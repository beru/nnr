// WebGPU ScatterElements (f32, reduction ∈ {none, add, min, max}).
//
// ONNX semantics:
//   Y = data                                 // copy
//   for each flat position i in indices:
//     out = unflatten(i, indices.shape)
//     out[axis] = indices[out]               // scatter axis is rewritten
//     Y[out] = updates[out]
//
// ~Gather's inverse. Unlike ScatterND (which uses index tuples to pick
// positions in `data`), ScatterElements uses an indices tensor with the
// *same rank as data* whose element at each position is the destination
// index along `axis`. The remaining axes of the output position match
// the original flat index's unflattened coordinates.
//
// Limits:
//   - data / updates / Y: f32 only
//   - indices: int32 or int64 (CPU-resident; read at reshape time so we
//     can normalize negatives and emit a u32 scatter index buffer)
//   - reduction ∈ {none, add, min, max, mul} — all via atomic CAS
//   - indices.ndim == data.ndim, rank ≤ 8
//
// Writes collide when two indices land on the same output cell. For
// reduction="none" ONNX allows any surviving write — matches "last write
// wins" — which is what naked non-atomic writes produce in parallel.

#include "nnr.h"
#include "attr_key.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <webgpu/webgpu_cpp.h>

namespace nnr {

namespace {

constexpr uint32_t WG = 256;

struct ScatterElements_operator_webgpu : public operator_t {
    enum red_mode_t { RED_NONE = 0, RED_ADD, RED_MIN, RED_MAX, RED_MUL };
    red_mode_t red_mode = RED_NONE;

    int axis = 0;
    int caxis = 0;
    uint32_t total_u     = 0;
    uint32_t ndim_u      = 0;
    uint32_t caxis_u     = 0;
    uint32_t idx_dims_u[8]   = {};
    uint32_t data_strides_u[8] = {};

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          idx_buf;
    uint32_t              idx_capacity = 0;
    uint32_t              idx_gen      = 0;
    wgpu::Buffer          meta_buf;

    // Cached BindGroup. Tensor-backed slots: [updates, Y]; idx_buf via idx_gen.
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[3] = {};   // [gen_updates, idx_gen, gen_Y]

    static int64_t read_idx(const tensor_t* t, size_t i) {
        if (t->type == NNR_DATA_TYPE_INT32) return ((const int32_t*)t->data)[i];
        return ((const int64_t*)t->data)[i];
    }

    bool init() override {
        if (inputs.size() != 3 || outputs.size() != 1) return false;
        if (!webgpu::device_ready()) return false;

        axis = attribute(attr_key_t::axis, (int32_t)0);
        std::string_view red = attribute(attr_key_t::reduction, "none");
        if      (red == "none") red_mode = RED_NONE;
        else if (red == "add")  red_mode = RED_ADD;
        else if (red == "min")  red_mode = RED_MIN;
        else if (red == "max")  red_mode = RED_MAX;
        else if (red == "mul")  red_mode = RED_MUL;
        else                     return false;

        auto& dev = webgpu::get_device();

        // Meta layout (all u32, 16-byte aligned):
        //   [0]  total          (num elements in indices)
        //   [1]  ndim
        //   [2]  caxis
        //   [3]  pad
        //   [4..11]  idx_dims      (up to 8 axes; pack as 2× vec4<u32>)
        //   [12..19] data_strides  (element strides for each axis of data; 2× vec4<u32>)
        const char* Y_decl = (red_mode == RED_NONE)
            ? "@group(0) @binding(2) var<storage, read_write> Y  : array<f32>;\n"
            : "@group(0) @binding(2) var<storage, read_write> Y  : array<atomic<u32>>;\n";

        std::string src =
            "struct Meta {\n"
            "  total        : u32,\n"
            "  ndim         : u32,\n"
            "  caxis        : u32,\n"
            "  _pad         : u32,\n"
            "  idx_dims_lo  : vec4<u32>,\n"
            "  idx_dims_hi  : vec4<u32>,\n"
            "  dstrides_lo  : vec4<u32>,\n"
            "  dstrides_hi  : vec4<u32>,\n"
            "};\n"
            "@group(0) @binding(0) var<storage, read>       U  : array<f32>;\n"
            "@group(0) @binding(1) var<storage, read>       I  : array<u32>;\n";
        src += Y_decl;
        src +=
            "@group(0) @binding(3) var<storage, read>       md : Meta;\n"
            "fn idx_dim(i : u32) -> u32 {\n"
            "  if (i < 4u) { return md.idx_dims_lo[i]; }\n"
            "  return md.idx_dims_hi[i - 4u];\n"
            "}\n"
            "fn dstride(i : u32) -> u32 {\n"
            "  if (i < 4u) { return md.dstrides_lo[i]; }\n"
            "  return md.dstrides_hi[i - 4u];\n"
            "}\n"
            "@compute @workgroup_size(256)\n"
            "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
            "  let i = gid.x;\n"
            "  if (i >= md.total) { return; }\n"
            // Unflatten i using indices.shape.
            "  var rem : u32 = i;\n"
            "  var coord : array<u32, 8>;\n"
            "  for (var d : u32 = md.ndim; d > 0u; d = d - 1u) {\n"
            "    let k = d - 1u;\n"
            "    let w = idx_dim(k);\n"
            "    coord[k] = rem % w;\n"
            "    rem = rem / w;\n"
            "  }\n"
            "  coord[md.caxis] = I[i];\n"
            "  var off : u32 = 0u;\n"
            "  for (var d : u32 = 0u; d < md.ndim; d = d + 1u) {\n"
            "    off = off + coord[d] * dstride(d);\n"
            "  }\n";
        if (red_mode == RED_NONE) {
            src += "  Y[off] = U[i];\n";
        } else {
            src +=
                "  let val = U[i];\n"
                "  var old_bits : u32 = atomicLoad(&Y[off]);\n"
                "  loop {\n"
                "    let old_val = bitcast<f32>(old_bits);\n";
            switch (red_mode) {
                case RED_ADD: src += "    let new_val = old_val + val;\n"; break;
                case RED_MIN: src += "    let new_val = min(old_val, val);\n"; break;
                case RED_MAX: src += "    let new_val = max(old_val, val);\n"; break;
                case RED_MUL: src += "    let new_val = old_val * val;\n"; break;
                default: break;
            }
            src +=
                "    let new_bits = bitcast<u32>(new_val);\n"
                "    if (new_bits == old_bits) { break; }\n"
                "    let r = atomicCompareExchangeWeak(&Y[off], old_bits, new_bits);\n"
                "    if (r.exchanged) { break; }\n"
                "    old_bits = r.old_value;\n"
                "  }\n";
        }
        src += "}\n";

        wgpu::ShaderSourceWGSL ws = {};
        ws.code = src.c_str();
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &ws;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

        wgpu::BindGroupLayoutEntry e[4] = {};
        for (int i = 0; i < 2; ++i) {
            e[i].binding = (uint32_t)i;
            e[i].visibility = wgpu::ShaderStage::Compute;
            e[i].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        }
        e[2].binding = 2; e[2].visibility = wgpu::ShaderStage::Compute;
        e[2].buffer.type = wgpu::BufferBindingType::Storage;
        e[3].binding = 3; e[3].visibility = wgpu::ShaderStage::Compute;
        e[3].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        wgpu::BindGroupLayoutDescriptor bgld = {};
        bgld.entryCount = 4; bgld.entries = e;
        bgl = dev.device.CreateBindGroupLayout(&bgld);

        wgpu::PipelineLayoutDescriptor pld = {};
        pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = &bgl;
        wgpu::PipelineLayout pl  = dev.device.CreatePipelineLayout(&pld);

        wgpu::ComputePipelineDescriptor cpd = {};
        cpd.layout             = pl;
        cpd.compute.module     = sm;
        cpd.compute.entryPoint = "main";
        pipeline = dev.device.CreateComputePipeline(&cpd);

        wgpu::BufferDescriptor md = {};
        md.size  = 80;   // 16B header + 2× vec4<u32> (idx_dims) + 2× vec4<u32> (dstrides)
        md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        meta_buf = dev.device.CreateBuffer(&md);
        return true;
    }

    bool reshape() override {
        const tensor_t* data    = inputs[0];
        const tensor_t* indices = inputs[1];
        const tensor_t* updates = inputs[2];
        tensor_t*       y       = outputs[0];
        if (data->type    != NNR_DATA_TYPE_FLOAT32) return false;
        if (updates->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (indices->type != NNR_DATA_TYPE_INT32 && indices->type != NNR_DATA_TYPE_INT64) return false;
        if (!indices->data) return false;
        if (data->ndim != indices->ndim) return false;
        if (updates->ndim != indices->ndim) return false;
        if (data->ndim > 8 || data->ndim < 1) return false;
        for (int d = 0; d < data->ndim; ++d) {
            if (indices->dims[d] != updates->dims[d]) return false;
            // Per ONNX: indices.dims[d] ≤ data.dims[d] for d != axis; ==
            // along axis is a typical use (full scatter), but the spec
            // allows indices to be smaller on non-axis too.
        }

        caxis = axis;
        if (caxis < 0) caxis += data->ndim;
        if (caxis < 0 || caxis >= data->ndim) return false;

        if (!y->reshape_identity(data)) return false;

        ndim_u = (uint32_t)data->ndim;
        caxis_u = (uint32_t)caxis;
        total_u = (uint32_t)indices->ndata;

        for (int k = 0; k < 8; ++k) {
            idx_dims_u[k]    = 0;
            data_strides_u[k] = 0;
        }
        for (int d = 0; d < data->ndim; ++d) {
            idx_dims_u[d] = (uint32_t)indices->dims[d];
            uint32_t s = 1;
            for (int j = d + 1; j < data->ndim; ++j) s *= (uint32_t)data->dims[j];
            data_strides_u[d] = s;
        }

        // Pre-convert scatter indices to u32. Negative indices are
        // normalized; out-of-range is treated as a reshape-time failure
        // so the op falls back to CPU, matching Gather's behavior.
        const uint32_t nflat = (uint32_t)indices->ndata;
        std::vector<uint32_t> host_idx((size_t)nflat, 0u);
        const int axis_dim = data->dims[caxis];
        for (uint32_t i = 0; i < nflat; ++i) {
            int64_t v = read_idx(indices, (size_t)i);
            if (v < 0) v += axis_dim;
            if (v < 0 || v >= axis_dim) return false;
            host_idx[i] = (uint32_t)v;
        }

        auto& dev = webgpu::get_device();
        uint32_t needed_bytes = (uint32_t)(host_idx.size() * sizeof(uint32_t));
        needed_bytes = (needed_bytes + 3u) & ~3u;
        if (needed_bytes < 16u) needed_bytes = 16u;
        if (idx_capacity < needed_bytes || !idx_buf) {
            wgpu::BufferDescriptor bd = {};
            bd.size  = needed_bytes;
            bd.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
            idx_buf      = dev.device.CreateBuffer(&bd);
            idx_capacity = needed_bytes;
            ++idx_gen;
        }
        dev.queue.WriteBuffer(idx_buf, 0, host_idx.data(),
                              host_idx.size() * sizeof(uint32_t));

        webgpu::ensure_buffer(data,    data->ndata    * sizeof(float));
        webgpu::ensure_buffer(updates, updates->ndata * sizeof(float));
        webgpu::ensure_buffer(y,       y->ndata       * sizeof(float));

        uint8_t buf[80] = {};
        auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(buf + off, &v, 4); };
        put_u32(0,  total_u);
        put_u32(4,  ndim_u);
        put_u32(8,  caxis_u);
        put_u32(12, 0u);
        for (int k = 0; k < 8; ++k) put_u32(16 + k * 4, idx_dims_u[k]);
        for (int k = 0; k < 8; ++k) put_u32(48 + k * 4, data_strides_u[k]);
        dev.queue.WriteBuffer(meta_buf, 0, buf, sizeof(buf));
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();
        webgpu::upload_if_needed(inputs[0]);
        webgpu::upload_if_needed(inputs[2]);

        auto* rd = webgpu::find(inputs[0]);
        auto* ru = webgpu::find(inputs[2]);
        auto* ry = webgpu::find(outputs[0]);

        wgpu::CommandEncoder& enc = webgpu::shared_encoder();
        enc.CopyBufferToBuffer(rd->buf, 0, ry->buf, 0, rd->size);

        uint32_t gen_u = webgpu::generation_of(inputs[2]);
        uint32_t gen_y = webgpu::generation_of(outputs[0]);
        if (!cached_bg || gen_u != cached_gen[0] || idx_gen != cached_gen[1] || gen_y != cached_gen[2]) {
            wgpu::BindGroupEntry be[4] = {};
            be[0].binding = 0; be[0].buffer = ru->buf;   be[0].offset = 0; be[0].size = ru->size;
            be[1].binding = 1; be[1].buffer = idx_buf;   be[1].offset = 0; be[1].size = idx_capacity;
            be[2].binding = 2; be[2].buffer = ry->buf;   be[2].offset = 0; be[2].size = ry->size;
            be[3].binding = 3; be[3].buffer = meta_buf;  be[3].offset = 0; be[3].size = 80;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 4; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_u;
            cached_gen[1] = idx_gen;
            cached_gen[2] = gen_y;
        }

        wgpu::ComputePassEncoder pass = enc.BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, cached_bg);
        uint32_t groups = (total_u + WG - 1) / WG;
        pass.DispatchWorkgroups(groups, 1, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_ScatterElements_webgpu(int, pool_t& pool) {
    return pool_new<ScatterElements_operator_webgpu>(pool);
}

} // namespace nnr
