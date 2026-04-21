// WebGPU ScatterND (f32, reduction ∈ {none, add, min, max}).
//
// Writes `updates` into a copy of `data` at positions specified by the
// `indices` tuples. Mirror of GatherND: each row of `indices` is a
// last_dim-tuple that picks the first `last_dim` axes of data; the write
// is the corresponding contiguous tail slice.
//
// Semantics:
//   Y = data  // copy, via CopyBufferToBuffer at exec() time
//   for each tuple t, inner s:
//     target = offset_from_tuple(t) + s
//     Y[target] = REDUCE(Y[target], updates[t * slice_size + s])
//
// Reductions:
//   - none: direct store (no races expected; with duplicate indices any
//     write may survive per ONNX spec — matches "last write wins")
//   - add/min/max: atomic CAS on bitcast<u32>(f32) — safe under parallel
//     writes to the same cell (duplicate indices). Scheme: load old u32,
//     bitcast to f32, compute REDUCE(old, val), CAS back. Loop until
//     exchange succeeds.
//
// All five reduction modes (none/add/min/max/mul) handled on GPU via CAS.
//
// Limits:
//   - data/updates f32 only
//   - indices int32/int64, CPU-resident
//   - last_dim ≤ 8 (fits the inner-stride vec4<u32> pair)

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

struct ScatterND_operator_webgpu : public operator_t {
    uint32_t total_u     = 0;    // num_tuples * slice_size
    uint32_t num_tuples_u = 0;
    uint32_t slice_size_u = 0;
    uint32_t last_dim_u   = 0;
    uint32_t inner_strides_u[8] = {};

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

    enum red_mode_t { RED_NONE = 0, RED_ADD, RED_MIN, RED_MAX, RED_MUL };
    red_mode_t red_mode = RED_NONE;

    bool init() override {
        if (inputs.size() != 3 || outputs.size() != 1) return false;
        if (!webgpu::device_ready()) return false;

        std::string_view red = attribute(attr_key_t::reduction, "none");
        if      (red == "none") red_mode = RED_NONE;
        else if (red == "add")  red_mode = RED_ADD;
        else if (red == "min")  red_mode = RED_MIN;
        else if (red == "max")  red_mode = RED_MAX;
        else if (red == "mul")  red_mode = RED_MUL;
        else                     return false;

        auto& dev = webgpu::get_device();

        // Body differs between "none" (direct store) and atomic variants.
        // All four share the same common prologue for unflattening and
        // computing the target offset.
        const char* body_none =
            "  Y[data_off + slice_off] = U[i];\n";
        const char* body_atomic_prologue =
            "  let val = U[i];\n"
            "  let off = data_off + slice_off;\n"
            "  var old_bits : u32 = atomicLoad(&Y[off]);\n"
            "  loop {\n"
            "    let old_val = bitcast<f32>(old_bits);\n";
        const char* body_atomic_epilogue =
            "    let new_bits = bitcast<u32>(new_val);\n"
            // Skip CAS if the reduction is a no-op (e.g. min of a value
            // greater than current).
            "    if (new_bits == old_bits) { break; }\n"
            "    let r = atomicCompareExchangeWeak(&Y[off], old_bits, new_bits);\n"
            "    if (r.exchanged) { break; }\n"
            "    old_bits = r.old_value;\n"
            "  }\n";

        const char* Y_decl_nonatomic =
            "@group(0) @binding(2) var<storage, read_write> Y  : array<f32>;\n";
        const char* Y_decl_atomic =
            "@group(0) @binding(2) var<storage, read_write> Y  : array<atomic<u32>>;\n";

        std::string src =
            "struct Meta {\n"
            "  total             : u32,\n"
            "  num_tuples        : u32,\n"
            "  slice_size        : u32,\n"
            "  last_dim          : u32,\n"
            "  inner_strides_lo  : vec4<u32>,\n"
            "  inner_strides_hi  : vec4<u32>,\n"
            "};\n"
            "@group(0) @binding(0) var<storage, read>       U  : array<f32>;\n"
            "@group(0) @binding(1) var<storage, read>       I  : array<u32>;\n";
        src += (red_mode == RED_NONE) ? Y_decl_nonatomic : Y_decl_atomic;
        src +=
            "@group(0) @binding(3) var<storage, read>       md : Meta;\n"
            "fn get_inner_stride(i : u32) -> u32 { if (i < 4u) { return md.inner_strides_lo[i]; } return md.inner_strides_hi[i - 4u]; }\n"
            "@compute @workgroup_size(256)\n"
            "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
            "  let i = gid.x;\n"
            "  if (i >= md.total) { return; }\n"
            "  let slice_off  = i % md.slice_size;\n"
            "  let tuple_idx  = i / md.slice_size;\n"
            "  let idx_base   = tuple_idx * md.last_dim;\n"
            "  var data_off   : u32 = 0u;\n"
            "  for (var k : u32 = 0u; k < md.last_dim; k = k + 1u) {\n"
            "    data_off = data_off + I[idx_base + k] * get_inner_stride(k);\n"
            "  }\n";
        if (red_mode == RED_NONE) {
            src += body_none;
        } else {
            src += body_atomic_prologue;
            switch (red_mode) {
                case RED_ADD: src += "    let new_val = old_val + val;\n"; break;
                case RED_MIN: src += "    let new_val = min(old_val, val);\n"; break;
                case RED_MAX: src += "    let new_val = max(old_val, val);\n"; break;
                case RED_MUL: src += "    let new_val = old_val * val;\n"; break;
                default: break;
            }
            src += body_atomic_epilogue;
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
        pld.bindGroupLayoutCount = 1;
        pld.bindGroupLayouts     = &bgl;
        wgpu::PipelineLayout pl  = dev.device.CreatePipelineLayout(&pld);

        wgpu::ComputePipelineDescriptor cpd = {};
        cpd.layout             = pl;
        cpd.compute.module     = sm;
        cpd.compute.entryPoint = "main";
        pipeline = dev.device.CreateComputePipeline(&cpd);

        wgpu::BufferDescriptor md = {};
        md.size  = 48;   // 16B header + 2 × vec4<u32>
        md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        meta_buf = dev.device.CreateBuffer(&md);
        return true;
    }

    bool reshape() override {
        const tensor_t* data    = inputs[0];
        const tensor_t* indices = inputs[1];
        const tensor_t* updates = inputs[2];
        tensor_t*       y       = outputs[0];
        if (!data || !indices || !updates || !y) return false;
        if (data->type    != NNR_DATA_TYPE_FLOAT32) return false;
        if (updates->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (indices->type != NNR_DATA_TYPE_INT32 && indices->type != NNR_DATA_TYPE_INT64) return false;
        if (!indices->data) return false;

        const int data_ndim = data->ndim;
        const int idx_ndim  = indices->ndim;
        if (idx_ndim < 1) return false;

        const int last_dim = indices->dims[idx_ndim - 1];
        if (last_dim < 1 || last_dim > data_ndim) return false;
        if (last_dim > 8) return false;

        // Output shape = data.shape (identity).
        if (!y->reshape_identity(data)) return false;

        num_tuples_u = 1;
        for (int i = 0; i < idx_ndim - 1; ++i)
            num_tuples_u *= (uint32_t)indices->dims[i];
        slice_size_u = 1;
        for (int d = last_dim; d < data_ndim; ++d)
            slice_size_u *= (uint32_t)data->dims[d];
        if (slice_size_u == 0) slice_size_u = 1;
        total_u = num_tuples_u * slice_size_u;
        last_dim_u = (uint32_t)last_dim;

        // updates must match [num_tuples, slice_size] when flattened.
        if ((uint32_t)updates->ndata != total_u) return false;

        for (int k = 0; k < 8; ++k) inner_strides_u[k] = 0;
        for (int k = 0; k < last_dim; ++k) {
            uint32_t s = 1;
            for (int j = k + 1; j < data_ndim; ++j) s *= (uint32_t)data->dims[j];
            inner_strides_u[k] = s;
        }

        // Pre-convert indices to u32 with bounds check (identical to GatherND).
        const uint32_t n_idx = num_tuples_u * (uint32_t)last_dim;
        std::vector<uint32_t> host_idx((size_t)n_idx, 0u);
        for (uint32_t t = 0; t < num_tuples_u; ++t)
          for (int k = 0; k < last_dim; ++k) {
              size_t flat = (size_t)t * (size_t)last_dim + (size_t)k;
              int64_t v = read_idx(indices, flat);
              int dim_size = data->dims[k];
              if (v < 0) v += dim_size;
              if (v < 0 || v >= dim_size) return false;
              host_idx[flat] = (uint32_t)v;
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

        uint8_t buf[48] = {};
        auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(buf + off, &v, 4); };
        put_u32(0,  total_u);
        put_u32(4,  num_tuples_u);
        put_u32(8,  slice_size_u);
        put_u32(12, last_dim_u);
        for (int k = 0; k < 8; ++k) put_u32(16 + k * 4, inner_strides_u[k]);
        dev.queue.WriteBuffer(meta_buf, 0, buf, sizeof(buf));
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();
        webgpu::upload_if_needed(inputs[0]);   // data
        webgpu::upload_if_needed(inputs[2]);   // updates

        auto* rd = webgpu::find(inputs[0]);
        auto* ru = webgpu::find(inputs[2]);
        auto* ry = webgpu::find(outputs[0]);

        wgpu::CommandEncoder& enc = webgpu::shared_encoder();
        // Seed Y with data. CopyBufferToBuffer is a straight memcpy.
        enc.CopyBufferToBuffer(rd->buf, 0, ry->buf, 0, rd->size);

        uint32_t gen_u = webgpu::generation_of(inputs[2]);
        uint32_t gen_y = webgpu::generation_of(outputs[0]);
        if (!cached_bg || gen_u != cached_gen[0] || idx_gen != cached_gen[1] || gen_y != cached_gen[2]) {
            wgpu::BindGroupEntry be[4] = {};
            be[0].binding = 0; be[0].buffer = ru->buf;   be[0].offset = 0; be[0].size = ru->size;
            be[1].binding = 1; be[1].buffer = idx_buf;   be[1].offset = 0; be[1].size = idx_capacity;
            be[2].binding = 2; be[2].buffer = ry->buf;   be[2].offset = 0; be[2].size = ry->size;
            be[3].binding = 3; be[3].buffer = meta_buf;  be[3].offset = 0; be[3].size = 48;
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

operator_t* resolver_default_op_ScatterND_webgpu(int, pool_t& pool) {
    return pool_new<ScatterND_operator_webgpu>(pool);
}

} // namespace nnr
