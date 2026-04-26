// WebGPU GatherND (f32, batch_dims ≥ 0). Each "index tuple" in `indices`
// picks a coord into `data`; the last indices dim is the tuple length
// (`last_k`), which selects the first `last_k` axes of data (after any
// batch_dims shared prefix). The result is the contiguous tail slice
// starting at that coord. Output shape follows ONNX:
//   data.shape[:batch_dims] ++ indices.shape[batch_dims:-1] ++ data.shape[batch_dims+last_k:]
//
// Indices are pre-converted on the CPU at reshape time (same pattern as
// Gather / OneHot) because WGSL has no i64 and we want the negative-index
// fix-up + out-of-range rejection to happen once, off the GPU.
//
// Limits: last_k ≤ 8 (fits the two-vec4 stride table); indices must be
// CPU-resident int32/int64; data must be f32. Anything outside these
// constraints returns false from reshape() → CPU fallback.

#include "nnr.h"
#include "attr_key.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <webgpu/webgpu_cpp.h>

namespace nnr {

namespace {

constexpr uint32_t WG = 256;

struct GatherND_operator_webgpu : public operator_t {
    int       batch_dims_attr = 0;
    uint32_t  total_u         = 0;
    uint32_t  batch_count_u   = 0;
    uint32_t  num_slices_u    = 0;
    uint32_t  slice_size_u    = 0;
    uint32_t  last_k_u        = 0;
    uint32_t  data_batch_stride_u = 0;
    uint32_t  inner_strides_u[8] = {};

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          idx_buf;
    uint32_t              idx_capacity = 0;
    uint32_t              idx_gen      = 0;
    wgpu::Buffer          meta_buf;

    // Cached BindGroup. Tensor-backed slots: [data, Y]; idx_buf tracked via idx_gen.
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[3] = {};   // [gen_data, idx_gen, gen_Y]

    static int64_t read_idx(const tensor_t* t, size_t i) {
        if (t->type == NNR_DATA_TYPE_INT32) return ((const int32_t*)t->data)[i];
        return ((const int64_t*)t->data)[i];
    }

    bool init() override {
        if (inputs.size() != 2 || outputs.size() != 1) return false;
        if (!webgpu::device_ready()) return false;

        batch_dims_attr = (int)attribute(attr_key_t::batch_dims, (int64_t)0);

        auto& dev = webgpu::get_device();

        std::string src =
            "struct Meta {\n"
            "  total              : u32,\n"
            "  batch_count        : u32,\n"
            "  num_slices         : u32,\n"
            "  slice_size         : u32,\n"
            "  last_k             : u32,\n"
            "  data_batch_stride  : u32,\n"
            "  _pad0              : u32,\n"
            "  _pad1              : u32,\n"
            "  inner_strides_lo   : vec4<u32>,\n"
            "  inner_strides_hi   : vec4<u32>,\n"
            "};\n"
            "@group(0) @binding(0) var<storage, read>       D  : array<f32>;\n"
            "@group(0) @binding(1) var<storage, read>       I  : array<u32>;\n"
            "@group(0) @binding(2) var<storage, read_write> Y  : array<f32>;\n"
            "@group(0) @binding(3) var<storage, read>       md : Meta;\n"
            "fn get_inner_stride(i : u32) -> u32 { if (i < 4u) { return md.inner_strides_lo[i]; } return md.inner_strides_hi[i - 4u]; }\n"
            "@compute @workgroup_size(256)\n"
            "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
            "  let i = gid.x;\n"
            "  if (i >= md.total) { return; }\n"
            "  let slice_off  = i % md.slice_size;\n"
            "  var tmp        = i / md.slice_size;\n"
            "  let slice_pos  = tmp % md.num_slices;\n"
            "  let batch_pos  = tmp / md.num_slices;\n"
            "  let idx_base   = (batch_pos * md.num_slices + slice_pos) * md.last_k;\n"
            "  var data_off   : u32 = 0u;\n"
            "  for (var k : u32 = 0u; k < md.last_k; k = k + 1u) {\n"
            "    data_off = data_off + I[idx_base + k] * get_inner_stride(k);\n"
            "  }\n"
            "  let src = batch_pos * md.data_batch_stride + data_off + slice_off;\n"
            "  Y[i] = D[src];\n"
            "}\n";

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
        md.size  = 64;   // 8 u32 header + 2 × vec4<u32> = 32 + 32 = 64 bytes
        md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        meta_buf = dev.device.CreateBuffer(&md);
        return true;
    }

    bool reshape() override {
        const tensor_t* data    = inputs[0];
        const tensor_t* indices = inputs[1];
        tensor_t*       y       = outputs[0];
        if (!data || !indices || !y) return false;
        if (data->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (indices->type != NNR_DATA_TYPE_INT32 && indices->type != NNR_DATA_TYPE_INT64) return false;
        if (!indices->data) return false;

        const int data_ndim    = data->ndim;
        const int indices_ndim = indices->ndim;
        if (indices_ndim < 1) return false;
        if (batch_dims_attr < 0 || batch_dims_attr >= data_ndim ||
            batch_dims_attr >= indices_ndim) return false;

        const int last_k = indices->dims[indices_ndim - 1];
        if (last_k < 1 || last_k > 8) return false;
        if (last_k > data_ndim - batch_dims_attr) return false;

        int out_dims[16] = {};
        int out_ndim = 0;
        for (int i = 0; i < batch_dims_attr; ++i)
            out_dims[out_ndim++] = data->dims[i];
        for (int i = batch_dims_attr; i < indices_ndim - 1; ++i)
            out_dims[out_ndim++] = indices->dims[i];
        for (int i = batch_dims_attr + last_k; i < data_ndim; ++i)
            out_dims[out_ndim++] = data->dims[i];
        if (out_ndim > 8) return false;
        if (out_ndim == 0) { out_dims[0] = 1; out_ndim = 1; }  // scalar → [1]
        if (!y->reshape(std::span<const int>(out_dims, out_ndim), data->type)) return false;

        batch_count_u = 1;
        for (int i = 0; i < batch_dims_attr; ++i)
            batch_count_u *= (uint32_t)data->dims[i];
        num_slices_u = 1;
        for (int i = batch_dims_attr; i < indices_ndim - 1; ++i)
            num_slices_u *= (uint32_t)indices->dims[i];
        slice_size_u = 1;
        for (int i = batch_dims_attr + last_k; i < data_ndim; ++i)
            slice_size_u *= (uint32_t)data->dims[i];
        if (num_slices_u == 0) num_slices_u = 1;
        if (slice_size_u == 0) slice_size_u = 1;
        total_u = batch_count_u * num_slices_u * slice_size_u;
        last_k_u = (uint32_t)last_k;

        data_batch_stride_u = 1;
        for (int i = batch_dims_attr; i < data_ndim; ++i)
            data_batch_stride_u *= (uint32_t)data->dims[i];

        for (int k = 0; k < 8; ++k) inner_strides_u[k] = 0;
        for (int k = 0; k < last_k; ++k) {
            uint32_t s = 1;
            for (int j = batch_dims_attr + k + 1; j < data_ndim; ++j)
                s *= (uint32_t)data->dims[j];
            inner_strides_u[k] = s;
        }

        // Pre-convert indices to u32 with negative fixup + bounds check. Out-of-
        // range indices cause reshape to fail → CPU fallback (which does the
        // same check). Total idx count = batch_count * num_slices * last_k.
        const uint32_t n_idx = batch_count_u * num_slices_u * (uint32_t)last_k;
        std::vector<uint32_t> host_idx((size_t)n_idx, 0u);
        for (uint32_t b = 0; b < batch_count_u; ++b)
          for (uint32_t s = 0; s < num_slices_u; ++s)
            for (int k = 0; k < last_k; ++k) {
                size_t flat = ((size_t)b * num_slices_u + s) * (size_t)last_k + (size_t)k;
                int64_t v = read_idx(indices, flat);
                int dim_size = data->dims[batch_dims_attr + k];
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

        webgpu::ensure_buffer(data, data->ndata * sizeof(float));
        webgpu::ensure_buffer(y,    y->ndata    * sizeof(float));

        uint8_t buf[64] = {};
        auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(buf + off, &v, 4); };
        put_u32(0,  total_u);
        put_u32(4,  batch_count_u);
        put_u32(8,  num_slices_u);
        put_u32(12, slice_size_u);
        put_u32(16, last_k_u);
        put_u32(20, data_batch_stride_u);
        for (int k = 0; k < 8; ++k) put_u32(32 + k * 4, inner_strides_u[k]);
        dev.queue.WriteBuffer(meta_buf, 0, buf, sizeof(buf));
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();
        webgpu::upload_if_needed(inputs[0]);

        auto* rd = webgpu::find(inputs[0]);
        auto* ry = webgpu::find(outputs[0]);
        uint32_t gen_d = webgpu::generation_of(inputs[0]);
        uint32_t gen_y = webgpu::generation_of(outputs[0]);
        if (!cached_bg || gen_d != cached_gen[0] || idx_gen != cached_gen[1] || gen_y != cached_gen[2]) {
            wgpu::BindGroupEntry be[4] = {};
            be[0].binding = 0; be[0].buffer = rd->buf;   be[0].offset = 0; be[0].size = rd->size;
            be[1].binding = 1; be[1].buffer = idx_buf;   be[1].offset = 0; be[1].size = idx_capacity;
            be[2].binding = 2; be[2].buffer = ry->buf;   be[2].offset = 0; be[2].size = ry->size;
            be[3].binding = 3; be[3].buffer = meta_buf;  be[3].offset = 0; be[3].size = 64;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 4; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_d;
            cached_gen[1] = idx_gen;
            cached_gen[2] = gen_y;
        }

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();
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

operator_t* resolver_default_op_GatherND_webgpu(int, pool_t& pool) {
    return pool_new<GatherND_operator_webgpu>(pool);
}

} // namespace nnr
