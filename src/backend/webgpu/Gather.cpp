#include "nnr.h"
#include "attr_key.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include "shaders.h"

#include <webgpu/webgpu_cpp.h>

#include <cstring>
#include <vector>

namespace nnr {

namespace {

constexpr uint32_t WG = 256;

// ONNX Gather along a single axis. The indices tensor lives on the CPU in
// practice — typically a token-id or position-index initializer — so we
// read it there and upload a u32 copy to the GPU. An int64 indices tensor
// with GPU-only residency (unusual) causes reshape to return false so the
// registry falls back to CPU.
struct Gather_operator_webgpu : public operator_t {
    uint32_t total_u       = 0;
    uint32_t prefix_count  = 0;
    uint32_t index_count   = 0;
    uint32_t suffix_count  = 0;
    uint32_t axis_dim_u    = 0;
    int      caxis         = 0;

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          idx_buf;     // u32 storage — sized/recreated per reshape
    wgpu::Buffer          params;
    uint32_t              idx_capacity = 0;
    uint32_t              idx_gen      = 0;  // bumped when idx_buf reallocated
    uint32_t              dispatch_gx  = 0;
    uint32_t              dispatch_gy  = 0;

    // Cached BindGroup. Tensor-backed slots: [data, Y]; idx_buf tracked via idx_gen.
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[3] = {};   // [gen_data, idx_gen, gen_Y]

    static int64_t read_idx(const tensor_t* t, int i) {
        if (t->type == NNR_DATA_TYPE_INT32) return ((const int32_t*)t->data)[i];
        return ((const int64_t*)t->data)[i];
    }

    bool init() override {
        if (inputs.size() != 2) return false;
        if (outputs.size() != 1) return false;
        // Early reject dtypes we can't handle so the loader falls back to
        // CPU. Without this the failure only surfaces at prepare()'s
        // reshape() call, which doesn't have a backend-fallback path.
        // Common miss: dynamic-shape graphs feed Gather with int64 data
        // from a Shape op; those need to stay on CPU.
        if (inputs[0] && inputs[0]->type != NNR_DATA_TYPE_UNDEFINED
            && inputs[0]->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (inputs[1] && inputs[1]->type != NNR_DATA_TYPE_UNDEFINED
            && inputs[1]->type != NNR_DATA_TYPE_INT32
            && inputs[1]->type != NNR_DATA_TYPE_INT64) return false;
        if (!webgpu::device_ready()) return false;

        auto& dev = webgpu::get_device();
        wgpu::ShaderSourceWGSL w = {};
        w.code = webgpu::wgsl::gather;
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &w;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

        wgpu::BindGroupLayoutEntry e[4] = {};
        e[0].binding = 0; e[0].visibility = wgpu::ShaderStage::Compute;
        e[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        e[1].binding = 1; e[1].visibility = wgpu::ShaderStage::Compute;
        e[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        e[2].binding = 2; e[2].visibility = wgpu::ShaderStage::Compute;
        e[2].buffer.type = wgpu::BufferBindingType::Storage;
        e[3].binding = 3; e[3].visibility = wgpu::ShaderStage::Compute;
        e[3].buffer.type = wgpu::BufferBindingType::Uniform;
        wgpu::BindGroupLayoutDescriptor bgld = {};
        bgld.entryCount = 4; bgld.entries = e;
        bgl = dev.device.CreateBindGroupLayout(&bgld);

        wgpu::PipelineLayoutDescriptor pld = {};
        pld.bindGroupLayoutCount = 1;
        pld.bindGroupLayouts = &bgl;
        wgpu::PipelineLayout pl = dev.device.CreatePipelineLayout(&pld);

        wgpu::ComputePipelineDescriptor cpd = {};
        cpd.layout = pl;
        cpd.compute.module = sm;
        cpd.compute.entryPoint = "main";
        pipeline = dev.device.CreateComputePipeline(&cpd);

        wgpu::BufferDescriptor pd = {};
        pd.size  = 32;
        pd.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        params = dev.device.CreateBuffer(&pd);
        return true;
    }

    bool reshape() override {
        const tensor_t* data    = inputs[0];
        const tensor_t* indices = inputs[1];
        tensor_t*       y       = outputs[0];
        if (data->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (indices->type != NNR_DATA_TYPE_INT32 && indices->type != NNR_DATA_TYPE_INT64) return false;
        if (!indices->data) return false;   // indices must be CPU-readable

        int dim = data->ndim;
        if (dim <= 0 || dim > 8) return false;

        int axis_attr = (int)attribute(attr_key_t::axis, (int64_t)0);
        caxis = axis_attr < 0 ? axis_attr + dim : axis_attr;
        if (caxis < 0 || caxis >= dim) return false;

        int axis_dim = data->dims[caxis];
        int n_idx    = (int)indices->ndata;

        // Output shape = data.shape[:axis] ++ indices.shape ++ data.shape[axis+1:].
        int out_dims[8] = {};
        int out_n = 0;
        for (int k = 0; k < caxis;  ++k) out_dims[out_n++] = data->dims[k];
        for (int k = 0; k < indices->ndim; ++k) {
            if (out_n >= 8) return false;
            out_dims[out_n++] = indices->dims[k];
        }
        for (int k = caxis + 1; k < dim; ++k) {
            if (out_n >= 8) return false;
            out_dims[out_n++] = data->dims[k];
        }
        // A rank-0 indices with rank-0 data collapse gives an empty shape;
        // wrap to shape [1] so downstream ops see a well-formed tensor.
        if (out_n == 0) { out_dims[0] = 1; out_n = 1; }
        if (!y->reshape(std::span<const int>(out_dims, out_n), data->type)) return false;

        // Flat products for the 3-part decomposition.
        prefix_count = 1; for (int k = 0;        k < caxis; ++k) prefix_count *= (uint32_t)data->dims[k];
        suffix_count = 1; for (int k = caxis + 1; k < dim;   ++k) suffix_count *= (uint32_t)data->dims[k];
        index_count  = (uint32_t)(n_idx == 0 ? 1 : n_idx);
        total_u      = prefix_count * index_count * suffix_count;
        axis_dim_u   = (uint32_t)axis_dim;

        // Build a u32 index buffer on CPU (normalizing negatives + bounds
        // checking), then upload. Reuse the buffer when large enough.
        std::vector<uint32_t> host_idx;
        host_idx.reserve(index_count);
        for (int i = 0; i < n_idx; ++i) {
            int64_t v = read_idx(indices, i);
            if (v < 0) v += axis_dim;
            if (v < 0 || v >= axis_dim) return false;
            host_idx.push_back((uint32_t)v);
        }
        if (host_idx.empty()) host_idx.push_back(0u);

        auto& dev = webgpu::get_device();
        uint32_t needed_bytes = (uint32_t)(host_idx.size() * sizeof(uint32_t));
        // Storage buffers must be a multiple of 4; round up.
        needed_bytes = (needed_bytes + 3u) & ~3u;
        if (needed_bytes < 16u) needed_bytes = 16u;    // platform minimum
        if (idx_capacity < needed_bytes || !idx_buf) {
            wgpu::BufferDescriptor bd = {};
            bd.size  = needed_bytes;
            bd.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
            idx_buf      = dev.device.CreateBuffer(&bd);
            idx_capacity = needed_bytes;
            ++idx_gen;
        }
        dev.queue.WriteBuffer(idx_buf, 0, host_idx.data(), host_idx.size() * sizeof(uint32_t));

        webgpu::ensure_buffer(data, data->ndata * sizeof(float));
        webgpu::ensure_buffer(y,    y->ndata    * sizeof(float));

        uint32_t groups = (total_u + WG - 1) / WG;
        webgpu::dispatch_1d_grid(groups, dispatch_gx, dispatch_gy);
        uint8_t u[32] = {};
        auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(u + off, &v, 4); };
        put_u32(0,  total_u);
        put_u32(4,  prefix_count);
        put_u32(8,  index_count);
        put_u32(12, suffix_count);
        put_u32(16, axis_dim_u);
        put_u32(20, dispatch_gx * WG);
        dev.queue.WriteBuffer(params, 0, u, sizeof(u));
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();
        webgpu::upload_if_needed(inputs[0]);

        auto* rx = webgpu::find(inputs[0]);
        auto* ry = webgpu::find(outputs[0]);
        uint32_t gen_x = webgpu::generation_of(inputs[0]);
        uint32_t gen_y = webgpu::generation_of(outputs[0]);
        if (!cached_bg || gen_x != cached_gen[0] || idx_gen != cached_gen[1] || gen_y != cached_gen[2]) {
            wgpu::BindGroupEntry be[4] = {};
            be[0].binding = 0; be[0].buffer = rx->buf;  be[0].offset = 0; be[0].size = rx->size;
            be[1].binding = 1; be[1].buffer = idx_buf;  be[1].offset = 0; be[1].size = idx_capacity;
            be[2].binding = 2; be[2].buffer = ry->buf;  be[2].offset = 0; be[2].size = ry->size;
            be[3].binding = 3; be[3].buffer = params;   be[3].offset = 0; be[3].size = 32;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 4; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_x;
            cached_gen[1] = idx_gen;
            cached_gen[2] = gen_y;
        }

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, cached_bg);
        pass.DispatchWorkgroups(dispatch_gx, dispatch_gy, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_Gather_webgpu(int, pool_t& pool) {
    return pool_new<Gather_operator_webgpu>(pool);
}

} // namespace nnr
