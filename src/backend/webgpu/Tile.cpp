// WebGPU Tile: replicate X along each axis by `repeats[k]` counts.
//
// ONNX semantics: Tile(X, repeats) outputs Y with Y.shape[k] = X.shape[k] *
// repeats[k]. Rank must match (repeats.ndata == X.ndim). Y[i0,i1,...] =
// X[i0 % X.shape[0], i1 % X.shape[1], ...]. This differs from Expand's
// broadcast semantics: Expand only works when X's axis is size 1 (broadcast
// to target), whereas Tile wraps across the full input axis.
//
// Implementation mirrors Expand: one thread per output element, unflatten
// the output index through out_dims, modulo each axis by in_dim to get the
// input coord, sum into x_flat via in_strides. Supports rank ≤ 8 and f32
// data; non-f32 / repeats that are not CPU-resident int64 fall back to CPU.
//
// The `repeats` tensor must carry CPU-resident int64 data (ONNX spec); the
// reshape reads it to compute output dims and doesn't upload it to the GPU.
// If a model produces `repeats` via a graph op it would be GPU-only and we
// return false → CPU fallback, same pattern as Gather's int64 indices.

#include "nnr.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include "shaders.h"

#include <webgpu/webgpu_cpp.h>

#include <cstring>

namespace nnr {

namespace {

constexpr uint32_t WG = 256;

struct Tile_operator_webgpu : public operator_t {
    uint32_t total = 0;
    uint32_t ndim  = 0;
    uint32_t out_dims_u[8]   = {};
    uint32_t in_dims_u[8]    = {};
    uint32_t in_strides_u[8] = {};

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          meta_buf;
    uint32_t              dispatch_gx = 0;
    uint32_t              dispatch_gy = 0;

    // Cached BindGroup. Tensor-backed slots: [X, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[2] = {};

    bool init() override {
        if (!is_inout_size(2, 1)) return false;
        // Early reject non-f32 data / non-int64 repeats / rank > 8 so the
        // loader falls back to CPU. Dynamic-shape graphs often Tile int64
        // shape tensors.
        if (inputs[0] && inputs[0]->type != NNR_DATA_TYPE_UNDEFINED
            && inputs[0]->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (inputs[1] && inputs[1]->type != NNR_DATA_TYPE_UNDEFINED
            && inputs[1]->type != NNR_DATA_TYPE_INT64) return false;
        if (inputs[0] && inputs[0]->ndim > 8) return false;
        if (!webgpu::device_ready()) return false;

        auto& dev = webgpu::get_device();
        wgpu::ShaderSourceWGSL w = {};
        w.code = webgpu::wgsl::tile;
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &w;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

        wgpu::BindGroupLayoutEntry e[3] = {};
        e[0].binding = 0; e[0].visibility = wgpu::ShaderStage::Compute;
        e[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        e[1].binding = 1; e[1].visibility = wgpu::ShaderStage::Compute;
        e[1].buffer.type = wgpu::BufferBindingType::Storage;
        e[2].binding = 2; e[2].visibility = wgpu::ShaderStage::Compute;
        e[2].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        wgpu::BindGroupLayoutDescriptor bgld = {};
        bgld.entryCount = 3; bgld.entries = e;
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

        // 16B header + 32B out_dims + 32B in_dims + 32B in_strides = 112, pad to 128.
        wgpu::BufferDescriptor md = {};
        md.size = 128;
        md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        meta_buf = dev.device.CreateBuffer(&md);
        return true;
    }

    bool reshape() override {
        const tensor_t* x       = inputs[0];
        const tensor_t* repeats = inputs[1];
        tensor_t*       y       = outputs[0];

        if (x->type != NNR_DATA_TYPE_FLOAT32)         return false;
        if (repeats->type != NNR_DATA_TYPE_INT64)     return false;
        if (repeats->ndim != 1)                       return false;
        if (!repeats->data)                           return false;  // must be CPU-resident
        if ((int)repeats->ndata != x->ndim)           return false;  // rank must match
        if (x->ndim > 8)                              return false;

        const int64_t* pr = (const int64_t*)repeats->data;
        int out_dims[8] = {};
        for (int k = 0; k < x->ndim; ++k) {
            int64_t r = pr[k];
            if (r <= 0) return false;   // zero/negative repeats → CPU
            out_dims[k] = x->dims[k] * (int)r;
        }
        if (!y->reshape(std::span<const int>(out_dims, x->ndim), x->type)) return false;

        // Natural row-major strides over input.
        uint32_t in_nat[8] = {};
        {
            uint32_t s = 1;
            for (int i = x->ndim - 1; i >= 0; --i) { in_nat[i] = s; s *= (uint32_t)x->dims[i]; }
        }
        for (int k = 0; k < 8; ++k) { out_dims_u[k] = 0; in_dims_u[k] = 1; in_strides_u[k] = 0; }
        for (int k = 0; k < x->ndim; ++k) {
            out_dims_u[k]   = (uint32_t)out_dims[k];
            in_dims_u[k]    = (uint32_t)x->dims[k];
            in_strides_u[k] = in_nat[k];
        }
        ndim  = (uint32_t)x->ndim;
        total = 1;
        for (int k = 0; k < x->ndim; ++k) total *= (uint32_t)out_dims[k];

        webgpu::ensure_buffer(x, x->ndata * sizeof(float));
        webgpu::ensure_buffer(y, y->ndata * sizeof(float));

        // Meta layout (matches tile.wgsl):
        //   [0..4)    total
        //   [4..8)    ndim
        //   [8..12)   grid_stride_x (threads-x for 2D dispatch split)
        //   [12..16)  pad
        //   [16..48)  out_dims    (8 x u32)
        //   [48..80)  in_dims     (8 x u32)
        //   [80..112) in_strides  (8 x u32)
        uint32_t groups = (total + WG - 1) / WG;
        webgpu::dispatch_1d_grid(groups, dispatch_gx, dispatch_gy);
        uint8_t buf[128] = {};
        auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(buf + off, &v, 4); };
        put_u32(0, total);
        put_u32(4, ndim);
        put_u32(8, dispatch_gx * WG);
        for (int i = 0; i < 8; ++i) put_u32(16 + i * 4, out_dims_u[i]);
        for (int i = 0; i < 8; ++i) put_u32(48 + i * 4, in_dims_u[i]);
        for (int i = 0; i < 8; ++i) put_u32(80 + i * 4, in_strides_u[i]);
        webgpu::get_device().queue.WriteBuffer(meta_buf, 0, buf, sizeof(buf));
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();
        webgpu::upload_if_needed(inputs[0]);

        auto* rx = webgpu::find(inputs[0]);
        auto* ry = webgpu::find(outputs[0]);

        uint32_t gen_x = webgpu::generation_of(inputs[0]);
        uint32_t gen_y = webgpu::generation_of(outputs[0]);
        if (!cached_bg || gen_x != cached_gen[0] || gen_y != cached_gen[1]) {
            wgpu::BindGroupEntry be[3] = {};
            be[0].binding = 0; be[0].buffer = rx->buf;   be[0].offset = 0; be[0].size = rx->size;
            be[1].binding = 1; be[1].buffer = ry->buf;   be[1].offset = 0; be[1].size = ry->size;
            be[2].binding = 2; be[2].buffer = meta_buf;  be[2].offset = 0; be[2].size = 128;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 3; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_x;
            cached_gen[1] = gen_y;
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

operator_t* resolver_default_op_Tile_webgpu(int, pool_t& pool) {
    return pool_new<Tile_operator_webgpu>(pool);
}

} // namespace nnr
