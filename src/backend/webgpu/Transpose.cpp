#include "nnr.h"
#include "attr_key.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include "shaders.h"

#include <webgpu/webgpu_cpp.h>

#include <cstring>

namespace nnr {

namespace {

struct Transpose_operator_webgpu : public operator_t {
    uint32_t perm_u[8]       = {};
    uint32_t in_strides_u[8] = {};
    uint32_t out_dims_u[8]   = {};
    uint32_t ndim  = 0;
    uint32_t total = 0;

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          meta_buf;
    uint32_t              dispatch_gx = 0;
    uint32_t              dispatch_gy = 0;

    // Cached BindGroup. Tensor-backed slots: [X, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[2] = {};

    bool init() override {
        if (!is_inout_size(1, 1))    return false;
        if (!webgpu::device_ready()) return false;

        auto& dev = webgpu::get_device();
        wgpu::ShaderSourceWGSL w = {};
        w.code = webgpu::wgsl::transpose;
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

        // total(4) + ndim(4) + 8B pad + 3 arrays of 8 u32 = 16 + 96 = 112 bytes,
        // rounded up to 128 for alignment headroom.
        wgpu::BufferDescriptor md = {};
        md.size = 128;
        md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        meta_buf = dev.device.CreateBuffer(&md);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t*       y = outputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (x->ndim <= 0 || x->ndim > 8)      return false;

        // perm attribute (default: reverse)
        int64_t* ints = nullptr;
        int nint = attribute(attr_key_t::perm, ints);
        int perm[8] = {};
        if (nint == x->ndim) {
            for (int i = 0; i < x->ndim; ++i) perm[i] = (int)ints[i];
        } else {
            for (int i = 0; i < x->ndim; ++i) perm[i] = x->ndim - 1 - i;
        }

        int dims[8];
        for (int i = 0; i < x->ndim; ++i) dims[i] = x->dims[perm[i]];
        if (!y->reshape(std::span<const int>(dims, x->ndim), x->type)) return false;

        ndim = (uint32_t)x->ndim;
        total = 1;
        for (int i = 0; i < x->ndim; ++i) total *= (uint32_t)dims[i];

        // Row-major (C-contig) input strides.
        uint32_t s = 1;
        for (int i = x->ndim - 1; i >= 0; --i) {
            in_strides_u[i] = s;
            s *= (uint32_t)x->dims[i];
        }
        for (int i = 0; i < x->ndim; ++i) {
            out_dims_u[i] = (uint32_t)dims[i];
            perm_u[i]     = (uint32_t)perm[i];
        }

        webgpu::ensure_buffer(x, x->ndata * sizeof(float));
        webgpu::ensure_buffer(y, y->ndata * sizeof(float));

        // Compute 2D dispatch shape so we don't trip WebGPU's per-dim
        // 65535 workgroup cap on multi-million-element intermediates.
        // Shader uses `grid_stride_x` to reconstruct 1D thread id from
        // (gid.x, gid.y).
        uint32_t wg_count = (total + 63) / 64;
        webgpu::dispatch_1d_grid(wg_count, dispatch_gx, dispatch_gy);
        uint32_t grid_stride_x = dispatch_gx * 64u;

        // Meta layout (std140-ish, 16-byte aligned):
        //   [0..4)    total
        //   [4..8)    ndim
        //   [8..16)   grid_stride_x + pad
        //   [16..48)  in_strides (8 x u32)
        //   [48..80)  out_dims
        //   [80..112) perm
        uint8_t buf[128] = {};
        auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(buf + off, &v, 4); };
        put_u32(0,  total);
        put_u32(4,  ndim);
        put_u32(8,  grid_stride_x);
        for (int i = 0; i < 8; ++i) put_u32(16 + i * 4,  in_strides_u[i]);
        for (int i = 0; i < 8; ++i) put_u32(48 + i * 4,  out_dims_u[i]);
        for (int i = 0; i < 8; ++i) put_u32(80 + i * 4,  perm_u[i]);
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

operator_t* resolver_default_op_Transpose_webgpu(int, pool_t& pool) {
    return pool_new<Transpose_operator_webgpu>(pool);
}

} // namespace nnr
