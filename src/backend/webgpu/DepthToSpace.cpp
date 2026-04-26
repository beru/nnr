// WebGPU DepthToSpace / SpaceToDepth — NCHW 4D reshape-by-index ops.
//
// DepthToSpace: [N, C, H, W] → [N, C/bs^2, H*bs, W*bs]. Two modes: DCR
// (default — depth-column-row) and CRD (column-row-depth). Used in
// super-resolution / sub-pixel convolution.
// SpaceToDepth: [N, C, H, W] → [N, C*bs^2, H/bs, W/bs]. Inverse of DCR-mode
// DepthToSpace. No mode attribute in ONNX.
//
// Scope: f32 only. Other dtypes → CPU. Non-4D → CPU. `blocksize` must
// divide C (for D2S) or divide H and W (for S2D).

#include "nnr.h"
#include "attr_key.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include "shaders.h"

#include <string_view>
#include <webgpu/webgpu_cpp.h>

namespace nnr {

namespace {

constexpr uint32_t WG = 256;

// Shared pipeline build — both ops have identical bind-group layouts and
// uniform shapes (8 u32). The only difference is which WGSL + which
// reshape/index-map the caller picks.
struct DtsStsCommon {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          uniforms;

    // Cached BindGroup. Tensor-backed slots: [X, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[2] = {};

    bool build(const char* wgsl_src) {
        auto& dev = webgpu::get_device();
        wgpu::ShaderSourceWGSL w = {};
        w.code = wgsl_src;
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &w;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

        wgpu::BindGroupLayoutEntry e[3] = {};
        e[0].binding = 0; e[0].visibility = wgpu::ShaderStage::Compute;
        e[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        e[1].binding = 1; e[1].visibility = wgpu::ShaderStage::Compute;
        e[1].buffer.type = wgpu::BufferBindingType::Storage;
        e[2].binding = 2; e[2].visibility = wgpu::ShaderStage::Compute;
        e[2].buffer.type = wgpu::BufferBindingType::Uniform;
        wgpu::BindGroupLayoutDescriptor bgld = {};
        bgld.entryCount = 3; bgld.entries = e;
        bgl = dev.device.CreateBindGroupLayout(&bgld);

        wgpu::PipelineLayoutDescriptor pld = {};
        pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = &bgl;
        wgpu::PipelineLayout pl = dev.device.CreatePipelineLayout(&pld);

        wgpu::ComputePipelineDescriptor cpd = {};
        cpd.layout = pl;
        cpd.compute.module = sm;
        cpd.compute.entryPoint = "main";
        pipeline = dev.device.CreateComputePipeline(&cpd);

        wgpu::BufferDescriptor ud = {};
        ud.size = 32;   // 8 x u32
        ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        uniforms = dev.device.CreateBuffer(&ud);
        return true;
    }

    // Called from the subclass reshape() once the cfg is filled in.
    void write_uniform(const uint32_t cfg[8]) {
        webgpu::get_device().queue.WriteBuffer(uniforms, 0, cfg, 32);
    }

    bool dispatch(const tensor_t* X, const tensor_t* Y, uint32_t total) {
        auto& dev = webgpu::get_device();
        webgpu::upload_if_needed(X);

        auto* rx = webgpu::find(X);
        auto* ry = webgpu::find(Y);
        uint32_t gen_x = webgpu::generation_of(X);
        uint32_t gen_y = webgpu::generation_of(Y);
        if (!cached_bg || gen_x != cached_gen[0] || gen_y != cached_gen[1]) {
            wgpu::BindGroupEntry be[3] = {};
            be[0].binding = 0; be[0].buffer = rx->buf;   be[0].offset = 0; be[0].size = rx->size;
            be[1].binding = 1; be[1].buffer = ry->buf;   be[1].offset = 0; be[1].size = ry->size;
            be[2].binding = 2; be[2].buffer = uniforms;  be[2].offset = 0; be[2].size = 32;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 3; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_x;
            cached_gen[1] = gen_y;
        }

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, cached_bg);
        uint32_t groups = (total + WG - 1) / WG;
        pass.DispatchWorkgroups(groups, 1, 1);
        pass.End();

        webgpu::mark_gpu_written(Y);
        return true;
    }
};

struct DepthToSpace_op_webgpu : public operator_t, DtsStsCommon {
    int blocksize = 0;
    int is_crd    = 0;
    uint32_t cfg[8] = {};

    bool init() override {
        if (!is_inout_size(1, 1)) return false;
        if (!webgpu::device_ready()) return false;
        blocksize = attribute(attr_key_t::blocksize, (int32_t)0);
        if (blocksize <= 0) return false;
        std::string_view mode = attribute(attr_key_t::mode, "DCR");
        is_crd = (mode == "CRD") ? 1 : 0;
        return build(webgpu::wgsl::depth_to_space);
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t*       y = outputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (x->ndim != 4) return false;
        int N = x->dims[0], C = x->dims[1], H = x->dims[2], W = x->dims[3];
        int bs2 = blocksize * blocksize;
        if (C % bs2 != 0) return false;
        int C_out = C / bs2;
        int out_dims[4] = { N, C_out, H * blocksize, W * blocksize };
        if (!y->reshape(std::span<const int>(out_dims, 4), x->type)) return false;

        cfg[0] = (uint32_t)y->ndata;
        cfg[1] = (uint32_t)N;
        cfg[2] = (uint32_t)C;
        cfg[3] = (uint32_t)H;
        cfg[4] = (uint32_t)W;
        cfg[5] = (uint32_t)C_out;
        cfg[6] = (uint32_t)blocksize;
        cfg[7] = (uint32_t)is_crd;

        webgpu::ensure_buffer(x, x->ndata * sizeof(float));
        webgpu::ensure_buffer(y, y->ndata * sizeof(float));
        write_uniform(cfg);
        return true;
    }

    bool exec() override {
        return dispatch(inputs[0], outputs[0], cfg[0]);
    }
};

struct SpaceToDepth_op_webgpu : public operator_t, DtsStsCommon {
    int blocksize = 0;
    uint32_t cfg[8] = {};

    bool init() override {
        if (!is_inout_size(1, 1)) return false;
        if (!webgpu::device_ready()) return false;
        blocksize = attribute(attr_key_t::blocksize, (int32_t)0);
        if (blocksize <= 0) return false;
        return build(webgpu::wgsl::space_to_depth);
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t*       y = outputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (x->ndim != 4) return false;
        int N = x->dims[0], C = x->dims[1], H = x->dims[2], W = x->dims[3];
        int bs = blocksize;
        if (H % bs != 0 || W % bs != 0) return false;
        int C_out = C * bs * bs;
        int out_dims[4] = { N, C_out, H / bs, W / bs };
        if (!y->reshape(std::span<const int>(out_dims, 4), x->type)) return false;

        cfg[0] = (uint32_t)y->ndata;
        cfg[1] = (uint32_t)N;
        cfg[2] = (uint32_t)C;
        cfg[3] = (uint32_t)H;
        cfg[4] = (uint32_t)W;
        cfg[5] = (uint32_t)C_out;
        cfg[6] = (uint32_t)bs;
        cfg[7] = 0;

        webgpu::ensure_buffer(x, x->ndata * sizeof(float));
        webgpu::ensure_buffer(y, y->ndata * sizeof(float));
        write_uniform(cfg);
        return true;
    }

    bool exec() override {
        return dispatch(inputs[0], outputs[0], cfg[0]);
    }
};

} // namespace

operator_t* resolver_default_op_DepthToSpace_webgpu(int, pool_t& pool) {
    return pool_new<DepthToSpace_op_webgpu>(pool);
}
operator_t* resolver_default_op_SpaceToDepth_webgpu(int, pool_t& pool) {
    return pool_new<SpaceToDepth_op_webgpu>(pool);
}

} // namespace nnr
