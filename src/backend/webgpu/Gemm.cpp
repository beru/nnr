#include "nnr.h"
#include "attr_key.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include "shaders.h"

#include <webgpu/webgpu_cpp.h>

#include <vector>

namespace nnr {

namespace {

// ONNX Gemm (opset 1+): Y = alpha * op(A) * op(B) + beta * C.
// Supports transA, transB, and bias C shaped as {}, [N], or [M, N].
// Other C broadcast shapes fall back to CPU via reshape() returning false.
struct Gemm_operator_webgpu : public operator_t {
    float alpha  = 1.0f;
    float beta   = 1.0f;
    int   transA = 0;
    int   transB = 0;

    int M = 0, N = 0, K = 0;
    int bias_kind = 0;  // 0=none, 1=per-col, 2=full

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          cfg_buf;
    wgpu::Buffer          zero_bias;  // bound when C absent

    // Cached BindGroup. Slots: [A, B, C|0, Y]. `cfg_buf` and `zero_bias`
    // are op-owned; zero_bias is stable for the op's lifetime once
    // allocated (reshape sets it up only when `c` is absent, and that
    // topology doesn't change post-init).
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[4] = {};

    bool init() override {
        if (inputs.size() < 2 || outputs.empty()) return false;
        if (!webgpu::device_ready())              return false;
        alpha  = attribute(attr_key_t::alpha,  1.0f);
        beta   = attribute(attr_key_t::beta,   1.0f);
        transA = attribute(attr_key_t::transA, (int32_t)0);
        transB = attribute(attr_key_t::transB, (int32_t)0);

        auto& dev = webgpu::get_device();
        wgpu::ShaderSourceWGSL w = {};
        w.code = webgpu::wgsl::gemm;
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &w;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

        wgpu::BindGroupLayoutEntry e[5] = {};
        for (int i = 0; i < 3; ++i) {
            e[i].binding = (uint32_t)i;
            e[i].visibility = wgpu::ShaderStage::Compute;
            e[i].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        }
        e[3].binding = 3; e[3].visibility = wgpu::ShaderStage::Compute;
        e[3].buffer.type = wgpu::BufferBindingType::Storage;
        e[4].binding = 4; e[4].visibility = wgpu::ShaderStage::Compute;
        e[4].buffer.type = wgpu::BufferBindingType::Uniform;
        wgpu::BindGroupLayoutDescriptor bgld = {};
        bgld.entryCount = 5; bgld.entries = e;
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

        wgpu::BufferDescriptor cd = {};
        cd.size = 48;  // 3 * vec4 = 48 bytes
        cd.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        cfg_buf = dev.device.CreateBuffer(&cd);
        return true;
    }

    bool reshape() override {
        const tensor_t* a = inputs[0];
        const tensor_t* b = inputs[1];
        const tensor_t* c = (inputs.size() > 2 && inputs[2]) ? inputs[2] : nullptr;
        tensor_t*       y = outputs[0];

        if (a->type != NNR_DATA_TYPE_FLOAT32 || b->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (a->ndim != 2 || b->ndim != 2) return false;

        int aM = transA ? a->dims[1] : a->dims[0];
        int aK = transA ? a->dims[0] : a->dims[1];
        int bK = transB ? b->dims[1] : b->dims[0];
        int bN = transB ? b->dims[0] : b->dims[1];
        if (aK != bK) return false;
        M = aM; K = aK; N = bN;

        bias_kind = 0;
        if (c) {
            if (c->type != NNR_DATA_TYPE_FLOAT32) return false;
            if (c->ndim == 1 && c->dims[0] == N)            bias_kind = 1;
            else if (c->ndim == 2 && c->dims[0] == M && c->dims[1] == N) bias_kind = 2;
            else return false;   // fall back for other broadcast shapes
        }

        int ydims[2] = { M, N };
        if (!y->reshape(std::span<const int>(ydims, 2), NNR_DATA_TYPE_FLOAT32)) return false;

        webgpu::ensure_buffer(a, a->ndata * sizeof(float));
        webgpu::ensure_buffer(b, b->ndata * sizeof(float));
        if (c) webgpu::ensure_buffer(c, c->ndata * sizeof(float));
        webgpu::ensure_buffer(y, y->ndata * sizeof(float));

        // Binding 2 (C) must always be valid. Bind a 1-element zero buffer
        // when bias is absent so WGSL array access is well-defined.
        if (!c) {
            auto& dev = webgpu::get_device();
            wgpu::BufferDescriptor bd = {};
            bd.size  = 16;
            bd.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
            zero_bias = dev.device.CreateBuffer(&bd);
            float zeros[4] = {};
            dev.queue.WriteBuffer(zero_bias, 0, zeros, sizeof(zeros));
        }

        // Cfg payload depends only on M/N/K + attrs (alpha/beta/transA/transB
        // from init, bias_kind from reshape) — none change per-exec, so
        // write here once.
        uint8_t cfgbuf[48] = {};
        auto put_u = [&](size_t off, uint32_t v) { std::memcpy(cfgbuf + off, &v, 4); };
        auto put_f = [&](size_t off, float v)    { std::memcpy(cfgbuf + off, &v, 4); };
        put_u(0,  (uint32_t)M);
        put_u(4,  (uint32_t)N);
        put_u(8,  (uint32_t)K);
        put_u(12, (uint32_t)transA);
        put_u(16, (uint32_t)transB);
        put_u(20, (uint32_t)bias_kind);
        put_f(32, alpha);
        put_f(36, beta);
        webgpu::get_device().queue.WriteBuffer(cfg_buf, 0, cfgbuf, sizeof(cfgbuf));
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();
        webgpu::upload_if_needed(inputs[0]);
        webgpu::upload_if_needed(inputs[1]);
        const tensor_t* c = (inputs.size() > 2 && inputs[2]) ? inputs[2] : nullptr;
        if (c) webgpu::upload_if_needed(c);

        // Cfg buffer was written in reshape() — its contents depend only
        // on M/N/K + attrs which are reshape-time constants.
        auto* ra = webgpu::find(inputs[0]);
        auto* rb = webgpu::find(inputs[1]);
        auto* ry = webgpu::find(outputs[0]);
        wgpu::Buffer c_buf;  uint64_t c_sz;
        if (c) { auto* rc = webgpu::find(c); c_buf = rc->buf; c_sz = rc->size; }
        else   { c_buf = zero_bias;                          c_sz = 16; }

        uint32_t gen_a = webgpu::generation_of(inputs[0]);
        uint32_t gen_b = webgpu::generation_of(inputs[1]);
        uint32_t gen_c = c ? webgpu::generation_of(c) : 0u;
        uint32_t gen_y = webgpu::generation_of(outputs[0]);
        if (!cached_bg || gen_a != cached_gen[0]
                       || gen_b != cached_gen[1]
                       || gen_c != cached_gen[2]
                       || gen_y != cached_gen[3]) {
            wgpu::BindGroupEntry be[5] = {};
            be[0].binding = 0; be[0].buffer = ra->buf;   be[0].offset = 0; be[0].size = ra->size;
            be[1].binding = 1; be[1].buffer = rb->buf;   be[1].offset = 0; be[1].size = rb->size;
            be[2].binding = 2; be[2].buffer = c_buf;     be[2].offset = 0; be[2].size = c_sz;
            be[3].binding = 3; be[3].buffer = ry->buf;   be[3].offset = 0; be[3].size = ry->size;
            be[4].binding = 4; be[4].buffer = cfg_buf;   be[4].offset = 0; be[4].size = 48;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 5; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_a;
            cached_gen[1] = gen_b;
            cached_gen[2] = gen_c;
            cached_gen[3] = gen_y;
        }

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, cached_bg);
        const uint32_t TILE = 16;
        pass.DispatchWorkgroups(((uint32_t)N + TILE - 1) / TILE,
                                ((uint32_t)M + TILE - 1) / TILE, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_Gemm_webgpu(int, pool_t& pool) {
    return pool_new<Gemm_operator_webgpu>(pool);
}

} // namespace nnr
