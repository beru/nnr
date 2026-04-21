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

// Large finite sentinels used when an ONNX Clip lacks a min/max bound.
// Exact f32 MIN/MAX would also work (we pass raw bytes, not WGSL literals),
// but these are comfortably far from the range of anything a real model
// will see while staying well clear of Tint's literal-edge restrictions
// should we ever want to inline them.
constexpr float FP_HI =  3.4e38f;
constexpr float FP_LO = -3.4e38f;

// ONNX Clip: elementwise clamp. min/max come from optional scalar inputs
// 1 and 2 (matches the CPU backend — the opset <11 attribute form is
// normalized to inputs upstream).
struct Clip_operator_webgpu : public operator_t {
    uint32_t n = 0;
    float    lo = FP_LO;
    float    hi = FP_HI;

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          params;

    // Cached BindGroup. Tensor-backed slots: [X, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[2] = {};

    bool init() override {
        if (inputs.size() < 1 || inputs.size() > 3) return false;
        if (outputs.size() != 1) return false;
        if (!webgpu::device_ready()) return false;

        auto& dev = webgpu::get_device();
        wgpu::ShaderSourceWGSL w = {};
        w.code = webgpu::wgsl::clip;
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
        pld.bindGroupLayoutCount = 1;
        pld.bindGroupLayouts = &bgl;
        wgpu::PipelineLayout pl = dev.device.CreatePipelineLayout(&pld);

        wgpu::ComputePipelineDescriptor cpd = {};
        cpd.layout = pl;
        cpd.compute.module = sm;
        cpd.compute.entryPoint = "main";
        pipeline = dev.device.CreateComputePipeline(&cpd);

        wgpu::BufferDescriptor d = {};
        d.size  = 16;
        d.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        params = dev.device.CreateBuffer(&d);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t*       y = outputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (!y->reshape_identity(x)) return false;

        // Opset 11+: min/max via optional scalar inputs 1, 2. Absent or empty
        // inputs mean "unbounded"; the CPU data pointer is our proxy for
        // "present and populated".
        lo = FP_LO; hi = FP_HI;
        if (inputs.size() >= 2 && inputs[1] && inputs[1]->data && inputs[1]->ndata >= 1) {
            if (inputs[1]->type != NNR_DATA_TYPE_FLOAT32) return false;
            lo = ((const float*)inputs[1]->data)[0];
        }
        if (inputs.size() >= 3 && inputs[2] && inputs[2]->data && inputs[2]->ndata >= 1) {
            if (inputs[2]->type != NNR_DATA_TYPE_FLOAT32) return false;
            hi = ((const float*)inputs[2]->data)[0];
        }
        n = (uint32_t)x->ndata;
        webgpu::ensure_buffer(x, x->ndata * sizeof(float));
        webgpu::ensure_buffer(y, y->ndata * sizeof(float));

        // Uniform contents depend only on reshape-time data — write here.
        uint8_t u[16] = {};
        std::memcpy(u + 0, &n,  4);
        std::memcpy(u + 4, &lo, 4);
        std::memcpy(u + 8, &hi, 4);
        webgpu::get_device().queue.WriteBuffer(params, 0, u, sizeof(u));
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
            be[0].binding = 0; be[0].buffer = rx->buf; be[0].offset = 0; be[0].size = rx->size;
            be[1].binding = 1; be[1].buffer = ry->buf; be[1].offset = 0; be[1].size = ry->size;
            be[2].binding = 2; be[2].buffer = params;  be[2].offset = 0; be[2].size = 16;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 3; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_x;
            cached_gen[1] = gen_y;
        }

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, cached_bg);
        uint32_t groups = (n + WG - 1) / WG;
        pass.DispatchWorkgroups(groups, 1, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_Clip_webgpu(int, pool_t& pool) {
    return pool_new<Clip_operator_webgpu>(pool);
}

} // namespace nnr
