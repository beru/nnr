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

constexpr uint32_t WG = 256;

// LeakyRelu with per-op alpha. Not absorbed into unary_elementwise_t since
// that template bakes the WGSL expression at init time, whereas alpha here
// has to be re-writable per model without recompiling WGSL.
struct LeakyRelu_operator_webgpu : public operator_t {
    uint32_t n = 0;
    float    alpha = 0.01f;

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          params;

    // Cached BindGroup. Tensor-backed slots: [X, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[2] = {};

    bool init() override {
        if (inputs.size() != 1 || outputs.size() != 1) return false;
        if (!webgpu::device_ready()) return false;

        auto& dev = webgpu::get_device();
        wgpu::ShaderSourceWGSL w = {};
        w.code = webgpu::wgsl::leaky_relu;
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

        wgpu::BufferDescriptor pd = {};
        pd.size  = 16;
        pd.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        params = dev.device.CreateBuffer(&pd);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t*       y = outputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (!y->reshape_identity(x)) return false;

        alpha = attribute(attr_key_t::alpha, 0.01f);
        n     = (uint32_t)x->ndata;

        webgpu::ensure_buffer(x, x->ndata * sizeof(float));
        webgpu::ensure_buffer(y, y->ndata * sizeof(float));

        // Uniform depends only on n + alpha — both reshape-time constants.
        uint32_t alpha_bits;
        std::memcpy(&alpha_bits, &alpha, sizeof(uint32_t));
        uint8_t u[16] = {};
        std::memcpy(u + 0, &n,          4);
        std::memcpy(u + 4, &alpha_bits, 4);
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

operator_t* resolver_default_op_LeakyRelu_webgpu(int, pool_t& pool) {
    return pool_new<LeakyRelu_operator_webgpu>(pool);
}

} // namespace nnr
