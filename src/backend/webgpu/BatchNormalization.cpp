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

// BatchNormalization — inference mode only (training_mode != 0 falls back to
// CPU). NCHW assumed: channel axis is 1, per_channel is the product of all
// spatial dims (axes 2..). All four parameter tensors are [C] vectors.
struct BatchNormalization_operator_webgpu : public operator_t {
    uint32_t total = 0, channels = 0, per_channel = 0, epsilon_bits = 0;

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          params;

    // Cached BindGroup. Tensor-backed slots: [X, scale, bias, mean, var, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[6] = {};

    bool init() override {
        if (inputs.size() != 5)  return false;
        if (outputs.size() != 1) return false;   // training_mode outputs not supported
        if (!webgpu::device_ready()) return false;

        auto& dev = webgpu::get_device();
        wgpu::ShaderSourceWGSL w = {};
        w.code = webgpu::wgsl::batchnorm;
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &w;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

        wgpu::BindGroupLayoutEntry e[7] = {};
        for (int i = 0; i < 5; ++i) {
            e[i].binding = (uint32_t)i;
            e[i].visibility = wgpu::ShaderStage::Compute;
            e[i].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        }
        e[5].binding = 5; e[5].visibility = wgpu::ShaderStage::Compute;
        e[5].buffer.type = wgpu::BufferBindingType::Storage;
        e[6].binding = 6; e[6].visibility = wgpu::ShaderStage::Compute;
        e[6].buffer.type = wgpu::BufferBindingType::Uniform;
        wgpu::BindGroupLayoutDescriptor bgld = {};
        bgld.entryCount = 7; bgld.entries = e;
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
        const tensor_t* x     = inputs[0];
        const tensor_t* scale = inputs[1];
        const tensor_t* bias  = inputs[2];
        const tensor_t* mean  = inputs[3];
        const tensor_t* var   = inputs[4];
        tensor_t*       y     = outputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (scale->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (bias->type  != NNR_DATA_TYPE_FLOAT32) return false;
        if (mean->type  != NNR_DATA_TYPE_FLOAT32) return false;
        if (var->type   != NNR_DATA_TYPE_FLOAT32) return false;
        if (x->ndim < 2) return false;
        int training_mode = (int)attribute(attr_key_t::training_mode, (int64_t)0);
        if (training_mode != 0) return false;

        int C = x->dims[1];
        if ((int)scale->ndata != C) return false;
        if ((int)bias->ndata  != C) return false;
        if ((int)mean->ndata  != C) return false;
        if ((int)var->ndata   != C) return false;

        if (!y->reshape_identity(x)) return false;

        channels    = (uint32_t)C;
        total       = (uint32_t)x->ndata;
        uint32_t pc = 1;
        for (int k = 2; k < x->ndim; ++k) pc *= (uint32_t)x->dims[k];
        per_channel = pc;
        float eps   = attribute(attr_key_t::epsilon, 1e-5f);
        std::memcpy(&epsilon_bits, &eps, sizeof(uint32_t));

        webgpu::ensure_buffer(x,     x->ndata     * sizeof(float));
        webgpu::ensure_buffer(scale, scale->ndata * sizeof(float));
        webgpu::ensure_buffer(bias,  bias->ndata  * sizeof(float));
        webgpu::ensure_buffer(mean,  mean->ndata  * sizeof(float));
        webgpu::ensure_buffer(var,   var->ndata   * sizeof(float));
        webgpu::ensure_buffer(y,     y->ndata     * sizeof(float));

        uint8_t u[16] = {};
        auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(u + off, &v, 4); };
        put_u32(0,  total);
        put_u32(4,  channels);
        put_u32(8,  per_channel);
        put_u32(12, epsilon_bits);
        webgpu::get_device().queue.WriteBuffer(params, 0, u, sizeof(u));
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();
        webgpu::upload_if_needed(inputs[0]);
        webgpu::upload_if_needed(inputs[1]);
        webgpu::upload_if_needed(inputs[2]);
        webgpu::upload_if_needed(inputs[3]);
        webgpu::upload_if_needed(inputs[4]);

        auto* rx  = webgpu::find(inputs[0]);
        auto* rs  = webgpu::find(inputs[1]);
        auto* rb  = webgpu::find(inputs[2]);
        auto* rm  = webgpu::find(inputs[3]);
        auto* rv  = webgpu::find(inputs[4]);
        auto* ry  = webgpu::find(outputs[0]);

        uint32_t gens[6] = {
            webgpu::generation_of(inputs[0]),
            webgpu::generation_of(inputs[1]),
            webgpu::generation_of(inputs[2]),
            webgpu::generation_of(inputs[3]),
            webgpu::generation_of(inputs[4]),
            webgpu::generation_of(outputs[0]),
        };
        bool mismatch = !cached_bg;
        for (int i = 0; i < 6; ++i) mismatch = mismatch || (gens[i] != cached_gen[i]);
        if (mismatch) {
            wgpu::BindGroupEntry be[7] = {};
            be[0].binding = 0; be[0].buffer = rx->buf;  be[0].offset = 0; be[0].size = rx->size;
            be[1].binding = 1; be[1].buffer = rs->buf;  be[1].offset = 0; be[1].size = rs->size;
            be[2].binding = 2; be[2].buffer = rb->buf;  be[2].offset = 0; be[2].size = rb->size;
            be[3].binding = 3; be[3].buffer = rm->buf;  be[3].offset = 0; be[3].size = rm->size;
            be[4].binding = 4; be[4].buffer = rv->buf;  be[4].offset = 0; be[4].size = rv->size;
            be[5].binding = 5; be[5].buffer = ry->buf;  be[5].offset = 0; be[5].size = ry->size;
            be[6].binding = 6; be[6].buffer = params;   be[6].offset = 0; be[6].size = 16;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 7; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            for (int i = 0; i < 6; ++i) cached_gen[i] = gens[i];
        }

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, cached_bg);
        uint32_t groups = (total + WG - 1) / WG;
        pass.DispatchWorkgroups(groups, 1, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_BatchNormalization_webgpu(int, pool_t& pool) {
    return pool_new<BatchNormalization_operator_webgpu>(pool);
}

} // namespace nnr
