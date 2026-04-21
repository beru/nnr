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

// LayerNormalization (opset 17+). Normalizes the suffix `dims[axis..ndim-1]`
// per ONNX spec — the normalization region is flattened into a single vector
// of length N = prod(dims[axis..]) and reduced in one pass. Works for axis =
// last (the transformer hot path, N = dims[-1]) as well as axis < last
// (multi-axis suffix, e.g. [B, S, D] with axis=-2 reduces S*D per row).
// Optional Mean / InvStdDev outputs (training signals) are not produced —
// the memory planner leaves them as unused tensors.
struct LayerNorm_operator_webgpu : public operator_t {
    int64_t axis_attr = -1;
    float   epsilon   = 1e-5f;

    int outer = 0;
    int N     = 0;

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          uniforms_dims;
    wgpu::Buffer          uniforms_cfg;
    wgpu::Buffer          zero_bias;   // bound when bias input is absent

    // Cached BindGroup. Tensor-backed slots: [X, scale, bias|0, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[4] = {};

    bool init() override {
        if (inputs.size() < 2 || outputs.empty()) return false;
        if (!webgpu::device_ready()) return false;
        axis_attr = attribute(attr_key_t::axis, (int64_t)-1);
        epsilon   = attribute(attr_key_t::epsilon, 1e-5f);

        auto& dev = webgpu::get_device();
        wgpu::ShaderSourceWGSL w = {};
        w.code = webgpu::wgsl::layer_norm;
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &w;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

        wgpu::BindGroupLayoutEntry e[6] = {};
        for (int i = 0; i < 3; ++i) {
            e[i].binding = (uint32_t)i;
            e[i].visibility = wgpu::ShaderStage::Compute;
            e[i].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        }
        e[3].binding = 3; e[3].visibility = wgpu::ShaderStage::Compute;
        e[3].buffer.type = wgpu::BufferBindingType::Storage;
        e[4].binding = 4; e[4].visibility = wgpu::ShaderStage::Compute;
        e[4].buffer.type = wgpu::BufferBindingType::Uniform;
        e[5].binding = 5; e[5].visibility = wgpu::ShaderStage::Compute;
        e[5].buffer.type = wgpu::BufferBindingType::Uniform;
        wgpu::BindGroupLayoutDescriptor bgld = {};
        bgld.entryCount = 6; bgld.entries = e;
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

        wgpu::BufferDescriptor ud = {};
        ud.size = 16;
        ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        uniforms_dims = dev.device.CreateBuffer(&ud);
        uniforms_cfg  = dev.device.CreateBuffer(&ud);

        // epsilon is set from attribute() in init() and never changes
        // afterward — write the cfg uniform once here.
        float cfg[4] = { epsilon, 0.0f, 0.0f, 0.0f };
        dev.queue.WriteBuffer(uniforms_cfg, 0, cfg, sizeof(cfg));
        return true;
    }

    bool reshape() override {
        const tensor_t* x     = inputs[0];
        const tensor_t* scale = inputs[1];
        const tensor_t* bias  = (inputs.size() > 2 && inputs[2]) ? inputs[2] : nullptr;
        tensor_t*       y     = outputs[0];

        if (x->type != NNR_DATA_TYPE_FLOAT32)     return false;
        if (scale->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (bias && bias->type != NNR_DATA_TYPE_FLOAT32) return false;

        int caxis = axis_attr < 0 ? (int)axis_attr + x->ndim : (int)axis_attr;
        if (caxis < 0 || caxis >= x->ndim) return false;

        outer = 1;
        for (int i = 0; i < caxis; ++i)    outer *= x->dims[i];
        N = 1;
        for (int i = caxis; i < x->ndim; ++i) N *= x->dims[i];

        if ((int)scale->ndata != N) return false;
        if (bias && (int)bias->ndata != N) return false;

        if (!y->reshape_identity(x)) return false;
        webgpu::ensure_buffer(x, x->ndata * sizeof(float));
        webgpu::ensure_buffer(scale, N * sizeof(float));
        if (bias) webgpu::ensure_buffer(bias, N * sizeof(float));
        webgpu::ensure_buffer(y, y->ndata * sizeof(float));

        // Bias is optional; when absent we bind a zero buffer of length N.
        if (!bias) {
            auto& dev = webgpu::get_device();
            wgpu::BufferDescriptor bd = {};
            bd.size = ((size_t)N * sizeof(float) + 3u) & ~size_t{3};
            bd.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
            zero_bias = dev.device.CreateBuffer(&bd);
            std::vector<float> zeros(N, 0.0f);
            dev.queue.WriteBuffer(zero_bias, 0, zeros.data(), N * sizeof(float));
        }

        // Dims uniform depends only on outer/N, both set above. Write once.
        uint32_t dims[4] = { (uint32_t)outer, (uint32_t)N, 0, 0 };
        webgpu::get_device().queue.WriteBuffer(uniforms_dims, 0, dims, sizeof(dims));
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();
        webgpu::upload_if_needed(inputs[0]);
        webgpu::upload_if_needed(inputs[1]);
        const tensor_t* bias = (inputs.size() > 2 && inputs[2]) ? inputs[2] : nullptr;
        if (bias) webgpu::upload_if_needed(bias);

        // uniforms_dims and uniforms_cfg were written in reshape()/init().
        auto* rx = webgpu::find(inputs[0]);
        auto* rs = webgpu::find(inputs[1]);
        auto* ry = webgpu::find(outputs[0]);
        wgpu::Buffer bias_buf;
        uint64_t     bias_size;
        if (bias) {
            auto* rb = webgpu::find(bias);
            bias_buf  = rb->buf;
            bias_size = rb->size;
        } else {
            bias_buf  = zero_bias;
            bias_size = (uint64_t)N * sizeof(float);
        }

        uint32_t gen_x    = webgpu::generation_of(inputs[0]);
        uint32_t gen_s    = webgpu::generation_of(inputs[1]);
        uint32_t gen_bias = bias ? webgpu::generation_of(bias) : 0u;
        uint32_t gen_y    = webgpu::generation_of(outputs[0]);
        if (!cached_bg || gen_x    != cached_gen[0]
                       || gen_s    != cached_gen[1]
                       || gen_bias != cached_gen[2]
                       || gen_y    != cached_gen[3]) {
            wgpu::BindGroupEntry be[6] = {};
            be[0].binding = 0; be[0].buffer = rx->buf;        be[0].offset = 0; be[0].size = rx->size;
            be[1].binding = 1; be[1].buffer = rs->buf;        be[1].offset = 0; be[1].size = rs->size;
            be[2].binding = 2; be[2].buffer = bias_buf;       be[2].offset = 0; be[2].size = bias_size;
            be[3].binding = 3; be[3].buffer = ry->buf;        be[3].offset = 0; be[3].size = ry->size;
            be[4].binding = 4; be[4].buffer = uniforms_dims;  be[4].offset = 0; be[4].size = 16;
            be[5].binding = 5; be[5].buffer = uniforms_cfg;   be[5].offset = 0; be[5].size = 16;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 6; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_x;
            cached_gen[1] = gen_s;
            cached_gen[2] = gen_bias;
            cached_gen[3] = gen_y;
        }

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, cached_bg);
        pass.DispatchWorkgroups((uint32_t)outer, 1, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_LayerNormalization_webgpu(int, pool_t& pool) {
    return pool_new<LayerNorm_operator_webgpu>(pool);
}

} // namespace nnr
