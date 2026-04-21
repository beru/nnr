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

constexpr uint32_t WG_STATS = 64;
constexpr uint32_t WG_APPLY = 256;

// InstanceNormalization (NCHW, any spatial rank). Per-sample, per-channel
// mean/variance over the spatial dims. Two-kernel implementation: a stats
// pass produces (mean, inv_std) for each (n, c), and an apply pass uses
// those to normalize + affine-transform every output element.
struct InstanceNormalization_operator_webgpu : public operator_t {
    uint32_t N = 0, C = 0, HW = 0;
    uint32_t epsilon_bits = 0;

    wgpu::ComputePipeline pipeline_stats;
    wgpu::ComputePipeline pipeline_apply;
    wgpu::BindGroupLayout bgl_stats;
    wgpu::BindGroupLayout bgl_apply;
    wgpu::Buffer          stats_buf;     // sized per reshape (N*C*2 floats)
    uint32_t              stats_capacity = 0;
    uint32_t              stats_gen      = 0;  // bumped when stats_buf reallocated
    wgpu::Buffer          params_stats;  // 16B
    wgpu::Buffer          params_apply;  // 16B

    // Cached BindGroups. Stats slots: [X, stats_buf]; Apply slots: [X, Scale, Bias, stats_buf, Y].
    wgpu::BindGroup cached_bg_stats;
    uint32_t        cached_gen_stats[2] = {};   // [gen_X, stats_gen]
    wgpu::BindGroup cached_bg_apply;
    uint32_t        cached_gen_apply[5] = {};   // [gen_X, gen_Scale, gen_Bias, stats_gen, gen_Y]

    bool init() override {
        if (inputs.size() != 3 || outputs.size() != 1) return false;
        if (!webgpu::device_ready()) return false;

        auto& dev = webgpu::get_device();

        auto make_module = [&](const char* src) {
            wgpu::ShaderSourceWGSL w = {};
            w.code = src;
            wgpu::ShaderModuleDescriptor d = {};
            d.nextInChain = &w;
            return dev.device.CreateShaderModule(&d);
        };

        // Stats pipeline: X (read), Stats (r/w), Params (uniform).
        {
            wgpu::BindGroupLayoutEntry e[3] = {};
            e[0].binding = 0; e[0].visibility = wgpu::ShaderStage::Compute;
            e[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
            e[1].binding = 1; e[1].visibility = wgpu::ShaderStage::Compute;
            e[1].buffer.type = wgpu::BufferBindingType::Storage;
            e[2].binding = 2; e[2].visibility = wgpu::ShaderStage::Compute;
            e[2].buffer.type = wgpu::BufferBindingType::Uniform;
            wgpu::BindGroupLayoutDescriptor bgld = {};
            bgld.entryCount = 3; bgld.entries = e;
            bgl_stats = dev.device.CreateBindGroupLayout(&bgld);

            wgpu::PipelineLayoutDescriptor pld = {};
            pld.bindGroupLayoutCount = 1;
            pld.bindGroupLayouts = &bgl_stats;
            wgpu::PipelineLayout pl = dev.device.CreatePipelineLayout(&pld);
            wgpu::ComputePipelineDescriptor cpd = {};
            cpd.layout = pl;
            cpd.compute.module = make_module(webgpu::wgsl::instance_norm_stats);
            cpd.compute.entryPoint = "main";
            pipeline_stats = dev.device.CreateComputePipeline(&cpd);
        }

        // Apply pipeline: X, Scale, Bias, Stats (all read), Y (write), Params.
        {
            wgpu::BindGroupLayoutEntry e[6] = {};
            for (int i = 0; i < 4; ++i) {
                e[i].binding = (uint32_t)i;
                e[i].visibility = wgpu::ShaderStage::Compute;
                e[i].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
            }
            e[4].binding = 4; e[4].visibility = wgpu::ShaderStage::Compute;
            e[4].buffer.type = wgpu::BufferBindingType::Storage;
            e[5].binding = 5; e[5].visibility = wgpu::ShaderStage::Compute;
            e[5].buffer.type = wgpu::BufferBindingType::Uniform;
            wgpu::BindGroupLayoutDescriptor bgld = {};
            bgld.entryCount = 6; bgld.entries = e;
            bgl_apply = dev.device.CreateBindGroupLayout(&bgld);

            wgpu::PipelineLayoutDescriptor pld = {};
            pld.bindGroupLayoutCount = 1;
            pld.bindGroupLayouts = &bgl_apply;
            wgpu::PipelineLayout pl = dev.device.CreatePipelineLayout(&pld);
            wgpu::ComputePipelineDescriptor cpd = {};
            cpd.layout = pl;
            cpd.compute.module = make_module(webgpu::wgsl::instance_norm_apply);
            cpd.compute.entryPoint = "main";
            pipeline_apply = dev.device.CreateComputePipeline(&cpd);
        }

        wgpu::BufferDescriptor pd = {};
        pd.size  = 16;
        pd.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        params_stats = dev.device.CreateBuffer(&pd);
        params_apply = dev.device.CreateBuffer(&pd);
        return true;
    }

    bool reshape() override {
        const tensor_t* X     = inputs[0];
        const tensor_t* Scale = inputs[1];
        const tensor_t* Bias  = inputs[2];
        tensor_t*       Y     = outputs[0];
        if (X->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (Scale->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (Bias->type  != NNR_DATA_TYPE_FLOAT32) return false;
        if (X->ndim < 2) return false;
        N  = (uint32_t)X->dims[0];
        C  = (uint32_t)X->dims[1];
        if ((int)Scale->ndata != (int)C) return false;
        if ((int)Bias->ndata  != (int)C) return false;
        uint32_t hw = 1;
        for (int k = 2; k < X->ndim; ++k) hw *= (uint32_t)X->dims[k];
        HW = hw;

        float eps = attribute(attr_key_t::epsilon, 1e-5f);
        std::memcpy(&epsilon_bits, &eps, sizeof(uint32_t));

        if (!Y->reshape_identity(X)) return false;

        auto& dev = webgpu::get_device();
        uint32_t needed = N * C * 2u * (uint32_t)sizeof(float);
        if (needed < 16u) needed = 16u;
        if (stats_capacity < needed || !stats_buf) {
            wgpu::BufferDescriptor bd = {};
            bd.size  = needed;
            bd.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
            stats_buf      = dev.device.CreateBuffer(&bd);
            stats_capacity = needed;
            ++stats_gen;
        }

        webgpu::ensure_buffer(X,     X->ndata     * sizeof(float));
        webgpu::ensure_buffer(Scale, Scale->ndata * sizeof(float));
        webgpu::ensure_buffer(Bias,  Bias->ndata  * sizeof(float));
        webgpu::ensure_buffer(Y,     Y->ndata     * sizeof(float));

        uint32_t NC   = N * C;
        uint32_t total = NC * HW;

        uint8_t pu_stats[16] = {};
        std::memcpy(pu_stats + 0, &NC,           4);
        std::memcpy(pu_stats + 4, &HW,           4);
        std::memcpy(pu_stats + 8, &epsilon_bits, 4);
        dev.queue.WriteBuffer(params_stats, 0, pu_stats, sizeof(pu_stats));

        uint8_t pu_apply[16] = {};
        std::memcpy(pu_apply + 0, &total, 4);
        std::memcpy(pu_apply + 4, &C,     4);
        std::memcpy(pu_apply + 8, &HW,    4);
        dev.queue.WriteBuffer(params_apply, 0, pu_apply, sizeof(pu_apply));
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();
        webgpu::upload_if_needed(inputs[0]);
        webgpu::upload_if_needed(inputs[1]);
        webgpu::upload_if_needed(inputs[2]);

        uint32_t NC    = N * C;
        uint32_t total = NC * HW;

        auto* rx = webgpu::find(inputs[0]);
        auto* rs = webgpu::find(inputs[1]);
        auto* rb = webgpu::find(inputs[2]);
        auto* ry = webgpu::find(outputs[0]);

        uint32_t gen_x = webgpu::generation_of(inputs[0]);
        uint32_t gen_s = webgpu::generation_of(inputs[1]);
        uint32_t gen_b = webgpu::generation_of(inputs[2]);
        uint32_t gen_y = webgpu::generation_of(outputs[0]);

        if (!cached_bg_stats || cached_gen_stats[0] != gen_x || cached_gen_stats[1] != stats_gen) {
            wgpu::BindGroupEntry be[3] = {};
            be[0].binding = 0; be[0].buffer = rx->buf;      be[0].offset = 0; be[0].size = rx->size;
            be[1].binding = 1; be[1].buffer = stats_buf;    be[1].offset = 0; be[1].size = stats_capacity;
            be[2].binding = 2; be[2].buffer = params_stats; be[2].offset = 0; be[2].size = 16;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl_stats; bgd.entryCount = 3; bgd.entries = be;
            cached_bg_stats = dev.device.CreateBindGroup(&bgd);
            cached_gen_stats[0] = gen_x;
            cached_gen_stats[1] = stats_gen;
        }

        if (!cached_bg_apply
            || cached_gen_apply[0] != gen_x
            || cached_gen_apply[1] != gen_s
            || cached_gen_apply[2] != gen_b
            || cached_gen_apply[3] != stats_gen
            || cached_gen_apply[4] != gen_y) {
            wgpu::BindGroupEntry be[6] = {};
            be[0].binding = 0; be[0].buffer = rx->buf;      be[0].offset = 0; be[0].size = rx->size;
            be[1].binding = 1; be[1].buffer = rs->buf;      be[1].offset = 0; be[1].size = rs->size;
            be[2].binding = 2; be[2].buffer = rb->buf;      be[2].offset = 0; be[2].size = rb->size;
            be[3].binding = 3; be[3].buffer = stats_buf;    be[3].offset = 0; be[3].size = stats_capacity;
            be[4].binding = 4; be[4].buffer = ry->buf;      be[4].offset = 0; be[4].size = ry->size;
            be[5].binding = 5; be[5].buffer = params_apply; be[5].offset = 0; be[5].size = 16;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl_apply; bgd.entryCount = 6; bgd.entries = be;
            cached_bg_apply = dev.device.CreateBindGroup(&bgd);
            cached_gen_apply[0] = gen_x;
            cached_gen_apply[1] = gen_s;
            cached_gen_apply[2] = gen_b;
            cached_gen_apply[3] = stats_gen;
            cached_gen_apply[4] = gen_y;
        }

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();
        pass.SetPipeline(pipeline_stats);
        pass.SetBindGroup(0, cached_bg_stats);
        pass.DispatchWorkgroups((NC + WG_STATS - 1) / WG_STATS, 1, 1);
        pass.SetPipeline(pipeline_apply);
        pass.SetBindGroup(0, cached_bg_apply);
        pass.DispatchWorkgroups((total + WG_APPLY - 1) / WG_APPLY, 1, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_InstanceNormalization_webgpu(int, pool_t& pool) {
    return pool_new<InstanceNormalization_operator_webgpu>(pool);
}

} // namespace nnr
