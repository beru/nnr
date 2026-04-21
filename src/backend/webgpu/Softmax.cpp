#include "nnr.h"
#include "attr_key.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include "shaders.h"

#include <webgpu/webgpu_cpp.h>

namespace nnr {

namespace {

// Softmax (opset 13+). Normalizes along a single axis per the opset-13+
// semantics (no coerced-to-2D). The kernel reduces N = dims[axis] values per
// (outer, inner) pair, where outer = prod(dims[..axis]) and inner =
// prod(dims[axis+1..]). The last-axis case (inner == 1) is the fast path
// that folds the stride-multiply to + j; the strided case (inner > 1) uses
// `base + j*stride` access.
struct Softmax_operator_webgpu : public operator_t {
    int axis_attr = -1;
    int outer = 0;
    int N     = 0;
    int inner = 0;

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          uniforms;

    // Cached BindGroup. Tensor-backed slots: [X, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[2] = {};

    bool init() override {
        if (!is_inout_size(1, 1))     return false;
        if (!webgpu::device_ready())  return false;
        axis_attr = attribute(attr_key_t::axis, -1);

        auto& dev = webgpu::get_device();
        wgpu::ShaderSourceWGSL w = {};
        w.code = webgpu::wgsl::softmax;
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

        wgpu::BufferDescriptor ud = {};
        ud.size = 16;
        ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        uniforms = dev.device.CreateBuffer(&ud);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t*       y = outputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return false;

        int caxis = axis_attr < 0 ? axis_attr + x->ndim : axis_attr;
        if (caxis < 0 || caxis >= x->ndim) return false;

        outer = 1;
        for (int i = 0; i < caxis; ++i)           outer *= x->dims[i];
        N = x->dims[caxis];
        inner = 1;
        for (int i = caxis + 1; i < x->ndim; ++i) inner *= x->dims[i];

        if (!y->reshape_identity(x)) return false;
        webgpu::ensure_buffer(x, x->ndata * sizeof(float));
        webgpu::ensure_buffer(y, y->ndata * sizeof(float));

        // Uniform depends only on outer/N/inner — write here.
        uint32_t u[4] = { (uint32_t)outer, (uint32_t)N, (uint32_t)inner, 0 };
        webgpu::get_device().queue.WriteBuffer(uniforms, 0, u, sizeof(u));
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();
        webgpu::upload_if_needed(inputs[0]);

        // Uniform was written in reshape() — its contents depend only on
        // outer/N/inner which are reshape-time constants.
        auto* rx = webgpu::find(inputs[0]);
        auto* ry = webgpu::find(outputs[0]);

        uint32_t gen_x = webgpu::generation_of(inputs[0]);
        uint32_t gen_y = webgpu::generation_of(outputs[0]);
        if (!cached_bg || gen_x != cached_gen[0] || gen_y != cached_gen[1]) {
            wgpu::BindGroupEntry be[3] = {};
            be[0].binding = 0; be[0].buffer = rx->buf; be[0].offset = 0; be[0].size = rx->size;
            be[1].binding = 1; be[1].buffer = ry->buf; be[1].offset = 0; be[1].size = ry->size;
            be[2].binding = 2; be[2].buffer = uniforms; be[2].offset = 0; be[2].size = 16;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 3; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_x;
            cached_gen[1] = gen_y;
        }

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, cached_bg);
        pass.DispatchWorkgroups((uint32_t)outer, (uint32_t)inner, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_Softmax_webgpu(int, pool_t& pool) {
    return pool_new<Softmax_operator_webgpu>(pool);
}

} // namespace nnr
