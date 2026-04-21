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

struct Expand_operator_webgpu : public operator_t {
    uint32_t total = 0;
    uint32_t ndim  = 0;
    uint32_t out_dims_u[8]   = {};
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
        if (!(inputs.size() == 2 && outputs.size() == 1)) return false;
        if (!webgpu::device_ready()) return false;

        auto& dev = webgpu::get_device();
        wgpu::ShaderSourceWGSL w = {};
        w.code = webgpu::wgsl::expand;
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

        wgpu::BufferDescriptor md = {};
        md.size = 128;
        md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        meta_buf = dev.device.CreateBuffer(&md);
        return true;
    }

    bool reshape() override {
        const tensor_t* data  = inputs[0];
        const tensor_t* shape = inputs[1];
        tensor_t*       y     = outputs[0];
        if (data->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (shape->type != NNR_DATA_TYPE_INT64)  return false;
        if (shape->ndim != 1)                    return false;
        if (!shape->data)                        return false;   // shape must be CPU-resident

        // Target shape is a 1D int64 tensor of length = target rank.
        int target_ndim = (int)shape->ndata;
        if (target_ndim < 0 || target_ndim > 8) return false;
        const int64_t* shape_data = (const int64_t*)shape->data;
        int target_dims[8] = {};
        for (int k = 0; k < target_ndim; ++k) target_dims[k] = (int)shape_data[k];

        // Bidirectional broadcast between data.shape and target_dims.
        int out_ndim = data->ndim > target_ndim ? data->ndim : target_ndim;
        if (out_ndim > 8) return false;
        int out_dims[8] = {};
        for (int k = 0; k < out_ndim; ++k) {
            int rd = k - (out_ndim - data->ndim);
            int rt = k - (out_ndim - target_ndim);
            int dd = rd >= 0 ? data->dims[rd] : 1;
            int dt = rt >= 0 ? target_dims[rt] : 1;
            if (dd != dt && dd != 1 && dt != 1) return false;
            out_dims[k] = dd > dt ? dd : dt;
        }
        if (!y->reshape(std::span<const int>(out_dims, out_ndim), data->type)) return false;

        // Natural row-major strides over data's own rank; broadcast-aware
        // strides aligned to output rank (stride 0 for size-1 / missing axes).
        uint32_t in_nat[8] = {};
        {
            uint32_t s = 1;
            for (int i = data->ndim - 1; i >= 0; --i) { in_nat[i] = s; s *= (uint32_t)data->dims[i]; }
        }
        for (int k = 0; k < 8; ++k) { out_dims_u[k] = 0; in_strides_u[k] = 0; }
        for (int k = 0; k < out_ndim; ++k) {
            out_dims_u[k] = (uint32_t)out_dims[k];
            int rd = k - (out_ndim - data->ndim);
            in_strides_u[k] = (rd >= 0 && data->dims[rd] != 1) ? in_nat[rd] : 0u;
        }
        ndim = (uint32_t)out_ndim;
        total = 1;
        for (int k = 0; k < out_ndim; ++k) total *= (uint32_t)out_dims[k];

        webgpu::ensure_buffer(data, data->ndata * sizeof(float));
        webgpu::ensure_buffer(y, y->ndata * sizeof(float));

        // Meta layout (matches expand.wgsl):
        //   [0..4)    total
        //   [4..8)    ndim
        //   [8..12)   grid_stride_x (threads-x for 2D dispatch split)
        //   [12..16)  pad
        //   [16..48)  out_dims     (8 x u32)
        //   [48..80)  in_strides   (8 x u32)
        uint32_t groups = (total + WG - 1) / WG;
        webgpu::dispatch_1d_grid(groups, dispatch_gx, dispatch_gy);
        uint8_t buf[128] = {};
        auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(buf + off, &v, 4); };
        put_u32(0, total);
        put_u32(4, ndim);
        put_u32(8, dispatch_gx * WG);
        for (int i = 0; i < 8; ++i) put_u32(16 + i * 4, out_dims_u[i]);
        for (int i = 0; i < 8; ++i) put_u32(48 + i * 4, in_strides_u[i]);
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

operator_t* resolver_default_op_Expand_webgpu(int, pool_t& pool) {
    return pool_new<Expand_operator_webgpu>(pool);
}

} // namespace nnr
