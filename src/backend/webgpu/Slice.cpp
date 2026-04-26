#include "nnr.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include "shaders.h"

#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace nnr {

namespace {

constexpr uint32_t WG = 256;

// ONNX Slice (opset 10+). Reads starts/ends/axes/steps from inputs 1..4.
// All index tensors live on CPU (they're typically model constants). Only
// the positive-step path runs on GPU; negative steps return false from
// reshape so the registry falls back to CPU.
struct Slice_operator_webgpu : public operator_t {
    uint32_t total = 0;
    uint32_t ndim  = 0;
    uint32_t base_flat = 0;
    uint32_t out_dims_u[8]    = {};
    uint32_t eff_strides_u[8] = {};
    bool     empty_output     = false;

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          meta_buf;
    uint32_t              dispatch_gx = 0;
    uint32_t              dispatch_gy = 0;

    // Cached BindGroup. Tensor-backed slots: [X, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[2] = {};

    static int64_t read_index(const tensor_t* t, int i) {
        if (t->type == NNR_DATA_TYPE_INT32) return ((const int32_t*)t->data)[i];
        return ((const int64_t*)t->data)[i];
    }

    static int clamp_index_pos(int64_t idx, int dim_size) {
        if (idx < 0) idx = 0;
        if (idx > dim_size) idx = dim_size;
        return (int)idx;
    }

    bool init() override {
        if (outputs.size() != 1) return false;
        if (inputs.size() < 3 || inputs.size() > 5) return false;  // opset 10+ only
        if (!webgpu::device_ready()) return false;

        auto& dev = webgpu::get_device();
        wgpu::ShaderSourceWGSL w = {};
        w.code = webgpu::wgsl::slice;
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
        md.size  = 128;
        md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        meta_buf = dev.device.CreateBuffer(&md);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t*       y = outputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return false;
        int dim = x->ndim;
        if (dim <= 0 || dim > 8) return false;

        const tensor_t* t_starts = inputs[1];
        const tensor_t* t_ends   = inputs[2];
        if (!t_starts->data || !t_ends->data) return false;
        if (t_starts->ndata != t_ends->ndata) return false;
        int nslices = (int)t_starts->ndata;

        int starts_per_axis[8], ends_per_axis[8], steps_per_axis[8];
        for (int i = 0; i < dim; ++i) { starts_per_axis[i] = 0; ends_per_axis[i] = x->dims[i]; steps_per_axis[i] = 1; }

        for (int i = 0; i < nslices; ++i) {
            int a;
            if (inputs.size() > 3 && inputs[3]) {
                const tensor_t* t_axes = inputs[3];
                if (!t_axes->data) return false;
                a = (int)read_index(t_axes, i);
                if (a < 0) a += dim;
            } else {
                a = i;
            }
            if (a < 0 || a >= dim) return false;

            int step = 1;
            if (inputs.size() > 4 && inputs[4]) {
                const tensor_t* t_steps = inputs[4];
                if (!t_steps->data) return false;
                step = (int)read_index(t_steps, i);
            }
            if (step <= 0) return false;   // negative/zero step → CPU fallback

            int64_t start_v = read_index(t_starts, i);
            int64_t end_v   = read_index(t_ends,   i);
            int d = x->dims[a];
            if (start_v >= -(int64_t)d && start_v < 0) start_v += d;
            if (end_v   >= -(int64_t)d && end_v   < 0) end_v   += d;

            starts_per_axis[a] = clamp_index_pos(start_v, d);
            ends_per_axis[a]   = clamp_index_pos(end_v,   d);
            steps_per_axis[a]  = step;
        }

        // Output dims and empty-slice detection.
        int out_dims[8];
        total = 1;
        empty_output = false;
        for (int i = 0; i < dim; ++i) {
            int s = starts_per_axis[i], e = ends_per_axis[i], st = steps_per_axis[i];
            int len = (e - s + st - 1) / st;
            if (len < 0) len = 0;
            if (len == 0) empty_output = true;
            out_dims[i] = len;
            total *= (uint32_t)len;
        }
        if (!y->reshape(std::span<const int>(out_dims, dim), x->type)) return false;

        // Natural row-major strides for input; base = sum(start[k] * in_stride[k]);
        // eff_stride[k] = step[k] * in_stride[k].
        uint32_t in_strides[8] = {};
        {
            uint32_t s = 1;
            for (int i = dim - 1; i >= 0; --i) { in_strides[i] = s; s *= (uint32_t)x->dims[i]; }
        }
        base_flat = 0;
        for (int k = 0; k < 8; ++k) { out_dims_u[k] = 0; eff_strides_u[k] = 0; }
        for (int i = 0; i < dim; ++i) {
            out_dims_u[i]    = (uint32_t)out_dims[i];
            eff_strides_u[i] = (uint32_t)steps_per_axis[i] * in_strides[i];
            base_flat       += (uint32_t)starts_per_axis[i] * in_strides[i];
        }
        ndim = (uint32_t)dim;

        webgpu::ensure_buffer(x, x->ndata * sizeof(float));
        webgpu::ensure_buffer(y, y->ndata * sizeof(float));

        // Meta layout (matches slice.wgsl):
        //   [0..4)    total
        //   [4..8)    ndim
        //   [8..12)   base
        //   [12..16)  grid_stride_x (threads-x for 2D dispatch split)
        //   [16..48)  out_dims     (8 x u32)
        //   [48..80)  eff_strides  (8 x u32)
        dispatch_gx = 0; dispatch_gy = 0;
        if (!(empty_output || total == 0)) {
            uint32_t groups = (total + WG - 1) / WG;
            webgpu::dispatch_1d_grid(groups, dispatch_gx, dispatch_gy);
            uint8_t buf[128] = {};
            auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(buf + off, &v, 4); };
            put_u32(0,  total);
            put_u32(4,  ndim);
            put_u32(8,  base_flat);
            put_u32(12, dispatch_gx * WG);
            for (int i = 0; i < 8; ++i) put_u32(16 + i * 4, out_dims_u[i]);
            for (int i = 0; i < 8; ++i) put_u32(48 + i * 4, eff_strides_u[i]);
            webgpu::get_device().queue.WriteBuffer(meta_buf, 0, buf, sizeof(buf));
        }
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();
        // Empty output: nothing to compute. Still mark gpu_written so the
        // residency tracker reflects that Y is the canonical copy.
        if (empty_output || total == 0) {
            webgpu::mark_gpu_written(outputs[0]);
            return true;
        }
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

operator_t* resolver_default_op_Slice_webgpu(int, pool_t& pool) {
    return pool_new<Slice_operator_webgpu>(pool);
}

} // namespace nnr
