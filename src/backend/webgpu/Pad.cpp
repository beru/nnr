#include "nnr.h"
#include "attr_key.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include "shaders.h"

#include <webgpu/webgpu_cpp.h>

#include <cstring>
#include <string_view>

namespace nnr {

namespace {

constexpr uint32_t WG = 256;

// ONNX Pad opset 11+ constant mode. Non-constant modes (reflect/edge/wrap)
// return false so the registry falls back to CPU. Pads and constant_value
// are read from inputs 1 (int64[2*rank]) and 2 (optional scalar). Older
// opsets pass pads/value as attributes; that path is not currently wired.
struct Pad_operator_webgpu : public operator_t {
    uint32_t total = 0;
    uint32_t ndim  = 0;
    uint32_t pad_value_bits = 0;
    uint32_t out_dims_u[8]    = {};
    uint32_t in_dims_u[8]     = {};
    uint32_t in_strides_u[8]  = {};
    uint32_t pad_starts_u[8]  = {};
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

    bool init() override {
        if (outputs.size() != 1) return false;
        if (inputs.size() < 2 || inputs.size() > 4) return false;
        if (!webgpu::device_ready()) return false;

        auto& dev = webgpu::get_device();
        wgpu::ShaderSourceWGSL w = {};
        w.code = webgpu::wgsl::pad;
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

        // Header (16B) + 6 u32[8] arrays (192B) = 208B, rounded up to 256.
        wgpu::BufferDescriptor md = {};
        md.size  = 256;
        md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        meta_buf = dev.device.CreateBuffer(&md);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t*       y = outputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return false;
        ndim = (uint32_t)x->ndim;
        if ((int)ndim <= 0 || (int)ndim > 8) return false;

        // Only constant mode is implemented; anything else goes to CPU.
        std::string_view mode = attribute(attr_key_t::mode, "constant");
        if (mode != "constant") return false;

        // Pads input: int64[2*rank] = [start_0..start_{R-1}, end_0..end_{R-1}].
        const tensor_t* t_pads = inputs[1];
        if (!t_pads->data) return false;
        if ((int)t_pads->ndata != 2 * (int)ndim) return false;

        int pad_start[8] = {}, pad_end[8] = {};
        for (uint32_t k = 0; k < ndim; ++k) {
            int64_t s = read_index(t_pads, (int)k);
            int64_t e = read_index(t_pads, (int)(ndim + k));
            if (s < 0 || e < 0) return false;   // negative pad (cropping) → CPU fallback
            pad_start[k] = (int)s;
            pad_end[k]   = (int)e;
        }

        // Constant value (optional input 2), default 0.
        float pad_value = 0.0f;
        if (inputs.size() > 2 && inputs[2] && inputs[2]->data) {
            const tensor_t* cv = inputs[2];
            if (cv->type != NNR_DATA_TYPE_FLOAT32) return false;
            if (cv->ndata < 1) return false;
            pad_value = ((const float*)cv->data)[0];
        }
        std::memcpy(&pad_value_bits, &pad_value, sizeof(uint32_t));

        // Output dims.
        int out_dims[8] = {};
        total = 1;
        empty_output = false;
        for (uint32_t k = 0; k < ndim; ++k) {
            int od = x->dims[k] + pad_start[k] + pad_end[k];
            if (od <= 0) { empty_output = true; od = 0; }
            out_dims[k] = od;
            total *= (uint32_t)od;
        }
        if (!y->reshape(std::span<const int>(out_dims, (int)ndim), x->type)) return false;

        // Natural row-major strides over input's own shape.
        for (int k = 0; k < 8; ++k) {
            out_dims_u[k] = 0; in_dims_u[k] = 0; in_strides_u[k] = 0; pad_starts_u[k] = 0;
        }
        {
            uint32_t s = 1;
            for (int k = (int)ndim - 1; k >= 0; --k) { in_strides_u[k] = s; s *= (uint32_t)x->dims[k]; }
        }
        for (uint32_t k = 0; k < ndim; ++k) {
            out_dims_u[k]   = (uint32_t)out_dims[k];
            in_dims_u[k]    = (uint32_t)x->dims[k];
            pad_starts_u[k] = (uint32_t)pad_start[k];
        }

        webgpu::ensure_buffer(x, x->ndata * sizeof(float));
        webgpu::ensure_buffer(y, y->ndata * sizeof(float));

        // Meta layout (matches pad.wgsl):
        //   [0..4)    total
        //   [4..8)    ndim
        //   [8..12)   pad_value (f32 bits)
        //   [12..16)  grid_stride_x (threads-x when 2D-splitting dispatch)
        //   [16..48)  out_dims      (8 x u32)
        //   [48..80)  in_dims       (8 x u32)
        //   [80..112) in_strides    (8 x u32)
        //   [112..144) pad_starts   (8 x u32)
        dispatch_gx = 0; dispatch_gy = 0;
        if (!(empty_output || total == 0)) {
            uint32_t groups = (total + WG - 1) / WG;
            webgpu::dispatch_1d_grid(groups, dispatch_gx, dispatch_gy);
            uint8_t buf[256] = {};
            auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(buf + off, &v, 4); };
            put_u32(0,  total);
            put_u32(4,  ndim);
            put_u32(8,  pad_value_bits);
            put_u32(12, dispatch_gx * WG);
            for (int k = 0; k < 8; ++k) put_u32(16  + k * 4, out_dims_u[k]);
            for (int k = 0; k < 8; ++k) put_u32(48  + k * 4, in_dims_u[k]);
            for (int k = 0; k < 8; ++k) put_u32(80  + k * 4, in_strides_u[k]);
            for (int k = 0; k < 8; ++k) put_u32(112 + k * 4, pad_starts_u[k]);
            webgpu::get_device().queue.WriteBuffer(meta_buf, 0, buf, sizeof(buf));
        }
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();
        if (empty_output || total == 0) { webgpu::mark_gpu_written(outputs[0]); return true; }
        webgpu::upload_if_needed(inputs[0]);

        auto* rx = webgpu::find(inputs[0]);
        auto* ry = webgpu::find(outputs[0]);

        uint32_t gen_x = webgpu::generation_of(inputs[0]);
        uint32_t gen_y = webgpu::generation_of(outputs[0]);
        if (!cached_bg || gen_x != cached_gen[0] || gen_y != cached_gen[1]) {
            wgpu::BindGroupEntry be[3] = {};
            be[0].binding = 0; be[0].buffer = rx->buf;   be[0].offset = 0; be[0].size = rx->size;
            be[1].binding = 1; be[1].buffer = ry->buf;   be[1].offset = 0; be[1].size = ry->size;
            be[2].binding = 2; be[2].buffer = meta_buf;  be[2].offset = 0; be[2].size = 256;
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

operator_t* resolver_default_op_Pad_webgpu(int, pool_t& pool) {
    return pool_new<Pad_operator_webgpu>(pool);
}

} // namespace nnr
