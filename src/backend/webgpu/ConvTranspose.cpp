// WebGPU ConvTranspose, 1D / 2D / 3D NCHW variants. Supports strides /
// pads / dilations / groups / output_padding / optional bias, plus all
// four auto_pad modes (NOTSET / VALID / SAME_UPPER / SAME_LOWER).
//
// Output shape per spatial axis:
//   - NOTSET / VALID (from formula):
//     L_out[i] = stride[i] * (L[i] - 1) + output_padding[i]
//                + (k[i] - 1) * dilation[i] + 1
//                - (pad_begin[i] + pad_end[i])
//   - SAME_UPPER / SAME_LOWER:
//     L_out[i] = L[i] * stride[i]
//     total_pad[i] = stride[i] * (L[i] - 1) + output_padding[i] + k_eff[i]
//                    - L_out[i]                  (clamped to 0 if negative)
//     SAME_UPPER: pad_begin = total_pad / 2;  pad_end = total_pad - pad_begin
//     SAME_LOWER: pad_end   = total_pad / 2;  pad_begin = total_pad - pad_end
//
// The shader (conv_transpose.wgsl) is rank-unified: treats every op as
// having 3 spatial axes (D, H, W) with unused ones set to size 1. For 1D
// we store (1, 1, L); for 2D (1, H, W). Kernel layout follows the same
// convention. This removes the need for per-rank shader variants.

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

constexpr uint32_t WG = 64;

struct ConvTranspose_operator_webgpu : public operator_t {
    uint32_t total_u = 0;
    uint32_t meta_vals[24] = {};   // matches Meta in conv_transpose.wgsl

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          meta_buf;
    wgpu::Buffer          dummy_bias;

    // Cached BindGroup. Tensor-backed slots: [X, W, bias|0, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[4] = {};

    bool init() override {
        if (inputs.size() < 2 || inputs.size() > 3) return false;
        if (outputs.size() != 1) return false;
        if (!webgpu::device_ready()) return false;

        auto& dev = webgpu::get_device();
        wgpu::ShaderSourceWGSL w = {};
        w.code = webgpu::wgsl::conv_transpose;
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
        e[4].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
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

        wgpu::BufferDescriptor md = {};
        md.size  = 128;  // room for 24 u32 + padding
        md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        meta_buf = dev.device.CreateBuffer(&md);

        wgpu::BufferDescriptor bd = {};
        bd.size  = 16;
        bd.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        dummy_bias = dev.device.CreateBuffer(&bd);
        uint8_t zeros[16] = {};
        dev.queue.WriteBuffer(dummy_bias, 0, zeros, sizeof(zeros));
        return true;
    }

    bool reshape() override {
        const tensor_t* X = inputs[0];
        const tensor_t* W = inputs[1];
        tensor_t*       Y = outputs[0];
        if (X->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (W->type != NNR_DATA_TYPE_FLOAT32) return false;

        // Accept rank 3 (1D: N,C,L), rank 4 (2D: N,C,H,W), rank 5 (3D: N,C,D,H,W).
        if (X->ndim < 3 || X->ndim > 5) return false;
        if (W->ndim != X->ndim)         return false;

        const int spatial_rank = X->ndim - 2;

        std::string_view auto_pad = attribute(attr_key_t::auto_pad, "NOTSET");
        if (auto_pad != "NOTSET" && auto_pad != "VALID"
            && auto_pad != "SAME_UPPER" && auto_pad != "SAME_LOWER") return false;

        int N  = X->dims[0];
        int C  = X->dims[1];   // C_in

        // Extract input spatial dims into a 3-wide array (D, H, W), padding
        // with 1 for ranks < 3.
        int in_shape[3] = { 1, 1, 1 };
        for (int i = 0; i < spatial_rank; ++i)
            in_shape[3 - spatial_rank + i] = X->dims[2 + i];

        int C_in_w      = W->dims[0];
        int M_per_group = W->dims[1];

        // Kernel spatial dims, similarly padded.
        int k_shape[3] = { 1, 1, 1 };
        for (int i = 0; i < spatial_rank; ++i)
            k_shape[3 - spatial_rank + i] = W->dims[2 + i];

        int group = (int)attribute(attr_key_t::group, (int64_t)1);
        if (group < 1) return false;
        if (C != C_in_w) return false;
        if (C % group != 0) return false;

        int M = M_per_group * group;   // C_out

        // Default attrs: strides / dilations = [1]*spatial_rank;
        // pads = [0]*(2*spatial_rank); output_padding = [0]*spatial_rank.
        int64_t* ints = nullptr;
        int strides[3] = { 1, 1, 1 };
        int nstride = attribute(attr_key_t::strides, ints);
        if (nstride > spatial_rank) return false;
        for (int i = 0; i < nstride; ++i)
            strides[3 - spatial_rank + i] = (int)ints[i];

        int dilations[3] = { 1, 1, 1 };
        int ndil = attribute(attr_key_t::dilations, ints);
        if (ndil > spatial_rank) return false;
        for (int i = 0; i < ndil; ++i)
            dilations[3 - spatial_rank + i] = (int)ints[i];

        // ONNX pads layout: [begin_1, begin_2, ..., end_1, end_2, ...].
        int pad_begin[3] = { 0, 0, 0 };
        int pad_end  [3] = { 0, 0, 0 };
        int npad = attribute(attr_key_t::pads, ints);
        if (npad > 2 * spatial_rank) return false;
        if (npad == 2 * spatial_rank) {
            for (int i = 0; i < spatial_rank; ++i) {
                pad_begin[3 - spatial_rank + i] = (int)ints[i];
                pad_end  [3 - spatial_rank + i] = (int)ints[spatial_rank + i];
            }
        }
        if (auto_pad == "VALID") {
            for (int i = 0; i < 3; ++i) { pad_begin[i] = 0; pad_end[i] = 0; }
        }
        for (int i = 0; i < 3; ++i) {
            if (pad_begin[i] < 0 || pad_end[i] < 0) return false;
        }

        int out_pad[3] = { 0, 0, 0 };
        int nop = attribute(attr_key_t::output_padding, ints);
        if (nop > spatial_rank) return false;
        for (int i = 0; i < nop; ++i)
            out_pad[3 - spatial_rank + i] = (int)ints[i];
        for (int i = 0; i < 3; ++i) if (out_pad[i] < 0) return false;

        // Explicit output_shape → CPU fallback (need to back-compute pads
        // with a different policy than auto_pad does).
        int nos = attribute(attr_key_t::output_shape, ints);
        if (nos > 0) return false;

        int out_shape[3] = { 1, 1, 1 };
        if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
            // Output shape is out = in * stride per spatial axis. The
            // pads are back-solved to make the geometry consistent.
            for (int i = 0; i < 3; ++i) {
                out_shape[i] = in_shape[i] * strides[i];
                if (i < 3 - spatial_rank) continue;  // unused axis (padded-1)
                int k_eff = (k_shape[i] - 1) * dilations[i] + 1;
                int total_pad = strides[i] * (in_shape[i] - 1) + out_pad[i]
                              + k_eff - out_shape[i];
                if (total_pad < 0) total_pad = 0;
                if (auto_pad == "SAME_UPPER") {
                    pad_begin[i] = total_pad / 2;
                    pad_end[i]   = total_pad - pad_begin[i];
                } else {
                    pad_end[i]   = total_pad / 2;
                    pad_begin[i] = total_pad - pad_end[i];
                }
            }
        } else {
            for (int i = 0; i < 3; ++i) {
                out_shape[i] = strides[i] * (in_shape[i] - 1) + out_pad[i]
                             + (k_shape[i] - 1) * dilations[i] + 1
                             - (pad_begin[i] + pad_end[i]);
                if (out_shape[i] <= 0) return false;
            }
        }
        for (int i = 0; i < 3; ++i) {
            if (out_shape[i] <= 0) return false;
        }

        // Output tensor: (N, M, <spatial_rank trailing dims>). Drop the
        // padding-1 prefix that we only used internally for the shader.
        int out_dims[5] = { N, M, 0, 0, 0 };
        for (int i = 0; i < spatial_rank; ++i)
            out_dims[2 + i] = out_shape[3 - spatial_rank + i];
        if (!Y->reshape(std::span<const int>(out_dims, X->ndim), X->type)) return false;

        int bias = (inputs.size() == 3 && inputs[2]) ? 1 : 0;
        if (bias) {
            if (inputs[2]->type != NNR_DATA_TYPE_FLOAT32) return false;
            if ((int)inputs[2]->ndata != M) return false;
        }

        total_u = (uint32_t)(N * M * out_shape[0] * out_shape[1] * out_shape[2]);

        // Populate meta in the order the shader expects (see
        // conv_transpose.wgsl / struct Meta).
        meta_vals[0]  = total_u;
        meta_vals[1]  = (uint32_t)N;
        meta_vals[2]  = (uint32_t)M;
        meta_vals[3]  = (uint32_t)out_shape[0];   // D_out
        meta_vals[4]  = (uint32_t)out_shape[1];   // H_out
        meta_vals[5]  = (uint32_t)out_shape[2];   // W_out
        meta_vals[6]  = (uint32_t)C;
        meta_vals[7]  = (uint32_t)group;
        meta_vals[8]  = (uint32_t)k_shape[0];     // kD
        meta_vals[9]  = (uint32_t)k_shape[1];     // kH
        meta_vals[10] = (uint32_t)k_shape[2];     // kW
        meta_vals[11] = (uint32_t)strides[0];     // stride_d
        meta_vals[12] = (uint32_t)strides[1];     // stride_h
        meta_vals[13] = (uint32_t)strides[2];     // stride_w
        meta_vals[14] = (uint32_t)pad_begin[0];   // pad_front
        meta_vals[15] = (uint32_t)pad_begin[1];   // pad_top
        meta_vals[16] = (uint32_t)pad_begin[2];   // pad_left
        meta_vals[17] = (uint32_t)dilations[0];   // dilation_d
        meta_vals[18] = (uint32_t)dilations[1];   // dilation_h
        meta_vals[19] = (uint32_t)dilations[2];   // dilation_w
        meta_vals[20] = (uint32_t)in_shape[0];    // D
        meta_vals[21] = (uint32_t)in_shape[1];    // H
        meta_vals[22] = (uint32_t)in_shape[2];    // W
        meta_vals[23] = (uint32_t)bias;

        webgpu::ensure_buffer(X, X->ndata * sizeof(float));
        webgpu::ensure_buffer(W, W->ndata * sizeof(float));
        if (bias) webgpu::ensure_buffer(inputs[2], inputs[2]->ndata * sizeof(float));
        webgpu::ensure_buffer(Y, Y->ndata * sizeof(float));

        // meta_vals depends only on shape + attribute data — write here so
        // exec() doesn't pay the WriteBuffer cost per dispatch.
        webgpu::get_device().queue.WriteBuffer(meta_buf, 0, meta_vals, sizeof(meta_vals));
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();
        webgpu::upload_if_needed(inputs[0]);
        webgpu::upload_if_needed(inputs[1]);
        bool has_bias = (inputs.size() == 3 && inputs[2]);
        if (has_bias) webgpu::upload_if_needed(inputs[2]);

        auto* rx = webgpu::find(inputs[0]);
        auto* rw = webgpu::find(inputs[1]);
        auto* rb = has_bias ? webgpu::find(inputs[2]) : nullptr;
        auto* ry = webgpu::find(outputs[0]);

        uint32_t gen_x = webgpu::generation_of(inputs[0]);
        uint32_t gen_w = webgpu::generation_of(inputs[1]);
        uint32_t gen_b = has_bias ? webgpu::generation_of(inputs[2]) : 0u;
        uint32_t gen_y = webgpu::generation_of(outputs[0]);
        if (!cached_bg || gen_x != cached_gen[0]
                       || gen_w != cached_gen[1]
                       || gen_b != cached_gen[2]
                       || gen_y != cached_gen[3]) {
            wgpu::BindGroupEntry be[5] = {};
            be[0].binding = 0; be[0].buffer = rx->buf;                be[0].offset = 0; be[0].size = rx->size;
            be[1].binding = 1; be[1].buffer = rw->buf;                be[1].offset = 0; be[1].size = rw->size;
            be[2].binding = 2;
            if (has_bias) { be[2].buffer = rb->buf; be[2].size = rb->size; }
            else          { be[2].buffer = dummy_bias; be[2].size = 16; }
            be[2].offset = 0;
            be[3].binding = 3; be[3].buffer = ry->buf;                be[3].offset = 0; be[3].size = ry->size;
            be[4].binding = 4; be[4].buffer = meta_buf;               be[4].offset = 0; be[4].size = sizeof(meta_vals);
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 5; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_x;
            cached_gen[1] = gen_w;
            cached_gen[2] = gen_b;
            cached_gen[3] = gen_y;
        }

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, cached_bg);
        uint32_t groups = (total_u + WG - 1) / WG;
        pass.DispatchWorkgroups(groups, 1, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_ConvTranspose_webgpu(int, pool_t& pool) {
    return pool_new<ConvTranspose_operator_webgpu>(pool);
}

} // namespace nnr
