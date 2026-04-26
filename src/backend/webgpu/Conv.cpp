#include "nnr.h"
#include "attr_key.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include "shaders.h"

#include <webgpu/webgpu_cpp.h>

#include <cstdlib>
#include <cstring>
#include <string_view>

namespace nnr {

namespace {

constexpr uint32_t WG = 64;
constexpr uint32_t MAX_TILED_W = 4096;   // matches conv_tiled.wgsl's MAX_WEIGHTS

// 2D NCHW direct convolution on WebGPU. Supports: strides, explicit pads,
// dilations, groups, and optional bias. 1D/3D Conv and non-NOTSET auto_pad
// variants return false from reshape so the CPU backend picks them up.
//
// Has two pipelines that share one bind-group layout:
//   - plain: one thread per output, scalar inner loops (conv.wgsl)
//   - tiled: 8×8 workgroup computes an 8×8 output tile for one (n, m);
//            all 64 threads cooperatively load W[m,:,:,:] into shared
//            memory and reuse it across the tile (conv_tiled.wgsl).
// Pick tiled whenever `C_in_per_group * kH * kW ≤ MAX_TILED_W` and the
// output plane is non-trivial; else fall back to plain.
struct Conv_operator_webgpu : public operator_t {
    uint32_t total_u = 0;
    uint32_t meta_vals[18] = {};   // matches Meta struct layout
    int H_out_i = 0, W_out_i = 0, N_i = 0, M_i = 0;
    bool use_tiled = false;

    wgpu::ComputePipeline pipeline;
    wgpu::ComputePipeline pipeline_tiled;
    bool                  tiled_built = false;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          meta_buf;
    wgpu::Buffer          dummy_bias;   // 16 B of zeros, used when inputs.size()==2

    // Cached BindGroup; rebuilt when any tensor-backed binding's buffer
    // handle changes. Slots: [X, W, bias|0, Y]. `meta_buf`, `dummy_bias`,
    // and the BGL are op-owned and stable, so generation bookkeeping
    // applies only to the tensors.
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[4] = {};

    bool init() override {
        if (inputs.size() < 2 || inputs.size() > 3) return false;
        if (outputs.size() != 1) return false;
        // Early-reject ranks / dtypes we can't handle so the loader falls
        // back to CPU. reshape() failures don't get a fallback path; init()
        // ones do. Covers 1D / 3D Conv (rank 3 / 5) and int / quant Conv
        // which must stay on CPU.
        if (inputs[0] && inputs[0]->type != NNR_DATA_TYPE_UNDEFINED
            && inputs[0]->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (inputs[1] && inputs[1]->type != NNR_DATA_TYPE_UNDEFINED
            && inputs[1]->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (inputs[0] && inputs[0]->ndim > 0 && inputs[0]->ndim != 4) return false;
        if (inputs[1] && inputs[1]->ndim > 0 && inputs[1]->ndim != 4) return false;
        if (!webgpu::device_ready()) return false;

        auto& dev = webgpu::get_device();
        wgpu::ShaderSourceWGSL w = {};
        w.code = webgpu::wgsl::conv;
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
        md.size  = 128;
        md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        meta_buf = dev.device.CreateBuffer(&md);

        // Stand-in bias buffer used when the model doesn't have one. Sized
        // to the minimum the runtime accepts (16 B) to keep the binding
        // layout uniform across pipelines.
        wgpu::BufferDescriptor bd = {};
        bd.size  = 16;
        bd.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        dummy_bias = dev.device.CreateBuffer(&bd);
        uint8_t zeros[16] = {};
        dev.queue.WriteBuffer(dummy_bias, 0, zeros, sizeof(zeros));
        return true;
    }

    // Build the tiled pipeline on first use. Reuses the plain pipeline's
    // bind-group layout since the bindings are identical.
    bool ensure_tiled_pipeline() {
        if (tiled_built) return true;
        auto& dev = webgpu::get_device();
        wgpu::ShaderSourceWGSL w = {};
        w.code = webgpu::wgsl::conv_tiled;
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &w;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

        wgpu::PipelineLayoutDescriptor pld = {};
        pld.bindGroupLayoutCount = 1;
        pld.bindGroupLayouts = &bgl;
        wgpu::PipelineLayout pl = dev.device.CreatePipelineLayout(&pld);

        wgpu::ComputePipelineDescriptor cpd = {};
        cpd.layout = pl;
        cpd.compute.module = sm;
        cpd.compute.entryPoint = "main";
        pipeline_tiled = dev.device.CreateComputePipeline(&cpd);
        tiled_built = true;
        return true;
    }

    bool reshape() override {
        const tensor_t* X = inputs[0];
        const tensor_t* W = inputs[1];
        tensor_t*       Y = outputs[0];
        if (X->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (W->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (X->ndim != 4 || W->ndim != 4)     return false;   // 2D only

        std::string_view auto_pad = attribute(attr_key_t::auto_pad, "NOTSET");
        if (auto_pad != "NOTSET" && auto_pad != "VALID"
            && auto_pad != "SAME_UPPER" && auto_pad != "SAME_LOWER") return false;

        int N  = X->dims[0];
        int C  = X->dims[1];
        int H  = X->dims[2];
        int Wd = X->dims[3];
        int M  = W->dims[0];
        int C_per_group = W->dims[1];
        int kH = W->dims[2];
        int kW = W->dims[3];

        int group = (int)attribute(attr_key_t::group, (int64_t)1);
        if (group < 1) return false;
        if (C % group != 0)      return false;
        if (M % group != 0)      return false;
        if (C / group != C_per_group) return false;

        int64_t* ints = nullptr;
        int nstride = attribute(attr_key_t::strides,   ints);
        int s_h = nstride >= 1 ? (int)ints[0] : 1;
        int s_w = nstride >= 2 ? (int)ints[1] : s_h;

        int ndil = attribute(attr_key_t::dilations, ints);
        int d_h = ndil >= 1 ? (int)ints[0] : 1;
        int d_w = ndil >= 2 ? (int)ints[1] : d_h;

        int npad = attribute(attr_key_t::pads, ints);
        int pad_top = 0, pad_left = 0, pad_bot = 0, pad_right = 0;
        if (npad >= 4) {
            pad_top   = (int)ints[0];
            pad_left  = (int)ints[1];
            pad_bot   = (int)ints[2];
            pad_right = (int)ints[3];
        }
        if (auto_pad == "VALID") { pad_top = pad_left = pad_bot = pad_right = 0; }

        int H_out, W_out;
        if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
            // Output shape is ceil(in / stride); pads are back-solved so the
            // geometry is consistent with that. Mirrors ConvTranspose's
            // SAME_* handling.
            H_out = (H + s_h - 1) / s_h;
            W_out = (Wd + s_w - 1) / s_w;
            int total_pad_h = (H_out - 1) * s_h + d_h * (kH - 1) + 1 - H;
            int total_pad_w = (W_out - 1) * s_w + d_w * (kW - 1) + 1 - Wd;
            if (total_pad_h < 0) total_pad_h = 0;
            if (total_pad_w < 0) total_pad_w = 0;
            if (auto_pad == "SAME_UPPER") {
                pad_top  = total_pad_h / 2; pad_bot   = total_pad_h - pad_top;
                pad_left = total_pad_w / 2; pad_right = total_pad_w - pad_left;
            } else {
                pad_bot   = total_pad_h / 2; pad_top  = total_pad_h - pad_bot;
                pad_right = total_pad_w / 2; pad_left = total_pad_w - pad_right;
            }
        } else {
            if (pad_top < 0 || pad_left < 0 || pad_bot < 0 || pad_right < 0) return false;
            H_out = (H + pad_top  + pad_bot   - d_h * (kH - 1) - 1) / s_h + 1;
            W_out = (Wd + pad_left + pad_right - d_w * (kW - 1) - 1) / s_w + 1;
        }
        if (H_out <= 0 || W_out <= 0) return false;

        int out_dims[4] = { N, M, H_out, W_out };
        if (!Y->reshape(std::span<const int>(out_dims, 4), X->type)) return false;

        int bias = (inputs.size() == 3 && inputs[2]) ? 1 : 0;
        if (bias) {
            if (inputs[2]->type != NNR_DATA_TYPE_FLOAT32) return false;
            if ((int)inputs[2]->ndata != M) return false;
        }

        total_u = (uint32_t)(N * M * H_out * W_out);
        meta_vals[0]  = total_u;
        meta_vals[1]  = (uint32_t)N;
        meta_vals[2]  = (uint32_t)M;
        meta_vals[3]  = (uint32_t)H_out;
        meta_vals[4]  = (uint32_t)W_out;
        meta_vals[5]  = (uint32_t)C;
        meta_vals[6]  = (uint32_t)group;
        meta_vals[7]  = (uint32_t)kH;
        meta_vals[8]  = (uint32_t)kW;
        meta_vals[9]  = (uint32_t)s_h;
        meta_vals[10] = (uint32_t)s_w;
        meta_vals[11] = (uint32_t)pad_top;
        meta_vals[12] = (uint32_t)pad_left;
        meta_vals[13] = (uint32_t)d_h;
        meta_vals[14] = (uint32_t)d_w;
        meta_vals[15] = (uint32_t)H;
        meta_vals[16] = (uint32_t)Wd;
        meta_vals[17] = (uint32_t)bias;

        webgpu::ensure_buffer(X, X->ndata * sizeof(float));
        webgpu::ensure_buffer(W, W->ndata * sizeof(float));
        if (bias) webgpu::ensure_buffer(inputs[2], inputs[2]->ndata * sizeof(float));
        webgpu::ensure_buffer(Y, Y->ndata * sizeof(float));

        H_out_i = H_out; W_out_i = W_out; N_i = N; M_i = M;

        // Decide between plain and tiled kernel. Tiled wins when the
        // weight tile fits in shared memory AND there's enough spatial
        // work to amortize the cooperative load (≥ 8×8 outputs typical).
        // For tiny outputs (e.g. 1×1 global-avg-pool-style), the plain
        // kernel's per-thread dispatch is fine.
        const uint32_t w_size = (uint32_t)(C_per_group * kH * kW);
        use_tiled = (w_size <= MAX_TILED_W) && (H_out >= 4) && (W_out >= 4);
        // Debug: force plain kernel via NNR_DISABLE_TILED_CONV.
        static const bool s_disable = []{
            const char* v = std::getenv("NNR_DISABLE_TILED_CONV");
            return v && *v;
        }();
        if (s_disable) use_tiled = false;
        if (use_tiled) ensure_tiled_pipeline();

        // meta_vals is fully populated above and depends only on shape +
        // attribute data — write to the GPU here so exec() doesn't pay
        // the WriteBuffer cost on every dispatch.
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
            be[4].binding = 4; be[4].buffer = meta_buf;               be[4].offset = 0; be[4].size = 128;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 5; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_x;
            cached_gen[1] = gen_w;
            cached_gen[2] = gen_b;
            cached_gen[3] = gen_y;
        }

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();
        if (use_tiled) {
            pass.SetPipeline(pipeline_tiled);
            pass.SetBindGroup(0, cached_bg);
            const uint32_t TX = 8, TY = 8;
            uint32_t gx = ((uint32_t)W_out_i + TX - 1) / TX;
            uint32_t gy = ((uint32_t)H_out_i + TY - 1) / TY;
            uint32_t gz = (uint32_t)(N_i * M_i);
            pass.DispatchWorkgroups(gx, gy, gz);
        } else {
            pass.SetPipeline(pipeline);
            pass.SetBindGroup(0, cached_bg);
            uint32_t groups = (total_u + WG - 1) / WG;
            pass.DispatchWorkgroups(groups, 1, 1);
        }
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_Conv_webgpu(int, pool_t& pool) {
    return pool_new<Conv_operator_webgpu>(pool);
}

} // namespace nnr
