// WebGPU Cast — elementwise dtype conversion.
//
// Seed dtype coverage for the WebGPU backend: f32, i32, u32, and int64.
// int64 is represented on the GPU as i32 — buffer.cpp narrows on upload
// and sign-extends back on download, which is right for the common index
// use cases (Gather axes, ArgMax outputs, shape tensors); values outside
// the i32 range silently lose their high bits, so models with true int64
// magnitudes must stay on CPU for that op. Any other source or target
// dtype (f64, f16, i8, bool-as-C-bool, the float8 variants, etc.) falls
// back to CPU via `return false` from reshape — same pattern used by
// MatMul/Conv when they see a dtype they don't implement.
//
// Pipelines are built lazily at reshape time because the input dtype isn't
// known at init time (only opset is known to solve_operator). One pipeline
// is cached per (in_ty, out_ty) pair encountered — in practice a given Cast
// node only sees one pair so this is effectively a one-shot build.

#include "nnr.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include <string>
#include <webgpu/webgpu_cpp.h>

namespace nnr {

namespace {

const char* wgsl_ty(data_type_t t) {
    switch (t) {
    case NNR_DATA_TYPE_FLOAT32: return "f32";
    case NNR_DATA_TYPE_INT32:   return "i32";
    case NNR_DATA_TYPE_UINT32:  return "u32";
    case NNR_DATA_TYPE_INT64:   return "i32";   // stored as i32 on GPU
    default:                    return nullptr;
    }
}

struct Cast_operator_webgpu : public operator_t {
    data_type_t to_type     = NNR_DATA_TYPE_UNDEFINED;
    data_type_t built_in    = NNR_DATA_TYPE_UNDEFINED;
    data_type_t built_out   = NNR_DATA_TYPE_UNDEFINED;

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          uniforms;
    uint32_t              dispatch_gx = 0;
    uint32_t              dispatch_gy = 0;

    // Cached BindGroup. Tensor-backed slots: [X, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[2] = {};

    bool init() override {
        if (!is_inout_size(1, 1)) return false;
        if (!webgpu::device_ready()) return false;

        // 'to' attribute holds the NNR dtype enum value (onnx_loader converts
        // the ONNX dtype id to NNR at load). Default = keep source dtype.
        to_type = (data_type_t)attribute(attr_key_t::to, (int64_t)NNR_DATA_TYPE_UNDEFINED);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t*       y = outputs[0];

        // If no 'to' attribute was set, default to input dtype (pass-through).
        data_type_t out_ty = (to_type == NNR_DATA_TYPE_UNDEFINED) ? x->type : to_type;

        const char* in_w  = wgsl_ty(x->type);
        const char* out_w = wgsl_ty(out_ty);
        if (!in_w || !out_w) return false;  // unsupported dtype → CPU fallback

        if (!y->reshape_identity(x, out_ty)) return false;

        if (!pipeline || built_in != x->type || built_out != out_ty) {
            if (!build_pipeline(in_w, out_w)) return false;
            built_in  = x->type;
            built_out = out_ty;
        }

        webgpu::ensure_buffer(x, (size_t)x->ndata * data_type_sizeof(x->type));
        webgpu::ensure_buffer(y, (size_t)y->ndata * data_type_sizeof(out_ty));

        // Uniform contents depend only on ndata — reshape-time constant.
        const uint32_t WG = 256;
        uint32_t groups = ((uint32_t)y->ndata + WG - 1) / WG;
        webgpu::dispatch_1d_grid(groups, dispatch_gx, dispatch_gy);
        uint32_t u[4] = { (uint32_t)y->ndata, dispatch_gx * WG, 0, 0 };
        webgpu::get_device().queue.WriteBuffer(uniforms, 0, u, sizeof(u));
        return true;
    }

    bool build_pipeline(const char* in_w, const char* out_w) {
        auto& dev = webgpu::get_device();

        std::string src =
            "struct Dims { n : u32, grid_stride_x : u32, _b : u32, _c : u32 };\n"
            "@group(0) @binding(0) var<storage, read>       X : array<";
        src += in_w;
        src += ">;\n"
               "@group(0) @binding(1) var<storage, read_write> Y : array<";
        src += out_w;
        src += ">;\n"
               "@group(0) @binding(2) var<uniform>             dims : Dims;\n"
               "@compute @workgroup_size(256)\n"
               "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
               "  let i = gid.y * dims.grid_stride_x + gid.x;\n"
               "  if (i >= dims.n) { return; }\n"
               "  Y[i] = ";
        src += out_w;
        src += "(X[i]);\n}\n";

        wgpu::ShaderSourceWGSL wgslSrc = {};
        wgslSrc.code = src.c_str();
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &wgslSrc;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

        if (!bgl) {
            wgpu::BindGroupLayoutEntry e[3] = {};
            e[0].binding = 0; e[0].visibility = wgpu::ShaderStage::Compute;
            e[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
            e[1].binding = 1; e[1].visibility = wgpu::ShaderStage::Compute;
            e[1].buffer.type = wgpu::BufferBindingType::Storage;
            e[2].binding = 2; e[2].visibility = wgpu::ShaderStage::Compute;
            e[2].buffer.type = wgpu::BufferBindingType::Uniform;
            wgpu::BindGroupLayoutDescriptor bgld = {};
            bgld.entryCount = 3;
            bgld.entries = e;
            bgl = dev.device.CreateBindGroupLayout(&bgld);

            wgpu::BufferDescriptor ud = {};
            ud.size = 16;
            ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
            uniforms = dev.device.CreateBuffer(&ud);
        }

        wgpu::PipelineLayoutDescriptor pld = {};
        pld.bindGroupLayoutCount = 1;
        pld.bindGroupLayouts = &bgl;
        wgpu::PipelineLayout pl = dev.device.CreatePipelineLayout(&pld);

        wgpu::ComputePipelineDescriptor cpd = {};
        cpd.layout = pl;
        cpd.compute.module = sm;
        cpd.compute.entryPoint = "main";
        pipeline = dev.device.CreateComputePipeline(&cpd);
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
            be[0].binding = 0; be[0].buffer = rx->buf;  be[0].offset = 0; be[0].size = rx->size;
            be[1].binding = 1; be[1].buffer = ry->buf;  be[1].offset = 0; be[1].size = ry->size;
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
        pass.DispatchWorkgroups(dispatch_gx, dispatch_gy, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_Cast_webgpu(int, pool_t& pool) {
    return pool_new<Cast_operator_webgpu>(pool);
}

} // namespace nnr
