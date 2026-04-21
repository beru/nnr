// WebGPU Range(start, limit, delta) — produce a 1D sequence.
//
// ONNX semantics: Y = [start, start+delta, start+2*delta, ..., < limit]
// (delta positive) or (> limit) (delta negative). All three inputs are
// 0-D scalars with the same dtype. Output length is
// max(0, ceil((limit - start) / delta)).
//
// Scope: f32 and i32 inputs. Other dtypes (i64 — common in ONNX for
// position indices, f64) fall back to CPU. Inputs must be CPU-resident
// scalars so the length can be computed at reshape-time. If a graph
// produces start/limit/delta as GPU-only tensors (rare but possible via
// upstream compute), we return false from reshape and let CPU handle it.
//
// The length is baked into the uniform and the kernel writes
// `start + i * delta` in parallel. Pipeline is cached per (in_type, out_type)
// — which in practice is always a single (same-dtype) case since ONNX
// Range's output dtype equals input dtype.

#include "nnr.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include <cstring>
#include <string>
#include <webgpu/webgpu_cpp.h>

namespace nnr {

namespace {

constexpr uint32_t WG = 256;

const char* wgsl_ty(data_type_t t) {
    switch (t) {
    case NNR_DATA_TYPE_FLOAT32: return "f32";
    case NNR_DATA_TYPE_INT32:   return "i32";
    default:                    return nullptr;
    }
}

// Scalar tensor read at reshape-time. Requires CPU-resident data.
bool read_scalar(const tensor_t* t, double& out) {
    if (!t->data) return false;
    if (!(t->ndim == 0 || (t->ndim == 1 && t->ndata == 1))) return false;
    switch (t->type) {
    case NNR_DATA_TYPE_FLOAT32: out = *(const float*)t->data;   return true;
    case NNR_DATA_TYPE_INT32:   out = *(const int32_t*)t->data; return true;
    default: return false;
    }
}

struct Range_operator_webgpu : public operator_t {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          uniforms;  // 16 bytes: {n, pad, start_f32, delta_f32}
    data_type_t           built_ty = NNR_DATA_TYPE_UNDEFINED;

    double start = 0, limit = 0, delta = 0;
    uint32_t n   = 0;

    // Cached BindGroup. Tensor-backed slot: [Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[1] = {};

    bool init() override {
        if (!is_inout_size(3, 1)) return false;
        if (!webgpu::device_ready()) return false;

        auto& dev = webgpu::get_device();
        wgpu::BindGroupLayoutEntry e[2] = {};
        e[0].binding = 0; e[0].visibility = wgpu::ShaderStage::Compute;
        e[0].buffer.type = wgpu::BufferBindingType::Storage;
        e[1].binding = 1; e[1].visibility = wgpu::ShaderStage::Compute;
        e[1].buffer.type = wgpu::BufferBindingType::Uniform;
        wgpu::BindGroupLayoutDescriptor bgld = {};
        bgld.entryCount = 2; bgld.entries = e;
        bgl = dev.device.CreateBindGroupLayout(&bgld);

        wgpu::BufferDescriptor ud = {};
        ud.size = 16;
        ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        uniforms = dev.device.CreateBuffer(&ud);
        return true;
    }

    bool build_pipeline(const char* out_ty) {
        auto& dev = webgpu::get_device();
        std::string src =
            "struct Cfg { n : u32, _pad : u32, start : f32, delta : f32 };\n"
            "@group(0) @binding(0) var<storage, read_write> Y   : array<";
        src += out_ty;
        src += ">;\n"
               "@group(0) @binding(1) var<uniform>             cfg : Cfg;\n"
               "@compute @workgroup_size(256)\n"
               "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
               "  let i = gid.x;\n"
               "  if (i >= cfg.n) { return; }\n"
               "  Y[i] = ";
        src += out_ty;
        src += "(cfg.start + f32(i) * cfg.delta);\n}\n";

        wgpu::ShaderSourceWGSL w = {};
        w.code = src.c_str();
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &w;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

        wgpu::PipelineLayoutDescriptor pld = {};
        pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = &bgl;
        wgpu::PipelineLayout pl = dev.device.CreatePipelineLayout(&pld);

        wgpu::ComputePipelineDescriptor cpd = {};
        cpd.layout = pl;
        cpd.compute.module = sm;
        cpd.compute.entryPoint = "main";
        pipeline = dev.device.CreateComputePipeline(&cpd);
        return true;
    }

    bool reshape() override {
        const tensor_t* s = inputs[0];
        const tensor_t* l = inputs[1];
        const tensor_t* d = inputs[2];
        tensor_t*       y = outputs[0];

        if (s->type != l->type || s->type != d->type) return false;
        const char* ty_w = wgsl_ty(s->type);
        if (!ty_w) return false;

        if (!read_scalar(s, start) || !read_scalar(l, limit) || !read_scalar(d, delta))
            return false;
        if (delta == 0) return false;
        double raw = (limit - start) / delta;
        int64_t len = raw > 0 ? (int64_t)ceil(raw) : 0;
        if (len < 0) len = 0;
        if (len > (int64_t)0x7fffffff) return false;
        n = (uint32_t)len;

        const int dims[1] = { (int)n };
        if (!y->reshape(std::span<const int>(dims, 1), s->type)) return false;

        if (!pipeline || built_ty != s->type) {
            if (!build_pipeline(ty_w)) return false;
            built_ty = s->type;
        }

        webgpu::ensure_buffer(y, (size_t)y->ndata * data_type_sizeof(s->type));

        // Uniform depends only on n/start/delta — reshape-time constants.
        float sf = (float)start, df = (float)delta;
        uint32_t u[4];
        u[0] = n; u[1] = 0;
        std::memcpy(&u[2], &sf, 4);
        std::memcpy(&u[3], &df, 4);
        webgpu::get_device().queue.WriteBuffer(uniforms, 0, u, sizeof(u));
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();

        auto* ry = webgpu::find(outputs[0]);
        uint32_t gen_y = webgpu::generation_of(outputs[0]);
        if (!cached_bg || gen_y != cached_gen[0]) {
            wgpu::BindGroupEntry be[2] = {};
            be[0].binding = 0; be[0].buffer = ry->buf;   be[0].offset = 0; be[0].size = ry->size;
            be[1].binding = 1; be[1].buffer = uniforms;  be[1].offset = 0; be[1].size = 16;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 2; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_y;
        }

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, cached_bg);
        uint32_t groups = (n + WG - 1) / WG;
        pass.DispatchWorkgroups(groups, 1, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_Range_webgpu(int, pool_t& pool) {
    return pool_new<Range_operator_webgpu>(pool);
}

} // namespace nnr
