// WebGPU ConstantOfShape — output a tensor of the given shape filled with
// a single scalar constant.
//
// ONNX: `value` attribute is a rank-0 or rank-1-length-1 tensor carrying
// both the output dtype and the fill value. Input is a 1-D int64 tensor
// of target dimensions (CPU-resident).
//
// Scope: f32 / i32 / u32 output dtypes, matching Cast's dtype coverage.
// Other dtypes → CPU fallback. The input shape tensor must be CPU-
// resident (typical since ONNX Constant folders fold it before runtime).

#include "nnr.h"
#include "attr_key.h"
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
    case NNR_DATA_TYPE_UINT32:  return "u32";
    default:                    return nullptr;
    }
}

struct ConstantOfShape_op_webgpu : public operator_t {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          uniforms;   // 16B: {n, 0, value_bits, 0}
    data_type_t           fill_type  = NNR_DATA_TYPE_FLOAT32;
    data_type_t           built_ty   = NNR_DATA_TYPE_UNDEFINED;
    uint32_t              value_bits = 0;
    uint32_t              n_out      = 0;

    // Cached BindGroup. Tensor-backed slot: [Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[1] = {};

    bool init() override {
        if (!is_inout_size(1, 1)) return false;
        if (!webgpu::device_ready()) return false;

        // Pull the scalar value from the `value` attribute. If missing,
        // default is f32 zero (per ONNX spec).
        attr_t* a = find_attr(attr_key_t::value);
        if (a && a->kind == attr_t::kind_t::TENSOR && a->tensor) {
            const tensor_t* vt = a->tensor;
            fill_type = vt->type;
            // Extract the scalar's raw bits from the attribute tensor.
            switch (vt->type) {
            case NNR_DATA_TYPE_FLOAT32:
                if (vt->data && vt->ndata > 0)
                    std::memcpy(&value_bits, vt->data, 4);
                break;
            case NNR_DATA_TYPE_INT32:
            case NNR_DATA_TYPE_UINT32:
                if (vt->data && vt->ndata > 0)
                    std::memcpy(&value_bits, vt->data, 4);
                break;
            default:
                // Other dtypes (int64 common for ONNX masks/indices) → CPU.
                return false;
            }
        } else {
            fill_type  = NNR_DATA_TYPE_FLOAT32;
            value_bits = 0;   // f32 0.0
        }

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

    bool build_pipeline(const char* ty) {
        auto& dev = webgpu::get_device();
        // value is stored as u32 bits; reinterpret via bitcast for the
        // target type so f32/i32/u32 all share one uniform layout.
        std::string src =
            "struct Cfg { n : u32, _a : u32, value_bits : u32, _b : u32 };\n"
            "@group(0) @binding(0) var<storage, read_write> Y   : array<";
        src += ty;
        src += ">;\n"
               "@group(0) @binding(1) var<uniform>             cfg : Cfg;\n"
               "@compute @workgroup_size(256)\n"
               "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
               "  let i = gid.x;\n"
               "  if (i >= cfg.n) { return; }\n"
               "  Y[i] = bitcast<";
        src += ty;
        src += ">(cfg.value_bits);\n}\n";

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
        const tensor_t* shape = inputs[0];
        tensor_t*       y     = outputs[0];
        if (shape->type != NNR_DATA_TYPE_INT64) return false;
        if (!shape->data || shape->ndim != 1)   return false;   // CPU-resident only

        const char* ty = wgsl_ty(fill_type);
        if (!ty) return false;

        int target_ndim = (int)shape->ndata;
        if (target_ndim < 0 || target_ndim > 8) return false;
        const int64_t* pd = (const int64_t*)shape->data;
        int dims[8] = {};
        for (int i = 0; i < target_ndim; ++i) dims[i] = (int)pd[i];
        if (!y->reshape(std::span<const int>(dims, target_ndim), fill_type)) return false;

        if (!pipeline || built_ty != fill_type) {
            if (!build_pipeline(ty)) return false;
            built_ty = fill_type;
        }

        webgpu::ensure_buffer(y, y->ndata * data_type_sizeof(fill_type));

        n_out = (uint32_t)y->ndata;
        uint32_t u[4] = { n_out, 0, value_bits, 0 };
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
        uint32_t groups = (n_out + WG - 1) / WG;
        pass.DispatchWorkgroups(groups, 1, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_ConstantOfShape_webgpu(int, pool_t& pool) {
    return pool_new<ConstantOfShape_op_webgpu>(pool);
}

} // namespace nnr
