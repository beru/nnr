// WebGPU QLinearAdd — quantized element-wise addition.
//
// Inputs: (A, A_scale, A_zp, B, B_scale, B_zp, Y_scale, Y_zp).
// Output: Y = clamp(round((A - A_zp) * A_scale / Y_scale +
//                         (B - B_zp) * B_scale / Y_scale + Y_zp), qmin, qmax)
//
// Algebra hoists the constant offset out of the inner loop:
//   sa = A_scale / Y_scale,  sb = B_scale / Y_scale
//   fixed = Y_zp - A_zp*sa - B_zp*sb
//   Y = clamp(round(A*sa + B*sb + fixed), qmin, qmax)
//
// Same-shape only in this first cut. Broadcasting (one operand is a scalar
// or has a different rank) → reshape() returns false → CPU fallback. The
// QDQ models in the bench are all same-shape Adds.

#include "nnr.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include <string>
#include <webgpu/webgpu_cpp.h>

namespace nnr {

namespace {

struct QLinearAdd_operator_webgpu : public operator_t {
    enum kind_t { K_NONE, K_U8, K_I8 };
    kind_t built_kind = K_NONE;

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          uniforms;   // 16 B: { n, n_packed, grid_stride_x, _pad }
    wgpu::Buffer          consts;     // 16 B: { sa, sb, fixed, _pad }
    uint32_t              dispatch_gx = 0;
    uint32_t              dispatch_gy = 0;

    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[3] = {};

    bool init() override {
        if (inputs.size() != 8 || outputs.size() != 1) return false;
        if (!webgpu::device_ready()) return false;
        layout_mask = LAYOUT_ALL;
        return true;
    }

    bool reshape() override {
        const tensor_t* a = inputs[0];
        const tensor_t* b = inputs[3];
        if (a->type != NNR_DATA_TYPE_UINT8 && a->type != NNR_DATA_TYPE_INT8) return false;
        if (b->type != a->type) return false;
        // Same-shape only. Treat scalar-broadcast as same-shape only when ndata matches.
        if (a->ndata != b->ndata) return false;
        // All scale tensors must be scalar f32 (per-tensor quant).
        if (inputs[1]->type != NNR_DATA_TYPE_FLOAT32 || inputs[1]->ndata != 1) return false;
        if (inputs[4]->type != NNR_DATA_TYPE_FLOAT32 || inputs[4]->ndata != 1) return false;
        if (inputs[6]->type != NNR_DATA_TYPE_FLOAT32 || inputs[6]->ndata != 1) return false;

        if (!outputs[0]->reshape_multi_broadcast(a, b, a->type)) return false;
        tensor_t* y = outputs[0];

        kind_t want = (a->type == NNR_DATA_TYPE_INT8) ? K_I8 : K_U8;
        if (!pipeline || built_kind != want) {
            if (!build_pipeline(want)) return false;
            built_kind = want;
        }

        webgpu::ensure_buffer(a, (size_t)a->ndata);
        webgpu::ensure_buffer(b, (size_t)b->ndata);
        webgpu::ensure_buffer(y, (size_t)y->ndata);

        // Upload scales and zero_points are CPU-side reads. Compute the
        // hoisted constants now so the shader doesn't reload them.
        float a_scale = ((const float*)inputs[1]->data)[0];
        float b_scale = ((const float*)inputs[4]->data)[0];
        float y_scale = ((const float*)inputs[6]->data)[0];
        int32_t a_zp = 0, b_zp = 0, y_zp = 0;
        if (inputs[2] && inputs[2]->ndata > 0) {
            if (a->type == NNR_DATA_TYPE_INT8) a_zp = ((const int8_t*)inputs[2]->data)[0];
            else                                a_zp = ((const uint8_t*)inputs[2]->data)[0];
        }
        if (inputs[5] && inputs[5]->ndata > 0) {
            if (a->type == NNR_DATA_TYPE_INT8) b_zp = ((const int8_t*)inputs[5]->data)[0];
            else                                b_zp = ((const uint8_t*)inputs[5]->data)[0];
        }
        if (inputs[7] && inputs[7]->ndata > 0) {
            if (a->type == NNR_DATA_TYPE_INT8) y_zp = ((const int8_t*)inputs[7]->data)[0];
            else                                y_zp = ((const uint8_t*)inputs[7]->data)[0];
        }
        float sa = a_scale / y_scale;
        float sb = b_scale / y_scale;
        float fixed = (float)y_zp - (sa * (float)a_zp + sb * (float)b_zp);
        float c[4] = { sa, sb, fixed, 0.0f };
        webgpu::get_device().queue.WriteBuffer(consts, 0, c, sizeof(c));

        const uint32_t WG = 256;
        const uint32_t n_packed = ((uint32_t)y->ndata + 3u) / 4u;
        uint32_t groups = (n_packed + WG - 1) / WG;
        webgpu::dispatch_1d_grid(groups, dispatch_gx, dispatch_gy);
        uint32_t u[4] = { (uint32_t)y->ndata, n_packed, dispatch_gx * WG, 0 };
        webgpu::get_device().queue.WriteBuffer(uniforms, 0, u, sizeof(u));
        return true;
    }

    bool build_pipeline(kind_t kind) {
        auto& dev = webgpu::get_device();
        const bool signed_t = (kind == K_I8);
        const int qmin = signed_t ? -128 : 0;
        const int qmax = signed_t ?  127 : 255;

        std::string src;
        src += "struct Dims { n: u32, n_packed: u32, grid_stride_x: u32, _p: u32 };\n";
        src += "struct K { sa: f32, sb: f32, fixed: f32, _p: f32 };\n";
        src += "@group(0) @binding(0) var<storage, read>       A : array<u32>;\n";  // packed bytes
        src += "@group(0) @binding(1) var<storage, read>       B : array<u32>;\n";
        src += "@group(0) @binding(2) var<storage, read_write> Y : array<u32>;\n";
        src += "@group(0) @binding(3) var<uniform>             dims : Dims;\n";
        src += "@group(0) @binding(4) var<uniform>             k    : K;\n";

        const char* extend_signed =
            "  return select(i32(raw), i32(raw) - 256, raw >= 128u);\n";
        const char* extend_unsigned =
            "  return i32(raw);\n";

        src += "fn extract(word: u32, k: u32) -> i32 {\n";
        src += "  let raw = (word >> (k * 8u)) & 0xFFu;\n";
        src += signed_t ? extend_signed : extend_unsigned;
        src += "}\n";

        src += "@compute @workgroup_size(256)\n";
        src += "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n";
        src += "  let i_packed = gid.y * dims.grid_stride_x + gid.x;\n";
        src += "  if (i_packed >= dims.n_packed) { return; }\n";
        src += "  let aw = A[i_packed];\n";
        src += "  let bw = B[i_packed];\n";
        src += "  var packed = 0u;\n";
        src += "  for (var p = 0u; p < 4u; p = p + 1u) {\n";
        src += "    let idx = i_packed * 4u + p;\n";
        src += "    if (idx >= dims.n) { break; }\n";
        src += "    let av = extract(aw, p);\n";
        src += "    let bv = extract(bw, p);\n";
        src += "    let f = f32(av) * k.sa + f32(bv) * k.sb + k.fixed;\n";
        src += "    var q = i32(round(f));\n";
        src += "    q = clamp(q, " + std::to_string(qmin) + ", " + std::to_string(qmax) + ");\n";
        src += "    packed = packed | ((u32(q) & 0xFFu) << (p * 8u));\n";
        src += "  }\n";
        src += "  Y[i_packed] = packed;\n";
        src += "}\n";

        wgpu::ShaderSourceWGSL wgslSrc = {};
        wgslSrc.code = src.c_str();
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &wgslSrc;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

        if (!bgl) {
            wgpu::BindGroupLayoutEntry e[5] = {};
            e[0].binding = 0; e[0].visibility = wgpu::ShaderStage::Compute;
            e[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
            e[1].binding = 1; e[1].visibility = wgpu::ShaderStage::Compute;
            e[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
            e[2].binding = 2; e[2].visibility = wgpu::ShaderStage::Compute;
            e[2].buffer.type = wgpu::BufferBindingType::Storage;
            e[3].binding = 3; e[3].visibility = wgpu::ShaderStage::Compute;
            e[3].buffer.type = wgpu::BufferBindingType::Uniform;
            e[4].binding = 4; e[4].visibility = wgpu::ShaderStage::Compute;
            e[4].buffer.type = wgpu::BufferBindingType::Uniform;
            wgpu::BindGroupLayoutDescriptor bgld = {};
            bgld.entryCount = 5; bgld.entries = e;
            bgl = dev.device.CreateBindGroupLayout(&bgld);

            wgpu::BufferDescriptor ud = {};
            ud.size = 16;
            ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
            uniforms = dev.device.CreateBuffer(&ud);
            consts   = dev.device.CreateBuffer(&ud);
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
        webgpu::upload_if_needed(inputs[3]);

        auto* ra = webgpu::find(inputs[0]);
        auto* rb = webgpu::find(inputs[3]);
        auto* ry = webgpu::find(outputs[0]);

        uint32_t gen_a = webgpu::generation_of(inputs[0]);
        uint32_t gen_b = webgpu::generation_of(inputs[3]);
        uint32_t gen_y = webgpu::generation_of(outputs[0]);
        auto pad4 = [](size_t n) -> uint64_t { return (n + 3) & ~size_t(3); };
        if (!cached_bg
            || gen_a != cached_gen[0] || gen_b != cached_gen[1] || gen_y != cached_gen[2]) {
            wgpu::BindGroupEntry be[5] = {};
            be[0].binding = 0; be[0].buffer = ra->buf;  be[0].offset = 0; be[0].size = pad4(ra->size);
            be[1].binding = 1; be[1].buffer = rb->buf;  be[1].offset = 0; be[1].size = pad4(rb->size);
            be[2].binding = 2; be[2].buffer = ry->buf;  be[2].offset = 0; be[2].size = pad4(ry->size);
            be[3].binding = 3; be[3].buffer = uniforms; be[3].offset = 0; be[3].size = 16;
            be[4].binding = 4; be[4].buffer = consts;   be[4].offset = 0; be[4].size = 16;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 5; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_a;
            cached_gen[1] = gen_b;
            cached_gen[2] = gen_y;
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

operator_t* resolver_default_op_QLinearAdd_webgpu(int, pool_t& pool) {
    return pool_new<QLinearAdd_operator_webgpu>(pool);
}

} // namespace nnr
