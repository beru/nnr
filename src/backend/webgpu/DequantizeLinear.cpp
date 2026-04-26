// WebGPU DequantizeLinear — element-wise uint8 / int8 → FP32 dequantization.
//
// Mirrors QuantizeLinear in input/output expectations: 8-bit input
// (uint8 or int8) packed as u32 on the GPU, f32 output, scalar or
// per-axis scale. Float8/Float4/int16/int32 inputs and blockwise
// quantization → reshape() returns false → CPU fallback.

#include "nnr.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include <string>
#include <webgpu/webgpu_cpp.h>

namespace nnr {

namespace {

struct DequantizeLinear_operator_webgpu : public operator_t {
    int axis = 1;
    int block_size = 0;

    enum kind_t { K_NONE, K_PT_U8, K_PT_I8, K_PA_U8, K_PA_I8 };
    kind_t built_kind = K_NONE;

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          uniforms;
    uint32_t              dispatch_gx = 0;
    uint32_t              dispatch_gy = 0;

    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[4] = {};

    bool init() override {
        if (inputs.size() < 2 || outputs.size() != 1) return false;
        if (!webgpu::device_ready()) return false;
        axis = attribute(attr_key_t::axis, (int32_t)1);
        block_size = attribute(attr_key_t::block_size, (int32_t)0);
        layout_mask = LAYOUT_ALL;
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        const tensor_t* sc = inputs[1];
        const tensor_t* zp = (inputs.size() > 2 && inputs[2] && inputs[2]->ndata > 0) ? inputs[2] : nullptr;

        if (x->type != NNR_DATA_TYPE_UINT8 && x->type != NNR_DATA_TYPE_INT8) return false;
        if (sc->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (zp && zp->type != x->type) return false;
        if (block_size != 0) return false;

        if (!outputs[0]->reshape_identity(x, NNR_DATA_TYPE_FLOAT32)) return false;
        tensor_t* y = outputs[0];

        const bool per_axis = (sc->ndata > 1);
        const bool signed_in = (x->type == NNR_DATA_TYPE_INT8);
        kind_t want = per_axis
            ? (signed_in ? K_PA_I8 : K_PA_U8)
            : (signed_in ? K_PT_I8 : K_PT_U8);

        if (per_axis) {
            int caxis = axis < 0 ? axis + x->ndim : axis;
            if (caxis < 0 || caxis >= x->ndim) return false;
            if ((size_t)x->dims[caxis] != sc->ndata) return false;
            if (zp && zp->ndata != sc->ndata) return false;
        }

        if (!pipeline || built_kind != want) {
            if (!build_pipeline(want)) return false;
            built_kind = want;
        }

        webgpu::ensure_buffer(x, (size_t)x->ndata);  // u8/i8 padded by ensure_buffer
        webgpu::ensure_buffer(sc, sc->ndata * sizeof(float));
        if (zp) webgpu::ensure_buffer(zp, zp->ndata);
        webgpu::ensure_buffer(y, (size_t)y->ndata * sizeof(float));

        // One thread = one output f32 element (no packing on the output side
        // since f32 is already 4 bytes/element).
        const uint32_t WG = 256;
        uint32_t groups = ((uint32_t)y->ndata + WG - 1) / WG;
        webgpu::dispatch_1d_grid(groups, dispatch_gx, dispatch_gy);

        uint32_t inner = 1, axis_size = 1;
        if (per_axis) {
            int caxis = axis < 0 ? axis + x->ndim : axis;
            axis_size = (uint32_t)x->dims[caxis];
            for (int i = caxis + 1; i < x->ndim; ++i) inner *= (uint32_t)x->dims[i];
        }
        uint32_t u[8] = {
            (uint32_t)y->ndata, dispatch_gx * WG, axis_size, inner,
            zp ? 1u : 0u, 0u, 0u, 0u
        };
        webgpu::get_device().queue.WriteBuffer(uniforms, 0, u, sizeof(u));
        return true;
    }

    bool build_pipeline(kind_t kind) {
        auto& dev = webgpu::get_device();

        const bool signed_in = (kind == K_PT_I8 || kind == K_PA_I8);
        const bool per_axis  = (kind == K_PA_U8 || kind == K_PA_I8);

        std::string src;
        src += "struct Dims { n: u32, grid_stride_x: u32, axis_size: u32, inner: u32, has_zp: u32, _p0: u32, _p1: u32, _p2: u32 };\n";
        src += "@group(0) @binding(0) var<storage, read>       X  : array<u32>;\n";  // packed bytes
        src += "@group(0) @binding(1) var<storage, read>       Sc : array<f32>;\n";
        src += "@group(0) @binding(2) var<storage, read>       Zp : array<u32>;\n";  // packed bytes
        src += "@group(0) @binding(3) var<storage, read_write> Y  : array<f32>;\n";
        src += "@group(0) @binding(4) var<uniform>             dims : Dims;\n";

        // Read one byte at element index `idx` from a packed-u32 array,
        // returning the value as i32 (signed-extended for INT8, zero-extended
        // for UINT8).
        const char* extend_signed =
            "  return select(i32(raw), i32(raw) - 256, raw >= 128u);\n";
        const char* extend_unsigned =
            "  return i32(raw);\n";

        src += "fn x_at(idx: u32) -> i32 {\n";
        src += "  let word = X[idx >> 2u];\n";
        src += "  let shift = (idx & 3u) * 8u;\n";
        src += "  let raw = (word >> shift) & 0xFFu;\n";
        src += signed_in ? extend_signed : extend_unsigned;
        src += "}\n";

        src += "fn zp_at(idx: u32) -> i32 {\n";
        src += "  let word = Zp[idx >> 2u];\n";
        src += "  let shift = (idx & 3u) * 8u;\n";
        src += "  let raw = (word >> shift) & 0xFFu;\n";
        src += signed_in ? extend_signed : extend_unsigned;
        src += "}\n";

        src += "@compute @workgroup_size(256)\n";
        src += "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n";
        src += "  let i = gid.y * dims.grid_stride_x + gid.x;\n";
        src += "  if (i >= dims.n) { return; }\n";
        if (per_axis) {
            src += "  let axis_idx = (i / dims.inner) % dims.axis_size;\n";
            src += "  let s  = Sc[axis_idx];\n";
            src += "  let zp = select(0, zp_at(axis_idx), dims.has_zp != 0u);\n";
        } else {
            src += "  let s  = Sc[0];\n";
            src += "  let zp = select(0, zp_at(0u), dims.has_zp != 0u);\n";
        }
        src += "  Y[i] = f32(x_at(i) - zp) * s;\n";
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
            e[2].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
            e[3].binding = 3; e[3].visibility = wgpu::ShaderStage::Compute;
            e[3].buffer.type = wgpu::BufferBindingType::Storage;
            e[4].binding = 4; e[4].visibility = wgpu::ShaderStage::Compute;
            e[4].buffer.type = wgpu::BufferBindingType::Uniform;
            wgpu::BindGroupLayoutDescriptor bgld = {};
            bgld.entryCount = 5; bgld.entries = e;
            bgl = dev.device.CreateBindGroupLayout(&bgld);

            wgpu::BufferDescriptor ud = {};
            ud.size = 32;
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
        webgpu::upload_if_needed(inputs[1]);
        const tensor_t* zp_t = (inputs.size() > 2 && inputs[2] && inputs[2]->ndata > 0) ? inputs[2] : nullptr;
        if (zp_t) webgpu::upload_if_needed(zp_t);

        auto* rx  = webgpu::find(inputs[0]);
        auto* rsc = webgpu::find(inputs[1]);
        auto* rzp = zp_t ? webgpu::find(zp_t) : nullptr;
        auto* ry  = webgpu::find(outputs[0]);

        uint32_t gen_x  = webgpu::generation_of(inputs[0]);
        uint32_t gen_sc = webgpu::generation_of(inputs[1]);
        uint32_t gen_zp = zp_t ? webgpu::generation_of(zp_t) : 0;
        uint32_t gen_y  = webgpu::generation_of(outputs[0]);
        auto pad4 = [](size_t n) -> uint64_t { return (n + 3) & ~size_t(3); };
        if (!cached_bg
            || gen_x  != cached_gen[0] || gen_sc != cached_gen[1]
            || gen_zp != cached_gen[2] || gen_y  != cached_gen[3]) {
            wgpu::BindGroupEntry be[5] = {};
            be[0].binding = 0; be[0].buffer = rx->buf;  be[0].offset = 0; be[0].size = pad4(rx->size);
            be[1].binding = 1; be[1].buffer = rsc->buf; be[1].offset = 0; be[1].size = pad4(rsc->size);
            if (rzp) {
                be[2].binding = 2; be[2].buffer = rzp->buf; be[2].offset = 0; be[2].size = pad4(rzp->size);
            } else {
                be[2].binding = 2; be[2].buffer = rsc->buf; be[2].offset = 0; be[2].size = pad4(rsc->size);
            }
            be[3].binding = 3; be[3].buffer = ry->buf;  be[3].offset = 0; be[3].size = pad4(ry->size);
            be[4].binding = 4; be[4].buffer = uniforms; be[4].offset = 0; be[4].size = 32;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 5; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_x;
            cached_gen[1] = gen_sc;
            cached_gen[2] = gen_zp;
            cached_gen[3] = gen_y;
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

operator_t* resolver_default_op_DequantizeLinear_webgpu(int, pool_t& pool) {
    return pool_new<DequantizeLinear_operator_webgpu>(pool);
}

} // namespace nnr
