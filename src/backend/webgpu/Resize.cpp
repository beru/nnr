// WebGPU Resize — 2D NCHW, mode="nearest" or mode="linear".
//
// Scope:
//   - X is 4D NCHW (batch, channel, height, width), float32.
//   - Output Y is [N, C, H_out, W_out] with scales[0] == scales[1] == 1.0
//     (batch and channel untouched — the universal PyTorch/ONNX export
//     convention for image upsample).
//   - mode = "nearest" or "linear" (2D bilinear). cubic → CPU fallback.
//   - coord mode in {half_pixel, pytorch_half_pixel, asymmetric,
//     align_corners, tf_half_pixel_for_nn, half_pixel_symmetric}.
//   - nearest mode in {round_prefer_floor, round_prefer_ceil, floor, ceil}.
//   - antialias=0, exclude_outside=0, no ROI (tf_crop_and_resize not
//     supported — CPU fallback).
//
// Shape determination mirrors the CPU op: input[1] may be ROI (ignored),
// input[2] scales, input[3] sizes. If sizes is present it wins; otherwise
// scales determine H_out / W_out.

#include "nnr.h"
#include "attr_key.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include <cstring>
#include <string_view>
#include <webgpu/webgpu_cpp.h>

namespace nnr {

namespace {

constexpr uint32_t WG = 64;

// Coordinate mode enum — kept tight with the WGSL branch numbering.
enum class coord_mode_u32 : uint32_t {
    half_pixel            = 0,
    pytorch_half_pixel    = 1,
    asymmetric            = 2,
    align_corners         = 3,
    tf_half_pixel_for_nn  = 4,
    half_pixel_symmetric  = 5,
};

enum class nearest_mode_u32 : uint32_t {
    round_prefer_floor = 0,
    round_prefer_ceil  = 1,
    floor_m            = 2,
    ceil_m             = 3,
};

enum class interp_mode_u32 : uint32_t {
    nearest = 0,
    linear  = 1,
};

struct Resize_operator_webgpu : public operator_t {
    coord_mode_u32   coord_mode   = coord_mode_u32::half_pixel;
    nearest_mode_u32 nearest_mode = nearest_mode_u32::round_prefer_floor;
    interp_mode_u32  interp_mode  = interp_mode_u32::nearest;

    int N = 0, C = 0, iH = 0, iW = 0, oH = 0, oW = 0;
    float scale_h = 1.0f, scale_w = 1.0f;

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          meta_buf;
    uint32_t              total_out = 0;

    // Cached BindGroup. Tensor-backed slots: [X, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[2] = {};

    // Single compiled shader; interp / coord / nearest mode selectors are
    // runtime branches driven by the meta buffer. One pipeline per backend
    // instance.
    static constexpr const char* kWgsl =
        "struct Meta {\n"
        "  total          : u32,\n"
        "  N              : u32,\n"
        "  C              : u32,\n"
        "  iH             : u32,\n"
        "  iW             : u32,\n"
        "  oH             : u32,\n"
        "  oW             : u32,\n"
        "  coord_mode     : u32,\n"
        "  nearest_mode   : u32,\n"
        "  scale_h_bits   : u32,\n"
        "  scale_w_bits   : u32,\n"
        "  interp_mode    : u32,\n"
        "};\n"
        "@group(0) @binding(0) var<storage, read>       X  : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> Y  : array<f32>;\n"
        "@group(0) @binding(2) var<storage, read>       md : Meta;\n"
        "fn transform_coord(out_coord : f32, scale : f32, in_size : u32, out_size : u32, mode : u32) -> f32 {\n"
        "  if (mode == 0u) { return (out_coord + 0.5) / scale - 0.5; }\n"             // half_pixel
        "  if (mode == 1u) {\n"                                                        // pytorch_half_pixel
        "    if (out_size > 1u) { return (out_coord + 0.5) / scale - 0.5; }\n"
        "    return 0.0;\n"
        "  }\n"
        "  if (mode == 2u) { return out_coord / scale; }\n"                            // asymmetric
        "  if (mode == 3u) {\n"                                                        // align_corners
        "    let len_resized = f32(in_size) * scale;\n"
        "    if (len_resized > 1.0) {\n"
        "      return out_coord * (f32(in_size) - 1.0) / (len_resized - 1.0);\n"
        "    }\n"
        "    return 0.0;\n"
        "  }\n"
        "  if (mode == 4u) { return (out_coord + 0.5) / scale; }\n"                    // tf_half_pixel_for_nn
        "  if (mode == 5u) {\n"                                                        // half_pixel_symmetric
        "    let adj = f32(out_size) / (scale * f32(in_size));\n"
        "    let center = f32(in_size) * 0.5;\n"
        "    let offset = center * (1.0 - adj);\n"
        "    return offset + (out_coord + 0.5) / scale - 0.5;\n"
        "  }\n"
        "  return out_coord / scale;\n"
        "}\n"
        "fn nearest_idx(coord : f32, in_size : u32, mode : u32) -> u32 {\n"
        "  var idx : i32;\n"
        "  if (mode == 0u) {\n"                                                        // round_prefer_floor
        // WGSL round: round-to-nearest-even. ONNX wants round-half-away-
        // from-zero with floor tiebreaker. Detect exact .5 and force floor.
        "    let f = floor(coord);\n"
        "    if (coord == f + 0.5) { idx = i32(f); }\n"
        "    else { idx = i32(round(coord)); }\n"
        "  } else if (mode == 1u) {\n"                                                 // round_prefer_ceil
        "    let f = floor(coord);\n"
        "    if (coord == f + 0.5) { idx = i32(f + 1.0); }\n"
        "    else { idx = i32(round(coord)); }\n"
        "  } else if (mode == 2u) {\n"                                                 // floor
        "    idx = i32(floor(coord));\n"
        "  } else {\n"                                                                 // ceil
        "    idx = i32(ceil(coord));\n"
        "  }\n"
        "  let hi = i32(in_size) - 1;\n"
        "  if (idx < 0)  { idx = 0; }\n"
        "  if (idx > hi) { idx = hi; }\n"
        "  return u32(idx);\n"
        "}\n"
        "fn clamp_i32(v : i32, hi : i32) -> i32 {\n"
        "  var r = v;\n"
        "  if (r < 0) { r = 0; }\n"
        "  if (r > hi) { r = hi; }\n"
        "  return r;\n"
        "}\n"
        "@compute @workgroup_size(64)\n"
        "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
        "  let i = gid.x;\n"
        "  if (i >= md.total) { return; }\n"
        "  let ow = i % md.oW;\n"
        "  var tmp = i / md.oW;\n"
        "  let oh = tmp % md.oH;\n"
        "  tmp = tmp / md.oH;\n"
        "  let c = tmp % md.C;\n"
        "  let n = tmp / md.C;\n"
        "  let scale_h = bitcast<f32>(md.scale_h_bits);\n"
        "  let scale_w = bitcast<f32>(md.scale_w_bits);\n"
        "  let fy = transform_coord(f32(oh), scale_h, md.iH, md.oH, md.coord_mode);\n"
        "  let fx = transform_coord(f32(ow), scale_w, md.iW, md.oW, md.coord_mode);\n"
        "  let nc_base = (n * md.C + c) * md.iH * md.iW;\n"
        "  if (md.interp_mode == 0u) {\n"                                               // nearest
        "    let iy = nearest_idx(fy, md.iH, md.nearest_mode);\n"
        "    let ix = nearest_idx(fx, md.iW, md.nearest_mode);\n"
        "    Y[i] = X[nc_base + iy * md.iW + ix];\n"
        "    return;\n"
        "  }\n"
        // interp_mode == 1 (linear / bilinear): 4-corner weighted sample,
        // edge-clamped. Standard ONNX "linear" semantics.
        "  let hi_y = i32(md.iH) - 1;\n"
        "  let hi_x = i32(md.iW) - 1;\n"
        "  let y0i = i32(floor(fy));\n"
        "  let x0i = i32(floor(fx));\n"
        "  let dy  = fy - f32(y0i);\n"
        "  let dx  = fx - f32(x0i);\n"
        "  let y0 = u32(clamp_i32(y0i,     hi_y));\n"
        "  let y1 = u32(clamp_i32(y0i + 1, hi_y));\n"
        "  let x0 = u32(clamp_i32(x0i,     hi_x));\n"
        "  let x1 = u32(clamp_i32(x0i + 1, hi_x));\n"
        "  let v00 = X[nc_base + y0 * md.iW + x0];\n"
        "  let v01 = X[nc_base + y0 * md.iW + x1];\n"
        "  let v10 = X[nc_base + y1 * md.iW + x0];\n"
        "  let v11 = X[nc_base + y1 * md.iW + x1];\n"
        "  let top = v00 * (1.0 - dx) + v01 * dx;\n"
        "  let bot = v10 * (1.0 - dx) + v11 * dx;\n"
        "  Y[i] = top * (1.0 - dy) + bot * dy;\n"
        "}\n";

    bool init() override {
        if (inputs.size() < 1 || outputs.size() != 1) return false;
        if (!webgpu::device_ready()) return false;

        // Scope check: mode must be "nearest" or "linear". Attribute read
        // happens here so we can reject at solve-time rather than reshape-
        // time. cubic and antialias fall back to CPU.
        std::string_view mode_str = attribute(attr_key_t::mode, "nearest");
        if      (mode_str == "nearest") interp_mode = interp_mode_u32::nearest;
        else if (mode_str == "linear")  interp_mode = interp_mode_u32::linear;
        else return false;

        std::string_view cm = attribute(attr_key_t::coordinate_transformation_mode,
                                        "half_pixel");
        if      (cm == "half_pixel")            coord_mode = coord_mode_u32::half_pixel;
        else if (cm == "pytorch_half_pixel")    coord_mode = coord_mode_u32::pytorch_half_pixel;
        else if (cm == "asymmetric")            coord_mode = coord_mode_u32::asymmetric;
        else if (cm == "align_corners")         coord_mode = coord_mode_u32::align_corners;
        else if (cm == "tf_half_pixel_for_nn")  coord_mode = coord_mode_u32::tf_half_pixel_for_nn;
        else if (cm == "half_pixel_symmetric")  coord_mode = coord_mode_u32::half_pixel_symmetric;
        else return false;  // tf_crop_and_resize needs ROI — CPU fallback.

        std::string_view nm = attribute(attr_key_t::nearest_mode,
                                        "round_prefer_floor");
        if      (nm == "round_prefer_floor") nearest_mode = nearest_mode_u32::round_prefer_floor;
        else if (nm == "round_prefer_ceil")  nearest_mode = nearest_mode_u32::round_prefer_ceil;
        else if (nm == "floor")              nearest_mode = nearest_mode_u32::floor_m;
        else if (nm == "ceil")               nearest_mode = nearest_mode_u32::ceil_m;
        else return false;

        // Antialias / exclude_outside / non-default keep_aspect_ratio / cubic
        // coeff are no-ops for nearest mode — silently ignored.
        int antialias = (int)attribute(attr_key_t::antialias, (int64_t)0);
        if (antialias) return false;

        auto& dev = webgpu::get_device();
        wgpu::ShaderSourceWGSL w = {};
        w.code = kWgsl;
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
        md.size = 48;   // 12 u32s (rounded naturally)
        md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        meta_buf = dev.device.CreateBuffer(&md);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t*       y = outputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (x->ndim != 4) return false;

        N  = x->dims[0];
        C  = x->dims[1];
        iH = x->dims[2];
        iW = x->dims[3];

        // ONNX opset 11/13+ positional: [X, roi, scales, sizes]. Some exports
        // omit roi (opset 19). Prefer sizes if present, else scales. We only
        // need scales/sizes to touch axes 2/3 (spatial) — other axes must be
        // passthrough (scale=1.0), otherwise CPU fallback.
        const tensor_t* sizes_t  = nullptr;
        const tensor_t* scales_t = nullptr;
        if (inputs.size() >= 4 && inputs[3] && inputs[3]->ndata > 0) sizes_t  = inputs[3];
        if (inputs.size() >= 3 && inputs[2] && inputs[2]->ndata > 0) scales_t = inputs[2];

        if (sizes_t) {
            if ((int)sizes_t->ndata != 4)                   return false;
            if (sizes_t->type != NNR_DATA_TYPE_INT64)       return false;
            const int64_t* sz = (const int64_t*)sizes_t->data;
            if ((int)sz[0] != N || (int)sz[1] != C)         return false;  // batch/channel must match
            oH = (int)sz[2];
            oW = (int)sz[3];
            if (oH <= 0 || oW <= 0)                         return false;
            scale_h = (iH > 0) ? (float)oH / (float)iH : 1.0f;
            scale_w = (iW > 0) ? (float)oW / (float)iW : 1.0f;
        } else if (scales_t) {
            if ((int)scales_t->ndata != 4)                  return false;
            if (scales_t->type != NNR_DATA_TYPE_FLOAT32)    return false;
            const float* sc = (const float*)scales_t->data;
            if (sc[0] != 1.0f || sc[1] != 1.0f)             return false;
            scale_h = sc[2];
            scale_w = sc[3];
            oH = (int)std::floor((float)iH * scale_h);
            oW = (int)std::floor((float)iW * scale_w);
            if (oH <= 0 || oW <= 0)                         return false;
        } else {
            return false;  // no scale or size info → CPU
        }

        int out_dims[4] = { N, C, oH, oW };
        if (!y->reshape(std::span<const int>(out_dims, 4), NNR_DATA_TYPE_FLOAT32))
            return false;

        webgpu::ensure_buffer(x, x->ndata * sizeof(float));
        webgpu::ensure_buffer(y, y->ndata * sizeof(float));

        // Pack meta as 12 u32s. `bitcast<f32>` in the shader reads them back
        // as floats; keeping scalar packing avoids vec4 alignment gymnastics
        // that a uniform buffer would otherwise demand.
        uint8_t buf[48] = {};
        auto put_u = [&](size_t off, uint32_t v) { std::memcpy(buf + off, &v, 4); };
        auto put_f = [&](size_t off, float v)    { std::memcpy(buf + off, &v, 4); };
        total_out = (uint32_t)N * (uint32_t)C * (uint32_t)oH * (uint32_t)oW;
        put_u(0,  total_out);
        put_u(4,  (uint32_t)N);
        put_u(8,  (uint32_t)C);
        put_u(12, (uint32_t)iH);
        put_u(16, (uint32_t)iW);
        put_u(20, (uint32_t)oH);
        put_u(24, (uint32_t)oW);
        put_u(28, (uint32_t)coord_mode);
        put_u(32, (uint32_t)nearest_mode);
        put_f(36, scale_h);
        put_f(40, scale_w);
        put_u(44, (uint32_t)interp_mode);
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
            be[2].binding = 2; be[2].buffer = meta_buf;  be[2].offset = 0; be[2].size = 48;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 3; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_x;
            cached_gen[1] = gen_y;
        }

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, cached_bg);
        const uint32_t groups = (total_out + WG - 1) / WG;
        pass.DispatchWorkgroups(groups, 1, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_Resize_webgpu(int, pool_t& pool) {
    return pool_new<Resize_operator_webgpu>(pool);
}

} // namespace nnr
