// WebGPU QLinearConv — quantized 2D NCHW convolution.
//
// Inputs (8 or 9):
//   0: X        (u8/i8)        [N, C, H, W]
//   1: x_scale  (f32)          scalar
//   2: x_zp     (u8/i8)        scalar
//   3: W        (u8/i8)        [M, C/groups, kH, kW]
//   4: w_scale  (f32)          scalar OR per-output-channel [M]
//   5: w_zp     (u8/i8)        scalar OR per-output-channel [M]
//   6: y_scale  (f32)          scalar
//   7: y_zp     (u8/i8)        scalar
//   8: B        (i32, opt)     [M]
//
// Output Y (u8/i8): same dtype as X.
//
// Per-output-element math (all in i32 until the final scale):
//   acc = sum_{c, kh, kw} (X[n,c,ih,iw] - x_zp) * (W[m,c,kh,kw] - w_zp[m])
//   if has_bias: acc += B[m]
//   y = clamp(round(acc * x_scale * w_scale[m] / y_scale + y_zp), qmin, qmax)
//
// First cut: one thread per output element, no tiling. The byte unpack
// inside the inner loop is doing more work than a fp32 conv, but it
// runs entirely on-device — this op was a CPU fallback before. Tiling
// + register fusion (post-op absorb) is the next pass.

#include "nnr.h"
#include "attr_key.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include <string>
#include <webgpu/webgpu_cpp.h>

namespace nnr {

namespace {

constexpr uint32_t WG = 64;

struct QLinearConv_operator_webgpu : public operator_t {
    // Conv geometry (filled in reshape).
    uint32_t meta_vals[20] = {};
    int H_out_i = 0, W_out_i = 0, N_i = 0, M_i = 0;

    // 3-bit kind = signed_x | (signed_w << 1) | (per_axis << 2)
    int  built_kind = -1;
    bool built_has_bias = false;

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          meta_buf;
    wgpu::Buffer          dummy_bias;  // 16 B zeros (i32) when no bias

    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[8] = {};

    bool init() override {
        if (inputs.size() != 8 && inputs.size() != 9) return false;
        if (outputs.size() != 1) return false;
        if (!webgpu::device_ready()) return false;
        // Reject unsupported X dtypes early.
        if (inputs[0] && inputs[0]->type != NNR_DATA_TYPE_UNDEFINED
            && inputs[0]->type != NNR_DATA_TYPE_UINT8
            && inputs[0]->type != NNR_DATA_TYPE_INT8) return false;
        if (inputs[0] && inputs[0]->ndim > 0 && inputs[0]->ndim != 4) return false;
        if (inputs[3] && inputs[3]->ndim > 0 && inputs[3]->ndim != 4) return false;
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        const tensor_t* w = inputs[3];
        if (x->type != NNR_DATA_TYPE_UINT8 && x->type != NNR_DATA_TYPE_INT8) return false;
        if (w->type != NNR_DATA_TYPE_UINT8 && w->type != NNR_DATA_TYPE_INT8) return false;
        // X and W may differ in signedness — fuse_qdq_compute converts u8 W
        // to i8 in place but leaves X u8.

        // Attributes (mirrors Conv.cpp API).
        std::string_view auto_pad = attribute(attr_key_t::auto_pad, "NOTSET");
        if (auto_pad != "NOTSET" && auto_pad != "VALID") return false;  // TODO: SAME_*
        int groups_i = (int)attribute(attr_key_t::group, (int64_t)1);
        if (groups_i < 1) return false;

        int64_t* ints = nullptr;
        int nstride = attribute(attr_key_t::strides, ints);
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
        int strides[2]   = {s_h, s_w};
        int dilations[2] = {d_h, d_w};
        int pads[4]      = {pad_top, pad_left, pad_bot, pad_right};

        const int N = x->dims[0];
        const int C = x->dims[1];
        const int H = x->dims[2];
        const int W = x->dims[3];
        const int M = w->dims[0];
        const int C_per_group = w->dims[1];
        const int kH = w->dims[2];
        const int kW = w->dims[3];
        if (C / groups_i != C_per_group) return false;

        // Output dims (NCHW only).
        auto eff = [](int dim, int k, int s, int d, int pa, int pb) {
            int eff_k = (k - 1) * d + 1;
            return (dim + pa + pb - eff_k) / s + 1;
        };
        int H_out = eff(H, kH, strides[0], dilations[0], pads[0], pads[2]);
        int W_out = eff(W, kW, strides[1], dilations[1], pads[1], pads[3]);
        if (H_out <= 0 || W_out <= 0) return false;

        small_vector<int> ydims = {N, M, H_out, W_out};
        if (!outputs[0]->reshape(ydims, x->type)) return false;
        tensor_t* y = outputs[0];

        // Scale tensors must be f32. Per-channel only on weight scale.
        if (inputs[1]->type != NNR_DATA_TYPE_FLOAT32 || inputs[1]->ndata != 1) return false;
        if (inputs[4]->type != NNR_DATA_TYPE_FLOAT32) return false;
        if ((size_t)inputs[4]->ndata != 1 && (size_t)inputs[4]->ndata != (size_t)M) return false;
        if (inputs[6]->type != NNR_DATA_TYPE_FLOAT32 || inputs[6]->ndata != 1) return false;
        // Zero-points: x_zp and y_zp must be scalar or empty. w_zp must be
        // scalar (per-tensor) or have ndata == M (per-channel matching scale).
        // Each zero-point's type must match its corresponding tensor's type
        // (X-side / W-side / Y-side may all differ in signedness).
        if (inputs[2]->ndata > 1) return false;
        if (inputs[7]->ndata > 1) return false;
        if (inputs[2]->ndata > 0 && inputs[2]->type != x->type) return false;
        if (inputs[7]->ndata > 0 && inputs[7]->type != x->type) return false;
        const bool per_axis = (inputs[4]->ndata > 1);
        if (per_axis) {
            if (inputs[5]->ndata != inputs[4]->ndata) return false;
            if (inputs[5]->type != w->type) return false;
        } else {
            if (inputs[5]->ndata > 1) return false;
            if (inputs[5]->ndata > 0 && inputs[5]->type != w->type) return false;
        }
        // Bias must be i32 [M] if present.
        const bool has_bias = (inputs.size() == 9 && inputs[8] && inputs[8]->ndata > 0);
        if (has_bias) {
            if (inputs[8]->type != NNR_DATA_TYPE_INT32) return false;
            if ((size_t)inputs[8]->ndata != (size_t)M) return false;
        }

        const bool signed_x = (x->type == NNR_DATA_TYPE_INT8);
        const bool signed_w = (w->type == NNR_DATA_TYPE_INT8);
        const int  want = (int)signed_x | ((int)signed_w << 1) | ((int)per_axis << 2);
        if (!pipeline || built_kind != want || built_has_bias != has_bias) {
            if (!build_pipeline(signed_x, signed_w, per_axis, has_bias)) return false;
            built_kind = want;
            built_has_bias = has_bias;
        }

        // Buffers — only allocate for tensors we actually bind. Scalar
        // x_zp / y_zp / x_scale / y_scale are read CPU-side once and
        // packed into the meta uniform (no GPU buffer needed).
        webgpu::ensure_buffer(x, (size_t)x->ndata);
        webgpu::ensure_buffer(w, (size_t)w->ndata);
        webgpu::ensure_buffer(inputs[4], inputs[4]->ndata * sizeof(float));
        if (per_axis) webgpu::ensure_buffer(inputs[5], (size_t)inputs[5]->ndata);
        if (has_bias) webgpu::ensure_buffer(inputs[8], (size_t)inputs[8]->ndata * sizeof(int32_t));
        webgpu::ensure_buffer(y, (size_t)y->ndata);

        // Pack scalars CPU-side into the meta buffer.
        // Scalars (x_scale, y_scale, x_zp, y_zp) are read once per element,
        // so plumbing them through the meta uniform avoids one bind slot.
        float x_scale = ((const float*)inputs[1]->data)[0];
        float y_scale = ((const float*)inputs[6]->data)[0];
        int32_t x_zp = 0, y_zp = 0;
        if (inputs[2]->ndata > 0) {
            x_zp = signed_x ? ((const int8_t*)inputs[2]->data)[0]
                            : ((const uint8_t*)inputs[2]->data)[0];
        }
        if (inputs[7]->ndata > 0) {
            // Y has the same dtype as X.
            y_zp = signed_x ? ((const int8_t*)inputs[7]->data)[0]
                            : ((const uint8_t*)inputs[7]->data)[0];
        }
        // For per-tensor weights, compute combined scale on host once.
        // For per-axis, the shader reads w_scale[m] and computes per-row.
        float w_scale_pt = 0.0f;
        int32_t w_zp_pt = 0;
        if (!per_axis) {
            w_scale_pt = ((const float*)inputs[4]->data)[0];
            if (inputs[5]->ndata > 0) {
                w_zp_pt = signed_w ? ((const int8_t*)inputs[5]->data)[0]
                                   : ((const uint8_t*)inputs[5]->data)[0];
            }
        }

        const uint32_t total = (uint32_t)y->ndata;
        meta_vals[0]  = total;
        meta_vals[1]  = (uint32_t)N;
        meta_vals[2]  = (uint32_t)M;
        meta_vals[3]  = (uint32_t)H_out;
        meta_vals[4]  = (uint32_t)W_out;
        meta_vals[5]  = (uint32_t)C;
        meta_vals[6]  = (uint32_t)groups_i;
        meta_vals[7]  = (uint32_t)kH;
        meta_vals[8]  = (uint32_t)kW;
        meta_vals[9]  = (uint32_t)strides[0];
        meta_vals[10] = (uint32_t)strides[1];
        meta_vals[11] = (uint32_t)pads[0];   // top
        meta_vals[12] = (uint32_t)pads[1];   // left
        meta_vals[13] = (uint32_t)dilations[0];
        meta_vals[14] = (uint32_t)dilations[1];
        meta_vals[15] = (uint32_t)H;
        meta_vals[16] = (uint32_t)W;
        // Pack scalar quant params: x_zp, y_zp, w_zp_pt as i32; x_scale, y_scale, w_scale_pt as f32.
        // Shader unions these via type punning isn't allowed in WGSL — use two buffers
        // (meta_buf int part + a float-only mini-uniform). Simpler: repurpose meta_buf
        // as half-int / half-float at fixed offsets. WGSL struct can mix u32/f32 fields.
        meta_vals[17] = (uint32_t)x_zp;
        meta_vals[18] = (uint32_t)y_zp;
        meta_vals[19] = (uint32_t)w_zp_pt;
        // float fields — reinterpret bits into the same buffer at different offsets.
        // We emit a struct with f32 fields after the u32 fields; WGSL aligns each
        // field naturally, all are 4 bytes.
        float meta_floats[3] = { x_scale, y_scale, w_scale_pt };
        struct MetaWire {
            uint32_t u[20];
            float    f[3];
            uint32_t has_bias;
        } wire;
        std::memcpy(wire.u, meta_vals, sizeof(meta_vals));
        std::memcpy(wire.f, meta_floats, sizeof(meta_floats));
        wire.has_bias = has_bias ? 1u : 0u;
        webgpu::get_device().queue.WriteBuffer(meta_buf, 0, &wire, sizeof(wire));

        N_i = N; M_i = M; H_out_i = H_out; W_out_i = W_out;
        return true;
    }

    bool build_pipeline(bool signed_x, bool signed_w, bool per_axis, bool has_bias) {
        auto& dev = webgpu::get_device();
        // Output Y has the same dtype as X (set in reshape via reshape_identity-like
        // path); qmin/qmax follow X's signedness.
        const int qmin = signed_x ? -128 : 0;
        const int qmax = signed_x ?  127 : 255;

        // Helper macro snippets for byte unpack.
        const char* unpack_signed =
            "  return select(i32(raw), i32(raw) - 256, raw >= 128u);\n";
        const char* unpack_unsigned =
            "  return i32(raw);\n";

        std::string s;
        s += "struct Meta {\n";
        s += "  total: u32, N: u32, M: u32, H_out: u32, W_out: u32,\n";
        s += "  C_in: u32, groups: u32, kH: u32, kW: u32,\n";
        s += "  stride_h: u32, stride_w: u32, pad_top: u32, pad_left: u32,\n";
        s += "  dilation_h: u32, dilation_w: u32, H: u32, W: u32,\n";
        s += "  x_zp: i32, y_zp: i32, w_zp_pt: i32,\n";
        s += "  x_scale: f32, y_scale: f32, w_scale_pt: f32,\n";
        s += "  has_bias: u32,\n";
        s += "};\n";

        s += "@group(0) @binding(0) var<storage, read>       X       : array<u32>;\n"; // packed bytes
        s += "@group(0) @binding(1) var<storage, read>       Wt      : array<u32>;\n"; // packed bytes
        s += "@group(0) @binding(2) var<storage, read>       Wscale  : array<f32>;\n"; // [M] or [1]
        s += "@group(0) @binding(3) var<storage, read>       Wzp     : array<u32>;\n"; // packed bytes
        s += "@group(0) @binding(4) var<storage, read>       Bias    : array<i32>;\n"; // [M] or [1]
        s += "@group(0) @binding(5) var<storage, read_write> Y       : array<u32>;\n"; // packed bytes (each thread writes one full u32 = 4 outputs)
        s += "@group(0) @binding(6) var<storage, read>       md      : Meta;\n";

        s += "fn x_at(idx: u32) -> i32 {\n";
        s += "  let word = X[idx >> 2u];\n";
        s += "  let shift = (idx & 3u) * 8u;\n";
        s += "  let raw = (word >> shift) & 0xFFu;\n";
        s += signed_x ? unpack_signed : unpack_unsigned;
        s += "}\n";

        s += "fn w_at(idx: u32) -> i32 {\n";
        s += "  let word = Wt[idx >> 2u];\n";
        s += "  let shift = (idx & 3u) * 8u;\n";
        s += "  let raw = (word >> shift) & 0xFFu;\n";
        s += signed_w ? unpack_signed : unpack_unsigned;
        s += "}\n";

        s += "fn w_zp_at(m: u32) -> i32 {\n";
        if (per_axis) {
            s += "  let word = Wzp[m >> 2u];\n";
            s += "  let shift = (m & 3u) * 8u;\n";
            s += "  let raw = (word >> shift) & 0xFFu;\n";
            s += signed_w ? unpack_signed : unpack_unsigned;
        } else {
            s += "  return md.w_zp_pt;\n";
        }
        s += "}\n";

        s += "fn w_scale_at(m: u32) -> f32 {\n";
        if (per_axis) {
            s += "  return Wscale[m];\n";
        } else {
            s += "  return md.w_scale_pt;\n";
        }
        s += "}\n";

        // Each thread computes 4 sequential output bytes and writes a full u32
        // word — no atomics, no contention. Total threads = ceil(total/4).
        // The 4 outputs are 4 sequential flat indices; in NCHW that's 4 ow
        // positions (when W_out >= 4) or spans an oh boundary (rare). Each
        // packed slot does its own conv compute, so boundary spans are fine.
        s += "@compute @workgroup_size(64)\n";
        s += "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n";
        s += "  let i_packed = gid.x;\n";
        s += "  let n_packed = (md.total + 3u) / 4u;\n";
        s += "  if (i_packed >= n_packed) { return; }\n";

        s += "  let M_per_group    = md.M    / md.groups;\n";
        s += "  let C_in_per_group = md.C_in / md.groups;\n";
        s += "  let x_batch_stride = md.C_in * md.H * md.W;\n";
        s += "  let x_chan_stride  = md.H * md.W;\n";
        s += "  let w_outch_stride = C_in_per_group * md.kH * md.kW;\n";
        s += "  let w_inch_stride  = md.kH * md.kW;\n";

        s += "  var packed : u32 = 0u;\n";
        s += "  for (var p : u32 = 0u; p < 4u; p = p + 1u) {\n";
        s += "    let i = i_packed * 4u + p;\n";
        s += "    if (i >= md.total) { break; }\n";
        s += "    let ow = i % md.W_out;\n";
        s += "    var tmp = i / md.W_out;\n";
        s += "    let oh = tmp % md.H_out;\n";
        s += "    tmp = tmp / md.H_out;\n";
        s += "    let m = tmp % md.M;\n";
        s += "    let n = tmp / md.M;\n";
        s += "    let g = m / M_per_group;\n";
        s += "    let base_x = n * x_batch_stride;\n";
        s += "    let base_w = m * w_outch_stride;\n";
        s += "    let wzp = w_zp_at(m);\n";
        s += "    var acc : i32 = 0;\n";
        s += "    for (var ic : u32 = 0u; ic < C_in_per_group; ic = ic + 1u) {\n";
        s += "      let c = g * C_in_per_group + ic;\n";
        s += "      let x_c_base = base_x + c * x_chan_stride;\n";
        s += "      let w_c_base = base_w + ic * w_inch_stride;\n";
        s += "      for (var kh : u32 = 0u; kh < md.kH; kh = kh + 1u) {\n";
        s += "        let ih_i = i32(oh * md.stride_h) + i32(kh * md.dilation_h) - i32(md.pad_top);\n";
        s += "        if (ih_i < 0 || ih_i >= i32(md.H)) { continue; }\n";
        s += "        let ih = u32(ih_i);\n";
        s += "        let x_h_base = x_c_base + ih * md.W;\n";
        s += "        let w_h_base = w_c_base + kh * md.kW;\n";
        s += "        for (var kw : u32 = 0u; kw < md.kW; kw = kw + 1u) {\n";
        s += "          let iw_i = i32(ow * md.stride_w) + i32(kw * md.dilation_w) - i32(md.pad_left);\n";
        s += "          if (iw_i < 0 || iw_i >= i32(md.W)) { continue; }\n";
        s += "          let iw = u32(iw_i);\n";
        s += "          let xv = x_at(x_h_base + iw) - md.x_zp;\n";
        s += "          let wv = w_at(w_h_base + kw) - wzp;\n";
        s += "          acc = acc + xv * wv;\n";
        s += "        }\n";
        s += "      }\n";
        s += "    }\n";
        if (has_bias) {
            s += "    acc = acc + Bias[m];\n";
        }
        s += "    let combined = md.x_scale * w_scale_at(m) / md.y_scale;\n";
        s += "    let scaled = f32(acc) * combined + f32(md.y_zp);\n";
        s += "    var q = i32(round(scaled));\n";
        s += "    q = clamp(q, " + std::to_string(qmin) + ", " + std::to_string(qmax) + ");\n";
        s += "    packed = packed | ((u32(q) & 0xFFu) << (p * 8u));\n";
        s += "  }\n";
        s += "  Y[i_packed] = packed;\n";
        s += "}\n";

        wgpu::ShaderSourceWGSL wgslSrc = {};
        wgslSrc.code = s.c_str();
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &wgslSrc;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

        if (!bgl) {
            wgpu::BindGroupLayoutEntry e[7] = {};
            for (int i = 0; i < 7; ++i) {
                e[i].binding = (uint32_t)i;
                e[i].visibility = wgpu::ShaderStage::Compute;
            }
            e[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;  // X
            e[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;  // W
            e[2].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;  // Wscale
            e[3].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;  // Wzp
            e[4].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;  // Bias
            e[5].buffer.type = wgpu::BufferBindingType::Storage;          // Y (atomic)
            e[6].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;  // Meta
            wgpu::BindGroupLayoutDescriptor bgld = {};
            bgld.entryCount = 7; bgld.entries = e;
            bgl = dev.device.CreateBindGroupLayout(&bgld);

            // Meta uniform: 20 u32 + 3 f32 + 1 u32 = 24 × 4 = 96 bytes.
            wgpu::BufferDescriptor md_desc = {};
            md_desc.size = 96;
            md_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
            meta_buf = dev.device.CreateBuffer(&md_desc);

            // Dummy bias (16 zero bytes = 4 i32 zeros).
            wgpu::BufferDescriptor db_desc = {};
            db_desc.size = 16;
            db_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
            dummy_bias = dev.device.CreateBuffer(&db_desc);
            uint32_t zeros[4] = {0, 0, 0, 0};
            dev.queue.WriteBuffer(dummy_bias, 0, zeros, sizeof(zeros));
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
        webgpu::upload_if_needed(inputs[4]);
        if (inputs[5]->ndata > 0) webgpu::upload_if_needed(inputs[5]);
        const bool has_bias = (inputs.size() == 9 && inputs[8] && inputs[8]->ndata > 0);
        if (has_bias) webgpu::upload_if_needed(inputs[8]);

        auto* rx  = webgpu::find(inputs[0]);
        auto* rw  = webgpu::find(inputs[3]);
        auto* rws = webgpu::find(inputs[4]);
        auto* rwz = inputs[5]->ndata > 0 ? webgpu::find(inputs[5]) : nullptr;
        auto* rb  = has_bias ? webgpu::find(inputs[8]) : nullptr;
        auto* ry  = webgpu::find(outputs[0]);

        uint32_t gen[6] = {
            webgpu::generation_of(inputs[0]),
            webgpu::generation_of(inputs[3]),
            webgpu::generation_of(inputs[4]),
            inputs[5]->ndata > 0 ? webgpu::generation_of(inputs[5]) : 0u,
            has_bias ? webgpu::generation_of(inputs[8]) : 0u,
            webgpu::generation_of(outputs[0]),
        };
        bool stale = !cached_bg;
        for (int i = 0; i < 6 && !stale; ++i) stale = (gen[i] != cached_gen[i]);
        // Storage bindings require 4-aligned sizes. Round up; capacity is
        // already 4-aligned in ensure_buffer, but `size` may not be (e.g.,
        // scalar u8 zero-points are 1 byte). Using capacity for the bind
        // size is safe — the shader bounds-checks all reads against the
        // logical element count via meta uniforms.
        auto pad4 = [](size_t n) -> uint64_t { return (n + 3) & ~size_t(3); };
        if (stale) {
            wgpu::BindGroupEntry be[7] = {};
            be[0].binding = 0; be[0].buffer = rx->buf;  be[0].offset = 0; be[0].size = pad4(rx->size);
            be[1].binding = 1; be[1].buffer = rw->buf;  be[1].offset = 0; be[1].size = pad4(rw->size);
            be[2].binding = 2; be[2].buffer = rws->buf; be[2].offset = 0; be[2].size = pad4(rws->size);
            if (rwz) { be[3].binding = 3; be[3].buffer = rwz->buf; be[3].offset = 0; be[3].size = pad4(rwz->size); }
            else     { be[3].binding = 3; be[3].buffer = rws->buf; be[3].offset = 0; be[3].size = pad4(rws->size); }
            if (rb)  { be[4].binding = 4; be[4].buffer = rb->buf;  be[4].offset = 0; be[4].size = pad4(rb->size); }
            else     { be[4].binding = 4; be[4].buffer = dummy_bias; be[4].offset = 0; be[4].size = 16; }
            be[5].binding = 5; be[5].buffer = ry->buf;  be[5].offset = 0; be[5].size = pad4(ry->size);
            be[6].binding = 6; be[6].buffer = meta_buf; be[6].offset = 0; be[6].size = 96;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 7; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            std::memcpy(cached_gen, gen, sizeof(gen));
        }

        // Each thread writes one u32 = 4 output bytes. Total threads = ceil(total/4).
        const uint32_t total = (uint32_t)outputs[0]->ndata;
        const uint32_t n_packed = (total + 3u) / 4u;
        uint32_t groups = (n_packed + WG - 1) / WG;
        uint32_t gx, gy;
        webgpu::dispatch_1d_grid(groups, gx, gy);

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, cached_bg);
        pass.DispatchWorkgroups(gx, gy, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_QLinearConv_webgpu(int, pool_t& pool) {
    return pool_new<QLinearConv_operator_webgpu>(pool);
}

} // namespace nnr
