// WebGPU QLinearMatMul — quantized 2D MatMul.
//
// Inputs (8):
//   0: A         (u8/i8)        [M, K]  (only 2D for now)
//   1: A_scale   (f32)          scalar
//   2: A_zp      (u8/i8)        scalar
//   3: B         (u8/i8)        [K, N]
//   4: B_scale   (f32)          scalar OR per-N
//   5: B_zp      (u8/i8)        scalar OR per-N
//   6: Y_scale   (f32)          scalar
//   7: Y_zp      (u8/i8)        scalar
//
// Output Y (u8/i8): same dtype as A.
//
// Math (per output element):
//   acc = sum_k (A[m,k] - A_zp) * (B[k,n] - B_zp[n])
//   y   = clamp(round(acc * A_scale * B_scale[n] / Y_scale + Y_zp), qmin, qmax)
//
// First cut: simple 16×16 workgroup tile, no shared-memory cooperative
// loads. One output element per thread. atomicOr packing for byte
// outputs (4 threads share a u32 word).

#include "nnr.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include <string>
#include <webgpu/webgpu_cpp.h>

namespace nnr {

namespace {

constexpr uint32_t TILE_M = 16;
constexpr uint32_t TILE_N = 16;

struct QLinearMatMul_operator_webgpu : public operator_t {
    int m_ = 0, n_ = 0, k_ = 0;

    enum kind_t { K_NONE, K_U8_PT, K_U8_PA, K_I8_PT, K_I8_PA };
    kind_t built_kind = K_NONE;

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          meta_buf;

    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[6] = {};

    bool init() override {
        if (inputs.size() != 8 || outputs.size() != 1) return false;
        if (!webgpu::device_ready()) return false;
        return true;
    }

    bool reshape() override {
        const tensor_t* a = inputs[0];
        const tensor_t* b = inputs[3];
        if (a->type != NNR_DATA_TYPE_UINT8 && a->type != NNR_DATA_TYPE_INT8) return false;
        if (b->type != a->type) return false;
        // 2D × 2D only for now.
        if (a->ndim != 2 || b->ndim != 2) return false;

        const int M = a->dims[0];
        const int K = a->dims[1];
        if (b->dims[0] != K) return false;
        const int N = b->dims[1];

        if (inputs[1]->type != NNR_DATA_TYPE_FLOAT32 || inputs[1]->ndata != 1) return false;
        if (inputs[4]->type != NNR_DATA_TYPE_FLOAT32) return false;
        if ((size_t)inputs[4]->ndata != 1 && (size_t)inputs[4]->ndata != (size_t)N) return false;
        if (inputs[6]->type != NNR_DATA_TYPE_FLOAT32 || inputs[6]->ndata != 1) return false;

        if (inputs[2]->ndata > 1) return false;
        if (inputs[7]->ndata > 1) return false;
        const bool per_axis = (inputs[4]->ndata > 1);
        if (per_axis) {
            if (inputs[5]->ndata != inputs[4]->ndata) return false;
            if (inputs[5]->type != a->type) return false;
        } else {
            if (inputs[5]->ndata > 1) return false;
            if (inputs[5]->ndata > 0 && inputs[5]->type != a->type) return false;
        }

        small_vector<int> ydims = {M, N};
        if (!outputs[0]->reshape(ydims, a->type)) return false;
        tensor_t* y = outputs[0];

        const bool signed_t = (a->type == NNR_DATA_TYPE_INT8);
        kind_t want = signed_t ? (per_axis ? K_I8_PA : K_I8_PT)
                                : (per_axis ? K_U8_PA : K_U8_PT);
        if (!pipeline || built_kind != want) {
            if (!build_pipeline(want)) return false;
            built_kind = want;
        }

        webgpu::ensure_buffer(a, (size_t)a->ndata);
        webgpu::ensure_buffer(b, (size_t)b->ndata);
        webgpu::ensure_buffer(inputs[4], inputs[4]->ndata * sizeof(float));
        if (per_axis) webgpu::ensure_buffer(inputs[5], (size_t)inputs[5]->ndata);
        webgpu::ensure_buffer(y, (size_t)y->ndata);

        // Pack scalars into meta.
        float a_scale = ((const float*)inputs[1]->data)[0];
        float y_scale = ((const float*)inputs[6]->data)[0];
        int32_t a_zp = 0, y_zp = 0;
        if (inputs[2]->ndata > 0) {
            a_zp = signed_t ? ((const int8_t*)inputs[2]->data)[0]
                            : ((const uint8_t*)inputs[2]->data)[0];
        }
        if (inputs[7]->ndata > 0) {
            y_zp = signed_t ? ((const int8_t*)inputs[7]->data)[0]
                            : ((const uint8_t*)inputs[7]->data)[0];
        }
        float b_scale_pt = 0.0f;
        int32_t b_zp_pt = 0;
        if (!per_axis) {
            b_scale_pt = ((const float*)inputs[4]->data)[0];
            if (inputs[5]->ndata > 0) {
                b_zp_pt = signed_t ? ((const int8_t*)inputs[5]->data)[0]
                                   : ((const uint8_t*)inputs[5]->data)[0];
            }
        }

        struct MetaWire {
            uint32_t M, N, K;
            int32_t  a_zp, y_zp, b_zp_pt;
            float    a_scale, y_scale, b_scale_pt;
            uint32_t _pad;
        } wire;
        wire.M = (uint32_t)M; wire.N = (uint32_t)N; wire.K = (uint32_t)K;
        wire.a_zp = a_zp; wire.y_zp = y_zp; wire.b_zp_pt = b_zp_pt;
        wire.a_scale = a_scale; wire.y_scale = y_scale; wire.b_scale_pt = b_scale_pt;
        wire._pad = 0;
        webgpu::get_device().queue.WriteBuffer(meta_buf, 0, &wire, sizeof(wire));

        m_ = M; n_ = N; k_ = K;
        return true;
    }

    bool build_pipeline(kind_t kind) {
        auto& dev = webgpu::get_device();
        const bool signed_t = (kind == K_I8_PT || kind == K_I8_PA);
        const bool per_axis = (kind == K_U8_PA || kind == K_I8_PA);
        const int qmin = signed_t ? -128 : 0;
        const int qmax = signed_t ?  127 : 255;

        const char* unpack_signed =
            "  return select(i32(raw), i32(raw) - 256, raw >= 128u);\n";
        const char* unpack_unsigned =
            "  return i32(raw);\n";

        std::string s;
        s += "struct Meta {\n";
        s += "  M: u32, N: u32, K: u32,\n";
        s += "  a_zp: i32, y_zp: i32, b_zp_pt: i32,\n";
        s += "  a_scale: f32, y_scale: f32, b_scale_pt: f32,\n";
        s += "  _pad: u32,\n};\n";

        s += "@group(0) @binding(0) var<storage, read>       A      : array<u32>;\n";
        s += "@group(0) @binding(1) var<storage, read>       Bt     : array<u32>;\n";
        s += "@group(0) @binding(2) var<storage, read>       Bscale : array<f32>;\n";
        s += "@group(0) @binding(3) var<storage, read>       Bzp    : array<u32>;\n";
        s += "@group(0) @binding(4) var<storage, read_write> Yatom  : array<atomic<u32>>;\n";
        s += "@group(0) @binding(5) var<storage, read>       md     : Meta;\n";

        s += "fn a_at(idx: u32) -> i32 {\n";
        s += "  let word = A[idx >> 2u];\n";
        s += "  let shift = (idx & 3u) * 8u;\n";
        s += "  let raw = (word >> shift) & 0xFFu;\n";
        s += signed_t ? unpack_signed : unpack_unsigned;
        s += "}\n";

        s += "fn b_at(idx: u32) -> i32 {\n";
        s += "  let word = Bt[idx >> 2u];\n";
        s += "  let shift = (idx & 3u) * 8u;\n";
        s += "  let raw = (word >> shift) & 0xFFu;\n";
        s += signed_t ? unpack_signed : unpack_unsigned;
        s += "}\n";

        s += "fn b_zp_at(n: u32) -> i32 {\n";
        if (per_axis) {
            s += "  let word = Bzp[n >> 2u];\n";
            s += "  let shift = (n & 3u) * 8u;\n";
            s += "  let raw = (word >> shift) & 0xFFu;\n";
            s += signed_t ? unpack_signed : unpack_unsigned;
        } else {
            s += "  return md.b_zp_pt;\n";
        }
        s += "}\n";

        s += "fn b_scale_at(n: u32) -> f32 {\n";
        s += per_axis ? "  return Bscale[n];\n" : "  return md.b_scale_pt;\n";
        s += "}\n";

        s += "@compute @workgroup_size(16, 16, 1)\n";
        s += "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n";
        s += "  let m = gid.y;\n";
        s += "  let n = gid.x;\n";
        s += "  if (m >= md.M || n >= md.N) { return; }\n";

        s += "  let bzp = b_zp_at(n);\n";
        s += "  var acc : i32 = 0;\n";
        s += "  for (var k : u32 = 0u; k < md.K; k = k + 1u) {\n";
        s += "    let av = a_at(m * md.K + k) - md.a_zp;\n";
        s += "    let bv = b_at(k * md.N + n) - bzp;\n";
        s += "    acc = acc + av * bv;\n";
        s += "  }\n";
        s += "  let combined = md.a_scale * b_scale_at(n) / md.y_scale;\n";
        s += "  let scaled = f32(acc) * combined + f32(md.y_zp);\n";
        s += "  var q = i32(round(scaled));\n";
        s += "  q = clamp(q, " + std::to_string(qmin) + ", " + std::to_string(qmax) + ");\n";

        s += "  let i = m * md.N + n;\n";
        s += "  let byte = u32(q & 0xFF);\n";
        s += "  let word_idx = i >> 2u;\n";
        s += "  let bit_shift = (i & 3u) * 8u;\n";
        s += "  atomicOr(&Yatom[word_idx], byte << bit_shift);\n";
        s += "}\n";

        wgpu::ShaderSourceWGSL wgslSrc = {};
        wgslSrc.code = s.c_str();
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &wgslSrc;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

        if (!bgl) {
            wgpu::BindGroupLayoutEntry e[6] = {};
            for (int i = 0; i < 6; ++i) {
                e[i].binding = (uint32_t)i;
                e[i].visibility = wgpu::ShaderStage::Compute;
            }
            e[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;  // A
            e[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;  // B
            e[2].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;  // Bscale
            e[3].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;  // Bzp
            e[4].buffer.type = wgpu::BufferBindingType::Storage;          // Y atomic
            e[5].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;  // Meta
            wgpu::BindGroupLayoutDescriptor bgld = {};
            bgld.entryCount = 6; bgld.entries = e;
            bgl = dev.device.CreateBindGroupLayout(&bgld);

            wgpu::BufferDescriptor md_desc = {};
            md_desc.size = 48;  // 12 fields × 4
            md_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
            meta_buf = dev.device.CreateBuffer(&md_desc);
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
        const bool per_axis = (inputs[4]->ndata > 1);
        if (per_axis) webgpu::upload_if_needed(inputs[5]);

        auto* ra = webgpu::find(inputs[0]);
        auto* rb = webgpu::find(inputs[3]);
        auto* rbs = webgpu::find(inputs[4]);
        auto* rbz = per_axis ? webgpu::find(inputs[5]) : nullptr;
        auto* ry = webgpu::find(outputs[0]);

        // Atomic OR pack — output must start at zero.
        webgpu::shared_encoder().ClearBuffer(ry->buf, 0, ry->size);

        uint32_t gen[5] = {
            webgpu::generation_of(inputs[0]),
            webgpu::generation_of(inputs[3]),
            webgpu::generation_of(inputs[4]),
            per_axis ? webgpu::generation_of(inputs[5]) : 0u,
            webgpu::generation_of(outputs[0]),
        };
        bool stale = !cached_bg;
        for (int i = 0; i < 5 && !stale; ++i) stale = (gen[i] != cached_gen[i]);
        auto pad4 = [](size_t n) -> uint64_t { return (n + 3) & ~size_t(3); };
        if (stale) {
            wgpu::BindGroupEntry be[6] = {};
            be[0].binding = 0; be[0].buffer = ra->buf;  be[0].offset = 0; be[0].size = pad4(ra->size);
            be[1].binding = 1; be[1].buffer = rb->buf;  be[1].offset = 0; be[1].size = pad4(rb->size);
            be[2].binding = 2; be[2].buffer = rbs->buf; be[2].offset = 0; be[2].size = pad4(rbs->size);
            if (rbz) { be[3].binding = 3; be[3].buffer = rbz->buf; be[3].offset = 0; be[3].size = pad4(rbz->size); }
            else     { be[3].binding = 3; be[3].buffer = rbs->buf; be[3].offset = 0; be[3].size = pad4(rbs->size); }
            be[4].binding = 4; be[4].buffer = ry->buf;  be[4].offset = 0; be[4].size = pad4(ry->size);
            be[5].binding = 5; be[5].buffer = meta_buf; be[5].offset = 0; be[5].size = 48;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 6; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            std::memcpy(cached_gen, gen, sizeof(gen));
        }

        const uint32_t gx = (n_ + TILE_N - 1) / TILE_N;
        const uint32_t gy = (m_ + TILE_M - 1) / TILE_M;
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

operator_t* resolver_default_op_QLinearMatMul_webgpu(int, pool_t& pool) {
    return pool_new<QLinearMatMul_operator_webgpu>(pool);
}

} // namespace nnr
