// WebGPU TopK — find the K largest (or smallest) values along one axis.
//
// ONNX: two outputs, values (same dtype as input) and indices (INT64).
// Attributes: axis (default -1), largest (1/0, default 1), sorted (1/0,
// default 1). For opset ≥ 10, K is inputs[1], an int64 scalar read CPU-side
// at reshape time; opset ≤ 9 has K as an attribute.
//
// Implementation is the simplest correct version: one thread per output
// slice (outer, inner), which loops K times and on each pass scans the
// reduce axis to find the next best element. O(axis_dim * K) per slice —
// fine for the typical K (1–64 for beam search / sampling) and typical
// axis sizes. Ties break toward the smaller index (matches CPU reference
// and ONNX "reproducibility" note).
//
// Dedup across passes uses a local `picked` array of K indices; K is
// bounded to 256 to keep the stack slot reasonable. Larger K falls back
// to CPU.
//
// Scope: f32 input only. Integer/half inputs → CPU fallback via registry.

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

constexpr uint32_t WG = 64;
// Cap K at 1024 — bounds output buffer footprint and makes axis_dim
// validation in reshape() trivial. Production detector heads use K <= 300.
constexpr int K_MAX = 1024;
// Cap axis_dim so the workgroup-shared dedup bitmap (1 bit per axis
// position) fits in a fixed shared-mem allocation. 32768 bits = 4 KB,
// well under the 16 KB workgroup-mem limit even with the reduction
// scratch. SSD-12 / yolo detector heads peak around axis=16000.
constexpr int AXIS_MAX = 32768;

struct TopK_op_webgpu : public operator_t {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          uniforms;
    int axis              = -1;
    int largest           = 1;
    int sorted            = 1;
    int caxis             = 0;
    int k_val             = 0;
    bool built            = false;
    uint32_t cfg[4]       = {};

    // Cached BindGroup. Tensor-backed slots: [X, V, I].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[3] = {};

    bool init() override {
        if (!webgpu::device_ready()) return false;
        // ONNX: TopK has 2 outputs. For opset ≥ 10, inputs=[X, K]; ≤ 9, inputs=[X].
        if (outputs.size() != 2) return false;

        axis    = attribute(attr_key_t::axis,    (int32_t)-1);
        largest = attribute(attr_key_t::largest, (int32_t)1);
        sorted  = attribute(attr_key_t::sorted,  (int32_t)1);

        auto& dev = webgpu::get_device();
        wgpu::BindGroupLayoutEntry e[4] = {};
        e[0].binding = 0; e[0].visibility = wgpu::ShaderStage::Compute;
        e[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        e[1].binding = 1; e[1].visibility = wgpu::ShaderStage::Compute;
        e[1].buffer.type = wgpu::BufferBindingType::Storage;
        e[2].binding = 2; e[2].visibility = wgpu::ShaderStage::Compute;
        e[2].buffer.type = wgpu::BufferBindingType::Storage;
        e[3].binding = 3; e[3].visibility = wgpu::ShaderStage::Compute;
        e[3].buffer.type = wgpu::BufferBindingType::Uniform;
        wgpu::BindGroupLayoutDescriptor bgld = {};
        bgld.entryCount = 4; bgld.entries = e;
        bgl = dev.device.CreateBindGroupLayout(&bgld);

        // One workgroup per output slice; 64 threads cooperate.
        // Each iteration r in [0, K) finds the next-best element in
        // axis_dim, skipping the indices selected in earlier rounds.
        // Dedup uses a workgroup-shared bitmap (1 bit per axis position)
        // so the inner check is O(1) instead of O(r). Per-thread scan is
        // strided by 64; a parallel reduction across the workgroup picks
        // the global winner, which thread 0 writes out and marks in the
        // bitmap before the next iteration.
        //
        // Why one workgroup per slice (not one thread): the original
        // single-thread design did K * axis_dim * O(K) work serially —
        // K=200, axis=16000 hit the Windows TDR (~2 s GPU watchdog),
        // hanging the device. The parallel design caps per-thread work
        // at K * (axis_dim / 64 + 64), well under the watchdog.
        //
        // `sorted=1` gives ordered output by construction. `sorted=0`
        // allows any order, which the sorted path satisfies — so the
        // attribute is ignored. `largest` flips the comparison.
        //
        // Tie-break: when v == best_v, prefer the smaller index.
        const char* better =
            (largest ? "(v > b_v) || (v == b_v && ii < b_i)"
                     : "(v < b_v) || (v == b_v && ii < b_i)");
        const char* sentinel_val =
            (largest ? "-3.4e38"
                     : "3.4e38");
        const int axis_words = (AXIS_MAX + 31) / 32;

        std::string src;
        src  = "struct Cfg { slices : u32, axis_dim : u32, inner : u32, k : u32 };\n";
        src += "@group(0) @binding(0) var<storage, read>       X   : array<f32>;\n";
        src += "@group(0) @binding(1) var<storage, read_write> V   : array<f32>;\n";
        src += "@group(0) @binding(2) var<storage, read_write> I   : array<i32>;\n";
        src += "@group(0) @binding(3) var<uniform>             cfg : Cfg;\n";
        src += "var<workgroup> picked_bm : array<atomic<u32>, ";
        src += std::to_string(axis_words);
        src += ">;\n";
        src += "var<workgroup> red_v : array<f32, 64>;\n";
        src += "var<workgroup> red_i : array<i32, 64>;\n";
        src += "@compute @workgroup_size(64)\n";
        src += "fn main(@builtin(workgroup_id) wgid : vec3<u32>,\n";
        src += "        @builtin(local_invocation_id) lid : vec3<u32>) {\n";
        src += "  let slice = wgid.x;\n";
        src += "  if (slice >= cfg.slices) { return; }\n";
        src += "  let tid = lid.x;\n";
        src += "  let outer_idx = slice / cfg.inner;\n";
        src += "  let inner_idx = slice % cfg.inner;\n";
        src += "  let base_x = outer_idx * cfg.axis_dim * cfg.inner + inner_idx;\n";
        src += "  let base_y = outer_idx * cfg.k * cfg.inner + inner_idx;\n";
        // Clear the dedup bitmap. Only words covering [0, axis_dim) need to be
        // zeroed; the sentinel-value reduction handles unselected lanes.
        src += "  let bm_words = (cfg.axis_dim + 31u) / 32u;\n";
        src += "  for (var w : u32 = tid; w < bm_words; w = w + 64u) {\n";
        src += "    atomicStore(&picked_bm[w], 0u);\n";
        src += "  }\n";
        src += "  workgroupBarrier();\n";
        src += "  for (var r : u32 = 0u; r < cfg.k; r = r + 1u) {\n";
        src += "    var b_v : f32 = ";
        src += sentinel_val;
        src += ";\n";
        src += "    var b_i : i32 = -1;\n";
        // Strided per-thread scan; bitmap check is O(1).
        src += "    for (var a : u32 = tid; a < cfg.axis_dim; a = a + 64u) {\n";
        src += "      let bit = (atomicLoad(&picked_bm[a >> 5u]) >> (a & 31u)) & 1u;\n";
        src += "      if (bit != 0u) { continue; }\n";
        src += "      let v = X[base_x + a * cfg.inner];\n";
        src += "      let ii : i32 = i32(a);\n";
        src += "      if (b_i < 0 || ";
        src += better;
        src += ") { b_v = v; b_i = ii; }\n";
        src += "    }\n";
        src += "    red_v[tid] = b_v;\n";
        src += "    red_i[tid] = b_i;\n";
        src += "    workgroupBarrier();\n";
        // Tree reduction across 64 threads (5 levels: 32→16→8→4→2→1).
        src += "    for (var off : u32 = 32u; off > 0u; off = off >> 1u) {\n";
        src += "      if (tid < off) {\n";
        src += "        let ov = red_v[tid + off];\n";
        src += "        let oi = red_i[tid + off];\n";
        src += "        let cur_i = red_i[tid];\n";
        src += "        let cur_v = red_v[tid];\n";
        src += "        let take = (cur_i < 0) || (oi >= 0 && (";
        // Inline winner check: take the `off`-side if it beats the current side.
        // Tie-break on smaller index.
        src += (largest ? "(ov > cur_v) || (ov == cur_v && oi < cur_i)"
                        : "(ov < cur_v) || (ov == cur_v && oi < cur_i)");
        src += "));\n";
        src += "        if (take) { red_v[tid] = ov; red_i[tid] = oi; }\n";
        src += "      }\n";
        src += "      workgroupBarrier();\n";
        src += "    }\n";
        // Thread 0 commits the winner and marks the bitmap.
        src += "    if (tid == 0u) {\n";
        src += "      let win_i = red_i[0];\n";
        src += "      V[base_y + r * cfg.inner] = red_v[0];\n";
        src += "      I[base_y + r * cfg.inner] = win_i;\n";
        src += "      if (win_i >= 0) {\n";
        src += "        let wu = u32(win_i);\n";
        src += "        atomicOr(&picked_bm[wu >> 5u], 1u << (wu & 31u));\n";
        src += "      }\n";
        src += "    }\n";
        src += "    workgroupBarrier();\n";
        src += "  }\n";
        src += "}\n";

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

        wgpu::BufferDescriptor ud = {};
        ud.size = 16;
        ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        uniforms = dev.device.CreateBuffer(&ud);
        built = true;
        return true;
    }

    int read_k() {
        if (opset >= 10) {
            if (inputs.size() < 2 || !inputs[1] || !inputs[1]->data) return -1;
            const tensor_t* kt = inputs[1];
            // Accept INT64 (ONNX spec) and INT32 (permissive).
            if (kt->type == NNR_DATA_TYPE_INT64)
                return (int)(*(const int64_t*)kt->data);
            if (kt->type == NNR_DATA_TYPE_INT32)
                return (int)(*(const int32_t*)kt->data);
            return -1;
        }
        return (int)attribute(attr_key_t::k, (int64_t)0);
    }

    bool reshape() override {
        if (!built) return false;
        const tensor_t* x = inputs[0];
        tensor_t* val_out = outputs[0];
        tensor_t* idx_out = outputs[1];

        if (x->type != NNR_DATA_TYPE_FLOAT32) return false;

        k_val = read_k();
        if (k_val <= 0 || k_val > K_MAX) return false;

        caxis = axis;
        if (caxis < 0) caxis += x->ndim;
        if (caxis < 0 || caxis >= x->ndim) return false;
        if (x->dims[caxis] == 0 || k_val > x->dims[caxis]) return false;

        // Workgroup-shared dedup bitmap is sized at compile time. Reject
        // axes larger than the bitmap can address; would need a global
        // storage bitmap or a chunked algorithm to handle in-place.
        if (x->dims[caxis] > AXIS_MAX) return false;

        int dims[MAX_NDIM];
        for (int i = 0; i < x->ndim; ++i) dims[i] = x->dims[i];
        dims[caxis] = k_val;

        if (!val_out->reshape(std::span<const int>(dims, x->ndim), NNR_DATA_TYPE_FLOAT32)) return false;
        if (!idx_out->reshape(std::span<const int>(dims, x->ndim), NNR_DATA_TYPE_INT64))   return false;

        uint32_t outer = 1, inner = 1;
        for (int i = 0;         i < caxis;    ++i) outer *= (uint32_t)x->dims[i];
        for (int i = caxis + 1; i < x->ndim;  ++i) inner *= (uint32_t)x->dims[i];

        cfg[0] = outer * inner;                   // total slices
        cfg[1] = (uint32_t)x->dims[caxis];        // axis dim
        cfg[2] = inner;                           // inner stride
        cfg[3] = (uint32_t)k_val;

        webgpu::ensure_buffer(x,       x->ndata * sizeof(float));
        webgpu::ensure_buffer(val_out, val_out->ndata * sizeof(float));
        webgpu::ensure_buffer(idx_out, idx_out->ndata * sizeof(int32_t));

        webgpu::get_device().queue.WriteBuffer(uniforms, 0, cfg, sizeof(cfg));
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();
        webgpu::upload_if_needed(inputs[0]);

        auto* rx = webgpu::find(inputs[0]);
        auto* rv = webgpu::find(outputs[0]);
        auto* ri = webgpu::find(outputs[1]);

        uint32_t gen_x = webgpu::generation_of(inputs[0]);
        uint32_t gen_v = webgpu::generation_of(outputs[0]);
        uint32_t gen_i = webgpu::generation_of(outputs[1]);
        if (!cached_bg || gen_x != cached_gen[0] || gen_v != cached_gen[1] || gen_i != cached_gen[2]) {
            wgpu::BindGroupEntry be[4] = {};
            be[0].binding = 0; be[0].buffer = rx->buf;   be[0].offset = 0; be[0].size = rx->size;
            be[1].binding = 1; be[1].buffer = rv->buf;   be[1].offset = 0; be[1].size = rv->size;
            be[2].binding = 2; be[2].buffer = ri->buf;   be[2].offset = 0; be[2].size = ri->size;
            be[3].binding = 3; be[3].buffer = uniforms;  be[3].offset = 0; be[3].size = 16;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 4; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_x;
            cached_gen[1] = gen_v;
            cached_gen[2] = gen_i;
        }

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, cached_bg);
        // One workgroup per slice — each workgroup's 64 threads collaborate
        // on a single TopK reduction.
        pass.DispatchWorkgroups(cfg[0], 1, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        webgpu::mark_gpu_written(outputs[1]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_TopK_webgpu(int, pool_t& pool) {
    return pool_new<TopK_op_webgpu>(pool);
}

} // namespace nnr
