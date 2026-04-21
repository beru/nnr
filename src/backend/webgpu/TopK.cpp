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
// Cap K at 1024 to keep the local `picked` array's footprint reasonable
// (4 KB of private memory per thread). Large enough for yolov10's K=300
// and similar detector topologies; K > 1024 still falls back to CPU.
constexpr int K_MAX = 1024;

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

        // One shader per (largest × sorted) combination. With `sorted=1`
        // outputs are ordered by value; with `sorted=0` ONNX allows any
        // ordering, but producing sorted output is strictly a superset
        // and the simplest implementation happens to always produce
        // sorted results — so we just ignore `sorted` in the shader.
        // The `largest` flag flips the compare direction.
        //
        // `picked` is a local array tracking indices already chosen in
        // prior passes. Bounded K_MAX=256; reshape() rejects larger K.
        //
        // Tie-break: when `v == best_v`, prefer the smaller index.
        const char* compare_new_beats_best =
            (largest ? "v > best_v || (v == best_v && ii < best_i)"
                     : "v < best_v || (v == best_v && ii < best_i)");
        const char* sentinel_val =
            (largest ? "-3.4e38"   // big negative; avoid edge fp literal
                     : "3.4e38");

        std::string src;
        src  = "struct Cfg { slices : u32, axis_dim : u32, inner : u32, k : u32 };\n";
        src += "@group(0) @binding(0) var<storage, read>       X   : array<f32>;\n";
        src += "@group(0) @binding(1) var<storage, read_write> V   : array<f32>;\n";
        src += "@group(0) @binding(2) var<storage, read_write> I   : array<i32>;\n";
        src += "@group(0) @binding(3) var<uniform>             cfg : Cfg;\n";
        src += "@compute @workgroup_size(64)\n";
        src += "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n";
        src += "  let slice = gid.x;\n";
        src += "  if (slice >= cfg.slices) { return; }\n";
        src += "  let outer_idx = slice / cfg.inner;\n";
        src += "  let inner_idx = slice % cfg.inner;\n";
        src += "  let base_x = outer_idx * cfg.axis_dim * cfg.inner + inner_idx;\n";
        src += "  let base_y = outer_idx * cfg.k * cfg.inner + inner_idx;\n";
        src += "  var picked : array<i32, ";
        src += std::to_string(K_MAX);
        src += ">;\n";
        src += "  for (var r : u32 = 0u; r < cfg.k; r = r + 1u) {\n";
        src += "    var best_v : f32 = ";
        src += sentinel_val;
        src += ";\n";
        src += "    var best_i : i32 = -1;\n";
        src += "    for (var a : u32 = 0u; a < cfg.axis_dim; a = a + 1u) {\n";
        src += "      let v = X[base_x + a * cfg.inner];\n";
        src += "      let ii : i32 = i32(a);\n";
        // Skip already-picked indices (linear scan of picked[0..r-1]).
        src += "      var already : bool = false;\n";
        src += "      for (var p : u32 = 0u; p < r; p = p + 1u) {\n";
        src += "        if (picked[p] == ii) { already = true; break; }\n";
        src += "      }\n";
        src += "      if (already) { continue; }\n";
        src += "      if (best_i < 0 || (";
        src += compare_new_beats_best;
        src += ")) { best_v = v; best_i = ii; }\n";
        src += "    }\n";
        src += "    picked[r] = best_i;\n";
        src += "    V[base_y + r * cfg.inner] = best_v;\n";
        src += "    I[base_y + r * cfg.inner] = best_i;\n";
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
        uint32_t groups = (cfg[0] + WG - 1) / WG;
        pass.DispatchWorkgroups(groups, 1, 1);
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
