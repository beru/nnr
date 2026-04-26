// WebGPU ArgMax / ArgMin — reduce-to-index along one axis.
//
// ONNX mandates int64 output. WebGPU storage buffers can't hold i64, so we
// store i32 — safe for any realistic tensor (no single axis exceeds 2^31
// indices). Cross-backend hazard: a CPU op downstream that reads this
// output expects int64 and will read garbage from the upper half. Same
// tradeoff as comparison ops emitting u32 vs BOOL; the broader bool/i64
// upload story in the continuation doc covers this. For pure-GPU chains
// (ArgMax → CPU-side `search_tensor` reading the int32 output as an index
// array) this works fine.
//
// Attributes: axis (int), keepdims (default 1), select_last_index
// (default 0). select_last_index=1 ties-break toward the highest index;
// the default breaks toward the lowest.
//
// Scope: f32 input only. Integer/half dtypes → CPU fallback.

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

struct ArgReduce_op_webgpu : public operator_t {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          uniforms;
    int axis              = 0;
    int keepdims          = 1;
    int select_last_index = 0;
    bool is_max           = true;
    bool built            = false;
    uint32_t cfg[4]       = {};

    // Cached BindGroup. Tensor-backed slots: [X, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[2] = {};

    ArgReduce_op_webgpu(bool max_variant) : is_max(max_variant) {}

    bool init() override {
        if (!is_inout_size(1, 1)) return false;
        if (!webgpu::device_ready()) return false;
        axis              = attribute(attr_key_t::axis, (int32_t)0);
        keepdims          = attribute(attr_key_t::keepdims, (int32_t)1);
        select_last_index = attribute(attr_key_t::select_last_index, (int32_t)0);

        auto& dev = webgpu::get_device();
        wgpu::BindGroupLayoutEntry e[3] = {};
        e[0].binding = 0; e[0].visibility = wgpu::ShaderStage::Compute;
        e[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        e[1].binding = 1; e[1].visibility = wgpu::ShaderStage::Compute;
        e[1].buffer.type = wgpu::BufferBindingType::Storage;
        e[2].binding = 2; e[2].visibility = wgpu::ShaderStage::Compute;
        e[2].buffer.type = wgpu::BufferBindingType::Uniform;
        wgpu::BindGroupLayoutDescriptor bgld = {};
        bgld.entryCount = 3; bgld.entries = e;
        bgl = dev.device.CreateBindGroupLayout(&bgld);

        // Runtime-compile one of four shaders ({max, min} × {first, last}).
        // Using an overridden compare op and an initial sentinel chosen to
        // guarantee index 0 is always the "starting best" (first scan iter
        // either keeps it or not).
        std::string src =
            "struct Cfg { total : u32, dim : u32, stride : u32, select_last : u32 };\n"
            "@group(0) @binding(0) var<storage, read>       X   : array<f32>;\n"
            "@group(0) @binding(1) var<storage, read_write> Y   : array<i32>;\n"
            "@group(0) @binding(2) var<uniform>             cfg : Cfg;\n"
            "@compute @workgroup_size(256)\n"
            "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
            "  let i = gid.x;\n"
            "  if (i >= cfg.total) { return; }\n"
            "  let inner_idx = i % cfg.stride;\n"
            "  let outer_idx = i / cfg.stride;\n"
            "  let base = outer_idx * cfg.dim * cfg.stride + inner_idx;\n"
            "  var best_v : f32 = X[base];\n"
            "  var best_i : i32 = 0;\n"
            "  for (var k : u32 = 1u; k < cfg.dim; k = k + 1u) {\n"
            "    let v = X[base + k * cfg.stride];\n"
            "    var update : bool;\n";
        // Four combinations of comparison: IsMax × select_last.
        if (is_max) {
            src += "    if (cfg.select_last != 0u) { update = v >= best_v; } else { update = v > best_v; }\n";
        } else {
            src += "    if (cfg.select_last != 0u) { update = v <= best_v; } else { update = v < best_v; }\n";
        }
        src += "    if (update) { best_v = v; best_i = i32(k); }\n"
               "  }\n"
               "  Y[i] = best_i;\n"
               "}\n";

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

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t*       y = outputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return false;
        int a = axis;
        if (a < 0) a += x->ndim;
        if (a < 0 || a >= x->ndim) return false;
        if (x->dims[a] == 0) return false;

        int dims[8] = {};
        int out_ndim = 0;
        if (keepdims) {
            out_ndim = x->ndim;
            for (int i = 0; i < x->ndim; ++i) dims[i] = (i == a) ? 1 : x->dims[i];
        } else {
            for (int i = 0; i < x->ndim; ++i) if (i != a) dims[out_ndim++] = x->dims[i];
        }
        // ONNX ArgMax/ArgMin outputs are INT64. The GPU shader emits i32
        // values, but buffer.cpp narrows int64 ↔ i32 at upload/download
        // time, so the declared tensor type matches what downstream CPU
        // consumers expect. Values > 2^31 would corrupt — rare for index
        // use cases. (See "cross-backend i64/bool hazard" in the doc.)
        if (!y->reshape(std::span<const int>(dims, out_ndim), NNR_DATA_TYPE_INT64)) return false;

        uint32_t stride = 1;
        for (int i = a + 1; i < x->ndim; ++i) stride *= (uint32_t)x->dims[i];
        cfg[0] = (uint32_t)y->ndata;
        cfg[1] = (uint32_t)x->dims[a];
        cfg[2] = stride;
        cfg[3] = (uint32_t)select_last_index;

        webgpu::ensure_buffer(x, x->ndata * sizeof(float));
        webgpu::ensure_buffer(y, y->ndata * sizeof(int32_t));

        webgpu::get_device().queue.WriteBuffer(uniforms, 0, cfg, sizeof(cfg));
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
            be[2].binding = 2; be[2].buffer = uniforms;  be[2].offset = 0; be[2].size = 16;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 3; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_x;
            cached_gen[1] = gen_y;
        }

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, cached_bg);
        uint32_t groups = (cfg[0] + WG - 1) / WG;
        pass.DispatchWorkgroups(groups, 1, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

struct ArgMax_op_webgpu : public ArgReduce_op_webgpu { ArgMax_op_webgpu() : ArgReduce_op_webgpu(true)  {} };
struct ArgMin_op_webgpu : public ArgReduce_op_webgpu { ArgMin_op_webgpu() : ArgReduce_op_webgpu(false) {} };

} // namespace

operator_t* resolver_default_op_ArgMax_webgpu(int, pool_t& pool) {
    return pool_new<ArgMax_op_webgpu>(pool);
}
operator_t* resolver_default_op_ArgMin_webgpu(int, pool_t& pool) {
    return pool_new<ArgMin_op_webgpu>(pool);
}

} // namespace nnr
