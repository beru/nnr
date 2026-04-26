// WebGPU CumSum — prefix sum along one axis.
//
// ONNX: Y[..., i, ...] = sum over j ∈ [0..i] of X[..., j, ...] along the
// named axis. Two flags:
//   - exclusive: if 1, sum up to but not including i.
//   - reverse:   if 1, sum from i to end instead of 0 to i.
//
// Scope: f32 input only. axis input (int32/int64 scalar) must be
// CPU-resident at reshape-time. Naive O(N * axis_dim) kernel — one thread
// per output element iterates along the axis. A parallel prefix-scan would
// be strictly faster but this is correctness first; the perf pass
// mentioned in the continuation doc covers this class of op.

#include "nnr.h"
#include "attr_key.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include <cstring>
#include <webgpu/webgpu_cpp.h>

namespace nnr {

namespace {

constexpr uint32_t WG = 256;

struct CumSum_operator_webgpu : public operator_t {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          uniforms;  // 32B: {total, outer, axis_dim, inner, exclusive, reverse, 0, 0}
    int exclusive = 0;
    int reverse   = 0;

    // Cached BindGroup. Tensor-backed slots: [X, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[2] = {};

    // One thread per output element. Decompose output flat index i into
    // (outer_idx, axis_idx, inner_idx), then sum the appropriate axis slice.
    static constexpr const char* kWgsl =
        "struct Cfg {\n"
        "  total     : u32,\n"
        "  outer     : u32,\n"
        "  axis_dim  : u32,\n"
        "  inner     : u32,\n"
        "  exclusive : u32,\n"
        "  reverse   : u32,\n"
        "  _a        : u32,\n"
        "  _b        : u32,\n"
        "};\n"
        "@group(0) @binding(0) var<storage, read>       X   : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> Y   : array<f32>;\n"
        "@group(0) @binding(2) var<uniform>             cfg : Cfg;\n"
        "@compute @workgroup_size(256)\n"
        "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
        "  let i = gid.x;\n"
        "  if (i >= cfg.total) { return; }\n"
        "  // i = ((outer_idx * axis_dim) + axis_idx) * inner + inner_idx\n"
        "  let inner_idx = i % cfg.inner;\n"
        "  var tmp = i / cfg.inner;\n"
        "  let axis_idx  = tmp % cfg.axis_dim;\n"
        "  let outer_idx = tmp / cfg.axis_dim;\n"
        "  let base = outer_idx * cfg.axis_dim * cfg.inner + inner_idx;\n"
        "  var acc : f32 = 0.0;\n"
        "  if (cfg.reverse == 0u) {\n"
        "    let end = select(axis_idx + 1u, axis_idx, cfg.exclusive != 0u);\n"
        "    for (var k : u32 = 0u; k < end; k = k + 1u) {\n"
        "      acc = acc + X[base + k * cfg.inner];\n"
        "    }\n"
        "  } else {\n"
        "    let start = select(axis_idx, axis_idx + 1u, cfg.exclusive != 0u);\n"
        "    for (var k : u32 = start; k < cfg.axis_dim; k = k + 1u) {\n"
        "      acc = acc + X[base + k * cfg.inner];\n"
        "    }\n"
        "  }\n"
        "  Y[i] = acc;\n"
        "}\n";

    bool init() override {
        if (!is_inout_size(2, 1)) return false;
        if (!webgpu::device_ready()) return false;
        exclusive = attribute(attr_key_t::exclusive, (int32_t)0);
        reverse   = attribute(attr_key_t::reverse,   (int32_t)0);

        auto& dev = webgpu::get_device();
        wgpu::ShaderSourceWGSL src = {};
        src.code = kWgsl;
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &src;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

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

        wgpu::PipelineLayoutDescriptor pld = {};
        pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = &bgl;
        wgpu::PipelineLayout pl = dev.device.CreatePipelineLayout(&pld);

        wgpu::ComputePipelineDescriptor cpd = {};
        cpd.layout = pl;
        cpd.compute.module = sm;
        cpd.compute.entryPoint = "main";
        pipeline = dev.device.CreateComputePipeline(&cpd);

        wgpu::BufferDescriptor ud = {};
        ud.size = 32;
        ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        uniforms = dev.device.CreateBuffer(&ud);
        return true;
    }

    bool reshape() override {
        const tensor_t* x    = inputs[0];
        const tensor_t* axt  = inputs[1];
        tensor_t*       y    = outputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (!axt->data || axt->ndata == 0) return false;

        int axis;
        if (axt->type == NNR_DATA_TYPE_INT32)      axis = *(const int32_t*)axt->data;
        else if (axt->type == NNR_DATA_TYPE_INT64) axis = (int)*(const int64_t*)axt->data;
        else return false;

        if (axis < 0) axis += x->ndim;
        if (axis < 0 || axis >= x->ndim) return false;

        if (!y->reshape_identity(x)) return false;

        uint32_t outer = 1;
        for (int i = 0; i < axis; ++i) outer *= (uint32_t)x->dims[i];
        uint32_t axis_dim = (uint32_t)x->dims[axis];
        uint32_t inner = 1;
        for (int i = axis + 1; i < x->ndim; ++i) inner *= (uint32_t)x->dims[i];

        cfg_vals[0] = (uint32_t)x->ndata;
        cfg_vals[1] = outer;
        cfg_vals[2] = axis_dim;
        cfg_vals[3] = inner;
        cfg_vals[4] = (uint32_t)exclusive;
        cfg_vals[5] = (uint32_t)reverse;
        cfg_vals[6] = 0;
        cfg_vals[7] = 0;

        webgpu::ensure_buffer(x, x->ndata * sizeof(float));
        webgpu::ensure_buffer(y, y->ndata * sizeof(float));

        webgpu::get_device().queue.WriteBuffer(uniforms, 0, cfg_vals, sizeof(cfg_vals));
        return true;
    }

    uint32_t cfg_vals[8] = {};

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
            be[2].binding = 2; be[2].buffer = uniforms;  be[2].offset = 0; be[2].size = 32;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 3; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_x;
            cached_gen[1] = gen_y;
        }

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, cached_bg);
        uint32_t groups = (cfg_vals[0] + WG - 1) / WG;
        pass.DispatchWorkgroups(groups, 1, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_CumSum_webgpu(int, pool_t& pool) {
    return pool_new<CumSum_operator_webgpu>(pool);
}

} // namespace nnr
