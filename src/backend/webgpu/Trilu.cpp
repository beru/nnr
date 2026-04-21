// WebGPU Trilu — keep only the upper (upper=1) or lower (upper=0) triangle
// of each 2D matrix in the last two axes of X; zero everything else.
// Optional second input `k` shifts the diagonal (scalar int64, CPU-resident).
//
// ONNX: output = where(keep, input, 0). Upper: keep where j >= i + k.
// Lower: keep where j <= i + k. i and j are the last two indices (row, col);
// leading axes are treated as independent matrix batches.
//
// Scope: f32 only for now. Bool/int Trilu falls back to CPU — the same
// pattern a bool-upload extension would complete later. Rank ≥ 2 required.

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

struct Trilu_operator_webgpu : public operator_t {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          uniforms;    // 16B: {total, rows, cols, k_plus_upper_bit}
    int upper = 1;
    uint32_t total = 0;

    // Cached BindGroup. Tensor-backed slots: [X, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[2] = {};

    static constexpr const char* kWgsl =
        "struct Cfg { total : u32, rows : u32, cols : u32, kflags : u32 };\n"
        "@group(0) @binding(0) var<storage, read>       X   : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> Y   : array<f32>;\n"
        "@group(0) @binding(2) var<uniform>             cfg : Cfg;\n"
        "@compute @workgroup_size(256)\n"
        "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
        "  let i = gid.x;\n"
        "  if (i >= cfg.total) { return; }\n"
        "  // Reinterpret kflags: high bit = upper flag, low 31 bits = (k + 2^30) so negative k survives u32.\n"
        "  let upper  = (cfg.kflags >> 31u) & 1u;\n"
        "  let k      = i32(cfg.kflags & 0x7fffffffu) - (1 << 30);\n"
        "  let mat    = cfg.rows * cfg.cols;\n"
        "  let idx_in_mat = i % mat;\n"
        "  let row = i32(idx_in_mat / cfg.cols);\n"
        "  let col = i32(idx_in_mat % cfg.cols);\n"
        "  var keep : bool;\n"
        "  if (upper != 0u) { keep = col >= row + k; }\n"
        "  else             { keep = col <= row + k; }\n"
        "  Y[i] = select(0.0, X[i], keep);\n"
        "}\n";

    bool init() override {
        if (outputs.size() != 1) return false;
        if (inputs.size() < 1 || inputs.size() > 2) return false;
        if (!webgpu::device_ready()) return false;
        upper = attribute(attr_key_t::upper, (int32_t)1);

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
        ud.size = 16;
        ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        uniforms = dev.device.CreateBuffer(&ud);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t*       y = outputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (x->ndim < 2) return false;
        if (!y->reshape_identity(x)) return false;
        webgpu::ensure_buffer(x, x->ndata * sizeof(float));
        webgpu::ensure_buffer(y, y->ndata * sizeof(float));

        total = (uint32_t)x->ndata;
        uint32_t rows  = (uint32_t)x->dims[x->ndim - 2];
        uint32_t cols  = (uint32_t)x->dims[x->ndim - 1];
        int64_t k = 0;
        if (inputs.size() > 1 && inputs[1] && inputs[1]->data && inputs[1]->ndata > 0) {
            k = *(const int64_t*)inputs[1]->data;
        }
        // Pack k (with a 2^30 bias so negative values survive u32) and the
        // upper flag (high bit) into one u32. Range ±(2^30-1), ample for
        // any realistic model.
        int64_t k_biased = k + (int64_t)(1 << 30);
        if (k_biased < 0 || k_biased >= (int64_t)0x7fffffff) return false;
        uint32_t kflags = (uint32_t)k_biased | (upper ? 0x80000000u : 0u);

        uint32_t u[4] = { total, rows, cols, kflags };
        webgpu::get_device().queue.WriteBuffer(uniforms, 0, u, sizeof(u));
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
        uint32_t groups = (total + WG - 1) / WG;
        pass.DispatchWorkgroups(groups, 1, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_Trilu_webgpu(int, pool_t& pool) {
    return pool_new<Trilu_operator_webgpu>(pool);
}

} // namespace nnr
