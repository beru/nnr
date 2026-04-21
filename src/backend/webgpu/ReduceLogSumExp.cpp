// WebGPU ReduceLogSumExp — numerically stable log-sum-exp reduction.
//
// The one-pass reduction pattern used by reduce_elementwise_t can't represent
// LogSumExp faithfully: naive `log(sum(exp(v)))` overflows for large `v`, and
// a running-rescale online form ties two accumulators together in a way the
// single-f32 `acc` template can't express. So this op defines its own shader
// with two sequential loops over the reduce domain inside the same kernel
// invocation — pass 1 tracks the row max, pass 2 accumulates exp(v - max).
// The final result is `log(sum) + max`, which is numerically stable.
//
// Reshape/exec are unchanged from `reduce_elementwise_t`: the meta buffer
// layout (in_dims / in_strides / is_reduce tables) is identical, so only
// init() needs to swap in a bespoke shader. The `*_expr()` virtuals return
// unused placeholders.

#include "reduce.h"

#include "device.h"
#include "pool.h"

#include <string>
#include <webgpu/webgpu_cpp.h>

namespace nnr {

namespace {

std::string make_logsumexp_wgsl() {
    // Meta layout matches `reduce_elementwise_t`'s. Two loops over the
    // reduce domain produce (max, sum-of-exp); the final write uses the
    // log-sum-exp identity `log(sum(exp(x))) == log(sum(exp(x - M))) + M`.
    return
        "struct Meta {\n"
        "  total         : u32,\n"
        "  ndim          : u32,\n"
        "  red_count     : u32,\n"
        "  _pad          : u32,\n"
        "  in_dims_lo    : vec4<u32>,\n"
        "  in_dims_hi    : vec4<u32>,\n"
        "  in_strides_lo : vec4<u32>,\n"
        "  in_strides_hi : vec4<u32>,\n"
        "  is_reduce_lo  : vec4<u32>,\n"
        "  is_reduce_hi  : vec4<u32>,\n"
        "};\n"
        "@group(0) @binding(0) var<storage, read>       X  : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> Y  : array<f32>;\n"
        "@group(0) @binding(2) var<storage, read>       md : Meta;\n"
        "fn get_in_dim(i : u32)    -> u32 { if (i < 4u) { return md.in_dims_lo[i]; }    return md.in_dims_hi[i - 4u]; }\n"
        "fn get_in_stride(i : u32) -> u32 { if (i < 4u) { return md.in_strides_lo[i]; } return md.in_strides_hi[i - 4u]; }\n"
        "fn get_is_reduce(i : u32) -> u32 { if (i < 4u) { return md.is_reduce_lo[i]; }  return md.is_reduce_hi[i - 4u]; }\n"
        "fn reduce_flat(base : u32, r : u32) -> u32 {\n"
        "  var rtmp   : u32 = r;\n"
        "  var x_flat : u32 = base;\n"
        "  for (var k : i32 = i32(md.ndim) - 1; k >= 0; k = k - 1) {\n"
        "    if (get_is_reduce(u32(k)) == 0u) { continue; }\n"
        "    let d = get_in_dim(u32(k));\n"
        "    let c = rtmp % d;\n"
        "    rtmp  = rtmp / d;\n"
        "    x_flat = x_flat + c * get_in_stride(u32(k));\n"
        "  }\n"
        "  return x_flat;\n"
        "}\n"
        "@compute @workgroup_size(64)\n"
        "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
        "  let o = gid.x;\n"
        "  if (o >= md.total) { return; }\n"
        "  // Base input flat index from the keep-axes of the output coord.\n"
        "  var base : u32 = 0u;\n"
        "  var tmp  : u32 = o;\n"
        "  for (var k : i32 = i32(md.ndim) - 1; k >= 0; k = k - 1) {\n"
        "    if (get_is_reduce(u32(k)) != 0u) { continue; }\n"
        "    let d = get_in_dim(u32(k));\n"
        "    let c = tmp % d;\n"
        "    tmp   = tmp / d;\n"
        "    base  = base + c * get_in_stride(u32(k));\n"
        "  }\n"
        "  // Pass 1: row max.\n"
        "  var row_max : f32 = -3.4e38;\n"
        "  for (var r : u32 = 0u; r < md.red_count; r = r + 1u) {\n"
        "    row_max = max(row_max, X[reduce_flat(base, r)]);\n"
        "  }\n"
        "  // Pass 2: sum of exp(x - row_max).\n"
        "  var row_sum : f32 = 0.0;\n"
        "  for (var r : u32 = 0u; r < md.red_count; r = r + 1u) {\n"
        "    row_sum = row_sum + exp(X[reduce_flat(base, r)] - row_max);\n"
        "  }\n"
        "  Y[o] = log(row_sum) + row_max;\n"
        "}\n";
}

struct ReduceLogSumExp_op_webgpu : webgpu::reduce_elementwise_t {
    // Bespoke kernel — these overrides are never read by the base class
    // since init() below builds its own shader + pipeline. Return trivial
    // strings so the base-class virtuals have concrete implementations.
    const char* init_expr()      const override { return "0.0"; }
    const char* transform_expr() const override { return "v"; }
    const char* merge_expr()     const override { return "a + b"; }
    const char* finalize_expr()  const override { return "acc"; }

    bool init() override {
        if (outputs.size() != 1) return false;
        if (inputs.size() < 1 || inputs.size() > 2) return false;
        if (!webgpu::device_ready()) return false;

        auto& dev = webgpu::get_device();
        std::string src = make_logsumexp_wgsl();
        wgpu::ShaderSourceWGSL w = {};
        w.code = src.c_str();
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
        md.size  = 128;
        md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        meta_buf = dev.device.CreateBuffer(&md);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_ReduceLogSumExp_webgpu(int, pool_t& pool) {
    return pool_new<ReduceLogSumExp_op_webgpu>(pool);
}

} // namespace nnr
