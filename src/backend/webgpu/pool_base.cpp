#include "pool_base.h"

#include "device.h"
#include "buffer.h"
#include "attr_key.h"

#include <cstring>
#include <string>
#include <string_view>

namespace nnr::webgpu {

namespace {

constexpr uint32_t WG = 64;

std::string make_pool_wgsl(const char* init, const char* combine, const char* finalize) {
    std::string s =
        "struct Meta {\n"
        "  total             : u32,\n"
        "  N                 : u32, C : u32, H_out : u32, W_out : u32,\n"
        "  kH                : u32, kW : u32,\n"
        "  stride_h          : u32, stride_w : u32,\n"
        "  pad_top           : u32, pad_left : u32,\n"
        "  dilation_h        : u32, dilation_w : u32,\n"
        "  H                 : u32, W : u32,\n"
        "  count_include_pad : u32,\n"
        "  grid_stride_x     : u32,\n"   // threads along x for 2D dispatch split
        "};\n"
        "@group(0) @binding(0) var<storage, read>       X  : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> Y  : array<f32>;\n"
        "@group(0) @binding(2) var<storage, read>       md : Meta;\n"
        "@compute @workgroup_size(64)\n"
        "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
        "  let i = gid.y * md.grid_stride_x + gid.x;\n"
        "  if (i >= md.total) { return; }\n"
        "  let ow = i % md.W_out;\n"
        "  var tmp = i / md.W_out;\n"
        "  let oh = tmp % md.H_out;\n"
        "  tmp    = tmp / md.H_out;\n"
        "  let c  = tmp % md.C;\n"
        "  let n  = tmp / md.C;\n"
        "  var acc : f32 = ";
    s += init;
    s += ";\n"
        "  var cnt : u32 = 0u;\n"
        "  let base_x = (n * md.C + c) * md.H * md.W;\n"
        "  for (var kh : u32 = 0u; kh < md.kH; kh = kh + 1u) {\n"
        "    let ih_i = i32(oh * md.stride_h) + i32(kh * md.dilation_h) - i32(md.pad_top);\n"
        "    if (ih_i < 0 || ih_i >= i32(md.H)) { continue; }\n"
        "    let ih = u32(ih_i);\n"
        "    for (var kw : u32 = 0u; kw < md.kW; kw = kw + 1u) {\n"
        "      let iw_i = i32(ow * md.stride_w) + i32(kw * md.dilation_w) - i32(md.pad_left);\n"
        "      if (iw_i < 0 || iw_i >= i32(md.W)) { continue; }\n"
        "      let iw = u32(iw_i);\n"
        "      let v : f32 = X[base_x + ih * md.W + iw];\n"
        "      acc = ";
    s += combine;
    s += ";\n"
        "      cnt = cnt + 1u;\n"
        "    }\n"
        "  }\n"
        "  let full = md.kH * md.kW;\n"
        "  let n_div : u32 = select(cnt, full, md.count_include_pad != 0u);\n"
        "  Y[i] = ";
    s += finalize;
    s += ";\n}\n";
    return s;
}

} // namespace

bool pool_elementwise_t::init() {
    if (inputs.size() != 1) return false;
    if (outputs.size() != 1) return false;   // MaxPool's optional indices output is unsupported
    if (!device_ready()) return false;

    auto& dev = get_device();
    std::string src = make_pool_wgsl(init_expr(), combine_expr(), finalize_expr());
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

bool pool_elementwise_t::reshape() {
    const tensor_t* X = inputs[0];
    tensor_t*       Y = outputs[0];
    if (X->type != NNR_DATA_TYPE_FLOAT32) return false;
    if (X->ndim != 4) return false;          // 2D pooling only for now

    int N = X->dims[0], C = X->dims[1], H = X->dims[2], W = X->dims[3];

    std::string_view auto_pad = attribute(attr_key_t::auto_pad, "NOTSET");
    if (auto_pad != "NOTSET" && auto_pad != "VALID"
        && auto_pad != "SAME_UPPER" && auto_pad != "SAME_LOWER") return false;
    if ((int)attribute(attr_key_t::ceil_mode, (int64_t)0) != 0) return false;

    int64_t* ints = nullptr;
    int nk = attribute(attr_key_t::kernel_shape, ints);
    if (nk != 2) return false;   // require 2D kernel for 4D input
    int kH = (int)ints[0], kW = (int)ints[1];

    int ns = attribute(attr_key_t::strides, ints);
    int s_h = ns >= 1 ? (int)ints[0] : 1;
    int s_w = ns >= 2 ? (int)ints[1] : s_h;

    int nd = attribute(attr_key_t::dilations, ints);
    int d_h = nd >= 1 ? (int)ints[0] : 1;
    int d_w = nd >= 2 ? (int)ints[1] : d_h;

    int np = attribute(attr_key_t::pads, ints);
    int p_t = 0, p_l = 0, p_b = 0, p_r = 0;
    if (np >= 4) { p_t = (int)ints[0]; p_l = (int)ints[1]; p_b = (int)ints[2]; p_r = (int)ints[3]; }
    if (auto_pad == "VALID") { p_t = p_l = p_b = p_r = 0; }

    int H_out, W_out;
    if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
        // Mirror Conv's SAME_* back-solve so output = ceil(input / stride).
        H_out = (H + s_h - 1) / s_h;
        W_out = (W + s_w - 1) / s_w;
        int total_pad_h = (H_out - 1) * s_h + d_h * (kH - 1) + 1 - H;
        int total_pad_w = (W_out - 1) * s_w + d_w * (kW - 1) + 1 - W;
        if (total_pad_h < 0) total_pad_h = 0;
        if (total_pad_w < 0) total_pad_w = 0;
        if (auto_pad == "SAME_UPPER") {
            p_t = total_pad_h / 2; p_b = total_pad_h - p_t;
            p_l = total_pad_w / 2; p_r = total_pad_w - p_l;
        } else {
            p_b = total_pad_h / 2; p_t = total_pad_h - p_b;
            p_r = total_pad_w / 2; p_l = total_pad_w - p_r;
        }
    } else {
        if (p_t < 0 || p_l < 0 || p_b < 0 || p_r < 0) return false;
        H_out = (H + p_t + p_b - d_h * (kH - 1) - 1) / s_h + 1;
        W_out = (W + p_l + p_r - d_w * (kW - 1) - 1) / s_w + 1;
    }
    if (H_out <= 0 || W_out <= 0) return false;

    int out_dims[4] = { N, C, H_out, W_out };
    if (!Y->reshape(std::span<const int>(out_dims, 4), X->type)) return false;

    int count_include_pad = (int)attribute(attr_key_t::count_include_pad, (int64_t)0);

    total_u = (uint32_t)(N * C * H_out * W_out);
    meta_vals[0]  = total_u;
    meta_vals[1]  = (uint32_t)N;
    meta_vals[2]  = (uint32_t)C;
    meta_vals[3]  = (uint32_t)H_out;
    meta_vals[4]  = (uint32_t)W_out;
    meta_vals[5]  = (uint32_t)kH;
    meta_vals[6]  = (uint32_t)kW;
    meta_vals[7]  = (uint32_t)s_h;
    meta_vals[8]  = (uint32_t)s_w;
    meta_vals[9]  = (uint32_t)p_t;
    meta_vals[10] = (uint32_t)p_l;
    meta_vals[11] = (uint32_t)d_h;
    meta_vals[12] = (uint32_t)d_w;
    meta_vals[13] = (uint32_t)H;
    meta_vals[14] = (uint32_t)W;
    meta_vals[15] = (uint32_t)count_include_pad;

    ensure_buffer(X, X->ndata * sizeof(float));
    ensure_buffer(Y, Y->ndata * sizeof(float));
    finalize_meta_for_dispatch();
    return true;
}

void pool_elementwise_t::finalize_meta_for_dispatch() {
    // Dispatch grid + grid_stride_x are pure functions of total_u.
    // Compute once and bake the meta payload here so exec() is a pure
    // dispatch path. Subclasses that override reshape() must call this
    // themselves after populating meta_vals[0..15] and total_u.
    uint32_t groups = (total_u + WG - 1) / WG;
    dispatch_1d_grid(groups, dispatch_gx, dispatch_gy);
    meta_vals[16] = dispatch_gx * WG;
    get_device().queue.WriteBuffer(meta_buf, 0, meta_vals, sizeof(meta_vals));
}

bool pool_elementwise_t::exec() {
    auto& dev = get_device();
    upload_if_needed(inputs[0]);

    auto* rx = find(inputs[0]);
    auto* ry = find(outputs[0]);

    uint32_t gen_x = generation_of(inputs[0]);
    uint32_t gen_y = generation_of(outputs[0]);
    if (!cached_bg || gen_x != cached_gen[0] || gen_y != cached_gen[1]) {
        wgpu::BindGroupEntry be[3] = {};
        be[0].binding = 0; be[0].buffer = rx->buf;   be[0].offset = 0; be[0].size = rx->size;
        be[1].binding = 1; be[1].buffer = ry->buf;   be[1].offset = 0; be[1].size = ry->size;
        be[2].binding = 2; be[2].buffer = meta_buf;  be[2].offset = 0; be[2].size = 128;
        wgpu::BindGroupDescriptor bgd = {};
        bgd.layout = bgl; bgd.entryCount = 3; bgd.entries = be;
        cached_bg = dev.device.CreateBindGroup(&bgd);
        cached_gen[0] = gen_x;
        cached_gen[1] = gen_y;
    }

    wgpu::ComputePassEncoder pass = shared_encoder().BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, cached_bg);
    pass.DispatchWorkgroups(dispatch_gx, dispatch_gy, 1);
    pass.End();

    mark_gpu_written(outputs[0]);
    return true;
}

} // namespace nnr::webgpu
