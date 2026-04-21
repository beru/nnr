#include "elementwise.h"

#include "device.h"
#include "buffer.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>

namespace nnr::webgpu {

namespace {

constexpr uint32_t WG = 256;

std::string make_unary_wgsl(const char* expr, const char* in_ty, const char* out_ty) {
    // grid_stride_x = threads-along-x on the host side (= gx * workgroup_size).
    // Host may 2D-split the dispatch past WebGPU's 65535-per-dim cap; the
    // shader reconstructs the flat id via gid.y * grid_stride_x + gid.x.
    std::string s =
        "struct Dims { n : u32, grid_stride_x : u32, _b : u32, _c : u32 };\n"
        "@group(0) @binding(0) var<storage, read>       X : array<";
    s += in_ty;
    s += ">;\n"
         "@group(0) @binding(1) var<storage, read_write> Y : array<";
    s += out_ty;
    s += ">;\n"
         "@group(0) @binding(2) var<uniform>             dims : Dims;\n"
         "@compute @workgroup_size(256)\n"
         "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
         "  let i = gid.y * dims.grid_stride_x + gid.x;\n"
         "  if (i >= dims.n) { return; }\n"
         "  let v : ";
    s += in_ty;
    s += " = X[i];\n"
         "  Y[i] = ";
    s += expr;
    s += ";\n}\n";
    return s;
}

// General-rank broadcast (up to 8 dims). Output index `o` is unflattened
// through `out_dims` in reverse; each axis's output coordinate is mapped to A
// and B via their own broadcast-aware strides (dim==1 or missing → stride 0).
// Meta lives in a storage buffer so array layout matches the C++ side
// (std140 uniform would force 16-byte strides on u32 arrays).
std::string make_binary_wgsl(const char* expr, const char* in_ty, const char* out_ty) {
    std::string s =
        "struct Meta {\n"
        "  total          : u32,\n"
        "  ndim           : u32,\n"
        "  grid_stride_x  : u32,\n"   // threads-along-x for 2D dispatch split
        "  _b             : u32,\n"
        "  out_dims_lo    : vec4<u32>,\n"
        "  out_dims_hi    : vec4<u32>,\n"
        "  a_strides_lo   : vec4<u32>,\n"
        "  a_strides_hi   : vec4<u32>,\n"
        "  b_strides_lo   : vec4<u32>,\n"
        "  b_strides_hi   : vec4<u32>,\n"
        "};\n"
        "@group(0) @binding(0) var<storage, read>       A  : array<";
    s += in_ty;
    s += ">;\n"
        "@group(0) @binding(1) var<storage, read>       B  : array<";
    s += in_ty;
    s += ">;\n"
        "@group(0) @binding(2) var<storage, read_write> Y  : array<";
    s += out_ty;
    s += ">;\n"
        "@group(0) @binding(3) var<storage, read>       md : Meta;\n"
        "fn get_out_dim(i : u32) -> u32 { if (i < 4u) { return md.out_dims_lo[i]; } return md.out_dims_hi[i - 4u]; }\n"
        "fn get_a_stride(i : u32) -> u32 { if (i < 4u) { return md.a_strides_lo[i]; } return md.a_strides_hi[i - 4u]; }\n"
        "fn get_b_stride(i : u32) -> u32 { if (i < 4u) { return md.b_strides_lo[i]; } return md.b_strides_hi[i - 4u]; }\n"
        "@compute @workgroup_size(256)\n"
        "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
        "  let o = gid.y * md.grid_stride_x + gid.x;\n"
        "  if (o >= md.total) { return; }\n"
        "  var a_flat : u32 = 0u;\n"
        "  var b_flat : u32 = 0u;\n"
        "  var tmp    : u32 = o;\n"
        "  for (var k : i32 = i32(md.ndim) - 1; k >= 0; k = k - 1) {\n"
        "    let d   = get_out_dim(u32(k));\n"
        "    let idx = tmp % d;\n"
        "    tmp     = tmp / d;\n"
        "    a_flat  = a_flat + idx * get_a_stride(u32(k));\n"
        "    b_flat  = b_flat + idx * get_b_stride(u32(k));\n"
        "  }\n"
        "  let a : ";
    s += in_ty;
    s += " = A[a_flat];\n"
         "  let b : ";
    s += in_ty;
    s += " = B[b_flat];\n"
         "  Y[o] = ";
    s += expr;
    s += ";\n}\n";
    return s;
}

wgpu::ShaderModule compile(const std::string& src) {
    auto& dev = get_device();
    wgpu::ShaderSourceWGSL w = {};
    w.code = src.c_str();
    wgpu::ShaderModuleDescriptor d = {};
    d.nextInChain = &w;
    return dev.device.CreateShaderModule(&d);
}

wgpu::BindGroupLayout make_bgl(int n_buffers, int has_uniform) {
    auto& dev = get_device();
    wgpu::BindGroupLayoutEntry e[8] = {};
    int idx = 0;
    for (int i = 0; i < n_buffers - 1; ++i, ++idx) {
        e[idx].binding = idx;
        e[idx].visibility = wgpu::ShaderStage::Compute;
        e[idx].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    }
    e[idx].binding = idx;
    e[idx].visibility = wgpu::ShaderStage::Compute;
    e[idx].buffer.type = wgpu::BufferBindingType::Storage;
    ++idx;
    if (has_uniform) {
        e[idx].binding = idx;
        e[idx].visibility = wgpu::ShaderStage::Compute;
        e[idx].buffer.type = wgpu::BufferBindingType::Uniform;
        ++idx;
    }
    wgpu::BindGroupLayoutDescriptor d = {};
    d.entryCount = idx;
    d.entries = e;
    return dev.device.CreateBindGroupLayout(&d);
}

wgpu::Buffer make_uniforms() {
    auto& dev = get_device();
    wgpu::BufferDescriptor d = {};
    d.size = 16;
    d.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    return dev.device.CreateBuffer(&d);
}

wgpu::ComputePipeline make_pipeline(wgpu::ShaderModule sm, wgpu::BindGroupLayout bgl) {
    auto& dev = get_device();
    wgpu::PipelineLayoutDescriptor pld = {};
    pld.bindGroupLayoutCount = 1;
    pld.bindGroupLayouts = &bgl;
    wgpu::PipelineLayout pl = dev.device.CreatePipelineLayout(&pld);
    wgpu::ComputePipelineDescriptor cpd = {};
    cpd.layout = pl;
    cpd.compute.module = sm;
    cpd.compute.entryPoint = "main";
    return dev.device.CreateComputePipeline(&cpd);
}

} // namespace

bool unary_elementwise_t::init() {
    if (!is_inout_size(1, 1)) return false;
    // Early-reject non-matching dtype so the loader falls back to CPU.
    data_type_t in_ty = input_dtype();
    if (inputs[0] && inputs[0]->type != NNR_DATA_TYPE_UNDEFINED
        && inputs[0]->type != in_ty) return false;
    if (!device_ready()) return false;
    auto sm = compile(make_unary_wgsl(op_expr(), input_wgsl_ty(), output_wgsl_ty()));
    bgl = make_bgl(/*n_buffers=*/2, /*has_uniform=*/1);
    pipeline = make_pipeline(sm, bgl);
    uniforms = make_uniforms();
    return true;
}

bool unary_elementwise_t::reshape() {
    const tensor_t* x = inputs[0];
    tensor_t*       y = outputs[0];
    data_type_t in_ty = input_dtype(), out_ty = output_dtype();
    if (x->type != in_ty) return false;
    if (!y->reshape_identity(x, out_ty)) return false;
    ensure_buffer(x, x->ndata * data_type_sizeof(in_ty));
    ensure_buffer(y, y->ndata * data_type_sizeof(out_ty));

    // Dispatch grid + uniform are pure functions of output ndata.
    // Compute and persist them now so exec() needs neither.
    uint32_t groups = ((uint32_t)y->ndata + WG - 1) / WG;
    dispatch_1d_grid(groups, dispatch_gx, dispatch_gy);
    uint32_t u[4] = { (uint32_t)y->ndata, dispatch_gx * WG, 0, 0 };
    get_device().queue.WriteBuffer(uniforms, 0, u, sizeof(u));
    return true;
}

bool unary_elementwise_t::exec() {
    auto& dev = get_device();
    upload_if_needed(inputs[0]);

    auto* rx = find(inputs[0]);
    auto* ry = find(outputs[0]);

    uint32_t gen_x = generation_of(inputs[0]);
    uint32_t gen_y = generation_of(outputs[0]);
    if (!cached_bg || gen_x != cached_gen[0] || gen_y != cached_gen[1]) {
        wgpu::BindGroupEntry be[3] = {};
        be[0].binding = 0; be[0].buffer = rx->buf; be[0].offset = 0; be[0].size = rx->size;
        be[1].binding = 1; be[1].buffer = ry->buf; be[1].offset = 0; be[1].size = ry->size;
        be[2].binding = 2; be[2].buffer = uniforms; be[2].offset = 0; be[2].size = 16;
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

bool binary_elementwise_t::init() {
    if (!is_inout_size(2, 1)) return false;
    // Early-reject dtypes we can't handle so the onnx_loader falls back to
    // CPU. Without this the failure surfaces at prepare()'s reshape() call
    // which has no backend-fallback path. Common miss: dynamic-shape
    // graphs do int64 arithmetic (Shape→Mul→Concat) and need CPU.
    data_type_t in_ty = input_dtype();
    if (inputs[0] && inputs[0]->type != NNR_DATA_TYPE_UNDEFINED
        && inputs[0]->type != in_ty) return false;
    if (inputs[1] && inputs[1]->type != NNR_DATA_TYPE_UNDEFINED
        && inputs[1]->type != in_ty) return false;
    if (!device_ready()) return false;
    auto& dev = get_device();

    auto sm = compile(make_binary_wgsl(op_expr(), input_wgsl_ty(), output_wgsl_ty()));

    wgpu::BindGroupLayoutEntry e[4] = {};
    e[0].binding = 0; e[0].visibility = wgpu::ShaderStage::Compute;
    e[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    e[1].binding = 1; e[1].visibility = wgpu::ShaderStage::Compute;
    e[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    e[2].binding = 2; e[2].visibility = wgpu::ShaderStage::Compute;
    e[2].buffer.type = wgpu::BufferBindingType::Storage;
    e[3].binding = 3; e[3].visibility = wgpu::ShaderStage::Compute;
    e[3].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    wgpu::BindGroupLayoutDescriptor bgld = {};
    bgld.entryCount = 4; bgld.entries = e;
    bgl = dev.device.CreateBindGroupLayout(&bgld);

    pipeline = make_pipeline(sm, bgl);

    // 16B header + 32B out_dims + 32B a_strides + 32B b_strides = 112B, round to 128.
    wgpu::BufferDescriptor md = {};
    md.size = 128;
    md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    meta_buf = dev.device.CreateBuffer(&md);
    return true;
}

bool binary_elementwise_t::reshape() {
    const tensor_t* a = inputs[0];
    const tensor_t* b = inputs[1];
    tensor_t*       y = outputs[0];
    data_type_t in_ty = input_dtype();
    if (a->type != in_ty || b->type != in_ty) return false;
    if (a->ndim > 8 || b->ndim > 8) return false;

    // Broadcast shape by right-aligning. Missing leading axes are size 1.
    int out_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    int out_dims[8] = {};
    for (int k = 0; k < out_ndim; ++k) {
        int ra = k - (out_ndim - a->ndim);   // index into a->dims, or < 0 if missing
        int rb = k - (out_ndim - b->ndim);
        int da = (ra >= 0) ? a->dims[ra] : 1;
        int db = (rb >= 0) ? b->dims[rb] : 1;
        if (da != db && da != 1 && db != 1) return false;
        out_dims[k] = da > db ? da : db;
    }
    data_type_t out_ty = output_dtype();
    if (!y->reshape(std::span<const int>(out_dims, out_ndim), out_ty)) return false;

    // Natural row-major strides for each input (over its own ndim).
    uint32_t a_nat[8] = {}, b_nat[8] = {};
    {
        uint32_t s = 1;
        for (int i = a->ndim - 1; i >= 0; --i) { a_nat[i] = s; s *= (uint32_t)a->dims[i]; }
    }
    {
        uint32_t s = 1;
        for (int i = b->ndim - 1; i >= 0; --i) { b_nat[i] = s; s *= (uint32_t)b->dims[i]; }
    }

    // Broadcast-aware strides aligned to the output axis count. Missing axes
    // or size-1 axes map to stride 0 so `idx * stride == 0` for any idx.
    for (int k = 0; k < 8; ++k) { out_dims_u[k] = 0; a_strides_u[k] = 0; b_strides_u[k] = 0; }
    for (int k = 0; k < out_ndim; ++k) {
        out_dims_u[k] = (uint32_t)out_dims[k];
        int ra = k - (out_ndim - a->ndim);
        int rb = k - (out_ndim - b->ndim);
        a_strides_u[k] = (ra >= 0 && a->dims[ra] != 1) ? a_nat[ra] : 0u;
        b_strides_u[k] = (rb >= 0 && b->dims[rb] != 1) ? b_nat[rb] : 0u;
    }
    ndim  = (uint32_t)out_ndim;
    total = 1;
    for (int k = 0; k < out_ndim; ++k) total *= (uint32_t)out_dims[k];

    size_t in_sz = data_type_sizeof(in_ty);
    ensure_buffer(a, a->ndata * in_sz);
    ensure_buffer(b, b->ndata * in_sz);
    ensure_buffer(y, y->ndata * data_type_sizeof(out_ty));

    // Dispatch grid + meta payload depend only on shape — write here so
    // exec() does only upload + bind + dispatch.
    uint32_t groups = (total + WG - 1) / WG;
    dispatch_1d_grid(groups, dispatch_gx, dispatch_gy);
    uint32_t grid_stride_x = dispatch_gx * WG;

    uint8_t buf[128] = {};
    auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(buf + off, &v, 4); };
    put_u32(0, total);
    put_u32(4, ndim);
    put_u32(8, grid_stride_x);
    for (int i = 0; i < 8; ++i) put_u32(16 + i * 4, out_dims_u[i]);
    for (int i = 0; i < 8; ++i) put_u32(48 + i * 4, a_strides_u[i]);
    for (int i = 0; i < 8; ++i) put_u32(80 + i * 4, b_strides_u[i]);
    get_device().queue.WriteBuffer(meta_buf, 0, buf, sizeof(buf));
    return true;
}

// ---------------------------------------------------------------------------
// fused_elementwise_chain_t — variable-arity N-stage chain composition
// ---------------------------------------------------------------------------
//
// Each entry in `stage_wgsl` is a fully-substituted WGSL expression in `v`
// (plus, optionally, reads of `S<k>[i]` or `S<k>[side_<k>_flat]`). WGSL
// `var v` lets us model pipeline composition as reassignment:
//
//   var v : f32 = X[i];
//   v = <stage0>;   // unary:  "max(v, 0.0)"
//   v = <stage1>;   // binary: "v + (S0[i])"      — Path U (same-shape)
//                   //     or  "v + (S0[side_0_flat])" — Path M (broadcast)
//   ...
//   Y[i] = v;
//
// Bindings:
//   Path U (needs_meta=false):
//     0        pipe X
//     1..N     S0..S{N-1}
//     N+1      Y
//     N+2      uniform dims {n, 0, 0, 0}
//   Path M (needs_meta=true):
//     0        pipe X
//     1..N     S0..S{N-1}
//     N+1      Y
//     N+2      storage md {total, ndim, out_dims, stride tables per side}
//
// Pipelines are cached by (mode, n_sides, stage list).

namespace {

std::string make_chain_wgsl_simple(const std::vector<std::string>& stage_wgsl,
                                   int n_sides)
{
    std::string s =
        "struct Dims { n : u32, _a : u32, _b : u32, _c : u32 };\n"
        "@group(0) @binding(0) var<storage, read>       X : array<f32>;\n";
    for (int k = 0; k < n_sides; ++k) {
        s += "@group(0) @binding(";
        s += std::to_string(1 + k);
        s += ") var<storage, read>       S";
        s += std::to_string(k);
        s += " : array<f32>;\n";
    }
    const int y_binding = 1 + n_sides;
    const int u_binding = 2 + n_sides;
    s += "@group(0) @binding(";
    s += std::to_string(y_binding);
    s += ") var<storage, read_write> Y : array<f32>;\n"
         "@group(0) @binding(";
    s += std::to_string(u_binding);
    s += ") var<uniform>             dims : Dims;\n"
         "@compute @workgroup_size(256)\n"
         "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
         "  let i = gid.x;\n"
         "  if (i >= dims.n) { return; }\n"
         "  var v : f32 = X[i];\n";
    for (const auto& stg : stage_wgsl) {
        s += "  v = ";
        s += stg;
        s += ";\n";
    }
    s += "  Y[i] = v;\n}\n";
    return s;
}

// Path M shader: emits a Meta struct with out_dims + per-side stride vec4
// pairs, an unflatten loop that accumulates `side_<k>_flat` for every
// binary stage, and the stage body. u32 arrays inside storage buffers use
// natural 4-byte alignment, but packing as `vec4<u32>` pairs guarantees
// 16-byte alignment across GPU drivers — matching the convention used by
// `binary_elementwise_t`'s broadcast shader.
std::string make_chain_wgsl_meta(const std::vector<std::string>& stage_wgsl,
                                 int n_sides)
{
    std::string s = "struct Meta {\n"
                    "  total          : u32,\n"
                    "  ndim           : u32,\n"
                    "  _pad0          : u32,\n"
                    "  _pad1          : u32,\n"
                    "  out_dims_lo    : vec4<u32>,\n"
                    "  out_dims_hi    : vec4<u32>,\n";
    for (int k = 0; k < n_sides; ++k) {
        s += "  s";
        s += std::to_string(k);
        s += "_strides_lo : vec4<u32>,\n";
        s += "  s";
        s += std::to_string(k);
        s += "_strides_hi : vec4<u32>,\n";
    }
    s += "};\n";

    s += "@group(0) @binding(0) var<storage, read>       X : array<f32>;\n";
    for (int k = 0; k < n_sides; ++k) {
        s += "@group(0) @binding(";
        s += std::to_string(1 + k);
        s += ") var<storage, read>       S";
        s += std::to_string(k);
        s += " : array<f32>;\n";
    }
    const int y_binding = 1 + n_sides;
    const int m_binding = 2 + n_sides;
    s += "@group(0) @binding(";
    s += std::to_string(y_binding);
    s += ") var<storage, read_write> Y : array<f32>;\n"
         "@group(0) @binding(";
    s += std::to_string(m_binding);
    s += ") var<storage, read>       md : Meta;\n";

    s += "fn get_out_dim(i : u32) -> u32 { if (i < 4u) { return md.out_dims_lo[i]; } return md.out_dims_hi[i - 4u]; }\n";
    for (int k = 0; k < n_sides; ++k) {
        auto ks = std::to_string(k);
        s += "fn get_s";
        s += ks;
        s += "_stride(i : u32) -> u32 { if (i < 4u) { return md.s";
        s += ks;
        s += "_strides_lo[i]; } return md.s";
        s += ks;
        s += "_strides_hi[i - 4u]; }\n";
    }

    s += "@compute @workgroup_size(256)\n"
         "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
         "  let i = gid.x;\n"
         "  if (i >= md.total) { return; }\n";
    for (int k = 0; k < n_sides; ++k) {
        s += "  var side_";
        s += std::to_string(k);
        s += "_flat : u32 = 0u;\n";
    }
    s += "  var tmp : u32 = i;\n"
         "  for (var k : i32 = i32(md.ndim) - 1; k >= 0; k = k - 1) {\n"
         "    let d = get_out_dim(u32(k));\n"
         "    let idx = tmp % d;\n"
         "    tmp = tmp / d;\n";
    for (int k = 0; k < n_sides; ++k) {
        auto ks = std::to_string(k);
        s += "    side_";
        s += ks;
        s += "_flat = side_";
        s += ks;
        s += "_flat + idx * get_s";
        s += ks;
        s += "_stride(u32(k));\n";
    }
    s += "  }\n"
         "  var v : f32 = X[i];\n";
    for (const auto& stg : stage_wgsl) {
        s += "  v = ";
        s += stg;
        s += ";\n";
    }
    s += "  Y[i] = v;\n}\n";
    return s;
}

wgpu::BindGroupLayout make_chain_bgl(int n_sides, bool needs_meta)
{
    auto& dev = get_device();
    const int total = 3 + n_sides;  // pipe + sides + output + (uniform | storage)
    std::vector<wgpu::BindGroupLayoutEntry> e(total);
    int idx = 0;
    for (int k = 0; k < 1 + n_sides; ++k, ++idx) {
        e[idx].binding = idx;
        e[idx].visibility = wgpu::ShaderStage::Compute;
        e[idx].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    }
    e[idx].binding = idx;
    e[idx].visibility = wgpu::ShaderStage::Compute;
    e[idx].buffer.type = wgpu::BufferBindingType::Storage;
    ++idx;
    e[idx].binding = idx;
    e[idx].visibility = wgpu::ShaderStage::Compute;
    e[idx].buffer.type = needs_meta
        ? wgpu::BufferBindingType::ReadOnlyStorage
        : wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutDescriptor d = {};
    d.entryCount = (uint32_t)total;
    d.entries = e.data();
    return dev.device.CreateBindGroupLayout(&d);
}

struct fused_chain_entry_t {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
};

std::unordered_map<std::string, fused_chain_entry_t>& fused_chain_cache()
{
    static std::unordered_map<std::string, fused_chain_entry_t> cache;
    return cache;
}

// Meta layout for Path M (storage buffer):
//   [0..4)     total
//   [4..8)     ndim
//   [8..16)    pad
//   [16..48)   out_dims  (8 × u32; only first ndim valid)
//   [48..80)   side_0 strides
//   [80..112)  side_1 strides
//   ...
//   48 + 32*k  side_k strides
constexpr size_t META_HEADER = 48;
constexpr size_t META_STRIDE_TABLE = 32;  // 8 u32 per side

} // namespace

bool fused_elementwise_chain_t::init() {
    if (stage_wgsl.size() < 2)                    return false;
    if ((int)inputs.size() != 1 + n_sides)        return false;
    if (outputs.size() != 1)                      return false;
    if (!device_ready())                          return false;
    if (needs_meta && n_sides < 1)                return false;

    std::string sig;
    sig += needs_meta ? 'M' : 'U';
    sig += std::to_string(n_sides);
    sig += '\x02';
    for (const auto& s : stage_wgsl) { sig += s; sig += '\x01'; }

    auto& cache = fused_chain_cache();
    auto it = cache.find(sig);
    if (it == cache.end()) {
        auto src = needs_meta
            ? make_chain_wgsl_meta(stage_wgsl, n_sides)
            : make_chain_wgsl_simple(stage_wgsl, n_sides);
        auto sm  = compile(src);
        fused_chain_entry_t e;
        e.bgl      = make_chain_bgl(n_sides, needs_meta);
        e.pipeline = make_pipeline(sm, e.bgl);
        it = cache.emplace(std::move(sig), std::move(e)).first;
    }
    bgl      = it->second.bgl;
    pipeline = it->second.pipeline;

    if (needs_meta) {
        const size_t sz = META_HEADER + META_STRIDE_TABLE * (size_t)n_sides;
        wgpu::BufferDescriptor md = {};
        md.size  = (sz + 15) & ~size_t(15);  // 16-byte round
        md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        meta_buf = get_device().device.CreateBuffer(&md);
    } else {
        uniforms = make_uniforms();
    }
    return true;
}

bool fused_elementwise_chain_t::reshape() {
    if ((int)inputs.size() != 1 + n_sides) return false;
    if (outputs.size() != 1)               return false;
    const tensor_t* x = inputs[0];
    tensor_t*       y = outputs[0];
    if (!x || !y)                            return false;
    if (x->type != NNR_DATA_TYPE_FLOAT32)    return false;
    if (x->ndim > 8)                          return false;

    // Validate each side input. Same-shape is always accepted. Broadcast —
    // side.ndim ≤ pipe.ndim, each right-aligned axis is 1 or equals pipe's —
    // is accepted only in Path M. Path U rejects any shape mismatch (must
    // match what the fusion pass decided at init time).
    for (int k = 0; k < n_sides; ++k) {
        const tensor_t* s = inputs[1 + k];
        if (!s)                                 return false;
        if (s->type != NNR_DATA_TYPE_FLOAT32)   return false;
        if (s->ndim > x->ndim)                  return false;
        bool exact = (s->ndim == x->ndim);
        if (exact) {
            for (int i = 0; i < s->ndim; ++i)
                if (s->dims[i] != x->dims[i]) { exact = false; break; }
        }
        if (!exact) {
            if (!needs_meta)                    return false;
            // Broadcast check: right-align and verify each side axis is 1
            // or equal to pipe's axis at that position.
            int off = x->ndim - s->ndim;
            for (int i = 0; i < s->ndim; ++i) {
                int d_side = s->dims[i];
                int d_pipe = x->dims[off + i];
                if (d_side != 1 && d_side != d_pipe) return false;
            }
        }
    }
    if (!y->reshape_identity(x, NNR_DATA_TYPE_FLOAT32)) return false;

    ensure_buffer(x, x->ndata * sizeof(float));
    for (int k = 0; k < n_sides; ++k)
        ensure_buffer(inputs[1 + k], inputs[1 + k]->ndata * sizeof(float));
    ensure_buffer(y, y->ndata * sizeof(float));

    // Uniform / meta payloads depend only on shape + per-side broadcast
    // tables — write here so exec() does only upload + bind + dispatch.
    auto& dev = get_device();
    if (!needs_meta) {
        uint32_t u[4] = { (uint32_t)y->ndata, 0, 0, 0 };
        dev.queue.WriteBuffer(uniforms, 0, u, sizeof(u));
    } else {
        const size_t sz = META_HEADER + META_STRIDE_TABLE * (size_t)n_sides;
        const size_t rounded = (sz + 15) & ~size_t(15);
        std::vector<uint8_t> buf(rounded, 0);
        auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(buf.data() + off, &v, 4); };
        put_u32(0, (uint32_t)y->ndata);
        put_u32(4, (uint32_t)x->ndim);
        for (int i = 0; i < x->ndim; ++i) put_u32(16 + i * 4, (uint32_t)x->dims[i]);

        for (int k = 0; k < n_sides; ++k) {
            const tensor_t* s = inputs[1 + k];
            uint32_t nat[8] = {};
            {
                uint32_t st = 1;
                for (int a = s->ndim - 1; a >= 0; --a) {
                    nat[a] = st;
                    st *= (uint32_t)s->dims[a];
                }
            }
            const int off = x->ndim - s->ndim;
            for (int a = 0; a < x->ndim; ++a) {
                int sa = a - off;
                uint32_t stride = 0;
                if (sa >= 0 && s->dims[sa] != 1) stride = nat[sa];
                put_u32(META_HEADER + 32 * (size_t)k + a * 4, stride);
            }
        }
        dev.queue.WriteBuffer(meta_buf, 0, buf.data(), buf.size());
    }
    return true;
}

bool fused_elementwise_chain_t::exec() {
    auto& dev = get_device();
    upload_if_needed(inputs[0]);
    for (int k = 0; k < n_sides; ++k) upload_if_needed(inputs[1 + k]);

    const tensor_t* x = inputs[0];
    tensor_t*       y = outputs[0];

    // Uniform / meta were written in reshape() — pure functions of shape.
    const int y_binding = 1 + n_sides;
    const int tail_binding = 2 + n_sides;
    const int total_bindings = 3 + n_sides;

    // Tensor-backed slots: pipe (1) + sides (n_sides) + output (1). The
    // tail binding (uniforms or meta_buf) is op-owned and stable, so it
    // doesn't participate in cache-invalidation bookkeeping.
    const int n_tensor_slots = 2 + n_sides;
    uint32_t cur_gens[16] = {};
    cur_gens[0] = generation_of(x);
    for (int k = 0; k < n_sides; ++k) cur_gens[1 + k] = generation_of(inputs[1 + k]);
    cur_gens[1 + n_sides] = generation_of(y);
    bool bg_valid = (bool)cached_bg;
    for (int k = 0; k < n_tensor_slots && bg_valid; ++k) {
        if (cur_gens[k] != cached_gen[k]) bg_valid = false;
    }
    if (!bg_valid) {
        std::vector<wgpu::BindGroupEntry> be(total_bindings);
        auto* rx = find(x);
        be[0].binding = 0; be[0].buffer = rx->buf; be[0].offset = 0; be[0].size = rx->size;
        for (int k = 0; k < n_sides; ++k) {
            auto* rs = find(inputs[1 + k]);
            be[1 + k].binding = 1 + k;
            be[1 + k].buffer  = rs->buf;
            be[1 + k].offset  = 0;
            be[1 + k].size    = rs->size;
        }
        auto* ry = find(y);
        be[y_binding].binding = y_binding;
        be[y_binding].buffer  = ry->buf;
        be[y_binding].offset  = 0;
        be[y_binding].size    = ry->size;
        be[tail_binding].binding = tail_binding;
        if (!needs_meta) {
            be[tail_binding].buffer = uniforms;
            be[tail_binding].size   = 16;
        } else {
            be[tail_binding].buffer = meta_buf;
            be[tail_binding].size   = (META_HEADER + META_STRIDE_TABLE * (size_t)n_sides + 15) & ~size_t(15);
        }
        be[tail_binding].offset = 0;

        wgpu::BindGroupDescriptor bgd = {};
        bgd.layout = bgl;
        bgd.entryCount = (uint32_t)total_bindings;
        bgd.entries = be.data();
        cached_bg = dev.device.CreateBindGroup(&bgd);
        for (int k = 0; k < n_tensor_slots; ++k) cached_gen[k] = cur_gens[k];
    }

    wgpu::ComputePassEncoder pass = shared_encoder().BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, cached_bg);
    uint32_t groups = ((uint32_t)y->ndata + WG - 1) / WG;
    pass.DispatchWorkgroups(groups, 1, 1);
    pass.End();
    mark_gpu_written(y);
    return true;
}

bool binary_elementwise_t::exec() {
    auto& dev = get_device();
    upload_if_needed(inputs[0]);
    upload_if_needed(inputs[1]);

    // meta_buf was written in reshape() — its contents are pure functions
    // of the broadcast shape and stride tables computed there.
    auto* ra = find(inputs[0]);
    auto* rb = find(inputs[1]);
    auto* ry = find(outputs[0]);

    uint32_t gen_a = generation_of(inputs[0]);
    uint32_t gen_b = generation_of(inputs[1]);
    uint32_t gen_y = generation_of(outputs[0]);
    if (!cached_bg || gen_a != cached_gen[0]
                   || gen_b != cached_gen[1]
                   || gen_y != cached_gen[2]) {
        wgpu::BindGroupEntry be[4] = {};
        be[0].binding = 0; be[0].buffer = ra->buf;   be[0].offset = 0; be[0].size = ra->size;
        be[1].binding = 1; be[1].buffer = rb->buf;   be[1].offset = 0; be[1].size = rb->size;
        be[2].binding = 2; be[2].buffer = ry->buf;   be[2].offset = 0; be[2].size = ry->size;
        be[3].binding = 3; be[3].buffer = meta_buf;  be[3].offset = 0; be[3].size = 128;
        wgpu::BindGroupDescriptor bgd = {};
        bgd.layout = bgl; bgd.entryCount = 4; bgd.entries = be;
        cached_bg = dev.device.CreateBindGroup(&bgd);
        cached_gen[0] = gen_a;
        cached_gen[1] = gen_b;
        cached_gen[2] = gen_y;
    }

    wgpu::ComputePassEncoder pass = shared_encoder().BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, cached_bg);
    pass.DispatchWorkgroups(dispatch_gx, dispatch_gy, 1);
    pass.End();
    mark_gpu_written(outputs[0]);
    return true;
}

// ---------------------------------------------------------------------------
// matmul_fused_chain_t — MatMul with absorbed elementwise chain epilogue
// ---------------------------------------------------------------------------
//
// Composed from the 16x16-tile matmul body + a chain-style epilogue that
// reassigns a local `var v : f32` through each `stage_wgsl` string before the
// output write. Same substitution convention the chain fusion pass already
// produces: Path U stages reference `S<k>[i]` with `i = row*N + col`, Path M
// stages reference `S<k>[side_<k>_flat]` which we compute inline from `row`
// and `col` via a per-side (stride_row, stride_col) pair packed in the meta
// buffer. Bind-group layout mirrors the MatMul seed op with the side storage
// bindings inserted between B and Y, and the Path-M meta appended after the
// Dims uniform.

namespace {

constexpr uint32_t MATMUL_TILE = 16;

// Path U shader builder.
//
// Bindings:
//   0       A
//   1       B
//   2..2+n_sides-1   S0..S{n_sides-1}
//   2+n_sides        C (read_write)
//   3+n_sides        Dims (uniform)
std::string make_matmul_chain_wgsl_simple(const std::vector<std::string>& stage_wgsl,
                                          int n_sides)
{
    std::string s =
        "struct Dims { M : u32, N : u32, K : u32, _pad : u32 };\n"
        "@group(0) @binding(0) var<storage, read>       A : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read>       B : array<f32>;\n";
    for (int k = 0; k < n_sides; ++k) {
        s += "@group(0) @binding(";
        s += std::to_string(2 + k);
        s += ") var<storage, read>       S";
        s += std::to_string(k);
        s += " : array<f32>;\n";
    }
    const int c_binding = 2 + n_sides;
    const int d_binding = 3 + n_sides;
    s += "@group(0) @binding(";
    s += std::to_string(c_binding);
    s += ") var<storage, read_write> C : array<f32>;\n"
         "@group(0) @binding(";
    s += std::to_string(d_binding);
    s += ") var<uniform>             dims : Dims;\n"
         "const TILE : u32 = 16u;\n"
         "var<workgroup> tA : array<array<f32, 16>, 16>;\n"
         "var<workgroup> tB : array<array<f32, 16>, 16>;\n"
         "@compute @workgroup_size(16, 16, 1)\n"
         "fn main(@builtin(global_invocation_id) gid : vec3<u32>,\n"
         "        @builtin(local_invocation_id)  lid : vec3<u32>) {\n"
         "  let row = gid.y;\n"
         "  let col = gid.x;\n"
         "  let M = dims.M;\n"
         "  let N = dims.N;\n"
         "  let K = dims.K;\n"
         "  var acc : f32 = 0.0;\n"
         "  let tiles = (K + TILE - 1u) / TILE;\n"
         "  for (var t : u32 = 0u; t < tiles; t = t + 1u) {\n"
         "    let a_col = t * TILE + lid.x;\n"
         "    let b_row = t * TILE + lid.y;\n"
         "    if (row < M && a_col < K) { tA[lid.y][lid.x] = A[row * K + a_col]; } else { tA[lid.y][lid.x] = 0.0; }\n"
         "    if (b_row < K && col < N) { tB[lid.y][lid.x] = B[b_row * N + col]; } else { tB[lid.y][lid.x] = 0.0; }\n"
         "    workgroupBarrier();\n"
         "    for (var kk : u32 = 0u; kk < TILE; kk = kk + 1u) {\n"
         "      acc = acc + tA[lid.y][kk] * tB[kk][lid.x];\n"
         "    }\n"
         "    workgroupBarrier();\n"
         "  }\n"
         "  if (row < M && col < N) {\n"
         "    let i = row * N + col;\n"
         "    var v : f32 = acc;\n";
    for (const auto& stg : stage_wgsl) {
        s += "    v = ";
        s += stg;
        s += ";\n";
    }
    s += "    C[i] = v;\n"
         "  }\n"
         "}\n";
    return s;
}

// Path M shader builder. Adds per-side (stride_row, stride_col) packed as
// `vec4<u32>` entries in a storage meta buffer (last two lanes unused). Each
// `side_<k>_flat` is computed inline: `row * stride_row + col * stride_col`.
std::string make_matmul_chain_wgsl_meta(const std::vector<std::string>& stage_wgsl,
                                        int n_sides)
{
    std::string s = "struct Meta {\n";
    for (int k = 0; k < n_sides; ++k) {
        s += "  s";
        s += std::to_string(k);
        s += "_strides : vec4<u32>,\n";
    }
    s += "};\n"
         "struct Dims { M : u32, N : u32, K : u32, _pad : u32 };\n"
         "@group(0) @binding(0) var<storage, read>       A : array<f32>;\n"
         "@group(0) @binding(1) var<storage, read>       B : array<f32>;\n";
    for (int k = 0; k < n_sides; ++k) {
        s += "@group(0) @binding(";
        s += std::to_string(2 + k);
        s += ") var<storage, read>       S";
        s += std::to_string(k);
        s += " : array<f32>;\n";
    }
    const int c_binding = 2 + n_sides;
    const int d_binding = 3 + n_sides;
    const int m_binding = 4 + n_sides;
    s += "@group(0) @binding(";
    s += std::to_string(c_binding);
    s += ") var<storage, read_write> C : array<f32>;\n"
         "@group(0) @binding(";
    s += std::to_string(d_binding);
    s += ") var<uniform>             dims : Dims;\n"
         "@group(0) @binding(";
    s += std::to_string(m_binding);
    s += ") var<storage, read>       md : Meta;\n"
         "const TILE : u32 = 16u;\n"
         "var<workgroup> tA : array<array<f32, 16>, 16>;\n"
         "var<workgroup> tB : array<array<f32, 16>, 16>;\n"
         "@compute @workgroup_size(16, 16, 1)\n"
         "fn main(@builtin(global_invocation_id) gid : vec3<u32>,\n"
         "        @builtin(local_invocation_id)  lid : vec3<u32>) {\n"
         "  let row = gid.y;\n"
         "  let col = gid.x;\n"
         "  let M = dims.M;\n"
         "  let N = dims.N;\n"
         "  let K = dims.K;\n"
         "  var acc : f32 = 0.0;\n"
         "  let tiles = (K + TILE - 1u) / TILE;\n"
         "  for (var t : u32 = 0u; t < tiles; t = t + 1u) {\n"
         "    let a_col = t * TILE + lid.x;\n"
         "    let b_row = t * TILE + lid.y;\n"
         "    if (row < M && a_col < K) { tA[lid.y][lid.x] = A[row * K + a_col]; } else { tA[lid.y][lid.x] = 0.0; }\n"
         "    if (b_row < K && col < N) { tB[lid.y][lid.x] = B[b_row * N + col]; } else { tB[lid.y][lid.x] = 0.0; }\n"
         "    workgroupBarrier();\n"
         "    for (var kk : u32 = 0u; kk < TILE; kk = kk + 1u) {\n"
         "      acc = acc + tA[lid.y][kk] * tB[kk][lid.x];\n"
         "    }\n"
         "    workgroupBarrier();\n"
         "  }\n"
         "  if (row < M && col < N) {\n"
         "    let i = row * N + col;\n";
    for (int k = 0; k < n_sides; ++k) {
        auto ks = std::to_string(k);
        s += "    let side_";
        s += ks;
        s += "_flat : u32 = row * md.s";
        s += ks;
        s += "_strides.x + col * md.s";
        s += ks;
        s += "_strides.y;\n";
    }
    s += "    var v : f32 = acc;\n";
    for (const auto& stg : stage_wgsl) {
        s += "    v = ";
        s += stg;
        s += ";\n";
    }
    s += "    C[i] = v;\n"
         "  }\n"
         "}\n";
    return s;
}

// -----------------------------------------------------------------------------
// GEMV variants — specialized for M=1 (skinny matmul). Avoids the 16x16-tile
// kernel's 15/16 wasted rows when only one output row is live. A single
// workgroup cooperates across the K dimension (split-K) and produces TILE_N
// output columns. Each workgroup_size lane accumulates TILE_N partial sums
// while striding through K, then a tree reduction in workgroup memory
// combines the partials, and the first TILE_N lanes apply the epilogue and
// write to C.
//
// Bindings and dims layout match the tile variants exactly — the same BGL /
// bind group / uniform buffer are reused; only the pipeline object differs.
//
// Dispatch (from exec()):  (ceil(N / TILE_N), 1, 1)

// 2D workgroup layout: lid.x varies across output cols (TILE_N per WG),
// lid.y splits the K dimension. This is specifically chosen so that
// subgroup-adjacent threads (x-major in lid layout) hit *adjacent* B
// columns on the same K row — so B reads are coalesced in memory.
// The prior 1D layout (WG=64, TILE_N=8 cols per thread) had adjacent
// threads reading K-strided B addresses (N floats apart), which on a
// typical iGPU costs 4× the cache-line fetches per subgroup issue.
constexpr uint32_t GEMV_TILE_N  = 8u;   // output cols per workgroup
constexpr uint32_t GEMV_K_SPLIT = 8u;   // K-reduction threads per col

namespace {

std::string gemv_prologue()
{
    std::string s;
    s += "const TILE_N  : u32 = ";
    s += std::to_string(GEMV_TILE_N);
    s += "u;\n";
    s += "const K_SPLIT : u32 = ";
    s += std::to_string(GEMV_K_SPLIT);
    s += "u;\n";
    // partial[col_local][k_idx] — we reduce along k_idx for each col_local.
    s += "var<workgroup> partial : array<array<f32, ";
    s += std::to_string(GEMV_K_SPLIT);
    s += ">, ";
    s += std::to_string(GEMV_TILE_N);
    s += ">;\n";
    return s;
}

// Common body up to the cross-k_idx reduction. Leaves partial[col_local][0]
// holding the full dot product for this WG's col_local output. Caller
// appends the epilogue (stage substitution + C write) — which runs only
// on the k_idx == 0 lanes since only they own one unique output col each.
//
// K_UNROLL=4 inside each thread's k-loop gives four independent FMA chains
// so the compiler can pipeline the dependent adds (each f32 FMA has ~4-cycle
// latency on typical iGPUs; 1 chain = 0.25 FMA/cycle, 4 chains = 1 FMA/cycle).
// The prior 1D layout got the same ILP from 8 output-col accumulators per
// thread; this layout had to re-create it along K instead since each thread
// now owns just one output col.
std::string gemv_reduce_body()
{
    return
        "@compute @workgroup_size(8, 8, 1)\n"
        "fn main(@builtin(workgroup_id)        wg  : vec3<u32>,\n"
        "        @builtin(local_invocation_id) lid : vec3<u32>) {\n"
        "  let col_local = lid.x;\n"
        "  let k_idx     = lid.y;\n"
        "  let col       = wg.x * TILE_N + col_local;\n"
        "  let N = dims.N;\n"
        "  let K = dims.K;\n"
        "  var acc0 : f32 = 0.0;\n"
        "  var acc1 : f32 = 0.0;\n"
        "  var acc2 : f32 = 0.0;\n"
        "  var acc3 : f32 = 0.0;\n"
        "  var acc4 : f32 = 0.0;\n"
        "  var acc5 : f32 = 0.0;\n"
        "  var acc6 : f32 = 0.0;\n"
        "  var acc7 : f32 = 0.0;\n"
        "  let kstep : u32 = K_SPLIT * 8u;\n"
        "  if (col < N) {\n"
        "    var kb : u32 = k_idx;\n"
        "    loop {\n"
        "      let k0 = kb;\n"
        "      let k1 = kb + K_SPLIT;\n"
        "      let k2 = kb + K_SPLIT * 2u;\n"
        "      let k3 = kb + K_SPLIT * 3u;\n"
        "      let k4 = kb + K_SPLIT * 4u;\n"
        "      let k5 = kb + K_SPLIT * 5u;\n"
        "      let k6 = kb + K_SPLIT * 6u;\n"
        "      let k7 = kb + K_SPLIT * 7u;\n"
        "      if (k7 < K) {\n"
        "        acc0 = acc0 + A[k0] * B[k0 * N + col];\n"
        "        acc1 = acc1 + A[k1] * B[k1 * N + col];\n"
        "        acc2 = acc2 + A[k2] * B[k2 * N + col];\n"
        "        acc3 = acc3 + A[k3] * B[k3 * N + col];\n"
        "        acc4 = acc4 + A[k4] * B[k4 * N + col];\n"
        "        acc5 = acc5 + A[k5] * B[k5 * N + col];\n"
        "        acc6 = acc6 + A[k6] * B[k6 * N + col];\n"
        "        acc7 = acc7 + A[k7] * B[k7 * N + col];\n"
        "      } else {\n"
        "        if (k0 < K) { acc0 = acc0 + A[k0] * B[k0 * N + col]; }\n"
        "        if (k1 < K) { acc1 = acc1 + A[k1] * B[k1 * N + col]; }\n"
        "        if (k2 < K) { acc2 = acc2 + A[k2] * B[k2 * N + col]; }\n"
        "        if (k3 < K) { acc3 = acc3 + A[k3] * B[k3 * N + col]; }\n"
        "        if (k4 < K) { acc4 = acc4 + A[k4] * B[k4 * N + col]; }\n"
        "        if (k5 < K) { acc5 = acc5 + A[k5] * B[k5 * N + col]; }\n"
        "        if (k6 < K) { acc6 = acc6 + A[k6] * B[k6 * N + col]; }\n"
        "        break;\n"
        "      }\n"
        "      kb = kb + kstep;\n"
        "    }\n"
        "  }\n"
        "  let acc = ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7));\n"
        "  partial[col_local][k_idx] = acc;\n"
        "  workgroupBarrier();\n"
        "  var stride : u32 = K_SPLIT / 2u;\n"
        "  loop {\n"
        "    if (stride == 0u) { break; }\n"
        "    if (k_idx < stride) {\n"
        "      partial[col_local][k_idx] =\n"
        "          partial[col_local][k_idx] + partial[col_local][k_idx + stride];\n"
        "    }\n"
        "    workgroupBarrier();\n"
        "    stride = stride / 2u;\n"
        "  }\n";
}

} // namespace

// Path U GEMV shader builder. Side indexing uses `S[col]` (M=1 so row=0).
std::string make_matmul_chain_wgsl_gemv_simple(const std::vector<std::string>& stage_wgsl,
                                               int n_sides)
{
    std::string s =
        "struct Dims { M : u32, N : u32, K : u32, _pad : u32 };\n"
        "@group(0) @binding(0) var<storage, read>       A : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read>       B : array<f32>;\n";
    for (int k = 0; k < n_sides; ++k) {
        s += "@group(0) @binding(";
        s += std::to_string(2 + k);
        s += ") var<storage, read>       S";
        s += std::to_string(k);
        s += " : array<f32>;\n";
    }
    const int c_binding = 2 + n_sides;
    const int d_binding = 3 + n_sides;
    s += "@group(0) @binding(";
    s += std::to_string(c_binding);
    s += ") var<storage, read_write> C : array<f32>;\n"
         "@group(0) @binding(";
    s += std::to_string(d_binding);
    s += ") var<uniform>             dims : Dims;\n";
    s += gemv_prologue();
    s += gemv_reduce_body();
    s +=
        "  if (k_idx == 0u && col < N) {\n"
        "    let i = col;\n"
        "    var v : f32 = partial[col_local][0];\n";
    for (const auto& stg : stage_wgsl) {
        s += "    v = ";
        s += stg;
        s += ";\n";
    }
    s += "    C[i] = v;\n"
         "  }\n"
         "}\n";
    return s;
}

// Path M GEMV shader builder. Each side index is `row * stride_row +
// col * stride_col`; row = 0 so only stride_col contributes.
std::string make_matmul_chain_wgsl_gemv_meta(const std::vector<std::string>& stage_wgsl,
                                             int n_sides)
{
    std::string s = "struct Meta {\n";
    for (int k = 0; k < n_sides; ++k) {
        s += "  s";
        s += std::to_string(k);
        s += "_strides : vec4<u32>,\n";
    }
    s += "};\n"
         "struct Dims { M : u32, N : u32, K : u32, _pad : u32 };\n"
         "@group(0) @binding(0) var<storage, read>       A : array<f32>;\n"
         "@group(0) @binding(1) var<storage, read>       B : array<f32>;\n";
    for (int k = 0; k < n_sides; ++k) {
        s += "@group(0) @binding(";
        s += std::to_string(2 + k);
        s += ") var<storage, read>       S";
        s += std::to_string(k);
        s += " : array<f32>;\n";
    }
    const int c_binding = 2 + n_sides;
    const int d_binding = 3 + n_sides;
    const int m_binding = 4 + n_sides;
    s += "@group(0) @binding(";
    s += std::to_string(c_binding);
    s += ") var<storage, read_write> C : array<f32>;\n"
         "@group(0) @binding(";
    s += std::to_string(d_binding);
    s += ") var<uniform>             dims : Dims;\n"
         "@group(0) @binding(";
    s += std::to_string(m_binding);
    s += ") var<storage, read>       md : Meta;\n";
    s += gemv_prologue();
    s += gemv_reduce_body();
    s +=
        "  if (k_idx == 0u && col < N) {\n"
        "    let i = col;\n";
    for (int k = 0; k < n_sides; ++k) {
        auto ks = std::to_string(k);
        s += "    let side_";
        s += ks;
        s += "_flat : u32 = col * md.s";
        s += ks;
        s += "_strides.y;\n";
    }
    s += "    var v : f32 = partial[col_local][0];\n";
    for (const auto& stg : stage_wgsl) {
        s += "    v = ";
        s += stg;
        s += ";\n";
    }
    s += "    C[i] = v;\n"
         "  }\n"
         "}\n";
    return s;
}

wgpu::BindGroupLayout make_matmul_chain_bgl(int n_sides, bool needs_meta)
{
    auto& dev = get_device();
    const int total = 4 + n_sides + (needs_meta ? 1 : 0);
    std::vector<wgpu::BindGroupLayoutEntry> e(total);
    int idx = 0;
    // A, B, sides — all ReadOnlyStorage.
    for (int k = 0; k < 2 + n_sides; ++k, ++idx) {
        e[idx].binding = idx;
        e[idx].visibility = wgpu::ShaderStage::Compute;
        e[idx].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    }
    // C — Storage.
    e[idx].binding = idx;
    e[idx].visibility = wgpu::ShaderStage::Compute;
    e[idx].buffer.type = wgpu::BufferBindingType::Storage;
    ++idx;
    // Dims — Uniform.
    e[idx].binding = idx;
    e[idx].visibility = wgpu::ShaderStage::Compute;
    e[idx].buffer.type = wgpu::BufferBindingType::Uniform;
    ++idx;
    // Path M: meta storage buffer appended.
    if (needs_meta) {
        e[idx].binding = idx;
        e[idx].visibility = wgpu::ShaderStage::Compute;
        e[idx].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        ++idx;
    }

    wgpu::BindGroupLayoutDescriptor d = {};
    d.entryCount = (uint32_t)total;
    d.entries = e.data();
    return dev.device.CreateBindGroupLayout(&d);
}

struct matmul_chain_cache_entry_t {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
};

std::unordered_map<std::string, matmul_chain_cache_entry_t>& matmul_chain_cache()
{
    static std::unordered_map<std::string, matmul_chain_cache_entry_t> cache;
    return cache;
}

} // namespace

bool matmul_fused_chain_t::init() {
    if (stage_wgsl.empty())                      return false;
    if ((int)inputs.size() != 2 + n_sides)       return false;
    if (outputs.size() != 1)                     return false;
    if (!device_ready())                         return false;
    if (needs_meta && n_sides < 1)               return false;

    std::string sig;
    sig += needs_meta ? 'M' : 'U';
    sig += std::to_string(n_sides);
    sig += '\x02';
    for (const auto& s : stage_wgsl) { sig += s; sig += '\x01'; }

    auto& cache = matmul_chain_cache();
    auto it = cache.find(sig);
    if (it == cache.end()) {
        auto src = needs_meta
            ? make_matmul_chain_wgsl_meta(stage_wgsl, n_sides)
            : make_matmul_chain_wgsl_simple(stage_wgsl, n_sides);
        auto sm  = compile(src);
        matmul_chain_cache_entry_t e;
        e.bgl      = make_matmul_chain_bgl(n_sides, needs_meta);
        e.pipeline = make_pipeline(sm, e.bgl);
        it = cache.emplace(std::move(sig), std::move(e)).first;
    }
    bgl      = it->second.bgl;
    pipeline = it->second.pipeline;

    uniforms = make_uniforms();

    if (needs_meta) {
        // Per side: 16-byte (vec4<u32>) slot. Two lanes unused per entry.
        const size_t sz = 16 * (size_t)n_sides;
        wgpu::BufferDescriptor md = {};
        md.size  = (sz + 15) & ~size_t(15);
        md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        meta_buf = get_device().device.CreateBuffer(&md);
    }
    return true;
}

bool matmul_fused_chain_t::reshape() {
    if ((int)inputs.size() != 2 + n_sides) return false;
    if (outputs.size() != 1)               return false;
    const tensor_t* a = inputs[0];
    const tensor_t* b = inputs[1];
    tensor_t*       y = outputs[0];
    if (!a || !b || !y)                    return false;
    if (a->type != NNR_DATA_TYPE_FLOAT32 || b->type != NNR_DATA_TYPE_FLOAT32) return false;

    // Two accepted shapes (mirrors the standalone MatMul's modes a/b):
    //   (a) 2D @ 2D
    //   (b) [..., base_m, K] @ [K, N] → [..., base_m, N]
    //       A's leading batch dims collapse into a single row extent; the
    //       16×16 tiled shader sees a flat [m, k] × [k, n] matmul where
    //       m = base_m * product(batch_dims). Output data lays out in
    //       row-major, so the higher-rank output shape is just a reshape
    //       of the same element sequence.
    //
    // Mode (c)/(d) — same-rank or broadcast batched matmul — would need
    // either Z-dispatch with per-batch offsets or the matmul_bcast shader;
    // out of scope here (those MatMuls in transformer fixtures are
    // consumed by Transpose, which isn't chain-absorbable anyway).
    int base_m = 0;
    int batch_prod = 1;
    int out_shape[MAX_NDIM];
    int out_ndim = 0;

    if (a->ndim == 2 && b->ndim == 2) {
        if (a->dims[1] != b->dims[0])          return false;
        base_m = a->dims[0];
        k      = a->dims[1];
        n      = b->dims[1];
        out_shape[0] = base_m;
        out_shape[1] = n;
        out_ndim = 2;
    } else if (a->ndim >= 3 && b->ndim == 2) {
        const int arank = a->ndim;
        k = a->dims[arank - 1];
        if (k != b->dims[0])                   return false;
        n = b->dims[1];
        base_m = a->dims[arank - 2];
        for (int i = 0; i < arank - 2; ++i) {
            out_shape[i]  = a->dims[i];
            batch_prod   *= a->dims[i];
        }
        out_shape[arank - 2] = base_m;
        out_shape[arank - 1] = n;
        out_ndim = arank;
    } else {
        return false;
    }

    m = base_m * batch_prod;

    if (!y->reshape(std::span<const int>(out_shape, out_ndim), NNR_DATA_TYPE_FLOAT32)) return false;

    // Validate each side against the flat [m, n] pipe. Two cases:
    //   - exact (s->ndata == m*n): side is the full output shape (any rank,
    //     row-major). Path U's `S[i] = S[row*N+col]` indexes it correctly;
    //     Path M uses stride_row=N, stride_col=1 for the same effect.
    //   - broadcast (Path M only): 1D or 2D side with right-aligned dims
    //     that are 1 or match [m, n]. 3D+ broadcast would need a richer
    //     meta layout; not required for the bias-add patterns we absorb.
    for (int kk = 0; kk < n_sides; ++kk) {
        const tensor_t* s = inputs[2 + kk];
        if (!s)                              return false;
        if (s->type != NNR_DATA_TYPE_FLOAT32) return false;
        const bool exact = ((size_t)s->ndata == (size_t)m * (size_t)n);
        if (!exact) {
            if (!needs_meta)                 return false;
            if (s->ndim > 2)                 return false;
            // Right-align side dims to [M, N]. Each side axis must be 1 or
            // equal to the pipe's corresponding axis.
            const int off = 2 - s->ndim;
            for (int i = 0; i < s->ndim; ++i) {
                int d_side = s->dims[i];
                int d_pipe = (off + i == 0) ? m : n;
                if (d_side != 1 && d_side != d_pipe) return false;
            }
        }
    }

    ensure_buffer(a, a->ndata * sizeof(float));
    auto& br = ensure_buffer(b, (size_t)k * n * sizeof(float));
    if (ctx && ctx->initializer_names.count(b->name)) br.is_weight = true;
    for (int kk = 0; kk < n_sides; ++kk)
        ensure_buffer(inputs[2 + kk], inputs[2 + kk]->ndata * sizeof(float));
    ensure_buffer(y, (size_t)m * n * sizeof(float));

    // Uniform / meta payloads are pure functions of m/n/k + side shapes,
    // all reshape-time data — write here so exec() doesn't re-queue them.
    auto& dev = get_device();
    uint32_t dims[4] = { (uint32_t)m, (uint32_t)n, (uint32_t)k, 0 };
    dev.queue.WriteBuffer(uniforms, 0, dims, sizeof(dims));

    if (needs_meta) {
        const size_t rounded = ((size_t)n_sides * 16 + 15) & ~size_t(15);
        std::vector<uint8_t> buf(rounded, 0);
        auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(buf.data() + off, &v, 4); };
        for (int kk = 0; kk < n_sides; ++kk) {
            const tensor_t* s = inputs[2 + kk];
            uint32_t stride_row = 0;
            uint32_t stride_col = 0;
            const bool exact = ((size_t)s->ndata == (size_t)m * (size_t)n);
            if (exact) {
                // Full-shape side — index as flat [row*N + col].
                stride_row = (uint32_t)n;
                stride_col = 1u;
            } else if (s->ndim == 2) {
                stride_col = (s->dims[1] != 1) ? 1u : 0u;
                stride_row = (s->dims[0] != 1) ? (uint32_t)s->dims[1] : 0u;
            } else if (s->ndim == 1) {
                stride_col = (s->dims[0] != 1) ? 1u : 0u;
                stride_row = 0;
            }
            put_u32((size_t)kk * 16 + 0, stride_row);
            put_u32((size_t)kk * 16 + 4, stride_col);
        }
        dev.queue.WriteBuffer(meta_buf, 0, buf.data(), buf.size());
    }

    // M=1 path: compile the GEMV-specialized shader once and let exec()
    // dispatch the skinny kernel instead of the 16×16 tile kernel. The
    // BGL is identical between variants (same bindings), so we reuse the
    // same cached_bg and just swap pipelines.
    is_gemv = (m == 1);
    if (is_gemv && !gemv_pipeline) {
        std::string sig;
        sig += 'G';  // distinguishes from the tile variant
        sig += needs_meta ? 'M' : 'U';
        sig += std::to_string(n_sides);
        sig += '\x02';
        for (const auto& s : stage_wgsl) { sig += s; sig += '\x01'; }

        auto& cache = matmul_chain_cache();
        auto it = cache.find(sig);
        if (it == cache.end()) {
            auto src = needs_meta
                ? make_matmul_chain_wgsl_gemv_meta(stage_wgsl, n_sides)
                : make_matmul_chain_wgsl_gemv_simple(stage_wgsl, n_sides);
            auto sm = compile(src);
            matmul_chain_cache_entry_t e;
            e.bgl      = bgl;   // reuse the BGL already built in init()
            e.pipeline = make_pipeline(sm, e.bgl);
            it = cache.emplace(std::move(sig), std::move(e)).first;
        }
        gemv_pipeline = it->second.pipeline;
    }
    return true;
}

bool matmul_fused_chain_t::exec() {
    auto& dev = get_device();
    upload_if_needed(inputs[0]);
    upload_if_needed(inputs[1]);
    for (int kk = 0; kk < n_sides; ++kk) upload_if_needed(inputs[2 + kk]);

    // uniforms / meta_buf were written in reshape().
    const int c_binding   = 2 + n_sides;
    const int d_binding   = 3 + n_sides;
    const int m_binding   = 4 + n_sides;
    const int total_binds = c_binding + 2 + (needs_meta ? 1 : 0);

    // Tensor-backed slots: A, B, sides..., Y = 3 + n_sides.
    const int n_tensor_slots = 3 + n_sides;
    uint32_t cur_gens[16] = {};
    cur_gens[0] = generation_of(inputs[0]);
    cur_gens[1] = generation_of(inputs[1]);
    for (int kk = 0; kk < n_sides; ++kk) cur_gens[2 + kk] = generation_of(inputs[2 + kk]);
    cur_gens[2 + n_sides] = generation_of(outputs[0]);
    bool bg_valid = (bool)cached_bg;
    for (int kk = 0; kk < n_tensor_slots && bg_valid; ++kk)
        if (cur_gens[kk] != cached_gen[kk]) bg_valid = false;
    if (!bg_valid) {
        std::vector<wgpu::BindGroupEntry> be(total_binds);
        auto* ra = find(inputs[0]);
        auto* rb = find(inputs[1]);
        be[0].binding = 0; be[0].buffer = ra->buf; be[0].offset = 0; be[0].size = ra->size;
        be[1].binding = 1; be[1].buffer = rb->buf; be[1].offset = 0; be[1].size = rb->size;
        for (int kk = 0; kk < n_sides; ++kk) {
            auto* rs = find(inputs[2 + kk]);
            be[2 + kk].binding = 2 + kk;
            be[2 + kk].buffer  = rs->buf;
            be[2 + kk].offset  = 0;
            be[2 + kk].size    = rs->size;
        }
        auto* ry = find(outputs[0]);
        be[c_binding].binding = c_binding;
        be[c_binding].buffer  = ry->buf;
        be[c_binding].offset  = 0;
        be[c_binding].size    = ry->size;
        be[d_binding].binding = d_binding;
        be[d_binding].buffer  = uniforms;
        be[d_binding].offset  = 0;
        be[d_binding].size    = 16;
        if (needs_meta) {
            const size_t rounded = ((size_t)n_sides * 16 + 15) & ~size_t(15);
            be[m_binding].binding = m_binding;
            be[m_binding].buffer  = meta_buf;
            be[m_binding].offset  = 0;
            be[m_binding].size    = rounded;
        }

        wgpu::BindGroupDescriptor bgd = {};
        bgd.layout     = bgl;
        bgd.entryCount = (uint32_t)total_binds;
        bgd.entries    = be.data();
        cached_bg = dev.device.CreateBindGroup(&bgd);
        for (int kk = 0; kk < n_tensor_slots; ++kk) cached_gen[kk] = cur_gens[kk];
    }

    wgpu::ComputePassEncoder pass = shared_encoder().BeginComputePass();
    if (is_gemv) {
        pass.SetPipeline(gemv_pipeline);
        pass.SetBindGroup(0, cached_bg);
        const uint32_t gx = ((uint32_t)n + GEMV_TILE_N - 1) / GEMV_TILE_N;
        pass.DispatchWorkgroups(gx, 1, 1);
    } else {
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, cached_bg);
        const uint32_t gx = ((uint32_t)n + MATMUL_TILE - 1) / MATMUL_TILE;
        const uint32_t gy = ((uint32_t)m + MATMUL_TILE - 1) / MATMUL_TILE;
        pass.DispatchWorkgroups(gx, gy, 1);
    }
    pass.End();
    mark_gpu_written(outputs[0]);
    return true;
}

// ---------------------------------------------------------------------------
// gemm_fused_chain_t — Gemm with absorbed elementwise chain epilogue
// ---------------------------------------------------------------------------
//
// Mirrors the matmul variant but adds α, β, transA, transB, and an optional
// C bias. The chain's stages run on the post-bias `v`, i.e. after
// `v = α·A·B + β·C`. A zero-filled dummy buffer is bound on the C slot when
// `has_bias` is false so the bind-group layout stays invariant (same trick
// the base Gemm shader uses).

namespace {

// Generates the full Gemm-with-chain-epilogue WGSL. transA / transB are
// baked into branches inside the tile load helpers so a single shader
// handles both orientations (matches the base Gemm approach, keeps cache
// entries sharing transform variants rather than multiplying them).
std::string make_gemm_chain_wgsl(const std::vector<std::string>& stage_wgsl,
                                 int n_sides, bool needs_meta)
{
    std::string s =
        "struct Cfg {\n"
        "  M : u32, N : u32, K : u32, transA : u32,\n"
        "  transB : u32, bias_kind : u32, _a : u32, _b : u32,\n"
        "  alpha : f32, beta : f32, _c : u32, _d : u32,\n"
        "};\n";
    if (needs_meta) {
        s += "struct Meta {\n";
        for (int k = 0; k < n_sides; ++k) {
            s += "  s";
            s += std::to_string(k);
            s += "_strides : vec4<u32>,\n";
        }
        s += "};\n";
    }
    s += "@group(0) @binding(0) var<storage, read>       A : array<f32>;\n"
         "@group(0) @binding(1) var<storage, read>       B : array<f32>;\n"
         "@group(0) @binding(2) var<storage, read>       C : array<f32>;\n";
    for (int k = 0; k < n_sides; ++k) {
        s += "@group(0) @binding(";
        s += std::to_string(3 + k);
        s += ") var<storage, read>       S";
        s += std::to_string(k);
        s += " : array<f32>;\n";
    }
    const int y_binding   = 3 + n_sides;
    const int cfg_binding = 4 + n_sides;
    const int md_binding  = 5 + n_sides;
    s += "@group(0) @binding(";
    s += std::to_string(y_binding);
    s += ") var<storage, read_write> Y : array<f32>;\n"
         "@group(0) @binding(";
    s += std::to_string(cfg_binding);
    s += ") var<uniform>             cfg : Cfg;\n";
    if (needs_meta) {
        s += "@group(0) @binding(";
        s += std::to_string(md_binding);
        s += ") var<storage, read>       md : Meta;\n";
    }
    s += "const TILE : u32 = 16u;\n"
         "var<workgroup> tA : array<array<f32, 16>, 16>;\n"
         "var<workgroup> tB : array<array<f32, 16>, 16>;\n"
         "fn load_A(r : u32, c : u32) -> f32 {\n"
         "  if (r >= cfg.M || c >= cfg.K) { return 0.0; }\n"
         "  if (cfg.transA == 0u) { return A[r * cfg.K + c]; }\n"
         "  return A[c * cfg.M + r];\n"
         "}\n"
         "fn load_B(r : u32, c : u32) -> f32 {\n"
         "  if (r >= cfg.K || c >= cfg.N) { return 0.0; }\n"
         "  if (cfg.transB == 0u) { return B[r * cfg.N + c]; }\n"
         "  return B[c * cfg.K + r];\n"
         "}\n"
         "@compute @workgroup_size(16, 16, 1)\n"
         "fn main(@builtin(global_invocation_id) gid : vec3<u32>,\n"
         "        @builtin(local_invocation_id)  lid : vec3<u32>) {\n"
         "  let row = gid.y;\n"
         "  let col = gid.x;\n"
         "  var acc : f32 = 0.0;\n"
         "  let tiles = (cfg.K + TILE - 1u) / TILE;\n"
         "  for (var t : u32 = 0u; t < tiles; t = t + 1u) {\n"
         "    tA[lid.y][lid.x] = load_A(row, t * TILE + lid.x);\n"
         "    tB[lid.y][lid.x] = load_B(t * TILE + lid.y, col);\n"
         "    workgroupBarrier();\n"
         "    for (var kk : u32 = 0u; kk < TILE; kk = kk + 1u) {\n"
         "      acc = acc + tA[lid.y][kk] * tB[kk][lid.x];\n"
         "    }\n"
         "    workgroupBarrier();\n"
         "  }\n"
         "  if (row < cfg.M && col < cfg.N) {\n"
         "    let i = row * cfg.N + col;\n"
         "    var v : f32 = cfg.alpha * acc;\n"
         "    if (cfg.bias_kind == 1u) {\n"
         "      v = v + cfg.beta * C[col];\n"
         "    } else if (cfg.bias_kind == 2u) {\n"
         "      v = v + cfg.beta * C[i];\n"
         "    }\n";
    if (needs_meta) {
        for (int k = 0; k < n_sides; ++k) {
            auto ks = std::to_string(k);
            s += "    let side_";
            s += ks;
            s += "_flat : u32 = row * md.s";
            s += ks;
            s += "_strides.x + col * md.s";
            s += ks;
            s += "_strides.y;\n";
        }
    }
    for (const auto& stg : stage_wgsl) {
        s += "    v = ";
        s += stg;
        s += ";\n";
    }
    s += "    Y[i] = v;\n"
         "  }\n"
         "}\n";
    return s;
}

wgpu::BindGroupLayout make_gemm_chain_bgl(int n_sides, bool needs_meta)
{
    auto& dev = get_device();
    const int total = 5 + n_sides + (needs_meta ? 1 : 0);
    std::vector<wgpu::BindGroupLayoutEntry> e(total);
    int idx = 0;
    // A, B, C, sides — all ReadOnlyStorage.
    for (int k = 0; k < 3 + n_sides; ++k, ++idx) {
        e[idx].binding = idx;
        e[idx].visibility = wgpu::ShaderStage::Compute;
        e[idx].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    }
    // Y — Storage.
    e[idx].binding = idx;
    e[idx].visibility = wgpu::ShaderStage::Compute;
    e[idx].buffer.type = wgpu::BufferBindingType::Storage;
    ++idx;
    // cfg — Uniform.
    e[idx].binding = idx;
    e[idx].visibility = wgpu::ShaderStage::Compute;
    e[idx].buffer.type = wgpu::BufferBindingType::Uniform;
    ++idx;
    if (needs_meta) {
        e[idx].binding = idx;
        e[idx].visibility = wgpu::ShaderStage::Compute;
        e[idx].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        ++idx;
    }
    wgpu::BindGroupLayoutDescriptor d = {};
    d.entryCount = (uint32_t)total;
    d.entries = e.data();
    return dev.device.CreateBindGroupLayout(&d);
}

struct gemm_chain_cache_entry_t {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
};

std::unordered_map<std::string, gemm_chain_cache_entry_t>& gemm_chain_cache()
{
    static std::unordered_map<std::string, gemm_chain_cache_entry_t> cache;
    return cache;
}

} // namespace

bool gemm_fused_chain_t::init() {
    if (stage_wgsl.empty())                  return false;
    if (!device_ready())                     return false;
    if (needs_meta && n_sides < 1)           return false;

    alpha    = attribute(attr_key_t::alpha,  1.0f);
    beta     = attribute(attr_key_t::beta,   1.0f);
    transA   = attribute(attr_key_t::transA, (int32_t)0);
    transB   = attribute(attr_key_t::transB, (int32_t)0);

    const int required = 2 + (has_bias ? 1 : 0) + n_sides;
    if ((int)inputs.size() != required)      return false;
    if (outputs.size() != 1)                 return false;

    // Cache shader variants by (mode, n_sides, stage list). transA / transB
    // are runtime-branched inside the shader (matches base Gemm), so they
    // don't split the cache. Bias presence changes shader behavior only via
    // cfg.bias_kind at runtime, so it also shares one shader variant.
    std::string sig;
    sig += needs_meta ? 'M' : 'U';
    sig += std::to_string(n_sides);
    sig += '\x02';
    for (const auto& s : stage_wgsl) { sig += s; sig += '\x01'; }

    auto& cache = gemm_chain_cache();
    auto it = cache.find(sig);
    if (it == cache.end()) {
        auto src = make_gemm_chain_wgsl(stage_wgsl, n_sides, needs_meta);
        auto sm  = compile(src);
        gemm_chain_cache_entry_t e;
        e.bgl      = make_gemm_chain_bgl(n_sides, needs_meta);
        e.pipeline = make_pipeline(sm, e.bgl);
        it = cache.emplace(std::move(sig), std::move(e)).first;
    }
    bgl      = it->second.bgl;
    pipeline = it->second.pipeline;

    wgpu::BufferDescriptor cd = {};
    cd.size  = 48;
    cd.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    cfg_buf  = get_device().device.CreateBuffer(&cd);

    if (needs_meta) {
        const size_t sz = 16 * (size_t)n_sides;
        wgpu::BufferDescriptor md = {};
        md.size  = (sz + 15) & ~size_t(15);
        md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        meta_buf = get_device().device.CreateBuffer(&md);
    }
    return true;
}

bool gemm_fused_chain_t::reshape() {
    const int required = 2 + (has_bias ? 1 : 0) + n_sides;
    if ((int)inputs.size() != required) return false;
    if (outputs.size() != 1)            return false;
    const tensor_t* a = inputs[0];
    const tensor_t* b = inputs[1];
    const tensor_t* c = has_bias ? inputs[2] : nullptr;
    tensor_t*       y = outputs[0];
    if (!a || !b || !y)                                   return false;
    if (a->ndim != 2 || b->ndim != 2)                     return false;
    if (a->type != NNR_DATA_TYPE_FLOAT32 || b->type != NNR_DATA_TYPE_FLOAT32) return false;

    int aM = transA ? a->dims[1] : a->dims[0];
    int aK = transA ? a->dims[0] : a->dims[1];
    int bK = transB ? b->dims[1] : b->dims[0];
    int bN = transB ? b->dims[0] : b->dims[1];
    if (aK != bK)                                         return false;
    m = aM; k = aK; n = bN;

    bias_kind = 0;
    if (c) {
        if (c->type != NNR_DATA_TYPE_FLOAT32)             return false;
        if (c->ndim == 1 && c->dims[0] == n)              bias_kind = 1;
        else if (c->ndim == 2 && c->dims[0] == m && c->dims[1] == n) bias_kind = 2;
        else return false;  // other broadcasts not supported; fusion aborts
    }

    int ydims[2] = { m, n };
    if (!y->reshape(std::span<const int>(ydims, 2), NNR_DATA_TYPE_FLOAT32)) return false;

    // Side offset in inputs[] = 2 + (has_bias ? 1 : 0).
    const int side_off = 2 + (has_bias ? 1 : 0);
    for (int kk = 0; kk < n_sides; ++kk) {
        const tensor_t* s = inputs[side_off + kk];
        if (!s)                                           return false;
        if (s->type != NNR_DATA_TYPE_FLOAT32)             return false;
        if (s->ndim > 2)                                  return false;
        bool exact = (s->ndim == 2 && s->dims[0] == m && s->dims[1] == n);
        if (!exact) {
            if (!needs_meta)                              return false;
            const int off = 2 - s->ndim;
            for (int i = 0; i < s->ndim; ++i) {
                int d_side = s->dims[i];
                int d_pipe = (off + i == 0) ? m : n;
                if (d_side != 1 && d_side != d_pipe)      return false;
            }
        }
    }

    ensure_buffer(a, a->ndata * sizeof(float));
    auto& br = ensure_buffer(b, b->ndata * sizeof(float));
    if (ctx && ctx->initializer_names.count(b->name)) br.is_weight = true;
    if (c) {
        auto& cr = ensure_buffer(c, c->ndata * sizeof(float));
        if (ctx && ctx->initializer_names.count(c->name)) cr.is_weight = true;
    }
    for (int kk = 0; kk < n_sides; ++kk)
        ensure_buffer(inputs[side_off + kk],
                      inputs[side_off + kk]->ndata * sizeof(float));
    ensure_buffer(y, y->ndata * sizeof(float));

    // Allocate the zero-fill C stand-in only when there's no real bias. Size
    // must be >= one f32 so WGSL `C[col]` / `C[i]` reads are well-defined
    // (the shader never takes the bias branch when bias_kind == 0, but the
    // driver still wants a valid binding).
    if (!has_bias) {
        auto& dev = get_device();
        wgpu::BufferDescriptor bd = {};
        bd.size  = 16;
        bd.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        zero_bias = dev.device.CreateBuffer(&bd);
        float zeros[4] = {};
        dev.queue.WriteBuffer(zero_bias, 0, zeros, sizeof(zeros));
    }

    // Cfg + meta payloads are pure functions of m/n/k/attrs/side shapes.
    auto& dev = get_device();
    uint8_t cfgbuf[48] = {};
    auto put_u = [&](size_t off, uint32_t v) { std::memcpy(cfgbuf + off, &v, 4); };
    auto put_f = [&](size_t off, float v)    { std::memcpy(cfgbuf + off, &v, 4); };
    put_u(0,  (uint32_t)m);
    put_u(4,  (uint32_t)n);
    put_u(8,  (uint32_t)k);
    put_u(12, (uint32_t)transA);
    put_u(16, (uint32_t)transB);
    put_u(20, (uint32_t)bias_kind);
    put_f(32, alpha);
    put_f(36, beta);
    dev.queue.WriteBuffer(cfg_buf, 0, cfgbuf, sizeof(cfgbuf));

    if (needs_meta) {
        const int side_off = 2 + (has_bias ? 1 : 0);
        const size_t rounded = ((size_t)n_sides * 16 + 15) & ~size_t(15);
        std::vector<uint8_t> mbuf(rounded, 0);
        auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(mbuf.data() + off, &v, 4); };
        for (int kk = 0; kk < n_sides; ++kk) {
            const tensor_t* s = inputs[side_off + kk];
            uint32_t stride_row = 0;
            uint32_t stride_col = 0;
            if (s->ndim == 2) {
                stride_col = (s->dims[1] != 1) ? 1u : 0u;
                stride_row = (s->dims[0] != 1) ? (uint32_t)s->dims[1] : 0u;
            } else if (s->ndim == 1) {
                stride_col = (s->dims[0] != 1) ? 1u : 0u;
                stride_row = 0;
            }
            put_u32((size_t)kk * 16 + 0, stride_row);
            put_u32((size_t)kk * 16 + 4, stride_col);
        }
        dev.queue.WriteBuffer(meta_buf, 0, mbuf.data(), mbuf.size());
    }
    return true;
}

bool gemm_fused_chain_t::exec() {
    auto& dev = get_device();
    upload_if_needed(inputs[0]);
    upload_if_needed(inputs[1]);
    if (has_bias) upload_if_needed(inputs[2]);
    const int side_off = 2 + (has_bias ? 1 : 0);
    for (int kk = 0; kk < n_sides; ++kk) upload_if_needed(inputs[side_off + kk]);

    // cfg_buf and meta_buf were written in reshape().
    auto* ra = find(inputs[0]);
    auto* rb = find(inputs[1]);
    wgpu::Buffer c_buf; uint64_t c_sz;
    if (has_bias) { auto* rc = find(inputs[2]); c_buf = rc->buf; c_sz = rc->size; }
    else          { c_buf = zero_bias;                          c_sz = 16;       }

    const int y_binding   = 3 + n_sides;
    const int cfg_binding = 4 + n_sides;
    const int md_binding  = 5 + n_sides;
    const int total_binds = y_binding + 2 + (needs_meta ? 1 : 0);

    // Tensor-backed slots: A, B, C|0, sides..., Y = 4 + n_sides. When
    // has_bias is false, slot 2 tracks the op-owned zero_bias via gen=0
    // (stable since the op's lifetime).
    const int n_tensor_slots = 4 + n_sides;
    uint32_t cur_gens[16] = {};
    cur_gens[0] = generation_of(inputs[0]);
    cur_gens[1] = generation_of(inputs[1]);
    cur_gens[2] = has_bias ? generation_of(inputs[2]) : 0u;
    for (int kk = 0; kk < n_sides; ++kk) cur_gens[3 + kk] = generation_of(inputs[side_off + kk]);
    cur_gens[3 + n_sides] = generation_of(outputs[0]);
    bool bg_valid = (bool)cached_bg;
    for (int kk = 0; kk < n_tensor_slots && bg_valid; ++kk)
        if (cur_gens[kk] != cached_gen[kk]) bg_valid = false;
    if (!bg_valid) {
        std::vector<wgpu::BindGroupEntry> be(total_binds);
        be[0].binding = 0; be[0].buffer = ra->buf; be[0].offset = 0; be[0].size = ra->size;
        be[1].binding = 1; be[1].buffer = rb->buf; be[1].offset = 0; be[1].size = rb->size;
        be[2].binding = 2; be[2].buffer = c_buf;   be[2].offset = 0; be[2].size = c_sz;
        for (int kk = 0; kk < n_sides; ++kk) {
            auto* rs = find(inputs[side_off + kk]);
            be[3 + kk].binding = 3 + kk;
            be[3 + kk].buffer  = rs->buf;
            be[3 + kk].offset  = 0;
            be[3 + kk].size    = rs->size;
        }
        auto* ry = find(outputs[0]);
        be[y_binding].binding = y_binding;
        be[y_binding].buffer  = ry->buf;
        be[y_binding].offset  = 0;
        be[y_binding].size    = ry->size;
        be[cfg_binding].binding = cfg_binding;
        be[cfg_binding].buffer  = cfg_buf;
        be[cfg_binding].offset  = 0;
        be[cfg_binding].size    = 48;
        if (needs_meta) {
            const size_t rounded = ((size_t)n_sides * 16 + 15) & ~size_t(15);
            be[md_binding].binding = md_binding;
            be[md_binding].buffer  = meta_buf;
            be[md_binding].offset  = 0;
            be[md_binding].size    = rounded;
        }

        wgpu::BindGroupDescriptor bgd = {};
        bgd.layout     = bgl;
        bgd.entryCount = (uint32_t)total_binds;
        bgd.entries    = be.data();
        cached_bg = dev.device.CreateBindGroup(&bgd);
        for (int kk = 0; kk < n_tensor_slots; ++kk) cached_gen[kk] = cur_gens[kk];
    }

    wgpu::ComputePassEncoder pass = shared_encoder().BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, cached_bg);
    const uint32_t TILE = 16;
    pass.DispatchWorkgroups(((uint32_t)n + TILE - 1) / TILE,
                            ((uint32_t)m + TILE - 1) / TILE, 1);
    pass.End();
    mark_gpu_written(outputs[0]);
    return true;
}

// ---------------------------------------------------------------------------
// conv_fused_chain_t — 2D NCHW Conv with absorbed elementwise chain
// ---------------------------------------------------------------------------
//
// Splices the chain's `stage_wgsl` into the Conv kernel's output epilogue
// (after the bias-add, before the final write). Same convention as the
// matmul/gemm variants: Path U references `S<k>[i]` using the linear
// output index, Path M references `S<k>[side_<k>_flat]` computed on the
// fly from the unflattened `(n, c_out, h_out, w_out)` coord and a per-side
// `(stride_n, stride_c, stride_h, stride_w)` packed in a second storage
// buffer.

namespace {

// Conv shader builder. transA / transB / groups / dilations / auto_pad / pad
// are all handled by the runtime meta buffer, same as the base Conv op, so
// variants don't split the pipeline cache. The only parameters that change
// shader text are n_sides, needs_meta, and the stage list — the same cache
// key the other fused ops use.
std::string make_conv_chain_wgsl(const std::vector<std::string>& stage_wgsl,
                                 int n_sides, bool needs_meta)
{
    std::string s =
        "struct Meta {\n"
        "  total       : u32,\n"
        "  N           : u32, C_out : u32, H_out : u32, W_out : u32,\n"
        "  C_in        : u32, groups : u32,\n"
        "  kH          : u32, kW    : u32,\n"
        "  stride_h    : u32, stride_w : u32,\n"
        "  pad_top     : u32, pad_left : u32,\n"
        "  dilation_h  : u32, dilation_w : u32,\n"
        "  H           : u32, W : u32,\n"
        "  has_bias    : u32,\n"
        "};\n";
    if (needs_meta) {
        s += "struct SideMeta {\n";
        for (int k = 0; k < n_sides; ++k) {
            s += "  s";
            s += std::to_string(k);
            s += "_strides : vec4<u32>,\n";  // (stride_n, stride_c, stride_h, stride_w)
        }
        s += "};\n";
    }
    s += "@group(0) @binding(0) var<storage, read>       X  : array<f32>;\n"
         "@group(0) @binding(1) var<storage, read>       Wt : array<f32>;\n"
         "@group(0) @binding(2) var<storage, read>       Bi : array<f32>;\n";
    for (int k = 0; k < n_sides; ++k) {
        s += "@group(0) @binding(";
        s += std::to_string(3 + k);
        s += ") var<storage, read>       S";
        s += std::to_string(k);
        s += " : array<f32>;\n";
    }
    const int y_binding  = 3 + n_sides;
    const int md_binding = 4 + n_sides;
    const int sd_binding = 5 + n_sides;
    s += "@group(0) @binding(";
    s += std::to_string(y_binding);
    s += ") var<storage, read_write> Y  : array<f32>;\n"
         "@group(0) @binding(";
    s += std::to_string(md_binding);
    s += ") var<storage, read>       md : Meta;\n";
    if (needs_meta) {
        s += "@group(0) @binding(";
        s += std::to_string(sd_binding);
        s += ") var<storage, read>       sd : SideMeta;\n";
    }
    s += "@compute @workgroup_size(64)\n"
         "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
         "  let i = gid.x;\n"
         "  if (i >= md.total) { return; }\n"
         "  let ow = i % md.W_out;\n"
         "  var tmp = i / md.W_out;\n"
         "  let oh = tmp % md.H_out;\n"
         "  tmp = tmp / md.H_out;\n"
         "  let m = tmp % md.C_out;\n"
         "  let n = tmp / md.C_out;\n"
         "  let M_per_group    = md.C_out / md.groups;\n"
         "  let C_in_per_group = md.C_in  / md.groups;\n"
         "  let g = m / M_per_group;\n"
         "  let x_batch_stride  = md.C_in * md.H * md.W;\n"
         "  let x_chan_stride   = md.H * md.W;\n"
         "  let w_outch_stride  = C_in_per_group * md.kH * md.kW;\n"
         "  let w_inch_stride   = md.kH * md.kW;\n"
         "  let base_x = n * x_batch_stride;\n"
         "  let base_w = m * w_outch_stride;\n"
         "  var acc : f32 = 0.0;\n"
         "  for (var ic : u32 = 0u; ic < C_in_per_group; ic = ic + 1u) {\n"
         "    let c = g * C_in_per_group + ic;\n"
         "    let x_c_base = base_x + c * x_chan_stride;\n"
         "    let w_c_base = base_w + ic * w_inch_stride;\n"
         "    for (var kh : u32 = 0u; kh < md.kH; kh = kh + 1u) {\n"
         "      let ih_i = i32(oh * md.stride_h) + i32(kh * md.dilation_h) - i32(md.pad_top);\n"
         "      if (ih_i < 0 || ih_i >= i32(md.H)) { continue; }\n"
         "      let ih = u32(ih_i);\n"
         "      let x_h_base = x_c_base + ih * md.W;\n"
         "      let w_h_base = w_c_base + kh * md.kW;\n"
         "      for (var kw : u32 = 0u; kw < md.kW; kw = kw + 1u) {\n"
         "        let iw_i = i32(ow * md.stride_w) + i32(kw * md.dilation_w) - i32(md.pad_left);\n"
         "        if (iw_i < 0 || iw_i >= i32(md.W)) { continue; }\n"
         "        let iw = u32(iw_i);\n"
         "        acc = acc + X[x_h_base + iw] * Wt[w_h_base + kw];\n"
         "      }\n"
         "    }\n"
         "  }\n"
         "  if (md.has_bias != 0u) { acc = acc + Bi[m]; }\n"
         "  var v : f32 = acc;\n";
    if (needs_meta) {
        for (int k = 0; k < n_sides; ++k) {
            auto ks = std::to_string(k);
            s += "  let side_";
            s += ks;
            s += "_flat : u32 = n * sd.s";
            s += ks;
            s += "_strides.x + m * sd.s";
            s += ks;
            s += "_strides.y + oh * sd.s";
            s += ks;
            s += "_strides.z + ow * sd.s";
            s += ks;
            s += "_strides.w;\n";
        }
    }
    for (const auto& stg : stage_wgsl) {
        s += "  v = ";
        s += stg;
        s += ";\n";
    }
    s += "  Y[i] = v;\n"
         "}\n";
    return s;
}

// -----------------------------------------------------------------------------
// Tiled (register-tiled + shared-memory weight cache) variant of the fused
// Conv chain. Mirrors `conv_tiled.wgsl`: each workgroup owns an 8×8 output
// tile for a single (n, m) pair; 64 threads cooperatively load
// W[m, :, :, :] into a 16 KB shared-memory scratchpad once, then reuse it
// across the whole tile. Cuts weight reads by ~64× versus the plain kernel
// where every thread re-reads its own weights from global memory.
//
// Bindings and meta layout are identical to the plain variant so the same
// BGL / bind group / meta buffer / side meta can back either pipeline.
//
// Dispatch (from exec()): (ceil(W_out/8), ceil(H_out/8), N * C_out).
// Required precondition: C_in_per_group * kH * kW ≤ MAX_TILED_W. Enforced
// at reshape() time.
//
// 4096 matches the stock Conv's tiled variant (conv_tiled.wgsl). Empirical
// note: on cnn_bench-sized convs (w_size ∈ {27, 288, 576}) the tiled path
// is a net loss at both 4096 and 1024 array sizes — per-op timing shows
// a ~10% drop but wall-clock overhead between dispatches grows more than
// it saves. Left behind NNR_WEBGPU_TILED_CONV_CHAIN=1 for workloads
// with larger weight tiles where the tradeoff may invert.
constexpr uint32_t CONV_CHAIN_MAX_TILED_W = 4096u;

std::string make_conv_chain_wgsl_tiled(const std::vector<std::string>& stage_wgsl,
                                       int n_sides, bool needs_meta)
{
    std::string s =
        "struct Meta {\n"
        "  total       : u32,\n"
        "  N           : u32, C_out : u32, H_out : u32, W_out : u32,\n"
        "  C_in        : u32, groups : u32,\n"
        "  kH          : u32, kW    : u32,\n"
        "  stride_h    : u32, stride_w : u32,\n"
        "  pad_top     : u32, pad_left : u32,\n"
        "  dilation_h  : u32, dilation_w : u32,\n"
        "  H           : u32, W : u32,\n"
        "  has_bias    : u32,\n"
        "};\n";
    if (needs_meta) {
        s += "struct SideMeta {\n";
        for (int k = 0; k < n_sides; ++k) {
            s += "  s";
            s += std::to_string(k);
            s += "_strides : vec4<u32>,\n";
        }
        s += "};\n";
    }
    s += "@group(0) @binding(0) var<storage, read>       X  : array<f32>;\n"
         "@group(0) @binding(1) var<storage, read>       Wt : array<f32>;\n"
         "@group(0) @binding(2) var<storage, read>       Bi : array<f32>;\n";
    for (int k = 0; k < n_sides; ++k) {
        s += "@group(0) @binding(";
        s += std::to_string(3 + k);
        s += ") var<storage, read>       S";
        s += std::to_string(k);
        s += " : array<f32>;\n";
    }
    const int y_binding  = 3 + n_sides;
    const int md_binding = 4 + n_sides;
    const int sd_binding = 5 + n_sides;
    s += "@group(0) @binding(";
    s += std::to_string(y_binding);
    s += ") var<storage, read_write> Y  : array<f32>;\n"
         "@group(0) @binding(";
    s += std::to_string(md_binding);
    s += ") var<storage, read>       md : Meta;\n";
    if (needs_meta) {
        s += "@group(0) @binding(";
        s += std::to_string(sd_binding);
        s += ") var<storage, read>       sd : SideMeta;\n";
    }
    s +=
        "const WG_X : u32 = 8u;\n"
        "const WG_Y : u32 = 8u;\n"
        "const WG_SIZE : u32 = 64u;\n"
        "const MAX_WEIGHTS : u32 = 4096u;\n"
        "var<workgroup> sh_w : array<f32, 4096>;\n"
        "@compute @workgroup_size(8, 8, 1)\n"
        "fn main(@builtin(workgroup_id)         wid : vec3<u32>,\n"
        "        @builtin(local_invocation_id)  lid : vec3<u32>) {\n"
        "  let nm = wid.z;\n"
        "  let n  = nm / md.C_out;\n"
        "  let m  = nm % md.C_out;\n"
        "  let M_per_group    = md.C_out / md.groups;\n"
        "  let C_in_per_group = md.C_in  / md.groups;\n"
        "  let g              = m / M_per_group;\n"
        "  let w_size = C_in_per_group * md.kH * md.kW;\n"
        "  let w_base = m * w_size;\n"
        "  let tid = lid.y * WG_X + lid.x;\n"
        "  for (var wi : u32 = tid; wi < w_size; wi = wi + WG_SIZE) {\n"
        "    sh_w[wi] = Wt[w_base + wi];\n"
        "  }\n"
        "  workgroupBarrier();\n"
        "  let oh = wid.y * WG_Y + lid.y;\n"
        "  let ow = wid.x * WG_X + lid.x;\n"
        "  if (oh >= md.H_out || ow >= md.W_out) { return; }\n"
        "  let c_start         = g * C_in_per_group;\n"
        "  let x_batch_stride  = md.C_in * md.H * md.W;\n"
        "  let x_chan_stride   = md.H * md.W;\n"
        "  let base_x          = n * x_batch_stride;\n"
        "  var acc : f32 = 0.0;\n"
        "  for (var ic : u32 = 0u; ic < C_in_per_group; ic = ic + 1u) {\n"
        "    let c = c_start + ic;\n"
        "    let x_c_base = base_x + c * x_chan_stride;\n"
        "    let w_c_base = ic * md.kH * md.kW;\n"
        "    for (var kh : u32 = 0u; kh < md.kH; kh = kh + 1u) {\n"
        "      let ih_i = i32(oh * md.stride_h) + i32(kh * md.dilation_h) - i32(md.pad_top);\n"
        "      if (ih_i < 0 || ih_i >= i32(md.H)) { continue; }\n"
        "      let ih = u32(ih_i);\n"
        "      let x_h_base = x_c_base + ih * md.W;\n"
        "      let w_h_base = w_c_base + kh * md.kW;\n"
        "      for (var kw : u32 = 0u; kw < md.kW; kw = kw + 1u) {\n"
        "        let iw_i = i32(ow * md.stride_w) + i32(kw * md.dilation_w) - i32(md.pad_left);\n"
        "        if (iw_i < 0 || iw_i >= i32(md.W)) { continue; }\n"
        "        let iw = u32(iw_i);\n"
        "        acc = acc + X[x_h_base + iw] * sh_w[w_h_base + kw];\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "  if (md.has_bias != 0u) { acc = acc + Bi[m]; }\n"
        "  let i = ((n * md.C_out + m) * md.H_out + oh) * md.W_out + ow;\n"
        "  var v : f32 = acc;\n";
    if (needs_meta) {
        for (int k = 0; k < n_sides; ++k) {
            auto ks = std::to_string(k);
            s += "  let side_";
            s += ks;
            s += "_flat : u32 = n * sd.s";
            s += ks;
            s += "_strides.x + m * sd.s";
            s += ks;
            s += "_strides.y + oh * sd.s";
            s += ks;
            s += "_strides.z + ow * sd.s";
            s += ks;
            s += "_strides.w;\n";
        }
    }
    for (const auto& stg : stage_wgsl) {
        s += "  v = ";
        s += stg;
        s += ";\n";
    }
    s += "  Y[i] = v;\n"
         "}\n";
    return s;
}

// -----------------------------------------------------------------------------
// Register-tiled variant: each thread computes TILE_W = 4 consecutive output
// columns for the same (n, m, oh). Weight-per-inner-loop-iteration is read
// once and reused across all 4 outputs; X reads overlap for adjacent ows so
// when stride_w=dilation_w=1 the 4 outputs share (TILE_W + kW - 1) unique X
// reads per (ic, kh, kw) instead of TILE_W × kW. Expected payoff: up to 4×
// fewer W reads and ~2× fewer X reads, which is the right lever for
// memory-bound small-channel first-layer convs like cnn_bench's c1.
//
// Dispatch (from exec()): (ceil(W_out/(8*4)), ceil(H_out/8), N * C_out).
// workgroup_size (8, 8, 1) = 64 threads.
//
// Bindings and meta layout are identical to the plain variant so the same
// BGL / bind group / meta buffer / side meta back either pipeline.
constexpr uint32_t CONV_CHAIN_REG_TILE_W = 4u;
constexpr uint32_t CONV_CHAIN_REG_WGX    = 8u;
constexpr uint32_t CONV_CHAIN_REG_WGY    = 8u;

// Wider tile variant (8 output cols per thread) for wide output fixtures.
// Same structure as the TILE_W=4 variant but with 8 accumulators; used when
// W_out is a multiple of (WGX * 8) so no threads have all-invalid outputs.
constexpr uint32_t CONV_CHAIN_REG_TILE_W_WIDE = 8u;

// 2D output-tile variant: 2 oh rows × 4 ow cols = 8 outputs per thread. The
// weight Wt[m, ic, kh, kw] is read once per (ic, kh, kw) inner iteration
// and reused across all 8 FMAs (vs 4 in the 1-row regtile). X row reads
// are also shared: kH=3 over 2 ohs touches 4 unique ih values vs 6 for
// two independent threads. Intended for memory-bound first-layer convs
// (iC small relative to kH*kW).
constexpr uint32_t CONV_CHAIN_REG_TILE_W_2D = 4u;
constexpr uint32_t CONV_CHAIN_REG_TILE_H_2D = 2u;

std::string make_conv_chain_wgsl_regtile_wide(const std::vector<std::string>& stage_wgsl,
                                              int n_sides, bool needs_meta);
std::string make_conv_chain_wgsl_regtile_2d(const std::vector<std::string>& stage_wgsl,
                                            int n_sides, bool needs_meta);
std::string make_conv_chain_wgsl_regtile_3x3s1(const std::vector<std::string>& stage_wgsl,
                                               int n_sides, bool needs_meta);

std::string make_conv_chain_wgsl_regtile(const std::vector<std::string>& stage_wgsl,
                                         int n_sides, bool needs_meta)
{
    std::string s =
        "struct Meta {\n"
        "  total       : u32,\n"
        "  N           : u32, C_out : u32, H_out : u32, W_out : u32,\n"
        "  C_in        : u32, groups : u32,\n"
        "  kH          : u32, kW    : u32,\n"
        "  stride_h    : u32, stride_w : u32,\n"
        "  pad_top     : u32, pad_left : u32,\n"
        "  dilation_h  : u32, dilation_w : u32,\n"
        "  H           : u32, W : u32,\n"
        "  has_bias    : u32,\n"
        "};\n";
    if (needs_meta) {
        s += "struct SideMeta {\n";
        for (int k = 0; k < n_sides; ++k) {
            s += "  s";
            s += std::to_string(k);
            s += "_strides : vec4<u32>,\n";
        }
        s += "};\n";
    }
    s += "@group(0) @binding(0) var<storage, read>       X  : array<f32>;\n"
         "@group(0) @binding(1) var<storage, read>       Wt : array<f32>;\n"
         "@group(0) @binding(2) var<storage, read>       Bi : array<f32>;\n";
    for (int k = 0; k < n_sides; ++k) {
        s += "@group(0) @binding(";
        s += std::to_string(3 + k);
        s += ") var<storage, read>       S";
        s += std::to_string(k);
        s += " : array<f32>;\n";
    }
    const int y_binding  = 3 + n_sides;
    const int md_binding = 4 + n_sides;
    const int sd_binding = 5 + n_sides;
    s += "@group(0) @binding(";
    s += std::to_string(y_binding);
    s += ") var<storage, read_write> Y  : array<f32>;\n"
         "@group(0) @binding(";
    s += std::to_string(md_binding);
    s += ") var<storage, read>       md : Meta;\n";
    if (needs_meta) {
        s += "@group(0) @binding(";
        s += std::to_string(sd_binding);
        s += ") var<storage, read>       sd : SideMeta;\n";
    }
    s +=
        "@compute @workgroup_size(8, 8, 1)\n"
        "fn main(@builtin(workgroup_id)         wid : vec3<u32>,\n"
        "        @builtin(local_invocation_id)  lid : vec3<u32>) {\n"
        "  let nm = wid.z;\n"
        "  let n  = nm / md.C_out;\n"
        "  let m  = nm % md.C_out;\n"
        "  let oh = wid.y * 8u + lid.y;\n"
        "  let ow_base = (wid.x * 8u + lid.x) * 4u;\n"
        "  if (oh >= md.H_out) { return; }\n"
        "  let M_per_group    = md.C_out / md.groups;\n"
        "  let C_in_per_group = md.C_in  / md.groups;\n"
        "  let g              = m / M_per_group;\n"
        "  let c_start         = g * C_in_per_group;\n"
        "  let x_batch_stride  = md.C_in * md.H * md.W;\n"
        "  let x_chan_stride   = md.H * md.W;\n"
        "  let w_outch_stride  = C_in_per_group * md.kH * md.kW;\n"
        "  let w_inch_stride   = md.kH * md.kW;\n"
        "  let base_x = n * x_batch_stride;\n"
        "  let base_w = m * w_outch_stride;\n"
        "  // Validity masks for 4 output cols — loop-invariant for this thread.\n"
        "  let ow0 = ow_base + 0u;\n"
        "  let ow1 = ow_base + 1u;\n"
        "  let ow2 = ow_base + 2u;\n"
        "  let ow3 = ow_base + 3u;\n"
        "  let ok0 = ow0 < md.W_out;\n"
        "  let ok1 = ow1 < md.W_out;\n"
        "  let ok2 = ow2 < md.W_out;\n"
        "  let ok3 = ow3 < md.W_out;\n"
        "  var acc0 : f32 = 0.0;\n"
        "  var acc1 : f32 = 0.0;\n"
        "  var acc2 : f32 = 0.0;\n"
        "  var acc3 : f32 = 0.0;\n"
        "  for (var ic : u32 = 0u; ic < C_in_per_group; ic = ic + 1u) {\n"
        "    let c = c_start + ic;\n"
        "    let x_c_base = base_x + c * x_chan_stride;\n"
        "    let w_c_base = base_w + ic * w_inch_stride;\n"
        "    for (var kh : u32 = 0u; kh < md.kH; kh = kh + 1u) {\n"
        "      let ih_i = i32(oh * md.stride_h) + i32(kh * md.dilation_h) - i32(md.pad_top);\n"
        "      if (ih_i < 0 || ih_i >= i32(md.H)) { continue; }\n"
        "      let ih = u32(ih_i);\n"
        "      let x_h_base = x_c_base + ih * md.W;\n"
        "      let w_h_base = w_c_base + kh * md.kW;\n"
        "      for (var kw : u32 = 0u; kw < md.kW; kw = kw + 1u) {\n"
        "        let w = Wt[w_h_base + kw];\n"
        "        let kw_dw = i32(kw * md.dilation_w) - i32(md.pad_left);\n"
        "        let iw0_i = i32(ow0 * md.stride_w) + kw_dw;\n"
        "        let iw1_i = i32(ow1 * md.stride_w) + kw_dw;\n"
        "        let iw2_i = i32(ow2 * md.stride_w) + kw_dw;\n"
        "        let iw3_i = i32(ow3 * md.stride_w) + kw_dw;\n"
        "        if (ok0 && iw0_i >= 0 && iw0_i < i32(md.W)) { acc0 = acc0 + X[x_h_base + u32(iw0_i)] * w; }\n"
        "        if (ok1 && iw1_i >= 0 && iw1_i < i32(md.W)) { acc1 = acc1 + X[x_h_base + u32(iw1_i)] * w; }\n"
        "        if (ok2 && iw2_i >= 0 && iw2_i < i32(md.W)) { acc2 = acc2 + X[x_h_base + u32(iw2_i)] * w; }\n"
        "        if (ok3 && iw3_i >= 0 && iw3_i < i32(md.W)) { acc3 = acc3 + X[x_h_base + u32(iw3_i)] * w; }\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "  let bias_val : f32 = select(0.0, Bi[m], md.has_bias != 0u);\n"
        "  acc0 = acc0 + bias_val;\n"
        "  acc1 = acc1 + bias_val;\n"
        "  acc2 = acc2 + bias_val;\n"
        "  acc3 = acc3 + bias_val;\n";

    // Helper lambda to emit one output's epilogue + write. Captured by const ref.
    auto emit_out = [&](const char* accv, const char* owv, const char* okv) {
        s += "  if (";
        s += okv;
        s += ") {\n";
        s += "    let ow = ";
        s += owv;
        s += ";\n";
        s += "    let i = ((n * md.C_out + m) * md.H_out + oh) * md.W_out + ow;\n";
        s += "    var v : f32 = ";
        s += accv;
        s += ";\n";
        if (needs_meta) {
            for (int k = 0; k < n_sides; ++k) {
                auto ks = std::to_string(k);
                s += "    let side_";
                s += ks;
                s += "_flat : u32 = n * sd.s";
                s += ks;
                s += "_strides.x + m * sd.s";
                s += ks;
                s += "_strides.y + oh * sd.s";
                s += ks;
                s += "_strides.z + ow * sd.s";
                s += ks;
                s += "_strides.w;\n";
            }
        }
        for (const auto& stg : stage_wgsl) {
            s += "    v = ";
            s += stg;
            s += ";\n";
        }
        s += "    Y[i] = v;\n"
             "  }\n";
    };

    emit_out("acc0", "ow0", "ok0");
    emit_out("acc1", "ow1", "ok1");
    emit_out("acc2", "ow2", "ok2");
    emit_out("acc3", "ow3", "ok3");

    s += "}\n";
    return s;
}

// TILE_W=8 variant. Same structure as make_conv_chain_wgsl_regtile; the
// two are kept separate rather than parameterized to keep each shader's
// register/accumulator count fixed at compile time. Dispatch uses
// (ceil(W_out/(8*8)), ceil(H_out/8), N*C_out) — so this variant only makes
// sense when W_out ≥ 64-ish (otherwise ~half the threads have all-invalid
// outputs and it underperforms the TILE_W=4 variant).
std::string make_conv_chain_wgsl_regtile_wide(const std::vector<std::string>& stage_wgsl,
                                              int n_sides, bool needs_meta)
{
    std::string s =
        "struct Meta {\n"
        "  total       : u32,\n"
        "  N           : u32, C_out : u32, H_out : u32, W_out : u32,\n"
        "  C_in        : u32, groups : u32,\n"
        "  kH          : u32, kW    : u32,\n"
        "  stride_h    : u32, stride_w : u32,\n"
        "  pad_top     : u32, pad_left : u32,\n"
        "  dilation_h  : u32, dilation_w : u32,\n"
        "  H           : u32, W : u32,\n"
        "  has_bias    : u32,\n"
        "};\n";
    if (needs_meta) {
        s += "struct SideMeta {\n";
        for (int k = 0; k < n_sides; ++k) {
            s += "  s";
            s += std::to_string(k);
            s += "_strides : vec4<u32>,\n";
        }
        s += "};\n";
    }
    s += "@group(0) @binding(0) var<storage, read>       X  : array<f32>;\n"
         "@group(0) @binding(1) var<storage, read>       Wt : array<f32>;\n"
         "@group(0) @binding(2) var<storage, read>       Bi : array<f32>;\n";
    for (int k = 0; k < n_sides; ++k) {
        s += "@group(0) @binding(";
        s += std::to_string(3 + k);
        s += ") var<storage, read>       S";
        s += std::to_string(k);
        s += " : array<f32>;\n";
    }
    const int y_binding  = 3 + n_sides;
    const int md_binding = 4 + n_sides;
    const int sd_binding = 5 + n_sides;
    s += "@group(0) @binding(";
    s += std::to_string(y_binding);
    s += ") var<storage, read_write> Y  : array<f32>;\n"
         "@group(0) @binding(";
    s += std::to_string(md_binding);
    s += ") var<storage, read>       md : Meta;\n";
    if (needs_meta) {
        s += "@group(0) @binding(";
        s += std::to_string(sd_binding);
        s += ") var<storage, read>       sd : SideMeta;\n";
    }
    s +=
        "@compute @workgroup_size(8, 8, 1)\n"
        "fn main(@builtin(workgroup_id)         wid : vec3<u32>,\n"
        "        @builtin(local_invocation_id)  lid : vec3<u32>) {\n"
        "  let nm = wid.z;\n"
        "  let n  = nm / md.C_out;\n"
        "  let m  = nm % md.C_out;\n"
        "  let oh = wid.y * 8u + lid.y;\n"
        "  let ow_base = (wid.x * 8u + lid.x) * 8u;\n"
        "  if (oh >= md.H_out) { return; }\n"
        "  let M_per_group    = md.C_out / md.groups;\n"
        "  let C_in_per_group = md.C_in  / md.groups;\n"
        "  let g              = m / M_per_group;\n"
        "  let c_start         = g * C_in_per_group;\n"
        "  let x_batch_stride  = md.C_in * md.H * md.W;\n"
        "  let x_chan_stride   = md.H * md.W;\n"
        "  let w_outch_stride  = C_in_per_group * md.kH * md.kW;\n"
        "  let w_inch_stride   = md.kH * md.kW;\n"
        "  let base_x = n * x_batch_stride;\n"
        "  let base_w = m * w_outch_stride;\n"
        "  let ow0 = ow_base + 0u;\n"
        "  let ow1 = ow_base + 1u;\n"
        "  let ow2 = ow_base + 2u;\n"
        "  let ow3 = ow_base + 3u;\n"
        "  let ow4 = ow_base + 4u;\n"
        "  let ow5 = ow_base + 5u;\n"
        "  let ow6 = ow_base + 6u;\n"
        "  let ow7 = ow_base + 7u;\n"
        "  let ok0 = ow0 < md.W_out;\n"
        "  let ok1 = ow1 < md.W_out;\n"
        "  let ok2 = ow2 < md.W_out;\n"
        "  let ok3 = ow3 < md.W_out;\n"
        "  let ok4 = ow4 < md.W_out;\n"
        "  let ok5 = ow5 < md.W_out;\n"
        "  let ok6 = ow6 < md.W_out;\n"
        "  let ok7 = ow7 < md.W_out;\n"
        "  var acc0 : f32 = 0.0;\n"
        "  var acc1 : f32 = 0.0;\n"
        "  var acc2 : f32 = 0.0;\n"
        "  var acc3 : f32 = 0.0;\n"
        "  var acc4 : f32 = 0.0;\n"
        "  var acc5 : f32 = 0.0;\n"
        "  var acc6 : f32 = 0.0;\n"
        "  var acc7 : f32 = 0.0;\n"
        "  for (var ic : u32 = 0u; ic < C_in_per_group; ic = ic + 1u) {\n"
        "    let c = c_start + ic;\n"
        "    let x_c_base = base_x + c * x_chan_stride;\n"
        "    let w_c_base = base_w + ic * w_inch_stride;\n"
        "    for (var kh : u32 = 0u; kh < md.kH; kh = kh + 1u) {\n"
        "      let ih_i = i32(oh * md.stride_h) + i32(kh * md.dilation_h) - i32(md.pad_top);\n"
        "      if (ih_i < 0 || ih_i >= i32(md.H)) { continue; }\n"
        "      let ih = u32(ih_i);\n"
        "      let x_h_base = x_c_base + ih * md.W;\n"
        "      let w_h_base = w_c_base + kh * md.kW;\n"
        "      for (var kw : u32 = 0u; kw < md.kW; kw = kw + 1u) {\n"
        "        let w = Wt[w_h_base + kw];\n"
        "        let kw_dw = i32(kw * md.dilation_w) - i32(md.pad_left);\n"
        "        let iw0_i = i32(ow0 * md.stride_w) + kw_dw;\n"
        "        let iw1_i = i32(ow1 * md.stride_w) + kw_dw;\n"
        "        let iw2_i = i32(ow2 * md.stride_w) + kw_dw;\n"
        "        let iw3_i = i32(ow3 * md.stride_w) + kw_dw;\n"
        "        let iw4_i = i32(ow4 * md.stride_w) + kw_dw;\n"
        "        let iw5_i = i32(ow5 * md.stride_w) + kw_dw;\n"
        "        let iw6_i = i32(ow6 * md.stride_w) + kw_dw;\n"
        "        let iw7_i = i32(ow7 * md.stride_w) + kw_dw;\n"
        "        if (ok0 && iw0_i >= 0 && iw0_i < i32(md.W)) { acc0 = acc0 + X[x_h_base + u32(iw0_i)] * w; }\n"
        "        if (ok1 && iw1_i >= 0 && iw1_i < i32(md.W)) { acc1 = acc1 + X[x_h_base + u32(iw1_i)] * w; }\n"
        "        if (ok2 && iw2_i >= 0 && iw2_i < i32(md.W)) { acc2 = acc2 + X[x_h_base + u32(iw2_i)] * w; }\n"
        "        if (ok3 && iw3_i >= 0 && iw3_i < i32(md.W)) { acc3 = acc3 + X[x_h_base + u32(iw3_i)] * w; }\n"
        "        if (ok4 && iw4_i >= 0 && iw4_i < i32(md.W)) { acc4 = acc4 + X[x_h_base + u32(iw4_i)] * w; }\n"
        "        if (ok5 && iw5_i >= 0 && iw5_i < i32(md.W)) { acc5 = acc5 + X[x_h_base + u32(iw5_i)] * w; }\n"
        "        if (ok6 && iw6_i >= 0 && iw6_i < i32(md.W)) { acc6 = acc6 + X[x_h_base + u32(iw6_i)] * w; }\n"
        "        if (ok7 && iw7_i >= 0 && iw7_i < i32(md.W)) { acc7 = acc7 + X[x_h_base + u32(iw7_i)] * w; }\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "  let bias_val : f32 = select(0.0, Bi[m], md.has_bias != 0u);\n"
        "  acc0 = acc0 + bias_val;\n"
        "  acc1 = acc1 + bias_val;\n"
        "  acc2 = acc2 + bias_val;\n"
        "  acc3 = acc3 + bias_val;\n"
        "  acc4 = acc4 + bias_val;\n"
        "  acc5 = acc5 + bias_val;\n"
        "  acc6 = acc6 + bias_val;\n"
        "  acc7 = acc7 + bias_val;\n";

    auto emit_out = [&](const char* accv, const char* owv, const char* okv) {
        s += "  if (";
        s += okv;
        s += ") {\n";
        s += "    let ow = ";
        s += owv;
        s += ";\n";
        s += "    let i = ((n * md.C_out + m) * md.H_out + oh) * md.W_out + ow;\n";
        s += "    var v : f32 = ";
        s += accv;
        s += ";\n";
        if (needs_meta) {
            for (int k = 0; k < n_sides; ++k) {
                auto ks = std::to_string(k);
                s += "    let side_";
                s += ks;
                s += "_flat : u32 = n * sd.s";
                s += ks;
                s += "_strides.x + m * sd.s";
                s += ks;
                s += "_strides.y + oh * sd.s";
                s += ks;
                s += "_strides.z + ow * sd.s";
                s += ks;
                s += "_strides.w;\n";
            }
        }
        for (const auto& stg : stage_wgsl) {
            s += "    v = ";
            s += stg;
            s += ";\n";
        }
        s += "    Y[i] = v;\n"
             "  }\n";
    };

    emit_out("acc0", "ow0", "ok0");
    emit_out("acc1", "ow1", "ok1");
    emit_out("acc2", "ow2", "ok2");
    emit_out("acc3", "ow3", "ok3");
    emit_out("acc4", "ow4", "ok4");
    emit_out("acc5", "ow5", "ok5");
    emit_out("acc6", "ow6", "ok6");
    emit_out("acc7", "ow7", "ok7");

    s += "}\n";
    return s;
}

// 2D output-tile variant (2 oh × 4 ow = 8 outputs per thread). Dispatches
// (ceil(W_out/(WGX*4)), ceil(H_out/(WGY*2)), N*C_out). Each thread has 8
// accumulators arranged as acc[row][col]. The inner (ic, kh, kw) loop
// reads 1 weight and reuses it across 8 FMAs (vs 4 in the 1D regtile);
// per kw iteration we issue 2 × 4 X reads, where the 2 ih values come
// from the same (oh_row, kh) pair on different oh_base rows. With kH=3
// we touch 4 unique ih values across (2 rows × 3 kh) instead of 6 for
// two separate threads — ~33% fewer X row fetches per thread-group.
//
// Used via NNR_WEBGPU_REGTILE_CONV_CHAIN_2D=1 / auto-selected for the
// memory-bound first-layer case (small iC relative to kH*kW).
std::string make_conv_chain_wgsl_regtile_2d(const std::vector<std::string>& stage_wgsl,
                                            int n_sides, bool needs_meta)
{
    std::string s =
        "struct Meta {\n"
        "  total       : u32,\n"
        "  N           : u32, C_out : u32, H_out : u32, W_out : u32,\n"
        "  C_in        : u32, groups : u32,\n"
        "  kH          : u32, kW    : u32,\n"
        "  stride_h    : u32, stride_w : u32,\n"
        "  pad_top     : u32, pad_left : u32,\n"
        "  dilation_h  : u32, dilation_w : u32,\n"
        "  H           : u32, W : u32,\n"
        "  has_bias    : u32,\n"
        "};\n";
    if (needs_meta) {
        s += "struct SideMeta {\n";
        for (int k = 0; k < n_sides; ++k) {
            s += "  s";
            s += std::to_string(k);
            s += "_strides : vec4<u32>,\n";
        }
        s += "};\n";
    }
    s += "@group(0) @binding(0) var<storage, read>       X  : array<f32>;\n"
         "@group(0) @binding(1) var<storage, read>       Wt : array<f32>;\n"
         "@group(0) @binding(2) var<storage, read>       Bi : array<f32>;\n";
    for (int k = 0; k < n_sides; ++k) {
        s += "@group(0) @binding(";
        s += std::to_string(3 + k);
        s += ") var<storage, read>       S";
        s += std::to_string(k);
        s += " : array<f32>;\n";
    }
    const int y_binding  = 3 + n_sides;
    const int md_binding = 4 + n_sides;
    const int sd_binding = 5 + n_sides;
    s += "@group(0) @binding(";
    s += std::to_string(y_binding);
    s += ") var<storage, read_write> Y  : array<f32>;\n"
         "@group(0) @binding(";
    s += std::to_string(md_binding);
    s += ") var<storage, read>       md : Meta;\n";
    if (needs_meta) {
        s += "@group(0) @binding(";
        s += std::to_string(sd_binding);
        s += ") var<storage, read>       sd : SideMeta;\n";
    }
    s +=
        "@compute @workgroup_size(8, 8, 1)\n"
        "fn main(@builtin(workgroup_id)         wid : vec3<u32>,\n"
        "        @builtin(local_invocation_id)  lid : vec3<u32>) {\n"
        "  let nm = wid.z;\n"
        "  let n  = nm / md.C_out;\n"
        "  let m  = nm % md.C_out;\n"
        "  // Each thread owns a 2×4 output tile.\n"
        "  let oh_base = (wid.y * 8u + lid.y) * 2u;\n"
        "  let ow_base = (wid.x * 8u + lid.x) * 4u;\n"
        "  let oh0 = oh_base + 0u;\n"
        "  let oh1 = oh_base + 1u;\n"
        "  if (oh0 >= md.H_out) { return; }\n"
        "  let oh0_ok = oh0 < md.H_out;\n"
        "  let oh1_ok = oh1 < md.H_out;\n"
        "  let M_per_group    = md.C_out / md.groups;\n"
        "  let C_in_per_group = md.C_in  / md.groups;\n"
        "  let g              = m / M_per_group;\n"
        "  let c_start         = g * C_in_per_group;\n"
        "  let x_batch_stride  = md.C_in * md.H * md.W;\n"
        "  let x_chan_stride   = md.H * md.W;\n"
        "  let w_outch_stride  = C_in_per_group * md.kH * md.kW;\n"
        "  let w_inch_stride   = md.kH * md.kW;\n"
        "  let base_x = n * x_batch_stride;\n"
        "  let base_w = m * w_outch_stride;\n"
        "  let ow0 = ow_base + 0u;\n"
        "  let ow1 = ow_base + 1u;\n"
        "  let ow2 = ow_base + 2u;\n"
        "  let ow3 = ow_base + 3u;\n"
        "  let ok0 = ow0 < md.W_out;\n"
        "  let ok1 = ow1 < md.W_out;\n"
        "  let ok2 = ow2 < md.W_out;\n"
        "  let ok3 = ow3 < md.W_out;\n"
        "  // acc[row][col] — 2 rows × 4 cols = 8 accumulators.\n"
        "  var a00 : f32 = 0.0; var a01 : f32 = 0.0;\n"
        "  var a02 : f32 = 0.0; var a03 : f32 = 0.0;\n"
        "  var a10 : f32 = 0.0; var a11 : f32 = 0.0;\n"
        "  var a12 : f32 = 0.0; var a13 : f32 = 0.0;\n"
        "  for (var ic : u32 = 0u; ic < C_in_per_group; ic = ic + 1u) {\n"
        "    let c = c_start + ic;\n"
        "    let x_c_base = base_x + c * x_chan_stride;\n"
        "    let w_c_base = base_w + ic * w_inch_stride;\n"
        "    for (var kh : u32 = 0u; kh < md.kH; kh = kh + 1u) {\n"
        "      let ih0_i = i32(oh0 * md.stride_h) + i32(kh * md.dilation_h) - i32(md.pad_top);\n"
        "      let ih1_i = i32(oh1 * md.stride_h) + i32(kh * md.dilation_h) - i32(md.pad_top);\n"
        "      let ih0_ok = ih0_i >= 0 && ih0_i < i32(md.H) && oh0_ok;\n"
        "      let ih1_ok = ih1_i >= 0 && ih1_i < i32(md.H) && oh1_ok;\n"
        "      let x_h0 = x_c_base + select(0u, u32(ih0_i) * md.W, ih0_ok);\n"
        "      let x_h1 = x_c_base + select(0u, u32(ih1_i) * md.W, ih1_ok);\n"
        "      let w_h_base = w_c_base + kh * md.kW;\n"
        "      for (var kw : u32 = 0u; kw < md.kW; kw = kw + 1u) {\n"
        "        let w = Wt[w_h_base + kw];\n"
        "        let kw_dw = i32(kw * md.dilation_w) - i32(md.pad_left);\n"
        "        let iw0_i = i32(ow0 * md.stride_w) + kw_dw;\n"
        "        let iw1_i = i32(ow1 * md.stride_w) + kw_dw;\n"
        "        let iw2_i = i32(ow2 * md.stride_w) + kw_dw;\n"
        "        let iw3_i = i32(ow3 * md.stride_w) + kw_dw;\n"
        "        let iw0_ok = iw0_i >= 0 && iw0_i < i32(md.W);\n"
        "        let iw1_ok = iw1_i >= 0 && iw1_i < i32(md.W);\n"
        "        let iw2_ok = iw2_i >= 0 && iw2_i < i32(md.W);\n"
        "        let iw3_ok = iw3_i >= 0 && iw3_i < i32(md.W);\n"
        "        if (ih0_ok) {\n"
        "          if (ok0 && iw0_ok) { a00 = a00 + X[x_h0 + u32(iw0_i)] * w; }\n"
        "          if (ok1 && iw1_ok) { a01 = a01 + X[x_h0 + u32(iw1_i)] * w; }\n"
        "          if (ok2 && iw2_ok) { a02 = a02 + X[x_h0 + u32(iw2_i)] * w; }\n"
        "          if (ok3 && iw3_ok) { a03 = a03 + X[x_h0 + u32(iw3_i)] * w; }\n"
        "        }\n"
        "        if (ih1_ok) {\n"
        "          if (ok0 && iw0_ok) { a10 = a10 + X[x_h1 + u32(iw0_i)] * w; }\n"
        "          if (ok1 && iw1_ok) { a11 = a11 + X[x_h1 + u32(iw1_i)] * w; }\n"
        "          if (ok2 && iw2_ok) { a12 = a12 + X[x_h1 + u32(iw2_i)] * w; }\n"
        "          if (ok3 && iw3_ok) { a13 = a13 + X[x_h1 + u32(iw3_i)] * w; }\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "  let bias_val : f32 = select(0.0, Bi[m], md.has_bias != 0u);\n"
        "  a00 = a00 + bias_val; a01 = a01 + bias_val;\n"
        "  a02 = a02 + bias_val; a03 = a03 + bias_val;\n"
        "  a10 = a10 + bias_val; a11 = a11 + bias_val;\n"
        "  a12 = a12 + bias_val; a13 = a13 + bias_val;\n";

    auto emit_out = [&](const char* accv,
                        const char* ohv, const char* oh_okv,
                        const char* owv, const char* ow_okv) {
        s += "  if (";
        s += oh_okv;
        s += " && ";
        s += ow_okv;
        s += ") {\n";
        s += "    let oh = ";
        s += ohv;
        s += ";\n";
        s += "    let ow = ";
        s += owv;
        s += ";\n";
        s += "    let i = ((n * md.C_out + m) * md.H_out + oh) * md.W_out + ow;\n";
        s += "    var v : f32 = ";
        s += accv;
        s += ";\n";
        if (needs_meta) {
            for (int k = 0; k < n_sides; ++k) {
                auto ks = std::to_string(k);
                s += "    let side_";
                s += ks;
                s += "_flat : u32 = n * sd.s";
                s += ks;
                s += "_strides.x + m * sd.s";
                s += ks;
                s += "_strides.y + oh * sd.s";
                s += ks;
                s += "_strides.z + ow * sd.s";
                s += ks;
                s += "_strides.w;\n";
            }
        }
        for (const auto& stg : stage_wgsl) {
            s += "    v = ";
            s += stg;
            s += ";\n";
        }
        s += "    Y[i] = v;\n"
             "  }\n";
    };

    // Row 0
    emit_out("a00", "oh0", "oh0_ok", "ow0", "ok0");
    emit_out("a01", "oh0", "oh0_ok", "ow1", "ok1");
    emit_out("a02", "oh0", "oh0_ok", "ow2", "ok2");
    emit_out("a03", "oh0", "oh0_ok", "ow3", "ok3");
    // Row 1
    emit_out("a10", "oh1", "oh1_ok", "ow0", "ok0");
    emit_out("a11", "oh1", "oh1_ok", "ow1", "ok1");
    emit_out("a12", "oh1", "oh1_ok", "ow2", "ok2");
    emit_out("a13", "oh1", "oh1_ok", "ow3", "ok3");

    s += "}\n";
    return s;
}

// Specialized regtile variant: assumes kH=kW=3, stride=dilation=1,
// pad_top=pad_left=1, groups=1. All of these are inlined as WGSL literals
// so tint can fully unroll the kh/kw loops (9 iterations with known
// constant taps) and fold stride/dilation/pad arithmetic. Same 4-output
// TILE_W=4 shape as the general regtile; only the constant kernel geometry
// changes. Enabled automatically when the ConvFusedChain op's attrs match.
std::string make_conv_chain_wgsl_regtile_3x3s1(const std::vector<std::string>& stage_wgsl,
                                               int n_sides, bool needs_meta)
{
    std::string s =
        "struct Meta {\n"
        "  total       : u32,\n"
        "  N           : u32, C_out : u32, H_out : u32, W_out : u32,\n"
        "  C_in        : u32, groups : u32,\n"
        "  kH          : u32, kW    : u32,\n"
        "  stride_h    : u32, stride_w : u32,\n"
        "  pad_top     : u32, pad_left : u32,\n"
        "  dilation_h  : u32, dilation_w : u32,\n"
        "  H           : u32, W : u32,\n"
        "  has_bias    : u32,\n"
        "};\n";
    if (needs_meta) {
        s += "struct SideMeta {\n";
        for (int k = 0; k < n_sides; ++k) {
            s += "  s";
            s += std::to_string(k);
            s += "_strides : vec4<u32>,\n";
        }
        s += "};\n";
    }
    s += "@group(0) @binding(0) var<storage, read>       X  : array<f32>;\n"
         "@group(0) @binding(1) var<storage, read>       Wt : array<f32>;\n"
         "@group(0) @binding(2) var<storage, read>       Bi : array<f32>;\n";
    for (int k = 0; k < n_sides; ++k) {
        s += "@group(0) @binding(";
        s += std::to_string(3 + k);
        s += ") var<storage, read>       S";
        s += std::to_string(k);
        s += " : array<f32>;\n";
    }
    const int y_binding  = 3 + n_sides;
    const int md_binding = 4 + n_sides;
    const int sd_binding = 5 + n_sides;
    s += "@group(0) @binding(";
    s += std::to_string(y_binding);
    s += ") var<storage, read_write> Y  : array<f32>;\n"
         "@group(0) @binding(";
    s += std::to_string(md_binding);
    s += ") var<storage, read>       md : Meta;\n";
    if (needs_meta) {
        s += "@group(0) @binding(";
        s += std::to_string(sd_binding);
        s += ") var<storage, read>       sd : SideMeta;\n";
    }
    s +=
        "@compute @workgroup_size(8, 8, 1)\n"
        "fn main(@builtin(workgroup_id)         wid : vec3<u32>,\n"
        "        @builtin(local_invocation_id)  lid : vec3<u32>) {\n"
        "  let nm = wid.z;\n"
        "  let n  = nm / md.C_out;\n"
        "  let m  = nm % md.C_out;\n"
        "  let oh = wid.y * 8u + lid.y;\n"
        "  let ow_base = (wid.x * 8u + lid.x) * 4u;\n"
        "  if (oh >= md.H_out) { return; }\n"
        "  let C_in = md.C_in;\n"   // groups=1 means C_in_per_group == C_in
        "  let x_batch_stride  = C_in * md.H * md.W;\n"
        "  let x_chan_stride   = md.H * md.W;\n"
        "  let w_outch_stride  = C_in * 9u;\n"  // kH*kW = 9
        "  let w_inch_stride   = 9u;\n"
        "  let base_x = n * x_batch_stride;\n"
        "  let base_w = m * w_outch_stride;\n"
        "  let ow0 = ow_base + 0u;\n"
        "  let ow1 = ow_base + 1u;\n"
        "  let ow2 = ow_base + 2u;\n"
        "  let ow3 = ow_base + 3u;\n"
        "  let ok0 = ow0 < md.W_out;\n"
        "  let ok1 = ow1 < md.W_out;\n"
        "  let ok2 = ow2 < md.W_out;\n"
        "  let ok3 = ow3 < md.W_out;\n"
        // Pre-compute the 3 valid-row flags once (kh = 0, 1, 2).
        "  let ih_kh0_i = i32(oh) + 0i - 1i;\n"    // stride=1, dilation=1, pad=1
        "  let ih_kh1_i = i32(oh) + 1i - 1i;\n"
        "  let ih_kh2_i = i32(oh) + 2i - 1i;\n"
        "  let ih_kh0_ok = ih_kh0_i >= 0 && ih_kh0_i < i32(md.H);\n"
        "  let ih_kh1_ok = ih_kh1_i >= 0 && ih_kh1_i < i32(md.H);\n"
        "  let ih_kh2_ok = ih_kh2_i >= 0 && ih_kh2_i < i32(md.H);\n"
        "  let ih_kh0 : u32 = select(0u, u32(ih_kh0_i), ih_kh0_ok);\n"
        "  let ih_kh1 : u32 = select(0u, u32(ih_kh1_i), ih_kh1_ok);\n"
        "  let ih_kh2 : u32 = select(0u, u32(ih_kh2_i), ih_kh2_ok);\n"
        // Per-thread iw offsets for kw=0,1,2 (stride=1, dilation=1, pad=1 → kw - 1).
        "  let iw0_kw0 = i32(ow0) - 1i;\n"
        "  let iw0_kw1 = i32(ow0) + 0i;\n"
        "  let iw0_kw2 = i32(ow0) + 1i;\n"
        "  let iw1_kw0 = i32(ow1) - 1i;\n"
        "  let iw1_kw1 = i32(ow1) + 0i;\n"
        "  let iw1_kw2 = i32(ow1) + 1i;\n"
        "  let iw2_kw0 = i32(ow2) - 1i;\n"
        "  let iw2_kw1 = i32(ow2) + 0i;\n"
        "  let iw2_kw2 = i32(ow2) + 1i;\n"
        "  let iw3_kw0 = i32(ow3) - 1i;\n"
        "  let iw3_kw1 = i32(ow3) + 0i;\n"
        "  let iw3_kw2 = i32(ow3) + 1i;\n"
        "  let W_i = i32(md.W);\n"
        "  let iw0_kw0_ok = iw0_kw0 >= 0 && iw0_kw0 < W_i;\n"
        "  let iw0_kw1_ok = iw0_kw1 >= 0 && iw0_kw1 < W_i;\n"
        "  let iw0_kw2_ok = iw0_kw2 >= 0 && iw0_kw2 < W_i;\n"
        "  let iw1_kw0_ok = iw1_kw0 >= 0 && iw1_kw0 < W_i;\n"
        "  let iw1_kw1_ok = iw1_kw1 >= 0 && iw1_kw1 < W_i;\n"
        "  let iw1_kw2_ok = iw1_kw2 >= 0 && iw1_kw2 < W_i;\n"
        "  let iw2_kw0_ok = iw2_kw0 >= 0 && iw2_kw0 < W_i;\n"
        "  let iw2_kw1_ok = iw2_kw1 >= 0 && iw2_kw1 < W_i;\n"
        "  let iw2_kw2_ok = iw2_kw2 >= 0 && iw2_kw2 < W_i;\n"
        "  let iw3_kw0_ok = iw3_kw0 >= 0 && iw3_kw0 < W_i;\n"
        "  let iw3_kw1_ok = iw3_kw1 >= 0 && iw3_kw1 < W_i;\n"
        "  let iw3_kw2_ok = iw3_kw2 >= 0 && iw3_kw2 < W_i;\n"
        "  var acc0 : f32 = 0.0;\n"
        "  var acc1 : f32 = 0.0;\n"
        "  var acc2 : f32 = 0.0;\n"
        "  var acc3 : f32 = 0.0;\n"
        // Single outer ic loop — kh/kw loops are fully unrolled manually.
        // Each iteration reads 9 weights and 3 rows of X (if valid).
        "  for (var ic : u32 = 0u; ic < C_in; ic = ic + 1u) {\n"
        "    let x_c_base = base_x + ic * x_chan_stride;\n"
        "    let w_c_base = base_w + ic * w_inch_stride;\n"
        "    let w00 = Wt[w_c_base + 0u]; let w01 = Wt[w_c_base + 1u]; let w02 = Wt[w_c_base + 2u];\n"
        "    let w10 = Wt[w_c_base + 3u]; let w11 = Wt[w_c_base + 4u]; let w12 = Wt[w_c_base + 5u];\n"
        "    let w20 = Wt[w_c_base + 6u]; let w21 = Wt[w_c_base + 7u]; let w22 = Wt[w_c_base + 8u];\n"
        // ROW kh=0
        "    if (ih_kh0_ok) {\n"
        "      let x_row = x_c_base + ih_kh0 * md.W;\n"
        "      if (ok0) {\n"
        "        if (iw0_kw0_ok) { acc0 = acc0 + X[x_row + u32(iw0_kw0)] * w00; }\n"
        "        if (iw0_kw1_ok) { acc0 = acc0 + X[x_row + u32(iw0_kw1)] * w01; }\n"
        "        if (iw0_kw2_ok) { acc0 = acc0 + X[x_row + u32(iw0_kw2)] * w02; }\n"
        "      }\n"
        "      if (ok1) {\n"
        "        if (iw1_kw0_ok) { acc1 = acc1 + X[x_row + u32(iw1_kw0)] * w00; }\n"
        "        if (iw1_kw1_ok) { acc1 = acc1 + X[x_row + u32(iw1_kw1)] * w01; }\n"
        "        if (iw1_kw2_ok) { acc1 = acc1 + X[x_row + u32(iw1_kw2)] * w02; }\n"
        "      }\n"
        "      if (ok2) {\n"
        "        if (iw2_kw0_ok) { acc2 = acc2 + X[x_row + u32(iw2_kw0)] * w00; }\n"
        "        if (iw2_kw1_ok) { acc2 = acc2 + X[x_row + u32(iw2_kw1)] * w01; }\n"
        "        if (iw2_kw2_ok) { acc2 = acc2 + X[x_row + u32(iw2_kw2)] * w02; }\n"
        "      }\n"
        "      if (ok3) {\n"
        "        if (iw3_kw0_ok) { acc3 = acc3 + X[x_row + u32(iw3_kw0)] * w00; }\n"
        "        if (iw3_kw1_ok) { acc3 = acc3 + X[x_row + u32(iw3_kw1)] * w01; }\n"
        "        if (iw3_kw2_ok) { acc3 = acc3 + X[x_row + u32(iw3_kw2)] * w02; }\n"
        "      }\n"
        "    }\n"
        // ROW kh=1
        "    if (ih_kh1_ok) {\n"
        "      let x_row = x_c_base + ih_kh1 * md.W;\n"
        "      if (ok0) {\n"
        "        if (iw0_kw0_ok) { acc0 = acc0 + X[x_row + u32(iw0_kw0)] * w10; }\n"
        "        if (iw0_kw1_ok) { acc0 = acc0 + X[x_row + u32(iw0_kw1)] * w11; }\n"
        "        if (iw0_kw2_ok) { acc0 = acc0 + X[x_row + u32(iw0_kw2)] * w12; }\n"
        "      }\n"
        "      if (ok1) {\n"
        "        if (iw1_kw0_ok) { acc1 = acc1 + X[x_row + u32(iw1_kw0)] * w10; }\n"
        "        if (iw1_kw1_ok) { acc1 = acc1 + X[x_row + u32(iw1_kw1)] * w11; }\n"
        "        if (iw1_kw2_ok) { acc1 = acc1 + X[x_row + u32(iw1_kw2)] * w12; }\n"
        "      }\n"
        "      if (ok2) {\n"
        "        if (iw2_kw0_ok) { acc2 = acc2 + X[x_row + u32(iw2_kw0)] * w10; }\n"
        "        if (iw2_kw1_ok) { acc2 = acc2 + X[x_row + u32(iw2_kw1)] * w11; }\n"
        "        if (iw2_kw2_ok) { acc2 = acc2 + X[x_row + u32(iw2_kw2)] * w12; }\n"
        "      }\n"
        "      if (ok3) {\n"
        "        if (iw3_kw0_ok) { acc3 = acc3 + X[x_row + u32(iw3_kw0)] * w10; }\n"
        "        if (iw3_kw1_ok) { acc3 = acc3 + X[x_row + u32(iw3_kw1)] * w11; }\n"
        "        if (iw3_kw2_ok) { acc3 = acc3 + X[x_row + u32(iw3_kw2)] * w12; }\n"
        "      }\n"
        "    }\n"
        // ROW kh=2
        "    if (ih_kh2_ok) {\n"
        "      let x_row = x_c_base + ih_kh2 * md.W;\n"
        "      if (ok0) {\n"
        "        if (iw0_kw0_ok) { acc0 = acc0 + X[x_row + u32(iw0_kw0)] * w20; }\n"
        "        if (iw0_kw1_ok) { acc0 = acc0 + X[x_row + u32(iw0_kw1)] * w21; }\n"
        "        if (iw0_kw2_ok) { acc0 = acc0 + X[x_row + u32(iw0_kw2)] * w22; }\n"
        "      }\n"
        "      if (ok1) {\n"
        "        if (iw1_kw0_ok) { acc1 = acc1 + X[x_row + u32(iw1_kw0)] * w20; }\n"
        "        if (iw1_kw1_ok) { acc1 = acc1 + X[x_row + u32(iw1_kw1)] * w21; }\n"
        "        if (iw1_kw2_ok) { acc1 = acc1 + X[x_row + u32(iw1_kw2)] * w22; }\n"
        "      }\n"
        "      if (ok2) {\n"
        "        if (iw2_kw0_ok) { acc2 = acc2 + X[x_row + u32(iw2_kw0)] * w20; }\n"
        "        if (iw2_kw1_ok) { acc2 = acc2 + X[x_row + u32(iw2_kw1)] * w21; }\n"
        "        if (iw2_kw2_ok) { acc2 = acc2 + X[x_row + u32(iw2_kw2)] * w22; }\n"
        "      }\n"
        "      if (ok3) {\n"
        "        if (iw3_kw0_ok) { acc3 = acc3 + X[x_row + u32(iw3_kw0)] * w20; }\n"
        "        if (iw3_kw1_ok) { acc3 = acc3 + X[x_row + u32(iw3_kw1)] * w21; }\n"
        "        if (iw3_kw2_ok) { acc3 = acc3 + X[x_row + u32(iw3_kw2)] * w22; }\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "  let bias_val : f32 = select(0.0, Bi[m], md.has_bias != 0u);\n"
        "  acc0 = acc0 + bias_val;\n"
        "  acc1 = acc1 + bias_val;\n"
        "  acc2 = acc2 + bias_val;\n"
        "  acc3 = acc3 + bias_val;\n";

    auto emit_out = [&](const char* accv, const char* owv, const char* okv) {
        s += "  if (";
        s += okv;
        s += ") {\n";
        s += "    let ow = ";
        s += owv;
        s += ";\n";
        s += "    let i = ((n * md.C_out + m) * md.H_out + oh) * md.W_out + ow;\n";
        s += "    var v : f32 = ";
        s += accv;
        s += ";\n";
        if (needs_meta) {
            for (int k = 0; k < n_sides; ++k) {
                auto ks = std::to_string(k);
                s += "    let side_";
                s += ks;
                s += "_flat : u32 = n * sd.s";
                s += ks;
                s += "_strides.x + m * sd.s";
                s += ks;
                s += "_strides.y + oh * sd.s";
                s += ks;
                s += "_strides.z + ow * sd.s";
                s += ks;
                s += "_strides.w;\n";
            }
        }
        for (const auto& stg : stage_wgsl) {
            s += "    v = ";
            s += stg;
            s += ";\n";
        }
        s += "    Y[i] = v;\n"
             "  }\n";
    };

    emit_out("acc0", "ow0", "ok0");
    emit_out("acc1", "ow1", "ok1");
    emit_out("acc2", "ow2", "ok2");
    emit_out("acc3", "ow3", "ok3");

    s += "}\n";
    return s;
}

wgpu::BindGroupLayout make_conv_chain_bgl(int n_sides, bool needs_meta)
{
    auto& dev = get_device();
    const int total = 5 + n_sides + (needs_meta ? 1 : 0);
    std::vector<wgpu::BindGroupLayoutEntry> e(total);
    int idx = 0;
    // X, Wt, Bi, sides — all ReadOnlyStorage.
    for (int k = 0; k < 3 + n_sides; ++k, ++idx) {
        e[idx].binding = idx;
        e[idx].visibility = wgpu::ShaderStage::Compute;
        e[idx].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    }
    // Y — Storage.
    e[idx].binding = idx;
    e[idx].visibility = wgpu::ShaderStage::Compute;
    e[idx].buffer.type = wgpu::BufferBindingType::Storage;
    ++idx;
    // md — ReadOnlyStorage (matches base Conv: meta is in storage, not uniform).
    e[idx].binding = idx;
    e[idx].visibility = wgpu::ShaderStage::Compute;
    e[idx].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    ++idx;
    if (needs_meta) {
        e[idx].binding = idx;
        e[idx].visibility = wgpu::ShaderStage::Compute;
        e[idx].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        ++idx;
    }
    wgpu::BindGroupLayoutDescriptor d = {};
    d.entryCount = (uint32_t)total;
    d.entries = e.data();
    return dev.device.CreateBindGroupLayout(&d);
}

struct conv_chain_cache_entry_t {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
};

std::unordered_map<std::string, conv_chain_cache_entry_t>& conv_chain_cache()
{
    static std::unordered_map<std::string, conv_chain_cache_entry_t> cache;
    return cache;
}

} // namespace

bool conv_fused_chain_t::init() {
    if (stage_wgsl.empty())             return false;
    if (!device_ready())                return false;
    if (needs_meta && n_sides < 1)      return false;
    // Inputs = X + W + (has_bias ? B : nothing) + sides
    const int required = 2 + (has_bias ? 1 : 0) + n_sides;
    if ((int)inputs.size() != required) return false;
    if (outputs.size() != 1)            return false;

    std::string sig;
    sig += needs_meta ? 'M' : 'U';
    sig += std::to_string(n_sides);
    sig += '\x02';
    for (const auto& s : stage_wgsl) { sig += s; sig += '\x01'; }

    auto& cache = conv_chain_cache();
    auto it = cache.find(sig);
    if (it == cache.end()) {
        auto src = make_conv_chain_wgsl(stage_wgsl, n_sides, needs_meta);
        auto sm  = compile(src);
        conv_chain_cache_entry_t e;
        e.bgl      = make_conv_chain_bgl(n_sides, needs_meta);
        e.pipeline = make_pipeline(sm, e.bgl);
        it = cache.emplace(std::move(sig), std::move(e)).first;
    }
    bgl      = it->second.bgl;
    pipeline = it->second.pipeline;

    // Tiled variant: same BGL (bindings identical), different shader. Cached
    // under a 'T'-prefixed signature so stage/side variants share compilation
    // across ops with the same chain shape. Compiled only when opted in via
    // NNR_WEBGPU_TILED_CONV_CHAIN=1 (see reshape() for the tradeoff notes).
    static const bool s_enable_tiled = []{
        const char* v = std::getenv("NNR_WEBGPU_TILED_CONV_CHAIN");
        return v && *v && *v != '0';
    }();
    if (s_enable_tiled) {
        std::string tsig = "T";
        tsig += sig;
        auto tit = cache.find(tsig);
        if (tit == cache.end()) {
            auto tsrc = make_conv_chain_wgsl_tiled(stage_wgsl, n_sides, needs_meta);
            auto tsm  = compile(tsrc);
            conv_chain_cache_entry_t e;
            e.bgl      = bgl;   // reuse the BGL from the plain variant
            e.pipeline = make_pipeline(tsm, e.bgl);
            tit = cache.emplace(std::move(tsig), std::move(e)).first;
        }
        tiled_pipeline = tit->second.pipeline;
    }

    // Register-tiled variant — compiled eagerly. The shader is the default
    // fast path for W_out >= 8 / non-trivial first-layer convs; it nearly
    // always beats or matches the plain kernel and is a large win for the
    // memory-bound small-channel case. Cached under 'R' prefix.
    {
        std::string rsig = "R";
        rsig += sig;
        auto rit = cache.find(rsig);
        if (rit == cache.end()) {
            auto rsrc = make_conv_chain_wgsl_regtile(stage_wgsl, n_sides, needs_meta);
            auto rsm  = compile(rsrc);
            conv_chain_cache_entry_t e;
            e.bgl      = bgl;
            e.pipeline = make_pipeline(rsm, e.bgl);
            rit = cache.emplace(std::move(rsig), std::move(e)).first;
        }
        regtile_pipeline = rit->second.pipeline;
    }

    // Wider register-tile variant (8 cols per thread). Enabled only when
    // W_out is large enough that all 8 lid.x lanes have valid outputs —
    // otherwise ~half the workgroup wastes work. Cached under 'W' prefix.
    {
        std::string wsig = "W";
        wsig += sig;
        auto wit = cache.find(wsig);
        if (wit == cache.end()) {
            auto wsrc = make_conv_chain_wgsl_regtile_wide(stage_wgsl, n_sides, needs_meta);
            auto wsm  = compile(wsrc);
            conv_chain_cache_entry_t e;
            e.bgl      = bgl;
            e.pipeline = make_pipeline(wsm, e.bgl);
            wit = cache.emplace(std::move(wsig), std::move(e)).first;
        }
        regtile_wide_pipeline = wit->second.pipeline;
    }

    // 2D output-tile variant (2 rows × 4 cols per thread). Measured
    // REGRESSION on cnn_bench c1 (270 μs vs 210 μs plain) — the extra
    // register pressure from 8 accumulators + row/col validity flags
    // hurts occupancy more than the shared-weight / shared-row savings
    // help. Kept opt-in via NNR_WEBGPU_REGTILE_CONV_CHAIN_2D=1 for
    // experiments on other shapes; pipeline compile is skipped when off.
    static const bool s_enable_2d_compile = []{
        const char* v = std::getenv("NNR_WEBGPU_REGTILE_CONV_CHAIN_2D");
        return v && *v && *v != '0';
    }();
    if (s_enable_2d_compile) {
        std::string dsig = "D";
        dsig += sig;
        auto dit = cache.find(dsig);
        if (dit == cache.end()) {
            auto dsrc = make_conv_chain_wgsl_regtile_2d(stage_wgsl, n_sides, needs_meta);
            auto dsm  = compile(dsrc);
            conv_chain_cache_entry_t e;
            e.bgl      = bgl;
            e.pipeline = make_pipeline(dsm, e.bgl);
            dit = cache.emplace(std::move(dsig), std::move(e)).first;
        }
        regtile_2d_pipeline = dit->second.pipeline;
    }

    // Specialized 3x3 regtile — detect attribute match now and compile
    // the fully-unrolled variant. kH/kW come from the weight tensor's
    // dims (initializer, so shape is known); the rest come from
    // attributes. Only useful when the fixture actually uses 3x3 convs
    // with stride=dilation=1 and pad=1 on all sides.
    {
        bool matches = false;
        if ((int)inputs.size() >= 2 && inputs[1] && inputs[1]->ndim == 4
            && inputs[1]->dims[2] == 3 && inputs[1]->dims[3] == 3) {
            int64_t* ints = nullptr;
            int nstride = attribute(attr_key_t::strides, ints);
            const int s_h = (nstride >= 1) ? (int)ints[0] : 1;
            const int s_w = (nstride >= 2) ? (int)ints[1] : s_h;
            int ndil = attribute(attr_key_t::dilations, ints);
            const int d_h = (ndil >= 1) ? (int)ints[0] : 1;
            const int d_w = (ndil >= 2) ? (int)ints[1] : d_h;
            int npad = attribute(attr_key_t::pads, ints);
            const int group = (int)attribute(attr_key_t::group, (int64_t)1);
            if (s_h == 1 && s_w == 1 && d_h == 1 && d_w == 1 && group == 1
                && npad >= 4 && ints
                && ints[0] == 1 && ints[1] == 1 && ints[2] == 1 && ints[3] == 1) {
                matches = true;
            }
        }
        has_3x3_s1 = matches;
        if (matches) {
            std::string psig = "P";  // P = Pattern-specialized (3x3 stride-1)
            psig += sig;
            auto pit = cache.find(psig);
            if (pit == cache.end()) {
                auto psrc = make_conv_chain_wgsl_regtile_3x3s1(stage_wgsl, n_sides, needs_meta);
                auto psm  = compile(psrc);
                conv_chain_cache_entry_t e;
                e.bgl      = bgl;
                e.pipeline = make_pipeline(psm, e.bgl);
                pit = cache.emplace(std::move(psig), std::move(e)).first;
            }
            regtile_3x3_pipeline = pit->second.pipeline;
        }
    }

    auto& dev = get_device();

    wgpu::BufferDescriptor md = {};
    md.size  = 128;
    md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    meta_buf = dev.device.CreateBuffer(&md);

    if (needs_meta) {
        const size_t sz = (size_t)n_sides * 16;
        wgpu::BufferDescriptor sd = {};
        sd.size  = (sz + 15) & ~size_t(15);
        sd.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        side_md  = dev.device.CreateBuffer(&sd);
    }

    // Dummy 16-byte zero bias for the !has_bias case — keeps the binding
    // layout invariant (matches base Conv pattern).
    if (!has_bias) {
        wgpu::BufferDescriptor bd = {};
        bd.size  = 16;
        bd.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        dummy_bias = dev.device.CreateBuffer(&bd);
        uint8_t zeros[16] = {};
        dev.queue.WriteBuffer(dummy_bias, 0, zeros, sizeof(zeros));
    }
    return true;
}

bool conv_fused_chain_t::reshape() {
    const int required = 2 + (has_bias ? 1 : 0) + n_sides;
    if ((int)inputs.size() != required) return false;
    if (outputs.size() != 1)            return false;

    const tensor_t* X = inputs[0];
    const tensor_t* W = inputs[1];
    tensor_t*       Y = outputs[0];
    if (!X || !W || !Y)                   return false;
    if (X->type != NNR_DATA_TYPE_FLOAT32) return false;
    if (W->type != NNR_DATA_TYPE_FLOAT32) return false;
    if (X->ndim != 4 || W->ndim != 4)     return false;

    std::string_view auto_pad = attribute(attr_key_t::auto_pad, "NOTSET");
    if (auto_pad != "NOTSET" && auto_pad != "VALID") return false;

    int N  = X->dims[0];
    int C  = X->dims[1];
    int H  = X->dims[2];
    int Wd = X->dims[3];
    int M  = W->dims[0];
    int C_per_group = W->dims[1];
    int kH = W->dims[2];
    int kW = W->dims[3];

    int group = (int)attribute(attr_key_t::group, (int64_t)1);
    if (group < 1)                        return false;
    if (C % group != 0)                   return false;
    if (M % group != 0)                   return false;
    if (C / group != C_per_group)         return false;

    int64_t* ints = nullptr;
    int nstride = attribute(attr_key_t::strides,   ints);
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
    if (pad_top < 0 || pad_left < 0 || pad_bot < 0 || pad_right < 0) return false;

    int H_out = (H  + pad_top  + pad_bot   - d_h * (kH - 1) - 1) / s_h + 1;
    int W_out = (Wd + pad_left + pad_right - d_w * (kW - 1) - 1) / s_w + 1;
    if (H_out <= 0 || W_out <= 0)         return false;

    n_out = N; c_out = M; h_out = H_out; w_out = W_out;

    int out_dims[4] = { N, M, H_out, W_out };
    if (!Y->reshape(std::span<const int>(out_dims, 4), X->type)) return false;

    if (has_bias) {
        const tensor_t* B = inputs[2];
        if (!B || B->type != NNR_DATA_TYPE_FLOAT32) return false;
        if ((int)B->ndata != M)                     return false;
    }

    // Validate sides; each side must broadcast into [N, M, H_out, W_out].
    const int side_off = 2 + (has_bias ? 1 : 0);
    for (int kk = 0; kk < n_sides; ++kk) {
        const tensor_t* s = inputs[side_off + kk];
        if (!s)                                     return false;
        if (s->type != NNR_DATA_TYPE_FLOAT32)       return false;
        if (s->ndim > 4)                            return false;
        bool exact = (s->ndim == 4
                   && s->dims[0] == N  && s->dims[1] == M
                   && s->dims[2] == H_out && s->dims[3] == W_out);
        if (!exact) {
            if (!needs_meta)                        return false;
            // Right-align side dims to [N, M, H_out, W_out].
            const int off = 4 - s->ndim;
            const int pipe_dims[4] = { N, M, H_out, W_out };
            for (int i = 0; i < s->ndim; ++i) {
                int d_side = s->dims[i];
                int d_pipe = pipe_dims[off + i];
                if (d_side != 1 && d_side != d_pipe) return false;
            }
        }
    }

    total_u = (uint32_t)(N * M * H_out * W_out);
    meta_vals[0]  = total_u;
    meta_vals[1]  = (uint32_t)N;
    meta_vals[2]  = (uint32_t)M;
    meta_vals[3]  = (uint32_t)H_out;
    meta_vals[4]  = (uint32_t)W_out;
    meta_vals[5]  = (uint32_t)C;
    meta_vals[6]  = (uint32_t)group;
    meta_vals[7]  = (uint32_t)kH;
    meta_vals[8]  = (uint32_t)kW;
    meta_vals[9]  = (uint32_t)s_h;
    meta_vals[10] = (uint32_t)s_w;
    meta_vals[11] = (uint32_t)pad_top;
    meta_vals[12] = (uint32_t)pad_left;
    meta_vals[13] = (uint32_t)d_h;
    meta_vals[14] = (uint32_t)d_w;
    meta_vals[15] = (uint32_t)H;
    meta_vals[16] = (uint32_t)Wd;
    meta_vals[17] = (uint32_t)(has_bias ? 1 : 0);

    // Tiled variant: uses a 16 KB shared-memory weight cache + 8×8 output
    // tile per workgroup. In theory this cuts weight re-reads ~64× versus
    // the plain kernel's one-thread-per-output layout. In practice, on
    // cnn_bench-sized convs (w_size ∈ {27, 288, 576}) the shared-memory
    // allocation dominates occupancy and L1 was already caching weights
    // across the workgroup — measured ~3× regression when enabled by
    // default. Left in place behind NNR_WEBGPU_TILED_CONV_CHAIN=1 for
    // larger-w_size workloads that might reverse the tradeoff.
    static const bool s_enable_tiled = []{
        const char* v = std::getenv("NNR_WEBGPU_TILED_CONV_CHAIN");
        return v && *v && *v != '0';
    }();
    const uint32_t w_size = (uint32_t)(C_per_group * kH * kW);
    use_tiled = s_enable_tiled
             && (w_size <= CONV_CHAIN_MAX_TILED_W)
             && (H_out >= 4) && (W_out >= 4);

    // Register-tiled variant: each thread computes 4 adjacent output cols.
    // Win comes from sharing each weight across 4 FMAs and sharing X reads
    // across overlapping kernel taps. Requires W_out wide enough that the
    // 4-wide tile isn't mostly wasted. An opt-out env var is provided for
    // bisecting regressions; tiled takes precedence when both are on.
    static const bool s_disable_regtile = []{
        const char* v = std::getenv("NNR_WEBGPU_DISABLE_REGTILE_CONV_CHAIN");
        return v && *v && *v != '0';
    }();
    use_regtile = !use_tiled && !s_disable_regtile && (W_out >= 8);

    // Wider (TILE_W=8) regtile path. Intended for wide outputs (W_out >=
    // WGX*TILE_W = 64) where the bigger tile amortizes weight loads and
    // kernel-overhead per thread. Measured neutral on cnn_bench c1 (the
    // largest wide-output Conv in the test suite) — register pressure
    // from 8 accumulators + 8 iw valid-flags roughly cancels the bigger
    // tile's savings. Left opt-in via NNR_WEBGPU_REGTILE_CONV_CHAIN_WIDE=1
    // for workloads with bigger W_out where the balance may tip.
    static const bool s_enable_wide = []{
        const char* v = std::getenv("NNR_WEBGPU_REGTILE_CONV_CHAIN_WIDE");
        return v && *v && *v != '0';
    }();
    use_regtile_wide = s_enable_wide && use_regtile && (W_out >= 64);

    // 2D output-tile regtile (2 rows × 4 cols = 8 outputs per thread).
    // Requires H_out and W_out each ≥ tile size so we're not mostly edge
    // threads. Gate behind env-var opt-in for safety; wide takes
    // precedence when both are on.
    static const bool s_enable_2d = []{
        const char* v = std::getenv("NNR_WEBGPU_REGTILE_CONV_CHAIN_2D");
        return v && *v && *v != '0';
    }();
    use_regtile_2d = s_enable_2d && use_regtile && !use_regtile_wide
                 && (W_out >= 8) && (H_out >= 8);

    // Specialized 3x3 stride-1 regtile. Only available when init() saw a
    // matching attribute set. Takes precedence over the general regtile
    // when eligible since the fully-unrolled kh/kw loops let tint schedule
    // more aggressively. Opt-out env var retained for bisecting.
    static const bool s_disable_3x3 = []{
        const char* v = std::getenv("NNR_WEBGPU_DISABLE_REGTILE_CONV_CHAIN_3X3");
        return v && *v && *v != '0';
    }();
    use_regtile_3x3 = has_3x3_s1 && use_regtile && !s_disable_3x3;

    ensure_buffer(X, X->ndata * sizeof(float));
    auto& wr = ensure_buffer(W, W->ndata * sizeof(float));
    if (ctx && ctx->initializer_names.count(W->name)) wr.is_weight = true;
    if (has_bias) {
        const tensor_t* B = inputs[2];
        auto& br = ensure_buffer(B, B->ndata * sizeof(float));
        if (ctx && ctx->initializer_names.count(B->name)) br.is_weight = true;
    }
    for (int kk = 0; kk < n_sides; ++kk)
        ensure_buffer(inputs[side_off + kk],
                      inputs[side_off + kk]->ndata * sizeof(float));
    ensure_buffer(Y, Y->ndata * sizeof(float));

    // meta_vals + per-side stride table are pure functions of shape +
    // attribute data. Write here so exec() doesn't re-queue them.
    auto& dev = get_device();
    dev.queue.WriteBuffer(meta_buf, 0, meta_vals, sizeof(meta_vals));
    if (needs_meta) {
        const size_t rounded = ((size_t)n_sides * 16 + 15) & ~size_t(15);
        std::vector<uint8_t> buf(rounded, 0);
        auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(buf.data() + off, &v, 4); };
        for (int kk = 0; kk < n_sides; ++kk) {
            const tensor_t* s = inputs[side_off + kk];
            uint32_t nat[4] = {};
            {
                uint32_t st = 1;
                for (int a = s->ndim - 1; a >= 0; --a) {
                    nat[a] = st;
                    st *= (uint32_t)s->dims[a];
                }
            }
            const int off = 4 - s->ndim;
            uint32_t strides[4] = {0, 0, 0, 0};
            for (int a = 0; a < 4; ++a) {
                int sa = a - off;
                if (sa >= 0 && s->dims[sa] != 1) strides[a] = nat[sa];
            }
            put_u32((size_t)kk * 16 + 0,  strides[0]);
            put_u32((size_t)kk * 16 + 4,  strides[1]);
            put_u32((size_t)kk * 16 + 8,  strides[2]);
            put_u32((size_t)kk * 16 + 12, strides[3]);
        }
        dev.queue.WriteBuffer(side_md, 0, buf.data(), buf.size());
    }
    return true;
}

bool conv_fused_chain_t::exec() {
    auto& dev = get_device();
    upload_if_needed(inputs[0]);
    upload_if_needed(inputs[1]);
    if (has_bias) upload_if_needed(inputs[2]);
    const int side_off = 2 + (has_bias ? 1 : 0);
    for (int kk = 0; kk < n_sides; ++kk) upload_if_needed(inputs[side_off + kk]);

    // meta_buf and side_md were written in reshape().
    auto* rx = find(inputs[0]);
    auto* rw = find(inputs[1]);
    auto* ry = find(outputs[0]);
    wgpu::Buffer b_buf; uint64_t b_sz;
    if (has_bias) { auto* rb = find(inputs[2]); b_buf = rb->buf; b_sz = rb->size; }
    else          { b_buf = dummy_bias;                          b_sz = 16;       }

    const int y_binding  = 3 + n_sides;
    const int md_binding = 4 + n_sides;
    const int sd_binding = 5 + n_sides;
    const int total_binds = y_binding + 2 + (needs_meta ? 1 : 0);

    // Tensor-backed slots: X, W, bias|0, sides..., Y.
    const int n_tensor_slots = 4 + n_sides;
    uint32_t cur_gens[16] = {};
    cur_gens[0] = generation_of(inputs[0]);
    cur_gens[1] = generation_of(inputs[1]);
    cur_gens[2] = has_bias ? generation_of(inputs[2]) : 0u;
    for (int kk = 0; kk < n_sides; ++kk) cur_gens[3 + kk] = generation_of(inputs[side_off + kk]);
    cur_gens[3 + n_sides] = generation_of(outputs[0]);
    bool bg_valid = (bool)cached_bg;
    for (int kk = 0; kk < n_tensor_slots && bg_valid; ++kk)
        if (cur_gens[kk] != cached_gen[kk]) bg_valid = false;
    if (!bg_valid) {
        std::vector<wgpu::BindGroupEntry> be(total_binds);
        be[0].binding = 0; be[0].buffer = rx->buf; be[0].offset = 0; be[0].size = rx->size;
        be[1].binding = 1; be[1].buffer = rw->buf; be[1].offset = 0; be[1].size = rw->size;
        be[2].binding = 2; be[2].buffer = b_buf;   be[2].offset = 0; be[2].size = b_sz;
        for (int kk = 0; kk < n_sides; ++kk) {
            auto* rs = find(inputs[side_off + kk]);
            be[3 + kk].binding = 3 + kk;
            be[3 + kk].buffer  = rs->buf;
            be[3 + kk].offset  = 0;
            be[3 + kk].size    = rs->size;
        }
        be[y_binding].binding = y_binding;
        be[y_binding].buffer  = ry->buf;
        be[y_binding].offset  = 0;
        be[y_binding].size    = ry->size;
        be[md_binding].binding = md_binding;
        be[md_binding].buffer  = meta_buf;
        be[md_binding].offset  = 0;
        be[md_binding].size    = 128;
        if (needs_meta) {
            const size_t rounded = ((size_t)n_sides * 16 + 15) & ~size_t(15);
            be[sd_binding].binding = sd_binding;
            be[sd_binding].buffer  = side_md;
            be[sd_binding].offset  = 0;
            be[sd_binding].size    = rounded;
        }

        wgpu::BindGroupDescriptor bgd = {};
        bgd.layout     = bgl;
        bgd.entryCount = (uint32_t)total_binds;
        bgd.entries    = be.data();
        cached_bg = dev.device.CreateBindGroup(&bgd);
        for (int kk = 0; kk < n_tensor_slots; ++kk) cached_gen[kk] = cur_gens[kk];
    }

    wgpu::ComputePassEncoder pass = shared_encoder().BeginComputePass();
    if (use_tiled) {
        pass.SetPipeline(tiled_pipeline);
        pass.SetBindGroup(0, cached_bg);
        const uint32_t TX = 8, TY = 8;
        const uint32_t gx = ((uint32_t)w_out + TX - 1) / TX;
        const uint32_t gy = ((uint32_t)h_out + TY - 1) / TY;
        const uint32_t gz = (uint32_t)(n_out * c_out);
        pass.DispatchWorkgroups(gx, gy, gz);
    } else if (use_regtile_wide) {
        pass.SetPipeline(regtile_wide_pipeline);
        pass.SetBindGroup(0, cached_bg);
        const uint32_t TX_eff = CONV_CHAIN_REG_WGX * CONV_CHAIN_REG_TILE_W_WIDE;  // 64 cols per WG
        const uint32_t TY     = CONV_CHAIN_REG_WGY;
        const uint32_t gx = ((uint32_t)w_out + TX_eff - 1) / TX_eff;
        const uint32_t gy = ((uint32_t)h_out + TY - 1) / TY;
        const uint32_t gz = (uint32_t)(n_out * c_out);
        pass.DispatchWorkgroups(gx, gy, gz);
    } else if (use_regtile_2d) {
        pass.SetPipeline(regtile_2d_pipeline);
        pass.SetBindGroup(0, cached_bg);
        const uint32_t TX_eff = CONV_CHAIN_REG_WGX * CONV_CHAIN_REG_TILE_W_2D;  // 32 cols per WG
        const uint32_t TY_eff = CONV_CHAIN_REG_WGY * CONV_CHAIN_REG_TILE_H_2D;  // 16 rows per WG
        const uint32_t gx = ((uint32_t)w_out + TX_eff - 1) / TX_eff;
        const uint32_t gy = ((uint32_t)h_out + TY_eff - 1) / TY_eff;
        const uint32_t gz = (uint32_t)(n_out * c_out);
        pass.DispatchWorkgroups(gx, gy, gz);
    } else if (use_regtile_3x3) {
        pass.SetPipeline(regtile_3x3_pipeline);
        pass.SetBindGroup(0, cached_bg);
        const uint32_t TX_eff = CONV_CHAIN_REG_WGX * CONV_CHAIN_REG_TILE_W;  // 32 cols per WG
        const uint32_t TY     = CONV_CHAIN_REG_WGY;
        const uint32_t gx = ((uint32_t)w_out + TX_eff - 1) / TX_eff;
        const uint32_t gy = ((uint32_t)h_out + TY - 1) / TY;
        const uint32_t gz = (uint32_t)(n_out * c_out);
        pass.DispatchWorkgroups(gx, gy, gz);
    } else if (use_regtile) {
        pass.SetPipeline(regtile_pipeline);
        pass.SetBindGroup(0, cached_bg);
        const uint32_t TX_eff = CONV_CHAIN_REG_WGX * CONV_CHAIN_REG_TILE_W;  // 32 cols per WG
        const uint32_t TY     = CONV_CHAIN_REG_WGY;                          // 8 rows per WG
        const uint32_t gx = ((uint32_t)w_out + TX_eff - 1) / TX_eff;
        const uint32_t gy = ((uint32_t)h_out + TY - 1) / TY;
        const uint32_t gz = (uint32_t)(n_out * c_out);
        pass.DispatchWorkgroups(gx, gy, gz);
    } else {
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, cached_bg);
        const uint32_t WG = 64;
        uint32_t groups = (total_u + WG - 1) / WG;
        pass.DispatchWorkgroups(groups, 1, 1);
    }
    pass.End();
    mark_gpu_written(outputs[0]);
    return true;
}

// ---------------------------------------------------------------------------
// layer_norm_fused_chain_t — LayerNorm (last-axis) with chain epilogue
// ---------------------------------------------------------------------------

namespace {

// Builds the LayerNorm shader with the chain epilogue spliced into the final
// output-write pass. The base 3-pass structure (sum → variance → apply) is
// preserved; we only modify the write expression. Side reads are
// `S<k>[off + j]` in Path U, or `S<k>[side_<k>_flat]` in Path M, where
// `side_<k>_flat` is computed by unflattening `i = off + j` over the full
// output shape and accumulating per-axis strides (size-1 / missing axes
// map to stride 0). See `make_chain_wgsl_meta` for the template.
std::string make_layer_norm_chain_wgsl(const std::vector<std::string>& stage_wgsl,
                                       int n_sides, bool needs_meta)
{
    std::string s =
        "struct Dims { outer : u32, N : u32, _a : u32, _b : u32 };\n"
        "struct Cfg  { eps : f32, _a : u32, _b : u32, _c : u32 };\n";
    if (needs_meta) {
        s += "struct SideMeta {\n"
             "  ndim          : u32,\n"
             "  _pad0         : u32,\n"
             "  _pad1         : u32,\n"
             "  _pad2         : u32,\n"
             "  out_dims_lo   : vec4<u32>,\n"
             "  out_dims_hi   : vec4<u32>,\n";
        for (int k = 0; k < n_sides; ++k) {
            s += "  s";
            s += std::to_string(k);
            s += "_strides_lo : vec4<u32>,\n";
            s += "  s";
            s += std::to_string(k);
            s += "_strides_hi : vec4<u32>,\n";
        }
        s += "};\n";
    }
    s += "@group(0) @binding(0) var<storage, read>       X     : array<f32>;\n"
         "@group(0) @binding(1) var<storage, read>       Scale : array<f32>;\n"
         "@group(0) @binding(2) var<storage, read>       Bias  : array<f32>;\n";
    for (int k = 0; k < n_sides; ++k) {
        s += "@group(0) @binding(";
        s += std::to_string(3 + k);
        s += ") var<storage, read>       S";
        s += std::to_string(k);
        s += " : array<f32>;\n";
    }
    const int y_binding    = 3 + n_sides;
    const int dims_binding = 4 + n_sides;
    const int cfg_binding  = 5 + n_sides;
    const int sd_binding   = 6 + n_sides;
    s += "@group(0) @binding(";
    s += std::to_string(y_binding);
    s += ") var<storage, read_write> Y     : array<f32>;\n"
         "@group(0) @binding(";
    s += std::to_string(dims_binding);
    s += ") var<uniform>             dims  : Dims;\n"
         "@group(0) @binding(";
    s += std::to_string(cfg_binding);
    s += ") var<uniform>             cfg   : Cfg;\n";
    if (needs_meta) {
        s += "@group(0) @binding(";
        s += std::to_string(sd_binding);
        s += ") var<storage, read>       sd    : SideMeta;\n";
        s += "fn get_out_dim(i : u32) -> u32 { if (i < 4u) { return sd.out_dims_lo[i]; } return sd.out_dims_hi[i - 4u]; }\n";
        for (int k = 0; k < n_sides; ++k) {
            auto ks = std::to_string(k);
            s += "fn get_s";
            s += ks;
            s += "_stride(i : u32) -> u32 { if (i < 4u) { return sd.s";
            s += ks;
            s += "_strides_lo[i]; } return sd.s";
            s += ks;
            s += "_strides_hi[i - 4u]; }\n";
        }
    }
    s += "const WG : u32 = 256u;\n"
         "var<workgroup> sh : array<f32, 256>;\n"
         "@compute @workgroup_size(256)\n"
         "fn main(@builtin(workgroup_id) wid : vec3<u32>,\n"
         "        @builtin(local_invocation_id) lid : vec3<u32>) {\n"
         "  let row = wid.x;\n"
         "  if (row >= dims.outer) { return; }\n"
         "  let N = dims.N;\n"
         "  let off = row * N;\n"
         "  let invN = 1.0 / f32(N);\n"
         "  // --- pass 1: sum ---\n"
         "  var sSum : f32 = 0.0;\n"
         "  for (var j : u32 = lid.x; j < N; j = j + WG) {\n"
         "    sSum = sSum + X[off + j];\n"
         "  }\n"
         "  sh[lid.x] = sSum;\n"
         "  workgroupBarrier();\n"
         "  var stride : u32 = WG / 2u;\n"
         "  loop {\n"
         "    if (stride == 0u) { break; }\n"
         "    if (lid.x < stride) { sh[lid.x] = sh[lid.x] + sh[lid.x + stride]; }\n"
         "    workgroupBarrier();\n"
         "    stride = stride / 2u;\n"
         "  }\n"
         "  let mean = sh[0] * invN;\n"
         "  workgroupBarrier();\n"
         "  // --- pass 2: sum of squared deltas ---\n"
         "  var ss : f32 = 0.0;\n"
         "  for (var j : u32 = lid.x; j < N; j = j + WG) {\n"
         "    let d = X[off + j] - mean;\n"
         "    ss = ss + d * d;\n"
         "  }\n"
         "  sh[lid.x] = ss;\n"
         "  workgroupBarrier();\n"
         "  stride = WG / 2u;\n"
         "  loop {\n"
         "    if (stride == 0u) { break; }\n"
         "    if (lid.x < stride) { sh[lid.x] = sh[lid.x] + sh[lid.x + stride]; }\n"
         "    workgroupBarrier();\n"
         "    stride = stride / 2u;\n"
         "  }\n"
         "  let variance = sh[0] * invN;\n"
         "  let inv_std = 1.0 / sqrt(variance + cfg.eps);\n"
         "  workgroupBarrier();\n"
         "  // --- pass 3: write normalized output + chain epilogue ---\n"
         "  for (var j : u32 = lid.x; j < N; j = j + WG) {\n"
         "    let i = off + j;\n"
         "    let normalized = (X[i] - mean) * inv_std;\n"
         "    var v : f32 = normalized * Scale[j] + Bias[j];\n";
    if (needs_meta) {
        for (int k = 0; k < n_sides; ++k) {
            s += "    var side_";
            s += std::to_string(k);
            s += "_flat : u32 = 0u;\n";
        }
        s += "    {\n"
             "      var tmp : u32 = i;\n"
             "      for (var k : i32 = i32(sd.ndim) - 1; k >= 0; k = k - 1) {\n"
             "        let d = get_out_dim(u32(k));\n"
             "        let idx = tmp % d;\n"
             "        tmp = tmp / d;\n";
        for (int k = 0; k < n_sides; ++k) {
            auto ks = std::to_string(k);
            s += "        side_";
            s += ks;
            s += "_flat = side_";
            s += ks;
            s += "_flat + idx * get_s";
            s += ks;
            s += "_stride(u32(k));\n";
        }
        s += "      }\n"
             "    }\n";
    }
    for (const auto& stg : stage_wgsl) {
        s += "    v = ";
        s += stg;
        s += ";\n";
    }
    s += "    Y[i] = v;\n"
         "  }\n"
         "}\n";
    return s;
}

wgpu::BindGroupLayout make_layer_norm_chain_bgl(int n_sides, bool needs_meta)
{
    auto& dev = get_device();
    // X, Scale, Bias, sides..., Y, dims, cfg [, sd]
    const int total = 6 + n_sides + (needs_meta ? 1 : 0);
    std::vector<wgpu::BindGroupLayoutEntry> e(total);
    int idx = 0;
    // X, Scale, Bias, sides — ReadOnlyStorage.
    for (int k = 0; k < 3 + n_sides; ++k, ++idx) {
        e[idx].binding = idx;
        e[idx].visibility = wgpu::ShaderStage::Compute;
        e[idx].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    }
    // Y — Storage.
    e[idx].binding = idx;
    e[idx].visibility = wgpu::ShaderStage::Compute;
    e[idx].buffer.type = wgpu::BufferBindingType::Storage;
    ++idx;
    // dims — Uniform.
    e[idx].binding = idx;
    e[idx].visibility = wgpu::ShaderStage::Compute;
    e[idx].buffer.type = wgpu::BufferBindingType::Uniform;
    ++idx;
    // cfg — Uniform.
    e[idx].binding = idx;
    e[idx].visibility = wgpu::ShaderStage::Compute;
    e[idx].buffer.type = wgpu::BufferBindingType::Uniform;
    ++idx;
    if (needs_meta) {
        // sd — ReadOnlyStorage (out_dims + per-side stride tables).
        e[idx].binding = idx;
        e[idx].visibility = wgpu::ShaderStage::Compute;
        e[idx].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        ++idx;
    }
    wgpu::BindGroupLayoutDescriptor d = {};
    d.entryCount = (uint32_t)total;
    d.entries = e.data();
    return dev.device.CreateBindGroupLayout(&d);
}

// Side-meta layout (storage buffer) for LN / Softmax Path M:
//   [0..4)     ndim
//   [4..16)    pad
//   [16..48)   out_dims  (8 × u32)
//   [48..80)   side_0 strides
//   [80..112)  side_1 strides
//   ...
//   48 + 32*k  side_k strides
// Matches fused_elementwise_chain_t's Path M layout minus the `total` u32
// in slot 0 (total is implicit — LN/Softmax already have outer*N).
constexpr size_t LN_SM_MD_HEADER = 48;
constexpr size_t LN_SM_MD_STRIDE_TABLE = 32;

inline size_t ln_sm_side_md_bytes(int n_sides) {
    const size_t raw = LN_SM_MD_HEADER + LN_SM_MD_STRIDE_TABLE * (size_t)n_sides;
    return (raw + 15) & ~size_t(15);
}

struct ln_chain_cache_entry_t {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
};

std::unordered_map<std::string, ln_chain_cache_entry_t>& ln_chain_cache()
{
    static std::unordered_map<std::string, ln_chain_cache_entry_t> cache;
    return cache;
}

} // namespace

bool layer_norm_fused_chain_t::init() {
    if (stage_wgsl.empty())             return false;
    if (!device_ready())                return false;
    if (needs_meta && n_sides < 1)      return false;
    const int required = 2 + (has_bias ? 1 : 0) + n_sides;
    if ((int)inputs.size() != required) return false;
    if (outputs.size() != 1)            return false;

    axis_attr = attribute(attr_key_t::axis, (int64_t)-1);
    epsilon   = attribute(attr_key_t::epsilon, 1e-5f);

    std::string sig;
    sig += needs_meta ? 'M' : 'U';
    sig += std::to_string(n_sides);
    sig += '\x02';
    for (const auto& s : stage_wgsl) { sig += s; sig += '\x01'; }

    auto& cache = ln_chain_cache();
    auto it = cache.find(sig);
    if (it == cache.end()) {
        auto src = make_layer_norm_chain_wgsl(stage_wgsl, n_sides, needs_meta);
        auto sm  = compile(src);
        ln_chain_cache_entry_t e;
        e.bgl      = make_layer_norm_chain_bgl(n_sides, needs_meta);
        e.pipeline = make_pipeline(sm, e.bgl);
        it = cache.emplace(std::move(sig), std::move(e)).first;
    }
    bgl      = it->second.bgl;
    pipeline = it->second.pipeline;

    auto& dev = get_device();
    wgpu::BufferDescriptor ud = {};
    ud.size = 16;
    ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    uniforms_dims = dev.device.CreateBuffer(&ud);
    uniforms_cfg  = dev.device.CreateBuffer(&ud);

    if (needs_meta) {
        wgpu::BufferDescriptor sd = {};
        sd.size  = ln_sm_side_md_bytes(n_sides);
        sd.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        side_md  = dev.device.CreateBuffer(&sd);
    }
    return true;
}

bool layer_norm_fused_chain_t::reshape() {
    const int required = 2 + (has_bias ? 1 : 0) + n_sides;
    if ((int)inputs.size() != required) return false;
    if (outputs.size() != 1)            return false;

    const tensor_t* x     = inputs[0];
    const tensor_t* scale = inputs[1];
    const tensor_t* bias  = has_bias ? inputs[2] : nullptr;
    tensor_t*       y     = outputs[0];

    if (!x || !scale || !y)                         return false;
    if (x->type != NNR_DATA_TYPE_FLOAT32)           return false;
    if (scale->type != NNR_DATA_TYPE_FLOAT32)       return false;
    if (bias && bias->type != NNR_DATA_TYPE_FLOAT32) return false;
    if (needs_meta && x->ndim > 8)                  return false;

    int caxis = axis_attr < 0 ? (int)axis_attr + x->ndim : (int)axis_attr;
    if (caxis < 0 || caxis >= x->ndim)              return false;

    outer = 1;
    for (int i = 0; i < caxis; ++i)    outer *= x->dims[i];
    N = 1;
    for (int i = caxis; i < x->ndim; ++i) N *= x->dims[i];
    if ((int)scale->ndata != N)                     return false;
    if (bias && (int)bias->ndata != N)              return false;

    if (!y->reshape_identity(x))                    return false;

    // Validate sides. Path U: same shape as X. Path M: side.ndim ≤ x.ndim
    // and each right-aligned side axis is 1 or equal to x's axis at that
    // position (NumPy/ONNX broadcast-into-pipe).
    const int side_off = 2 + (has_bias ? 1 : 0);
    for (int kk = 0; kk < n_sides; ++kk) {
        const tensor_t* s = inputs[side_off + kk];
        if (!s)                                       return false;
        if (s->type != NNR_DATA_TYPE_FLOAT32)         return false;
        bool exact = (s->ndim == x->ndim && s->ndata == x->ndata);
        if (exact) {
            for (int i = 0; i < s->ndim; ++i)
                if (s->dims[i] != x->dims[i]) { exact = false; break; }
        }
        if (!exact) {
            if (!needs_meta)                          return false;
            if (s->ndim > x->ndim)                    return false;
            int off = x->ndim - s->ndim;
            for (int i = 0; i < s->ndim; ++i) {
                int d_side = s->dims[i];
                int d_pipe = x->dims[off + i];
                if (d_side != 1 && d_side != d_pipe)  return false;
            }
        }
    }

    ensure_buffer(x, x->ndata * sizeof(float));
    auto& sr = ensure_buffer(scale, N * sizeof(float));
    if (ctx && ctx->initializer_names.count(scale->name)) sr.is_weight = true;
    if (bias) {
        auto& br = ensure_buffer(bias, N * sizeof(float));
        if (ctx && ctx->initializer_names.count(bias->name)) br.is_weight = true;
    }
    for (int kk = 0; kk < n_sides; ++kk)
        ensure_buffer(inputs[side_off + kk],
                      inputs[side_off + kk]->ndata * sizeof(float));
    ensure_buffer(y, y->ndata * sizeof(float));

    if (!has_bias) {
        auto& dev = get_device();
        wgpu::BufferDescriptor bd = {};
        bd.size = ((size_t)N * sizeof(float) + 3u) & ~size_t{3};
        bd.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        zero_bias = dev.device.CreateBuffer(&bd);
        std::vector<float> zeros((size_t)N, 0.0f);
        dev.queue.WriteBuffer(zero_bias, 0, zeros.data(), (size_t)N * sizeof(float));
    }

    // Uniform / meta payloads are pure functions of shape + epsilon.
    auto& dev = get_device();
    uint32_t dims[4] = { (uint32_t)outer, (uint32_t)N, 0, 0 };
    dev.queue.WriteBuffer(uniforms_dims, 0, dims, sizeof(dims));
    float cfg[4] = { epsilon, 0, 0, 0 };
    dev.queue.WriteBuffer(uniforms_cfg, 0, cfg, sizeof(cfg));

    if (needs_meta) {
        const int side_off = 2 + (has_bias ? 1 : 0);
        const tensor_t* x = inputs[0];
        const size_t sz = ln_sm_side_md_bytes(n_sides);
        std::vector<uint8_t> buf(sz, 0);
        auto put_u32 = [&](size_t off, uint32_t v) {
            std::memcpy(buf.data() + off, &v, 4);
        };
        put_u32(0, (uint32_t)x->ndim);
        for (int i = 0; i < x->ndim; ++i) put_u32(16 + i * 4, (uint32_t)x->dims[i]);
        for (int kk = 0; kk < n_sides; ++kk) {
            const tensor_t* s = inputs[side_off + kk];
            uint32_t nat[8] = {};
            uint32_t st = 1;
            for (int a = s->ndim - 1; a >= 0; --a) {
                nat[a] = st;
                st *= (uint32_t)s->dims[a];
            }
            const int off = x->ndim - s->ndim;
            for (int a = 0; a < x->ndim; ++a) {
                int sa = a - off;
                uint32_t stride = 0;
                if (sa >= 0 && s->dims[sa] != 1) stride = nat[sa];
                put_u32(LN_SM_MD_HEADER + LN_SM_MD_STRIDE_TABLE * (size_t)kk + a * 4, stride);
            }
        }
        dev.queue.WriteBuffer(side_md, 0, buf.data(), buf.size());
    }
    return true;
}

bool layer_norm_fused_chain_t::exec() {
    auto& dev = get_device();
    upload_if_needed(inputs[0]);
    upload_if_needed(inputs[1]);
    if (has_bias) upload_if_needed(inputs[2]);
    const int side_off = 2 + (has_bias ? 1 : 0);
    for (int kk = 0; kk < n_sides; ++kk) upload_if_needed(inputs[side_off + kk]);

    // uniforms_dims, uniforms_cfg, side_md were written in reshape().
    auto* rx  = find(inputs[0]);
    auto* rs  = find(inputs[1]);
    auto* ry  = find(outputs[0]);
    wgpu::Buffer bias_buf; uint64_t bias_size;
    if (has_bias) {
        auto* rb = find(inputs[2]);
        bias_buf  = rb->buf;
        bias_size = rb->size;
    } else {
        bias_buf  = zero_bias;
        bias_size = (uint64_t)N * sizeof(float);
    }

    const int y_binding    = 3 + n_sides;
    const int dims_binding = 4 + n_sides;
    const int cfg_binding  = 5 + n_sides;
    const int sd_binding   = 6 + n_sides;
    const int total_binds  = 6 + n_sides + (needs_meta ? 1 : 0);

    // Tensor-backed slots: X, scale, bias|0, sides..., Y.
    const int n_tensor_slots = 4 + n_sides;
    uint32_t cur_gens[16] = {};
    cur_gens[0] = generation_of(inputs[0]);
    cur_gens[1] = generation_of(inputs[1]);
    cur_gens[2] = has_bias ? generation_of(inputs[2]) : 0u;
    for (int kk = 0; kk < n_sides; ++kk) cur_gens[3 + kk] = generation_of(inputs[side_off + kk]);
    cur_gens[3 + n_sides] = generation_of(outputs[0]);
    bool bg_valid = (bool)cached_bg;
    for (int kk = 0; kk < n_tensor_slots && bg_valid; ++kk)
        if (cur_gens[kk] != cached_gen[kk]) bg_valid = false;
    if (!bg_valid) {
        std::vector<wgpu::BindGroupEntry> be(total_binds);
        be[0].binding = 0; be[0].buffer = rx->buf;   be[0].offset = 0; be[0].size = rx->size;
        be[1].binding = 1; be[1].buffer = rs->buf;   be[1].offset = 0; be[1].size = rs->size;
        be[2].binding = 2; be[2].buffer = bias_buf;  be[2].offset = 0; be[2].size = bias_size;
        for (int kk = 0; kk < n_sides; ++kk) {
            auto* rsd = find(inputs[side_off + kk]);
            be[3 + kk].binding = 3 + kk;
            be[3 + kk].buffer  = rsd->buf;
            be[3 + kk].offset  = 0;
            be[3 + kk].size    = rsd->size;
        }
        be[y_binding].binding    = y_binding;
        be[y_binding].buffer     = ry->buf;
        be[y_binding].offset     = 0;
        be[y_binding].size       = ry->size;
        be[dims_binding].binding = dims_binding;
        be[dims_binding].buffer  = uniforms_dims;
        be[dims_binding].offset  = 0;
        be[dims_binding].size    = 16;
        be[cfg_binding].binding  = cfg_binding;
        be[cfg_binding].buffer   = uniforms_cfg;
        be[cfg_binding].offset   = 0;
        be[cfg_binding].size     = 16;
        if (needs_meta) {
            be[sd_binding].binding = sd_binding;
            be[sd_binding].buffer  = side_md;
            be[sd_binding].offset  = 0;
            be[sd_binding].size    = ln_sm_side_md_bytes(n_sides);
        }

        wgpu::BindGroupDescriptor bgd = {};
        bgd.layout     = bgl;
        bgd.entryCount = (uint32_t)total_binds;
        bgd.entries    = be.data();
        cached_bg = dev.device.CreateBindGroup(&bgd);
        for (int kk = 0; kk < n_tensor_slots; ++kk) cached_gen[kk] = cur_gens[kk];
    }

    wgpu::ComputePassEncoder pass = shared_encoder().BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, cached_bg);
    pass.DispatchWorkgroups((uint32_t)outer, 1, 1);
    pass.End();
    mark_gpu_written(outputs[0]);
    return true;
}

// ---------------------------------------------------------------------------
// softmax_fused_chain_t — Softmax (last-axis) with chain epilogue
// ---------------------------------------------------------------------------

namespace {

// Softmax with an absorbed chain. The base 3-pass structure (rowmax →
// exp-and-sum → divide) is preserved; the chain stages run on each output
// element during the third pass, between the divide and the write. Path M
// (broadcast sides) adds an unflatten-and-stride prelude inside pass 3
// that computes each `side_<k>_flat` from the linear output index
// `i = base + j*stride`, using the full X shape (ndim + out_dims) and
// per-side stride tables in a separate storage meta buffer.
std::string make_softmax_chain_wgsl(const std::vector<std::string>& stage_wgsl,
                                    int n_sides, bool needs_meta)
{
    std::string s =
        "struct Dims { outer : u32, N : u32, inner : u32, _a : u32 };\n";
    if (needs_meta) {
        s += "struct SideMeta {\n"
             "  ndim          : u32,\n"
             "  _pad0         : u32,\n"
             "  _pad1         : u32,\n"
             "  _pad2         : u32,\n"
             "  out_dims_lo   : vec4<u32>,\n"
             "  out_dims_hi   : vec4<u32>,\n";
        for (int k = 0; k < n_sides; ++k) {
            s += "  s";
            s += std::to_string(k);
            s += "_strides_lo : vec4<u32>,\n";
            s += "  s";
            s += std::to_string(k);
            s += "_strides_hi : vec4<u32>,\n";
        }
        s += "};\n";
    }
    s += "@group(0) @binding(0) var<storage, read>       X    : array<f32>;\n";
    for (int k = 0; k < n_sides; ++k) {
        s += "@group(0) @binding(";
        s += std::to_string(1 + k);
        s += ") var<storage, read>       S";
        s += std::to_string(k);
        s += " : array<f32>;\n";
    }
    const int y_binding    = 1 + n_sides;
    const int dims_binding = 2 + n_sides;
    const int sd_binding   = 3 + n_sides;
    s += "@group(0) @binding(";
    s += std::to_string(y_binding);
    s += ") var<storage, read_write> Y    : array<f32>;\n"
         "@group(0) @binding(";
    s += std::to_string(dims_binding);
    s += ") var<uniform>             dims : Dims;\n";
    if (needs_meta) {
        s += "@group(0) @binding(";
        s += std::to_string(sd_binding);
        s += ") var<storage, read>       sd   : SideMeta;\n";
        s += "fn get_out_dim(i : u32) -> u32 { if (i < 4u) { return sd.out_dims_lo[i]; } return sd.out_dims_hi[i - 4u]; }\n";
        for (int k = 0; k < n_sides; ++k) {
            auto ks = std::to_string(k);
            s += "fn get_s";
            s += ks;
            s += "_stride(i : u32) -> u32 { if (i < 4u) { return sd.s";
            s += ks;
            s += "_strides_lo[i]; } return sd.s";
            s += ks;
            s += "_strides_hi[i - 4u]; }\n";
        }
    }
    s += "const WG : u32 = 256u;\n"
         "var<workgroup> sh : array<f32, 256>;\n"
         "@compute @workgroup_size(256)\n"
         "fn main(@builtin(workgroup_id) wid : vec3<u32>,\n"
         "        @builtin(local_invocation_id) lid : vec3<u32>) {\n"
         "  let o = wid.x;\n"
         "  let p = wid.y;\n"
         "  if (o >= dims.outer || p >= dims.inner) { return; }\n"
         "  let N      = dims.N;\n"
         "  let stride = dims.inner;\n"
         "  let base   = o * N * stride + p;\n"
         "  // --- 1. rowwise max ---\n"
         "  var m : f32 = -3.4e38;\n"
         "  for (var j : u32 = lid.x; j < N; j = j + WG) {\n"
         "    m = max(m, X[base + j * stride]);\n"
         "  }\n"
         "  sh[lid.x] = m;\n"
         "  workgroupBarrier();\n"
         "  var s : u32 = WG / 2u;\n"
         "  loop {\n"
         "    if (s == 0u) { break; }\n"
         "    if (lid.x < s) { sh[lid.x] = max(sh[lid.x], sh[lid.x + s]); }\n"
         "    workgroupBarrier();\n"
         "    s = s / 2u;\n"
         "  }\n"
         "  let row_max = sh[0];\n"
         "  workgroupBarrier();\n"
         "  // --- 2. exp(x - max), store to Y, accumulate sum ---\n"
         "  var sum : f32 = 0.0;\n"
         "  for (var j : u32 = lid.x; j < N; j = j + WG) {\n"
         "    let i = base + j * stride;\n"
         "    let e = exp(X[i] - row_max);\n"
         "    Y[i] = e;\n"
         "    sum = sum + e;\n"
         "  }\n"
         "  sh[lid.x] = sum;\n"
         "  workgroupBarrier();\n"
         "  s = WG / 2u;\n"
         "  loop {\n"
         "    if (s == 0u) { break; }\n"
         "    if (lid.x < s) { sh[lid.x] = sh[lid.x] + sh[lid.x + s]; }\n"
         "    workgroupBarrier();\n"
         "    s = s / 2u;\n"
         "  }\n"
         "  let row_sum = sh[0];\n"
         "  let inv = 1.0 / row_sum;\n"
         "  workgroupBarrier();\n"
         "  // --- 3. normalize + chain epilogue ---\n"
         "  for (var j : u32 = lid.x; j < N; j = j + WG) {\n"
         "    let i = base + j * stride;\n"
         "    var v : f32 = Y[i] * inv;\n";
    if (needs_meta) {
        for (int k = 0; k < n_sides; ++k) {
            s += "    var side_";
            s += std::to_string(k);
            s += "_flat : u32 = 0u;\n";
        }
        s += "    {\n"
             "      var tmp : u32 = i;\n"
             "      for (var k : i32 = i32(sd.ndim) - 1; k >= 0; k = k - 1) {\n"
             "        let d = get_out_dim(u32(k));\n"
             "        let idx = tmp % d;\n"
             "        tmp = tmp / d;\n";
        for (int k = 0; k < n_sides; ++k) {
            auto ks = std::to_string(k);
            s += "        side_";
            s += ks;
            s += "_flat = side_";
            s += ks;
            s += "_flat + idx * get_s";
            s += ks;
            s += "_stride(u32(k));\n";
        }
        s += "      }\n"
             "    }\n";
    }
    for (const auto& stg : stage_wgsl) {
        s += "    v = ";
        s += stg;
        s += ";\n";
    }
    s += "    Y[i] = v;\n"
         "  }\n"
         "}\n";
    return s;
}

wgpu::BindGroupLayout make_softmax_chain_bgl(int n_sides, bool needs_meta)
{
    auto& dev = get_device();
    // X, sides..., Y, dims [, sd]
    const int total = 3 + n_sides + (needs_meta ? 1 : 0);
    std::vector<wgpu::BindGroupLayoutEntry> e(total);
    int idx = 0;
    // X + sides — ReadOnlyStorage.
    for (int k = 0; k < 1 + n_sides; ++k, ++idx) {
        e[idx].binding = idx;
        e[idx].visibility = wgpu::ShaderStage::Compute;
        e[idx].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    }
    // Y — Storage.
    e[idx].binding = idx;
    e[idx].visibility = wgpu::ShaderStage::Compute;
    e[idx].buffer.type = wgpu::BufferBindingType::Storage;
    ++idx;
    // dims — Uniform.
    e[idx].binding = idx;
    e[idx].visibility = wgpu::ShaderStage::Compute;
    e[idx].buffer.type = wgpu::BufferBindingType::Uniform;
    ++idx;
    if (needs_meta) {
        // sd — ReadOnlyStorage (out_dims + per-side stride tables).
        e[idx].binding = idx;
        e[idx].visibility = wgpu::ShaderStage::Compute;
        e[idx].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        ++idx;
    }
    wgpu::BindGroupLayoutDescriptor d = {};
    d.entryCount = (uint32_t)total;
    d.entries = e.data();
    return dev.device.CreateBindGroupLayout(&d);
}

struct sm_chain_cache_entry_t {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
};

std::unordered_map<std::string, sm_chain_cache_entry_t>& sm_chain_cache()
{
    static std::unordered_map<std::string, sm_chain_cache_entry_t> cache;
    return cache;
}

} // namespace

bool softmax_fused_chain_t::init() {
    if (stage_wgsl.empty())                  return false;
    if (!device_ready())                     return false;
    if (needs_meta && n_sides < 1)           return false;
    const int required = 1 + n_sides;
    if ((int)inputs.size() != required)      return false;
    if (outputs.size() != 1)                 return false;

    axis_attr = (int)attribute(attr_key_t::axis, (int64_t)-1);

    std::string sig;
    sig += needs_meta ? 'M' : 'U';
    sig += std::to_string(n_sides);
    sig += '\x02';
    for (const auto& s : stage_wgsl) { sig += s; sig += '\x01'; }

    auto& cache = sm_chain_cache();
    auto it = cache.find(sig);
    if (it == cache.end()) {
        auto src = make_softmax_chain_wgsl(stage_wgsl, n_sides, needs_meta);
        auto sm  = compile(src);
        sm_chain_cache_entry_t e;
        e.bgl      = make_softmax_chain_bgl(n_sides, needs_meta);
        e.pipeline = make_pipeline(sm, e.bgl);
        it = cache.emplace(std::move(sig), std::move(e)).first;
    }
    bgl      = it->second.bgl;
    pipeline = it->second.pipeline;

    auto& dev = get_device();
    wgpu::BufferDescriptor ud = {};
    ud.size = 16;
    ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    uniforms = dev.device.CreateBuffer(&ud);

    if (needs_meta) {
        wgpu::BufferDescriptor sd = {};
        sd.size  = ln_sm_side_md_bytes(n_sides);
        sd.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        side_md  = dev.device.CreateBuffer(&sd);
    }
    return true;
}

bool softmax_fused_chain_t::reshape() {
    const int required = 1 + n_sides;
    if ((int)inputs.size() != required) return false;
    if (outputs.size() != 1)            return false;

    const tensor_t* x = inputs[0];
    tensor_t*       y = outputs[0];
    if (!x || !y)                       return false;
    if (x->type != NNR_DATA_TYPE_FLOAT32) return false;
    if (needs_meta && x->ndim > 8)      return false;

    int caxis = axis_attr < 0 ? axis_attr + x->ndim : axis_attr;
    if (caxis < 0 || caxis >= x->ndim)  return false;

    outer = 1;
    for (int i = 0; i < caxis; ++i)           outer *= x->dims[i];
    N = x->dims[caxis];
    inner = 1;
    for (int i = caxis + 1; i < x->ndim; ++i) inner *= x->dims[i];

    if (!y->reshape_identity(x))        return false;

    // Validate sides. Path U: same shape as X. Path M: side.ndim ≤ x.ndim
    // and each right-aligned side axis is 1 or equal to x's axis at that
    // position (NumPy/ONNX broadcast-into-pipe).
    for (int kk = 0; kk < n_sides; ++kk) {
        const tensor_t* s = inputs[1 + kk];
        if (!s)                                      return false;
        if (s->type != NNR_DATA_TYPE_FLOAT32)        return false;
        bool exact = (s->ndim == x->ndim && s->ndata == x->ndata);
        if (exact) {
            for (int i = 0; i < s->ndim; ++i)
                if (s->dims[i] != x->dims[i]) { exact = false; break; }
        }
        if (!exact) {
            if (!needs_meta)                         return false;
            if (s->ndim > x->ndim)                   return false;
            int off = x->ndim - s->ndim;
            for (int i = 0; i < s->ndim; ++i) {
                int d_side = s->dims[i];
                int d_pipe = x->dims[off + i];
                if (d_side != 1 && d_side != d_pipe) return false;
            }
        }
    }

    ensure_buffer(x, x->ndata * sizeof(float));
    for (int kk = 0; kk < n_sides; ++kk)
        ensure_buffer(inputs[1 + kk], inputs[1 + kk]->ndata * sizeof(float));
    ensure_buffer(y, y->ndata * sizeof(float));

    // Uniform / meta payloads are pure functions of shape.
    auto& dev = get_device();
    uint32_t u[4] = { (uint32_t)outer, (uint32_t)N, (uint32_t)inner, 0 };
    dev.queue.WriteBuffer(uniforms, 0, u, sizeof(u));
    if (needs_meta) {
        const size_t sz = ln_sm_side_md_bytes(n_sides);
        std::vector<uint8_t> buf(sz, 0);
        auto put_u32 = [&](size_t off, uint32_t v) {
            std::memcpy(buf.data() + off, &v, 4);
        };
        put_u32(0, (uint32_t)x->ndim);
        for (int i = 0; i < x->ndim; ++i) put_u32(16 + i * 4, (uint32_t)x->dims[i]);
        for (int kk = 0; kk < n_sides; ++kk) {
            const tensor_t* s = inputs[1 + kk];
            uint32_t nat[8] = {};
            uint32_t st = 1;
            for (int a = s->ndim - 1; a >= 0; --a) {
                nat[a] = st;
                st *= (uint32_t)s->dims[a];
            }
            const int off = x->ndim - s->ndim;
            for (int a = 0; a < x->ndim; ++a) {
                int sa = a - off;
                uint32_t stride = 0;
                if (sa >= 0 && s->dims[sa] != 1) stride = nat[sa];
                put_u32(LN_SM_MD_HEADER + LN_SM_MD_STRIDE_TABLE * (size_t)kk + a * 4, stride);
            }
        }
        dev.queue.WriteBuffer(side_md, 0, buf.data(), buf.size());
    }
    return true;
}

bool softmax_fused_chain_t::exec() {
    auto& dev = get_device();
    upload_if_needed(inputs[0]);
    for (int kk = 0; kk < n_sides; ++kk) upload_if_needed(inputs[1 + kk]);

    // uniforms and side_md were written in reshape().
    const int y_binding    = 1 + n_sides;
    const int dims_binding = 2 + n_sides;
    const int sd_binding   = 3 + n_sides;
    const int total_binds  = 3 + n_sides + (needs_meta ? 1 : 0);

    // Tensor-backed slots: X, sides..., Y.
    const int n_tensor_slots = 2 + n_sides;
    uint32_t cur_gens[16] = {};
    cur_gens[0] = generation_of(inputs[0]);
    for (int kk = 0; kk < n_sides; ++kk) cur_gens[1 + kk] = generation_of(inputs[1 + kk]);
    cur_gens[1 + n_sides] = generation_of(outputs[0]);
    bool bg_valid = (bool)cached_bg;
    for (int kk = 0; kk < n_tensor_slots && bg_valid; ++kk)
        if (cur_gens[kk] != cached_gen[kk]) bg_valid = false;
    if (!bg_valid) {
        std::vector<wgpu::BindGroupEntry> be(total_binds);
        auto* rx = find(inputs[0]);
        be[0].binding = 0; be[0].buffer = rx->buf; be[0].offset = 0; be[0].size = rx->size;
        for (int kk = 0; kk < n_sides; ++kk) {
            auto* rs = find(inputs[1 + kk]);
            be[1 + kk].binding = 1 + kk;
            be[1 + kk].buffer  = rs->buf;
            be[1 + kk].offset  = 0;
            be[1 + kk].size    = rs->size;
        }
        auto* ry = find(outputs[0]);
        be[y_binding].binding    = y_binding;
        be[y_binding].buffer     = ry->buf;
        be[y_binding].offset     = 0;
        be[y_binding].size       = ry->size;
        be[dims_binding].binding = dims_binding;
        be[dims_binding].buffer  = uniforms;
        be[dims_binding].offset  = 0;
        be[dims_binding].size    = 16;
        if (needs_meta) {
            be[sd_binding].binding = sd_binding;
            be[sd_binding].buffer  = side_md;
            be[sd_binding].offset  = 0;
            be[sd_binding].size    = ln_sm_side_md_bytes(n_sides);
        }

        wgpu::BindGroupDescriptor bgd = {};
        bgd.layout     = bgl;
        bgd.entryCount = (uint32_t)total_binds;
        bgd.entries    = be.data();
        cached_bg = dev.device.CreateBindGroup(&bgd);
        for (int kk = 0; kk < n_tensor_slots; ++kk) cached_gen[kk] = cur_gens[kk];
    }

    wgpu::ComputePassEncoder pass = shared_encoder().BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, cached_bg);
    pass.DispatchWorkgroups((uint32_t)outer, (uint32_t)inner, 1);
    pass.End();
    mark_gpu_written(outputs[0]);
    return true;
}

} // namespace nnr::webgpu
