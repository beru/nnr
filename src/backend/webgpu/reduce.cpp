#include "reduce.h"

#include "device.h"
#include "buffer.h"
#include "attr_key.h"

#include <cstring>
#include <string>

namespace nnr::webgpu {

namespace {

constexpr uint32_t WG = 64;

// One WGSL source per (init, transform, merge, finalize) tuple —
// runtime-compiled by the base class so every reduce op gets its own
// pipeline with no switching in the kernel.
//
// Kernel scheme: one WORKGROUP per output element, 64 threads cooperate on
// the reduce domain with strided accumulation + workgroup tree reduction.
// Stream step: acc = merge(acc, transform(v)). Tree step: a, b in shared
// memory merged pairwise. Scales roughly linearly with red_count up to
// ~64x parallelism per output, vs the previous 1-thread-per-output scalar
// loop.
std::string make_reduce_wgsl(const char* init, const char* transform,
                             const char* merge, const char* finalize) {
    std::string s =
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
        "var<workgroup> wg_tmp : array<f32, 64>;\n"
        "@compute @workgroup_size(64)\n"
        "fn main(@builtin(workgroup_id) wgid : vec3<u32>,\n"
        "        @builtin(local_invocation_id) lid : vec3<u32>) {\n"
        // Flatten the 2D workgroup grid — host dispatches up to 65535 on
        // the X axis then wraps to Y so large `total` values still fit.
        "  let o  = wgid.y * 65535u + wgid.x;\n"
        "  let tx = lid.x;\n"
        // For out-of-range workgroups (total not a multiple of WG), skip
        // the main body but still participate in the tree reduction with
        // init values so the barriers remain balanced.
        "  var in_range : bool = (o < md.total);\n"
        // Base input flat index from keep-axes of the output coord. Reduce
        // axes are skipped during the unflatten loop — their contribution
        // comes from the inner reduction loop.
        "  var base : u32 = 0u;\n"
        "  if (in_range) {\n"
        "    var tmp : u32 = o;\n"
        "    for (var k : i32 = i32(md.ndim) - 1; k >= 0; k = k - 1) {\n"
        "      if (get_is_reduce(u32(k)) != 0u) { continue; }\n"
        "      let d = get_in_dim(u32(k));\n"
        "      let c = tmp % d;\n"
        "      tmp   = tmp / d;\n"
        "      base  = base + c * get_in_stride(u32(k));\n"
        "    }\n"
        "  }\n"
        // Each thread walks its strided subset of the reduce domain.
        "  var acc : f32 = ";
    s += init;
    s += ";\n"
        "  if (in_range) {\n"
        "    for (var r : u32 = tx; r < md.red_count; r = r + 64u) {\n"
        "      var rtmp   : u32 = r;\n"
        "      var x_flat : u32 = base;\n"
        "      for (var k : i32 = i32(md.ndim) - 1; k >= 0; k = k - 1) {\n"
        "        if (get_is_reduce(u32(k)) == 0u) { continue; }\n"
        "        let d = get_in_dim(u32(k));\n"
        "        let c = rtmp % d;\n"
        "        rtmp  = rtmp / d;\n"
        "        x_flat = x_flat + c * get_in_stride(u32(k));\n"
        "      }\n"
        "      let v : f32 = X[x_flat];\n"
        "      let b : f32 = ";      // stream step: transform v -> b, then merge(acc, b)
    s += transform;
    s += ";\n"
        "      let a : f32 = acc;\n"
        "      acc = ";
    s += merge;
    s += ";\n"
        "    }\n"
        "  }\n"
        "  wg_tmp[tx] = acc;\n"
        "  workgroupBarrier();\n"
        // Tree reduction within the workgroup. All threads participate
        // (in-range check above ensures out-of-range threads have the
        // init sentinel, which is a no-op for sum / max / min / prod).
        "  var stride : u32 = 32u;\n"
        "  loop {\n"
        "    if (stride == 0u) { break; }\n"
        "    if (tx < stride) {\n"
        "      let a : f32 = wg_tmp[tx];\n"
        "      let b : f32 = wg_tmp[tx + stride];\n"
        "      wg_tmp[tx] = ";
    s += merge;
    s += ";\n"
        "    }\n"
        "    workgroupBarrier();\n"
        "    stride = stride / 2u;\n"
        "  }\n"
        "  if (tx == 0u && in_range) {\n"
        "    let acc : f32 = wg_tmp[0];\n"
        "    let n   : u32 = md.red_count;\n"
        "    Y[o] = ";
    s += finalize;
    s += ";\n  }\n}\n";
    return s;
}

int64_t read_axes_index(const tensor_t* t, int i) {
    if (t->type == NNR_DATA_TYPE_INT32) return ((const int32_t*)t->data)[i];
    return ((const int64_t*)t->data)[i];
}

} // namespace

bool reduce_elementwise_t::init() {
    if (outputs.size() != 1) return false;
    if (inputs.size() < 1 || inputs.size() > 2) return false;
    if (!device_ready()) return false;

    auto& dev = get_device();
    std::string src = make_reduce_wgsl(init_expr(), transform_expr(),
                                       merge_expr(), finalize_expr());
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

    // Header (16B) + 3 u32[8] arrays (96B) = 112B, rounded to 128.
    wgpu::BufferDescriptor md = {};
    md.size  = 128;
    md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    meta_buf = dev.device.CreateBuffer(&md);
    return true;
}

bool reduce_elementwise_t::reshape() {
    const tensor_t* x = inputs[0];
    tensor_t*       y = outputs[0];
    if (x->type != NNR_DATA_TYPE_FLOAT32) return false;
    int dim = x->ndim;
    if (dim <= 0 || dim > 8) return false;
    ndim = (uint32_t)dim;

    // axes: from input 1 (opset 13+ for ReduceSum, 18+ for the others) or
    // from attribute. Missing/empty axes means reduce over all axes, unless
    // opset 18+'s noop_with_empty_axes is set.
    bool axes_set[8] = {};
    int  axes_count  = 0;

    auto set_axis = [&](int a) {
        if (a < 0) a += dim;
        if (a < 0 || a >= dim) return false;
        if (!axes_set[a]) { axes_set[a] = true; axes_count += 1; }
        return true;
    };

    bool axes_provided = false;
    if (inputs.size() >= 2 && inputs[1] && inputs[1]->data && inputs[1]->ndata > 0) {
        axes_provided = true;
        const tensor_t* t = inputs[1];
        for (int i = 0; i < (int)t->ndata; ++i) {
            if (!set_axis((int)read_axes_index(t, i))) return false;
        }
    } else {
        int64_t* attr_axes = nullptr;
        int nax = attribute(attr_key_t::axes, attr_axes);
        if (nax > 0) {
            axes_provided = true;
            for (int i = 0; i < nax; ++i) {
                if (!set_axis((int)attr_axes[i])) return false;
            }
        }
    }

    if (!axes_provided) {
        int noop = (int)attribute(attr_key_t::noop_with_empty_axes, (int64_t)0);
        if (noop) {
            // Identity: copy input shape.
            if (!y->reshape_identity(x)) return false;
            for (int k = 0; k < 8; ++k) { in_dims_u[k] = 0; in_strides_u[k] = 0; is_reduce_u[k] = 0; }
            red_count = 1;
            total = (uint32_t)x->ndata;
            uint32_t s = 1;
            for (int k = dim - 1; k >= 0; --k) {
                in_dims_u[k]    = (uint32_t)x->dims[k];
                in_strides_u[k] = s;
                s *= (uint32_t)x->dims[k];
            }
            ensure_buffer(x, x->ndata * sizeof(float));
            ensure_buffer(y, y->ndata * sizeof(float));

            // Identity path: write meta now so exec() doesn't.
            uint8_t mbuf[128] = {};
            auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(mbuf + off, &v, 4); };
            put_u32(0,  total);
            put_u32(4,  ndim);
            put_u32(8,  red_count);
            for (int k = 0; k < 8; ++k) put_u32(16 + k * 4, in_dims_u[k]);
            for (int k = 0; k < 8; ++k) put_u32(48 + k * 4, in_strides_u[k]);
            for (int k = 0; k < 8; ++k) put_u32(80 + k * 4, is_reduce_u[k]);
            get_device().queue.WriteBuffer(meta_buf, 0, mbuf, sizeof(mbuf));
            return true;
        }
        // Reduce over all axes.
        for (int k = 0; k < dim; ++k) { axes_set[k] = true; }
        axes_count = dim;
    }

    int keepdims = (int)attribute(attr_key_t::keepdims, (int64_t)1);

    // Output shape: either keep all ranks with reduced axes at 1, or drop
    // reduced axes entirely.
    int out_dims[8] = {};
    int out_ndim = 0;
    for (int k = 0; k < dim; ++k) {
        if (axes_set[k]) {
            if (keepdims) out_dims[out_ndim++] = 1;
        } else {
            out_dims[out_ndim++] = x->dims[k];
        }
    }
    // Scalar reduction (all axes, no keepdims) — represent as shape [1].
    if (out_ndim == 0) { out_dims[0] = 1; out_ndim = 1; }
    if (!y->reshape(std::span<const int>(out_dims, out_ndim), x->type)) return false;

    // Input strides + is_reduce flags + reduce-count.
    for (int k = 0; k < 8; ++k) { in_dims_u[k] = 0; in_strides_u[k] = 0; is_reduce_u[k] = 0; }
    {
        uint32_t s = 1;
        for (int k = dim - 1; k >= 0; --k) {
            in_dims_u[k]    = (uint32_t)x->dims[k];
            in_strides_u[k] = s;
            s *= (uint32_t)x->dims[k];
        }
    }
    red_count = 1;
    total     = 1;
    for (int k = 0; k < dim; ++k) {
        if (axes_set[k]) { is_reduce_u[k] = 1; red_count *= (uint32_t)x->dims[k]; }
        else             {                     total     *= (uint32_t)x->dims[k]; }
    }
    // Guard: reduce of a zero-sized axis means "no values to reduce" —
    // the expected-count formula (e.g. Mean divides by 0) is model-broken,
    // so reject upfront rather than produce NaN silently.
    if (red_count == 0) return false;

    ensure_buffer(x, x->ndata * sizeof(float));
    ensure_buffer(y, y->ndata * sizeof(float));

    // Meta payload is a pure function of shape + axes config.
    uint8_t mbuf[128] = {};
    auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(mbuf + off, &v, 4); };
    put_u32(0,  total);
    put_u32(4,  ndim);
    put_u32(8,  red_count);
    for (int k = 0; k < 8; ++k) put_u32(16 + k * 4, in_dims_u[k]);
    for (int k = 0; k < 8; ++k) put_u32(48 + k * 4, in_strides_u[k]);
    for (int k = 0; k < 8; ++k) put_u32(80 + k * 4, is_reduce_u[k]);
    get_device().queue.WriteBuffer(meta_buf, 0, mbuf, sizeof(mbuf));
    return true;
}

bool reduce_elementwise_t::exec() {
    auto& dev = get_device();
    upload_if_needed(inputs[0]);

    // meta_buf was written in reshape() — its contents are pure functions
    // of total/ndim/red_count and the cached stride / dim / is_reduce
    // tables, all reshape-time data.
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
    // One workgroup per output; each workgroup's 64 threads cooperatively
    // reduce the corresponding slice. WebGPU limits DispatchWorkgroups.x
    // to 65535, so large `total` values need to span the Y axis too.
    const uint32_t MAX_X = 65535u;
    uint32_t gx = total > MAX_X ? MAX_X : total;
    uint32_t gy = (total + MAX_X - 1) / MAX_X;
    pass.DispatchWorkgroups(gx, gy, 1);
    pass.End();

    mark_gpu_written(outputs[0]);
    return true;
}

} // namespace nnr::webgpu
