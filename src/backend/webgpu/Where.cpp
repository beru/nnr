// WebGPU Where(cond, a, b) → cond ? a : b, elementwise with full three-way
// NumPy/ONNX broadcasting (rank ≤ 8).
//
// Scope: cond is u32, i32, or bool (non-zero is true), a and b are f32
// with the same dtype. Output dtype = a->type. Broadcast rule: right-
// align each input's shape; each axis pair must match or be 1. Missing
// leading axes are size 1. Output shape is the elementwise max of the
// aligned input shapes. An input's stride for a given output axis is 0
// when that axis is size-1 in the input (or missing), else the natural
// row-major stride.
//
// BOOL cond: ONNX stores bool as u8 on CPU. `upload_if_needed` in
// buffer.cpp widens each byte to a u32 before uploading, so from the
// shader's perspective a BOOL cond looks identical to a u32 cond.
//
// CPU fallbacks:
//   - Non-u32/i32/bool cond.
//   - Non-f32 a/b.
//   - rank > 8 or mismatched-broadcast shapes.
//
// Shader layout mirrors binary_elementwise_t's ternary-ized form: per-axis
// out_dims + cond_strides + a_strides + b_strides packed as vec4 pairs for
// predictable 16B alignment. The same 128B meta-buffer pattern that
// Expand/Tile/binary use.

#include "nnr.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include <cstring>
#include <webgpu/webgpu_cpp.h>

namespace nnr {

namespace {

constexpr uint32_t WG = 256;

struct Where_operator_webgpu : public operator_t {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          meta_buf;

    // Cached BindGroup. Tensor-backed slots: [C, A, B, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[4] = {};

    uint32_t total = 0;
    uint32_t ndim  = 0;
    uint32_t out_dims_u[8]  = {};
    uint32_t c_strides_u[8] = {};
    uint32_t a_strides_u[8] = {};
    uint32_t b_strides_u[8] = {};

    static constexpr const char* kWgsl =
        "struct Meta {\n"
        "  total          : u32,\n"
        "  ndim           : u32,\n"
        "  _a             : u32,\n"
        "  _b             : u32,\n"
        "  out_dims_lo    : vec4<u32>,\n"
        "  out_dims_hi    : vec4<u32>,\n"
        "  c_strides_lo   : vec4<u32>,\n"
        "  c_strides_hi   : vec4<u32>,\n"
        "  a_strides_lo   : vec4<u32>,\n"
        "  a_strides_hi   : vec4<u32>,\n"
        "  b_strides_lo   : vec4<u32>,\n"
        "  b_strides_hi   : vec4<u32>,\n"
        "};\n"
        "@group(0) @binding(0) var<storage, read>       C  : array<u32>;\n"
        "@group(0) @binding(1) var<storage, read>       A  : array<f32>;\n"
        "@group(0) @binding(2) var<storage, read>       B  : array<f32>;\n"
        "@group(0) @binding(3) var<storage, read_write> Y  : array<f32>;\n"
        "@group(0) @binding(4) var<storage, read>       md : Meta;\n"
        "fn get_out_dim(i : u32)  -> u32 { if (i < 4u) { return md.out_dims_lo[i]; }  return md.out_dims_hi[i - 4u]; }\n"
        "fn get_c_stride(i : u32) -> u32 { if (i < 4u) { return md.c_strides_lo[i]; } return md.c_strides_hi[i - 4u]; }\n"
        "fn get_a_stride(i : u32) -> u32 { if (i < 4u) { return md.a_strides_lo[i]; } return md.a_strides_hi[i - 4u]; }\n"
        "fn get_b_stride(i : u32) -> u32 { if (i < 4u) { return md.b_strides_lo[i]; } return md.b_strides_hi[i - 4u]; }\n"
        "@compute @workgroup_size(256)\n"
        "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
        "  let o = gid.x;\n"
        "  if (o >= md.total) { return; }\n"
        "  var c_flat : u32 = 0u;\n"
        "  var a_flat : u32 = 0u;\n"
        "  var b_flat : u32 = 0u;\n"
        "  var tmp    : u32 = o;\n"
        "  for (var k : i32 = i32(md.ndim) - 1; k >= 0; k = k - 1) {\n"
        "    let d   = get_out_dim(u32(k));\n"
        "    let idx = tmp % d;\n"
        "    tmp     = tmp / d;\n"
        "    c_flat  = c_flat + idx * get_c_stride(u32(k));\n"
        "    a_flat  = a_flat + idx * get_a_stride(u32(k));\n"
        "    b_flat  = b_flat + idx * get_b_stride(u32(k));\n"
        "  }\n"
        "  Y[o] = select(B[b_flat], A[a_flat], C[c_flat] != 0u);\n"
        "}\n";

    bool init() override {
        if (!is_inout_size(3, 1)) return false;
        if (!webgpu::device_ready()) return false;
        auto& dev = webgpu::get_device();

        wgpu::ShaderSourceWGSL src = {};
        src.code = kWgsl;
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &src;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

        wgpu::BindGroupLayoutEntry e[5] = {};
        // C, A, B, meta all read-only storage; Y is writable storage.
        for (int i = 0; i < 3; ++i) {
            e[i].binding = i; e[i].visibility = wgpu::ShaderStage::Compute;
            e[i].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        }
        e[3].binding = 3; e[3].visibility = wgpu::ShaderStage::Compute;
        e[3].buffer.type = wgpu::BufferBindingType::Storage;
        e[4].binding = 4; e[4].visibility = wgpu::ShaderStage::Compute;
        e[4].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        wgpu::BindGroupLayoutDescriptor bgld = {};
        bgld.entryCount = 5; bgld.entries = e;
        bgl = dev.device.CreateBindGroupLayout(&bgld);

        wgpu::PipelineLayoutDescriptor pld = {};
        pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = &bgl;
        wgpu::PipelineLayout pl = dev.device.CreatePipelineLayout(&pld);

        wgpu::ComputePipelineDescriptor cpd = {};
        cpd.layout = pl;
        cpd.compute.module = sm;
        cpd.compute.entryPoint = "main";
        pipeline = dev.device.CreateComputePipeline(&cpd);

        // 16B header + 32B out_dims + 32B cond_strides + 32B a_strides +
        // 32B b_strides = 144B — round up to 160 for alignment predictability.
        wgpu::BufferDescriptor md = {};
        md.size = 160;
        md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        meta_buf = dev.device.CreateBuffer(&md);
        return true;
    }

    // Natural row-major strides over a tensor's own rank.
    static void natural_strides(const tensor_t* t, uint32_t out[8]) {
        uint32_t s = 1;
        for (int i = t->ndim - 1; i >= 0; --i) { out[i] = s; s *= (uint32_t)t->dims[i]; }
    }

    bool reshape() override {
        const tensor_t* c = inputs[0];
        const tensor_t* a = inputs[1];
        const tensor_t* b = inputs[2];
        tensor_t*       y = outputs[0];

        if (c->type != NNR_DATA_TYPE_UINT32
         && c->type != NNR_DATA_TYPE_INT32
         && c->type != NNR_DATA_TYPE_BOOL) return false;
        if (a->type != NNR_DATA_TYPE_FLOAT32 || b->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (c->ndim > 8 || a->ndim > 8 || b->ndim > 8) return false;

        int out_ndim = c->ndim;
        if (a->ndim > out_ndim) out_ndim = a->ndim;
        if (b->ndim > out_ndim) out_ndim = b->ndim;
        if (out_ndim > 8) return false;

        int out_dims[8] = {};
        for (int k = 0; k < out_ndim; ++k) {
            int rc = k - (out_ndim - c->ndim);
            int ra = k - (out_ndim - a->ndim);
            int rb = k - (out_ndim - b->ndim);
            int dc = (rc >= 0) ? c->dims[rc] : 1;
            int da = (ra >= 0) ? a->dims[ra] : 1;
            int db = (rb >= 0) ? b->dims[rb] : 1;
            // Three-way broadcast: all non-1 dims must match.
            int m = 1;
            if (dc != 1) m = dc;
            if (da != 1) { if (m != 1 && m != da) return false; m = da; }
            if (db != 1) { if (m != 1 && m != db) return false; m = db; }
            if ((dc != 1 && dc != m) || (da != 1 && da != m) || (db != 1 && db != m)) return false;
            out_dims[k] = m;
        }
        if (!y->reshape(std::span<const int>(out_dims, out_ndim), a->type)) return false;

        uint32_t c_nat[8] = {}, a_nat[8] = {}, b_nat[8] = {};
        natural_strides(c, c_nat);
        natural_strides(a, a_nat);
        natural_strides(b, b_nat);

        for (int k = 0; k < 8; ++k) {
            out_dims_u[k] = 0;
            c_strides_u[k] = 0; a_strides_u[k] = 0; b_strides_u[k] = 0;
        }
        for (int k = 0; k < out_ndim; ++k) {
            out_dims_u[k] = (uint32_t)out_dims[k];
            int rc = k - (out_ndim - c->ndim);
            int ra = k - (out_ndim - a->ndim);
            int rb = k - (out_ndim - b->ndim);
            c_strides_u[k] = (rc >= 0 && c->dims[rc] != 1) ? c_nat[rc] : 0u;
            a_strides_u[k] = (ra >= 0 && a->dims[ra] != 1) ? a_nat[ra] : 0u;
            b_strides_u[k] = (rb >= 0 && b->dims[rb] != 1) ? b_nat[rb] : 0u;
        }
        ndim = (uint32_t)out_ndim;
        total = 1;
        for (int k = 0; k < out_ndim; ++k) total *= (uint32_t)out_dims[k];

        webgpu::ensure_buffer(c, (size_t)c->ndata * sizeof(uint32_t));
        webgpu::ensure_buffer(a, (size_t)a->ndata * sizeof(float));
        webgpu::ensure_buffer(b, (size_t)b->ndata * sizeof(float));
        webgpu::ensure_buffer(y, (size_t)y->ndata * sizeof(float));

        // Meta layout (matches kWgsl's struct Meta):
        //   [0..4)    total
        //   [4..8)    ndim
        //   [8..16)   pad
        //   [16..48)  out_dims    (8 x u32)
        //   [48..80)  c_strides   (8 x u32)
        //   [80..112) a_strides   (8 x u32)
        //   [112..144) b_strides  (8 x u32)
        uint8_t buf[160] = {};
        auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(buf + off, &v, 4); };
        put_u32(0, total);
        put_u32(4, ndim);
        for (int i = 0; i < 8; ++i) put_u32( 16 + i * 4, out_dims_u[i]);
        for (int i = 0; i < 8; ++i) put_u32( 48 + i * 4, c_strides_u[i]);
        for (int i = 0; i < 8; ++i) put_u32( 80 + i * 4, a_strides_u[i]);
        for (int i = 0; i < 8; ++i) put_u32(112 + i * 4, b_strides_u[i]);
        webgpu::get_device().queue.WriteBuffer(meta_buf, 0, buf, sizeof(buf));
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();
        webgpu::upload_if_needed(inputs[0]);
        webgpu::upload_if_needed(inputs[1]);
        webgpu::upload_if_needed(inputs[2]);

        auto* rc = webgpu::find(inputs[0]);
        auto* ra = webgpu::find(inputs[1]);
        auto* rb = webgpu::find(inputs[2]);
        auto* ry = webgpu::find(outputs[0]);

        uint32_t gen_c = webgpu::generation_of(inputs[0]);
        uint32_t gen_a = webgpu::generation_of(inputs[1]);
        uint32_t gen_b = webgpu::generation_of(inputs[2]);
        uint32_t gen_y = webgpu::generation_of(outputs[0]);
        if (!cached_bg || gen_c != cached_gen[0] || gen_a != cached_gen[1]
                       || gen_b != cached_gen[2] || gen_y != cached_gen[3]) {
            wgpu::BindGroupEntry be[5] = {};
            be[0].binding = 0; be[0].buffer = rc->buf;   be[0].offset = 0; be[0].size = rc->size;
            be[1].binding = 1; be[1].buffer = ra->buf;   be[1].offset = 0; be[1].size = ra->size;
            be[2].binding = 2; be[2].buffer = rb->buf;   be[2].offset = 0; be[2].size = rb->size;
            be[3].binding = 3; be[3].buffer = ry->buf;   be[3].offset = 0; be[3].size = ry->size;
            be[4].binding = 4; be[4].buffer = meta_buf;  be[4].offset = 0; be[4].size = 160;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 5; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_c;
            cached_gen[1] = gen_a;
            cached_gen[2] = gen_b;
            cached_gen[3] = gen_y;
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

operator_t* resolver_default_op_Where_webgpu(int, pool_t& pool) {
    return pool_new<Where_operator_webgpu>(pool);
}

} // namespace nnr
