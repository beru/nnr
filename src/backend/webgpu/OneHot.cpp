// WebGPU OneHot. Supports f32/i32/u32 values (most common) with int32/int64
// indices pre-converted at reshape time (like Gather). One thread per output
// element: each thread computes its (outer, d, inner) coord and compares `d`
// to the index at the corresponding (outer, inner) position in the indices
// tensor. In-range matches → on_value; otherwise → off_value.
//
// Negative indices and out-of-range indices are handled at reshape time —
// negatives get +depth added; anything still out-of-range is stored as a
// sentinel (UINT32_MAX) that will never equal any valid `d < depth`, so the
// shader writes off_value for those slots without extra branching.
//
// Other values dtypes (f64, i64, float8, bool-as-C-bool, etc.) return false
// from reshape() so the registry falls back to CPU. int64 indices without
// CPU residency also fall back.

#include "nnr.h"
#include "attr_key.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <webgpu/webgpu_cpp.h>

namespace nnr {

namespace {

constexpr uint32_t WG = 256;

const char* wgsl_ty(data_type_t t) {
    switch (t) {
    case NNR_DATA_TYPE_FLOAT32: return "f32";
    case NNR_DATA_TYPE_INT32:   return "i32";
    case NNR_DATA_TYPE_UINT32:  return "u32";
    default:                    return nullptr;
    }
}

struct OneHot_operator_webgpu : public operator_t {
    int      axis_attr = -1;
    int      caxis     = 0;
    uint32_t total_u   = 0;
    uint32_t outer_u   = 0;
    uint32_t depth_u   = 0;
    uint32_t inner_u   = 0;

    data_type_t built_value_ty = NNR_DATA_TYPE_UNDEFINED;

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          idx_buf;
    uint32_t              idx_capacity = 0;
    uint32_t              idx_gen      = 0;
    wgpu::Buffer          uniforms;

    // Cached BindGroup. Tensor-backed slots: [values, Y]; idx_buf tracked via idx_gen.
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[3] = {};   // [idx_gen, gen_values, gen_Y]

    static int64_t read_idx(const tensor_t* t, size_t i) {
        if (t->type == NNR_DATA_TYPE_INT32) return ((const int32_t*)t->data)[i];
        return ((const int64_t*)t->data)[i];
    }

    static int64_t read_scalar(const tensor_t* t) {
        switch (t->type) {
        case NNR_DATA_TYPE_INT32:   return *(const int32_t*)t->data;
        case NNR_DATA_TYPE_INT64:   return *(const int64_t*)t->data;
        case NNR_DATA_TYPE_UINT32:  return (int64_t)*(const uint32_t*)t->data;
        case NNR_DATA_TYPE_FLOAT32: return (int64_t)*(const float*)t->data;
        default:                    return -1;
        }
    }

    bool init() override {
        if (inputs.size() != 3 || outputs.size() != 1) return false;
        if (!webgpu::device_ready()) return false;

        axis_attr = (int)attribute(attr_key_t::axis, (int64_t)-1);

        auto& dev = webgpu::get_device();
        wgpu::BindGroupLayoutEntry e[4] = {};
        for (int i = 0; i < 2; ++i) {
            e[i].binding = (uint32_t)i;
            e[i].visibility = wgpu::ShaderStage::Compute;
            e[i].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        }
        e[2].binding = 2; e[2].visibility = wgpu::ShaderStage::Compute;
        e[2].buffer.type = wgpu::BufferBindingType::Storage;
        e[3].binding = 3; e[3].visibility = wgpu::ShaderStage::Compute;
        e[3].buffer.type = wgpu::BufferBindingType::Uniform;
        wgpu::BindGroupLayoutDescriptor bgld = {};
        bgld.entryCount = 4; bgld.entries = e;
        bgl = dev.device.CreateBindGroupLayout(&bgld);

        wgpu::BufferDescriptor ud = {};
        ud.size  = 16;
        ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        uniforms = dev.device.CreateBuffer(&ud);
        return true;
    }

    bool build_pipeline(const char* val_ty) {
        auto& dev = webgpu::get_device();

        std::string src =
            "struct Cfg { total : u32, outer : u32, depth : u32, inner : u32 };\n"
            "@group(0) @binding(0) var<storage, read>       Ind : array<u32>;\n"
            "@group(0) @binding(1) var<storage, read>       Val : array<";
        src += val_ty;
        src += ">;\n"
               "@group(0) @binding(2) var<storage, read_write> Y   : array<";
        src += val_ty;
        src += ">;\n"
               "@group(0) @binding(3) var<uniform>             cfg : Cfg;\n"
               "@compute @workgroup_size(256)\n"
               "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
               "  let i = gid.x;\n"
               "  if (i >= cfg.total) { return; }\n"
               "  let inner_pos = i % cfg.inner;\n"
               "  let d         = (i / cfg.inner) % cfg.depth;\n"
               "  let outer_pos = i / (cfg.inner * cfg.depth);\n"
               "  let idx_flat  = outer_pos * cfg.inner + inner_pos;\n"
               "  let stored    = Ind[idx_flat];\n"
               "  Y[i] = select(Val[0], Val[1], stored == d);\n"
               "}\n";

        wgpu::ShaderSourceWGSL ws = {};
        ws.code = src.c_str();
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &ws;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

        wgpu::PipelineLayoutDescriptor pld = {};
        pld.bindGroupLayoutCount = 1;
        pld.bindGroupLayouts     = &bgl;
        wgpu::PipelineLayout pl  = dev.device.CreatePipelineLayout(&pld);

        wgpu::ComputePipelineDescriptor cpd = {};
        cpd.layout             = pl;
        cpd.compute.module     = sm;
        cpd.compute.entryPoint = "main";
        pipeline = dev.device.CreateComputePipeline(&cpd);
        return (bool)pipeline;
    }

    bool reshape() override {
        const tensor_t* indices  = inputs[0];
        const tensor_t* depth_t  = inputs[1];
        const tensor_t* values_t = inputs[2];
        tensor_t*       y        = outputs[0];
        if (!indices || !depth_t || !values_t || !y) return false;

        if (indices->type != NNR_DATA_TYPE_INT32 && indices->type != NNR_DATA_TYPE_INT64)
            return false;
        if (!indices->data) return false;   // must be CPU-readable
        if (!depth_t->data) return false;
        if (depth_t->ndata < 1) return false;

        const char* val_ty = wgsl_ty(values_t->type);
        if (!val_ty) return false;          // unsupported values dtype → CPU fallback
        if (values_t->ndata < 2) return false;

        int64_t depth = read_scalar(depth_t);
        if (depth <= 0) return false;

        int out_ndim = indices->ndim + 1;
        if (out_ndim > 8) return false;

        caxis = axis_attr < 0 ? axis_attr + out_ndim : axis_attr;
        if (caxis < 0 || caxis >= out_ndim) return false;

        int out_dims[8] = {};
        int d = 0;
        for (int i = 0; i < out_ndim; ++i) {
            if (i == caxis) out_dims[i] = (int)depth;
            else            out_dims[i] = indices->dims[d++];
        }
        if (!y->reshape(std::span<const int>(out_dims, out_ndim), values_t->type))
            return false;

        outer_u = 1;
        for (int i = 0; i < caxis; ++i) outer_u *= (uint32_t)out_dims[i];
        depth_u = (uint32_t)depth;
        inner_u = 1;
        for (int i = caxis + 1; i < out_ndim; ++i) inner_u *= (uint32_t)out_dims[i];
        total_u = outer_u * depth_u * inner_u;

        const uint32_t n_idx = outer_u * inner_u;

        if (!pipeline || built_value_ty != values_t->type) {
            if (!build_pipeline(val_ty)) return false;
            built_value_ty = values_t->type;
        }

        // Build a u32 index buffer on CPU — normalize negatives, sentinel
        // out-of-range so the shader never matches any valid `d`.
        std::vector<uint32_t> host_idx((size_t)n_idx, UINT32_MAX);
        for (uint32_t i = 0; i < n_idx; ++i) {
            int64_t v = read_idx(indices, i);
            if (v < 0) v += depth;
            if (v >= 0 && v < depth) host_idx[i] = (uint32_t)v;
        }

        auto& dev = webgpu::get_device();
        uint32_t needed_bytes = (uint32_t)(host_idx.size() * sizeof(uint32_t));
        needed_bytes = (needed_bytes + 3u) & ~3u;
        if (needed_bytes < 16u) needed_bytes = 16u;
        if (idx_capacity < needed_bytes || !idx_buf) {
            wgpu::BufferDescriptor bd = {};
            bd.size  = needed_bytes;
            bd.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
            idx_buf      = dev.device.CreateBuffer(&bd);
            idx_capacity = needed_bytes;
            ++idx_gen;
        }
        dev.queue.WriteBuffer(idx_buf, 0, host_idx.data(),
                              host_idx.size() * sizeof(uint32_t));

        // values tensor — sized to at least 16 bytes so WebGPU doesn't complain.
        const size_t val_bytes = values_t->ndata * 4;
        webgpu::ensure_buffer(values_t, val_bytes < 16 ? 16 : val_bytes);
        webgpu::ensure_buffer(y, (size_t)y->ndata * 4);

        uint32_t u[4] = { total_u, outer_u, depth_u, inner_u };
        dev.queue.WriteBuffer(uniforms, 0, u, sizeof(u));
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();
        webgpu::upload_if_needed(inputs[2]);  // values

        auto* rv = webgpu::find(inputs[2]);
        auto* ry = webgpu::find(outputs[0]);
        uint32_t gen_v = webgpu::generation_of(inputs[2]);
        uint32_t gen_y = webgpu::generation_of(outputs[0]);
        if (!cached_bg || idx_gen != cached_gen[0] || gen_v != cached_gen[1] || gen_y != cached_gen[2]) {
            wgpu::BindGroupEntry be[4] = {};
            be[0].binding = 0; be[0].buffer = idx_buf;  be[0].offset = 0; be[0].size = idx_capacity;
            be[1].binding = 1; be[1].buffer = rv->buf;  be[1].offset = 0; be[1].size = rv->size;
            be[2].binding = 2; be[2].buffer = ry->buf;  be[2].offset = 0; be[2].size = ry->size;
            be[3].binding = 3; be[3].buffer = uniforms; be[3].offset = 0; be[3].size = 16;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 4; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = idx_gen;
            cached_gen[1] = gen_v;
            cached_gen[2] = gen_y;
        }

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, cached_bg);
        uint32_t groups = (total_u + WG - 1) / WG;
        pass.DispatchWorkgroups(groups, 1, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_OneHot_webgpu(int, pool_t& pool) {
    return pool_new<OneHot_operator_webgpu>(pool);
}

} // namespace nnr
