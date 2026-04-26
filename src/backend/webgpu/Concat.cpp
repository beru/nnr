#include "nnr.h"
#include "attr_key.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include "shaders.h"

#include <webgpu/webgpu_cpp.h>

#include <cstring>

namespace nnr {

namespace {

constexpr int MAX_INPUTS = 16;
constexpr uint32_t WG = 64;

struct Concat_operator_webgpu : public operator_t {
    int caxis = 0;
    int ndim  = 0;

    // Output shape + natural row-major strides (over the output's own dims).
    uint32_t out_dims_u[8]    = {};
    uint32_t out_strides_u[8] = {};

    // Per-input: its dims (same as output except on caxis), its natural
    // strides, element count, and the output's flat offset where its slice
    // starts (= axis_offset * out_strides[caxis]).
    int      n_inputs         = 0;
    uint32_t in_dims_u[MAX_INPUTS][8]    = {};
    uint32_t in_strides_u[MAX_INPUTS][8] = {};
    uint32_t totals[MAX_INPUTS]          = {};
    uint32_t flat_offsets[MAX_INPUTS]    = {};

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          meta_bufs[MAX_INPUTS];   // one per input

    // Cached BindGroups — one per input. Each tracks [gen_in[i], gen_y].
    wgpu::BindGroup cached_bgs[MAX_INPUTS];
    uint32_t        cached_gens[MAX_INPUTS][2] = {};
    uint32_t        dispatches_gx_r[MAX_INPUTS] = {};
    uint32_t        dispatches_gy_r[MAX_INPUTS] = {};

    bool init() override {
        if (inputs.size() < 1 || inputs.size() > MAX_INPUTS) return false;
        if (outputs.size() != 1) return false;
        if (!webgpu::device_ready()) return false;

        auto& dev = webgpu::get_device();
        wgpu::ShaderSourceWGSL w = {};
        w.code = webgpu::wgsl::concat;
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

        // Pre-allocate one 128B meta buffer per input so a single command
        // encoder can record all N dispatches without per-exec allocation.
        wgpu::BufferDescriptor md = {};
        md.size  = 128;
        md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        n_inputs = (int)inputs.size();
        for (int i = 0; i < n_inputs; ++i) meta_bufs[i] = dev.device.CreateBuffer(&md);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t*       y = outputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return false;
        ndim = x->ndim;
        if (ndim <= 0 || ndim > 8) return false;

        int axis_attr = (int)attribute(attr_key_t::axis, (int64_t)1);
        caxis = axis_attr < 0 ? axis_attr + ndim : axis_attr;
        if (caxis < 0 || caxis >= ndim) return false;

        // Output dims: same as input[0], with caxis summed across all inputs.
        int out_dims[8];
        for (int j = 0; j < ndim; ++j) out_dims[j] = x->dims[j];
        int axis_sum = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            const tensor_t* xi = inputs[i];
            if (xi->type != NNR_DATA_TYPE_FLOAT32) return false;
            if (xi->ndim != ndim) return false;
            for (int j = 0; j < ndim; ++j) {
                if (j == caxis) continue;
                if (xi->dims[j] != x->dims[j]) return false;
            }
            axis_sum += xi->dims[caxis];
        }
        out_dims[caxis] = axis_sum;
        if (!y->reshape(std::span<const int>(out_dims, ndim), x->type)) return false;

        // Output natural row-major strides.
        for (int j = 0; j < 8; ++j) out_strides_u[j] = 0;
        {
            uint32_t s = 1;
            for (int j = ndim - 1; j >= 0; --j) { out_strides_u[j] = s; s *= (uint32_t)out_dims[j]; }
        }
        for (int j = 0; j < ndim; ++j) out_dims_u[j] = (uint32_t)out_dims[j];

        // Per-input metadata: natural strides, element count, flat offset.
        int axis_off = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            const tensor_t* xi = inputs[i];
            for (int j = 0; j < 8; ++j) { in_dims_u[i][j] = 0; in_strides_u[i][j] = 0; }
            uint32_t s = 1;
            for (int j = ndim - 1; j >= 0; --j) { in_strides_u[i][j] = s; s *= (uint32_t)xi->dims[j]; }
            for (int j = 0; j < ndim; ++j) in_dims_u[i][j] = (uint32_t)xi->dims[j];
            totals[i]       = (uint32_t)xi->ndata;
            flat_offsets[i] = (uint32_t)axis_off * out_strides_u[caxis];
            axis_off       += xi->dims[caxis];

            webgpu::ensure_buffer(xi, xi->ndata * sizeof(float));
        }
        webgpu::ensure_buffer(y, y->ndata * sizeof(float));

        // Meta struct layout (matches concat.wgsl):
        //   [0..4)    total
        //   [4..8)    ndim
        //   [8..12)   flat_offset
        //   [12..16)  grid_stride_x (threads-x for 2D dispatch split)
        //   [16..48)  in_dims      (8 x u32)
        //   [48..80)  in_strides   (8 x u32)
        //   [80..112) dst_strides  (8 x u32)
        for (int i = 0; i < n_inputs; ++i) {
            uint32_t groups = (totals[i] + WG - 1) / WG;
            uint32_t gx = 0, gy = 0;
            webgpu::dispatch_1d_grid(groups, gx, gy);
            uint8_t buf[128] = {};
            auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(buf + off, &v, 4); };
            put_u32(0,  totals[i]);
            put_u32(4,  (uint32_t)ndim);
            put_u32(8,  flat_offsets[i]);
            put_u32(12, gx * WG);
            for (int j = 0; j < 8; ++j) put_u32(16 + j * 4, in_dims_u[i][j]);
            for (int j = 0; j < 8; ++j) put_u32(48 + j * 4, in_strides_u[i][j]);
            for (int j = 0; j < 8; ++j) put_u32(80 + j * 4, out_strides_u[j]);
            webgpu::get_device().queue.WriteBuffer(meta_bufs[i], 0, buf, sizeof(buf));
            dispatches_gx_r[i] = gx;
            dispatches_gy_r[i] = gy;
        }
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();
        tensor_t* y = outputs[0];
        auto* ry = webgpu::find(y);

        for (int i = 0; i < n_inputs; ++i) {
            webgpu::upload_if_needed(inputs[i]);
        }

        uint32_t gen_y = webgpu::generation_of(y);

        // Single command encoder records all N dispatches; each writes a
        // disjoint slice of Y so ordering among dispatches is irrelevant.
        wgpu::CommandEncoder& enc = webgpu::shared_encoder();
        for (int i = 0; i < n_inputs; ++i) {
            auto* rx = webgpu::find(inputs[i]);
            uint32_t gen_x = webgpu::generation_of(inputs[i]);
            if (!cached_bgs[i] || cached_gens[i][0] != gen_x || cached_gens[i][1] != gen_y) {
                wgpu::BindGroupEntry be[3] = {};
                be[0].binding = 0; be[0].buffer = rx->buf;        be[0].offset = 0; be[0].size = rx->size;
                be[1].binding = 1; be[1].buffer = ry->buf;        be[1].offset = 0; be[1].size = ry->size;
                be[2].binding = 2; be[2].buffer = meta_bufs[i];   be[2].offset = 0; be[2].size = 128;
                wgpu::BindGroupDescriptor bgd = {};
                bgd.layout = bgl; bgd.entryCount = 3; bgd.entries = be;
                cached_bgs[i] = dev.device.CreateBindGroup(&bgd);
                cached_gens[i][0] = gen_x;
                cached_gens[i][1] = gen_y;
            }

            wgpu::ComputePassEncoder pass = enc.BeginComputePass();
            pass.SetPipeline(pipeline);
            pass.SetBindGroup(0, cached_bgs[i]);
            pass.DispatchWorkgroups(dispatches_gx_r[i], dispatches_gy_r[i], 1);
            pass.End();
        }

        webgpu::mark_gpu_written(y);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_Concat_webgpu(int, pool_t& pool) {
    return pool_new<Concat_operator_webgpu>(pool);
}

} // namespace nnr
