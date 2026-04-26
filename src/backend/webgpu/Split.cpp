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

constexpr int MAX_OUTPUTS = 16;
constexpr uint32_t WG = 256;

// Split: inverse of Concat. One input is divided along `axis` into N
// outputs of sizes given by either the `split` input (opset 13+) or a
// `split` attribute (older opsets); if neither is present, the axis is
// split equally among outputs.
//
// Each output is a contiguous slab of the input along `axis`. We reuse the
// Slice kernel: for output i, set base = offset_i * in_stride[axis] and
// eff_strides = input's natural row-major strides.
struct Split_operator_webgpu : public operator_t {
    int caxis = 0;
    int ndim  = 0;

    int      n_outputs = 0;
    uint32_t totals[MAX_OUTPUTS]        = {};
    uint32_t bases[MAX_OUTPUTS]         = {};
    uint32_t out_dims_u[MAX_OUTPUTS][8] = {};
    uint32_t in_strides_u[8]            = {};

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          meta_bufs[MAX_OUTPUTS];

    // Cached BindGroups — one per output. Each tracks [gen_x, gen_out[i]].
    wgpu::BindGroup cached_bgs[MAX_OUTPUTS];
    uint32_t        cached_gens[MAX_OUTPUTS][2] = {};

    static int64_t read_index(const tensor_t* t, int i) {
        if (t->type == NNR_DATA_TYPE_INT32) return ((const int32_t*)t->data)[i];
        return ((const int64_t*)t->data)[i];
    }

    bool init() override {
        if (inputs.size() < 1 || inputs.size() > 2) return false;
        if (outputs.size() < 1 || outputs.size() > MAX_OUTPUTS) return false;
        if (!webgpu::device_ready()) return false;

        auto& dev = webgpu::get_device();
        wgpu::ShaderSourceWGSL w = {};
        w.code = webgpu::wgsl::slice;  // reuse slice.wgsl (base + eff_stride per-axis)
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
        n_outputs = (int)outputs.size();
        for (int i = 0; i < n_outputs; ++i) meta_bufs[i] = dev.device.CreateBuffer(&md);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return false;
        ndim = x->ndim;
        if (ndim <= 0 || ndim > 8) return false;

        int axis_attr = (int)attribute(attr_key_t::axis, (int64_t)0);
        caxis = axis_attr < 0 ? axis_attr + ndim : axis_attr;
        if (caxis < 0 || caxis >= ndim) return false;

        // Resolve split sizes: opset 13+ input, opset <13 attribute, else equal.
        int split_sizes[MAX_OUTPUTS] = {};
        if (inputs.size() == 2 && inputs[1] && inputs[1]->data) {
            const tensor_t* ts = inputs[1];
            if ((int)ts->ndata != n_outputs) return false;
            for (int i = 0; i < n_outputs; ++i) split_sizes[i] = (int)read_index(ts, i);
        } else {
            int64_t* attr_split = nullptr;
            int ns = attribute(attr_key_t::split, attr_split);
            if (ns == n_outputs) {
                for (int i = 0; i < n_outputs; ++i) split_sizes[i] = (int)attr_split[i];
            } else {
                int dim = x->dims[caxis];
                if (dim % n_outputs != 0) return false;
                int per = dim / n_outputs;
                for (int i = 0; i < n_outputs; ++i) split_sizes[i] = per;
            }
        }
        int sum = 0;
        for (int i = 0; i < n_outputs; ++i) {
            if (split_sizes[i] < 0) return false;
            sum += split_sizes[i];
        }
        if (sum != x->dims[caxis]) return false;

        // Input natural row-major strides (same for all outputs).
        for (int k = 0; k < 8; ++k) in_strides_u[k] = 0;
        {
            uint32_t s = 1;
            for (int i = ndim - 1; i >= 0; --i) { in_strides_u[i] = s; s *= (uint32_t)x->dims[i]; }
        }

        // Per-output: shape, total, base offset along caxis.
        int axis_off = 0;
        for (int i = 0; i < n_outputs; ++i) {
            tensor_t* yi = outputs[i];
            int od[8];
            for (int k = 0; k < ndim; ++k) od[k] = x->dims[k];
            od[caxis] = split_sizes[i];
            if (!yi->reshape(std::span<const int>(od, ndim), x->type)) return false;
            for (int k = 0; k < 8; ++k) out_dims_u[i][k] = 0;
            for (int k = 0; k < ndim; ++k) out_dims_u[i][k] = (uint32_t)od[k];
            totals[i] = (uint32_t)yi->ndata;
            bases[i]  = (uint32_t)axis_off * in_strides_u[caxis];
            axis_off += split_sizes[i];
            webgpu::ensure_buffer(yi, yi->ndata * sizeof(float));
        }
        webgpu::ensure_buffer(x, x->ndata * sizeof(float));

        // Write meta for each output (slice.wgsl's Meta layout):
        //   [0..4)    total
        //   [4..8)    ndim
        //   [8..12)   base
        //   [12..16)  pad
        //   [16..48)  out_dims     (8 x u32)
        //   [48..80)  eff_strides  (8 x u32)
        for (int i = 0; i < n_outputs; ++i) {
            uint8_t buf[128] = {};
            auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(buf + off, &v, 4); };
            put_u32(0, totals[i]);
            put_u32(4, (uint32_t)ndim);
            put_u32(8, bases[i]);
            for (int k = 0; k < 8; ++k) put_u32(16 + k * 4, out_dims_u[i][k]);
            for (int k = 0; k < 8; ++k) put_u32(48 + k * 4, in_strides_u[k]);
            webgpu::get_device().queue.WriteBuffer(meta_bufs[i], 0, buf, sizeof(buf));
        }
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();
        webgpu::upload_if_needed(inputs[0]);
        auto* rx = webgpu::find(inputs[0]);
        uint32_t gen_x = webgpu::generation_of(inputs[0]);

        wgpu::CommandEncoder& enc = webgpu::shared_encoder();
        for (int i = 0; i < n_outputs; ++i) {
            if (totals[i] == 0) continue;
            auto* ry = webgpu::find(outputs[i]);
            uint32_t gen_y = webgpu::generation_of(outputs[i]);
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
            uint32_t groups = (totals[i] + WG - 1) / WG;
            pass.DispatchWorkgroups(groups, 1, 1);
            pass.End();
        }

        for (int i = 0; i < n_outputs; ++i) webgpu::mark_gpu_written(outputs[i]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_Split_webgpu(int, pool_t& pool) {
    return pool_new<Split_operator_webgpu>(pool);
}

} // namespace nnr
