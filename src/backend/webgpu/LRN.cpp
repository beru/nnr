// WebGPU LRN — Local Response Normalization (across-channel only).
//
// y[n, c, h, w] = x[n, c, h, w] /
//   pow(bias + alpha/size * sum_{c'=c-r..c+r}(x[n, c', h, w]^2), beta)
// where r = (size - 1) / 2 and out-of-range c' are clamped (ONNX uses zero
// padding: contributions beyond [0, C) are skipped).
//
// f32 only. Each thread computes one output element. Inner loop visits
// `size` channel neighbours at the same spatial position — small (size is
// typically 5 in legacy googlenet) so no tiling needed.

#include "nnr.h"
#include "attr_key.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include <string>
#include <webgpu/webgpu_cpp.h>

namespace nnr {

namespace {

struct LRN_operator_webgpu : public operator_t {
    float alpha = 1e-4f;
    float beta  = 0.75f;
    float bias  = 1.0f;
    int   size  = 5;

    bool built = false;
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          uniforms;
    uint32_t              dispatch_gx = 0;
    uint32_t              dispatch_gy = 0;

    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[2] = {};

    bool init() override {
        if (inputs.size() != 1 || outputs.size() != 1) return false;
        if (!webgpu::device_ready()) return false;
        alpha = (float)attribute(attr_key_t::alpha, 1e-4f);
        beta  = (float)attribute(attr_key_t::beta,  0.75f);
        bias  = (float)attribute(attr_key_t::bias,  1.0f);
        size  = (int)attribute(attr_key_t::size, (int64_t)5);
        if (size < 1) return false;
        layout_mask = LAYOUT_NCHW;
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (x->ndim != 4) return false;

        if (!outputs[0]->reshape_identity(x, NNR_DATA_TYPE_FLOAT32)) return false;
        tensor_t* y = outputs[0];

        if (!built) {
            if (!build_pipeline()) return false;
            built = true;
        }

        webgpu::ensure_buffer(x, (size_t)x->ndata * sizeof(float));
        webgpu::ensure_buffer(y, (size_t)y->ndata * sizeof(float));

        const uint32_t WG = 256;
        uint32_t groups = ((uint32_t)y->ndata + WG - 1) / WG;
        webgpu::dispatch_1d_grid(groups, dispatch_gx, dispatch_gy);

        // Uniform: { N, C, H, W, size, half, total, grid_stride_x, alpha_div_size, beta, bias, _pad }
        uint32_t N = (uint32_t)x->dims[0];
        uint32_t C = (uint32_t)x->dims[1];
        uint32_t H = (uint32_t)x->dims[2];
        uint32_t W = (uint32_t)x->dims[3];
        struct U {
            uint32_t N, C, H, W;
            uint32_t size, half, total, grid_stride_x;
            float    alpha_div_size, beta, bias;
            float    _pad;
        } u;
        u.N = N; u.C = C; u.H = H; u.W = W;
        u.size = (uint32_t)size;
        u.half = (uint32_t)((size - 1) / 2);
        u.total = (uint32_t)y->ndata;
        u.grid_stride_x = dispatch_gx * WG;
        u.alpha_div_size = alpha / (float)size;
        u.beta = beta;
        u.bias = bias;
        u._pad = 0.0f;
        webgpu::get_device().queue.WriteBuffer(uniforms, 0, &u, sizeof(u));
        return true;
    }

    bool build_pipeline() {
        auto& dev = webgpu::get_device();

        const char* src = R"WGSL(
struct U {
    N: u32, C: u32, H: u32, W: u32,
    size: u32, half: u32, total: u32, grid_stride_x: u32,
    alpha_div_size: f32, beta: f32, bias: f32, _pad: f32,
};
@group(0) @binding(0) var<storage, read>       X : array<f32>;
@group(0) @binding(1) var<storage, read_write> Y : array<f32>;
@group(0) @binding(2) var<uniform>             u : U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * u.grid_stride_x + gid.x;
    if (i >= u.total) { return; }
    let HW = u.H * u.W;
    let CHW = u.C * HW;
    let n = i / CHW;
    let rem = i % CHW;
    let c = rem / HW;
    let hw = rem % HW;
    let n_base = n * CHW;
    var sumsq : f32 = 0.0;
    let c_lo : u32 = select(0u, c - u.half, c >= u.half);
    let c_hi_excl : u32 = min(c + u.half + 1u, u.C);
    for (var cc : u32 = c_lo; cc < c_hi_excl; cc = cc + 1u) {
        let v = X[n_base + cc * HW + hw];
        sumsq = sumsq + v * v;
    }
    let denom = pow(u.bias + u.alpha_div_size * sumsq, u.beta);
    Y[i] = X[i] / denom;
}
)WGSL";

        wgpu::ShaderSourceWGSL wgslSrc = {};
        wgslSrc.code = src;
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &wgslSrc;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

        if (!bgl) {
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

            wgpu::BufferDescriptor ud = {};
            ud.size = 48;
            ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
            uniforms = dev.device.CreateBuffer(&ud);
        }

        wgpu::PipelineLayoutDescriptor pld = {};
        pld.bindGroupLayoutCount = 1;
        pld.bindGroupLayouts = &bgl;
        wgpu::PipelineLayout pl = dev.device.CreatePipelineLayout(&pld);

        wgpu::ComputePipelineDescriptor cpd = {};
        cpd.layout = pl;
        cpd.compute.module = sm;
        cpd.compute.entryPoint = "main";
        pipeline = dev.device.CreateComputePipeline(&cpd);
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
            be[0].binding = 0; be[0].buffer = rx->buf;  be[0].offset = 0; be[0].size = rx->size;
            be[1].binding = 1; be[1].buffer = ry->buf;  be[1].offset = 0; be[1].size = ry->size;
            be[2].binding = 2; be[2].buffer = uniforms; be[2].offset = 0; be[2].size = 48;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 3; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_x;
            cached_gen[1] = gen_y;
        }

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, cached_bg);
        pass.DispatchWorkgroups(dispatch_gx, dispatch_gy, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_LRN_webgpu(int, pool_t& pool) {
    return pool_new<LRN_operator_webgpu>(pool);
}

} // namespace nnr
