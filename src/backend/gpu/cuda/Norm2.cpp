#if defined(NNR_USE_CUDA)

// InstanceNormalization (NCHW): normalize over H×W per (N,C), scale/bias per channel.
// RMSNormalization (last-axis): y = x * rsqrt(mean(x^2) + eps) * scale, no mean subtract.
// f32 only.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "attr_key.h"

namespace nnr {

operator_t* resolver_default_op_InstanceNormalization(int opset, pool_t& pool);
operator_t* resolver_default_op_RMSNormalization     (int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* norm2_source() {
    return R"CUDA(
extern "C" {

// Block per (n,c). Threads reduce H*W to compute mean + var, then normalize and
// apply per-channel scale/bias.
__global__ void in_norm_f32(const float* __restrict__ x,
                            const float* __restrict__ scale,  // (C,)
                            const float* __restrict__ bias,   // (C,)
                            float* __restrict__ y,
                            int C, int HW, float eps)
{
    extern __shared__ float smem[];   // 2 * nth
    const int nc  = blockIdx.x;       // covers n*C
    const int c   = nc % C;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    const float* xr = x + (size_t)nc * HW;
    float* yr       = y + (size_t)nc * HW;

    float s = 0.f, ss = 0.f;
    for (int j = tid; j < HW; j += nth) {
        float v = xr[j]; s += v; ss += v * v;
    }
    smem[tid] = s; smem[nth + tid] = ss;
    __syncthreads();
    for (int off = nth >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            smem[tid]       += smem[tid + off];
            smem[nth + tid] += smem[nth + tid + off];
        }
        __syncthreads();
    }
    float mean = smem[0]     / (float)HW;
    float meansq = smem[nth] / (float)HW;
    float var = meansq - mean * mean;
    if (var < 0.f) var = 0.f;
    float inv = rsqrtf(var + eps);
    float sc  = scale[c], bs = bias[c];

    for (int j = tid; j < HW; j += nth) {
        yr[j] = (xr[j] - mean) * inv * sc + bs;
    }
}

// Block per outer row (last-axis). RMS = sqrt(mean(x^2) + eps).
__global__ void rms_norm_lastaxis_f32(const float* __restrict__ x,
                                      const float* __restrict__ scale,  // may be null
                                      float* __restrict__ y,
                                      int D, int outer, float eps, int has_scale)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= outer) return;
    const float* xr = x + (size_t)row * D;
    float* yr       = y + (size_t)row * D;
    const int tid = threadIdx.x, nth = blockDim.x;

    float ss = 0.f;
    for (int j = tid; j < D; j += nth) { float v = xr[j]; ss += v * v; }
    smem[tid] = ss; __syncthreads();
    for (int off = nth >> 1; off > 0; off >>= 1) {
        if (tid < off) smem[tid] += smem[tid + off];
        __syncthreads();
    }
    float inv = rsqrtf(smem[0] / (float)D + eps);

    for (int j = tid; j < D; j += nth) {
        float v = xr[j] * inv;
        if (has_scale) v *= scale[j];
        yr[j] = v;
    }
}

} // extern "C"
)CUDA";
}

struct InstanceNorm_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    float eps = 1e-5f;
    int NC = 0, C = 0, HW = 0;

    bool init() override {
        if (!(inputs.size() == 3 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_InstanceNormalization(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        eps = attribute(attr_key_t::epsilon, 1e-5f);
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* x = inputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32 || x->ndim != 4) return true;
        C = x->dims[1];
        HW = x->dims[2] * x->dims[3];
        NC = x->dims[0] * C;
        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        CUfunction f = be->nvrtc.get("nnr_norm2_f32", norm2_source(),
                                     "in_norm_f32", gpu::nvrtc_arch_option(be->device));
        if (!f) { return fallback->exec(); }

        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_s = (float*)be->cache->ensure_device(inputs[1]);
        float* d_b = (float*)be->cache->ensure_device(inputs[2]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_s || !d_b || !d_y) { return fallback->exec(); }

        unsigned block = 256;
        while ((int)block > HW && block > 32) block >>= 1;
        if (block < 32) block = 32;
        unsigned shared = 2 * block * sizeof(float);
        int _C = C, _HW = HW; float _eps = eps;
        void* args[] = { &d_x, &d_s, &d_b, &d_y, &_C, &_HW, &_eps };
        if (!gpu::nvrtc_launch(be->device, f, (unsigned)NC, 1, 1, block, 1, 1, args, shared)) {
            return fallback->exec();
        }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

struct RMSNorm_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    float eps = 1e-5f;
    int axis = -1, D = 0, outer = 0;

    bool init() override {
        if (inputs.size() < 1 || outputs.size() != 1) return false;
        fallback = resolver_default_op_RMSNormalization(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        eps  = attribute(attr_key_t::epsilon, 1e-5f);
        axis = (int)attribute(attr_key_t::axis, (int64_t)-1);
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* x = inputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return true;
        int caxis = axis < 0 ? axis + x->ndim : axis;
        if (caxis != x->ndim - 1) return true;
        D = x->dims[caxis];
        outer = 1; for (int d = 0; d < caxis; ++d) outer *= x->dims[d];
        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        CUfunction f = be->nvrtc.get("nnr_norm2_f32", norm2_source(),
                                     "rms_norm_lastaxis_f32", gpu::nvrtc_arch_option(be->device));
        if (!f) { return fallback->exec(); }

        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_s = (inputs.size() >= 2 && inputs[1]) ? (float*)be->cache->ensure_device(inputs[1]) : nullptr;
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        unsigned block = 256;
        while ((int)block > D && block > 32) block >>= 1;
        if (block < 32) block = 32;
        unsigned shared = block * sizeof(float);
        int _D = D, _outer = outer; float _eps = eps;
        int has_scale = d_s ? 1 : 0;
        void* args[] = { &d_x, &d_s, &d_y, &_D, &_outer, &_eps, &has_scale };
        if (!gpu::nvrtc_launch(be->device, f, (unsigned)outer, 1, 1, block, 1, 1, args, shared)) {
            return fallback->exec();
        }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_InstanceNormalization(int opset, pool_t& pool) {
    return pool_new<InstanceNorm_cuda>(pool);
}
operator_t* resolver_cuda_op_RMSNormalization(int opset, pool_t& pool) {
    return pool_new<RMSNorm_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
