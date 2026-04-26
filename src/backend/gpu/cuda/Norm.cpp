#if defined(NNR_USE_CUDA)

// BatchNormalization (inference) and LayerNormalization via NVRTC.
// f32 only. NCHW for BN; last-axis LN (axis=-1, standard transformer layout).

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "attr_key.h"

namespace nnr {

operator_t* resolver_default_op_BatchNormalization(int opset, pool_t& pool);
operator_t* resolver_default_op_LayerNormalization(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* norm_source() {
    return R"CUDA(
#ifndef INFINITY
#define INFINITY __int_as_float(0x7F800000)
#endif
extern "C" {

// BatchNorm inference: y = scale[c] * (x - mean[c]) / sqrt(var[c] + eps) + bias[c]
// Interpreted as (outer, C, inner). Thread per element.
__global__ void bn_infer_f32(const float* __restrict__ x,
                             const float* __restrict__ scale,
                             const float* __restrict__ bias,
                             const float* __restrict__ mean,
                             const float* __restrict__ var,
                             float* __restrict__ y,
                             unsigned long long n_total,
                             int C, int inner, float eps)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n_total) return;
    int c = (int)((i / (unsigned long long)inner) % (unsigned long long)C);
    float inv_std = rsqrtf(var[c] + eps);
    y[i] = (x[i] - mean[c]) * inv_std * scale[c] + bias[c];
}

// LayerNorm (last-axis): y = scale * (x - mean) / sqrt(var + eps) + bias
// Layout: (outer, D) — D is the normalized axis. Block per outer row.
__global__ void ln_lastaxis_f32(const float* __restrict__ x,
                                const float* __restrict__ scale,  // may be null (identity)
                                const float* __restrict__ bias,   // may be null (zero)
                                float* __restrict__ y,
                                int D,
                                int outer,
                                float eps,
                                int has_scale,
                                int has_bias)
{
    const int row = blockIdx.x;
    if (row >= outer) return;
    const float* xr = x + (size_t)row * D;
    float* yr       = y + (size_t)row * D;

    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    extern __shared__ float smem[];    // 2 * nth: sums, sqsums

    // pass 1: mean + mean-of-squares
    float s = 0.f, ss = 0.f;
    for (int j = tid; j < D; j += nth) {
        float v = xr[j];
        s  += v;
        ss += v * v;
    }
    smem[tid]        = s;
    smem[nth + tid]  = ss;
    __syncthreads();
    for (int off = nth >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            smem[tid]       += smem[tid + off];
            smem[nth + tid] += smem[nth + tid + off];
        }
        __syncthreads();
    }
    float mean = smem[0]       / (float)D;
    float meansq = smem[nth]   / (float)D;
    float var = meansq - mean * mean;
    if (var < 0.f) var = 0.f;
    float inv_std = rsqrtf(var + eps);

    // pass 2: normalize + affine
    for (int j = tid; j < D; j += nth) {
        float v = (xr[j] - mean) * inv_std;
        if (has_scale) v *= scale[j];
        if (has_bias)  v += bias[j];
        yr[j] = v;
    }
}

} // extern "C"
)CUDA";
}

// -------------------- BatchNormalization --------------------

struct BatchNorm_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    float eps = 1e-5f;
    unsigned long long n_total = 0;
    int C = 0, inner = 0;

    bool init() override {
        if (!(inputs.size() == 5 && outputs.size() >= 1)) return false;
        fallback = resolver_default_op_BatchNormalization(opset, ctx->attr_pool);
        fallback->ctx = ctx;
        fallback->opset = opset;
        fallback->op_type = op_type;
        fallback->inputs = inputs;
        fallback->outputs = outputs;
        fallback->attrs = attrs;
        fallback->init();
        eps = attribute(attr_key_t::epsilon, 1e-5f);
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* x = inputs[0];
        int training_mode = (int)attribute(attr_key_t::training_mode, (int64_t)0);
        if (training_mode) return true;
        if (x->type != NNR_DATA_TYPE_FLOAT32 || x->ndim < 2) return true;

        C = x->dims[1];
        size_t inr = 1;
        for (int d = 2; d < x->ndim; ++d) inr *= x->dims[d];
        inner = (int)inr;
        n_total = (unsigned long long)x->ndata;

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }

        CUfunction f = be->nvrtc.get("nnr_norm_f32", norm_source(),
                                     "bn_infer_f32", gpu::nvrtc_arch_option(be->device));
        if (!f) { return fallback->exec(); }

        float* d_x     = (float*)be->cache->ensure_device(inputs[0]);
        float* d_scale = (float*)be->cache->ensure_device(inputs[1]);
        float* d_bias  = (float*)be->cache->ensure_device(inputs[2]);
        float* d_mean  = (float*)be->cache->ensure_device(inputs[3]);
        float* d_var   = (float*)be->cache->ensure_device(inputs[4]);
        float* d_y     = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_scale || !d_bias || !d_mean || !d_var || !d_y) {
            return fallback->exec();
        }

        int _C = C, _inner = inner; float _eps = eps;
        void* args[] = { &d_x, &d_scale, &d_bias, &d_mean, &d_var, &d_y,
                         &n_total, &_C, &_inner, &_eps };
        unsigned block = 256;
        unsigned grid = (unsigned)((n_total + block - 1) / block);
        if (!gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args)) {
            return fallback->exec();
        }

        be->cache->mark_written(outputs[0]);
        return true;
    }
};

// -------------------- LayerNormalization --------------------

struct LayerNorm_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    float eps = 1e-5f;
    int axis = -1;
    int D = 0, outer = 0;

    bool init() override {
        if (!(inputs.size() >= 1 && outputs.size() >= 1)) return false;
        fallback = resolver_default_op_LayerNormalization(opset, ctx->attr_pool);
        fallback->ctx = ctx;
        fallback->opset = opset;
        fallback->op_type = op_type;
        fallback->inputs = inputs;
        fallback->outputs = outputs;
        fallback->attrs = attrs;
        fallback->init();
        eps  = attribute(attr_key_t::epsilon, 1e-5f);
        axis = (int)attribute(attr_key_t::axis, (int64_t)-1);
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* x = inputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32 || x->ndim < 1) return true;

        int caxis = axis < 0 ? axis + x->ndim : axis;
        if (caxis != x->ndim - 1) return true;  // only last-axis supported

        D = x->dims[caxis];
        outer = 1;
        for (int d = 0; d < caxis; ++d) outer *= x->dims[d];

        // Training mode outputs (mean, inv_std) not produced by this path.
        // If the user requested them, defer.
        if (outputs.size() > 1 && outputs[1]) return true;

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }

        CUfunction f = be->nvrtc.get("nnr_norm_f32", norm_source(),
                                     "ln_lastaxis_f32", gpu::nvrtc_arch_option(be->device));
        if (!f) { return fallback->exec(); }

        float* d_x     = (float*)be->cache->ensure_device(inputs[0]);
        float* d_scale = (inputs.size() >= 2 && inputs[1]) ? (float*)be->cache->ensure_device(inputs[1]) : nullptr;
        float* d_bias  = (inputs.size() >= 3 && inputs[2]) ? (float*)be->cache->ensure_device(inputs[2]) : nullptr;
        float* d_y     = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        int _D = D, _outer = outer; float _eps = eps;
        int has_scale = d_scale ? 1 : 0;
        int has_bias  = d_bias  ? 1 : 0;
        void* args[] = { &d_x, &d_scale, &d_bias, &d_y,
                         &_D, &_outer, &_eps, &has_scale, &has_bias };

        unsigned block = 256;
        while ((int)block > D && block > 32) block >>= 1;
        if (block < 32) block = 32;
        unsigned shared = 2 * block * sizeof(float);
        if (!gpu::nvrtc_launch(be->device, f, (unsigned)outer, 1, 1,
                               block, 1, 1, args, shared)) {
            return fallback->exec();
        }

        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_BatchNormalization(int opset, pool_t& pool) {
    return pool_new<BatchNorm_cuda>(pool);
}

operator_t* resolver_cuda_op_LayerNormalization(int opset, pool_t& pool) {
    return pool_new<LayerNorm_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
