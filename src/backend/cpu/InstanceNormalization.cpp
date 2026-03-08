#include "nnr.h"
#include "util.h"
#include "allocator.h"
#include "thread_pool.h"
#ifdef NNR_ARCH_X64
#include "backend/x64/ops_x64.h"
#elifdef NNR_ARCH_ARM64
#include "backend/arm/conv_neon.h"
#endif
// hsum512f, compute_mean_var_avx512, affine_avx512 are in backend/x64/ops_x64.h

namespace nnr {

namespace {

struct InstanceNormalization_operator : public operator_t {
    float epsilon;
    // Pre-computed per-(N,C) linear transform for scroll strip pass:
    // y[i] = cached_alpha[nc] * x[i] + cached_beta[nc]
    // Arena-allocated in scroll_pre_exec(); freed in bulk when arena scope exits.
    float* cached_alpha = nullptr;
    float* cached_beta = nullptr;

    bool init() override {
        if (!(inputs.size() == 3 && outputs.size() >= 1)) {
            return false;
        }
        epsilon = attribute(attr_key_t::epsilon, 1e-05f);
        return true;
    }

    scroll_info_t scroll_info() const override {
        if (inputs[0]->ndim != 4) return {};
        return { .scrollable = true, .needs_pre_pass = true };
    }

    bool scroll_pre_exec() override {
        const tensor_t* x = inputs[0];
        const float* px = (const float*)x->data;
        const float* pscale = (const float*)inputs[1]->data;
        const float* pb = (const float*)inputs[2]->data;

        int N = x->dims[0], C = x->dims[1];
        int NC = N * C;
        int spatial = x->dims[2] * x->dims[3];
        float inv_sp = 1.0f / spatial;

        // Single arena allocation for both arrays to avoid invalidation
        // if a second alloc triggers arena growth.
        float* buf = ctx->arena.alloc_arr<float>(NC * 2);
        if (!buf) return false;

        cached_alpha = buf;
        cached_beta = buf + NC;

        for (int j = 0; j < NC; ++j) {
            int c = j % C;
            const float* src = px + (size_t)j * spatial;
            float mean, var;
#ifdef NNR_ARCH_X64
            compute_mean_var_avx512(src, spatial, mean, var);
#elifdef NNR_ARCH_ARM64
            compute_mean_var_neon(src, spatial, mean, var);
#else
            { float s = 0, s2 = 0;
              for (int i = 0; i < spatial; ++i) { s += src[i]; s2 += src[i] * src[i]; }
              mean = s / spatial; var = s2 / spatial - mean * mean; }
#endif
            var = fmaxf(var, 0.f);
            float alpha = pscale[c] / sqrtf(var + epsilon);
            cached_alpha[j] = alpha;
            cached_beta[j] = pb[c] - alpha * mean;
        }
        return true;
    }

    bool exec_strip(int out_row_start, int out_rows,
        int in_row_start, int in_rows) override
    {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];

        int NC = x->dims[0] * x->dims[1];
        int iH = x->dims[2];
        int W = x->dims[3];
        int oH = y->dims[2];
        int clamp_H = ring_out.orig_H > 0 ? ring_out.orig_H : oH;
        int out_end = std::min(out_row_start + out_rows, clamp_H);
        int count = (out_end - out_row_start) * W;
        if (count <= 0) return true;

        const float* px = (const float*)x->data;
        float* py = (float*)y->data;

        for (int nc = 0; nc < NC; ++nc) {
            const float* src = px + (size_t)nc * iH * W + (size_t)out_row_start * W;
            float* dst = py + (size_t)nc * oH * W + (size_t)out_row_start * W;
            float a = cached_alpha[nc];
            float b = cached_beta[nc];
#ifdef NNR_ARCH_X64
            affine_avx512(dst, src, count, a, b);
#elifdef NNR_ARCH_ARM64
            affine_neon(dst, src, count, a, b);
#else
            for (int i = 0; i < count; ++i)
                dst[i] = a * src[i] + b;
#endif
        }
        return true;
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        const tensor_t* scale = inputs[1];
        const tensor_t* b = inputs[2];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        const T* pscale = (const T*)scale->data;
        const T* pb = (const T*)b->data;
        T* py = (T*)y->data;
        int N = x->dims[0];
        int C = x->dims[1];
        int NC = N * C;
        int channel = std::reduce(x->dims + 2, x->dims + x->ndim, 1, std::multiplies<>{});
        float inv_ch = 1.0f / channel;
#if defined(NNR_ARCH_X64) || defined(NNR_ARCH_ARM64)
        if constexpr (std::is_same_v<T, float>) {
            nnr::for_static(0, NC, NC > 4, [&](int j) {
                const float* src = px + (size_t)j * channel;
                float* dst = py + (size_t)j * channel;
                int jc = j % C;
                float mean, var;
#ifdef NNR_ARCH_X64
                compute_mean_var_avx512(src, channel, mean, var);
#else
                compute_mean_var_neon(src, channel, mean, var);
#endif
                var = fmaxf(var, 0.f);
                float alpha = pscale[jc] / sqrtf(var + epsilon);
                float b_shift = (float)pb[jc] - alpha * mean;
#ifdef NNR_ARCH_X64
                affine_avx512(dst, src, channel, alpha, b_shift);
#else
                affine_neon(dst, src, channel, alpha, b_shift);
#endif
            });
            return true;
        }
#endif
        for (int j = 0; j < NC; ++j) {
            const T* src = px + (size_t)j * channel;
            T* dst = py + (size_t)j * channel;
            int jc = j % C;
            // Mean
            T sum = 0;
            for (int i = 0; i < channel; ++i)
                sum += src[i];
            T mean = sum * inv_ch;
            // Variance (d*d instead of pow)
            T var_sum = 0;
            for (int i = 0; i < channel; ++i) {
                T d = src[i] - mean;
                var_sum += d * d;
            }
            // Precompute alpha = scale/denom, beta = bias
            float alpha = (float)(pscale[jc] / sqrt((double)(var_sum * inv_ch) + epsilon));
            float beta = (float)pb[jc];
            // Fused normalize: dst = alpha * (src - mean) + beta
            for (int i = 0; i < channel; ++i)
                dst[i] = (T)(alpha * (float)(src[i] - mean) + beta);
        }
        return true;
    }

    bool exec() override {
        return typed_exec<InstanceNormalization_operator,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace {

// @nnr-meta-op mt=static scroll=yes
operator_t* resolver_default_op_InstanceNormalization(int opset, pool_t& pool) { return pool_new<InstanceNormalization_operator>(pool); }

} // namespace nnr
