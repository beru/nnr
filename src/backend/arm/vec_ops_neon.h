#pragma once
// ARM NEON vectorized operations: clip, reduction, dot product.
// 128-bit (4 floats per vector) counterpart of x64/vec_ops_avx2.h.

#if defined(__aarch64__) || defined(_M_ARM64)

#ifdef _MSC_VER
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif
#include <algorithm>
#include <cmath>

namespace nnr::neon {

// Horizontal sum of 4 floats in a NEON register.
// @nnr-meta isa=NEON dtype=fp32
inline float hsum(float32x4_t v)
{
    return vaddvq_f32(v);  // ARMv8: single-instruction horizontal add
}

// Vectorized clamp: py[i] = clamp(px[i], minv, maxv)
// @nnr-meta isa=NEON dtype=fp32
inline void clip(const float* __restrict px, float* __restrict py, size_t len, float minv, float maxv)
{
    float32x4_t vmin = vdupq_n_f32(minv);
    float32x4_t vmax = vdupq_n_f32(maxv);
    size_t i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t v = vld1q_f32(px + i);
        v = vmaxq_f32(v, vmin);
        v = vminq_f32(v, vmax);
        vst1q_f32(py + i, v);
    }
    for (; i < len; ++i) {
        float v = px[i];
        v = std::max(v, minv);
        v = std::min(v, maxv);
        py[i] = v;
    }
}

// Fused bias-add + clamp: data[i] = clamp(data[i] + bias, minv, maxv)
// @nnr-meta isa=NEON dtype=fp32 fusion=post_op
inline void bias_clip(float* __restrict data, int len, float bias, float minv, float maxv)
{
    float32x4_t vb = vdupq_n_f32(bias);
    float32x4_t vmin = vdupq_n_f32(minv);
    float32x4_t vmax = vdupq_n_f32(maxv);
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t v = vaddq_f32(vld1q_f32(data + i), vb);
        vst1q_f32(data + i, vminq_f32(vmaxq_f32(v, vmin), vmax));
    }
    for (; i < len; ++i)
        data[i] = std::clamp(data[i] + bias, minv, maxv);
}

// Fused bias-add + relu: data[i] = max(data[i] + bias, 0)
// @nnr-meta isa=NEON dtype=fp32 fusion=post_op
inline void bias_relu(float* __restrict data, int len, float bias)
{
    float32x4_t vb = vdupq_n_f32(bias);
    float32x4_t vz = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t v = vaddq_f32(vld1q_f32(data + i), vb);
        vst1q_f32(data + i, vmaxq_f32(v, vz));
    }
    for (; i < len; ++i)
        data[i] = std::max(data[i] + bias, 0.0f);
}

// Bias-add only: data[i] += bias
// @nnr-meta isa=NEON dtype=fp32 fusion=post_op
inline void bias_add(float* __restrict data, int len, float bias)
{
    if (bias == 0.0f) return;
    float32x4_t vb = vdupq_n_f32(bias);
    int i = 0;
    for (; i + 4 <= len; i += 4)
        vst1q_f32(data + i, vaddq_f32(vld1q_f32(data + i), vb));
    for (; i < len; ++i)
        data[i] += bias;
}

// HardSwish: data[i] = (data[i] + bias) * clamp((data[i] + bias) / 6 + 0.5, 0, 1)
// @nnr-meta isa=NEON dtype=fp32 fusion=post_op
inline void bias_hardswish(float* __restrict data, int len, float bias)
{
    float32x4_t vb   = vdupq_n_f32(bias);
    float32x4_t v6   = vdupq_n_f32(1.0f / 6.0f);
    float32x4_t vhalf = vdupq_n_f32(0.5f);
    float32x4_t vzero = vdupq_n_f32(0.0f);
    float32x4_t vone  = vdupq_n_f32(1.0f);
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t x = vaddq_f32(vld1q_f32(data + i), vb);
        float32x4_t t = vminq_f32(vmaxq_f32(
            vaddq_f32(vmulq_f32(x, v6), vhalf), vzero), vone);
        vst1q_f32(data + i, vmulq_f32(x, t));
    }
    for (; i < len; ++i) {
        float x = data[i] + bias;
        data[i] = x * std::max(0.0f, std::min(1.0f, x / 6.0f + 0.5f));
    }
}

// HardSwish (no bias): py[i] = px[i] * clamp(px[i] / 6 + 0.5, 0, 1)
// @nnr-meta isa=NEON dtype=fp32
inline void hardswish(const float* __restrict px, float* __restrict py, size_t len)
{
    float32x4_t v6   = vdupq_n_f32(1.0f / 6.0f);
    float32x4_t vhalf = vdupq_n_f32(0.5f);
    float32x4_t vzero = vdupq_n_f32(0.0f);
    float32x4_t vone  = vdupq_n_f32(1.0f);
    size_t i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t x = vld1q_f32(px + i);
        float32x4_t t = vminq_f32(vmaxq_f32(
            vaddq_f32(vmulq_f32(x, v6), vhalf), vzero), vone);
        vst1q_f32(py + i, vmulq_f32(x, t));
    }
    for (; i < len; ++i) {
        float x = px[i];
        py[i] = x * std::max(0.0f, std::min(1.0f, x / 6.0f + 0.5f));
    }
}

// Vectorized leaky relu: py[i] = px[i] >= 0 ? px[i] : px[i] * alpha
// @nnr-meta isa=NEON dtype=fp32
inline void leaky_relu(const float* __restrict px, float* __restrict py, int len, float alpha)
{
    float32x4_t va = vdupq_n_f32(alpha);
    float32x4_t vz = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t v = vld1q_f32(px + i);
        float32x4_t neg = vmulq_f32(v, va);
        uint32x4_t mask = vcltq_f32(v, vz);
        vst1q_f32(py + i, vbslq_f32(mask, neg, v));
    }
    for (; i < len; ++i)
        py[i] = px[i] >= 0 ? px[i] : px[i] * alpha;
}

// Fused bias-add + leaky relu: data[i] = (data[i]+bias) >= 0 ? (data[i]+bias) : (data[i]+bias)*alpha
// @nnr-meta isa=NEON dtype=fp32 fusion=post_op
inline void bias_leaky_relu(float* __restrict data, int len, float bias, float alpha)
{
    float32x4_t vb = vdupq_n_f32(bias);
    float32x4_t va = vdupq_n_f32(alpha);
    float32x4_t vz = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t v = vaddq_f32(vld1q_f32(data + i), vb);
        float32x4_t neg = vmulq_f32(v, va);
        uint32x4_t mask = vcltq_f32(v, vz);
        vst1q_f32(data + i, vbslq_f32(mask, neg, v));
    }
    for (; i < len; ++i) {
        float v = data[i] + bias;
        data[i] = v >= 0 ? v : v * alpha;
    }
}

// Vectorized horizontal sum of a float array with 4 independent accumulators.
// @nnr-meta isa=NEON dtype=fp32
inline float reduce_sum(const float* data, int len)
{
    float32x4_t s0 = vdupq_n_f32(0.0f);
    float32x4_t s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f);
    float32x4_t s3 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 16 <= len; i += 16) {
        s0 = vaddq_f32(s0, vld1q_f32(data + i));
        s1 = vaddq_f32(s1, vld1q_f32(data + i + 4));
        s2 = vaddq_f32(s2, vld1q_f32(data + i + 8));
        s3 = vaddq_f32(s3, vld1q_f32(data + i + 12));
    }
    s0 = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
    for (; i + 4 <= len; i += 4)
        s0 = vaddq_f32(s0, vld1q_f32(data + i));
    float total = vaddvq_f32(s0);
    for (; i < len; i++)
        total += data[i];
    return total;
}

// Vectorized dot product with 4 independent accumulators to hide FMA latency.
// @nnr-meta isa=NEON dtype=fp32
inline float dot_product(const float* __restrict a, const float* __restrict b, int len)
{
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 16 <= len; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),      vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    // Reduce 4 accumulators to 1
    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);
    for (; i + 4 <= len; i += 4)
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    float s = vaddvq_f32(acc0);
    for (; i < len; ++i)
        s += a[i] * b[i];
    return s;
}

// Affine transform: dst[i] = alpha[i] * src[i] + beta[i]  (per-channel, NHWC)
// @nnr-meta isa=NEON dtype=fp32
inline void affine_channel(const float* __restrict src, float* __restrict dst,
                           const float* __restrict alpha, const float* __restrict beta, int C)
{
    int c = 0;
    for (; c + 4 <= C; c += 4) {
        float32x4_t va = vld1q_f32(alpha + c);
        float32x4_t vb = vld1q_f32(beta + c);
        vst1q_f32(dst + c, vfmaq_f32(vb, va, vld1q_f32(src + c)));
    }
    for (; c < C; ++c)
        dst[c] = alpha[c] * src[c] + beta[c];
}

// Affine transform: dst[i] = a * src[i] + b  (broadcast scalar)
// @nnr-meta isa=NEON dtype=fp32
inline void affine_broadcast(const float* __restrict src, float* __restrict dst,
                             int len, float a, float b)
{
    float32x4_t va = vdupq_n_f32(a);
    float32x4_t vb = vdupq_n_f32(b);
    int i = 0;
    for (; i + 4 <= len; i += 4)
        vst1q_f32(dst + i, vfmaq_f32(vb, va, vld1q_f32(src + i)));
    for (; i < len; ++i)
        dst[i] = a * src[i] + b;
}

} // namespace nnr::neon

#endif // __aarch64__ || _M_ARM64
