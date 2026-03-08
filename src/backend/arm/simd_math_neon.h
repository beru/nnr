#pragma once
// NEON vectorized math functions: exp, sigmoid, SiLU (Swish).
// Fast polynomial approximation matching the AVX-512 version.
// Max relative error ~1.5e-7 (sufficient for inference).

#ifdef NNR_ARCH_ARM64
#include <arm_neon.h>

namespace nnr {

// Fast exp(x) for 4 floats using NEON.
// Range reduction to [-ln2/2, ln2/2] + degree-5 minimax polynomial.
// @nnr-meta isa=NEON dtype=fp32
static inline float32x4_t exp_neon_ps(float32x4_t x) {
    const float32x4_t log2e  = vdupq_n_f32(1.44269504089f);
    const float32x4_t ln2_hi = vdupq_n_f32(0.693145751953125f);
    const float32x4_t ln2_lo = vdupq_n_f32(1.428606765330187e-06f);
    const float32x4_t one    = vdupq_n_f32(1.0f);

    // Clamp to prevent overflow/underflow
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));
    x = vminq_f32(x, vdupq_n_f32(88.0f));

    // Range reduction: n = round(x * log2(e))
    float32x4_t t = vmulq_f32(x, log2e);
    float32x4_t n = vrndnq_f32(t);  // round to nearest

    // r = x - n * ln(2) (two-part for precision)
    float32x4_t r = vfmsq_f32(x, n, ln2_hi);   // r = x - n * ln2_hi
    r = vfmsq_f32(r, n, ln2_lo);                 // r -= n * ln2_lo

    // Polynomial: exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120
    const float32x4_t c2 = vdupq_n_f32(0.500000000f);
    const float32x4_t c3 = vdupq_n_f32(0.166666672f);
    const float32x4_t c4 = vdupq_n_f32(0.041666664f);
    const float32x4_t c5 = vdupq_n_f32(0.008333345f);

    float32x4_t p = vfmaq_f32(c4, c5, r);
    p = vfmaq_f32(c3, p, r);
    p = vfmaq_f32(c2, p, r);
    p = vfmaq_f32(one, p, r);
    p = vfmaq_f32(one, p, r);

    // Reconstruct: 2^n * exp(r) via integer exponent manipulation
    int32x4_t ni = vcvtnq_s32_f32(n);
    ni = vaddq_s32(ni, vdupq_n_s32(127));
    ni = vshlq_n_s32(ni, 23);
    float32x4_t scale = vreinterpretq_f32_s32(ni);

    return vmulq_f32(p, scale);
}

// sigmoid(x) = 1 / (1 + exp(-x))
// @nnr-meta isa=NEON dtype=fp32
static inline float32x4_t sigmoid_neon_ps(float32x4_t x) {
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t neg_x = vnegq_f32(x);
    float32x4_t exp_neg = exp_neon_ps(neg_x);
    // 1 / (1 + exp(-x)) using Newton-Raphson reciprocal
    float32x4_t denom = vaddq_f32(one, exp_neg);
    float32x4_t recip = vrecpeq_f32(denom);
    recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));  // 1 Newton step
    recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));  // 2nd step for ~24-bit precision
    return recip;
}

// SiLU(x) = x * sigmoid(x)
// @nnr-meta isa=NEON dtype=fp32
static inline float32x4_t silu_neon_ps(float32x4_t x) {
    return vmulq_f32(x, sigmoid_neon_ps(x));
}

// Apply sigmoid to a contiguous float array (threaded, NEON).
// @nnr-meta isa=NEON dtype=fp32
inline void sigmoid_neon(float* data, size_t n) {
    constexpr size_t CHUNK = 16384;
    int nchunks = (int)((n + CHUNK - 1) / CHUNK);
    nnr::for_static(0, nchunks, nchunks > 1, [&](int c) {
        size_t start = (size_t)c * CHUNK;
        size_t end = std::min(start + CHUNK, n);
        size_t i = start;
        for (; i + 4 <= end; i += 4) {
            float32x4_t v = vld1q_f32(data + i);
            vst1q_f32(data + i, sigmoid_neon_ps(v));
        }
        for (; i < end; i++)
            data[i] = 1.0f / (1.0f + expf(-data[i]));
    });
}

// Apply SiLU (x * sigmoid(x)): src → dst (threaded, NEON).
// @nnr-meta isa=NEON dtype=fp32
inline void silu_neon(const float* src, float* dst, size_t n) {
    constexpr size_t CHUNK = 16384;
    int nchunks = (int)((n + CHUNK - 1) / CHUNK);
    nnr::for_static(0, nchunks, nchunks > 1, [&](int c) {
        size_t start = (size_t)c * CHUNK;
        size_t end = std::min(start + CHUNK, n);
        size_t i = start;
        for (; i + 4 <= end; i += 4) {
            float32x4_t v = vld1q_f32(src + i);
            vst1q_f32(dst + i, silu_neon_ps(v));
        }
        for (; i < end; i++)
            dst[i] = src[i] / (1.0f + expf(-src[i]));
    });
}

} // namespace nnr
#endif // NNR_ARCH_ARM64
