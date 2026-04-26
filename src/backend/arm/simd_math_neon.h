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

// tanh(x) = 2 * sigmoid(2x) - 1
// @nnr-meta isa=NEON dtype=fp32
static inline float32x4_t tanh_neon_ps(float32x4_t x) {
    const float32x4_t two = vdupq_n_f32(2.0f);
    const float32x4_t neg_one = vdupq_n_f32(-1.0f);
    float32x4_t s = sigmoid_neon_ps(vmulq_f32(two, x));
    return vfmaq_f32(neg_one, two, s);  // -1 + 2*s
}

// Fast erf(x) for 4 floats using NEON.
// Abramowitz & Stegun 7.1.26 (5-term polynomial × exp(-x²)). Max error ~1.5e-7.
//   t = 1 / (1 + p*|x|)
//   erf(x) = sign(x) * (1 - (a1·t + a2·t² + a3·t³ + a4·t⁴ + a5·t⁵) · exp(-x²))
// @nnr-meta isa=NEON dtype=fp32
static inline float32x4_t erf_neon_ps(float32x4_t x) {
    const uint32x4_t sign_mask = vdupq_n_u32(0x80000000u);
    uint32x4_t sign = vandq_u32(vreinterpretq_u32_f32(x), sign_mask);
    float32x4_t ax = vabsq_f32(x);

    const float32x4_t p  = vdupq_n_f32(0.3275911f);
    const float32x4_t a1 = vdupq_n_f32(0.254829592f);
    const float32x4_t a2 = vdupq_n_f32(-0.284496736f);
    const float32x4_t a3 = vdupq_n_f32(1.421413741f);
    const float32x4_t a4 = vdupq_n_f32(-1.453152027f);
    const float32x4_t a5 = vdupq_n_f32(1.061405429f);
    const float32x4_t one = vdupq_n_f32(1.0f);

    // t = 1 / (1 + p*|x|) via rcpe + two Newton steps (~24-bit precision)
    float32x4_t denom = vfmaq_f32(one, p, ax);
    float32x4_t t = vrecpeq_f32(denom);
    t = vmulq_f32(t, vrecpsq_f32(denom, t));
    t = vmulq_f32(t, vrecpsq_f32(denom, t));

    // Horner: ((((a5·t + a4)·t + a3)·t + a2)·t + a1)·t
    float32x4_t poly = vfmaq_f32(a4, a5, t);
    poly = vfmaq_f32(a3, poly, t);
    poly = vfmaq_f32(a2, poly, t);
    poly = vfmaq_f32(a1, poly, t);
    poly = vmulq_f32(poly, t);

    // exp(-x²)
    float32x4_t ex = exp_neon_ps(vnegq_f32(vmulq_f32(ax, ax)));

    // 1 - poly·exp(-x²), then re-apply sign
    float32x4_t r = vfmsq_f32(one, poly, ex);
    return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(r), sign));
}

// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
// @nnr-meta isa=NEON dtype=fp32
static inline float32x4_t gelu_neon_ps(float32x4_t x) {
    const float32x4_t inv_sqrt2 = vdupq_n_f32(0.7071067811865476f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t e = erf_neon_ps(vmulq_f32(x, inv_sqrt2));
    return vmulq_f32(vmulq_f32(half, x), vaddq_f32(one, e));
}

// Single-threaded inplace sigmoid kernel — for callers that already provide
// outer parallelism (e.g. scroll strip, conv epilogue).
// @nnr-meta isa=NEON dtype=fp32
static inline void sigmoid_neon_kernel(float* data, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(data + i, sigmoid_neon_ps(vld1q_f32(data + i)));
    for (; i < n; i++)
        data[i] = 1.0f / (1.0f + expf(-data[i]));
}

// Single-threaded SiLU kernel (in-place safe: src == dst OK).
// @nnr-meta isa=NEON dtype=fp32
static inline void silu_neon_kernel(const float* src, float* dst, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(dst + i, silu_neon_ps(vld1q_f32(src + i)));
    for (; i < n; i++)
        dst[i] = src[i] / (1.0f + expf(-src[i]));
}

// Single-threaded tanh kernel (in-place safe: src == dst OK).
// @nnr-meta isa=NEON dtype=fp32
static inline void tanh_neon_kernel(const float* src, float* dst, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(dst + i, tanh_neon_ps(vld1q_f32(src + i)));
    for (; i < n; i++)
        dst[i] = tanhf(src[i]);
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

// Apply GELU (exact, erf-based): dst[i] = 0.5 * x * (1 + erf(x / sqrt(2))) (threaded, NEON).
// @nnr-meta isa=NEON dtype=fp32
inline void gelu_neon(const float* src, float* dst, size_t n) {
    constexpr size_t CHUNK = 16384;
    int nchunks = (int)((n + CHUNK - 1) / CHUNK);
    nnr::for_static(0, nchunks, nchunks > 1, [&](int c) {
        size_t start = (size_t)c * CHUNK;
        size_t end = std::min(start + CHUNK, n);
        size_t i = start;
        for (; i + 4 <= end; i += 4) {
            float32x4_t v = vld1q_f32(src + i);
            vst1q_f32(dst + i, gelu_neon_ps(v));
        }
        const float inv_sqrt2_s = 0.7071067811865476f;
        for (; i < end; i++)
            dst[i] = 0.5f * src[i] * (1.0f + std::erf(src[i] * inv_sqrt2_s));
    });
}

// Apply GELU to FP16 array: widen to FP32, compute, narrow back (threaded, NEON).
// External layout is uint16_t bit-pattern for FP16 (NNR convention).
// @nnr-meta isa=NEON dtype=fp16
inline void gelu_neon_fp16(const uint16_t* src, uint16_t* dst, size_t n) {
    constexpr size_t CHUNK = 16384;
    int nchunks = (int)((n + CHUNK - 1) / CHUNK);
    nnr::for_static(0, nchunks, nchunks > 1, [&](int c) {
        size_t start = (size_t)c * CHUNK;
        size_t end = std::min(start + CHUNK, n);
        size_t i = start;
        for (; i + 8 <= end; i += 8) {
            float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(src + i));
            float32x4_t lo = vcvt_f32_f16(vget_low_f16(vh));
            float32x4_t hi = vcvt_f32_f16(vget_high_f16(vh));
            float16x4_t glo = vcvt_f16_f32(gelu_neon_ps(lo));
            float16x4_t ghi = vcvt_f16_f32(gelu_neon_ps(hi));
            vst1q_u16(dst + i, vreinterpretq_u16_f16(vcombine_f16(glo, ghi)));
        }
        if (i < end) {
            // 4-lane tail via widen→compute→narrow.
            for (; i + 4 <= end; i += 4) {
                float16x4_t h = vreinterpret_f16_u16(vld1_u16(src + i));
                float32x4_t f = vcvt_f32_f16(h);
                float16x4_t g = vcvt_f16_f32(gelu_neon_ps(f));
                vst1_u16(dst + i, vreinterpret_u16_f16(g));
            }
            // 1..3 leftover via single-lane convert.
            const float inv_sqrt2_s = 0.7071067811865476f;
            for (; i < end; i++) {
                float16x4_t h1 = vreinterpret_f16_u16(vdup_n_u16(src[i]));
                float f = vgetq_lane_f32(vcvt_f32_f16(h1), 0);
                float g = 0.5f * f * (1.0f + std::erf(f * inv_sqrt2_s));
                float16x4_t gh = vcvt_f16_f32(vdupq_n_f32(g));
                dst[i] = vget_lane_u16(vreinterpret_u16_f16(gh), 0);
            }
        }
    });
}

} // namespace nnr
#endif // NNR_ARCH_ARM64
