#pragma once
// AVX-2 LRN (Local Response Normalization) kernels.
// Mirrors lrn_avx512.h; uses bit-manipulation for log2/exp2 since AVX-2
// lacks _mm512_getexp_ps / getmant_ps / scalef_ps.

#include <immintrin.h>
#include "cpu_features.h"

namespace nnr {

// Fast SIMD log2(x) approximation (~20-bit accuracy) using IEEE-754 bit math.
// log2(x) = exponent(x) + log2(mantissa), mantissa in [1, 2).
// @nnr-meta isa=AVX2 dtype=fp32
inline __m256 fast_log2_avx2(__m256 x) {
    __m256i ix = _mm256_castps_si256(x);
    // exponent = ((ix >> 23) & 0xff) - 127
    __m256i exp_bits = _mm256_srli_epi32(ix, 23);
    exp_bits = _mm256_and_si256(exp_bits, _mm256_set1_epi32(0xff));
    exp_bits = _mm256_sub_epi32(exp_bits, _mm256_set1_epi32(127));
    __m256 exponent = _mm256_cvtepi32_ps(exp_bits);

    // mantissa: clear exponent, set bias to 127 → mantissa in [1, 2)
    __m256i mant_bits = _mm256_and_si256(ix, _mm256_set1_epi32(0x007fffff));
    mant_bits = _mm256_or_si256(mant_bits, _mm256_set1_epi32(0x3f800000));
    __m256 m = _mm256_castsi256_ps(mant_bits);

    // Polynomial: log2(m) ≈ -1/3 * m^3 + 4/3 * m^2 - 2 * m + 1 for m in [1, 2)
    __m256 a3 = _mm256_set1_ps(-0.3333333f);
    __m256 a2 = _mm256_set1_ps(1.3333333f);
    __m256 a1 = _mm256_set1_ps(-2.0f);
    __m256 a0 = _mm256_set1_ps(1.0f);
    __m256 log2_m = _mm256_fmadd_ps(a3, m, a2);
    log2_m = _mm256_fmadd_ps(log2_m, m, a1);
    log2_m = _mm256_fmadd_ps(log2_m, m, a0);
    return _mm256_add_ps(exponent, log2_m);
}

// Fast SIMD exp2(x) approximation (~20-bit accuracy) — manual 2^n reconstruction.
// @nnr-meta isa=AVX2 dtype=fp32
inline __m256 fast_exp2_avx2(__m256 x) {
    __m256 n = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 f = _mm256_sub_ps(x, n);

    __m256 c3 = _mm256_set1_ps(0.079442f);
    __m256 c2 = _mm256_set1_ps(0.227411f);
    __m256 c1 = _mm256_set1_ps(0.693147f);
    __m256 c0 = _mm256_set1_ps(1.0f);
    __m256 exp2_f = _mm256_fmadd_ps(c3, f, c2);
    exp2_f = _mm256_fmadd_ps(exp2_f, f, c1);
    exp2_f = _mm256_fmadd_ps(exp2_f, f, c0);

    // Reconstruct 2^n by manipulating IEEE-754 exponent
    __m256i n_int = _mm256_cvtps_epi32(n);
    __m256i bias  = _mm256_set1_epi32(127);
    __m256i e     = _mm256_slli_epi32(_mm256_add_epi32(n_int, bias), 23);
    __m256  pow2n = _mm256_castsi256_ps(e);
    return _mm256_mul_ps(exp2_f, pow2n);
}

// pow(base, -0.5) = rsqrt(base), with Newton-Raphson refinement.
struct pow_neg_050_avx2 {
    static __m256 scale(__m256 base) {
        __m256 half = _mm256_set1_ps(0.5f);
        __m256 three = _mm256_set1_ps(3.0f);
        __m256 y0 = _mm256_rsqrt_ps(base);  // ~12-bit
        // NR: y1 = 0.5 * y0 * (3 - base * y0^2) → ~24-bit
        return _mm256_mul_ps(half,
            _mm256_mul_ps(y0, _mm256_fnmadd_ps(base, _mm256_mul_ps(y0, y0), three)));
    }
};

// pow(base, -0.75) = rsqrt(base) * rsqrt(sqrt(base)), with NR refinement.
struct pow_neg_075_avx2 {
    static __m256 scale(__m256 base) {
        __m256 half = _mm256_set1_ps(0.5f);
        __m256 three = _mm256_set1_ps(3.0f);
        __m256 y0 = _mm256_rsqrt_ps(base);
        __m256 rsqrt_base = _mm256_mul_ps(half,
            _mm256_mul_ps(y0, _mm256_fnmadd_ps(base, _mm256_mul_ps(y0, y0), three)));
        __m256 sqrt_base = _mm256_sqrt_ps(base);
        __m256 y1 = _mm256_rsqrt_ps(sqrt_base);
        __m256 rsqrt_sqrt_base = _mm256_mul_ps(half,
            _mm256_mul_ps(y1, _mm256_fnmadd_ps(sqrt_base, _mm256_mul_ps(y1, y1), three)));
        return _mm256_mul_ps(rsqrt_base, rsqrt_sqrt_base);
    }
};

// pow(base, -1.0) = rcp(base) with NR step.
struct pow_neg_100_avx2 {
    static __m256 scale(__m256 base) {
        __m256 r0 = _mm256_rcp_ps(base);  // ~12-bit
        // NR: r1 = r0 * (2 - base * r0) → ~24-bit
        __m256 two = _mm256_set1_ps(2.0f);
        return _mm256_mul_ps(r0, _mm256_fnmadd_ps(base, r0, two));
    }
};

// Generic pow(base, -beta) = exp2(-beta * log2(base)).
struct pow_neg_general_avx2 {
    float beta;
    __m256 scale(__m256 base) const {
        __m256 log2_base = fast_log2_avx2(base);
        __m256 vbeta = _mm256_set1_ps(-beta);
        return fast_exp2_avx2(_mm256_mul_ps(vbeta, log2_base));
    }
};

// Templated AVX-2 LRN inner loop. PowFn::scale(base) returns base^(-beta).
template <typename PowFn>
// @nnr-meta isa=AVX2 dtype=fp32 layout=NCHW tiling=spatial
inline void lrn_channel_avx2(const float* __restrict input, float* __restrict output,
    int nc, int C, int spatial, int c0, int c1, float alpha, float bias, PowFn pow_fn, int& hw)
{
    const float* xc = input + (size_t)nc * spatial;
    float* yc = output + (size_t)nc * spatial;
    int n = nc / C;
    __m256 valpha = _mm256_set1_ps(alpha);
    __m256 vbias = _mm256_set1_ps(bias);

    for (; hw + 8 <= spatial; hw += 8) {
        __m256 sum = _mm256_setzero_ps();
        for (int ci = c0; ci <= c1; ++ci) {
            __m256 v = _mm256_loadu_ps(input + ((size_t)n * C + ci) * spatial + hw);
            sum = _mm256_fmadd_ps(v, v, sum);
        }
        __m256 base = _mm256_fmadd_ps(valpha, sum, vbias);
        __m256 xv = _mm256_loadu_ps(xc + hw);
        _mm256_storeu_ps(yc + hw, _mm256_mul_ps(xv, pow_fn.scale(base)));
    }
}

} // namespace nnr
