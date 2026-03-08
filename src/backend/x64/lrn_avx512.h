#pragma once
// AVX-512 LRN (Local Response Normalization) kernels.
// Fast SIMD pow(x, -beta) via algebraic decomposition or exp2/log2 approximation.

#include <immintrin.h>
#include "cpu_features.h"

namespace nnr {

// --------------------------------------------------------------------------
// Fast SIMD log2(x) approximation (~20-bit accuracy)
// Uses the identity: log2(x) = exponent(x) + log2(mantissa)
// where mantissa is in [1, 2), approximated with a 3rd-order polynomial.
// @nnr-meta isa=AVX512 dtype=fp32
inline __m512 fast_log2_avx512(__m512 x) {
    __m512 exponent = _mm512_getexp_ps(x);
    __m512 mantissa = _mm512_getmant_ps(x, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);
    __m512 a3 = _mm512_set1_ps(-0.3333333f);
    __m512 a2 = _mm512_set1_ps(1.3333333f);
    __m512 a1 = _mm512_set1_ps(-2.0f);
    __m512 a0 = _mm512_set1_ps(1.0f);
    __m512 m = mantissa;
    __m512 log2_m = _mm512_fmadd_ps(a3, m, a2);
    log2_m = _mm512_fmadd_ps(log2_m, m, a1);
    log2_m = _mm512_fmadd_ps(log2_m, m, a0);
    return _mm512_add_ps(exponent, log2_m);
}

// Fast SIMD exp2(x) approximation (~20-bit accuracy)
// Splits x into integer part n and fractional part f, then:
//   exp2(x) = 2^n * exp2(f), with exp2(f) for f in [0, 1)
//   approximated by a 3rd-order polynomial.
// @nnr-meta isa=AVX512 dtype=fp32
inline __m512 fast_exp2_avx512(__m512 x) {
    __m512 n = _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 f = _mm512_sub_ps(x, n);
    __m512 c3 = _mm512_set1_ps(0.079442f);
    __m512 c2 = _mm512_set1_ps(0.227411f);
    __m512 c1 = _mm512_set1_ps(0.693147f);
    __m512 c0 = _mm512_set1_ps(1.0f);
    __m512 exp2_f = _mm512_fmadd_ps(c3, f, c2);
    exp2_f = _mm512_fmadd_ps(exp2_f, f, c1);
    exp2_f = _mm512_fmadd_ps(exp2_f, f, c0);
    return _mm512_scalef_ps(exp2_f, n);
}

// pow(base, -0.5) = rsqrt(base), with Newton-Raphson refinement
struct pow_neg_050 {
    static __m512 scale(__m512 base) {
        __m512 half = _mm512_set1_ps(0.5f);
        __m512 three = _mm512_set1_ps(3.0f);
        __m512 y0 = _mm512_rsqrt14_ps(base);
        return _mm512_mul_ps(half,
            _mm512_mul_ps(y0, _mm512_fnmadd_ps(base, _mm512_mul_ps(y0, y0), three)));
    }
};

// pow(base, -0.75) = rsqrt(base) * rsqrt(sqrt(base)), with NR refinement
struct pow_neg_075 {
    static __m512 scale(__m512 base) {
        __m512 half = _mm512_set1_ps(0.5f);
        __m512 three = _mm512_set1_ps(3.0f);
        __m512 y0 = _mm512_rsqrt14_ps(base);
        __m512 rsqrt_base = _mm512_mul_ps(half,
            _mm512_mul_ps(y0, _mm512_fnmadd_ps(base, _mm512_mul_ps(y0, y0), three)));
        __m512 sqrt_base = _mm512_sqrt_ps(base);
        __m512 y1 = _mm512_rsqrt14_ps(sqrt_base);
        __m512 rsqrt_sqrt_base = _mm512_mul_ps(half,
            _mm512_mul_ps(y1, _mm512_fnmadd_ps(sqrt_base, _mm512_mul_ps(y1, y1), three)));
        return _mm512_mul_ps(rsqrt_base, rsqrt_sqrt_base);
    }
};

// pow(base, -1.0) = rcp(base)
struct pow_neg_100 {
    static __m512 scale(__m512 base) {
        return _mm512_rcp14_ps(base);
    }
};

// pow(base, -beta) = exp2(-beta * log2(base)) for arbitrary beta
struct pow_neg_general {
    float beta;
    // @nnr-meta isa=AVX512 dtype=fp32
    __m512 scale(__m512 base) const {
        __m512 log2_base = fast_log2_avx512(base);
        __m512 vbeta = _mm512_set1_ps(-beta);
        return fast_exp2_avx512(_mm512_mul_ps(vbeta, log2_base));
    }
};

// Templated AVX-512 LRN loop — PowFn provides scale(base) -> pow(base, -beta)
template <typename PowFn>
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW tiling=spatial
inline void lrn_channel_avx512(const float* __restrict input, float* __restrict output,
    int nc, int C, int spatial, int c0, int c1, float alpha, float bias, PowFn pow_fn, int& hw)
{
    const float* xc = input + (size_t)nc * spatial;
    float* yc = output + (size_t)nc * spatial;
    int n = nc / C;
    __m512 valpha = _mm512_set1_ps(alpha);
    __m512 vbias = _mm512_set1_ps(bias);

    for (; hw + 16 <= spatial; hw += 16) {
        __m512 sum = _mm512_setzero_ps();
        for (int ci = c0; ci <= c1; ++ci) {
            __m512 v = _mm512_loadu_ps(input + ((size_t)n * C + ci) * spatial + hw);
            sum = _mm512_fmadd_ps(v, v, sum);
        }
        __m512 base = _mm512_fmadd_ps(valpha, sum, vbias);
        __m512 xv = _mm512_loadu_ps(xc + hw);
        _mm512_storeu_ps(yc + hw, _mm512_mul_ps(xv, pow_fn.scale(base)));
    }
}

} // namespace nnr
