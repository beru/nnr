#pragma once
// AVX-512 vectorized math functions: exp, sigmoid, SiLU (Swish).
// Fast polynomial approximation — sufficient for neural network inference.

#include "cpu_features.h"
#ifdef NNR_ARCH_X64
#include <immintrin.h>

namespace nnr {

// Fast exp(x) for 16 floats using AVX-512.
// Range reduction to [-ln2/2, ln2/2] + degree-5 minimax polynomial.
// Uses vscalefps for the 2^n reconstruction (one instruction vs the manual
// cvt+add+slli+cast+mul chain). Max relative error ~1.5e-7.
// @nnr-meta isa=AVX512 dtype=fp32
static inline __m512 exp512_ps(__m512 x) {
    const __m512 log2e  = _mm512_set1_ps(1.44269504089f);
    const __m512 ln2_hi = _mm512_set1_ps(0.693145751953125f);
    const __m512 ln2_lo = _mm512_set1_ps(1.428606765330187e-06f);
    const __m512 one    = _mm512_set1_ps(1.0f);

    // Clamp to prevent overflow/underflow in float exponent
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.0f));
    x = _mm512_min_ps(x, _mm512_set1_ps(88.0f));

    // Range reduction: exp(x) = 2^n * exp(r)
    // n = round(x * log2(e)), r = x - n * ln(2)
    __m512 t = _mm512_mul_ps(x, log2e);
    __m512 n = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, ln2_hi, x);   // r = x - n * ln2_hi
    r = _mm512_fnmadd_ps(n, ln2_lo, r);           // r -= n * ln2_lo (precision)

    // Polynomial approximation: exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120
    const __m512 c2 = _mm512_set1_ps(0.500000000f);
    const __m512 c3 = _mm512_set1_ps(0.166666672f);
    const __m512 c4 = _mm512_set1_ps(0.041666664f);
    const __m512 c5 = _mm512_set1_ps(0.008333345f);

    __m512 p = _mm512_fmadd_ps(c5, r, c4);
    p = _mm512_fmadd_ps(p, r, c3);
    p = _mm512_fmadd_ps(p, r, c2);
    p = _mm512_fmadd_ps(p, r, one);
    p = _mm512_fmadd_ps(p, r, one);

    // Reconstruct: p * 2^n in one instruction (AVX-512F).
    return _mm512_scalef_ps(p, n);
}

// sigmoid(x) = 1 / (1 + exp(-x))
// Uses rcp14 + Newton-Raphson instead of vdivps (14→28-bit precision, ~10 cycles faster).
// @nnr-meta isa=AVX512 dtype=fp32
static inline __m512 sigmoid512_ps(__m512 x) {
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 neg_x = _mm512_xor_ps(x, _mm512_set1_ps(-0.0f)); // negate via sign flip
    __m512 exp_neg = exp512_ps(neg_x);
    __m512 denom = _mm512_add_ps(one, exp_neg);
    // rcp14 (14-bit) + 1 Newton step → ~28-bit precision (sufficient for float32)
    __m512 rcp = _mm512_rcp14_ps(denom);
    // Newton: rcp' = rcp * (2 - denom * rcp)
    rcp = _mm512_mul_ps(rcp, _mm512_fnmadd_ps(denom, rcp, _mm512_set1_ps(2.0f)));
    return rcp;
}

// SiLU(x) = x * sigmoid(x)
// @nnr-meta isa=AVX512 dtype=fp32
static inline __m512 silu512_ps(__m512 x) {
    return _mm512_mul_ps(x, sigmoid512_ps(x));
}

// Apply sigmoid to a contiguous float array (threaded, AVX-512).
// @nnr-meta isa=AVX512 dtype=fp32
void sigmoid_avx512(float* data, size_t n);

// Apply SiLU (x * sigmoid(x)) in-place (threaded, AVX-512).
// Reads from src, writes to dst. src == dst for in-place.
// @nnr-meta isa=AVX512 dtype=fp32
void silu_avx512(const float* src, float* dst, size_t n);

// Apply element-wise multiply: dst[i] = a[i] * b[i] (threaded, AVX-512).
// @nnr-meta isa=AVX512 dtype=fp32
void mul_avx512(const float* a, const float* b, float* dst, size_t n);

// Apply element-wise sub: dst[i] = a[i] - b[i] (threaded, AVX-512).
// @nnr-meta isa=AVX512 dtype=fp32
void sub_avx512(const float* a, const float* b, float* dst, size_t n);

// Apply element-wise div: dst[i] = a[i] / b[i] (threaded, AVX-512).
// @nnr-meta isa=AVX512 dtype=fp32
void div_avx512(const float* a, const float* b, float* dst, size_t n);

// Apply GELU (exact, erf-based): dst[i] = 0.5 * x * (1 + erf(x / sqrt(2))) (threaded, AVX-512).
// @nnr-meta isa=AVX512 dtype=fp32
void gelu_avx512(const float* src, float* dst, size_t n);

// Fast erf(x) for 16 floats using AVX-512.
// Abramowitz & Stegun polynomial approximation (max error ~1.5e-7).
// @nnr-meta isa=AVX512 dtype=fp32
static inline __m512 erf512_ps(__m512 x) {
    // erf(x) = sign(x) * (1 - t * exp(-x² + polynomial(t)))
    // where t = 1/(1 + 0.3275911 * |x|)
    const __m512 sign_mask = _mm512_set1_ps(-0.0f);
    __m512 sign = _mm512_and_ps(x, sign_mask);
    __m512 ax = _mm512_andnot_ps(sign_mask, x);  // |x|

    const __m512 p  = _mm512_set1_ps(0.3275911f);
    const __m512 a1 = _mm512_set1_ps(0.254829592f);
    const __m512 a2 = _mm512_set1_ps(-0.284496736f);
    const __m512 a3 = _mm512_set1_ps(1.421413741f);
    const __m512 a4 = _mm512_set1_ps(-1.453152027f);
    const __m512 a5 = _mm512_set1_ps(1.061405429f);
    const __m512 one = _mm512_set1_ps(1.0f);

    __m512 t = _mm512_rcp14_ps(_mm512_fmadd_ps(p, ax, one));
    // Newton step for 1/(1+p*|x|)
    __m512 denom = _mm512_fmadd_ps(p, ax, one);
    t = _mm512_mul_ps(t, _mm512_fnmadd_ps(denom, t, _mm512_set1_ps(2.0f)));

    // Horner: poly = ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
    __m512 poly = _mm512_fmadd_ps(a5, t, a4);
    poly = _mm512_fmadd_ps(poly, t, a3);
    poly = _mm512_fmadd_ps(poly, t, a2);
    poly = _mm512_fmadd_ps(poly, t, a1);
    poly = _mm512_mul_ps(poly, t);

    // exp(-x²)
    __m512 neg_x2 = _mm512_xor_ps(_mm512_mul_ps(ax, ax), sign_mask);
    __m512 ex = exp512_ps(neg_x2);

    // erf = sign * (1 - poly * exp(-x²))
    __m512 result = _mm512_fnmadd_ps(poly, ex, one);
    return _mm512_xor_ps(result, sign);
}

} // namespace nnr
#endif // NNR_ARCH_X64
