#pragma once
// AVX-2 vectorized math functions: exp, sigmoid, SiLU (Swish), tanh, erf, gelu.
// Mirrors simd_math_avx512.h for AVX-2-only x64 hardware (Zen2/Skylake/Haswell etc.).
// Polynomial accuracy matches the AVX-512 versions; only vector width differs (8 vs 16).

#include "cpu_features.h"
#ifdef NNR_ARCH_X64
#include <immintrin.h>

namespace nnr {

// Fast exp(x) for 8 floats using AVX-2 + FMA.
// Range reduction to [-ln2/2, ln2/2] + degree-5 minimax polynomial.
// Manual 2^n reconstruction (AVX-2 has no vscalefps). Max relative error ~1.5e-7.
// @nnr-meta isa=AVX2 dtype=fp32
static inline __m256 exp256_ps(__m256 x) {
    const __m256 log2e  = _mm256_set1_ps(1.44269504089f);
    const __m256 ln2_hi = _mm256_set1_ps(0.693145751953125f);
    const __m256 ln2_lo = _mm256_set1_ps(1.428606765330187e-06f);
    const __m256 one    = _mm256_set1_ps(1.0f);

    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));

    // Round-to-nearest k = round(x * log2e)
    __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // r = x - k * ln2 (high+low for accuracy)
    __m256 r = _mm256_fnmadd_ps(fx, ln2_hi, x);
    r = _mm256_fnmadd_ps(fx, ln2_lo, r);

    // Degree-5 Horner polynomial for exp(r) on [-ln2/2, ln2/2]
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, one));

    // Reconstruct 2^k by manipulating IEEE-754 exponent: 2^k = (k + 127) << 23
    __m256i k_int = _mm256_cvtps_epi32(fx);
    __m256i bias  = _mm256_set1_epi32(127);
    __m256i e     = _mm256_slli_epi32(_mm256_add_epi32(k_int, bias), 23);
    __m256  pow2k = _mm256_castsi256_ps(e);

    return _mm256_mul_ps(p, pow2k);
}

// sigmoid(x) = 1 / (1 + exp(-x))
// @nnr-meta isa=AVX2 dtype=fp32
static inline __m256 sigmoid256_ps(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 e = exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), x));
    return _mm256_div_ps(one, _mm256_add_ps(one, e));
}

// silu(x) = x * sigmoid(x)
// @nnr-meta isa=AVX2 dtype=fp32
static inline __m256 silu256_ps(__m256 x) {
    return _mm256_mul_ps(x, sigmoid256_ps(x));
}

// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1), with clamping to avoid overflow.
// @nnr-meta isa=AVX2 dtype=fp32
static inline __m256 tanh256_ps(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 lim = _mm256_set1_ps(8.0f);
    __m256 ax = _mm256_min_ps(_mm256_max_ps(x, _mm256_sub_ps(_mm256_setzero_ps(), lim)), lim);
    __m256 e2 = exp256_ps(_mm256_add_ps(ax, ax));
    return _mm256_div_ps(_mm256_sub_ps(e2, one), _mm256_add_ps(e2, one));
}

// Single-threaded inline kernels (mirror sigmoid_avx512_kernel et al.).
// Used from post_fn fusion where outer for_static is already active.
static inline void sigmoid_avx2_kernel(float* data, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        _mm256_storeu_ps(data + i, sigmoid256_ps(v));
    }
    for (; i < n; i++)
        data[i] = 1.0f / (1.0f + expf(-data[i]));
}

static inline void silu_avx2_kernel(const float* src, float* dst, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, silu256_ps(v));
    }
    for (; i < n; i++)
        dst[i] = src[i] / (1.0f + expf(-src[i]));
}

static inline void tanh_avx2_kernel(const float* src, float* dst, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, tanh256_ps(v));
    }
    for (; i < n; i++)
        dst[i] = tanhf(src[i]);
}

// Threaded entry points: same names as AVX-512 versions but with avx2 suffix.
// @nnr-meta isa=AVX2 dtype=fp32
void sigmoid_avx2(float* data, size_t n);
// @nnr-meta isa=AVX2 dtype=fp32
void silu_avx2(const float* src, float* dst, size_t n);
// @nnr-meta isa=AVX2 dtype=fp32
void mul_avx2(const float* a, const float* b, float* dst, size_t n);
// @nnr-meta isa=AVX2 dtype=fp32
void sub_avx2(const float* a, const float* b, float* dst, size_t n);
// @nnr-meta isa=AVX2 dtype=fp32
void div_avx2(const float* a, const float* b, float* dst, size_t n);
// @nnr-meta isa=AVX2 dtype=fp32
void gelu_avx2(const float* src, float* dst, size_t n);

// Broadcast binary ops with general numpy-style broadcast (mirrors AVX-512).
// @nnr-meta isa=AVX2 dtype=fp32
void mul_broadcast_avx2(const float* a, const float* b, float* y,
    const int* dims, const int* a_bstr, const int* b_bstr, int ndim);
void add_broadcast_avx2(const float* a, const float* b, float* y,
    const int* dims, const int* a_bstr, const int* b_bstr, int ndim);
void sub_broadcast_avx2(const float* a, const float* b, float* y,
    const int* dims, const int* a_bstr, const int* b_bstr, int ndim);
void div_broadcast_avx2(const float* a, const float* b, float* y,
    const int* dims, const int* a_bstr, const int* b_bstr, int ndim);

// Fast erf(x) for 8 floats using AVX-2 + FMA.
// Abramowitz & Stegun polynomial approximation (max error ~1.5e-7).
// @nnr-meta isa=AVX2 dtype=fp32
static inline __m256 erf256_ps(__m256 x) {
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 sign = _mm256_and_ps(x, sign_mask);
    __m256 ax = _mm256_andnot_ps(sign_mask, x);

    const __m256 p  = _mm256_set1_ps(0.3275911f);
    const __m256 a1 = _mm256_set1_ps(0.254829592f);
    const __m256 a2 = _mm256_set1_ps(-0.284496736f);
    const __m256 a3 = _mm256_set1_ps(1.421413741f);
    const __m256 a4 = _mm256_set1_ps(-1.453152027f);
    const __m256 a5 = _mm256_set1_ps(1.061405429f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);

    __m256 t = _mm256_rcp_ps(_mm256_fmadd_ps(p, ax, one));
    // Newton step (rcp_ps is ~12-bit only): t = t * (2 - denom*t)
    __m256 denom = _mm256_fmadd_ps(p, ax, one);
    t = _mm256_mul_ps(t, _mm256_fnmadd_ps(denom, t, two));

    __m256 poly = _mm256_fmadd_ps(a5, t, a4);
    poly = _mm256_fmadd_ps(poly, t, a3);
    poly = _mm256_fmadd_ps(poly, t, a2);
    poly = _mm256_fmadd_ps(poly, t, a1);
    poly = _mm256_mul_ps(poly, t);

    __m256 neg_x2 = _mm256_xor_ps(_mm256_mul_ps(ax, ax), sign_mask);
    __m256 ex = exp256_ps(neg_x2);

    __m256 result = _mm256_fnmadd_ps(poly, ex, one);
    return _mm256_xor_ps(result, sign);
}

} // namespace nnr
#endif // NNR_ARCH_X64
