#pragma once
// AVX-2 softmax row kernel.
// Mirrors softmax_avx512.h; uses simd_math_avx2.h for exp.

#ifdef NNR_ARCH_X64
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include "backend/x64/simd_math_avx2.h"

namespace nnr {

// Specialized exp for softmax: input is always <= 0 (x - max), so no upper
// clamp is needed. Manual 2^n reconstruction (AVX-2 has no scalefps).
// Same degree-5 poly as exp256_ps.
// @nnr-meta isa=AVX2 dtype=fp32
static inline __m256 softmax_exp_avx2(__m256 x) {
    const __m256 log2e  = _mm256_set1_ps(1.44269504089f);
    const __m256 ln2_hi = _mm256_set1_ps(0.693145751953125f);
    const __m256 ln2_lo = _mm256_set1_ps(1.428606765330187e-06f);
    const __m256 one    = _mm256_set1_ps(1.0f);
    const __m256 c2 = _mm256_set1_ps(0.500000000f);
    const __m256 c3 = _mm256_set1_ps(0.166666672f);
    const __m256 c4 = _mm256_set1_ps(0.041666664f);
    const __m256 c5 = _mm256_set1_ps(0.008333345f);

    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));

    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, log2e),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, ln2_hi, x);
    r = _mm256_fnmadd_ps(n, ln2_lo, r);
    __m256 p = _mm256_fmadd_ps(c5, r, c4);
    p = _mm256_fmadd_ps(p, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, one);
    p = _mm256_fmadd_ps(p, r, one);

    // Reconstruct 2^n manually.
    __m256i n_int = _mm256_cvtps_epi32(n);
    __m256i bias  = _mm256_set1_epi32(127);
    __m256i e     = _mm256_slli_epi32(_mm256_add_epi32(n_int, bias), 23);
    __m256  pow2n = _mm256_castsi256_ps(e);
    return _mm256_mul_ps(p, pow2n);
}

// AVX-2 softmax for one row: find max, compute exp(x-max), normalize.
// Sum-exp pass unrolled 3x (24 elts/iter) for ILP.
// @nnr-meta isa=AVX2 dtype=fp32
inline void softmax_row_avx2(const float* row, float* out, int len) {
    // 1. Find max
    __m256 vmax = _mm256_set1_ps(-1e30f);
    int j = 0;
    for (; j + 8 <= len; j += 8)
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(row + j));
    // Horizontal max of 8 lanes.
    alignas(32) float buf[8];
    _mm256_store_ps(buf, vmax);
    float maxv = buf[0];
    for (int k = 1; k < 8; k++) maxv = std::max(maxv, buf[k]);
    for (; j < len; j++)
        maxv = std::max(maxv, row[j]);

    // 2. exp(x - max) and sum. 3x unroll → 24 elts/iter.
    __m256 vsum0 = _mm256_setzero_ps();
    __m256 vsum1 = _mm256_setzero_ps();
    __m256 vsum2 = _mm256_setzero_ps();
    __m256 vmx = _mm256_set1_ps(maxv);
    j = 0;
    for (; j + 24 <= len; j += 24) {
        __m256 v0 = softmax_exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(row + j),     vmx));
        __m256 v1 = softmax_exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(row + j + 8), vmx));
        __m256 v2 = softmax_exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(row + j + 16), vmx));
        _mm256_storeu_ps(out + j,      v0);
        _mm256_storeu_ps(out + j + 8,  v1);
        _mm256_storeu_ps(out + j + 16, v2);
        vsum0 = _mm256_add_ps(vsum0, v0);
        vsum1 = _mm256_add_ps(vsum1, v1);
        vsum2 = _mm256_add_ps(vsum2, v2);
    }
    for (; j + 8 <= len; j += 8) {
        __m256 v = softmax_exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(row + j), vmx));
        _mm256_storeu_ps(out + j, v);
        vsum0 = _mm256_add_ps(vsum0, v);
    }
    __m256 vsum = _mm256_add_ps(_mm256_add_ps(vsum0, vsum1), vsum2);
    _mm256_store_ps(buf, vsum);
    float sum = buf[0] + buf[1] + buf[2] + buf[3]
              + buf[4] + buf[5] + buf[6] + buf[7];
    for (; j < len; j++) {
        out[j] = expf(row[j] - maxv);
        sum += out[j];
    }

    // 3. Normalize
    if (sum != 0) {
        float inv = 1.0f / sum;
        __m256 vinv = _mm256_set1_ps(inv);
        j = 0;
        for (; j + 8 <= len; j += 8)
            _mm256_storeu_ps(out + j,
                _mm256_mul_ps(_mm256_loadu_ps(out + j), vinv));
        for (; j < len; j++)
            out[j] *= inv;
    }
}

} // namespace nnr

#endif // NNR_ARCH_X64
