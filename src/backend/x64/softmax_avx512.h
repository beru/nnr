#pragma once
// AVX-512 softmax row kernel.
// Uses polynomial exp approximation from simd_math_avx512.h.

#ifdef NNR_ARCH_X64
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include "backend/x64/simd_math_avx512.h"

namespace nnr {

// Specialized exp for softmax: input is always <= 0 (x - max), so no upper
// clamp is needed. Uses vscalefps for 2^n (one instruction vs bit manip).
// Same degree-5 poly as exp512_ps.
// @nnr-meta isa=AVX512 dtype=fp32
static inline __m512 softmax_exp_avx512(__m512 x) {
    const __m512 log2e  = _mm512_set1_ps(1.44269504089f);
    const __m512 ln2_hi = _mm512_set1_ps(0.693145751953125f);
    const __m512 ln2_lo = _mm512_set1_ps(1.428606765330187e-06f);
    const __m512 one    = _mm512_set1_ps(1.0f);
    const __m512 c2 = _mm512_set1_ps(0.500000000f);
    const __m512 c3 = _mm512_set1_ps(0.166666672f);
    const __m512 c4 = _mm512_set1_ps(0.041666664f);
    const __m512 c5 = _mm512_set1_ps(0.008333345f);

    // Clamp lower (softmax inputs are <= 0, so no upper clamp needed).
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.0f));

    __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, log2e),
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, ln2_hi, x);
    r = _mm512_fnmadd_ps(n, ln2_lo, r);
    __m512 p = _mm512_fmadd_ps(c5, r, c4);
    p = _mm512_fmadd_ps(p, r, c3);
    p = _mm512_fmadd_ps(p, r, c2);
    p = _mm512_fmadd_ps(p, r, one);
    p = _mm512_fmadd_ps(p, r, one);
    return _mm512_scalef_ps(p, n);
}

// AVX-512 softmax for one row: find max, compute exp(x-max), normalize.
// Sum-exp pass unrolled 3x (48 elts/iter) for ILP — softmax is exp-bound and
// three independent exp chains hide the per-exp latency on Zen 4.
// @nnr-meta isa=AVX512 dtype=fp32
inline void softmax_row_avx512(const float* row, float* out, int len) {
    // 1. Find max
    __m512 vmax = _mm512_set1_ps(-1e30f);
    int j = 0;
    for (; j + 16 <= len; j += 16)
        vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(row + j));
    float maxv = _mm512_reduce_max_ps(vmax);
    for (; j < len; j++)
        maxv = std::max(maxv, row[j]);

    // 2. exp(x - max) and sum. 3x unroll → 48 elts/iter when possible.
    __m512 vsum0 = _mm512_setzero_ps();
    __m512 vsum1 = _mm512_setzero_ps();
    __m512 vsum2 = _mm512_setzero_ps();
    __m512 vmx = _mm512_set1_ps(maxv);
    j = 0;
    for (; j + 48 <= len; j += 48) {
        __m512 v0 = softmax_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(row + j),      vmx));
        __m512 v1 = softmax_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(row + j + 16), vmx));
        __m512 v2 = softmax_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(row + j + 32), vmx));
        _mm512_storeu_ps(out + j,      v0);
        _mm512_storeu_ps(out + j + 16, v1);
        _mm512_storeu_ps(out + j + 32, v2);
        vsum0 = _mm512_add_ps(vsum0, v0);
        vsum1 = _mm512_add_ps(vsum1, v1);
        vsum2 = _mm512_add_ps(vsum2, v2);
    }
    for (; j + 16 <= len; j += 16) {
        __m512 v = softmax_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(row + j), vmx));
        _mm512_storeu_ps(out + j, v);
        vsum0 = _mm512_add_ps(vsum0, v);
    }
    __m512 vsum = _mm512_add_ps(_mm512_add_ps(vsum0, vsum1), vsum2);
    float sum = _mm512_reduce_add_ps(vsum);
    for (; j < len; j++) {
        out[j] = expf(row[j] - maxv);
        sum += out[j];
    }

    // 3. Normalize
    if (sum != 0) {
        __m512 vinv = _mm512_set1_ps(1.0f / sum);
        j = 0;
        for (; j + 16 <= len; j += 16)
            _mm512_storeu_ps(out + j,
                _mm512_mul_ps(_mm512_loadu_ps(out + j), vinv));
        float inv = 1.0f / sum;
        for (; j < len; j++)
            out[j] *= inv;
    }
}

} // namespace nnr

#endif // NNR_ARCH_X64
