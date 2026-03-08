#pragma once
// x64 AVX-512/AVX2 helpers for conv kernels (im2col, bias, etc.)
// Called from kernel/conv.h via #ifdef NNR_ARCH_X64 dispatch.

#ifdef NNR_ARCH_X64
#include <immintrin.h>
#include "cpu_features.h"

namespace nnr {

// Stride-2 im2col row: gather even-indexed floats via permutex2var.
// Returns the output-width position after vectorized processing.
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=IM2COL
inline int im2col_stride2_avx512(float* __restrict drow, const float* __restrict src,
    int ow, int w0, int w1, int iW, int iw_base)
{
    static const int even_idx[16] = {0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30};
    __m512i vidx = _mm512_loadu_si512(even_idx);
    int w1_vec = std::min(w1, w0 + (iW - iw_base - w0 * 2 - 30) / 2);
    for (; ow + 16 <= w1_vec; ow += 16) {
        __m512 lo = _mm512_loadu_ps(src + ow * 2);
        __m512 hi = _mm512_loadu_ps(src + ow * 2 + 16);
        _mm512_storeu_ps(drow + ow, _mm512_permutex2var_ps(lo, vidx, hi));
    }
    return ow;
}

// Vectorized NHWC bias add: Y[spatial × M] += bias[M]
// @nnr-meta isa=[AVX512,AVX2] dtype=fp32 layout=NHWC fusion=post_op
inline void nhwc_bias_add_x64(float* Y, const float* bias, int spatial, int M) {
    if (has_avx512()) {
        for (int s = 0; s < spatial; s++) {
            float* row = Y + s * M;
            int m = 0;
            for (; m + 16 <= M; m += 16)
                _mm512_storeu_ps(row + m, _mm512_add_ps(
                    _mm512_loadu_ps(row + m), _mm512_loadu_ps(bias + m)));
            for (; m < M; m++)
                row[m] += bias[m];
        }
    } else if (detect_isa() == isa_t::avx2) {
        for (int s = 0; s < spatial; s++) {
            float* row = Y + s * M;
            int m = 0;
            for (; m + 8 <= M; m += 8)
                _mm256_storeu_ps(row + m, _mm256_add_ps(
                    _mm256_loadu_ps(row + m), _mm256_loadu_ps(bias + m)));
            for (; m < M; m++)
                row[m] += bias[m];
        }
    } else {
        for (int s = 0; s < spatial; s++)
            for (int m = 0; m < M; m++)
                Y[s * M + m] += bias[m];
    }
}

// Column-wise bias add: data[j] += bias[j] for j in [0, len).
// Used by NHWC GEMM post-ops where bias is per-column.
// @nnr-meta isa=[AVX512,AVX2] dtype=fp32 layout=NHWC fusion=post_op
inline int col_bias_add_x64(float* data, const float* bias, int len) {
    int j = 0;
    if (has_avx512()) {
        for (; j + 16 <= len; j += 16)
            _mm512_storeu_ps(data + j, _mm512_add_ps(
                _mm512_loadu_ps(data + j), _mm512_loadu_ps(bias + j)));
    } else if (detect_isa() == isa_t::avx2) {
        for (; j + 8 <= len; j += 8)
            _mm256_storeu_ps(data + j, _mm256_add_ps(
                _mm256_loadu_ps(data + j), _mm256_loadu_ps(bias + j)));
    }
    return j;
}

} // namespace nnr

#endif // NNR_ARCH_X64
