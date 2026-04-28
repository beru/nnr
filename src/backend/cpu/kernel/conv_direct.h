#pragma once
// Direct NCHW convolution: pre-pad input + direct micro-kernel (no im2col).
// Dispatches to AVX-512 or falls back to false (caller uses im2col path).
#include "gemm.h"
#include <algorithm>
#include <cstring>

#ifdef NNR_ARCH_X64
#include "backend/x64/conv_direct_nchw_avx512.h"
#endif

namespace nnr {

// Pre-pad NCHW input: dst[C, padH, padW] with zero borders.
// src is [C, H, W] contiguous. padT/padB/padL/padR are top/bottom/left/right padding.
inline void prepad_nchw(float* __restrict dst, const float* __restrict src,
    int C, int H, int W, int padT, int padB, int padL, int padR)
{
    const int padH = H + padT + padB;
    const int padW = W + padL + padR;
    memset(dst, 0, (size_t)C * padH * padW * sizeof(float));
    for (int c = 0; c < C; c++) {
        float* dc = dst + (size_t)c * padH * padW + padT * padW + padL;
        const float* sc = src + (size_t)c * H * W;
        for (int h = 0; h < H; h++)
            memcpy(dc + h * padW, sc + h * W, W * sizeof(float));
    }
}

// Build offset table for direct conv K-loop.
// offsets[k] = ic * padH * padW + kh * padW + kw
// where k = ic * kH * kW + kh * kW + kw
inline void build_conv_offsets(int* __restrict offsets,
    int kC, int kH, int kW, int padH, int padW)
{
    const int kHW = kH * kW;
    const int chStride = padH * padW;
    int k = 0;
    for (int ic = 0; ic < kC; ic++) {
        const int ic_off = ic * chStride;
        for (int kh = 0; kh < kH; kh++) {
            const int row_off = ic_off + kh * padW;
            for (int kw = 0; kw < kW; kw++)
                offsets[k++] = row_off + kw;
        }
    }
}

// Direct NCHW conv: ISA dispatch.
// Returns true if handled, false to fall back to im2col path.
template <typename PostFn>
inline bool conv_direct(
    int M, int oH, int oW, int K,
    const float* __restrict packed_A,
    const float* __restrict padded,
    const int*   __restrict offsets,
    int padW, int sH,
    float* __restrict Y,
    const PostFn& post_fn)
{
#ifdef NNR_ARCH_X64
    if (!has_avx512()) {
        // AVX-512-only kernel; fall back to im2col path on AVX-2-only hosts.
        (void)M;(void)oH;(void)oW;(void)K;(void)packed_A;(void)padded;
        (void)offsets;(void)padW;(void)sH;(void)Y;(void)post_fn;
        return false;
    }
    return avx512::conv_direct_nchw(M, oH, oW, K, packed_A, padded, offsets,
        padW, sH, Y, post_fn);
#else
    (void)M;(void)oH;(void)oW;(void)K;(void)packed_A;(void)padded;
    (void)offsets;(void)padW;(void)sH;(void)Y;(void)post_fn;
    return false;
#endif
}

} // namespace nnr
