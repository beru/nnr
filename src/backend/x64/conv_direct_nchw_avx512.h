#pragma once
// Direct NCHW convolution — AVX-512, non-depthwise.
// Eliminates im2col by loading input directly from pre-padded buffer via offset table.
// Reuses pack_a weights from gemm_avx512.h (same [ni][nk][JBLK*KC] layout).
//
// Micro-kernel: MR=8 output channels × NR=16 spatial positions.
// K-loop: load 16 input values at padded[offsets[k] + oh*padW + ow], broadcast 8 weights.
// Registers: 8 accumulators + 1 B load + 1 broadcast = 10 zmm (well within 32).

#include <immintrin.h>
#include <algorithm>
#include <cfloat>
#include <cstring>
#include "cpu_features.h"

namespace nnr { namespace avx512 {

// 8×16 micro-kernel: accumulate kc K-steps.
// padded_row = padded base shifted to current output row (padded + oh * sH * padW)
// offsets: per-K-step offset from padded_row; input = padded_row + offsets[k] + ow
NNR_NOINLINE
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=Direct tiling=[MR,NR]
static void direct_nchw_ukernel_8x16(
    int kc,
    const float* __restrict pa,
    const float* __restrict padded_row,
    const int*   __restrict offsets,
    int ow,
    float* __restrict pc0, float* __restrict pc1,
    float* __restrict pc2, float* __restrict pc3,
    float* __restrict pc4, float* __restrict pc5,
    float* __restrict pc6, float* __restrict pc7,
    bool zero_acc)
{
    __m512 c0, c1, c2, c3, c4, c5, c6, c7;
    if (zero_acc) {
        c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = _mm512_setzero_ps();
    } else {
        c0 = _mm512_loadu_ps(pc0); c1 = _mm512_loadu_ps(pc1);
        c2 = _mm512_loadu_ps(pc2); c3 = _mm512_loadu_ps(pc3);
        c4 = _mm512_loadu_ps(pc4); c5 = _mm512_loadu_ps(pc5);
        c6 = _mm512_loadu_ps(pc6); c7 = _mm512_loadu_ps(pc7);
    }

    // 2-way unrolled K-loop
    int k = 0;
    for (; k + 2 <= kc; k += 2) {
        __m512 b0 = _mm512_loadu_ps(padded_row + offsets[k] + ow);
        c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 0]), b0, c0);
        c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 1]), b0, c1);
        c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 2]), b0, c2);
        c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 3]), b0, c3);
        c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 4]), b0, c4);
        c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 5]), b0, c5);
        c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 6]), b0, c6);
        c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 7]), b0, c7);

        __m512 b1 = _mm512_loadu_ps(padded_row + offsets[k + 1] + ow);
        c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa[(k + 1) * 8 + 0]), b1, c0);
        c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa[(k + 1) * 8 + 1]), b1, c1);
        c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa[(k + 1) * 8 + 2]), b1, c2);
        c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa[(k + 1) * 8 + 3]), b1, c3);
        c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa[(k + 1) * 8 + 4]), b1, c4);
        c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa[(k + 1) * 8 + 5]), b1, c5);
        c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa[(k + 1) * 8 + 6]), b1, c6);
        c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa[(k + 1) * 8 + 7]), b1, c7);
    }
    for (; k < kc; k++) {
        __m512 b0 = _mm512_loadu_ps(padded_row + offsets[k] + ow);
        c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 0]), b0, c0);
        c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 1]), b0, c1);
        c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 2]), b0, c2);
        c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 3]), b0, c3);
        c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 4]), b0, c4);
        c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 5]), b0, c5);
        c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 6]), b0, c6);
        c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 7]), b0, c7);
    }

    _mm512_storeu_ps(pc0, c0); _mm512_storeu_ps(pc1, c1);
    _mm512_storeu_ps(pc2, c2); _mm512_storeu_ps(pc3, c3);
    _mm512_storeu_ps(pc4, c4); _mm512_storeu_ps(pc5, c5);
    _mm512_storeu_ps(pc6, c6); _mm512_storeu_ps(pc7, c7);
}

// Masked version for spatial remainder (< 16 elements)
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=Direct tiling=[MR,NR]
static void direct_nchw_ukernel_8x_masked(
    int kc, __mmask16 mask,
    const float* __restrict pa,
    const float* __restrict padded_row,
    const int*   __restrict offsets,
    int ow,
    float** __restrict pc,
    bool zero_acc)
{
    __m512 c0, c1, c2, c3, c4, c5, c6, c7;
    if (zero_acc) {
        c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = _mm512_setzero_ps();
    } else {
        c0 = _mm512_maskz_loadu_ps(mask, pc[0]); c1 = _mm512_maskz_loadu_ps(mask, pc[1]);
        c2 = _mm512_maskz_loadu_ps(mask, pc[2]); c3 = _mm512_maskz_loadu_ps(mask, pc[3]);
        c4 = _mm512_maskz_loadu_ps(mask, pc[4]); c5 = _mm512_maskz_loadu_ps(mask, pc[5]);
        c6 = _mm512_maskz_loadu_ps(mask, pc[6]); c7 = _mm512_maskz_loadu_ps(mask, pc[7]);
    }

    for (int k = 0; k < kc; k++) {
        __m512 b0 = _mm512_maskz_loadu_ps(mask, padded_row + offsets[k] + ow);
        c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 0]), b0, c0);
        c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 1]), b0, c1);
        c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 2]), b0, c2);
        c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 3]), b0, c3);
        c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 4]), b0, c4);
        c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 5]), b0, c5);
        c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 6]), b0, c6);
        c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa[k * 8 + 7]), b0, c7);
    }

    _mm512_mask_storeu_ps(pc[0], mask, c0); _mm512_mask_storeu_ps(pc[1], mask, c1);
    _mm512_mask_storeu_ps(pc[2], mask, c2); _mm512_mask_storeu_ps(pc[3], mask, c3);
    _mm512_mask_storeu_ps(pc[4], mask, c4); _mm512_mask_storeu_ps(pc[5], mask, c5);
    _mm512_mask_storeu_ps(pc[6], mask, c6); _mm512_mask_storeu_ps(pc[7], mask, c7);
}

// Tiled direct NCHW convolution: Y[M, oH, oW] = W × gather(padded_input)
//
// packed_A:  from pack_a(), format [M/JBLK][K/KC][JBLK*KC]
// padded:    pre-padded input [kC, padH, padW] with zero borders
// offsets:   precomputed [K] offsets: offsets[k] = ic*padH*padW + kh*padW + kw
// padW:      padded width (stride between rows in padded buffer)
// sH:        vertical stride (padded_row = padded + oh*sH*padW)
//
// Post-op: fused bias + relu/clip via PostFn (same interface as dgemm_packed_a).
template <typename PostFn>
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=Direct tiling=[K,MR,NR] fusion=post_op
inline bool conv_direct_nchw(
    int M, int oH, int oW,
    int K,
    const float* __restrict packed_A,
    const float* __restrict padded,
    const int*   __restrict offsets,
    int padW,
    int sH,
    float* __restrict Y,
    const PostFn& post_fn)
{
    if (!has_avx512()) return false;

    constexpr int JBLK = 64;
    constexpr int KC = 256;
    constexpr bool can_fuse = PostFn::per_row_bias;

    const int spatial = oH * oW;
    const int nk = (K + KC - 1) / KC;
    const int ni = (M + JBLK - 1) / JBLK;

    // Tiling: K-blocks OUTER, spatial INNER.
    // This ensures packed_a is loaded once per KC-block and reused across all spatial positions.
    // Accumulators live in the output buffer between KC-blocks (load/store overhead is small
    // because the output for 8 rows × spatial fits in L1/L2).
    for (int ib = 0; ib < ni; ib++) {
        const int i0 = ib * JBLK;
        const int ie = std::min(i0 + JBLK, M);
        const int iw = ie - i0;
        const int nfull = iw / 8;
        const int nrem = iw % 8;

        // --- 8-row groups: K-outer, spatial-inner ---
        for (int grp = 0; grp < nfull; grp++) {
            const int m_base = i0 + grp * 8;

            for (int kt = 0; kt < nk; kt++) {
                const int k0 = kt * KC;
                const int kc = std::min(KC, K - k0);
                const float* pa = packed_A + ((size_t)ib * nk + kt) * JBLK * KC
                                + (size_t)grp * 8 * KC;
                const bool zero = (kt == 0);

                for (int oh = 0; oh < oH; oh++) {
                    const float* padded_row = padded + oh * sH * padW;

                    int ow = 0;
                    for (; ow + 16 <= oW; ow += 16) {
                        float* pc[8];
                        for (int r = 0; r < 8; r++)
                            pc[r] = Y + (size_t)(m_base + r) * spatial + oh * oW + ow;
                        direct_nchw_ukernel_8x16(kc, pa, padded_row, offsets + k0, ow,
                            pc[0], pc[1], pc[2], pc[3], pc[4], pc[5], pc[6], pc[7], zero);
                    }

                    if (ow < oW) {
                        const __mmask16 mask = (__mmask16)((1u << (oW - ow)) - 1);
                        float* pc[8];
                        for (int r = 0; r < 8; r++)
                            pc[r] = Y + (size_t)(m_base + r) * spatial + oh * oW + ow;
                        direct_nchw_ukernel_8x_masked(kc, mask, pa, padded_row,
                            offsets + k0, ow, pc, zero);
                    }
                }
            }

            // Post-op: bias + activation (after all K-blocks)
            if constexpr (can_fuse) {
                if (post_fn.kind != post_op_kind::none || post_fn.bias) {
                    const float* bp = post_fn.bias ? post_fn.bias + post_fn.bias_off + m_base
                                                   : fused_zero_bias;
                    float fmin = post_fn.clip_min;
                    float fmax = post_fn.clip_max;

                    for (int oh = 0; oh < oH; oh++) {
                        int ow = 0;
                        for (; ow + 16 <= oW; ow += 16) {
                            float* pc[8];
                            for (int r = 0; r < 8; r++)
                                pc[r] = Y + (size_t)(m_base + r) * spatial + oh * oW + ow;
                            __m512 c0 = _mm512_loadu_ps(pc[0]), c1 = _mm512_loadu_ps(pc[1]);
                            __m512 c2 = _mm512_loadu_ps(pc[2]), c3 = _mm512_loadu_ps(pc[3]);
                            __m512 c4 = _mm512_loadu_ps(pc[4]), c5 = _mm512_loadu_ps(pc[5]);
                            __m512 c6 = _mm512_loadu_ps(pc[6]), c7 = _mm512_loadu_ps(pc[7]);
                            fuse_nchw_8(c0, c1, c2, c3, c4, c5, c6, c7, bp, fmin, fmax);
                            _mm512_storeu_ps(pc[0], c0); _mm512_storeu_ps(pc[1], c1);
                            _mm512_storeu_ps(pc[2], c2); _mm512_storeu_ps(pc[3], c3);
                            _mm512_storeu_ps(pc[4], c4); _mm512_storeu_ps(pc[5], c5);
                            _mm512_storeu_ps(pc[6], c6); _mm512_storeu_ps(pc[7], c7);
                        }
                        if (ow < oW) {
                            const __mmask16 mask = (__mmask16)((1u << (oW - ow)) - 1);
                            for (int r = 0; r < 8; r++) {
                                float* pr = Y + (size_t)(m_base + r) * spatial + oh * oW + ow;
                                __m512 c = _mm512_maskz_loadu_ps(mask, pr);
                                fuse_nchw_1(c, bp[r], fmin, fmax);
                                _mm512_mask_storeu_ps(pr, mask, c);
                            }
                        }
                    }
                }
            }
        }

        // --- Remainder rows (< 8): 1-row, K-outer ---
        for (int r = 0; r < nrem; r++) {
            const int m = i0 + nfull * 8 + r;

            for (int kt = 0; kt < nk; kt++) {
                const int k0 = kt * KC;
                const int kc = std::min(KC, K - k0);
                const float* pa = packed_A + ((size_t)ib * nk + kt) * JBLK * KC
                                + (size_t)nfull * 8 * KC + (size_t)r * KC;
                const bool zero = (kt == 0);

                for (int oh = 0; oh < oH; oh++) {
                    const float* padded_row = padded + oh * sH * padW;

                    int ow = 0;
                    for (; ow + 16 <= oW; ow += 16) {
                        float* pc_r = Y + (size_t)m * spatial + oh * oW + ow;
                        __m512 acc = zero ? _mm512_setzero_ps() : _mm512_loadu_ps(pc_r);
                        for (int k = 0; k < kc; k++) {
                            __m512 b = _mm512_loadu_ps(padded_row + offsets[k0 + k] + ow);
                            acc = _mm512_fmadd_ps(_mm512_set1_ps(pa[k]), b, acc);
                        }
                        _mm512_storeu_ps(pc_r, acc);
                    }
                    if (ow < oW) {
                        const int nr = oW - ow;
                        const __mmask16 mask = (__mmask16)((1u << nr) - 1);
                        float* pc_r = Y + (size_t)m * spatial + oh * oW + ow;
                        __m512 acc = zero ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(mask, pc_r);
                        for (int k = 0; k < kc; k++) {
                            __m512 b = _mm512_maskz_loadu_ps(mask, padded_row + offsets[k0 + k] + ow);
                            acc = _mm512_fmadd_ps(_mm512_set1_ps(pa[k]), b, acc);
                        }
                        _mm512_mask_storeu_ps(pc_r, mask, acc);
                    }
                }
            }

            // Post-op for remainder row
            if constexpr (can_fuse) {
                if (post_fn.kind != post_op_kind::none || post_fn.bias) {
                    float bv = post_fn.bias ? post_fn.bias[post_fn.bias_off + m] : 0.0f;
                    float fmin = post_fn.clip_min, fmax = post_fn.clip_max;
                    for (int oh = 0; oh < oH; oh++) {
                        int ow = 0;
                        for (; ow + 16 <= oW; ow += 16) {
                            float* p = Y + (size_t)m * spatial + oh * oW + ow;
                            __m512 c = _mm512_loadu_ps(p);
                            fuse_nchw_1(c, bv, fmin, fmax);
                            _mm512_storeu_ps(p, c);
                        }
                        if (ow < oW) {
                            const __mmask16 mask = (__mmask16)((1u << (oW - ow)) - 1);
                            float* p = Y + (size_t)m * spatial + oh * oW + ow;
                            __m512 c = _mm512_maskz_loadu_ps(mask, p);
                            fuse_nchw_1(c, bv, fmin, fmax);
                            _mm512_mask_storeu_ps(p, mask, c);
                        }
                    }
                }
            }
        }

        // General post_fn for non-fused cases
        if constexpr (can_fuse) {
            if (post_fn.post_fn && post_fn.kind == post_op_kind::none) {
                post_fn.post_fn(Y + (size_t)i0 * spatial, iw,
                    spatial, spatial, post_fn.fused_op,
                    post_fn.bias ? post_fn.bias + post_fn.bias_off + i0 : nullptr,
                    post_fn.c_base_offset + (int)((size_t)i0 * spatial));
            }
        }
    }

    return true;
}

}} // namespace nnr::avx512
