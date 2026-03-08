#pragma once
// Direct convolution for first-layer Conv (small input channels, e.g., RGB).
// Skips im2col entirely. Vectorizes over output channels (16 per ZMM register).
//
// Loop order: outer=output pixels (tiled), inner=kernel (IC×KH×KW).
// Accumulators stay in registers across all 147 kernel positions (3×7×7).
// 14 ZMM accumulators + 1 weight vector = 15 registers.
//
// Weights: repacked as [IC][KH][KW][OC/16][16] (37.6KB for 64×3×7×7 — fits L1).

#ifdef NNR_ARCH_X64

#include <immintrin.h>
#include <algorithm>
#include <cstring>
#include <cfloat>
#include "thread_pool.h"
#include "backend/x64/layout_x64.h"  // transpose_16x16_avx512

namespace nnr {

// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=FirstLayer fusion=post_op
inline bool conv_first_layer_avx512(
    float* __restrict y,
    const float* __restrict x,
    const float* __restrict w,    // repacked [IC, KH, KW, OC_blocks, 16]
    const float* __restrict bias,
    int IC, int iH, int iW,
    int OC, int oH, int oW,
    int KH, int KW,
    int sH, int sW,
    int padH, int padW,
    operator_t::post_fn_t post_fn,
    const operator_t* fused_op)
{
    if (!has_avx512()) return false;
    const int OC_blocks = (OC + 15) / 16;
    const int kSpatial = KH * KW;
    const int spatial = oH * oW;
    constexpr int WT = 14;  // output width tile

    // Pre-padded input: copy x into a padded buffer with zero borders so the
    // FMA inner loop is fully branchless (no ih/iw bounds checks).
    // Layout: [IC][pH][pW]. Lives at the front of NNR_POOL_SCRATCH(0); all
    // worker threads read from it. Filled once on the main thread before dispatch.
    const int pH = iH + 2 * padH;
    const int pW = iW + 2 * padW;
    const size_t prepad_elems = (size_t)IC * pH * pW;
    const size_t row_elems = (size_t)oW * OC_blocks * 16;
    NNR_POOL_ENSURE_SCRATCH((prepad_elems + row_elems) * sizeof(float));

    {
        float* prepad_w = (float*)NNR_POOL_SCRATCH(0);
        memset(prepad_w, 0, prepad_elems * sizeof(float));
        for (int ic = 0; ic < IC; ic++) {
            for (int ih = 0; ih < iH; ih++) {
                memcpy(
                    prepad_w + ((size_t)ic * pH + (ih + padH)) * pW + padW,
                    x        + ((size_t)ic * iH +  ih)         * iW,
                    (size_t)iW * sizeof(float));
            }
        }
    }
    const float* prepad = (const float*)NNR_POOL_SCRATCH(0);

    nnr::for_dynamic(0, oH, oH >= 4, [&](int tid, int oh) {
        // Row buf layout: [ob][ow][16] — per-ob slab is oW*16 floats contiguous.
        // (Previously [ow][ob*16+lane] which had 128B stride per ob lane and
        // forced a slow scalar gather in the final NCHW transpose.)
        float* buf = (float*)NNR_POOL_SCRATCH(tid) + prepad_elems;

        // For each OC block (16 output channels at a time)
        for (int ob = 0; ob < OC_blocks; ob++) {
            float* out_ob = buf + (size_t)ob * oW * 16;

            __m512 vbias;
            if (bias) {
                int oc0 = ob * 16;
                if (oc0 + 16 <= OC) {
                    vbias = _mm512_loadu_ps(bias + oc0);
                } else {
                    float tmp[16] = {};
                    for (int i = 0; i < OC - oc0; i++) tmp[i] = bias[oc0 + i];
                    vbias = _mm512_loadu_ps(tmp);
                }
            } else {
                vbias = _mm512_setzero_ps();
            }
            // No bias-init store loop — we initialize accumulators directly
            // from vbias inside the tile loop and store the final result.

            // Main loop: tile over oW, inner loop over IC×KH×KW
            int ow = 0;
            for (; ow + WT <= oW; ow += WT) {
                __m512 a0  = vbias;
                __m512 a1  = vbias;
                __m512 a2  = vbias;
                __m512 a3  = vbias;
                __m512 a4  = vbias;
                __m512 a5  = vbias;
                __m512 a6  = vbias;
                __m512 a7  = vbias;
                __m512 a8  = vbias;
                __m512 a9  = vbias;
                __m512 a10 = vbias;
                __m512 a11 = vbias;
                __m512 a12 = vbias;
                __m512 a13 = vbias;

                // Inner loop: all kernel positions. Branchless — prepad has
                // zero borders baked in, so ih_p ∈ [0,pH) and iw_p ∈ [0,pW).
                for (int ic = 0; ic < IC; ic++) {
                    const float* p_ic = prepad + (size_t)ic * pH * pW;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih_p = oh * sH + kh;
                        const float* x_row = p_ic + (size_t)ih_p * pW;
                        for (int kw = 0; kw < KW; kw++) {
                            __m512 wv = _mm512_loadu_ps(
                                w + (((size_t)ic * kSpatial + kh * KW + kw) * OC_blocks + ob) * 16);

                            #define FMA_T(T) \
                                a##T = _mm512_fmadd_ps( \
                                    _mm512_set1_ps(x_row[(ow + T) * sW + kw]), wv, a##T);
                            FMA_T(0)  FMA_T(1)  FMA_T(2)  FMA_T(3)
                            FMA_T(4)  FMA_T(5)  FMA_T(6)  FMA_T(7)
                            FMA_T(8)  FMA_T(9)  FMA_T(10) FMA_T(11)
                            FMA_T(12) FMA_T(13)
                            #undef FMA_T
                        }
                    }
                }

                _mm512_storeu_ps(out_ob + (size_t)(ow+0)  * 16, a0);
                _mm512_storeu_ps(out_ob + (size_t)(ow+1)  * 16, a1);
                _mm512_storeu_ps(out_ob + (size_t)(ow+2)  * 16, a2);
                _mm512_storeu_ps(out_ob + (size_t)(ow+3)  * 16, a3);
                _mm512_storeu_ps(out_ob + (size_t)(ow+4)  * 16, a4);
                _mm512_storeu_ps(out_ob + (size_t)(ow+5)  * 16, a5);
                _mm512_storeu_ps(out_ob + (size_t)(ow+6)  * 16, a6);
                _mm512_storeu_ps(out_ob + (size_t)(ow+7)  * 16, a7);
                _mm512_storeu_ps(out_ob + (size_t)(ow+8)  * 16, a8);
                _mm512_storeu_ps(out_ob + (size_t)(ow+9)  * 16, a9);
                _mm512_storeu_ps(out_ob + (size_t)(ow+10) * 16, a10);
                _mm512_storeu_ps(out_ob + (size_t)(ow+11) * 16, a11);
                _mm512_storeu_ps(out_ob + (size_t)(ow+12) * 16, a12);
                _mm512_storeu_ps(out_ob + (size_t)(ow+13) * 16, a13);
            }

            // Remainder pixels (< WT) — init each accumulator from vbias.
            for (; ow < oW; ow++) {
                __m512 a = vbias;
                for (int ic = 0; ic < IC; ic++) {
                    const float* p_ic = prepad + (size_t)ic * pH * pW;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih_p = oh * sH + kh;
                        const float* x_row = p_ic + (size_t)ih_p * pW;
                        for (int kw = 0; kw < KW; kw++) {
                            __m512 wv = _mm512_loadu_ps(
                                w + (((size_t)ic * kSpatial + kh * KW + kw) * OC_blocks + ob) * 16);
                            a = _mm512_fmadd_ps(_mm512_set1_ps(x_row[ow * sW + kw]), wv, a);
                        }
                    }
                }
                _mm512_storeu_ps(out_ob + (size_t)ow * 16, a);
            }
        }

        // Final NCHW transpose: per ob, [oW × 16] → 16 OC planes via
        // transpose_16x16_avx512 over each 16-wide ow slab. Scalar tail for
        // the ow remainder and for partial-OC last blocks (rare — none of
        // our target stems hit OC % 16 != 0).
        for (int ob = 0; ob < OC_blocks; ob++) {
            const float* src_ob = buf + (size_t)ob * oW * 16;
            float* dst_ob = y + (size_t)ob * 16 * spatial + oh * oW;
            int oc_in_block = std::min(16, OC - ob * 16);
            if (oc_in_block == 16) {
                int ow2 = 0;
                for (; ow2 + 16 <= oW; ow2 += 16) {
                    nnr::transpose_16x16_avx512(
                        src_ob + (size_t)ow2 * 16, /*src_stride=*/16,
                        dst_ob + ow2,              /*dst_stride=*/spatial);
                }
                for (; ow2 < oW; ow2++) {
                    for (int c = 0; c < 16; c++)
                        dst_ob[(size_t)c * spatial + ow2] = src_ob[(size_t)ow2 * 16 + c];
                }
            } else {
                for (int ow2 = 0; ow2 < oW; ow2++)
                    for (int c = 0; c < oc_in_block; c++)
                        dst_ob[(size_t)c * spatial + ow2] = src_ob[(size_t)ow2 * 16 + c];
            }
        }
        if (post_fn) {
            for (int oc = 0; oc < OC; oc++) {
                float* y_row = y + (size_t)oc * spatial + oh * oW;
                int offset = (int)((size_t)oc * spatial + oh * oW);
                post_fn(y_row, 1, oW, oW, fused_op, nullptr, offset);
            }
        }
    });

    return true;
}

// @nnr-meta isa=scalar dtype=fp32
inline void pack_weights_first_layer(
    float* __restrict dst,
    const float* __restrict src,
    int OC, int IC, int KH, int KW)
{
    int OC_blocks = (OC + 15) / 16;
    int kSpatial = KH * KW;
    memset(dst, 0, (size_t)IC * kSpatial * OC_blocks * 16 * sizeof(float));
    for (int ic = 0; ic < IC; ic++)
        for (int k = 0; k < kSpatial; k++)
            for (int oc = 0; oc < OC; oc++)
                dst[((size_t)ic * kSpatial + k) * OC_blocks * 16 + oc] =
                    src[((size_t)oc * IC + ic) * kSpatial + k];
}

// @nnr-meta isa=scalar dtype=fp32
inline size_t pack_weights_first_layer_size(int OC, int IC, int KH, int KW)
{
    return (size_t)IC * KH * KW * ((OC + 15) / 16) * 16;
}

} // namespace nnr

#endif
