#pragma once
// Direct convolution for last-layer Conv (small output channels, e.g., RGB output).
// Skips im2col entirely. Vectorizes over output width (16 pixels per ZMM register).
//
// 8 ZMM accumulators cover 128 output pixels — hides FMA latency (8 chains > 4-cycle lat).
// Weight layout: original NCHW [OC, IC, KH, KW] — no repacking needed (scalar broadcast).
// Input: pre-padded NCHW buffer [IC, pH, pW] with zero padding.

#ifdef NNR_ARCH_X64

#include <immintrin.h>
#include <cstring>

namespace nnr {

// Pre-pad NCHW input into workspace buffer.
// Output: [IC, pH, pW] with zeros outside the valid region.
// @nnr-meta isa=scalar dtype=fp32 layout=NCHW
inline void conv_last_layer_prepad(
    float* __restrict pad_buf,
    const float* __restrict x,    // [IC, iH, iW] NCHW
    int IC, int iH, int iW,
    int pH, int pW,
    int padH_top, int padW_left)
{
    memset(pad_buf, 0, (size_t)IC * pH * pW * sizeof(float));
    for (int ic = 0; ic < IC; ic++)
        for (int h = 0; h < iH; h++)
            memcpy(pad_buf + (size_t)ic * pH * pW + (h + padH_top) * pW + padW_left,
                   x + (size_t)ic * iH * iW + h * iW,
                   iW * sizeof(float));
}

// Pre-pad NHWC input directly into NCHW padded workspace.
// Combines NHWC→NCHW conversion and padding in one pass.
// @nnr-meta isa=scalar dtype=fp32 layout=NHWC
inline void conv_last_layer_prepad_nhwc(
    float* __restrict pad_buf,
    const float* __restrict x,    // [iH, iW, IC] NHWC
    int IC, int iH, int iW,
    int pH, int pW,
    int padH_top, int padW_left)
{
    memset(pad_buf, 0, (size_t)IC * pH * pW * sizeof(float));
    for (int h = 0; h < iH; h++)
        for (int w = 0; w < iW; w++) {
            const float* src = x + ((size_t)h * iW + w) * IC;
            for (int c = 0; c < IC; c++)
                pad_buf[(size_t)c * pH * pW + (h + padH_top) * pW + (w + padW_left)] = src[c];
        }
}

// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=LastLayer fusion=post_op
inline void conv_last_layer_avx512(
    float* __restrict y,           // [OC, OH, OW] NCHW output
    const float* __restrict w,     // [OC, IC, KH, KW] original layout
    const float* __restrict bias,
    int IC, int OC, int oH, int oW,
    int KH, int KW,
    int pH, int pW,                // padded input dimensions
    const float* __restrict pad_buf, // [IC, pH, pW] pre-padded input
    operator_t::post_fn_t post_fn,
    const operator_t* fused_op)
{
    const int spatial = oH * oW;
    const int kSpatial = KH * KW;

    // Process each output channel
    for (int oc = 0; oc < OC; oc++) {
        const float b = bias ? bias[oc] : 0.0f;
        const __m512 vbias = _mm512_set1_ps(b);
        float* y_oc = y + (size_t)oc * spatial;
        const float* w_oc = w + (size_t)oc * IC * kSpatial;

        nnr::for_static(0, oH, oH >= 4, [&](int oh) {
            float* out_row = y_oc + oh * oW;
            int ow = 0;

            // Main loop: 8 × 16 = 128 pixels per iteration
            for (; ow + 128 <= oW; ow += 128) {
                __m512 a0 = _mm512_setzero_ps();
                __m512 a1 = _mm512_setzero_ps();
                __m512 a2 = _mm512_setzero_ps();
                __m512 a3 = _mm512_setzero_ps();
                __m512 a4 = _mm512_setzero_ps();
                __m512 a5 = _mm512_setzero_ps();
                __m512 a6 = _mm512_setzero_ps();
                __m512 a7 = _mm512_setzero_ps();

                for (int ic = 0; ic < IC; ic++) {
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh + kh;  // padded coords
                        const float* in_row = pad_buf + (size_t)ic * pH * pW + ih * pW;
                        const float* wk = w_oc + (size_t)ic * kSpatial + kh * KW;

                        for (int kw = 0; kw < KW; kw++) {
                            __m512 wv = _mm512_set1_ps(wk[kw]);
                            int iw = ow + kw;  // padded coords
                            a0 = _mm512_fmadd_ps(wv, _mm512_loadu_ps(in_row + iw),       a0);
                            a1 = _mm512_fmadd_ps(wv, _mm512_loadu_ps(in_row + iw + 16),  a1);
                            a2 = _mm512_fmadd_ps(wv, _mm512_loadu_ps(in_row + iw + 32),  a2);
                            a3 = _mm512_fmadd_ps(wv, _mm512_loadu_ps(in_row + iw + 48),  a3);
                            a4 = _mm512_fmadd_ps(wv, _mm512_loadu_ps(in_row + iw + 64),  a4);
                            a5 = _mm512_fmadd_ps(wv, _mm512_loadu_ps(in_row + iw + 80),  a5);
                            a6 = _mm512_fmadd_ps(wv, _mm512_loadu_ps(in_row + iw + 96),  a6);
                            a7 = _mm512_fmadd_ps(wv, _mm512_loadu_ps(in_row + iw + 112), a7);
                        }
                    }
                }

                _mm512_storeu_ps(out_row + ow,       _mm512_add_ps(a0, vbias));
                _mm512_storeu_ps(out_row + ow + 16,  _mm512_add_ps(a1, vbias));
                _mm512_storeu_ps(out_row + ow + 32,  _mm512_add_ps(a2, vbias));
                _mm512_storeu_ps(out_row + ow + 48,  _mm512_add_ps(a3, vbias));
                _mm512_storeu_ps(out_row + ow + 64,  _mm512_add_ps(a4, vbias));
                _mm512_storeu_ps(out_row + ow + 80,  _mm512_add_ps(a5, vbias));
                _mm512_storeu_ps(out_row + ow + 96,  _mm512_add_ps(a6, vbias));
                _mm512_storeu_ps(out_row + ow + 112, _mm512_add_ps(a7, vbias));
            }

            // Middle loop: 16 pixels per iteration
            for (; ow + 16 <= oW; ow += 16) {
                __m512 acc = _mm512_setzero_ps();
                for (int ic = 0; ic < IC; ic++) {
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh + kh;
                        const float* in_row = pad_buf + (size_t)ic * pH * pW + ih * pW;
                        const float* wk = w_oc + (size_t)ic * kSpatial + kh * KW;
                        for (int kw = 0; kw < KW; kw++)
                            acc = _mm512_fmadd_ps(_mm512_set1_ps(wk[kw]),
                                _mm512_loadu_ps(in_row + ow + kw), acc);
                    }
                }
                _mm512_storeu_ps(out_row + ow, _mm512_add_ps(acc, vbias));
            }

            // Tail: masked store for remaining pixels
            if (ow < oW) {
                __mmask16 mask = (__mmask16)((1u << (oW - ow)) - 1);
                __m512 acc = _mm512_setzero_ps();
                for (int ic = 0; ic < IC; ic++) {
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh + kh;
                        const float* in_row = pad_buf + (size_t)ic * pH * pW + ih * pW;
                        const float* wk = w_oc + (size_t)ic * kSpatial + kh * KW;
                        for (int kw = 0; kw < KW; kw++)
                            acc = _mm512_fmadd_ps(_mm512_set1_ps(wk[kw]),
                                _mm512_loadu_ps(in_row + ow + kw), acc);
                    }
                }
                _mm512_mask_storeu_ps(out_row + ow, mask, _mm512_add_ps(acc, vbias));
            }

            // Apply post-op for this row
            if (post_fn) {
                int offset = (int)((size_t)oc * spatial + oh * oW);
                post_fn(out_row, 1, oW, oW, fused_op, nullptr, offset);
            }
        });
    }
}

// Workspace size for last-layer kernel (pre-padded input buffer)
// @nnr-meta isa=scalar dtype=fp32
inline size_t conv_last_layer_workspace(int IC, int iH, int iW,
    int padH_top, int padH_bot, int padW_left, int padW_right)
{
    int pH = iH + padH_top + padH_bot;
    int pW = iW + padW_left + padW_right + 15;  // +15 for SIMD overread
    return (size_t)IC * pH * pW * sizeof(float);
}

} // namespace nnr

#endif
