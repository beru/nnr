#pragma once
// Direct convolution for last-layer Conv (small output channels, e.g., RGB output).
// ARM NEON version: vectorizes over output width (4 pixels per q-register).
//
// 8 q-register accumulators = 32 output pixels per main loop iteration —
// hides FMA latency (8 chains > 4-cycle latency).
// Weight layout: original NCHW [OC, IC, KH, KW] — no repacking needed (scalar broadcast).
// Input: pre-padded NCHW buffer [IC, pH, pW] with zero padding.
//
// IC tiling: when IC > IC_TILE, processes input channels in blocks to keep
// the pad_buf working set in L2 (~8 channels × pH × pW ≈ 384KB).
// Partial results accumulate in the output buffer between tiles.

#ifdef NNR_ARCH_ARM64

#include <arm_neon.h>
#include <cstring>
#include <algorithm>

#include "cpu_features.h"

namespace nnr {

// Pre-pad NCHW input into workspace buffer.
// Output: [IC, pH, pW] with zeros outside the valid region.
// @nnr-meta isa=scalar dtype=fp32 layout=NCHW special=LastLayer
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
// @nnr-meta isa=scalar dtype=fp32 layout=NHWC special=LastLayer
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

// @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=LastLayer tiling=spatial fusion=post_op
inline void conv_last_layer_neon(
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
    // IC tile size: keep pad_buf working set in L2.
    // pad_buf per tile = IC_TILE × pH × pW × 4 B; target half of L2 so the
    // weight panel and adjacent working sets have room to coexist. On
    // typical first-layer shapes (~108×111) this yields IC_TILE≈8 at 512 KB L2.
    int IC_TILE;
    {
        const size_t l2_half = (size_t)cpu_features().l2_kb * 1024 / 2;
        const size_t per_ch = (size_t)pH * pW * sizeof(float);
        int t = (per_ch > 0) ? (int)(l2_half / per_ch) : 8;
        IC_TILE = std::clamp(t, 1, 32);
    }
    const int num_ic_tiles = (IC + IC_TILE - 1) / IC_TILE;

    for (int oc = 0; oc < OC; oc++) {
        const float b = bias ? bias[oc] : 0.0f;
        float* y_oc = y + (size_t)oc * spatial;
        const float* w_oc = w + (size_t)oc * IC * kSpatial;

        for (int ic_tile = 0; ic_tile < num_ic_tiles; ic_tile++) {
            const int ic0 = ic_tile * IC_TILE;
            const int ic1 = std::min(ic0 + IC_TILE, IC);
            const bool is_first = (ic_tile == 0);
            const bool is_last = (ic_tile == num_ic_tiles - 1);

            nnr::for_static(0, oH, oH >= 4, [&](int oh) {
                float* out_row = y_oc + oh * oW;
                int ow = 0;

                // Main loop: 8 × 4 = 32 pixels per iteration
                for (; ow + 32 <= oW; ow += 32) {
                    float32x4_t a0, a1, a2, a3, a4, a5, a6, a7;
                    if (is_first) {
                        a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = vdupq_n_f32(0.f);
                    } else {
                        a0 = vld1q_f32(out_row + ow);
                        a1 = vld1q_f32(out_row + ow +  4);
                        a2 = vld1q_f32(out_row + ow +  8);
                        a3 = vld1q_f32(out_row + ow + 12);
                        a4 = vld1q_f32(out_row + ow + 16);
                        a5 = vld1q_f32(out_row + ow + 20);
                        a6 = vld1q_f32(out_row + ow + 24);
                        a7 = vld1q_f32(out_row + ow + 28);
                    }

                    for (int ic = ic0; ic < ic1; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh + kh;
                            const float* in_row = pad_buf + (size_t)ic * pH * pW + ih * pW;
                            const float* wk = w_oc + (size_t)ic * kSpatial + kh * KW;
                            for (int kw = 0; kw < KW; kw++) {
                                float32x4_t wv = vdupq_n_f32(wk[kw]);
                                int iw = ow + kw;
                                a0 = vfmaq_f32(a0, wv, vld1q_f32(in_row + iw));
                                a1 = vfmaq_f32(a1, wv, vld1q_f32(in_row + iw +  4));
                                a2 = vfmaq_f32(a2, wv, vld1q_f32(in_row + iw +  8));
                                a3 = vfmaq_f32(a3, wv, vld1q_f32(in_row + iw + 12));
                                a4 = vfmaq_f32(a4, wv, vld1q_f32(in_row + iw + 16));
                                a5 = vfmaq_f32(a5, wv, vld1q_f32(in_row + iw + 20));
                                a6 = vfmaq_f32(a6, wv, vld1q_f32(in_row + iw + 24));
                                a7 = vfmaq_f32(a7, wv, vld1q_f32(in_row + iw + 28));
                            }
                        }
                    }

                    if (is_last) {
                        float32x4_t vb = vdupq_n_f32(b);
                        vst1q_f32(out_row + ow,      vaddq_f32(a0, vb));
                        vst1q_f32(out_row + ow +  4, vaddq_f32(a1, vb));
                        vst1q_f32(out_row + ow +  8, vaddq_f32(a2, vb));
                        vst1q_f32(out_row + ow + 12, vaddq_f32(a3, vb));
                        vst1q_f32(out_row + ow + 16, vaddq_f32(a4, vb));
                        vst1q_f32(out_row + ow + 20, vaddq_f32(a5, vb));
                        vst1q_f32(out_row + ow + 24, vaddq_f32(a6, vb));
                        vst1q_f32(out_row + ow + 28, vaddq_f32(a7, vb));
                    } else {
                        vst1q_f32(out_row + ow,      a0);
                        vst1q_f32(out_row + ow +  4, a1);
                        vst1q_f32(out_row + ow +  8, a2);
                        vst1q_f32(out_row + ow + 12, a3);
                        vst1q_f32(out_row + ow + 16, a4);
                        vst1q_f32(out_row + ow + 20, a5);
                        vst1q_f32(out_row + ow + 24, a6);
                        vst1q_f32(out_row + ow + 28, a7);
                    }
                }

                // Middle loop: 4 pixels per iteration
                for (; ow + 4 <= oW; ow += 4) {
                    float32x4_t acc;
                    if (is_first)
                        acc = vdupq_n_f32(0.f);
                    else
                        acc = vld1q_f32(out_row + ow);

                    for (int ic = ic0; ic < ic1; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh + kh;
                            const float* in_row = pad_buf + (size_t)ic * pH * pW + ih * pW;
                            const float* wk = w_oc + (size_t)ic * kSpatial + kh * KW;
                            for (int kw = 0; kw < KW; kw++)
                                acc = vfmaq_f32(acc, vdupq_n_f32(wk[kw]),
                                    vld1q_f32(in_row + ow + kw));
                        }
                    }
                    if (is_last)
                        vst1q_f32(out_row + ow, vaddq_f32(acc, vdupq_n_f32(b)));
                    else
                        vst1q_f32(out_row + ow, acc);
                }

                // Tail: scalar
                for (; ow < oW; ow++) {
                    float acc = is_first ? 0.f : out_row[ow];
                    for (int ic = ic0; ic < ic1; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh + kh;
                            const float* in_row = pad_buf + (size_t)ic * pH * pW + ih * pW;
                            const float* wk = w_oc + (size_t)ic * kSpatial + kh * KW;
                            for (int kw = 0; kw < KW; kw++)
                                acc += wk[kw] * in_row[ow + kw];
                        }
                    }
                    out_row[ow] = is_last ? acc + b : acc;
                }

                if (is_last && post_fn) {
                    int offset = (int)((size_t)oc * spatial + oh * oW);
                    post_fn(out_row, 1, oW, oW, fused_op, nullptr, offset);
                }
            });
        }
    }
}

// Workspace size for last-layer kernel (pre-padded input buffer).
// +3 overread padding for NEON vld1q_f32 safety.
// @nnr-meta isa=scalar dtype=fp32 layout=NCHW special=LastLayer
inline size_t conv_last_layer_workspace(int IC, int iH, int iW,
    int padH_top, int padH_bot, int padW_left, int padW_right)
{
    int pH = iH + padH_top + padH_bot;
    int pW = iW + padW_left + padW_right + 3;  // +3 for NEON overread
    return (size_t)IC * pH * pW * sizeof(float);
}

} // namespace nnr

#endif // NNR_ARCH_ARM64
