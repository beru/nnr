#pragma once
// NHWC-direct FP16 conv with NEON widening FMA (`vfmlalq_{low,high}_f16`).
//
// Input  NHWC [pH, pW, IC] FP16 bit pattern (spatial zero-pad applied by caller).
// Output NHWC [oH, oW, OC] FP32 — caller narrows to FP16 if downstream op needs it.
// Weights: FP16, OIHW source, packed into [ks][ob][ic][8] FP16 lanes by
//          pack_weights_fp16_direct_nhwc_neon; OC padded up to a multiple of 8
//          with 0.0 so the kernel can issue a full vld1q_u16 per OC group.
//
// Tile: OC=8 × OW=4 main body + OW=1 tail (also used when oW<4).  Per inner
// (kh, kw, ic) step we issue ONE f16x8 weight load shared by all 4 OW pixels
// plus 4 scalar f16 activation loads/broadcasts; that feeds 8 `vfmlalq_*` FMAs
// (low+high per OW pixel), amortising the weight bandwidth 4× over the
// single-pixel shape.  Named accumulators (8 × `float32x4_t`) — MSVC spills
// `float32x4_t acc[N]` to stack when any access uses a variable index (see
// memory:feedback_msvc_array_of_neon_vectors).
//
// Runtime-gated by `has_neon_fp16()` (all three NNR ARM targets satisfy this).
// External element type is `uint16_t` to sidestep the MSVC-vs-GCC split on
// `float16_t` native types (same convention as `gemm_fp16_neon.h`).

#if defined(__aarch64__) || defined(_M_ARM64)

#ifdef _MSC_VER
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif
#include <algorithm>
#include <cstdint>
#include <cstring>
#include "cpu_features.h"
#include "thread_pool.h"

namespace nnr::fp16::neon {

// Packed size: [KH*KW][OC/8][IC] × 8 FP16 lanes.
inline size_t pack_weights_fp16_direct_nhwc_neon_size(int OC, int IC, int KH, int KW)
{
    int OC8 = (OC + 7) / 8;
    return (size_t)KH * KW * OC8 * IC * 8 * sizeof(uint16_t);
}

// Repack OIHW [OC, IC, KH, KW] FP16 weights into [KH*KW][OC/8][IC][8] NHWC layout.
// OC tail (OC % 8 != 0) is zero-padded; no IC padding is needed because the inner
// step consumes one IC at a time.
inline void pack_weights_fp16_direct_nhwc_neon(
    uint16_t* __restrict dst,
    const uint16_t* __restrict src, // OIHW FP16 bit pattern
    int OC, int IC, int KH, int KW)
{
    int OC8 = (OC + 7) / 8;
    int kSpatial = KH * KW;
    std::memset(dst, 0, pack_weights_fp16_direct_nhwc_neon_size(OC, IC, KH, KW));

    for (int oc = 0; oc < OC; oc++) {
        int ob = oc / 8, lane = oc % 8;
        for (int ic = 0; ic < IC; ic++) {
            for (int ks = 0; ks < kSpatial; ks++) {
                size_t didx = ((size_t)ks * OC8 + ob) * IC * 8
                            + (size_t)ic * 8 + lane;
                dst[didx] = src[((size_t)oc * IC + ic) * kSpatial + ks];
            }
        }
    }
}

// NHWC FP16 direct conv. Output is FP32.
//   y:        [oH, oW, OC]   float
//   x_padded: [pH, pW, IC]   uint16 (FP16 bit pattern, spatially zero-padded)
//   w_packed: pack from pack_weights_fp16_direct_nhwc_neon
inline bool conv_fp16_direct_nhwc_neon(
    float* __restrict y,
    const uint16_t* __restrict x_padded,
    const uint16_t* __restrict w_packed,
    int IC, int pH, int pW,
    int OC, int oH, int oW,
    int KH, int KW, int sH, int sW)
{
    if (!has_neon_fp16()) return false;
    (void)pH;

    const int OC8 = (OC + 7) / 8;
    const int OW_T = oW / 4;             // OW=4 tile count
    const int OW_tail_begin = OW_T * 4;  // first OW index in tail

    nnr::for_static(0, oH, oH >= 4, [&](int oh) {
        // ── OW=4 main tile ───────────────────────────────────────────────
        for (int owt = 0; owt < OW_T; owt++) {
            const int ow0 = owt * 4;
            float* y_row = y + ((size_t)oh * oW + ow0) * OC;

            for (int ob = 0; ob < OC8; ob++) {
                int oc_base = ob * 8;
                float32x4_t acc_lo0 = vdupq_n_f32(0), acc_hi0 = vdupq_n_f32(0);
                float32x4_t acc_lo1 = vdupq_n_f32(0), acc_hi1 = vdupq_n_f32(0);
                float32x4_t acc_lo2 = vdupq_n_f32(0), acc_hi2 = vdupq_n_f32(0);
                float32x4_t acc_lo3 = vdupq_n_f32(0), acc_hi3 = vdupq_n_f32(0);

                for (int kh = 0; kh < KH; kh++) {
                    const int ih = oh * sH + kh;
                    for (int kw = 0; kw < KW; kw++) {
                        const int ks = kh * KW + kw;
                        const uint16_t* w_grp = w_packed
                            + ((size_t)ks * OC8 + ob) * IC * 8;
                        const int iw_base = ow0 * sW + kw;
                        const uint16_t* x_p0 = x_padded
                            + ((size_t)ih * pW + iw_base + 0 * sW) * IC;
                        const uint16_t* x_p1 = x_padded
                            + ((size_t)ih * pW + iw_base + 1 * sW) * IC;
                        const uint16_t* x_p2 = x_padded
                            + ((size_t)ih * pW + iw_base + 2 * sW) * IC;
                        const uint16_t* x_p3 = x_padded
                            + ((size_t)ih * pW + iw_base + 3 * sW) * IC;

                        for (int ic = 0; ic < IC; ic++) {
                            float16x8_t wv = vreinterpretq_f16_u16(
                                vld1q_u16(w_grp + (size_t)ic * 8));
                            float16x8_t x0 = vreinterpretq_f16_u16(vdupq_n_u16(x_p0[ic]));
                            float16x8_t x1 = vreinterpretq_f16_u16(vdupq_n_u16(x_p1[ic]));
                            float16x8_t x2 = vreinterpretq_f16_u16(vdupq_n_u16(x_p2[ic]));
                            float16x8_t x3 = vreinterpretq_f16_u16(vdupq_n_u16(x_p3[ic]));
                            acc_lo0 = vfmlalq_low_f16 (acc_lo0, x0, wv);
                            acc_hi0 = vfmlalq_high_f16(acc_hi0, x0, wv);
                            acc_lo1 = vfmlalq_low_f16 (acc_lo1, x1, wv);
                            acc_hi1 = vfmlalq_high_f16(acc_hi1, x1, wv);
                            acc_lo2 = vfmlalq_low_f16 (acc_lo2, x2, wv);
                            acc_hi2 = vfmlalq_high_f16(acc_hi2, x2, wv);
                            acc_lo3 = vfmlalq_low_f16 (acc_lo3, x3, wv);
                            acc_hi3 = vfmlalq_high_f16(acc_hi3, x3, wv);
                        }
                    }
                }

                if (oc_base + 8 <= OC) {
                    vst1q_f32(y_row + 0 * OC + oc_base + 0, acc_lo0);
                    vst1q_f32(y_row + 0 * OC + oc_base + 4, acc_hi0);
                    vst1q_f32(y_row + 1 * OC + oc_base + 0, acc_lo1);
                    vst1q_f32(y_row + 1 * OC + oc_base + 4, acc_hi1);
                    vst1q_f32(y_row + 2 * OC + oc_base + 0, acc_lo2);
                    vst1q_f32(y_row + 2 * OC + oc_base + 4, acc_hi2);
                    vst1q_f32(y_row + 3 * OC + oc_base + 0, acc_lo3);
                    vst1q_f32(y_row + 3 * OC + oc_base + 4, acc_hi3);
                } else {
                    alignas(16) float tmp[4][8];
                    vst1q_f32(tmp[0] + 0, acc_lo0); vst1q_f32(tmp[0] + 4, acc_hi0);
                    vst1q_f32(tmp[1] + 0, acc_lo1); vst1q_f32(tmp[1] + 4, acc_hi1);
                    vst1q_f32(tmp[2] + 0, acc_lo2); vst1q_f32(tmp[2] + 4, acc_hi2);
                    vst1q_f32(tmp[3] + 0, acc_lo3); vst1q_f32(tmp[3] + 4, acc_hi3);
                    int tail = OC - oc_base;
                    for (int p = 0; p < 4; p++)
                        for (int t = 0; t < tail; t++)
                            y_row[p * OC + oc_base + t] = tmp[p][t];
                }
            }
        }

        // ── OW=1 tail (0..3 remaining pixels; also covers oW < 4) ────────
        for (int ow = OW_tail_begin; ow < oW; ow++) {
            float* y_pix = y + ((size_t)oh * oW + ow) * OC;
            for (int ob = 0; ob < OC8; ob++) {
                int oc_base = ob * 8;
                float32x4_t acc_lo = vdupq_n_f32(0);
                float32x4_t acc_hi = vdupq_n_f32(0);

                for (int kh = 0; kh < KH; kh++) {
                    const int ih = oh * sH + kh;
                    for (int kw = 0; kw < KW; kw++) {
                        const int iw = ow * sW + kw;
                        const int ks = kh * KW + kw;
                        const uint16_t* x_pix = x_padded
                            + ((size_t)ih * pW + iw) * IC;
                        const uint16_t* w_grp = w_packed
                            + ((size_t)ks * OC8 + ob) * IC * 8;

                        for (int ic = 0; ic < IC; ic++) {
                            float16x8_t xv = vreinterpretq_f16_u16(
                                vdupq_n_u16(x_pix[ic]));
                            float16x8_t wv = vreinterpretq_f16_u16(
                                vld1q_u16(w_grp + (size_t)ic * 8));
                            acc_lo = vfmlalq_low_f16 (acc_lo, xv, wv);
                            acc_hi = vfmlalq_high_f16(acc_hi, xv, wv);
                        }
                    }
                }

                if (oc_base + 8 <= OC) {
                    vst1q_f32(y_pix + oc_base + 0, acc_lo);
                    vst1q_f32(y_pix + oc_base + 4, acc_hi);
                } else {
                    alignas(16) float tmp[8];
                    vst1q_f32(tmp + 0, acc_lo);
                    vst1q_f32(tmp + 4, acc_hi);
                    int tail = OC - oc_base;
                    for (int t = 0; t < tail; t++)
                        y_pix[oc_base + t] = tmp[t];
                }
            }
        }
    });
    return true;
}

} // namespace nnr::fp16::neon

#endif // aarch64
