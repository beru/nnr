#pragma once
// NCHW-direct FP16 conv with NEON widening FMA (`vfmlalq_{low,high}_f16`).
//
// Input  NCHW [IC, pH, pW] FP16 bit pattern (spatial zero-pad by caller).
// Output NCHW [OC, oH, oW] FP32 — caller narrows to FP16 if downstream op needs it.
// Weights: FP16, OIHW source, packed by pack_weights_fp16_direct_neon into
//          [OC/8][KH*KW][IC][8] FP16 lanes. OC tail is zero-padded so the
//          kernel always issues a full `vld1q_u16` (8 FP16 = 16 bytes) per step.
//
// Main tile: OC=8 × OW=4 (stride=1 rows only).  Each (kh, kw, ic) inner step:
//     1 × vld1_u16  activation  — 4 contiguous IW values  (8 bytes)
//     1 × vld1q_u16 weights     — 8 OC values            (16 bytes)
//     8 × vfmlalq_low_f16       → 8 FMAs, 64 FP ops; OW dim vectorised in low
//                                  4 lanes (high half of the f16x8 is unused).
//   Matches the NHWC OC=8 × OW=4 tile's density (64 FP ops / 24 bytes) and
//   gives a short OW=0..3 tail (vs OW=0..7 if we went OW=8 × OC=4), which
//   matters most on MobileNet-style 7×7 / 14×14 / 12-wide shapes.
//
// sW > 1 and OW-remainder: single-pixel scalar-broadcast path, reusing the
// vfmlalq_low_f16 widening FMA with the 4-OC weight vector duplicated into
// both halves of an 8-lane vec (only lo half is consumed).  Handles any
// stride and any 0..3 remaining pixels.
//
// 8 named `float32x4_t` accumulators per OC group in the main tile (MSVC
// stack-spills `acc[N]` arrays; see feedback_msvc_array_of_neon_vectors).

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

// Packed size: [OC/8][KH*KW][IC] × 8 FP16 lanes.
inline size_t pack_weights_fp16_direct_neon_size(int OC, int IC, int KH, int KW)
{
    int OC8 = (OC + 7) / 8;
    return (size_t)OC8 * KH * KW * IC * 8 * sizeof(uint16_t);
}

// Repack OIHW [OC, IC, KH, KW] FP16 weights into [OC/8][ks][ic][8] lanes.
// OC tail (OC % 8 != 0) is zero-padded.
inline void pack_weights_fp16_direct_neon(
    uint16_t* __restrict dst,
    const uint16_t* __restrict src, // OIHW FP16 bit pattern
    int OC, int IC, int KH, int KW)
{
    int OC8 = (OC + 7) / 8;
    int kSpatial = KH * KW;
    std::memset(dst, 0, pack_weights_fp16_direct_neon_size(OC, IC, KH, KW));

    for (int oc = 0; oc < OC; oc++) {
        int ob = oc / 8, lane = oc % 8;
        for (int ic = 0; ic < IC; ic++) {
            for (int ks = 0; ks < kSpatial; ks++) {
                size_t didx = ((size_t)ob * kSpatial + ks) * IC * 8
                            + (size_t)ic * 8 + lane;
                dst[didx] = src[((size_t)oc * IC + ic) * kSpatial + ks];
            }
        }
    }
}

// NCHW FP16 direct conv. Output is FP32.
//   y:        [OC, oH, oW]    float
//   x_padded: [IC, pH, pW]    uint16 (FP16, spatially zero-padded)
//   w_packed: pack from pack_weights_fp16_direct_neon
inline bool conv_fp16_direct_neon(
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
    const int kSpatial = KH * KW;
    const size_t x_slab = (size_t)pH * pW;
    const size_t y_slab = (size_t)oH * oW;
    const int OW_T          = (sW == 1) ? (oW / 4) : 0;
    const int OW_tail_begin = OW_T * 4;

    nnr::for_static(0, oH, oH >= 4, [&](int oh) {
        for (int ob = 0; ob < OC8; ob++) {
            const int oc_base = ob * 8;
            const uint16_t* w_base = w_packed
                + (size_t)ob * kSpatial * IC * 8;

            // ── OW=4 main tile (sW==1 only) ──────────────────────────────
            for (int owt = 0; owt < OW_T; owt++) {
                const int ow0 = owt * 4;
                float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
                float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
                float32x4_t a4 = vdupq_n_f32(0), a5 = vdupq_n_f32(0);
                float32x4_t a6 = vdupq_n_f32(0), a7 = vdupq_n_f32(0);

                for (int kh = 0; kh < KH; kh++) {
                    const int ih = oh * sH + kh;
                    for (int kw = 0; kw < KW; kw++) {
                        const int ks = kh * KW + kw;
                        const int iw_base = ow0 + kw;  // sW=1
                        const uint16_t* w_ks = w_base + (size_t)ks * IC * 8;

                        for (int ic = 0; ic < IC; ic++) {
                            // Load 4 contiguous IW FP16 values; duplicate into
                            // both halves of an 8-lane vec so vfmlalq_low can
                            // consume the low 4 lanes.
                            uint16x4_t x4 = vld1_u16(
                                x_padded + (size_t)ic * x_slab
                                + (size_t)ih * pW + iw_base);
                            float16x8_t xv = vreinterpretq_f16_u16(
                                vcombine_u16(x4, x4));
                            // Load 8 OC weights once; lane-broadcast per OC.
                            uint16x8_t w8 = vld1q_u16(w_ks + (size_t)ic * 8);
                            float16x8_t b0 = vreinterpretq_f16_u16(vdupq_laneq_u16(w8, 0));
                            float16x8_t b1 = vreinterpretq_f16_u16(vdupq_laneq_u16(w8, 1));
                            float16x8_t b2 = vreinterpretq_f16_u16(vdupq_laneq_u16(w8, 2));
                            float16x8_t b3 = vreinterpretq_f16_u16(vdupq_laneq_u16(w8, 3));
                            float16x8_t b4 = vreinterpretq_f16_u16(vdupq_laneq_u16(w8, 4));
                            float16x8_t b5 = vreinterpretq_f16_u16(vdupq_laneq_u16(w8, 5));
                            float16x8_t b6 = vreinterpretq_f16_u16(vdupq_laneq_u16(w8, 6));
                            float16x8_t b7 = vreinterpretq_f16_u16(vdupq_laneq_u16(w8, 7));
                            a0 = vfmlalq_low_f16(a0, xv, b0);
                            a1 = vfmlalq_low_f16(a1, xv, b1);
                            a2 = vfmlalq_low_f16(a2, xv, b2);
                            a3 = vfmlalq_low_f16(a3, xv, b3);
                            a4 = vfmlalq_low_f16(a4, xv, b4);
                            a5 = vfmlalq_low_f16(a5, xv, b5);
                            a6 = vfmlalq_low_f16(a6, xv, b6);
                            a7 = vfmlalq_low_f16(a7, xv, b7);
                        }
                    }
                }

                auto store_oc = [&](int oc, float32x4_t acc) {
                    if (oc >= OC) return;
                    vst1q_f32(y + (size_t)oc * y_slab
                        + (size_t)oh * oW + ow0, acc);
                };
                store_oc(oc_base + 0, a0);
                store_oc(oc_base + 1, a1);
                store_oc(oc_base + 2, a2);
                store_oc(oc_base + 3, a3);
                store_oc(oc_base + 4, a4);
                store_oc(oc_base + 5, a5);
                store_oc(oc_base + 6, a6);
                store_oc(oc_base + 7, a7);
            }

            // ── OW tail (and full row when sW>1): single-pixel per step ──
            // Two float32x4 accumulators (lo = OCs 0..3, hi = OCs 4..7).
            const int tail_start = (sW == 1) ? OW_tail_begin : 0;
            for (int ow = tail_start; ow < oW; ow++) {
                float32x4_t acc_lo = vdupq_n_f32(0);
                float32x4_t acc_hi = vdupq_n_f32(0);
                for (int kh = 0; kh < KH; kh++) {
                    const int ih = oh * sH + kh;
                    for (int kw = 0; kw < KW; kw++) {
                        const int iw = ow * sW + kw;
                        const int ks = kh * KW + kw;
                        const uint16_t* w_ks = w_base + (size_t)ks * IC * 8;
                        for (int ic = 0; ic < IC; ic++) {
                            uint16_t xs = x_padded[
                                (size_t)ic * x_slab
                                + (size_t)ih * pW + iw];
                            float16x8_t x_bcast = vreinterpretq_f16_u16(
                                vdupq_n_u16(xs));
                            float16x8_t w8 = vreinterpretq_f16_u16(
                                vld1q_u16(w_ks + (size_t)ic * 8));
                            acc_lo = vfmlalq_low_f16 (acc_lo, x_bcast, w8);
                            acc_hi = vfmlalq_high_f16(acc_hi, x_bcast, w8);
                        }
                    }
                }
                alignas(16) float tmp[8];
                vst1q_f32(tmp + 0, acc_lo);
                vst1q_f32(tmp + 4, acc_hi);
                int tail = std::min(8, OC - oc_base);
                for (int t = 0; t < tail; t++)
                    y[(size_t)(oc_base + t) * y_slab
                      + (size_t)oh * oW + ow] = tmp[t];
            }
        }
    });
    return true;
}

} // namespace nnr::fp16::neon

#endif // aarch64
