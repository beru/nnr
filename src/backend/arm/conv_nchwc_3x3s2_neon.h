#pragma once
// NEON NCHW8c direct Conv specialized for KH=KW=3, strideH=strideW=2.
//
// Shape: input  [ICb, iH, iW, 8]  output [OCb, oH, oW, 8]
//        weight [OCb, IC, 3, 3, 8]  (8 OC per block contiguous at innermost)
//
// Why a dedicated kernel: stride-2 3×3 is a MobileNet-v2 / ResNet entry-block
// hot path. Hard-unrolling KH×KW=9 and stride-2 iw offsets lets the compiler
// fold addressing math into immediates rather than running a generic loop.
//
// Counterpart of x64/conv_nchwc_3x3s2_avx512.h.
// MVP tile: F=1 OC block (8 channels) × OW tile = 4 spatial positions.
// Follow-ups: F=2 (16 OCs per iter), OW=6, interior-vs-edge split, stride-1.

#if defined(__aarch64__) || defined(_M_ARM64)

#ifdef _MSC_VER
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif
#include <algorithm>
#include <cstdint>
#include <cstring>
#include "thread_pool.h"

#ifndef NNR_FORCEINLINE
#ifdef _MSC_VER
#define NNR_FORCEINLINE __forceinline
#else
#define NNR_FORCEINLINE inline __attribute__((always_inline))
#endif
#endif

namespace nnr {

// Pack [OC, IC, 3, 3] OIHW → [OCb, IC, 3, 3, 8]. block=8.
inline size_t pack_weight_nchwc8_3x3_size(int OC, int IC) {
    int OCb = (OC + 7) / 8;
    return (size_t)OCb * IC * 3 * 3 * 8 * sizeof(float);
}

inline void pack_weight_nchwc8_3x3(
    float* __restrict dst,
    const float* __restrict src,  // [OC, IC, 3, 3]
    int OC, int IC)
{
    int OCb = (OC + 7) / 8;
    std::memset(dst, 0, pack_weight_nchwc8_3x3_size(OC, IC));
    for (int ob = 0; ob < OCb; ob++) {
        for (int ic = 0; ic < IC; ic++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    for (int l = 0; l < 8; l++) {
                        int oc = ob * 8 + l;
                        if (oc >= OC) continue;
                        size_t d = ((((size_t)ob * IC + ic) * 3 + kh) * 3 + kw) * 8 + l;
                        dst[d] = src[((size_t)oc * IC + ic) * 9 + kh * 3 + kw];
                    }
                }
            }
        }
    }
}

// 3×3 stride-2 NCHW8c kernel. Expects input padded so that for every
// (oh, ow) in [oh_start, oh_end) × [0, oW), all 9 kernel taps at
// (oh*2 + kh - padH, ow*2 + kw - padW) are in [0, pH) × [0, pW).
// Caller handles edges (or passes oH_interior).
inline void conv_nchwc8_3x3s2_neon(
    float* __restrict out,              // [OCb, oH, oW, 8]
    const float* __restrict x_padded,   // [ICb, pH, pW, 8]
    const float* __restrict w_packed,   // [OCb, IC, 3, 3, 8] from pack_weight_nchwc8_3x3
    const float* __restrict bias,       // [OC] or nullptr
    int IC, int pH, int pW,
    int OC, int oH, int oW)
{
    (void)pH;
    const int OCb = (OC + 7) / 8;
    const int ICb = (IC + 7) / 8;

    nnr::for_static(0, OCb * oH, oH >= 4, [&](int job) {
        int ob = job / oH;
        int oh = job % oH;

        int oc_base = ob * 8;
        // Bias can be partial at the last OC block (OC % 8 != 0). Load via a
        // zero-initialized scratch so the invalid lanes are 0 — otherwise a
        // direct `vld1q_f32(bias + oc_base + 4)` reads past the end of the
        // bias vector (undefined; MSVC/stack-layout-lucky but fails under GCC).
        float32x4_t bv_lo, bv_hi;
        if (bias) {
            float bias_buf[8] = {0};
            int n_valid = std::min(8, OC - oc_base);
            std::memcpy(bias_buf, bias + oc_base, (size_t)n_valid * sizeof(float));
            bv_lo = vld1q_f32(bias_buf);
            bv_hi = vld1q_f32(bias_buf + 4);
        } else {
            bv_lo = vdupq_n_f32(0.0f);
            bv_hi = vdupq_n_f32(0.0f);
        }

        float* out_row = out + ((size_t)ob * oH + oh) * oW * 8;

        int ih_base = oh * 2;  // pre-padded input — caller handles padding

        constexpr int WT = 4;
        int ow = 0;
        for (; ow + WT <= oW; ow += WT) {
            int iw_base = ow * 2;

            float32x4_t acc_lo[WT], acc_hi[WT];
            for (int t = 0; t < WT; t++) {
                acc_lo[t] = bv_lo;
                acc_hi[t] = bv_hi;
            }

            for (int ic = 0; ic < IC; ic++) {
                int ib = ic >> 3, il = ic & 7;
                for (int kh = 0; kh < 3; kh++) {
                    int ih = ih_base + kh;
                    const float* in_row = x_padded
                        + (((size_t)ib * pH + ih) * pW + iw_base) * 8;
                    for (int kw = 0; kw < 3; kw++) {
                        const float* wptr = w_packed
                            + (((((size_t)ob * IC + ic) * 3) + kh) * 3 + kw) * 8;
                        float32x4_t wv_lo = vld1q_f32(wptr + 0);
                        float32x4_t wv_hi = vld1q_f32(wptr + 4);

                        // 4 spatial positions at stride 2: offsets 2t+kw in 8-wide blocks.
                        for (int t = 0; t < WT; t++) {
                            const float* x_slot = in_row + (t * 2 + kw) * 8;
                            float a = x_slot[il];
                            acc_lo[t] = vfmaq_n_f32(acc_lo[t], wv_lo, a);
                            acc_hi[t] = vfmaq_n_f32(acc_hi[t], wv_hi, a);
                        }
                    }
                }
            }

            for (int t = 0; t < WT; t++) {
                vst1q_f32(out_row + (ow + t) * 8 + 0, acc_lo[t]);
                vst1q_f32(out_row + (ow + t) * 8 + 4, acc_hi[t]);
            }
        }

        // OW tail (<4 positions)
        for (; ow < oW; ow++) {
            float32x4_t acc_lo = bv_lo;
            float32x4_t acc_hi = bv_hi;
            int iw_base = ow * 2;
            for (int ic = 0; ic < IC; ic++) {
                int ib = ic >> 3, il = ic & 7;
                for (int kh = 0; kh < 3; kh++) {
                    int ih = ih_base + kh;
                    for (int kw = 0; kw < 3; kw++) {
                        const float* wptr = w_packed
                            + (((((size_t)ob * IC + ic) * 3) + kh) * 3 + kw) * 8;
                        float32x4_t wv_lo = vld1q_f32(wptr + 0);
                        float32x4_t wv_hi = vld1q_f32(wptr + 4);
                        const float* in_row = x_padded
                            + (((size_t)ib * pH + ih) * pW + iw_base + kw) * 8;
                        float a = in_row[il];
                        acc_lo = vfmaq_n_f32(acc_lo, wv_lo, a);
                        acc_hi = vfmaq_n_f32(acc_hi, wv_hi, a);
                    }
                }
            }
            vst1q_f32(out_row + ow * 8 + 0, acc_lo);
            vst1q_f32(out_row + ow * 8 + 4, acc_hi);
        }
        (void)ICb;
    });
}

} // namespace nnr

#endif // aarch64
