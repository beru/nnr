#pragma once
// NHWC-direct int8 conv with NEON SDOT.
//
// Input  NHWC [pH, pW, IC_padded] uint8 (pre-padded with x_zp, IC_padded = round_up(IC, 4)).
// Output NHWC [oH, oW, OC] int32 raw Σ(x_uint8 * w_int8) matching NCHW kernel semantics
//        (caller applies -x_zp·w_sum[oc] + scale + bias + requantize).
// Weights: same SDOT pack as conv_int8_direct_neon.h (OC=4 × IC=4 tile).
//
// NHWC packing is a natural fit for SDOT — the 4 IC values needed per (ih, iw, ig) slot
// are contiguous in memory, so one 32-bit load replaces the 4 scalar channel reads needed
// in NCHW. MVP is output-pixel-at-a-time with OC tile = 4; a proper OW tile (+ memcpy-free
// OOB indirection á la x64 jit_nhwc_direct_gemm_nr16) is a follow-up.

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

namespace nnr {

inline bool conv_int8_direct_nhwc_neon(
    int32_t* __restrict y_i32,         // NHWC [oH, oW, OC]
    const uint8_t* __restrict x_padded, // NHWC [pH, pW, IC_padded] padded with x_zp
    const int8_t* __restrict w_packed,  // SDOT pack from pack_weights_int8_direct_neon
    const int32_t* __restrict w_sum,
    int IC, int pH, int pW,
    int OC, int oH, int oW,
    int KH, int KW, int sH, int sW)
{
    if (!has_neon_dotprod()) return false;
    (void)pH;

    const int OC4 = (OC + 3) / 4;
    const int IC4 = (IC + 3) / 4;
    const int IC_padded = IC4 * 4;

    nnr::for_static(0, oH, oH >= 4, [&](int oh) {
        for (int ow = 0; ow < oW; ow++) {
            int32_t* y_pix = y_i32 + ((size_t)oh * oW + ow) * OC;

            for (int ob = 0; ob < OC4; ob++) {
                int oc_base = ob * 4;
                int32x4_t acc = vdupq_n_s32(0);

                for (int kh = 0; kh < KH; kh++) {
                    const int ih = oh * sH + kh;
                    for (int kw = 0; kw < KW; kw++) {
                        const int iw = ow * sW + kw;
                        const int ks = kh * KW + kw;
                        const uint8_t* x_pix = x_padded
                            + ((size_t)ih * pW + iw) * IC_padded;

                        for (int ig = 0; ig < IC4; ig++) {
                            int8x16_t wv = vld1q_s8(
                                w_packed + ((size_t)ks * IC4 + ig) * OC4 * 16
                                + (size_t)ob * 16);
                            uint32_t xv;
                            std::memcpy(&xv, x_pix + ig * 4, 4);  // 4 contiguous IC bytes
                            int8x16_t xvec = vreinterpretq_s8_s32(
                                vdupq_n_s32((int32_t)(xv ^ 0x80808080u)));
                            acc = vdotq_s32(acc, xvec, wv);
                        }
                    }
                }

                // Fold +128·w_sum[oc] to match Σ(x_uint8 * w_int8) semantics
                int32_t ws_arr[4] = {
                    oc_base + 0 < OC ? w_sum[oc_base + 0] : 0,
                    oc_base + 1 < OC ? w_sum[oc_base + 1] : 0,
                    oc_base + 2 < OC ? w_sum[oc_base + 2] : 0,
                    oc_base + 3 < OC ? w_sum[oc_base + 3] : 0,
                };
                int32x4_t ws = vshlq_n_s32(vld1q_s32(ws_arr), 7);  // *128
                acc = vaddq_s32(acc, ws);

                // Store up to 4 OCs for this pixel. Full-vector store when ob*4+4 ≤ OC,
                // else lane-scatter for tail.
                if (oc_base + 4 <= OC) {
                    vst1q_s32(y_pix + oc_base, acc);
                } else {
                    // Tail (1..3 OCs). vgetq_lane_s32 requires a constant lane index,
                    // so unroll rather than loop.
                    int tail = OC - oc_base;
                    if (tail >= 1) y_pix[oc_base + 0] = vgetq_lane_s32(acc, 0);
                    if (tail >= 2) y_pix[oc_base + 1] = vgetq_lane_s32(acc, 1);
                    if (tail >= 3) y_pix[oc_base + 2] = vgetq_lane_s32(acc, 2);
                }
            }
        }
    });
    return true;
}

// =============================================================================
// SMMLA variant — NHWC int8 conv using the ARMv8.6 i8mm `vmmlaq_s32` primitive.
// Runtime-gated by caller via has_neon_i8mm(). Uses its own pack layout (2-OC × 8-IC
// groups, 16-byte panels), matching the GEMM SMMLA approach in gemm_int8_neon.h.
// Tile: OC=4 × OW=2 (2 output pixels × 2 OC-pairs = 2 SMMLAs per IC-group-of-8).
// =============================================================================
#if defined(__ARM_FEATURE_MATMUL_INT8) || (defined(_MSC_VER) && defined(_M_ARM64))

// Packed weight size: [KH*KW][IC/8][OC/2] × 16-byte panels.
inline size_t pack_weights_int8_direct_nhwc_smmla_size(int OC, int IC, int KH, int KW)
{
    int OC2 = (OC + 1) / 2;
    int IC8 = (IC + 7) / 8;
    return (size_t)KH * KW * IC8 * OC2 * 16;
}

// Repack [OC, IC, KH, KW] int8 weights to NHWC SMMLA layout.
// Each 16-byte panel = [OC(2ob)_IC(8ig)..IC(8ig+7) | OC(2ob+1)_IC(8ig)..IC(8ig+7)].
inline void pack_weights_int8_direct_nhwc_smmla(
    int8_t* __restrict dst,
    const int8_t* __restrict src,  // OIHW
    int OC, int IC, int KH, int KW)
{
    int OC2 = (OC + 1) / 2;
    int IC8 = (IC + 7) / 8;
    int kSpatial = KH * KW;
    std::memset(dst, 0, pack_weights_int8_direct_nhwc_smmla_size(OC, IC, KH, KW));

    for (int oc = 0; oc < OC; oc++) {
        int ob = oc / 2, lane = oc % 2;
        for (int ic = 0; ic < IC; ic++) {
            int ig = ic / 8, ic_off = ic % 8;
            for (int ks = 0; ks < kSpatial; ks++) {
                size_t didx = ((size_t)ks * IC8 + ig) * OC2 * 16
                            + (size_t)ob * 16 + (size_t)lane * 8 + ic_off;
                dst[didx] = src[((size_t)oc * IC + ic) * kSpatial + ks];
            }
        }
    }
}

// NHWC int8 direct conv using SMMLA (2 OW-pixels × 2 OC-pairs tile).
// x_padded is NHWC [pH, pW, IC_padded] where IC_padded = round_up(IC, 8),
// pre-filled with x_zp in the padded region (same convention as the SDOT path,
// but padded to IC8 rather than IC4).
inline bool conv_int8_direct_nhwc_smmla(
    int32_t* __restrict y_i32,          // NHWC [oH, oW, OC] int32
    const uint8_t* __restrict x_padded, // NHWC [pH, pW, IC_padded] padded with x_zp
    const int8_t* __restrict w_packed,  // SMMLA pack (2-OC × 8-IC panels)
    const int32_t* __restrict w_sum,
    int IC, int pH, int pW,
    int OC, int oH, int oW,
    int KH, int KW, int sH, int sW)
{
    if (!has_neon_i8mm()) return false;
    (void)pH;

    const int OC2 = (OC + 1) / 2;
    const int IC8 = (IC + 7) / 8;
    const int IC_padded = IC8 * 8;

    // Load 8 contiguous IC bytes from pixel (ih, iw) at ig's 8-IC window, XOR 0x80.
    auto load_pix8 = [&](int ih, int iw, int ig) -> int8x8_t {
        const uint8_t* p = x_padded + ((size_t)ih * pW + iw) * IC_padded + (size_t)ig * 8;
        uint8x8_t u = vld1_u8(p);
        return vreinterpret_s8_u8(veor_u8(u, vdup_n_u8(0x80)));
    };

    nnr::for_static(0, oH, oH >= 4, [&](int oh) {
        int ow = 0;
        for (; ow + 2 <= oW; ow += 2) {
            const int iw0 = ow * sW;
            const int iw1 = (ow + 1) * sW;

            // OC main body: process 4 OC-pairs (8 OCs) per outer iteration so one
            // A-build feeds 4 SMMLAs — the single biggest lever past the basic tile
            // because SMMLA issue isn't the bottleneck, the byte-strided A rebuild is.
            // Register usage: 4 acc + 4 W + 1 A = 9 live vregs (NEON has 32).
            int ob = 0;
            for (; ob + 4 <= OC2; ob += 4) {
                int oc_base = ob * 2;
                int32x4_t acc_a = vdupq_n_s32(0);
                int32x4_t acc_b = vdupq_n_s32(0);
                int32x4_t acc_c = vdupq_n_s32(0);
                int32x4_t acc_d = vdupq_n_s32(0);

                for (int kh = 0; kh < KH; kh++) {
                    const int ih = oh * sH + kh;
                    for (int kw = 0; kw < KW; kw++) {
                        const int ks = kh * KW + kw;
                        const int iw0k = iw0 + kw;
                        const int iw1k = iw1 + kw;

                        for (int ig = 0; ig < IC8; ig++) {
                            int8x8_t p0 = load_pix8(ih, iw0k, ig);
                            int8x8_t p1 = load_pix8(ih, iw1k, ig);
                            int8x16_t av = vcombine_s8(p0, p1);

                            const int8_t* wp = w_packed + ((size_t)ks * IC8 + ig) * OC2 * 16
                                             + (size_t)ob * 16;
                            int8x16_t wv_a = vld1q_s8(wp +  0);
                            int8x16_t wv_b = vld1q_s8(wp + 16);
                            int8x16_t wv_c = vld1q_s8(wp + 32);
                            int8x16_t wv_d = vld1q_s8(wp + 48);

                            acc_a = vmmlaq_s32(acc_a, av, wv_a);
                            acc_b = vmmlaq_s32(acc_b, av, wv_b);
                            acc_c = vmmlaq_s32(acc_c, av, wv_c);
                            acc_d = vmmlaq_s32(acc_d, av, wv_d);
                        }
                    }
                }

                // Fold +128·w_sum and scatter-store 8 OCs × 2 pixels.
                int32_t ws[8] = {
                    (oc_base + 0 < OC) ? w_sum[oc_base + 0] * 128 : 0,
                    (oc_base + 1 < OC) ? w_sum[oc_base + 1] * 128 : 0,
                    (oc_base + 2 < OC) ? w_sum[oc_base + 2] * 128 : 0,
                    (oc_base + 3 < OC) ? w_sum[oc_base + 3] * 128 : 0,
                    (oc_base + 4 < OC) ? w_sum[oc_base + 4] * 128 : 0,
                    (oc_base + 5 < OC) ? w_sum[oc_base + 5] * 128 : 0,
                    (oc_base + 6 < OC) ? w_sum[oc_base + 6] * 128 : 0,
                    (oc_base + 7 < OC) ? w_sum[oc_base + 7] * 128 : 0,
                };
                int32_t wsa_a[4] = { ws[0], ws[1], ws[0], ws[1] };
                int32_t wsa_b[4] = { ws[2], ws[3], ws[2], ws[3] };
                int32_t wsa_c[4] = { ws[4], ws[5], ws[4], ws[5] };
                int32_t wsa_d[4] = { ws[6], ws[7], ws[6], ws[7] };
                acc_a = vaddq_s32(acc_a, vld1q_s32(wsa_a));
                acc_b = vaddq_s32(acc_b, vld1q_s32(wsa_b));
                acc_c = vaddq_s32(acc_c, vld1q_s32(wsa_c));
                acc_d = vaddq_s32(acc_d, vld1q_s32(wsa_d));

                int32_t* y_p0 = y_i32 + ((size_t)oh * oW + ow + 0) * OC;
                int32_t* y_p1 = y_i32 + ((size_t)oh * oW + ow + 1) * OC;
                // Pixel 0 gets lanes 0,1 of each acc; pixel 1 gets lanes 2,3.
                if (oc_base + 0 < OC) y_p0[oc_base + 0] = vgetq_lane_s32(acc_a, 0);
                if (oc_base + 1 < OC) y_p0[oc_base + 1] = vgetq_lane_s32(acc_a, 1);
                if (oc_base + 2 < OC) y_p0[oc_base + 2] = vgetq_lane_s32(acc_b, 0);
                if (oc_base + 3 < OC) y_p0[oc_base + 3] = vgetq_lane_s32(acc_b, 1);
                if (oc_base + 4 < OC) y_p0[oc_base + 4] = vgetq_lane_s32(acc_c, 0);
                if (oc_base + 5 < OC) y_p0[oc_base + 5] = vgetq_lane_s32(acc_c, 1);
                if (oc_base + 6 < OC) y_p0[oc_base + 6] = vgetq_lane_s32(acc_d, 0);
                if (oc_base + 7 < OC) y_p0[oc_base + 7] = vgetq_lane_s32(acc_d, 1);
                if (oc_base + 0 < OC) y_p1[oc_base + 0] = vgetq_lane_s32(acc_a, 2);
                if (oc_base + 1 < OC) y_p1[oc_base + 1] = vgetq_lane_s32(acc_a, 3);
                if (oc_base + 2 < OC) y_p1[oc_base + 2] = vgetq_lane_s32(acc_b, 2);
                if (oc_base + 3 < OC) y_p1[oc_base + 3] = vgetq_lane_s32(acc_b, 3);
                if (oc_base + 4 < OC) y_p1[oc_base + 4] = vgetq_lane_s32(acc_c, 2);
                if (oc_base + 5 < OC) y_p1[oc_base + 5] = vgetq_lane_s32(acc_c, 3);
                if (oc_base + 6 < OC) y_p1[oc_base + 6] = vgetq_lane_s32(acc_d, 2);
                if (oc_base + 7 < OC) y_p1[oc_base + 7] = vgetq_lane_s32(acc_d, 3);
            }
            // OC2 tail: 0..3 OC-pairs remaining, one-at-a-time for simplicity.
            for (; ob < OC2; ob++) {
                int oc_base = ob * 2;
                int32x4_t acc = vdupq_n_s32(0);

                for (int kh = 0; kh < KH; kh++) {
                    const int ih = oh * sH + kh;
                    for (int kw = 0; kw < KW; kw++) {
                        const int ks = kh * KW + kw;
                        const int iw0k = iw0 + kw;
                        const int iw1k = iw1 + kw;
                        for (int ig = 0; ig < IC8; ig++) {
                            int8x8_t p0 = load_pix8(ih, iw0k, ig);
                            int8x8_t p1 = load_pix8(ih, iw1k, ig);
                            int8x16_t av = vcombine_s8(p0, p1);
                            int8x16_t wv = vld1q_s8(
                                w_packed + ((size_t)ks * IC8 + ig) * OC2 * 16
                                + (size_t)ob * 16);
                            acc = vmmlaq_s32(acc, av, wv);
                        }
                    }
                }

                int32_t ws0 = (oc_base + 0 < OC) ? w_sum[oc_base + 0] : 0;
                int32_t ws1 = (oc_base + 1 < OC) ? w_sum[oc_base + 1] : 0;
                int32_t wsa[4] = { ws0 * 128, ws1 * 128, ws0 * 128, ws1 * 128 };
                acc = vaddq_s32(acc, vld1q_s32(wsa));

                int32_t* y_p0 = y_i32 + ((size_t)oh * oW + ow + 0) * OC;
                int32_t* y_p1 = y_i32 + ((size_t)oh * oW + ow + 1) * OC;
                if (oc_base + 0 < OC) y_p0[oc_base + 0] = vgetq_lane_s32(acc, 0);
                if (oc_base + 1 < OC) y_p0[oc_base + 1] = vgetq_lane_s32(acc, 1);
                if (oc_base + 0 < OC) y_p1[oc_base + 0] = vgetq_lane_s32(acc, 2);
                if (oc_base + 1 < OC) y_p1[oc_base + 1] = vgetq_lane_s32(acc, 3);
            }
        }
        // OW tail (single pixel): reuse the SMMLA op with the pixel duplicated in both
        // "rows"; lanes 2,3 of the result mirror lanes 0,1 and we only store the pair
        // from lanes 0,1. Simpler than carrying a separate SDOT-shaped path, and the
        // wasted compute is ≤ 1 pixel per row — amortized away on any realistic oW.
        for (; ow < oW; ow++) {
            const int iw = ow * sW;

            for (int ob = 0; ob < OC2; ob++) {
                int oc_base = ob * 2;
                int32x4_t acc = vdupq_n_s32(0);

                for (int kh = 0; kh < KH; kh++) {
                    const int ih = oh * sH + kh;
                    for (int kw = 0; kw < KW; kw++) {
                        const int ks = kh * KW + kw;
                        const int iwk = iw + kw;
                        for (int ig = 0; ig < IC8; ig++) {
                            int8x8_t p = load_pix8(ih, iwk, ig);
                            int8x16_t av = vcombine_s8(p, p);

                            int8x16_t wv = vld1q_s8(
                                w_packed + ((size_t)ks * IC8 + ig) * OC2 * 16
                                + (size_t)ob * 16);

                            acc = vmmlaq_s32(acc, av, wv);
                        }
                    }
                }

                int32_t ws0 = (oc_base + 0 < OC) ? w_sum[oc_base + 0] : 0;
                int32_t ws1 = (oc_base + 1 < OC) ? w_sum[oc_base + 1] : 0;
                int32_t wsa[4] = { ws0 * 128, ws1 * 128, ws0 * 128, ws1 * 128 };
                acc = vaddq_s32(acc, vld1q_s32(wsa));

                int32_t* y_p = y_i32 + ((size_t)oh * oW + ow) * OC;
                if (oc_base + 0 < OC) y_p[oc_base + 0] = vgetq_lane_s32(acc, 0);
                if (oc_base + 1 < OC) y_p[oc_base + 1] = vgetq_lane_s32(acc, 1);
            }
        }
    });
    return true;
}

#endif // __ARM_FEATURE_MATMUL_INT8 || (_MSC_VER && _M_ARM64)

} // namespace nnr

#endif // aarch64
