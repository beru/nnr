#pragma once
// Direct int8 convolution with NEON SDOT (ARMv8.4-A dotprod) — no im2col buffer.
//
// Vectorizes over output channels (4 per int32x4_t register) and output width (4 tile).
// Uses SDOT: 4×(int8×int8)→int32 per lane, 16 MACs per instruction.
//
// Weight layout: [KH*KW][IC/4][OC/4] panels of 16 bytes (4 OC × 4 IC in SDOT order).
// Input: uint8 NCHW, pre-padded with x_zp to eliminate boundary checks (same as x64).
// Output: int32 [OC, oH*oW] matching x64/conv_int8_direct_avx512 semantics — raw
//   dot products y[oc, s] = Σ_{ic,kh,kw} x_uint8[ic, ih, iw] * w_int8[oc, ic, kh, kw].
//   Zero-point compensation (x_zp * w_sum[oc]) is applied by the caller, unchanged.
//
// Because SDOT is signed×signed, uint8 inputs are shifted to int8 via XOR 0x80 on-the-fly.
// The resulting SDOT value equals Σ((x-128)*w); we fold +128*w_sum[oc] into the accumulator
// at the end so the externally-visible output matches the x64 kernel's exactly.
//
// Counterpart of x64/conv_int8_direct_avx512.h.
// MVP: OC_tile=4, OW_tile=4, single-row OH dispatch via for_static.
// Follow-ups: i8mm SMMLA (4× throughput), wider OW tiles, scroll-friendly layouts.

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
#include "backend/arm/conv_int8_nhwc_direct_neon.h"  // SMMLA pack + transpose helpers

namespace nnr {

// Packed weight size in bytes, matching the NEON SDOT-direct layout.
inline size_t pack_weights_int8_direct_neon_size(int OC, int IC, int KH, int KW)
{
    int OC4 = (OC + 3) / 4;
    int IC4 = (IC + 3) / 4;
    return (size_t)KH * KW * IC4 * OC4 * 16;
}

// Repack [OC, IC, KH, KW] int8 weights to SDOT-direct layout:
//   dst[((ks*IC4 + ig)*OC4 + ob)*16 + lane*4 + ic_off] = w[ob*4+lane, ig*4+ic_off, kh, kw]
inline void pack_weights_int8_direct_neon(
    int8_t* __restrict dst,
    const int8_t* __restrict src,  // OIHW
    int OC, int IC, int KH, int KW)
{
    int OC4 = (OC + 3) / 4;
    int IC4 = (IC + 3) / 4;
    int kSpatial = KH * KW;
    std::memset(dst, 0, pack_weights_int8_direct_neon_size(OC, IC, KH, KW));

    for (int oc = 0; oc < OC; oc++) {
        int ob = oc / 4, lane = oc % 4;
        for (int ic = 0; ic < IC; ic++) {
            int ig = ic / 4, ic_off = ic % 4;
            for (int ks = 0; ks < kSpatial; ks++) {
                size_t didx = ((size_t)ks * IC4 + ig) * OC4 * 16
                            + (size_t)ob * 16 + lane * 4 + ic_off;
                dst[didx] = src[((size_t)oc * IC + ic) * kSpatial + ks];
            }
        }
    }
}

// Per-OC weight sums (over IC×KH×KW). Identical to x64's compute_weight_sums_int8_direct.
inline void compute_weight_sums_int8_direct_neon(
    int32_t* __restrict w_sum,
    const int8_t* __restrict src,
    int OC, int IC, int KH, int KW)
{
    int CHW = IC * KH * KW;
    for (int oc = 0; oc < OC; oc++) {
        int32_t s = 0;
        const int8_t* row = src + (size_t)oc * CHW;
        for (int k = 0; k < CHW; k++)
            s += (int32_t)row[k];
        w_sum[oc] = s;
    }
}

// Direct int8 conv kernel (NEON SDOT).
//   y_i32:    int32 [OC, oH*oW] raw Σ(x_uint8 * w_int8), matching x64 semantics.
//   x_padded: uint8 [IC_padded, pH, pW] pre-padded with x_zp (IC_padded = round_up(IC, 4)).
//   w_packed: SDOT-packed weights from pack_weights_int8_direct_neon().
//   w_sum:    per-OC weight sums from compute_weight_sums_int8_direct_neon().
//             (Used to fold +128·w_sum[oc] into output; caller still applies -x_zp·w_sum[oc].)
inline bool conv_int8_direct_neon(
    int32_t* __restrict y_i32,
    const uint8_t* __restrict x_padded,
    const int8_t* __restrict w_packed,
    const int32_t* __restrict w_sum,
    int IC, int pH, int pW,
    int OC, int oH, int oW,
    int KH, int KW,
    int sH, int sW)
{
    if (!has_neon_dotprod()) return false;
    (void)pH;  // pH only used implicitly via plane_stride

    const int OC4 = (OC + 3) / 4;
    const int IC4 = (IC + 3) / 4;
    const int spatial = oH * oW;
    const size_t plane_stride = (size_t)pH * pW;
    constexpr int WT = 4;  // output-width tile

    nnr::for_static(0, oH, oH >= 4, [&](int oh) {
        for (int ob = 0; ob < OC4; ob++) {
            int oc_base = ob * 4;
            int32_t ws_arr[4] = {
                oc_base + 0 < OC ? w_sum[oc_base + 0] : 0,
                oc_base + 1 < OC ? w_sum[oc_base + 1] : 0,
                oc_base + 2 < OC ? w_sum[oc_base + 2] : 0,
                oc_base + 3 < OC ? w_sum[oc_base + 3] : 0,
            };
            int32x4_t shift128 = vshlq_n_s32(vld1q_s32(ws_arr), 7);  // *128

            int ow = 0;
            for (; ow + WT <= oW; ow += WT) {
                int32x4_t a0 = vdupq_n_s32(0);
                int32x4_t a1 = vdupq_n_s32(0);
                int32x4_t a2 = vdupq_n_s32(0);
                int32x4_t a3 = vdupq_n_s32(0);

                for (int kh = 0; kh < KH; kh++) {
                    const int ih = oh * sH + kh;
                    for (int kw = 0; kw < KW; kw++) {
                        const int ks = kh * KW + kw;
                        for (int ig = 0; ig < IC4; ig++) {
                            int8x16_t wv = vld1q_s8(
                                w_packed + ((size_t)ks * IC4 + ig) * OC4 * 16
                                + (size_t)ob * 16);

                            const uint8_t* p0 = x_padded + (size_t)(ig*4 + 0) * plane_stride + ih * pW;
                            const uint8_t* p1 = p0 + plane_stride;
                            const uint8_t* p2 = p0 + 2 * plane_stride;
                            const uint8_t* p3 = p0 + 3 * plane_stride;

                            #define NNR_SDOT_T(T) { \
                                const int iw = (ow + (T)) * sW + kw; \
                                const uint32_t xv = (uint32_t)p0[iw] \
                                    | ((uint32_t)p1[iw] << 8) \
                                    | ((uint32_t)p2[iw] << 16) \
                                    | ((uint32_t)p3[iw] << 24); \
                                int8x16_t xvec = vreinterpretq_s8_s32( \
                                    vdupq_n_s32((int32_t)(xv ^ 0x80808080u))); \
                                a##T = vdotq_s32(a##T, xvec, wv); \
                            }
                            NNR_SDOT_T(0) NNR_SDOT_T(1)
                            NNR_SDOT_T(2) NNR_SDOT_T(3)
                            #undef NNR_SDOT_T
                        }
                    }
                }

                // Fold +128·w_sum[oc] so output matches Σ(x_uint8 * w_int8).
                a0 = vaddq_s32(a0, shift128);
                a1 = vaddq_s32(a1, shift128);
                a2 = vaddq_s32(a2, shift128);
                a3 = vaddq_s32(a3, shift128);

                // Scatter: lane `lane` of a[t] → y_i32[oc_base+lane][oh*oW + ow+t].
                // 4 OCs × 4 pixels = 16 lane stores per tile (MVP; transpose + contiguous is a follow-up).
                #define NNR_STORE_T(T) { \
                    int s_out = oh * oW + ow + (T); \
                    if (oc_base + 0 < OC) y_i32[(oc_base + 0) * spatial + s_out] = vgetq_lane_s32(a##T, 0); \
                    if (oc_base + 1 < OC) y_i32[(oc_base + 1) * spatial + s_out] = vgetq_lane_s32(a##T, 1); \
                    if (oc_base + 2 < OC) y_i32[(oc_base + 2) * spatial + s_out] = vgetq_lane_s32(a##T, 2); \
                    if (oc_base + 3 < OC) y_i32[(oc_base + 3) * spatial + s_out] = vgetq_lane_s32(a##T, 3); \
                }
                NNR_STORE_T(0) NNR_STORE_T(1)
                NNR_STORE_T(2) NNR_STORE_T(3)
                #undef NNR_STORE_T
            }

            // Tail pixels (< WT)
            for (; ow < oW; ow++) {
                int32x4_t a = vdupq_n_s32(0);
                for (int kh = 0; kh < KH; kh++) {
                    const int ih = oh * sH + kh;
                    for (int kw = 0; kw < KW; kw++) {
                        const int ks = kh * KW + kw;
                        const int iw = ow * sW + kw;
                        for (int ig = 0; ig < IC4; ig++) {
                            int8x16_t wv = vld1q_s8(
                                w_packed + ((size_t)ks * IC4 + ig) * OC4 * 16
                                + (size_t)ob * 16);

                            const uint8_t* pb = x_padded + (size_t)(ig*4) * plane_stride + ih * pW + iw;
                            const uint32_t xv = (uint32_t)pb[0]
                                | ((uint32_t)pb[plane_stride] << 8)
                                | ((uint32_t)pb[2 * plane_stride] << 16)
                                | ((uint32_t)pb[3 * plane_stride] << 24);
                            int8x16_t xvec = vreinterpretq_s8_s32(
                                vdupq_n_s32((int32_t)(xv ^ 0x80808080u)));
                            a = vdotq_s32(a, xvec, wv);
                        }
                    }
                }
                a = vaddq_s32(a, shift128);
                int s_out = oh * oW + ow;
                if (oc_base + 0 < OC) y_i32[(oc_base + 0) * spatial + s_out] = vgetq_lane_s32(a, 0);
                if (oc_base + 1 < OC) y_i32[(oc_base + 1) * spatial + s_out] = vgetq_lane_s32(a, 1);
                if (oc_base + 2 < OC) y_i32[(oc_base + 2) * spatial + s_out] = vgetq_lane_s32(a, 2);
                if (oc_base + 3 < OC) y_i32[(oc_base + 3) * spatial + s_out] = vgetq_lane_s32(a, 3);
            }
        }
    });

    return true;
}

// =============================================================================
// SMMLA variant for NCHW input.
//
// SMMLA needs 8 contiguous IC bytes per spatial pixel — a natural fit for NHWC
// but strided in NCHW (plane-stride apart). Rather than build an SMMLA-shaped
// kernel that walks NCHW with 8 scalar loads per pixel, we pre-transpose the
// activation to NHWC in caller-provided scratch and then run the SMMLA body
// with NCHW-layout output stores. This reuses the NHWC SMMLA weight pack
// unchanged (weight layout is determined by SMMLA, not activation layout).
//
// Transpose + compute tradeoff at C=64 56×56: ~400 KB of scratch (L2-resident),
// ≈4 µs input transpose against ≈30 µs for the SMMLA body — transpose is
// dominated by the conv so the SMMLA win survives on intended shapes.
// =============================================================================
#if defined(__ARM_FEATURE_MATMUL_INT8) || (defined(_MSC_VER) && defined(_M_ARM64))

inline size_t pack_weights_int8_direct_nchw_smmla_size(int OC, int IC, int KH, int KW)
{
    return pack_weights_int8_direct_nhwc_smmla_size(OC, IC, KH, KW);
}

inline void pack_weights_int8_direct_nchw_smmla(
    int8_t* __restrict dst, const int8_t* __restrict src /*OIHW*/,
    int OC, int IC, int KH, int KW)
{
    pack_weights_int8_direct_nhwc_smmla(dst, src, OC, IC, KH, KW);
}

// Scratch size (bytes) for the pre-transposed NHWC activation buffer.
inline size_t conv_int8_direct_nchw_smmla_x_scratch_size(int IC, int pH, int pW)
{
    int IC8 = (IC + 7) / 8;
    return (size_t)pH * pW * IC8 * 8;
}

// 8×8 uint8 matrix transpose. r[i][j] → s[j][i]. Standard 3-level vtrn butterfly.
static inline void transpose_u8_8x8(const uint8x8_t r[8], uint8x8_t s[8])
{
    // Level 0: byte-pair interleave within (0,1),(2,3),(4,5),(6,7).
    uint8x8x2_t t01 = vtrn_u8(r[0], r[1]);
    uint8x8x2_t t23 = vtrn_u8(r[2], r[3]);
    uint8x8x2_t t45 = vtrn_u8(r[4], r[5]);
    uint8x8x2_t t67 = vtrn_u8(r[6], r[7]);
    // Level 1: halfword-pair interleave.
    uint16x4x2_t u02 = vtrn_u16(vreinterpret_u16_u8(t01.val[0]), vreinterpret_u16_u8(t23.val[0]));
    uint16x4x2_t u13 = vtrn_u16(vreinterpret_u16_u8(t01.val[1]), vreinterpret_u16_u8(t23.val[1]));
    uint16x4x2_t u46 = vtrn_u16(vreinterpret_u16_u8(t45.val[0]), vreinterpret_u16_u8(t67.val[0]));
    uint16x4x2_t u57 = vtrn_u16(vreinterpret_u16_u8(t45.val[1]), vreinterpret_u16_u8(t67.val[1]));
    // Level 2: word-pair interleave.
    uint32x2x2_t v04 = vtrn_u32(vreinterpret_u32_u16(u02.val[0]), vreinterpret_u32_u16(u46.val[0]));
    uint32x2x2_t v15 = vtrn_u32(vreinterpret_u32_u16(u13.val[0]), vreinterpret_u32_u16(u57.val[0]));
    uint32x2x2_t v26 = vtrn_u32(vreinterpret_u32_u16(u02.val[1]), vreinterpret_u32_u16(u46.val[1]));
    uint32x2x2_t v37 = vtrn_u32(vreinterpret_u32_u16(u13.val[1]), vreinterpret_u32_u16(u57.val[1]));
    s[0] = vreinterpret_u8_u32(v04.val[0]);
    s[4] = vreinterpret_u8_u32(v04.val[1]);
    s[1] = vreinterpret_u8_u32(v15.val[0]);
    s[5] = vreinterpret_u8_u32(v15.val[1]);
    s[2] = vreinterpret_u8_u32(v26.val[0]);
    s[6] = vreinterpret_u8_u32(v26.val[1]);
    s[3] = vreinterpret_u8_u32(v37.val[0]);
    s[7] = vreinterpret_u8_u32(v37.val[1]);
}

// NCHW [IC_padded_src, pH, pW] → NHWC [pH, pW, IC_padded_dst].
//
// Vectorized 8-channel × 8-pixel tile transpose: loads 8 vectors of 8 bytes (8 planes
// × 8 consecutive width positions), transposes in-register with the vtrn butterfly,
// then writes 8 eight-byte NHWC vectors each to a stride-IC_padded_dst address. Each
// cache line on the write side now receives 8 bytes per touch instead of 1 — an 8×
// reduction in store-side cache traffic vs. a scalar byte-at-a-time pass.
//
// Channels in [IC_padded_src, IC_padded_dst) are filled with x_zp to extend the NCHW
// IC4-padded source up to the IC8 alignment the SMMLA kernel needs.
inline void transpose_nchw_to_nhwc_u8_with_zp_pad(
    uint8_t* __restrict dst, const uint8_t* __restrict src,
    int IC_padded_src, int pH, int pW,
    int IC_padded_dst, uint8_t x_zp)
{
    const size_t plane = (size_t)pH * pW;
    const size_t row_dst = (size_t)pW * IC_padded_dst;

    // Main path assumes IC_padded_src is a multiple of 8 (holds for any IC ≥ 5 since
    // IC4*4 ∈ {8,12,16,...} there; IC ∈ {1..4} falls through to the scalar remainder
    // loop below).
    int c = 0;
    for (; c + 8 <= IC_padded_src; c += 8) {
        const uint8_t* sp[8];
        for (int i = 0; i < 8; i++) sp[i] = src + (size_t)(c + i) * plane;
        for (int h = 0; h < pH; h++) {
            int w = 0;
            for (; w + 8 <= pW; w += 8) {
                uint8x8_t r[8];
                for (int i = 0; i < 8; i++) r[i] = vld1_u8(sp[i] + h * pW + w);
                uint8x8_t s8[8];
                transpose_u8_8x8(r, s8);
                for (int i = 0; i < 8; i++) {
                    uint8_t* d = dst + ((size_t)h * pW + w + i) * IC_padded_dst + c;
                    vst1_u8(d, s8[i]);
                }
            }
            // Width tail (< 8 pixels): fall back to byte-wise for the remainder.
            for (; w < pW; w++) {
                uint8_t* d = dst + ((size_t)h * pW + w) * IC_padded_dst + c;
                for (int i = 0; i < 8; i++) d[i] = sp[i][h * pW + w];
            }
        }
    }
    // Channel tail (< 8 remaining source channels): scalar per-byte.
    for (; c < IC_padded_src; c++) {
        const uint8_t* sp = src + (size_t)c * plane;
        uint8_t* dc = dst + c;
        for (int h = 0; h < pH; h++) {
            const uint8_t* shw = sp + h * pW;
            uint8_t* dhw = dc + h * row_dst;
            for (int w = 0; w < pW; w++)
                dhw[(size_t)w * IC_padded_dst] = shw[w];
        }
    }
    // Zero-point pad for channels [IC_padded_src, IC_padded_dst).
    for (; c < IC_padded_dst; c++) {
        uint8_t* dc = dst + c;
        for (int h = 0; h < pH; h++) {
            uint8_t* dhw = dc + h * row_dst;
            for (int w = 0; w < pW; w++)
                dhw[(size_t)w * IC_padded_dst] = x_zp;
        }
    }
}

// SMMLA NCHW direct conv. NCHW-facing I/O; internally operates on NHWC scratch.
//   y_i32:       int32 NCHW [OC, oH*oW] raw Σ(x_uint8 * w_int8).
//   x_padded:    uint8 NCHW [IC_padded_nchw, pH, pW] (IC_padded_nchw = round_up(IC, 4)).
//   w_packed:    SMMLA pack from pack_weights_int8_direct_nchw_smmla().
//   w_sum:       per-OC weight sums (same as SDOT path).
//   x_zp:        x quantization zero-point (used to pad the IC8 gap in scratch).
//   x_scratch:   caller-provided buffer of size conv_int8_direct_nchw_smmla_x_scratch_size().
inline bool conv_int8_direct_nchw_smmla(
    int32_t* __restrict y_i32,
    const uint8_t* __restrict x_padded,
    const int8_t* __restrict w_packed,
    const int32_t* __restrict w_sum,
    int IC, int pH, int pW,
    int OC, int oH, int oW,
    int KH, int KW, int sH, int sW,
    uint8_t x_zp,
    uint8_t* __restrict x_scratch)
{
    if (!has_neon_i8mm()) return false;
    (void)pH;

    const int IC4 = (IC + 3) / 4;
    const int IC_padded_src = IC4 * 4;
    const int IC8 = (IC + 7) / 8;
    const int IC_padded = IC8 * 8;
    const int OC2 = (OC + 1) / 2;
    const int spatial = oH * oW;

    transpose_nchw_to_nhwc_u8_with_zp_pad(
        x_scratch, x_padded, IC_padded_src, pH, pW, IC_padded, x_zp);

    const uint8_t* x_nhwc = x_scratch;

    auto load_pix8 = [&](int ih, int iw, int ig) -> int8x8_t {
        const uint8_t* p = x_nhwc + ((size_t)ih * pW + iw) * IC_padded + (size_t)ig * 8;
        uint8x8_t u = vld1_u8(p);
        return vreinterpret_s8_u8(veor_u8(u, vdup_n_u8(0x80)));
    };

    nnr::for_static(0, oH, oH >= 4, [&](int oh) {
        int ow = 0;
        for (; ow + 2 <= oW; ow += 2) {
            const int iw0 = ow * sW;
            const int iw1 = (ow + 1) * sW;
            const int s0 = oh * oW + ow;
            const int s1 = s0 + 1;

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
                            const int8_t* wp = w_packed
                                + ((size_t)ks * IC8 + ig) * OC2 * 16
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

                // NCHW stores: lanes 0,1 → pixel s0 ; lanes 2,3 → pixel s1.
                #define NNR_SMMLA_NCHW_STORE(OFF, ACC, L0, L1) \
                    if (oc_base + (OFF) < OC) { \
                        y_i32[(oc_base + (OFF)) * spatial + s0] = vgetq_lane_s32(ACC, L0); \
                        y_i32[(oc_base + (OFF)) * spatial + s1] = vgetq_lane_s32(ACC, L1); \
                    }
                NNR_SMMLA_NCHW_STORE(0, acc_a, 0, 2)
                NNR_SMMLA_NCHW_STORE(1, acc_a, 1, 3)
                NNR_SMMLA_NCHW_STORE(2, acc_b, 0, 2)
                NNR_SMMLA_NCHW_STORE(3, acc_b, 1, 3)
                NNR_SMMLA_NCHW_STORE(4, acc_c, 0, 2)
                NNR_SMMLA_NCHW_STORE(5, acc_c, 1, 3)
                NNR_SMMLA_NCHW_STORE(6, acc_d, 0, 2)
                NNR_SMMLA_NCHW_STORE(7, acc_d, 1, 3)
                #undef NNR_SMMLA_NCHW_STORE
            }
            // OC2 tail (0..3 remaining pairs).
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

                if (oc_base + 0 < OC) {
                    y_i32[(oc_base + 0) * spatial + s0] = vgetq_lane_s32(acc, 0);
                    y_i32[(oc_base + 0) * spatial + s1] = vgetq_lane_s32(acc, 2);
                }
                if (oc_base + 1 < OC) {
                    y_i32[(oc_base + 1) * spatial + s0] = vgetq_lane_s32(acc, 1);
                    y_i32[(oc_base + 1) * spatial + s1] = vgetq_lane_s32(acc, 3);
                }
            }
        }
        // OW tail: single pixel via duplicated SMMLA rows.
        for (; ow < oW; ow++) {
            const int iw = ow * sW;
            const int s_out = oh * oW + ow;
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

                if (oc_base + 0 < OC) y_i32[(oc_base + 0) * spatial + s_out] = vgetq_lane_s32(acc, 0);
                if (oc_base + 1 < OC) y_i32[(oc_base + 1) * spatial + s_out] = vgetq_lane_s32(acc, 1);
            }
        }
    });
    return true;
}

#endif // __ARM_FEATURE_MATMUL_INT8 || (_MSC_VER && _M_ARM64)

} // namespace nnr

#endif // aarch64
