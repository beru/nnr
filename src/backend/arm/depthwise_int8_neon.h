#pragma once
// NEON int8 depthwise convolution (NHWC layout) with fused requantize.
//
// Counterpart of x64/depthwise_int8_avx512.h. Vectorizes across channels (8 per iter,
// two int32x4_t accumulators). Uses an indirection buffer to handle padding without
// inner-loop branches — same external structure as the x64 version.
//
// Depthwise has no K reduction across channels, so SDOT doesn't help; we do plain
// int32 MLA (vmlaq_s32) on zero-point-subtracted operands.
//
// Main tile is OW=4 × C=8: 4 output pixels processed in parallel per channel
// group, giving 8 live int32x4 accumulators to break the per-output MLA dep chain
// and amortizing each filter load 4× across output pixels. OW=1 fall-through
// path handles the 0-3 output tail.

#if defined(__aarch64__) || defined(_M_ARM64)

#ifdef _MSC_VER
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include "thread_pool.h"

namespace nnr::int8::neon {

// Repack depthwise weights from [OC, 1, kH, kW] to [kH*kW, C_pad].
// C_pad = round_up(OC, 4) — 4-aligned for NEON int32x4_t stride.
inline size_t repack_depthwise_weights_neon_size(int OC, int kH, int kW) {
    int OC_pad = (OC + 3) & ~3;
    return (size_t)kH * kW * OC_pad;
}

inline void repack_depthwise_weights_neon(
    int8_t* dst, const int8_t* src,
    int OC, int kH, int kW)
{
    int OC_pad = (OC + 3) & ~3;
    int kHW = kH * kW;
    std::memset(dst, 0, (size_t)kHW * OC_pad);
    for (int oc = 0; oc < OC; oc++)
        for (int k = 0; k < kHW; k++)
            dst[k * OC_pad + oc] = src[oc * kHW + k];
}

// Build indirection buffer. Identical semantics to x64's build_depthwise_indirection.
inline void build_depthwise_indirection_neon(
    const uint8_t** ind,
    const uint8_t* x_nhwc,
    const uint8_t* zero_buf,
    int oH, int oW, int iH, int iW, int C,
    int kH, int kW, int sH, int sW, int pH, int pW,
    int dH, int dW)
{
    int kHW = kH * kW;
    for (int oh = 0; oh < oH; oh++) {
        for (int ow = 0; ow < oW; ow++) {
            int out_idx = oh * oW + ow;
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh * sH - pH + kh * dH;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = ow * sW - pW + kw * dW;
                    int k = kh * kW + kw;
                    if ((unsigned)ih < (unsigned)iH && (unsigned)iw < (unsigned)iW)
                        ind[out_idx * kHW + k] = x_nhwc + ((size_t)ih * iW + iw) * C;
                    else
                        ind[out_idx * kHW + k] = zero_buf;
                }
            }
        }
    }
}

// Requantize + store 4 int32 accumulators as 4 uint8 bytes.
static inline void dw_requantize_store4_neon(
    uint8_t* dst,
    int32x4_t acc,
    float32x4_t vcs, float32x4_t vbias,
    float32x4_t v_inv_y, float32x4_t v_y_zp,
    float32x4_t v_qmin, float32x4_t v_qmax)
{
    float32x4_t facc = vfmaq_f32(vbias, vcvtq_f32_s32(acc), vcs);
    float32x4_t q = vmulq_f32(facc, v_inv_y);
    q = vrndnq_f32(q);                               // round to nearest even
    q = vaddq_f32(q, v_y_zp);
    q = vmaxq_f32(vminq_f32(q, v_qmax), v_qmin);
    int32x4_t qi = vcvtq_s32_f32(q);
    // Saturating narrow 4 × int32 → 4 × uint8 via int16 intermediate
    uint8_t buf[8];
    uint16x4_t q16 = vqmovun_s32(qi);                // sat to [0, 0xFFFF]
    uint8x8_t  q8  = vqmovn_u16(vcombine_u16(q16, q16));
    vst1_u8(buf, q8);
    std::memcpy(dst, buf, 4);
}

// Requantize 8 int32 (lo+hi) accumulators and store as 8 contiguous uint8 bytes.
// Only safe when 8 channels fit at dst — use for the 8-channel main tile.
static inline void dw_requantize_store8_neon(
    uint8_t* dst,
    int32x4_t acc_lo, int32x4_t acc_hi,
    float32x4_t vcs_lo, float32x4_t vcs_hi,
    float32x4_t vbias_lo, float32x4_t vbias_hi,
    float32x4_t v_inv_y, float32x4_t v_y_zp,
    float32x4_t v_qmin, float32x4_t v_qmax)
{
    float32x4_t f_lo = vfmaq_f32(vbias_lo, vcvtq_f32_s32(acc_lo), vcs_lo);
    float32x4_t f_hi = vfmaq_f32(vbias_hi, vcvtq_f32_s32(acc_hi), vcs_hi);
    float32x4_t q_lo = vaddq_f32(vrndnq_f32(vmulq_f32(f_lo, v_inv_y)), v_y_zp);
    float32x4_t q_hi = vaddq_f32(vrndnq_f32(vmulq_f32(f_hi, v_inv_y)), v_y_zp);
    q_lo = vmaxq_f32(vminq_f32(q_lo, v_qmax), v_qmin);
    q_hi = vmaxq_f32(vminq_f32(q_hi, v_qmax), v_qmin);
    uint16x4_t q16_lo = vqmovun_s32(vcvtq_s32_f32(q_lo));
    uint16x4_t q16_hi = vqmovun_s32(vcvtq_s32_f32(q_hi));
    uint8x8_t  q8 = vqmovn_u16(vcombine_u16(q16_lo, q16_hi));
    vst1_u8(dst, q8);
}

// Main kernel. Signature mirrors the x64 depthwise_int8_nhwc_avx512 closely.
inline void depthwise_int8_nhwc_neon(
    uint8_t* __restrict y_nhwc,
    const uint8_t* const* __restrict indirection,
    const int8_t* __restrict filter,     // [kH*kW, C_pad] from repack_depthwise_weights_neon
    const float* __restrict combined_scale,   // [C]
    const float* __restrict bias_f,           // [C] or nullptr
    int C, int output_count, int kernel_size,
    int x_zp, int w_zp,
    float inv_y_scale, float y_zp_f, float qmin, float qmax)
{
    const int C_pad = (C + 3) & ~3;
    int32x4_t v_x_zp = vdupq_n_s32(x_zp);
    int32x4_t v_w_zp = vdupq_n_s32(w_zp);
    int16x8_t v_x_zp_s16 = vdupq_n_s16((int16_t)x_zp);  // (x - x_zp) fits in int16
    int16x8_t v_w_zp_s16 = vdupq_n_s16((int16_t)w_zp);  // (w - w_zp) fits in int16
    float32x4_t v_inv_y = vdupq_n_f32(inv_y_scale);
    float32x4_t v_y_zp  = vdupq_n_f32(y_zp_f);
    float32x4_t v_qmin  = vdupq_n_f32(qmin);
    float32x4_t v_qmax  = vdupq_n_f32(qmax);

    // Fast path applies when w_zp == 0 (symmetric weight quantization, i.e.
    // every test case here and the typical QLinearConv usage).  Precompute
    // w_sum[c] = sum_k(w_k) so the inner loop can skip the x-zp subtract:
    //   sum((x - x_zp) * w) = sum(x * w) - x_zp * w_sum
    // The correction -x_zp * w_sum is folded into the effective bias at
    // requantize time. Saves ~1 vsubq_s16 per pixel per k-step.
    constexpr int W_SUM_MAX_C = 2048;
    alignas(16) int32_t w_sum_buf[W_SUM_MAX_C];
    const bool wzp0_fast = (w_zp == 0) && (C <= W_SUM_MAX_C);
    if (wzp0_fast) {
        int c = 0;
        for (; c + 8 <= C; c += 8) {
            int16x8_t acc = vdupq_n_s16(0);
            for (int k = 0; k < kernel_size; k++)
                acc = vaddq_s16(acc, vmovl_s8(vld1_s8(filter + k * C_pad + c)));
            vst1q_s32(w_sum_buf + c,     vmovl_s16(vget_low_s16(acc)));
            vst1q_s32(w_sum_buf + c + 4, vmovl_s16(vget_high_s16(acc)));
        }
        for (; c < C; c++) {
            int32_t s = 0;
            for (int k = 0; k < kernel_size; k++) s += filter[k * C_pad + c];
            w_sum_buf[c] = s;
        }
    }
    float32x4_t v_neg_xzp_f = vdupq_n_f32((float)(-x_zp));

    int op = 0;
    // OW=4 tile: 4 output pixels share each filter load across the K loop.
    // 8 parallel int32x4 accumulators (4 pixels × lo/hi) break the per-output
    // MLA dependency chain — the main throughput lever over the OW=1 MVP.
    for (; op + 4 <= output_count; op += 4) {
        int c = 0;
        // Fast 4-pixel × 8-channel tile for w_zp=0: no x-zp subtract in the
        // inner loop; correction folded into bias at requantize time.
        if (wzp0_fast) {
            for (; c + 8 <= C; c += 8) {
                int32x4_t a0_lo = vdupq_n_s32(0), a0_hi = vdupq_n_s32(0);
                int32x4_t a1_lo = vdupq_n_s32(0), a1_hi = vdupq_n_s32(0);
                int32x4_t a2_lo = vdupq_n_s32(0), a2_hi = vdupq_n_s32(0);
                int32x4_t a3_lo = vdupq_n_s32(0), a3_hi = vdupq_n_s32(0);
                for (int k = 0; k < kernel_size; k++) {
                    int16x8_t ws = vmovl_s8(vld1_s8(filter + k * C_pad + c));
                    int16x4_t ws_lo = vget_low_s16(ws);
                    int16x4_t ws_hi = vget_high_s16(ws);

                    const uint8_t* const* ind_k = indirection + (size_t)op * kernel_size;
                    #define DW_FAST_PIXEL(P, A_LO, A_HI) do { \
                        uint8x8_t xu = vld1_u8(ind_k[(P) * kernel_size + k] + c); \
                        int16x8_t xs = vreinterpretq_s16_u16(vmovl_u8(xu)); \
                        A_LO = vmlal_s16(A_LO, vget_low_s16(xs),  ws_lo); \
                        A_HI = vmlal_s16(A_HI, vget_high_s16(xs), ws_hi); \
                    } while (0)
                    DW_FAST_PIXEL(0, a0_lo, a0_hi);
                    DW_FAST_PIXEL(1, a1_lo, a1_hi);
                    DW_FAST_PIXEL(2, a2_lo, a2_hi);
                    DW_FAST_PIXEL(3, a3_lo, a3_hi);
                    #undef DW_FAST_PIXEL
                }
                // Requantize with bias_eff = bias + (-x_zp) * cs * w_sum
                float32x4_t vcs_lo = vld1q_f32(combined_scale + c);
                float32x4_t vcs_hi = vld1q_f32(combined_scale + c + 4);
                float32x4_t vws_lo = vcvtq_f32_s32(vld1q_s32(w_sum_buf + c));
                float32x4_t vws_hi = vcvtq_f32_s32(vld1q_s32(w_sum_buf + c + 4));
                float32x4_t vbe_lo = bias_f ? vld1q_f32(bias_f + c)     : vdupq_n_f32(0.0f);
                float32x4_t vbe_hi = bias_f ? vld1q_f32(bias_f + c + 4) : vdupq_n_f32(0.0f);
                vbe_lo = vfmaq_f32(vbe_lo, v_neg_xzp_f, vmulq_f32(vcs_lo, vws_lo));
                vbe_hi = vfmaq_f32(vbe_hi, v_neg_xzp_f, vmulq_f32(vcs_hi, vws_hi));
                dw_requantize_store8_neon(y_nhwc + (size_t)(op + 0) * C + c,
                    a0_lo, a0_hi, vcs_lo, vcs_hi, vbe_lo, vbe_hi, v_inv_y, v_y_zp, v_qmin, v_qmax);
                dw_requantize_store8_neon(y_nhwc + (size_t)(op + 1) * C + c,
                    a1_lo, a1_hi, vcs_lo, vcs_hi, vbe_lo, vbe_hi, v_inv_y, v_y_zp, v_qmin, v_qmax);
                dw_requantize_store8_neon(y_nhwc + (size_t)(op + 2) * C + c,
                    a2_lo, a2_hi, vcs_lo, vcs_hi, vbe_lo, vbe_hi, v_inv_y, v_y_zp, v_qmin, v_qmax);
                dw_requantize_store8_neon(y_nhwc + (size_t)(op + 3) * C + c,
                    a3_lo, a3_hi, vcs_lo, vcs_hi, vbe_lo, vbe_hi, v_inv_y, v_y_zp, v_qmin, v_qmax);
            }
        }
        // Slow 4-pixel × 8-channel tile: general w_zp. For w_zp=0 this is
        // skipped because the fast path already consumed the 8-ch blocks.
        for (; c + 8 <= C; c += 8) {
            int32x4_t a0_lo = vdupq_n_s32(0), a0_hi = vdupq_n_s32(0);
            int32x4_t a1_lo = vdupq_n_s32(0), a1_hi = vdupq_n_s32(0);
            int32x4_t a2_lo = vdupq_n_s32(0), a2_hi = vdupq_n_s32(0);
            int32x4_t a3_lo = vdupq_n_s32(0), a3_hi = vdupq_n_s32(0);
            for (int k = 0; k < kernel_size; k++) {
                const int8_t* wp = filter + k * C_pad + c;
                int16x8_t ws = vsubq_s16(vmovl_s8(vld1_s8(wp)), v_w_zp_s16);
                int16x4_t ws_lo = vget_low_s16(ws);
                int16x4_t ws_hi = vget_high_s16(ws);

                const uint8_t* const* ind_k = indirection + (size_t)op * kernel_size;
                #define DW_OW4_8CH_PIXEL(P, A_LO, A_HI) do { \
                    uint8x8_t xu = vld1_u8(ind_k[(P) * kernel_size + k] + c); \
                    int16x8_t xs = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(xu)), v_x_zp_s16); \
                    A_LO = vmlal_s16(A_LO, vget_low_s16(xs),  ws_lo); \
                    A_HI = vmlal_s16(A_HI, vget_high_s16(xs), ws_hi); \
                } while (0)
                DW_OW4_8CH_PIXEL(0, a0_lo, a0_hi);
                DW_OW4_8CH_PIXEL(1, a1_lo, a1_hi);
                DW_OW4_8CH_PIXEL(2, a2_lo, a2_hi);
                DW_OW4_8CH_PIXEL(3, a3_lo, a3_hi);
                #undef DW_OW4_8CH_PIXEL
            }
            float32x4_t vcs_lo = vld1q_f32(combined_scale + c);
            float32x4_t vcs_hi = vld1q_f32(combined_scale + c + 4);
            float32x4_t vbias_lo = bias_f ? vld1q_f32(bias_f + c)     : vdupq_n_f32(0.0f);
            float32x4_t vbias_hi = bias_f ? vld1q_f32(bias_f + c + 4) : vdupq_n_f32(0.0f);
            dw_requantize_store8_neon(y_nhwc + (size_t)(op + 0) * C + c,
                a0_lo, a0_hi, vcs_lo, vcs_hi, vbias_lo, vbias_hi, v_inv_y, v_y_zp, v_qmin, v_qmax);
            dw_requantize_store8_neon(y_nhwc + (size_t)(op + 1) * C + c,
                a1_lo, a1_hi, vcs_lo, vcs_hi, vbias_lo, vbias_hi, v_inv_y, v_y_zp, v_qmin, v_qmax);
            dw_requantize_store8_neon(y_nhwc + (size_t)(op + 2) * C + c,
                a2_lo, a2_hi, vcs_lo, vcs_hi, vbias_lo, vbias_hi, v_inv_y, v_y_zp, v_qmin, v_qmax);
            dw_requantize_store8_neon(y_nhwc + (size_t)(op + 3) * C + c,
                a3_lo, a3_hi, vcs_lo, vcs_hi, vbias_lo, vbias_hi, v_inv_y, v_y_zp, v_qmin, v_qmax);
        }
        // 4-pixel × 4-channel tail (fires when C%8==4, e.g. C=12)
        for (; c + 4 <= C; c += 4) {
            int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
            int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
            for (int k = 0; k < kernel_size; k++) {
                const int8_t* wp = filter + k * C_pad + c;
                int32_t wv; std::memcpy(&wv, wp, 4);
                int8x8_t  wu = vreinterpret_s8_s32(vdup_n_s32(wv));
                int16x8_t w16 = vmovl_s8(wu);
                int32x4_t ws = vsubq_s32(vmovl_s16(vget_low_s16(w16)), v_w_zp);

                const uint8_t* const* ind_k = indirection + (size_t)op * kernel_size;
                #define DW_OW4_4CH_PIXEL(P, ACC) do { \
                    uint32_t xv; std::memcpy(&xv, ind_k[(P) * kernel_size + k] + c, 4); \
                    uint8x8_t xu = vreinterpret_u8_u32(vdup_n_u32(xv)); \
                    uint16x8_t x16 = vmovl_u8(xu); \
                    int32x4_t xs = vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(x16))), v_x_zp); \
                    ACC = vmlaq_s32(ACC, xs, ws); \
                } while (0)
                DW_OW4_4CH_PIXEL(0, a0);
                DW_OW4_4CH_PIXEL(1, a1);
                DW_OW4_4CH_PIXEL(2, a2);
                DW_OW4_4CH_PIXEL(3, a3);
                #undef DW_OW4_4CH_PIXEL
            }
            float32x4_t vcs   = vld1q_f32(combined_scale + c);
            float32x4_t vbias = bias_f ? vld1q_f32(bias_f + c) : vdupq_n_f32(0.0f);
            dw_requantize_store4_neon(y_nhwc + (size_t)(op + 0) * C + c, a0, vcs, vbias, v_inv_y, v_y_zp, v_qmin, v_qmax);
            dw_requantize_store4_neon(y_nhwc + (size_t)(op + 1) * C + c, a1, vcs, vbias, v_inv_y, v_y_zp, v_qmin, v_qmax);
            dw_requantize_store4_neon(y_nhwc + (size_t)(op + 2) * C + c, a2, vcs, vbias, v_inv_y, v_y_zp, v_qmin, v_qmax);
            dw_requantize_store4_neon(y_nhwc + (size_t)(op + 3) * C + c, a3, vcs, vbias, v_inv_y, v_y_zp, v_qmin, v_qmax);
        }
        // Scalar C-tail (< 4 channels) for the 4 pixels — rare shape, keep simple.
        for (; c < C; c++) {
            for (int p = 0; p < 4; p++) {
                int32_t acc = 0;
                for (int k = 0; k < kernel_size; k++) {
                    int32_t xv = indirection[(size_t)(op + p) * kernel_size + k][c];
                    int32_t wv = filter[k * C_pad + c];
                    acc += (xv - x_zp) * (wv - w_zp);
                }
                float v = (float)acc * combined_scale[c] + (bias_f ? bias_f[c] : 0.0f);
                float q = v * inv_y_scale;
                q = std::nearbyint(q) + y_zp_f;
                q = std::min(std::max(q, qmin), qmax);
                y_nhwc[(size_t)(op + p) * C + c] = (uint8_t)(int32_t)q;
            }
        }
    }

    // OW=1 tail: remaining 0-3 output pixels (original single-pixel path).
    for (; op < output_count; op++) {
        int c = 0;
        // 8-channel tile
        for (; c + 8 <= C; c += 8) {
            int32x4_t acc_lo = vdupq_n_s32(0);
            int32x4_t acc_hi = vdupq_n_s32(0);
            for (int k = 0; k < kernel_size; k++) {
                const uint8_t* xp = indirection[op * kernel_size + k] + c;
                const int8_t*  wp = filter + k * C_pad + c;
                uint8x8_t xu = vld1_u8(xp);
                int8x8_t  wu = vld1_s8(wp);
                uint16x8_t x16 = vmovl_u8(xu);
                int16x8_t  w16 = vmovl_s8(wu);
                int32x4_t xs_lo = vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(x16))), v_x_zp);
                int32x4_t xs_hi = vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(x16))), v_x_zp);
                int32x4_t ws_lo = vsubq_s32(vmovl_s16(vget_low_s16(w16)), v_w_zp);
                int32x4_t ws_hi = vsubq_s32(vmovl_s16(vget_high_s16(w16)), v_w_zp);
                acc_lo = vmlaq_s32(acc_lo, xs_lo, ws_lo);
                acc_hi = vmlaq_s32(acc_hi, xs_hi, ws_hi);
            }
            float32x4_t vcs_lo = vld1q_f32(combined_scale + c);
            float32x4_t vcs_hi = vld1q_f32(combined_scale + c + 4);
            float32x4_t vbias_lo = bias_f ? vld1q_f32(bias_f + c) : vdupq_n_f32(0.0f);
            float32x4_t vbias_hi = bias_f ? vld1q_f32(bias_f + c + 4) : vdupq_n_f32(0.0f);
            dw_requantize_store8_neon(y_nhwc + (size_t)op * C + c,
                acc_lo, acc_hi, vcs_lo, vcs_hi, vbias_lo, vbias_hi, v_inv_y, v_y_zp, v_qmin, v_qmax);
        }
        // 4-channel tail
        for (; c + 4 <= C; c += 4) {
            int32x4_t acc = vdupq_n_s32(0);
            for (int k = 0; k < kernel_size; k++) {
                const uint8_t* xp = indirection[op * kernel_size + k] + c;
                const int8_t*  wp = filter + k * C_pad + c;
                uint32_t xv; std::memcpy(&xv, xp, 4);
                int32_t  wv; std::memcpy(&wv, wp, 4);
                uint8x8_t xu = vreinterpret_u8_u32(vdup_n_u32(xv));
                int8x8_t  wu = vreinterpret_s8_s32(vdup_n_s32(wv));
                uint16x8_t x16 = vmovl_u8(xu);
                int16x8_t  w16 = vmovl_s8(wu);
                int32x4_t xs = vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(x16))), v_x_zp);
                int32x4_t ws = vsubq_s32(vmovl_s16(vget_low_s16(w16)), v_w_zp);
                acc = vmlaq_s32(acc, xs, ws);
            }
            float32x4_t vcs = vld1q_f32(combined_scale + c);
            float32x4_t vbias = bias_f ? vld1q_f32(bias_f + c) : vdupq_n_f32(0.0f);
            dw_requantize_store4_neon(y_nhwc + (size_t)op * C + c,
                acc, vcs, vbias, v_inv_y, v_y_zp, v_qmin, v_qmax);
        }
        // Scalar tail (< 4 channels)
        for (; c < C; c++) {
            int32_t acc = 0;
            for (int k = 0; k < kernel_size; k++) {
                int32_t xv = indirection[op * kernel_size + k][c];
                int32_t wv = filter[k * C_pad + c];
                acc += (xv - x_zp) * (wv - w_zp);
            }
            float v = (float)acc * combined_scale[c] + (bias_f ? bias_f[c] : 0.0f);
            float q = v * inv_y_scale;
            q = std::nearbyint(q) + y_zp_f;
            q = std::min(std::max(q, qmin), qmax);
            y_nhwc[(size_t)op * C + c] = (uint8_t)(int32_t)q;
        }
    }
}

// Threaded wrapper — splits output pixels across rows.
inline void depthwise_int8_nhwc_neon_mt(
    uint8_t* y_nhwc,
    const uint8_t* const* indirection,
    const int8_t* filter,
    const float* combined_scale,
    const float* bias_f,
    int C, int oH, int oW, int kernel_size,
    int x_zp, int w_zp,
    float inv_y_scale, float y_zp_f, float qmin, float qmax)
{
    bool par = oH >= 4 && C >= 16;
    nnr::for_static(0, oH, par, [&](int oh) {
        int op_start = oh * oW;
        depthwise_int8_nhwc_neon(
            y_nhwc + (size_t)op_start * C,
            indirection + (size_t)op_start * kernel_size,
            filter, combined_scale, bias_f,
            C, oW, kernel_size,
            x_zp, w_zp,
            inv_y_scale, y_zp_f, qmin, qmax);
    });
}

} // namespace nnr::int8::neon

#endif // aarch64
