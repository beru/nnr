#pragma once
// NHWC-layout depthwise convolution for FP16 via NEON widening FMA
// (`vfmlalq_{low,high}_f16`), FP32 accumulator, FP32 output.
//
// Depthwise: output[n, oh, ow, c] = Σ_kh Σ_kw X[n, ih, iw, c] · W[c, kh, kw]
//                                  (+ bias[c]; channel-independent, M = C).
//
// Layout:
//   x:  NHWC [N, iH, iW, C] uint16 (FP16 bit pattern), caller supplies; this
//       kernel does inline boundary checks rather than pre-padding (matches
//       depthwise_nhwc_neon.h; pre-padding extra C channels has no win since
//       depthwise has per-channel weights and no cross-channel reuse).
//   w:  repacked [kH*kW, C] uint16 (row-major, channel-contiguous per kernel
//       slot; same convention as depthwise_nhwc_neon.h's FP32 w_repacked).
//   y:  NHWC [N, oH, oW, C] float32 — caller narrows to FP16 if needed.
//   bias: [C] FP32 (optional; may be nullptr).
//
// Tile: C=8 per step for the vector body, plus a C-tail scalar fallback for
// 1..7 remaining channels. Bias is loaded vectorised when available.
//
// Runtime-gated by `has_neon_fp16()`.

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

// Repack OIHW [C, 1, kH, kW] FP16 depthwise weights into [kH*kW, C] layout.
inline size_t repack_weights_depthwise_fp16_nhwc_size(int C, int kH, int kW)
{
    return (size_t)kH * kW * C * sizeof(uint16_t);
}

inline void repack_weights_depthwise_fp16_nhwc(
    uint16_t* __restrict dst,
    const uint16_t* __restrict src,  // OIHW [C, 1, kH, kW]
    int C, int kH, int kW)
{
    int kSpatial = kH * kW;
    for (int c = 0; c < C; c++)
        for (int ks = 0; ks < kSpatial; ks++)
            dst[(size_t)ks * C + c] = src[(size_t)c * kSpatial + ks];
}

// One output pixel of a depthwise conv over channels 0..C-1.
//   out:  [C] float32 destination
//   xn:   [iH, iW, C] FP16 input (one batch)
//   w:    [kH*kW, C] FP16 repacked weights
//   bias: [C] FP32 (nullable)
//   ih0:  oh * sH - pH_begin
//   iw0:  ow * sW - pW_begin
inline void depthwise_fp16_nhwc_pixel_neon(
    float* __restrict out,
    const uint16_t* __restrict xn,
    const uint16_t* __restrict w,
    const float* __restrict bias,
    int C, int kH, int kW, int iH, int iW, int dH, int dW,
    int ih0, int iw0)
{
    int c = 0;
    for (; c + 8 <= C; c += 8) {
        float32x4_t acc_lo = bias ? vld1q_f32(bias + c)     : vdupq_n_f32(0);
        float32x4_t acc_hi = bias ? vld1q_f32(bias + c + 4) : vdupq_n_f32(0);
        for (int kh = 0; kh < kH; kh++) {
            int ih = ih0 + kh * dH;
            if (ih < 0 || ih >= iH) continue;
            for (int kw = 0; kw < kW; kw++) {
                int iw = iw0 + kw * dW;
                if (iw < 0 || iw >= iW) continue;
                float16x8_t x8 = vreinterpretq_f16_u16(vld1q_u16(
                    xn + (size_t)(ih * iW + iw) * C + c));
                float16x8_t w8 = vreinterpretq_f16_u16(vld1q_u16(
                    w  + (size_t)(kh * kW + kw) * C + c));
                acc_lo = vfmlalq_low_f16 (acc_lo, x8, w8);
                acc_hi = vfmlalq_high_f16(acc_hi, x8, w8);
            }
        }
        vst1q_f32(out + c + 0, acc_lo);
        vst1q_f32(out + c + 4, acc_hi);
    }
    // C-tail: 1..7 scalar channels.  Depthwise weights are small so an
    // FP32 scalar loop here is cheap.
    for (; c < C; c++) {
        float acc = bias ? bias[c] : 0.0f;
        for (int kh = 0; kh < kH; kh++) {
            int ih = ih0 + kh * dH;
            if (ih < 0 || ih >= iH) continue;
            for (int kw = 0; kw < kW; kw++) {
                int iw = iw0 + kw * dW;
                if (iw < 0 || iw >= iW) continue;
                uint16_t xu = xn[(size_t)(ih * iW + iw) * C + c];
                uint16_t wu = w [(size_t)(kh * kW + kw) * C + c];
                float16x4_t xv = vreinterpret_f16_u16(vdup_n_u16(xu));
                float16x4_t wv = vreinterpret_f16_u16(vdup_n_u16(wu));
                // Scalar widen+FMA via the 4-wide low variant; lane 0 holds the result.
                float32x2_t r = vfmlal_low_f16(vdup_n_f32(acc), xv, wv);
                acc = vget_lane_f32(r, 0);
            }
        }
        out[c] = acc;
    }
}

// Full depthwise exec (single batch; batch loop is caller's responsibility).
//   y:       [oH, oW, C] float
//   xn:      [iH, iW, C] FP16
//   w:       [kH*kW, C] FP16 repacked (from repack_weights_depthwise_fp16_nhwc)
//   bias:    [C] FP32 (nullable)
//   pH_b/pH_e, pW_b/pW_e: begin/end spatial pads
inline bool depthwise_fp16_nhwc_neon(
    float* __restrict y,
    const uint16_t* __restrict xn,
    const uint16_t* __restrict w,
    const float* __restrict bias,
    int C, int iH, int iW,
    int oH, int oW,
    int kH, int kW, int sH, int sW,
    int dH, int dW,
    int pH_b, int pW_b)
{
    if (!has_neon_fp16()) return false;
    nnr::for_static(0, oH, oH >= 4, [&](int oh) {
        int ih0 = oh * sH - pH_b;
        for (int ow = 0; ow < oW; ow++) {
            int iw0 = ow * sW - pW_b;
            float* out = y + (size_t)(oh * oW + ow) * C;
            depthwise_fp16_nhwc_pixel_neon(
                out, xn, w, bias,
                C, kH, kW, iH, iW, dH, dW,
                ih0, iw0);
        }
    });
    return true;
}

} // namespace nnr::fp16::neon

#endif // aarch64
