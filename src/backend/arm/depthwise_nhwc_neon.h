#pragma once
// ARM NEON NHWC depthwise convolution: per-pixel channel loop.
// Called from Conv_depthwise.h.

#ifdef NNR_ARCH_ARM64
#include <arm_neon.h>

namespace nnr {

// Process one output pixel's channels for NHWC depthwise conv.
// Returns the channel position after vectorized processing.
// @nnr-meta isa=NEON dtype=fp32 layout=NHWC special=DW
inline int depthwise_nhwc_pixel_neon(
    float* out, const float* xn, const float* w_repacked, const float* bias,
    int C, int kH, int kW, int iH, int iW, int dH, int dW,
    int ih0, int iw0)
{
    int c = 0;
    for (; c + 16 <= C; c += 16) {
        float32x4_t acc0 = bias ? vld1q_f32(bias + c)      : vdupq_n_f32(0);
        float32x4_t acc1 = bias ? vld1q_f32(bias + c + 4)  : vdupq_n_f32(0);
        float32x4_t acc2 = bias ? vld1q_f32(bias + c + 8)  : vdupq_n_f32(0);
        float32x4_t acc3 = bias ? vld1q_f32(bias + c + 12) : vdupq_n_f32(0);
        for (int kh = 0; kh < kH; kh++) {
            int ih = ih0 + kh * dH;
            if (ih < 0 || ih >= iH) continue;
            for (int kw = 0; kw < kW; kw++) {
                int iw = iw0 + kw * dW;
                if (iw < 0 || iw >= iW) continue;
                size_t xoff = (size_t)(ih * iW + iw) * C + c;
                size_t woff = (size_t)(kh * kW + kw) * C + c;
                acc0 = vfmaq_f32(acc0, vld1q_f32(xn + xoff),      vld1q_f32(w_repacked + woff));
                acc1 = vfmaq_f32(acc1, vld1q_f32(xn + xoff + 4),  vld1q_f32(w_repacked + woff + 4));
                acc2 = vfmaq_f32(acc2, vld1q_f32(xn + xoff + 8),  vld1q_f32(w_repacked + woff + 8));
                acc3 = vfmaq_f32(acc3, vld1q_f32(xn + xoff + 12), vld1q_f32(w_repacked + woff + 12));
            }
        }
        vst1q_f32(out + c,      acc0);
        vst1q_f32(out + c + 4,  acc1);
        vst1q_f32(out + c + 8,  acc2);
        vst1q_f32(out + c + 12, acc3);
    }
    for (; c + 4 <= C; c += 4) {
        float32x4_t acc = bias ? vld1q_f32(bias + c) : vdupq_n_f32(0);
        for (int kh = 0; kh < kH; kh++) {
            int ih = ih0 + kh * dH;
            if (ih < 0 || ih >= iH) continue;
            for (int kw = 0; kw < kW; kw++) {
                int iw = iw0 + kw * dW;
                if (iw < 0 || iw >= iW) continue;
                acc = vfmaq_f32(acc,
                    vld1q_f32(xn + (ih * iW + iw) * C + c),
                    vld1q_f32(w_repacked + (kh * kW + kw) * C + c));
            }
        }
        vst1q_f32(out + c, acc);
    }
    return c;
}

} // namespace nnr

#endif // NNR_ARCH_ARM64
