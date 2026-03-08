#pragma once
// ARM NEON inline direct convolution for scroll-tiling strip processing.
// Called from Conv.cpp exec_strip for small-channel NCHW paths.

#ifdef NNR_ARCH_ARM64
#include <arm_neon.h>
#include <algorithm>

namespace nnr {

// Direct conv strip: compute output rows [out_row_start, out_end) for one channel.
// xc: input channel [iH, iW], wc: weights [kH, kW], yc: output strip.
// Handles stride 1 and 2 with NEON vectorization in the safe interior.
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=Direct tiling=spatial fusion=post_op
inline void conv_direct_strip_neon(
    float* yc, const float* xc, const float* wc, float bv,
    int kH, int kW, int iH, int iW, int oW,
    int sH, int sW, int pH, int pW, int dH, int dW,
    int out_row_start, int out_end)
{
    int ow_safe_lo = pW > 0 ? (pW + sW - 1) / sW : 0;
    int ow_safe_hi = std::min(oW, (iW + pW - kW) / sW + 1);
    float32x4_t vbias = vdupq_n_f32(bv);
    for (int oh = out_row_start; oh < out_end; ++oh) {
        int ih0 = oh * sH - pH;
        int kh0 = std::max(0, -ih0), kh1 = std::min(kH, iH - ih0);
        float* dst = yc + (oh - out_row_start) * oW;
        // Left edge (scalar)
        for (int ow = 0; ow < ow_safe_lo && ow < oW; ++ow) {
            int iw0 = ow * sW - pW;
            int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
            float sum = bv;
            for (int kh = kh0; kh < kh1; ++kh)
                for (int kw = kw0; kw < kw1; ++kw)
                    sum += xc[(ih0 + kh) * iW + (iw0 + kw)] * wc[kh * kW + kw];
            dst[ow] = sum;
        }
        // Interior (NEON)
        int ow = ow_safe_lo;
        if (sW == 1) {
            for (; ow + 4 <= ow_safe_hi; ow += 4) {
                int iw0 = ow - pW;
                float32x4_t vsum = vbias;
                for (int kh = kh0; kh < kh1; ++kh) {
                    const float* row = xc + (ih0 + kh) * iW + iw0;
                    for (int kw = 0; kw < kW; ++kw)
                        vsum = vfmaq_n_f32(vsum, vld1q_f32(row + kw), wc[kh * kW + kw]);
                }
                vst1q_f32(dst + ow, vsum);
            }
        } else if (sW == 2) {
            for (; ow + 4 <= ow_safe_hi; ow += 4) {
                int iw0 = ow * 2 - pW;
                float32x4_t vsum = vbias;
                for (int kh = kh0; kh < kh1; ++kh) {
                    const float* row = xc + (ih0 + kh) * iW + iw0;
                    for (int kw = 0; kw < kW; ++kw) {
                        float32x4x2_t pairs = vld2q_f32(row + kw);
                        vsum = vfmaq_n_f32(vsum, pairs.val[0], wc[kh * kW + kw]);
                    }
                }
                vst1q_f32(dst + ow, vsum);
            }
        }
        // Right edge (scalar)
        for (; ow < oW; ++ow) {
            int iw0 = ow * sW - pW;
            int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
            float sum = bv;
            for (int kh = kh0; kh < kh1; ++kh)
                for (int kw = kw0; kw < kw1; ++kw)
                    sum += xc[(ih0 + kh) * iW + (iw0 + kw)] * wc[kh * kW + kw];
            dst[ow] = sum;
        }
    }
}

} // namespace nnr

#endif // NNR_ARCH_ARM64
