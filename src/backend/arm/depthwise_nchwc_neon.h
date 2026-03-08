#pragma once
// ARM NEON NCHW8c depthwise 2D convolution: per-pixel kernel.
//
// Counterpart of x64/depthwise_nhwc_x64.h::depthwise_nchwc_pixel_avx512.
// Called from Conv_depthwise.h::exec_depthwise_2d_nchwc.
//
// Block width is 8 on ARM (NCHW8c): each output pixel owns 8 channels laid
// out contiguously. 8 floats = 2 NEON qregs, so a single output pixel uses
// two float32x4_t accumulators. The caller loops oh×ow.
//
// This is M2 of the ARM NCHWc plan. M1 shipped the 1×1 pointwise kernel
// in conv_nchwc_neon.h; this file adds the DW kernel needed for mobilenet
// chains.

#ifdef NNR_ARCH_ARM64

#ifdef _MSC_VER
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif

namespace nnr {

// NCHWc depthwise: process one output pixel with block=8 (NEON).
// inp: [iH, iW, 8] for this channel block
// wt:  [kH, kW, 8] for this channel block
// bi:  [8] bias for this channel block
// out: [oH, oW, 8] for this channel block
// @nnr-meta isa=NEON dtype=fp32 layout=BLOCKED_8 special=[DW,NCHWc]
inline void depthwise_nchwc_pixel_neon(
    float* out, const float* inp, const float* wt, const float* bi,
    int kH, int kW, int iH, int iW, int oW, int block,
    int oh, int ow, int sH, int sW, int pH, int pW)
{
    // block is always 8 on ARM NCHW8c — parameter kept for API parity with
    // the x64 AVX-512 (block=16) entry point.
    float32x4_t acc_lo = vld1q_f32(bi);
    float32x4_t acc_hi = vld1q_f32(bi + 4);
    const int ih0 = oh * sH - pH;
    const int iw0 = ow * sW - pW;
    for (int kh = 0; kh < kH; kh++) {
        int ih = ih0 + kh;
        if (ih < 0 || ih >= iH) continue;
        for (int kw = 0; kw < kW; kw++) {
            int iw = iw0 + kw;
            if (iw < 0 || iw >= iW) continue;
            const float* xp = &inp[((size_t)ih * iW + iw) * block];
            const float* wp = &wt[((size_t)kh * kW + kw) * block];
            acc_lo = vfmaq_f32(acc_lo, vld1q_f32(xp),     vld1q_f32(wp));
            acc_hi = vfmaq_f32(acc_hi, vld1q_f32(xp + 4), vld1q_f32(wp + 4));
        }
    }
    float* op = &out[((size_t)oh * oW + ow) * block];
    vst1q_f32(op,     acc_lo);
    vst1q_f32(op + 4, acc_hi);
}

} // namespace nnr

#endif // NNR_ARCH_ARM64
