#pragma once
// ARM NEON depthwise 2D convolution backend.
// 128-bit (4 floats per vector) counterpart of x64/depthwise_avx2.h.

#if defined(__aarch64__) || defined(_M_ARM64)

#ifdef _MSC_VER
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif
#include <algorithm>
#include "thread_pool.h"

namespace nnr::neon {

#if 0 // LOCUST
;empty_init = "{}"
;for S in [1, 2]:
;    suffix = "" if S == 1 else "_s2"

;    if S == 1:
// NEON float depthwise conv, stride=1, dilation=1 specialization.
// Processes 4 output pixels per iteration.
// Boundary handling: scalar gather into temp buffer (NEON lacks masked loads).
;        pass
;    else:
// NEON float depthwise conv, stride=2, dilation=1 specialization.
// 4 output elements need input at stride-2 positions: iw, iw+2, iw+4, iw+6.
// Interior: use vld2q_f32 to deinterleave even/odd indices from 8 contiguous floats.
// Boundary: scalar gather.
;        pass
inline bool depthwise_2d@suffix@(
    tensor_t* y, const tensor_t* x, const tensor_t* w, float* bias,
    int pH, int pW,
    operator_t::post_fn_t post_fn, const operator_t* fused_op)
{
    const int kH = w->dims[2], kW = w->dims[3];
    const int iH = x->dims[2], iW = x->dims[3];
    const int oN = y->dims[0], oC = y->dims[1], oH = y->dims[2], oW = y->dims[3];
    const float* xd = (const float*)x->data;
    float* yd = (float*)y->data;
    const float* wd = (const float*)w->data;

    const int iC = x->dims[1];
    nnr::for_static(0, oN * oC, oN * oC > 4, [&](int nc) {
        int n = nc / oC, c = nc % oC;
        int ic = (int)((size_t)c * iC / oC);
        const float* xc = xd + ((size_t)n * iC + ic) * iH * iW;
        const float* wc = wd + (size_t)c * kH * kW;
        float* yc = yd + ((size_t)n * oC + c) * oH * oW;
        float32x4_t vbias = vdupq_n_f32(bias ? bias[c] : 0.0f);
        for (int oh = 0; oh < oH; ++oh) {
;    if S == 1:
            int ih0 = oh - pH;
;        pass
;    else:
            int ih0 = oh * 2 - pH;
;        pass
            int ow = 0;
            for (; ow + 4 <= oW; ow += 4) {
;    if S == 1:
                int iw0 = ow - pW;
;        pass
;    else:
                int iw0 = ow * 2 - pW;
;        pass
                float32x4_t acc = vbias;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = ih0 + kh;
                    if (ih < 0 || ih >= iH) continue;
                    const float* xrow = xc + ih * iW;
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = iw0 + kw;
                        float32x4_t wv = vdupq_n_f32(wc[kh * kW + kw]);
;    if S == 1:
                        if (iw >= 0 && iw + 3 < iW) {
                            acc = vfmaq_f32(acc, wv, vld1q_f32(xrow + iw));
                        } else {
                            // Boundary: scalar gather into temp buffer
                            alignas(16) float tmp[4] = @empty_init@;
                            for (int p = 0; p < 4; ++p) {
                                int iw_p = iw + p;
                                if (iw_p >= 0 && iw_p < iW)
                                    tmp[p] = xrow[iw_p];
                            }
                            acc = vfmaq_f32(acc, wv, vld1q_f32(tmp));
                        }
;        pass
;    else:
                        // Need input at positions: iw, iw+2, iw+4, iw+6
                        if (iw >= 0 && iw + 7 < iW) {
                            // Interior: load 8 contiguous, deinterleave to get even indices
                            float32x4x2_t pair = vld2q_f32(xrow + iw);
                            // pair.val[0] = even-indexed elements: xrow[iw], xrow[iw+2], xrow[iw+4], xrow[iw+6]
                            acc = vfmaq_f32(acc, wv, pair.val[0]);
                        } else {
                            // Boundary: scalar gather into temp
                            alignas(16) float tmp[4] = @empty_init@;
                            for (int p = 0; p < 4; ++p) {
                                int iw_p = iw + p * 2;
                                if (iw_p >= 0 && iw_p < iW)
                                    tmp[p] = xrow[iw_p];
                            }
                            acc = vfmaq_f32(acc, wv, vld1q_f32(tmp));
                        }
;        pass
                    }
                }
                vst1q_f32(yc + oh * oW + ow, acc);
            }
            // Scalar tail
            for (; ow < oW; ++ow) {
;    if S == 1:
                int iw0 = ow - pW;
;        pass
;    else:
                int iw0 = ow * 2 - pW;
;        pass
                float sum = bias ? bias[c] : 0.0f;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = ih0 + kh;
                    if (ih < 0 || ih >= iH) continue;
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = iw0 + kw;
                        if (iw < 0 || iw >= iW) continue;
                        sum += xc[ih * iW + iw] * wc[kh * kW + kw];
                    }
                }
                yc[oh * oW + ow] = sum;
            }
        }
        if (post_fn) {
            int toff = (int)(yc - yd);
            post_fn(yc, 1, oH * oW, oH * oW, fused_op, nullptr, toff);
        }
    });
    return true;
}

;    pass
#else // LOCUST

// NEON float depthwise conv, stride=1, dilation=1 specialization.
// Processes 4 output pixels per iteration.
// Boundary handling: scalar gather into temp buffer (NEON lacks masked loads).
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=DW fusion=post_op
inline bool depthwise_2d(
    tensor_t* y, const tensor_t* x, const tensor_t* w, float* bias,
    int pH, int pW,
    operator_t::post_fn_t post_fn, const operator_t* fused_op)
{
    const int kH = w->dims[2], kW = w->dims[3];
    const int iH = x->dims[2], iW = x->dims[3];
    const int oN = y->dims[0], oC = y->dims[1], oH = y->dims[2], oW = y->dims[3];
    const float* xd = (const float*)x->data;
    float* yd = (float*)y->data;
    const float* wd = (const float*)w->data;

    const int iC = x->dims[1];
    nnr::for_static(0, oN * oC, oN * oC > 4, [&](int nc) {
        int n = nc / oC, c = nc % oC;
        int ic = (int)((size_t)c * iC / oC);
        const float* xc = xd + ((size_t)n * iC + ic) * iH * iW;
        const float* wc = wd + (size_t)c * kH * kW;
        float* yc = yd + ((size_t)n * oC + c) * oH * oW;
        float32x4_t vbias = vdupq_n_f32(bias ? bias[c] : 0.0f);
        // 3×3 specialization: 3 independent row accumulators hide FMLA latency.
        // Row-pointer-to-zeros eliminates per-kh boundary branches.
        if (kH == 3 && kW == 3) {
            alignas(16) static const float zeros[256] = {};
            const float* zp = zeros;  // invalid rows point here
            for (int oh = 0; oh < oH; ++oh) {
                int ih0 = oh - pH;
                const float* r0 = (ih0     >= 0 && ih0     < iH) ? xc + ih0       * iW - pW : zp;
                const float* r1 = (ih0 + 1 >= 0 && ih0 + 1 < iH) ? xc + (ih0 + 1) * iW - pW : zp;
                const float* r2 = (ih0 + 2 >= 0 && ih0 + 2 < iH) ? xc + (ih0 + 2) * iW - pW : zp;
                int ow = 0;
                for (; ow + 4 <= oW; ow += 4) {
                    int iw0 = ow;  // already offset by -pW in row pointers
                    float32x4_t a0 = vbias;
                    float32x4_t a1 = vdupq_n_f32(0);
                    float32x4_t a2 = vdupq_n_f32(0);
                    for (int kw = 0; kw < 3; ++kw) {
                        int iw = iw0 + kw;
                        // Row pointers have -pW baked in, so the contiguous 4-wide load from
                        // r*+iw covers actual columns [iw-pW, iw-pW+3]; bounds check them directly.
                        int iw_in = iw - pW;
                        float32x4_t w0 = vdupq_n_f32(wc[0 + kw]);
                        float32x4_t w1 = vdupq_n_f32(wc[3 + kw]);
                        float32x4_t w2 = vdupq_n_f32(wc[6 + kw]);
                        if (iw_in >= 0 && iw_in + 3 < iW) {
                            a0 = vfmaq_f32(a0, w0, vld1q_f32(r0 + iw));
                            a1 = vfmaq_f32(a1, w1, vld1q_f32(r1 + iw));
                            a2 = vfmaq_f32(a2, w2, vld1q_f32(r2 + iw));
                        } else {
                            alignas(16) float t0[4] = {}, t1[4] = {}, t2[4] = {};
                            for (int p = 0; p < 4; ++p) {
                                int iwp = iw_in + p;
                                if (iwp >= 0 && iwp < iW) {
                                    t0[p] = r0[iw + p]; t1[p] = r1[iw + p]; t2[p] = r2[iw + p];
                                }
                            }
                            a0 = vfmaq_f32(a0, w0, vld1q_f32(t0));
                            a1 = vfmaq_f32(a1, w1, vld1q_f32(t1));
                            a2 = vfmaq_f32(a2, w2, vld1q_f32(t2));
                        }
                    }
                    vst1q_f32(yc + oh * oW + ow, vaddq_f32(vaddq_f32(a0, a1), a2));
                }
                for (; ow < oW; ++ow) {
                    int iw0 = ow - pW;
                    float sum = bias ? bias[c] : 0.0f;
                    for (int kh = 0; kh < 3; ++kh) {
                        int ih = ih0 + kh;
                        if (ih < 0 || ih >= iH) continue;
                        for (int kw = 0; kw < 3; ++kw) {
                            int iw = iw0 + kw;
                            if (iw < 0 || iw >= iW) continue;
                            sum += xc[ih * iW + iw] * wc[kh * kW + kw];
                        }
                    }
                    yc[oh * oW + ow] = sum;
                }
            }
        } else {
        // Generic kH×kW path (original)
        for (int oh = 0; oh < oH; ++oh) {
            int ih0 = oh - pH;
            int ow = 0;
            for (; ow + 4 <= oW; ow += 4) {
                int iw0 = ow - pW;
                float32x4_t acc = vbias;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = ih0 + kh;
                    if (ih < 0 || ih >= iH) continue;
                    const float* xrow = xc + ih * iW;
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = iw0 + kw;
                        float32x4_t wv = vdupq_n_f32(wc[kh * kW + kw]);
                        if (iw >= 0 && iw + 3 < iW) {
                            acc = vfmaq_f32(acc, wv, vld1q_f32(xrow + iw));
                        } else {
                            // Boundary: scalar gather into temp buffer
                            alignas(16) float tmp[4] = {};
                            for (int p = 0; p < 4; ++p) {
                                int iw_p = iw + p;
                                if (iw_p >= 0 && iw_p < iW)
                                    tmp[p] = xrow[iw_p];
                            }
                            acc = vfmaq_f32(acc, wv, vld1q_f32(tmp));
                        }
                    }
                }
                vst1q_f32(yc + oh * oW + ow, acc);
            }
            // Scalar tail
            for (; ow < oW; ++ow) {
                int iw0 = ow - pW;
                float sum = bias ? bias[c] : 0.0f;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = ih0 + kh;
                    if (ih < 0 || ih >= iH) continue;
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = iw0 + kw;
                        if (iw < 0 || iw >= iW) continue;
                        sum += xc[ih * iW + iw] * wc[kh * kW + kw];
                    }
                }
                yc[oh * oW + ow] = sum;
            }
        }
        } // end generic
        if (post_fn) {
            int toff = (int)(yc - yd);
            post_fn(yc, 1, oH * oW, oH * oW, fused_op, nullptr, toff);
        }
    });
    return true;
}


// NEON float depthwise conv, stride=2, dilation=1 specialization.
// 4 output elements need input at stride-2 positions: iw, iw+2, iw+4, iw+6.
// Interior: use vld2q_f32 to deinterleave even/odd indices from 8 contiguous floats.
// Boundary: scalar gather.
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=DW fusion=post_op
inline bool depthwise_2d_s2(
    tensor_t* y, const tensor_t* x, const tensor_t* w, float* bias,
    int pH, int pW,
    operator_t::post_fn_t post_fn, const operator_t* fused_op)
{
    const int kH = w->dims[2], kW = w->dims[3];
    const int iH = x->dims[2], iW = x->dims[3];
    const int oN = y->dims[0], oC = y->dims[1], oH = y->dims[2], oW = y->dims[3];
    const float* xd = (const float*)x->data;
    float* yd = (float*)y->data;
    const float* wd = (const float*)w->data;

    const int iC = x->dims[1];
    nnr::for_static(0, oN * oC, oN * oC > 4, [&](int nc) {
        int n = nc / oC, c = nc % oC;
        int ic = (int)((size_t)c * iC / oC);
        const float* xc = xd + ((size_t)n * iC + ic) * iH * iW;
        const float* wc = wd + (size_t)c * kH * kW;
        float* yc = yd + ((size_t)n * oC + c) * oH * oW;
        float32x4_t vbias = vdupq_n_f32(bias ? bias[c] : 0.0f);
        for (int oh = 0; oh < oH; ++oh) {
            int ih0 = oh * 2 - pH;
            int ow = 0;
            for (; ow + 4 <= oW; ow += 4) {
                int iw0 = ow * 2 - pW;
                float32x4_t acc = vbias;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = ih0 + kh;
                    if (ih < 0 || ih >= iH) continue;
                    const float* xrow = xc + ih * iW;
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = iw0 + kw;
                        float32x4_t wv = vdupq_n_f32(wc[kh * kW + kw]);
                        // Need input at positions: iw, iw+2, iw+4, iw+6
                        if (iw >= 0 && iw + 7 < iW) {
                            // Interior: load 8 contiguous, deinterleave to get even indices
                            float32x4x2_t pair = vld2q_f32(xrow + iw);
                            // pair.val[0] = even-indexed elements: xrow[iw], xrow[iw+2], xrow[iw+4], xrow[iw+6]
                            acc = vfmaq_f32(acc, wv, pair.val[0]);
                        } else {
                            // Boundary: scalar gather into temp
                            alignas(16) float tmp[4] = {};
                            for (int p = 0; p < 4; ++p) {
                                int iw_p = iw + p * 2;
                                if (iw_p >= 0 && iw_p < iW)
                                    tmp[p] = xrow[iw_p];
                            }
                            acc = vfmaq_f32(acc, wv, vld1q_f32(tmp));
                        }
                    }
                }
                vst1q_f32(yc + oh * oW + ow, acc);
            }
            // Scalar tail
            for (; ow < oW; ++ow) {
                int iw0 = ow * 2 - pW;
                float sum = bias ? bias[c] : 0.0f;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = ih0 + kh;
                    if (ih < 0 || ih >= iH) continue;
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = iw0 + kw;
                        if (iw < 0 || iw >= iW) continue;
                        sum += xc[ih * iW + iw] * wc[kh * kW + kw];
                    }
                }
                yc[oh * oW + ow] = sum;
            }
        }
        if (post_fn) {
            int toff = (int)(yc - yd);
            post_fn(yc, 1, oH * oW, oH * oW, fused_op, nullptr, toff);
        }
    });
    return true;
}

#endif // LOCUST

} // namespace nnr::neon

#endif // __aarch64__ || _M_ARM64
