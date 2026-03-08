#pragma once
// AVX-512 fast paths for depthwise direct convolution (single-channel strip).
// Called from Conv_operator::depthwise_strip_channel() for scroll execution.
//
// Two specializations:
//   - Stride-1, dilation-1: vectorized 16-wide output with masked boundary loads
//   - Stride-2, dilation-1: permutex2var gather for strided input access

#include <immintrin.h>
#include "cpu_features.h"

namespace nnr {

struct operator_t;  // forward declaration

// Returns true if handled by AVX-512 fast path, false to fall through to scalar.
// post_fn signature matches operator_t::post_fn_t.
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=DW fusion=post_op
inline bool depthwise_strip_channel_avx512(
    const float* xc, const float* wc, float* yc, float bv,
    int kH, int kW, int iH, int iW, int oW,
    int sH, int sW, int dH, int dW, int pH, int pW,
    int out_row_start, int out_end, int spatial_strip,
    void (*post_fn)(float*, int, int, int, const operator_t*, const float*, int),
    const operator_t* fused_op, int tensor_offset)
{
    if (!has_avx512()) return false;

    // Stride-1, dilation-1 fast path
    if (dH == 1 && dW == 1 && sH == 1 && sW == 1) {
        __m512 vbias = _mm512_set1_ps(bv);
        for (int oh = out_row_start; oh < out_end; ++oh) {
            int ih0 = oh - pH;
            int ow = 0;
            for (; ow + 16 <= oW; ow += 16) {
                int iw0 = ow - pW;
                __m512 acc = vbias;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = ih0 + kh;
                    if (ih < 0 || ih >= iH) continue;
                    const float* xrow = xc + ih * iW;
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = iw0 + kw;
                        if (iw >= 0 && iw + 15 < iW) {
                            acc = _mm512_fmadd_ps(
                                _mm512_set1_ps(wc[kh * kW + kw]),
                                _mm512_loadu_ps(xrow + iw), acc);
                        } else {
                            __mmask16 mask = 0;
                            for (int p = 0; p < 16; ++p) {
                                int iw_p = iw + p;
                                if (iw_p >= 0 && iw_p < iW)
                                    mask |= (1 << p);
                            }
                            acc = _mm512_fmadd_ps(
                                _mm512_set1_ps(wc[kh * kW + kw]),
                                _mm512_maskz_loadu_ps(mask, xrow + iw), acc);
                        }
                    }
                }
                _mm512_storeu_ps(yc + (oh - out_row_start) * oW + ow, acc);
            }
            for (; ow < oW; ++ow) {
                int iw0 = ow - pW;
                float sum = bv;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = ih0 + kh;
                    if (ih < 0 || ih >= iH) continue;
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = iw0 + kw;
                        if (iw < 0 || iw >= iW) continue;
                        sum += xc[ih * iW + iw] * wc[kh * kW + kw];
                    }
                }
                yc[(oh - out_row_start) * oW + ow] = sum;
            }
        }
        if (post_fn) post_fn(yc, 1, spatial_strip, spatial_strip, fused_op, nullptr, tensor_offset);
        return true;
    }

    // Stride-2, dilation-1 fast path
    if (dH == 1 && dW == 1 && sH == 2 && sW == 2) {
        const __m512i perm_s2 = _mm512_set_epi32(
            30, 28, 26, 24, 22, 20, 18, 16,
            14, 12, 10, 8, 6, 4, 2, 0);
        const __m512i gather_s2 = _mm512_set_epi32(
            30, 28, 26, 24, 22, 20, 18, 16,
            14, 12, 10, 8, 6, 4, 2, 0);
        __m512 vbias = _mm512_set1_ps(bv);
        for (int oh = out_row_start; oh < out_end; ++oh) {
            int ih0 = oh * 2 - pH;
            int ow = 0;
            for (; ow + 16 <= oW; ow += 16) {
                int iw0 = ow * 2 - pW;
                __m512 acc = vbias;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = ih0 + kh;
                    if (ih < 0 || ih >= iH) continue;
                    const float* xrow = xc + ih * iW;
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = iw0 + kw;
                        if (iw >= 0 && iw + 31 < iW) {
                            __m512 lo = _mm512_loadu_ps(xrow + iw);
                            __m512 hi = _mm512_loadu_ps(xrow + iw + 16);
                            __m512 vals = _mm512_permutex2var_ps(lo, perm_s2, hi);
                            acc = _mm512_fmadd_ps(
                                _mm512_set1_ps(wc[kh * kW + kw]), vals, acc);
                        } else {
                            __mmask16 mask = 0;
                            for (int p = 0; p < 16; ++p) {
                                int iw_p = iw + p * 2;
                                if (iw_p >= 0 && iw_p < iW)
                                    mask |= (1 << p);
                            }
                            __m512 vals = _mm512_mask_i32gather_ps(
                                _mm512_setzero_ps(), mask, gather_s2, xrow + iw, 4);
                            acc = _mm512_fmadd_ps(
                                _mm512_set1_ps(wc[kh * kW + kw]), vals, acc);
                        }
                    }
                }
                _mm512_storeu_ps(yc + (oh - out_row_start) * oW + ow, acc);
            }
            for (; ow < oW; ++ow) {
                int iw0 = ow * 2 - pW;
                float sum = bv;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = ih0 + kh;
                    if (ih < 0 || ih >= iH) continue;
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = iw0 + kw;
                        if (iw < 0 || iw >= iW) continue;
                        sum += xc[ih * iW + iw] * wc[kh * kW + kw];
                    }
                }
                yc[(oh - out_row_start) * oW + ow] = sum;
            }
        }
        if (post_fn) post_fn(yc, 1, spatial_strip, spatial_strip, fused_op, nullptr, tensor_offset);
        return true;
    }

    return false;  // not handled
}

} // namespace nnr
