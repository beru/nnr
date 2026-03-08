#pragma once
// AVX2+FMA depthwise 2D convolution backend.
// 256-bit (8 floats per vector) counterpart of avx512/depthwise.h.

#include <immintrin.h>
#include "thread_pool.h"

namespace nnr::avx2 {

// Precomputed boundary masks for depthwise convolution.
// Table-based: a single unaligned load from a 16-element table produces the mask.
// Replaces per-element mask construction in the inner loop.
// @nnr-meta isa=AVX2 dtype=fp32 layout=NCHW
inline __m256i boundary_mask_left(int first_valid)
{
    // first_valid = number of invalid lanes on the left (0..7)
    alignas(32) static const int32_t table[16] = {
        0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1
    };
    return _mm256_loadu_si256((const __m256i*)(table + 8 - first_valid));
}
// @nnr-meta isa=AVX2 dtype=fp32 layout=NCHW
inline __m256i boundary_mask_right(int last_valid)
{
    // last_valid = number of valid lanes from the left (1..8)
    alignas(32) static const int32_t table[16] = {
        -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0
    };
    return _mm256_loadu_si256((const __m256i*)(table + 8 - last_valid));
}

#if 0 // LOCUST
;empty_init = "{}"
;for S in [1, 2]:
;    suffix = "" if S == 1 else "_s2"
;    stride_comment = f"stride={S}"

// AVX2 float depthwise conv, @stride_comment@, dilation=1 specialization.
;    if S == 1:
// Processes 8 output pixels per iteration with masked boundary loads.
;        pass
;    else:
// 8 output elements need input at stride-2 positions: iw, iw+2, ..., iw+14.
// Interior: load 16 contiguous floats into two __m256, extract even indices
// using _mm256_permutevar8x32_ps + blend.
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

;    if S == 2:
    // Permutation to extract even-indexed floats from a pair of __m256:
    // From lo[0..7]: take indices 0,2,4,6 -> positions 0,1,2,3
    // From hi[0..7]: take indices 0,2,4,6 -> positions 4,5,6,7
    const __m256i perm_even = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);

;        pass
    const int iC = x->dims[1];
    nnr::for_static(0, oN * oC, oN * oC > 4, [&](int nc) {
        int n = nc / oC, c = nc % oC;
        int ic = (int)((size_t)c * iC / oC);
        const float* xc = xd + ((size_t)n * iC + ic) * iH * iW;
        const float* wc = wd + (size_t)c * kH * kW;
        float* yc = yd + ((size_t)n * oC + c) * oH * oW;
        __m256 vbias = _mm256_set1_ps(bias ? bias[c] : 0.0f);
        for (int oh = 0; oh < oH; ++oh) {
;    if S == 1:
            int ih0 = oh - pH;
;        pass
;    else:
            int ih0 = oh * 2 - pH;
;        pass
            int ow = 0;
            for (; ow + 8 <= oW; ow += 8) {
;    if S == 1:
                int iw0 = ow - pW;
;        pass
;    else:
                int iw0 = ow * 2 - pW;
;        pass
                __m256 acc = vbias;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = ih0 + kh;
                    if (ih < 0 || ih >= iH) continue;
                    const float* xrow = xc + ih * iW;
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = iw0 + kw;
;    if S == 1:
                        if (iw >= 0 && iw + 7 < iW) {
                            acc = _mm256_fmadd_ps(
                                _mm256_set1_ps(wc[kh * kW + kw]),
                                _mm256_loadu_ps(xrow + iw), acc);
                        } else {
                            // Boundary: table-based mask (no per-element loop)
                            int first = std::max(0, -iw);
                            int last = std::min(8, iW - iw);
                            if (first >= last) continue;
                            __m256i mask = _mm256_and_si256(
                                boundary_mask_left(first),
                                boundary_mask_right(last));
                            acc = _mm256_fmadd_ps(
                                _mm256_set1_ps(wc[kh * kW + kw]),
                                _mm256_maskload_ps(xrow + iw, mask), acc);
                        }
;        pass
;    else:
                        // Need input at positions: iw, iw+2, ..., iw+14
                        if (iw >= 0 && iw + 15 < iW) {
                            // Interior: load 16 floats, extract even indices
                            __m256 lo = _mm256_loadu_ps(xrow + iw);
                            __m256 hi = _mm256_loadu_ps(xrow + iw + 8);
                            __m256 lo_even = _mm256_permutevar8x32_ps(lo, perm_even);
                            __m256 hi_even = _mm256_permutevar8x32_ps(hi, perm_even);
                            // Blend: take lower 128 from lo_even, upper 128 from hi_even
                            __m256 vals = _mm256_blend_ps(lo_even, hi_even, 0xF0);
                            acc = _mm256_fmadd_ps(
                                _mm256_set1_ps(wc[kh * kW + kw]), vals, acc);
                        } else {
                            // Boundary: scalar gather into temp array
                            alignas(32) float tmp[8] = @empty_init@;
                            for (int p = 0; p < 8; ++p) {
                                int iw_p = iw + p * 2;
                                if (iw_p >= 0 && iw_p < iW)
                                    tmp[p] = xrow[iw_p];
                            }
                            acc = _mm256_fmadd_ps(
                                _mm256_set1_ps(wc[kh * kW + kw]),
                                _mm256_load_ps(tmp), acc);
                        }
;        pass
                    }
                }
                _mm256_storeu_ps(yc + oh * oW + ow, acc);
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

// AVX2 float depthwise conv, stride=1, dilation=1 specialization.
// Processes 8 output pixels per iteration with masked boundary loads.
// @nnr-meta isa=AVX2 dtype=fp32 layout=NCHW special=DW fusion=post_op
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
        __m256 vbias = _mm256_set1_ps(bias ? bias[c] : 0.0f);
        alignas(32) static const float zeros[512] = {};
        if (kH == 3 && kW == 3) {
            // Pre-broadcast all 9 weights once.
            __m256 w00=_mm256_set1_ps(wc[0]), w01=_mm256_set1_ps(wc[1]), w02=_mm256_set1_ps(wc[2]);
            __m256 w10=_mm256_set1_ps(wc[3]), w11=_mm256_set1_ps(wc[4]), w12=_mm256_set1_ps(wc[5]);
            __m256 w20=_mm256_set1_ps(wc[6]), w21=_mm256_set1_ps(wc[7]), w22=_mm256_set1_ps(wc[8]);
            // Interior: iw0+0 >= 0 (=> ow >= pW) and iw0+2+7 < iW (=> ow < iW+pW-9).
            const int ow_int_start = pW;
            const int ow_int_end   = iW + pW - 9;
            for (int oh = 0; oh < oH; ++oh) {
                int ih0 = oh - pH;
                const float* r0 = (ih0     >= 0 && ih0     < iH) ? xc + ih0     * iW : zeros;
                const float* r1 = (ih0 + 1 >= 0 && ih0 + 1 < iH) ? xc + (ih0+1) * iW : zeros;
                const float* r2 = (ih0 + 2 >= 0 && ih0 + 2 < iH) ? xc + (ih0+2) * iW : zeros;
                int ow = 0;
                // Left boundary tiles.
                for (; ow + 8 <= oW && ow < ow_int_start; ow += 8) {
                    int iw0 = ow - pW;
                    __m256 a0 = vbias, a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps();
                    for (int kw = 0; kw < 3; ++kw) {
                        int iw = iw0 + kw;
                        int first = std::max(0, -iw), last = std::min(8, iW - iw);
                        if (first >= last) continue;
                        __m256i m = _mm256_and_si256(boundary_mask_left(first), boundary_mask_right(last));
                        a0 = _mm256_fmadd_ps(_mm256_set1_ps(wc[    kw]), _mm256_maskload_ps(r0+iw, m), a0);
                        a1 = _mm256_fmadd_ps(_mm256_set1_ps(wc[3 + kw]), _mm256_maskload_ps(r1+iw, m), a1);
                        a2 = _mm256_fmadd_ps(_mm256_set1_ps(wc[6 + kw]), _mm256_maskload_ps(r2+iw, m), a2);
                    }
                    _mm256_storeu_ps(yc + oh*oW + ow, _mm256_add_ps(_mm256_add_ps(a0, a1), a2));
                }
                // Interior tiles: 3 independent FMA chains, no branches.
                for (; ow + 8 <= oW && ow < ow_int_end; ow += 8) {
                    int iw0 = ow - pW;
                    __m256 a0 = vbias, a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps();
                    a0 = _mm256_fmadd_ps(w00, _mm256_loadu_ps(r0 + iw0    ), a0);
                    a0 = _mm256_fmadd_ps(w01, _mm256_loadu_ps(r0 + iw0 + 1), a0);
                    a0 = _mm256_fmadd_ps(w02, _mm256_loadu_ps(r0 + iw0 + 2), a0);
                    a1 = _mm256_fmadd_ps(w10, _mm256_loadu_ps(r1 + iw0    ), a1);
                    a1 = _mm256_fmadd_ps(w11, _mm256_loadu_ps(r1 + iw0 + 1), a1);
                    a1 = _mm256_fmadd_ps(w12, _mm256_loadu_ps(r1 + iw0 + 2), a1);
                    a2 = _mm256_fmadd_ps(w20, _mm256_loadu_ps(r2 + iw0    ), a2);
                    a2 = _mm256_fmadd_ps(w21, _mm256_loadu_ps(r2 + iw0 + 1), a2);
                    a2 = _mm256_fmadd_ps(w22, _mm256_loadu_ps(r2 + iw0 + 2), a2);
                    _mm256_storeu_ps(yc + oh*oW + ow, _mm256_add_ps(_mm256_add_ps(a0, a1), a2));
                }
                // Right boundary tiles.
                for (; ow + 8 <= oW; ow += 8) {
                    int iw0 = ow - pW;
                    __m256 a0 = vbias, a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps();
                    for (int kw = 0; kw < 3; ++kw) {
                        int iw = iw0 + kw;
                        int first = std::max(0, -iw), last = std::min(8, iW - iw);
                        if (first >= last) continue;
                        __m256i m = _mm256_and_si256(boundary_mask_left(first), boundary_mask_right(last));
                        a0 = _mm256_fmadd_ps(_mm256_set1_ps(wc[    kw]), _mm256_maskload_ps(r0+iw, m), a0);
                        a1 = _mm256_fmadd_ps(_mm256_set1_ps(wc[3 + kw]), _mm256_maskload_ps(r1+iw, m), a1);
                        a2 = _mm256_fmadd_ps(_mm256_set1_ps(wc[6 + kw]), _mm256_maskload_ps(r2+iw, m), a2);
                    }
                    _mm256_storeu_ps(yc + oh*oW + ow, _mm256_add_ps(_mm256_add_ps(a0, a1), a2));
                }
                // Scalar tail.
                for (; ow < oW; ++ow) {
                    int iw0 = ow - pW;
                    float sum = bias ? bias[c] : 0.0f;
                    for (int kh = 0; kh < 3; ++kh) {
                        int ih = ih0 + kh;
                        if (ih < 0 || ih >= iH) continue;
                        for (int kw = 0; kw < 3; ++kw) {
                            int iw = iw0 + kw;
                            if (iw < 0 || iw >= iW) continue;
                            sum += xc[ih * iW + iw] * wc[kh * 3 + kw];
                        }
                    }
                    yc[oh * oW + ow] = sum;
                }
            }
        } else if (kH == 5 && kW == 5) {
            // Pre-broadcast all 25 weights once.
            __m256 w00=_mm256_set1_ps(wc[ 0]), w01=_mm256_set1_ps(wc[ 1]), w02=_mm256_set1_ps(wc[ 2]), w03=_mm256_set1_ps(wc[ 3]), w04=_mm256_set1_ps(wc[ 4]);
            __m256 w10=_mm256_set1_ps(wc[ 5]), w11=_mm256_set1_ps(wc[ 6]), w12=_mm256_set1_ps(wc[ 7]), w13=_mm256_set1_ps(wc[ 8]), w14=_mm256_set1_ps(wc[ 9]);
            __m256 w20=_mm256_set1_ps(wc[10]), w21=_mm256_set1_ps(wc[11]), w22=_mm256_set1_ps(wc[12]), w23=_mm256_set1_ps(wc[13]), w24=_mm256_set1_ps(wc[14]);
            __m256 w30=_mm256_set1_ps(wc[15]), w31=_mm256_set1_ps(wc[16]), w32=_mm256_set1_ps(wc[17]), w33=_mm256_set1_ps(wc[18]), w34=_mm256_set1_ps(wc[19]);
            __m256 w40=_mm256_set1_ps(wc[20]), w41=_mm256_set1_ps(wc[21]), w42=_mm256_set1_ps(wc[22]), w43=_mm256_set1_ps(wc[23]), w44=_mm256_set1_ps(wc[24]);
            // Interior: iw0+4+7 < iW (=> ow < iW+pW-11).
            const int ow_int_start = pW;
            const int ow_int_end   = iW + pW - 11;
            for (int oh = 0; oh < oH; ++oh) {
                int ih0 = oh - pH;
                const float* r0 = (ih0     >= 0 && ih0     < iH) ? xc + ih0     * iW : zeros;
                const float* r1 = (ih0 + 1 >= 0 && ih0 + 1 < iH) ? xc + (ih0+1) * iW : zeros;
                const float* r2 = (ih0 + 2 >= 0 && ih0 + 2 < iH) ? xc + (ih0+2) * iW : zeros;
                const float* r3 = (ih0 + 3 >= 0 && ih0 + 3 < iH) ? xc + (ih0+3) * iW : zeros;
                const float* r4 = (ih0 + 4 >= 0 && ih0 + 4 < iH) ? xc + (ih0+4) * iW : zeros;
                int ow = 0;
                // Left boundary tiles.
                for (; ow + 8 <= oW && ow < ow_int_start; ow += 8) {
                    int iw0 = ow - pW;
                    __m256 a0=vbias, a1=_mm256_setzero_ps(), a2=_mm256_setzero_ps(), a3=_mm256_setzero_ps(), a4=_mm256_setzero_ps();
                    for (int kw = 0; kw < 5; ++kw) {
                        int iw = iw0 + kw;
                        int first = std::max(0, -iw), last = std::min(8, iW - iw);
                        if (first >= last) continue;
                        __m256i m = _mm256_and_si256(boundary_mask_left(first), boundary_mask_right(last));
                        a0 = _mm256_fmadd_ps(_mm256_set1_ps(wc[    kw]), _mm256_maskload_ps(r0+iw, m), a0);
                        a1 = _mm256_fmadd_ps(_mm256_set1_ps(wc[ 5 +kw]), _mm256_maskload_ps(r1+iw, m), a1);
                        a2 = _mm256_fmadd_ps(_mm256_set1_ps(wc[10 +kw]), _mm256_maskload_ps(r2+iw, m), a2);
                        a3 = _mm256_fmadd_ps(_mm256_set1_ps(wc[15 +kw]), _mm256_maskload_ps(r3+iw, m), a3);
                        a4 = _mm256_fmadd_ps(_mm256_set1_ps(wc[20 +kw]), _mm256_maskload_ps(r4+iw, m), a4);
                    }
                    _mm256_storeu_ps(yc + oh*oW + ow, _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(_mm256_add_ps(a2, a3), a4)));
                }
                // Interior tiles: 5 independent FMA chains, no branches.
                for (; ow + 8 <= oW && ow < ow_int_end; ow += 8) {
                    int iw0 = ow - pW;
                    __m256 a0=vbias, a1=_mm256_setzero_ps(), a2=_mm256_setzero_ps(), a3=_mm256_setzero_ps(), a4=_mm256_setzero_ps();
                    a0 = _mm256_fmadd_ps(w00, _mm256_loadu_ps(r0 + iw0    ), a0);
                    a0 = _mm256_fmadd_ps(w01, _mm256_loadu_ps(r0 + iw0 + 1), a0);
                    a0 = _mm256_fmadd_ps(w02, _mm256_loadu_ps(r0 + iw0 + 2), a0);
                    a0 = _mm256_fmadd_ps(w03, _mm256_loadu_ps(r0 + iw0 + 3), a0);
                    a0 = _mm256_fmadd_ps(w04, _mm256_loadu_ps(r0 + iw0 + 4), a0);
                    a1 = _mm256_fmadd_ps(w10, _mm256_loadu_ps(r1 + iw0    ), a1);
                    a1 = _mm256_fmadd_ps(w11, _mm256_loadu_ps(r1 + iw0 + 1), a1);
                    a1 = _mm256_fmadd_ps(w12, _mm256_loadu_ps(r1 + iw0 + 2), a1);
                    a1 = _mm256_fmadd_ps(w13, _mm256_loadu_ps(r1 + iw0 + 3), a1);
                    a1 = _mm256_fmadd_ps(w14, _mm256_loadu_ps(r1 + iw0 + 4), a1);
                    a2 = _mm256_fmadd_ps(w20, _mm256_loadu_ps(r2 + iw0    ), a2);
                    a2 = _mm256_fmadd_ps(w21, _mm256_loadu_ps(r2 + iw0 + 1), a2);
                    a2 = _mm256_fmadd_ps(w22, _mm256_loadu_ps(r2 + iw0 + 2), a2);
                    a2 = _mm256_fmadd_ps(w23, _mm256_loadu_ps(r2 + iw0 + 3), a2);
                    a2 = _mm256_fmadd_ps(w24, _mm256_loadu_ps(r2 + iw0 + 4), a2);
                    a3 = _mm256_fmadd_ps(w30, _mm256_loadu_ps(r3 + iw0    ), a3);
                    a3 = _mm256_fmadd_ps(w31, _mm256_loadu_ps(r3 + iw0 + 1), a3);
                    a3 = _mm256_fmadd_ps(w32, _mm256_loadu_ps(r3 + iw0 + 2), a3);
                    a3 = _mm256_fmadd_ps(w33, _mm256_loadu_ps(r3 + iw0 + 3), a3);
                    a3 = _mm256_fmadd_ps(w34, _mm256_loadu_ps(r3 + iw0 + 4), a3);
                    a4 = _mm256_fmadd_ps(w40, _mm256_loadu_ps(r4 + iw0    ), a4);
                    a4 = _mm256_fmadd_ps(w41, _mm256_loadu_ps(r4 + iw0 + 1), a4);
                    a4 = _mm256_fmadd_ps(w42, _mm256_loadu_ps(r4 + iw0 + 2), a4);
                    a4 = _mm256_fmadd_ps(w43, _mm256_loadu_ps(r4 + iw0 + 3), a4);
                    a4 = _mm256_fmadd_ps(w44, _mm256_loadu_ps(r4 + iw0 + 4), a4);
                    _mm256_storeu_ps(yc + oh*oW + ow, _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(_mm256_add_ps(a2, a3), a4)));
                }
                // Right boundary tiles.
                for (; ow + 8 <= oW; ow += 8) {
                    int iw0 = ow - pW;
                    __m256 a0=vbias, a1=_mm256_setzero_ps(), a2=_mm256_setzero_ps(), a3=_mm256_setzero_ps(), a4=_mm256_setzero_ps();
                    for (int kw = 0; kw < 5; ++kw) {
                        int iw = iw0 + kw;
                        int first = std::max(0, -iw), last = std::min(8, iW - iw);
                        if (first >= last) continue;
                        __m256i m = _mm256_and_si256(boundary_mask_left(first), boundary_mask_right(last));
                        a0 = _mm256_fmadd_ps(_mm256_set1_ps(wc[    kw]), _mm256_maskload_ps(r0+iw, m), a0);
                        a1 = _mm256_fmadd_ps(_mm256_set1_ps(wc[ 5 +kw]), _mm256_maskload_ps(r1+iw, m), a1);
                        a2 = _mm256_fmadd_ps(_mm256_set1_ps(wc[10 +kw]), _mm256_maskload_ps(r2+iw, m), a2);
                        a3 = _mm256_fmadd_ps(_mm256_set1_ps(wc[15 +kw]), _mm256_maskload_ps(r3+iw, m), a3);
                        a4 = _mm256_fmadd_ps(_mm256_set1_ps(wc[20 +kw]), _mm256_maskload_ps(r4+iw, m), a4);
                    }
                    _mm256_storeu_ps(yc + oh*oW + ow, _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(_mm256_add_ps(a2, a3), a4)));
                }
                // Scalar tail.
                for (; ow < oW; ++ow) {
                    int iw0 = ow - pW;
                    float sum = bias ? bias[c] : 0.0f;
                    for (int kh = 0; kh < 5; ++kh) {
                        int ih = ih0 + kh;
                        if (ih < 0 || ih >= iH) continue;
                        for (int kw = 0; kw < 5; ++kw) {
                            int iw = iw0 + kw;
                            if (iw < 0 || iw >= iW) continue;
                            sum += xc[ih * iW + iw] * wc[kh * 5 + kw];
                        }
                    }
                    yc[oh * oW + ow] = sum;
                }
            }
        } else {
            for (int oh = 0; oh < oH; ++oh) {
                int ih0 = oh - pH;
                int ow = 0;
                for (; ow + 8 <= oW; ow += 8) {
                    int iw0 = ow - pW;
                    __m256 acc = vbias;
                    for (int kh = 0; kh < kH; ++kh) {
                        int ih = ih0 + kh;
                        if (ih < 0 || ih >= iH) continue;
                        const float* xrow = xc + ih * iW;
                        for (int kw = 0; kw < kW; ++kw) {
                            int iw = iw0 + kw;
                            if (iw >= 0 && iw + 7 < iW) {
                                acc = _mm256_fmadd_ps(
                                    _mm256_set1_ps(wc[kh * kW + kw]),
                                    _mm256_loadu_ps(xrow + iw), acc);
                            } else {
                                // Boundary: table-based mask (no per-element loop)
                                int first = std::max(0, -iw);
                                int last = std::min(8, iW - iw);
                                if (first >= last) continue;
                                __m256i mask = _mm256_and_si256(
                                    boundary_mask_left(first),
                                    boundary_mask_right(last));
                                acc = _mm256_fmadd_ps(
                                    _mm256_set1_ps(wc[kh * kW + kw]),
                                    _mm256_maskload_ps(xrow + iw, mask), acc);
                            }
                        }
                    }
                    _mm256_storeu_ps(yc + oh * oW + ow, acc);
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
        }
        if (post_fn) {
            int toff = (int)(yc - yd);
            post_fn(yc, 1, oH * oW, oH * oW, fused_op, nullptr, toff);
        }
    });
    return true;
}


// AVX2 float depthwise conv, stride=2, dilation=1 specialization.
// 8 output elements need input at stride-2 positions: iw, iw+2, ..., iw+14.
// Interior: load 16 contiguous floats into two __m256, extract even indices
// using _mm256_permutevar8x32_ps + blend.
// Boundary: scalar gather.
// @nnr-meta isa=AVX2 dtype=fp32 layout=NCHW special=DW fusion=post_op
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

    // Permutation to extract even-indexed floats from a pair of __m256:
    // From lo[0..7]: take indices 0,2,4,6 -> positions 0,1,2,3
    // From hi[0..7]: take indices 0,2,4,6 -> positions 4,5,6,7
    const __m256i perm_even = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);

    const int iC = x->dims[1];
    nnr::for_static(0, oN * oC, oN * oC > 4, [&](int nc) {
        int n = nc / oC, c = nc % oC;
        int ic = (int)((size_t)c * iC / oC);
        const float* xc = xd + ((size_t)n * iC + ic) * iH * iW;
        const float* wc = wd + (size_t)c * kH * kW;
        float* yc = yd + ((size_t)n * oC + c) * oH * oW;
        __m256 vbias = _mm256_set1_ps(bias ? bias[c] : 0.0f);
        for (int oh = 0; oh < oH; ++oh) {
            int ih0 = oh * 2 - pH;
            int ow = 0;
            for (; ow + 8 <= oW; ow += 8) {
                int iw0 = ow * 2 - pW;
                __m256 acc = vbias;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = ih0 + kh;
                    if (ih < 0 || ih >= iH) continue;
                    const float* xrow = xc + ih * iW;
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = iw0 + kw;
                        // Need input at positions: iw, iw+2, ..., iw+14
                        if (iw >= 0 && iw + 15 < iW) {
                            // Interior: load 16 floats, extract even indices
                            __m256 lo = _mm256_loadu_ps(xrow + iw);
                            __m256 hi = _mm256_loadu_ps(xrow + iw + 8);
                            __m256 lo_even = _mm256_permutevar8x32_ps(lo, perm_even);
                            __m256 hi_even = _mm256_permutevar8x32_ps(hi, perm_even);
                            // Blend: take lower 128 from lo_even, upper 128 from hi_even
                            __m256 vals = _mm256_blend_ps(lo_even, hi_even, 0xF0);
                            acc = _mm256_fmadd_ps(
                                _mm256_set1_ps(wc[kh * kW + kw]), vals, acc);
                        } else {
                            // Boundary: scalar gather into temp array
                            alignas(32) float tmp[8] = {};
                            for (int p = 0; p < 8; ++p) {
                                int iw_p = iw + p * 2;
                                if (iw_p >= 0 && iw_p < iW)
                                    tmp[p] = xrow[iw_p];
                            }
                            acc = _mm256_fmadd_ps(
                                _mm256_set1_ps(wc[kh * kW + kw]),
                                _mm256_load_ps(tmp), acc);
                        }
                    }
                }
                _mm256_storeu_ps(yc + oh * oW + ow, acc);
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

} // namespace nnr::avx2
