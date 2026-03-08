#pragma once
// x64 AVX-512/AVX2 generic depthwise convolution kernels.
// Called from kernel/depthwise.h. Handles any kH×kW with stride 1 and 2.

#ifdef NNR_ARCH_X64
#include <immintrin.h>
#include <algorithm>
#include <vector>
#include "nnr.h"
#include "thread_pool.h"
#include "cpu_features.h"

namespace nnr {

// Full depthwise conv2d (all output rows). Returns true if handled.
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=DW fusion=post_op
inline bool depthwise_conv2d_x64(float* output, const float* input, const float* weight,
    const float* bias, int N, int C, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW,
    operator_t::post_fn_t post_fn, const operator_t* fused_op)
{
    int NC = N * C;
    int spatial = oH * oW;

    if (has_avx512() && dH == 1 && dW == 1) {
        if (sH == 1 && sW == 1) {
            // Pre-compute masks outside the per-channel lambda (they don't depend on nc).
            __mmask16 omask = oW <= 16 ? (__mmask16)((1u << oW) - 1) : 0;
            std::vector<__mmask16> kmasks_buf;
            if (oW <= 16) {
                kmasks_buf.resize(kH * kW);
                for (int ki = 0; ki < kH * kW; ++ki) {
                    int iw = -pW + (ki % kW);
                    int mf = std::max(0, -iw), ml = std::min(oW, iW - iw);
                    kmasks_buf[ki] = (ml > mf) ? (__mmask16)(((1u << ml) - 1) & ~((1u << mf) - 1)) : 0;
                }
            }
            const __mmask16* kmasks = kmasks_buf.data();
            nnr::for_static(0, NC, NC > 4, [&](int nc) {
                int c = nc % C;
                const float* xc = input + (size_t)nc * iH * iW;
                const float* wc = weight + (size_t)c * kH * kW;
                float* yc = output + (size_t)nc * spatial;
                __m512 vbias = _mm512_set1_ps(bias ? bias[c] : 0.0f);
                if (oW <= 16) {
                    for (int oh = 0; oh < oH; ++oh) {
                        int ih0 = oh - pH;
                        __m512 acc = vbias;
                        for (int kh = 0; kh < kH; ++kh) {
                            int ih = ih0 + kh;
                            if (ih < 0 || ih >= iH) continue;
                            const float* xrow = xc + ih * iW;
                            for (int kw = 0; kw < kW; ++kw) {
                                __mmask16 mask = kmasks[kh * kW + kw];
                                if (mask) acc = _mm512_fmadd_ps(
                                    _mm512_set1_ps(wc[kh * kW + kw]),
                                    _mm512_maskz_loadu_ps(mask, xrow + (-pW + kw)), acc);
                            }
                        }
                        _mm512_mask_storeu_ps(yc + oh * oW, omask, acc);
                    }
                } else {
                for (int oh = 0; oh < oH; ++oh) {
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
                                    int mf = std::max(0, -iw), ml = std::min(16, iW - iw);
                                    __mmask16 mask = (ml > mf) ? (__mmask16)((0xFFFFu >> (16 - ml + mf)) << mf) : 0;
                                    acc = _mm512_fmadd_ps(
                                        _mm512_set1_ps(wc[kh * kW + kw]),
                                        _mm512_maskz_loadu_ps(mask, xrow + iw), acc);
                                }
                            }
                        }
                        _mm512_storeu_ps(yc + oh * oW + ow, acc);
                    }
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
                } // end oW > 16
                if (post_fn) post_fn(yc, 1, spatial, spatial, fused_op, nullptr, 0);
            });
            return true;
        }
        if (sH == 2 && sW == 2) {
            const __m512i perm_s2 = _mm512_set_epi32(
                30, 28, 26, 24, 22, 20, 18, 16,
                14, 12, 10, 8, 6, 4, 2, 0);
            const __m512i gather_s2 = _mm512_set_epi32(
                30, 28, 26, 24, 22, 20, 18, 16,
                14, 12, 10, 8, 6, 4, 2, 0);
            nnr::for_static(0, NC, NC > 4, [&](int nc) {
                int c = nc % C;
                const float* xc = input + (size_t)nc * iH * iW;
                const float* wc = weight + (size_t)c * kH * kW;
                float* yc = output + (size_t)nc * spatial;
                __m512 vbias = _mm512_set1_ps(bias ? bias[c] : 0.0f);
                for (int oh = 0; oh < oH; ++oh) {
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
                                    int mf = std::max(0, (-iw + 1) / 2), ml = std::min(16, (iW - iw + 1) / 2);
                                    __mmask16 mask = (ml > mf) ? (__mmask16)((0xFFFFu >> (16 - ml + mf)) << mf) : 0;
                                    __m512 vals = _mm512_mask_i32gather_ps(
                                        _mm512_setzero_ps(), mask, gather_s2, xrow + iw, 4);
                                    acc = _mm512_fmadd_ps(
                                        _mm512_set1_ps(wc[kh * kW + kw]), vals, acc);
                                }
                            }
                        }
                        _mm512_storeu_ps(yc + oh * oW + ow, acc);
                    }
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
                if (post_fn) post_fn(yc, 1, spatial, spatial, fused_op, nullptr, 0);
            });
            return true;
        }
    }

    if (detect_isa() == isa_t::avx2 && dH == 1 && dW == 1) {
        if (sH == 1 && sW == 1) {
            nnr::for_static(0, NC, NC > 4, [&](int nc) {
                int c = nc % C;
                const float* xc = input + (size_t)nc * iH * iW;
                const float* wc = weight + (size_t)c * kH * kW;
                float* yc = output + (size_t)nc * spatial;
                __m256 vbias = _mm256_set1_ps(bias ? bias[c] : 0.0f);
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
                                    alignas(32) int32_t mask_arr[8];
                                    for (int p = 0; p < 8; ++p) {
                                        int iw_p = iw + p;
                                        mask_arr[p] = (iw_p >= 0 && iw_p < iW) ? -1 : 0;
                                    }
                                    __m256i mask = _mm256_load_si256((const __m256i*)mask_arr);
                                    acc = _mm256_fmadd_ps(
                                        _mm256_set1_ps(wc[kh * kW + kw]),
                                        _mm256_maskload_ps(xrow + iw, mask), acc);
                                }
                            }
                        }
                        _mm256_storeu_ps(yc + oh * oW + ow, acc);
                    }
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
                if (post_fn) post_fn(yc, 1, spatial, spatial, fused_op, nullptr, 0);
            });
            return true;
        }
        if (sH == 2 && sW == 2) {
            const __m256i perm_even = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
            nnr::for_static(0, NC, NC > 4, [&](int nc) {
                int c = nc % C;
                const float* xc = input + (size_t)nc * iH * iW;
                const float* wc = weight + (size_t)c * kH * kW;
                float* yc = output + (size_t)nc * spatial;
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
                                if (iw >= 0 && iw + 15 < iW) {
                                    __m256 lo = _mm256_loadu_ps(xrow + iw);
                                    __m256 hi = _mm256_loadu_ps(xrow + iw + 8);
                                    __m256 lo_even = _mm256_permutevar8x32_ps(lo, perm_even);
                                    __m256 hi_even = _mm256_permutevar8x32_ps(hi, perm_even);
                                    __m256 vals = _mm256_blend_ps(lo_even, hi_even, 0xF0);
                                    acc = _mm256_fmadd_ps(
                                        _mm256_set1_ps(wc[kh * kW + kw]), vals, acc);
                                } else {
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
                if (post_fn) post_fn(yc, 1, spatial, spatial, fused_op, nullptr, 0);
            });
            return true;
        }
    return false;
}

// Strip depthwise conv2d (rows oh_start..oh_end). Returns true if handled.
inline bool depthwise_conv2d_strip_x64(float* output, const float* input, const float* weight,
    const float* bias, int N, int C, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW,
    int oh_start, int oh_end,
    operator_t::post_fn_t post_fn, const operator_t* fused_op)
{
    int NC = N * C;
    int spatial = oH * oW;
    int strip_len = (oh_end - oh_start) * oW;

    if (has_avx512() && dH == 1 && dW == 1) {
        if (sH == 1 && sW == 1) {
            nnr::for_static(0, NC, NC > 4, [&](int nc) {
                int c = nc % C;
                const float* xc = input + (size_t)nc * iH * iW;
                const float* wc = weight + (size_t)c * kH * kW;
                float* yc = output + (size_t)nc * spatial;
                __m512 vbias = _mm512_set1_ps(bias ? bias[c] : 0.0f);
                for (int oh = oh_start; oh < oh_end; ++oh) {
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
                                    int mf = std::max(0, -iw), ml = std::min(16, iW - iw);
                                    __mmask16 mask = (ml > mf) ? (__mmask16)((0xFFFFu >> (16 - ml + mf)) << mf) : 0;
                                    acc = _mm512_fmadd_ps(
                                        _mm512_set1_ps(wc[kh * kW + kw]),
                                        _mm512_maskz_loadu_ps(mask, xrow + iw), acc);
                                }
                            }
                        }
                        _mm512_storeu_ps(yc + oh * oW + ow, acc);
                    }
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
                if (post_fn) post_fn(yc + oh_start * oW, 1, strip_len, strip_len, fused_op, nullptr, 0);
            });
            return true;
        }
        if (sH == 2 && sW == 2) {
            const __m512i perm_s2 = _mm512_set_epi32(
                30, 28, 26, 24, 22, 20, 18, 16,
                14, 12, 10, 8, 6, 4, 2, 0);
            const __m512i gather_s2 = _mm512_set_epi32(
                30, 28, 26, 24, 22, 20, 18, 16,
                14, 12, 10, 8, 6, 4, 2, 0);
            nnr::for_static(0, NC, NC > 4, [&](int nc) {
                int c = nc % C;
                const float* xc = input + (size_t)nc * iH * iW;
                const float* wc = weight + (size_t)c * kH * kW;
                float* yc = output + (size_t)nc * spatial;
                __m512 vbias = _mm512_set1_ps(bias ? bias[c] : 0.0f);
                for (int oh = oh_start; oh < oh_end; ++oh) {
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
                                    int mf = std::max(0, (-iw + 1) / 2), ml = std::min(16, (iW - iw + 1) / 2);
                                    __mmask16 mask = (ml > mf) ? (__mmask16)((0xFFFFu >> (16 - ml + mf)) << mf) : 0;
                                    __m512 vals = _mm512_mask_i32gather_ps(
                                        _mm512_setzero_ps(), mask, gather_s2, xrow + iw, 4);
                                    acc = _mm512_fmadd_ps(
                                        _mm512_set1_ps(wc[kh * kW + kw]), vals, acc);
                                }
                            }
                        }
                        _mm512_storeu_ps(yc + oh * oW + ow, acc);
                    }
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
                if (post_fn) post_fn(yc + oh_start * oW, 1, strip_len, strip_len, fused_op, nullptr, 0);
            });
            return true;
        }
    }

    if (detect_isa() == isa_t::avx2 && dH == 1 && dW == 1) {
        if (sH == 1 && sW == 1) {
            nnr::for_static(0, NC, NC > 4, [&](int nc) {
                int c = nc % C;
                const float* xc = input + (size_t)nc * iH * iW;
                const float* wc = weight + (size_t)c * kH * kW;
                float* yc = output + (size_t)nc * spatial;
                __m256 vbias = _mm256_set1_ps(bias ? bias[c] : 0.0f);
                for (int oh = oh_start; oh < oh_end; ++oh) {
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
                                    alignas(32) int32_t mask_arr[8];
                                    for (int p = 0; p < 8; ++p) {
                                        int iw_p = iw + p;
                                        mask_arr[p] = (iw_p >= 0 && iw_p < iW) ? -1 : 0;
                                    }
                                    __m256i mask = _mm256_load_si256((const __m256i*)mask_arr);
                                    acc = _mm256_fmadd_ps(
                                        _mm256_set1_ps(wc[kh * kW + kw]),
                                        _mm256_maskload_ps(xrow + iw, mask), acc);
                                }
                            }
                        }
                        _mm256_storeu_ps(yc + oh * oW + ow, acc);
                    }
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
                if (post_fn) post_fn(yc + oh_start * oW, 1, strip_len, strip_len, fused_op, nullptr, 0);
            });
            return true;
        }
        if (sH == 2 && sW == 2) {
            const __m256i perm_even = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
            nnr::for_static(0, NC, NC > 4, [&](int nc) {
                int c = nc % C;
                const float* xc = input + (size_t)nc * iH * iW;
                const float* wc = weight + (size_t)c * kH * kW;
                float* yc = output + (size_t)nc * spatial;
                __m256 vbias = _mm256_set1_ps(bias ? bias[c] : 0.0f);
                for (int oh = oh_start; oh < oh_end; ++oh) {
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
                                if (iw >= 0 && iw + 15 < iW) {
                                    __m256 lo = _mm256_loadu_ps(xrow + iw);
                                    __m256 hi = _mm256_loadu_ps(xrow + iw + 8);
                                    __m256 lo_even = _mm256_permutevar8x32_ps(lo, perm_even);
                                    __m256 hi_even = _mm256_permutevar8x32_ps(hi, perm_even);
                                    __m256 vals = _mm256_blend_ps(lo_even, hi_even, 0xF0);
                                    acc = _mm256_fmadd_ps(
                                        _mm256_set1_ps(wc[kh * kW + kw]), vals, acc);
                                } else {
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
                if (post_fn) post_fn(yc + oh_start * oW, 1, strip_len, strip_len, fused_op, nullptr, 0);
            });
            return true;
        }
    }
    return false;
}

} // namespace nnr

#endif // NNR_ARCH_X64
