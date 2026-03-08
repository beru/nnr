#pragma once
// AVX-512 depthwise 2D convolution backend.
// Included from Conv_depthwise.h.

#include <immintrin.h>
#include "allocator.h"
#include "thread_pool.h"
#ifdef NNR_USE_XBYAK
#include "backend/x64/jit_depthwise_avx512.h"
#include "backend/cpu/kernel/post_ops.h"
#endif

namespace nnr::avx512 {

// Masked SIMD path for oW <= 16: entire output row fits in one vector.
// Noinline to keep icache clean for the standard path (oW > 16).
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=DW fusion=post_op
static NNR_NOINLINE void depthwise_2d_small_ow(
    const int kH, const int kW, const int iH, const int iW,
    const int oN, const int oC, const int oH, const int oW,
    const float* xd, float* yd, const float* wd, float* bias,
    int pH, int pW, const int iC,
    operator_t::post_fn_t post_fn, const operator_t* fused_op,
    arena_t& arena)
{
    __mmask16 omask = (__mmask16)((1u << oW) - 1);
    arena_vector<__mmask16> kmasks(kH * kW, __mmask16{0}, arena_allocator<__mmask16>{arena});
    for (int ki = 0; ki < kH * kW; ++ki) {
        int iw = -pW + (ki % kW);
        int mf = std::max(0, -iw), ml = std::min(oW, iW - iw);
        kmasks[ki] = (ml > mf) ? (__mmask16)(((1u << ml) - 1) & ~((1u << mf) - 1)) : 0;
    }
    nnr::for_static(0, oN * oC, oN * oC > 4, [&, omask, kmasks](int nc) {
        int n = nc / oC, c = nc % oC;
        int ic = (int)((size_t)c * iC / oC);
        const float* xc = xd + ((size_t)n * iC + ic) * iH * iW;
        const float* wc = wd + (size_t)c * kH * kW;
        float* yc = yd + ((size_t)n * oC + c) * oH * oW;
        __m512 vbias = _mm512_set1_ps(bias ? bias[c] : 0.0f);
#if 0 // LOCUST
;for KH, KW in [(5, 5), (3, 3)]:
;    if KH == 5:
        if (kH == 5 && kW == 5) {
;        pass
;    else:
        } else if (kH == 3 && kW == 3) {
;        pass
            for (int oh = 0; oh < oH; ++oh) {
                int ih0 = oh - pH;
;    accs_decl = ", ".join(f"a{kh} = _mm512_setzero_ps()" for kh in range(1, KH))
                __m512 a0 = vbias, @accs_decl@;
                alignas(64) static const float zeros[512] = {};
;    for kh in range(KH):
                const float* r@kh@ = (ih0 + @kh@ >= 0 && ih0 + @kh@ < iH) ? xc + (ih0 + @kh@) * iW - pW : zeros;
;        pass
                for (int kw = 0; kw < @KW@; ++kw) {
;    for kh in range(KH):
;        WI_BASE = kh * KW
                    __m512 w@kh@ = _mm512_set1_ps(wc[@WI_BASE@ + kw]);
;        pass
;    for kh in range(KH):
;        WI_BASE = kh * KW
                    __m512 v@kh@ = _mm512_maskz_loadu_ps(kmasks[@WI_BASE@ + kw], r@kh@ + kw);
;        pass
;    for kh in range(KH):
                    a@kh@ = _mm512_fmadd_ps(w@kh@, v@kh@, a@kh@);
;        pass
                }
;    if KH == 5:
;        reduce = "_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(_mm512_add_ps(a2, a3), a4))"
;    else:
;        reduce = "_mm512_add_ps(_mm512_add_ps(a0, a1), a2)"
;    pass
                _mm512_mask_storeu_ps(yc + oh * oW, omask, @reduce@);
            }
;    pass
#else // LOCUST
        if (kH == 5 && kW == 5) {
            for (int oh = 0; oh < oH; ++oh) {
                int ih0 = oh - pH;
                __m512 a0 = vbias, a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps(), a4 = _mm512_setzero_ps();
                alignas(64) static const float zeros[512] = {};
                const float* r0 = (ih0 + 0 >= 0 && ih0 + 0 < iH) ? xc + (ih0 + 0) * iW - pW : zeros;
                const float* r1 = (ih0 + 1 >= 0 && ih0 + 1 < iH) ? xc + (ih0 + 1) * iW - pW : zeros;
                const float* r2 = (ih0 + 2 >= 0 && ih0 + 2 < iH) ? xc + (ih0 + 2) * iW - pW : zeros;
                const float* r3 = (ih0 + 3 >= 0 && ih0 + 3 < iH) ? xc + (ih0 + 3) * iW - pW : zeros;
                const float* r4 = (ih0 + 4 >= 0 && ih0 + 4 < iH) ? xc + (ih0 + 4) * iW - pW : zeros;
                for (int kw = 0; kw < 5; ++kw) {
                    __m512 w0 = _mm512_set1_ps(wc[0 + kw]);
                    __m512 w1 = _mm512_set1_ps(wc[5 + kw]);
                    __m512 w2 = _mm512_set1_ps(wc[10 + kw]);
                    __m512 w3 = _mm512_set1_ps(wc[15 + kw]);
                    __m512 w4 = _mm512_set1_ps(wc[20 + kw]);
                    __m512 v0 = _mm512_maskz_loadu_ps(kmasks[0 + kw], r0 + kw);
                    __m512 v1 = _mm512_maskz_loadu_ps(kmasks[5 + kw], r1 + kw);
                    __m512 v2 = _mm512_maskz_loadu_ps(kmasks[10 + kw], r2 + kw);
                    __m512 v3 = _mm512_maskz_loadu_ps(kmasks[15 + kw], r3 + kw);
                    __m512 v4 = _mm512_maskz_loadu_ps(kmasks[20 + kw], r4 + kw);
                    a0 = _mm512_fmadd_ps(w0, v0, a0);
                    a1 = _mm512_fmadd_ps(w1, v1, a1);
                    a2 = _mm512_fmadd_ps(w2, v2, a2);
                    a3 = _mm512_fmadd_ps(w3, v3, a3);
                    a4 = _mm512_fmadd_ps(w4, v4, a4);
                }
                _mm512_mask_storeu_ps(yc + oh * oW, omask, _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(_mm512_add_ps(a2, a3), a4)));
            }
        } else if (kH == 3 && kW == 3) {
            for (int oh = 0; oh < oH; ++oh) {
                int ih0 = oh - pH;
                __m512 a0 = vbias, a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps();
                alignas(64) static const float zeros[512] = {};
                const float* r0 = (ih0 + 0 >= 0 && ih0 + 0 < iH) ? xc + (ih0 + 0) * iW - pW : zeros;
                const float* r1 = (ih0 + 1 >= 0 && ih0 + 1 < iH) ? xc + (ih0 + 1) * iW - pW : zeros;
                const float* r2 = (ih0 + 2 >= 0 && ih0 + 2 < iH) ? xc + (ih0 + 2) * iW - pW : zeros;
                for (int kw = 0; kw < 3; ++kw) {
                    __m512 w0 = _mm512_set1_ps(wc[0 + kw]);
                    __m512 w1 = _mm512_set1_ps(wc[3 + kw]);
                    __m512 w2 = _mm512_set1_ps(wc[6 + kw]);
                    __m512 v0 = _mm512_maskz_loadu_ps(kmasks[0 + kw], r0 + kw);
                    __m512 v1 = _mm512_maskz_loadu_ps(kmasks[3 + kw], r1 + kw);
                    __m512 v2 = _mm512_maskz_loadu_ps(kmasks[6 + kw], r2 + kw);
                    a0 = _mm512_fmadd_ps(w0, v0, a0);
                    a1 = _mm512_fmadd_ps(w1, v1, a1);
                    a2 = _mm512_fmadd_ps(w2, v2, a2);
                }
                _mm512_mask_storeu_ps(yc + oh * oW, omask, _mm512_add_ps(_mm512_add_ps(a0, a1), a2));
            }
#endif // LOCUST
        } else {
            for (int oh = 0; oh < oH; ++oh) {
                int ih0 = oh - pH;
                __m512 acc = vbias;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = ih0 + kh;
                    if (ih < 0 || ih >= iH) continue;
                    const float* xrow = xc + ih * iW - pW;
                    for (int kw = 0; kw < kW; ++kw) {
                        __mmask16 m = kmasks[kh * kW + kw];
                        acc = _mm512_fmadd_ps(_mm512_set1_ps(wc[kh * kW + kw]),
                            _mm512_maskz_loadu_ps(m, xrow + kw), acc);
                    }
                }
                _mm512_mask_storeu_ps(yc + oh * oW, omask, acc);
            }
        }
        if (post_fn) {
            int toff = (int)(yc - yd);
            post_fn(yc, 1, oH * oW, oH * oW, fused_op, nullptr, toff);
        }
    });
}

// AVX-512 float depthwise conv, stride=1, dilation=1 specialization.
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=DW fusion=post_op
inline bool depthwise_2d(
    tensor_t* y, const tensor_t* x, const tensor_t* w, float* bias,
    int pH, int pW,
    operator_t::post_fn_t post_fn, const operator_t* fused_op,
    arena_t& arena)
{
    const int kH = w->dims[2], kW = w->dims[3];
    const int iH = x->dims[2], iW = x->dims[3];
    const int oN = y->dims[0], oC = y->dims[1], oH = y->dims[2], oW = y->dims[3];
    const float* xd = (const float*)x->data;
    float* yd = (float*)y->data;
    const float* wd = (const float*)w->data;

    const int iC = x->dims[1];

#ifdef NNR_USE_XBYAK
    // Try JIT path for 3×3 stride=1 with fusable activation
    if (oW > 16 && kH == 3 && kW == 3) {
        jit_activation_t act = jit_activation_t::none;
        jit_epilogue_params_t epi{};
        bool can_jit = true;
        if (post_fn == relu_post_fn) {
            act = jit_activation_t::relu;
        } else if (post_fn == clip_post_fn) {
            auto* p = reinterpret_cast<const clip_post_params_t*>(fused_op);
            act = jit_activation_t::clip;
            epi = {act, p->min_val, p->max_val, 0.0f};
        } else if (post_fn) {
            can_jit = false;  // unrecognized post_fn — fall through to intrinsics
        }
        bool jit_ran = false;
        if (can_jit) {
            jit_execute(
                jit_dw_eligible(iH, iW, oH, oW, pH, pW),
                [&] { return resolve_jit_depthwise(iH, iW, oH, oW, pH, pW, act, epi); },
                [&](auto fn) {
                    nnr::for_static(0, oN * oC, oN * oC > 4, [&](int nc) {
                        int n = nc / oC, c = nc % oC;
                        int ic = (int)((size_t)c * iC / oC);
                        const float* xc = xd + ((size_t)n * iC + ic) * iH * iW;
                        const float* wc = wd + (size_t)c * kH * kW;
                        float* yc = yd + ((size_t)n * oC + c) * oH * oW;
                        fn(xc, yc, wc, bias ? &bias[c] : nullptr);
                    });
                    jit_ran = true;
                },
                [&] { /* fall through to intrinsics below */ }
            );
            if (jit_ran) return true;
        }
    }
#endif

    if (oW <= 16) {
        depthwise_2d_small_ow(kH, kW, iH, iW, oN, oC, oH, oW, xd, yd, wd, bias, pH, pW, iC, post_fn, fused_op, arena);
    } else {
        nnr::for_static(0, oN * oC, oN * oC > 4, [&](int nc) {
            int n = nc / oC, c = nc % oC;
            int ic = (int)((size_t)c * iC / oC);
            const float* xc = xd + ((size_t)n * iC + ic) * iH * iW;
            const float* wc = wd + (size_t)c * kH * kW;
            float* yc = yd + ((size_t)n * oC + c) * oH * oW;
            __m512 vbias = _mm512_set1_ps(bias ? bias[c] : 0.0f);
            if (kH == 3 && kW == 3) {
                // Pre-broadcast all 9 weights once (constant across all oh/ow tiles).
                __m512 w00 = _mm512_set1_ps(wc[0]), w01 = _mm512_set1_ps(wc[1]), w02 = _mm512_set1_ps(wc[2]);
                __m512 w10 = _mm512_set1_ps(wc[3]), w11 = _mm512_set1_ps(wc[4]), w12 = _mm512_set1_ps(wc[5]);
                __m512 w20 = _mm512_set1_ps(wc[6]), w21 = _mm512_set1_ps(wc[7]), w22 = _mm512_set1_ps(wc[8]);
                // Interior ow range: iw0+0 >= 0 (=> ow >= pW) and iw0+2+15 < iW (=> ow < iW+pW-17).
                const int ow_int_start = pW;
                const int ow_int_end   = iW + pW - 17;
                alignas(64) static const float zeros[512] = {};
                for (int oh = 0; oh < oH; ++oh) {
                    int ih0 = oh - pH;
                    const float* r0 = (ih0     >= 0 && ih0     < iH) ? xc + ih0     * iW : zeros;
                    const float* r1 = (ih0 + 1 >= 0 && ih0 + 1 < iH) ? xc + (ih0+1) * iW : zeros;
                    const float* r2 = (ih0 + 2 >= 0 && ih0 + 2 < iH) ? xc + (ih0+2) * iW : zeros;
                    int ow = 0;
                    // Left boundary tiles (masked loads per kw).
                    for (; ow + 16 <= oW && ow < ow_int_start; ow += 16) {
                        int iw0 = ow - pW;
                        __m512 a0 = vbias, a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps();
                        for (int kw = 0; kw < 3; ++kw) {
                            int iw = iw0 + kw;
                            int mf = std::max(0, -iw), ml = std::min(16, iW - iw);
                            __mmask16 m = (ml > mf) ? (__mmask16)((0xFFFFu >> (16 - ml + mf)) << mf) : 0;
                            a0 = _mm512_fmadd_ps(_mm512_set1_ps(wc[    kw]), _mm512_maskz_loadu_ps(m, r0+iw), a0);
                            a1 = _mm512_fmadd_ps(_mm512_set1_ps(wc[3 + kw]), _mm512_maskz_loadu_ps(m, r1+iw), a1);
                            a2 = _mm512_fmadd_ps(_mm512_set1_ps(wc[6 + kw]), _mm512_maskz_loadu_ps(m, r2+iw), a2);
                        }
                        _mm512_storeu_ps(yc + oh*oW + ow, _mm512_add_ps(_mm512_add_ps(a0, a1), a2));
                    }
                    // Interior tiles: pre-broadcast weights, 3 independent FMA chains, no branches.
                    for (; ow + 16 <= oW && ow < ow_int_end; ow += 16) {
                        int iw0 = ow - pW;
                        __m512 a0 = vbias, a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps();
                        a0 = _mm512_fmadd_ps(w00, _mm512_loadu_ps(r0 + iw0    ), a0);
                        a0 = _mm512_fmadd_ps(w01, _mm512_loadu_ps(r0 + iw0 + 1), a0);
                        a0 = _mm512_fmadd_ps(w02, _mm512_loadu_ps(r0 + iw0 + 2), a0);
                        a1 = _mm512_fmadd_ps(w10, _mm512_loadu_ps(r1 + iw0    ), a1);
                        a1 = _mm512_fmadd_ps(w11, _mm512_loadu_ps(r1 + iw0 + 1), a1);
                        a1 = _mm512_fmadd_ps(w12, _mm512_loadu_ps(r1 + iw0 + 2), a1);
                        a2 = _mm512_fmadd_ps(w20, _mm512_loadu_ps(r2 + iw0    ), a2);
                        a2 = _mm512_fmadd_ps(w21, _mm512_loadu_ps(r2 + iw0 + 1), a2);
                        a2 = _mm512_fmadd_ps(w22, _mm512_loadu_ps(r2 + iw0 + 2), a2);
                        _mm512_storeu_ps(yc + oh*oW + ow, _mm512_add_ps(_mm512_add_ps(a0, a1), a2));
                    }
                    // Right boundary tiles (same masked logic as left).
                    for (; ow + 16 <= oW; ow += 16) {
                        int iw0 = ow - pW;
                        __m512 a0 = vbias, a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps();
                        for (int kw = 0; kw < 3; ++kw) {
                            int iw = iw0 + kw;
                            int mf = std::max(0, -iw), ml = std::min(16, iW - iw);
                            __mmask16 m = (ml > mf) ? (__mmask16)((0xFFFFu >> (16 - ml + mf)) << mf) : 0;
                            a0 = _mm512_fmadd_ps(_mm512_set1_ps(wc[    kw]), _mm512_maskz_loadu_ps(m, r0+iw), a0);
                            a1 = _mm512_fmadd_ps(_mm512_set1_ps(wc[3 + kw]), _mm512_maskz_loadu_ps(m, r1+iw), a1);
                            a2 = _mm512_fmadd_ps(_mm512_set1_ps(wc[6 + kw]), _mm512_maskz_loadu_ps(m, r2+iw), a2);
                        }
                        _mm512_storeu_ps(yc + oh*oW + ow, _mm512_add_ps(_mm512_add_ps(a0, a1), a2));
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
                __m512 w00=_mm512_set1_ps(wc[ 0]), w01=_mm512_set1_ps(wc[ 1]), w02=_mm512_set1_ps(wc[ 2]), w03=_mm512_set1_ps(wc[ 3]), w04=_mm512_set1_ps(wc[ 4]);
                __m512 w10=_mm512_set1_ps(wc[ 5]), w11=_mm512_set1_ps(wc[ 6]), w12=_mm512_set1_ps(wc[ 7]), w13=_mm512_set1_ps(wc[ 8]), w14=_mm512_set1_ps(wc[ 9]);
                __m512 w20=_mm512_set1_ps(wc[10]), w21=_mm512_set1_ps(wc[11]), w22=_mm512_set1_ps(wc[12]), w23=_mm512_set1_ps(wc[13]), w24=_mm512_set1_ps(wc[14]);
                __m512 w30=_mm512_set1_ps(wc[15]), w31=_mm512_set1_ps(wc[16]), w32=_mm512_set1_ps(wc[17]), w33=_mm512_set1_ps(wc[18]), w34=_mm512_set1_ps(wc[19]);
                __m512 w40=_mm512_set1_ps(wc[20]), w41=_mm512_set1_ps(wc[21]), w42=_mm512_set1_ps(wc[22]), w43=_mm512_set1_ps(wc[23]), w44=_mm512_set1_ps(wc[24]);
                // Interior ow: iw0+4+15 < iW (=> ow < iW+pW-19).
                const int ow_int_start = pW;
                const int ow_int_end   = iW + pW - 19;
                alignas(64) static const float zeros[512] = {};
                for (int oh = 0; oh < oH; ++oh) {
                    int ih0 = oh - pH;
                    const float* r0 = (ih0     >= 0 && ih0     < iH) ? xc + ih0     * iW : zeros;
                    const float* r1 = (ih0 + 1 >= 0 && ih0 + 1 < iH) ? xc + (ih0+1) * iW : zeros;
                    const float* r2 = (ih0 + 2 >= 0 && ih0 + 2 < iH) ? xc + (ih0+2) * iW : zeros;
                    const float* r3 = (ih0 + 3 >= 0 && ih0 + 3 < iH) ? xc + (ih0+3) * iW : zeros;
                    const float* r4 = (ih0 + 4 >= 0 && ih0 + 4 < iH) ? xc + (ih0+4) * iW : zeros;
                    int ow = 0;
                    // Left boundary tiles.
                    for (; ow + 16 <= oW && ow < ow_int_start; ow += 16) {
                        int iw0 = ow - pW;
                        __m512 a0=vbias, a1=_mm512_setzero_ps(), a2=_mm512_setzero_ps(), a3=_mm512_setzero_ps(), a4=_mm512_setzero_ps();
                        for (int kw = 0; kw < 5; ++kw) {
                            int iw = iw0 + kw;
                            int mf = std::max(0, -iw), ml = std::min(16, iW - iw);
                            __mmask16 m = (ml > mf) ? (__mmask16)((0xFFFFu >> (16 - ml + mf)) << mf) : 0;
                            a0 = _mm512_fmadd_ps(_mm512_set1_ps(wc[    kw]), _mm512_maskz_loadu_ps(m, r0+iw), a0);
                            a1 = _mm512_fmadd_ps(_mm512_set1_ps(wc[ 5 +kw]), _mm512_maskz_loadu_ps(m, r1+iw), a1);
                            a2 = _mm512_fmadd_ps(_mm512_set1_ps(wc[10 +kw]), _mm512_maskz_loadu_ps(m, r2+iw), a2);
                            a3 = _mm512_fmadd_ps(_mm512_set1_ps(wc[15 +kw]), _mm512_maskz_loadu_ps(m, r3+iw), a3);
                            a4 = _mm512_fmadd_ps(_mm512_set1_ps(wc[20 +kw]), _mm512_maskz_loadu_ps(m, r4+iw), a4);
                        }
                        _mm512_storeu_ps(yc + oh*oW + ow, _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(_mm512_add_ps(a2, a3), a4)));
                    }
                    // Interior tiles: 5 independent FMA chains, no branches.
                    for (; ow + 16 <= oW && ow < ow_int_end; ow += 16) {
                        int iw0 = ow - pW;
                        __m512 a0=vbias, a1=_mm512_setzero_ps(), a2=_mm512_setzero_ps(), a3=_mm512_setzero_ps(), a4=_mm512_setzero_ps();
                        a0 = _mm512_fmadd_ps(w00, _mm512_loadu_ps(r0 + iw0    ), a0);
                        a0 = _mm512_fmadd_ps(w01, _mm512_loadu_ps(r0 + iw0 + 1), a0);
                        a0 = _mm512_fmadd_ps(w02, _mm512_loadu_ps(r0 + iw0 + 2), a0);
                        a0 = _mm512_fmadd_ps(w03, _mm512_loadu_ps(r0 + iw0 + 3), a0);
                        a0 = _mm512_fmadd_ps(w04, _mm512_loadu_ps(r0 + iw0 + 4), a0);
                        a1 = _mm512_fmadd_ps(w10, _mm512_loadu_ps(r1 + iw0    ), a1);
                        a1 = _mm512_fmadd_ps(w11, _mm512_loadu_ps(r1 + iw0 + 1), a1);
                        a1 = _mm512_fmadd_ps(w12, _mm512_loadu_ps(r1 + iw0 + 2), a1);
                        a1 = _mm512_fmadd_ps(w13, _mm512_loadu_ps(r1 + iw0 + 3), a1);
                        a1 = _mm512_fmadd_ps(w14, _mm512_loadu_ps(r1 + iw0 + 4), a1);
                        a2 = _mm512_fmadd_ps(w20, _mm512_loadu_ps(r2 + iw0    ), a2);
                        a2 = _mm512_fmadd_ps(w21, _mm512_loadu_ps(r2 + iw0 + 1), a2);
                        a2 = _mm512_fmadd_ps(w22, _mm512_loadu_ps(r2 + iw0 + 2), a2);
                        a2 = _mm512_fmadd_ps(w23, _mm512_loadu_ps(r2 + iw0 + 3), a2);
                        a2 = _mm512_fmadd_ps(w24, _mm512_loadu_ps(r2 + iw0 + 4), a2);
                        a3 = _mm512_fmadd_ps(w30, _mm512_loadu_ps(r3 + iw0    ), a3);
                        a3 = _mm512_fmadd_ps(w31, _mm512_loadu_ps(r3 + iw0 + 1), a3);
                        a3 = _mm512_fmadd_ps(w32, _mm512_loadu_ps(r3 + iw0 + 2), a3);
                        a3 = _mm512_fmadd_ps(w33, _mm512_loadu_ps(r3 + iw0 + 3), a3);
                        a3 = _mm512_fmadd_ps(w34, _mm512_loadu_ps(r3 + iw0 + 4), a3);
                        a4 = _mm512_fmadd_ps(w40, _mm512_loadu_ps(r4 + iw0    ), a4);
                        a4 = _mm512_fmadd_ps(w41, _mm512_loadu_ps(r4 + iw0 + 1), a4);
                        a4 = _mm512_fmadd_ps(w42, _mm512_loadu_ps(r4 + iw0 + 2), a4);
                        a4 = _mm512_fmadd_ps(w43, _mm512_loadu_ps(r4 + iw0 + 3), a4);
                        a4 = _mm512_fmadd_ps(w44, _mm512_loadu_ps(r4 + iw0 + 4), a4);
                        _mm512_storeu_ps(yc + oh*oW + ow, _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(_mm512_add_ps(a2, a3), a4)));
                    }
                    // Right boundary tiles.
                    for (; ow + 16 <= oW; ow += 16) {
                        int iw0 = ow - pW;
                        __m512 a0=vbias, a1=_mm512_setzero_ps(), a2=_mm512_setzero_ps(), a3=_mm512_setzero_ps(), a4=_mm512_setzero_ps();
                        for (int kw = 0; kw < 5; ++kw) {
                            int iw = iw0 + kw;
                            int mf = std::max(0, -iw), ml = std::min(16, iW - iw);
                            __mmask16 m = (ml > mf) ? (__mmask16)((0xFFFFu >> (16 - ml + mf)) << mf) : 0;
                            a0 = _mm512_fmadd_ps(_mm512_set1_ps(wc[    kw]), _mm512_maskz_loadu_ps(m, r0+iw), a0);
                            a1 = _mm512_fmadd_ps(_mm512_set1_ps(wc[ 5 +kw]), _mm512_maskz_loadu_ps(m, r1+iw), a1);
                            a2 = _mm512_fmadd_ps(_mm512_set1_ps(wc[10 +kw]), _mm512_maskz_loadu_ps(m, r2+iw), a2);
                            a3 = _mm512_fmadd_ps(_mm512_set1_ps(wc[15 +kw]), _mm512_maskz_loadu_ps(m, r3+iw), a3);
                            a4 = _mm512_fmadd_ps(_mm512_set1_ps(wc[20 +kw]), _mm512_maskz_loadu_ps(m, r4+iw), a4);
                        }
                        _mm512_storeu_ps(yc + oh*oW + ow, _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(_mm512_add_ps(a2, a3), a4)));
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
                    for (; ow + 16 <= oW; ow += 16) {
                        int iw0 = ow - pW;
                        __m512 acc = vbias;
                        for (int kh = 0; kh < kH; ++kh) {
                            int ih = ih0 + kh;
                            if (ih < 0 || ih >= iH) continue;
                            const float* xrow = xc + ih * iW;
                            for (int kw = 0; kw < kW; ++kw) {
                                int iw = iw0 + kw;
                                __m512 wv = _mm512_set1_ps(wc[kh * kW + kw]);
                                if (iw >= 0 && iw + 15 < iW) {
                                    acc = _mm512_fmadd_ps(wv,
                                        _mm512_loadu_ps(xrow + iw), acc);
                                } else {
                                    int mf = std::max(0, -iw), ml = std::min(16, iW - iw);
                                    __mmask16 mask = (ml > mf) ? (__mmask16)((0xFFFFu >> (16 - ml + mf)) << mf) : 0;
                                    acc = _mm512_fmadd_ps(wv,
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
            }
            if (post_fn) {
                int toff = (int)(yc - yd);
                post_fn(yc, 1, oH * oW, oH * oW, fused_op, nullptr, toff);
            }
        });
    }
    return true;
}

// AVX-512 float depthwise conv, stride=2, dilation=1 specialization.
// 16 output elements need input at stride-2 positions: iw, iw+2, ..., iw+30.
// Interior: load 32 contiguous floats, permutex2var to extract even indices.
// Boundary: masked gather.
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=DW fusion=post_op
inline bool depthwise_2d_s2(
    tensor_t* y, const tensor_t* x, const tensor_t* w, float* bias,
    int pH, int pW,
    operator_t::post_fn_t post_fn, const operator_t* fused_op,
    arena_t& arena)
{
    const int kH = w->dims[2], kW = w->dims[3];
    const int iH = x->dims[2], iW = x->dims[3];
    const int oN = y->dims[0], oC = y->dims[1], oH = y->dims[2], oW = y->dims[3];
    const float* xd = (const float*)x->data;
    float* yd = (float*)y->data;
    const float* wd = (const float*)w->data;

    const __m512i perm_s2 = _mm512_set_epi32(
        30, 28, 26, 24, 22, 20, 18, 16,
        14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i gather_s2 = _mm512_set_epi32(
        30, 28, 26, 24, 22, 20, 18, 16,
        14, 12, 10, 8, 6, 4, 2, 0);

    const int iC = x->dims[1];
    if (oW <= 16) {
        __mmask16 omask = (__mmask16)((1u << oW) - 1);
        arena_vector<__mmask16> kmasks(kH * kW, __mmask16{0}, arena_allocator<__mmask16>{arena});
        for (int ki = 0; ki < kH * kW; ++ki) {
            int iw = -pW + (ki % kW);
            int mf = std::max(0, (-iw + 1) / 2), ml = std::min(oW, (iW - iw + 1) / 2);
            kmasks[ki] = (ml > mf) ? (__mmask16)(((1u << ml) - 1) & ~((1u << mf) - 1)) : 0;
        }
        nnr::for_static(0, oN * oC, oN * oC > 4, [&, omask, kmasks](int nc) {
            int n = nc / oC, c = nc % oC;
            int ic = (int)((size_t)c * iC / oC);
            const float* xc = xd + ((size_t)n * iC + ic) * iH * iW;
            const float* wc = wd + (size_t)c * kH * kW;
            float* yc = yd + ((size_t)n * oC + c) * oH * oW;
            __m512 vbias = _mm512_set1_ps(bias ? bias[c] : 0.0f);
#if 0 // LOCUST
;for KH, KW in [(5, 5), (3, 3)]:
;    if KH == 5:
                if (kH == 5 && kW == 5) {
;        pass
;    else:
                } else if (kH == 3 && kW == 3) {
;        pass
                    for (int oh = 0; oh < oH; ++oh) {
                        int ih0 = oh * 2 - pH;
;    accs_decl = ", ".join(f"a{kh} = _mm512_setzero_ps()" for kh in range(1, KH))
                        __m512 a0 = vbias, @accs_decl@;
                        alignas(64) static const float zeros[512] = {};
;    for kh in range(KH):
                        const float* r@kh@ = (ih0 + @kh@ >= 0 && ih0 + @kh@ < iH) ? xc + (ih0 + @kh@) * iW - pW : zeros;
;        pass
                        for (int kw = 0; kw < @KW@; ++kw) {
;    for kh in range(KH):
;        WI_BASE = kh * KW
                            __m512 w@kh@ = _mm512_set1_ps(wc[@WI_BASE@ + kw]);
;        pass
;    for kh in range(KH):
;        WI_BASE = kh * KW
                            __m512 v@kh@ = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), kmasks[@WI_BASE@ + kw], gather_s2, r@kh@ + kw, 4);
;        pass
;    for kh in range(KH):
                            a@kh@ = _mm512_fmadd_ps(w@kh@, v@kh@, a@kh@);
;        pass
                        }
;    if KH == 5:
;        reduce = "_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(_mm512_add_ps(a2, a3), a4))"
;    else:
;        reduce = "_mm512_add_ps(_mm512_add_ps(a0, a1), a2)"
;    pass
                        _mm512_mask_storeu_ps(yc + oh * oW, omask, @reduce@);
                    }
;    pass
#else // LOCUST
                if (kH == 5 && kW == 5) {
                    for (int oh = 0; oh < oH; ++oh) {
                        int ih0 = oh * 2 - pH;
                        __m512 a0 = vbias, a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps(), a4 = _mm512_setzero_ps();
                        alignas(64) static const float zeros[512] = {};
                        const float* r0 = (ih0 + 0 >= 0 && ih0 + 0 < iH) ? xc + (ih0 + 0) * iW - pW : zeros;
                        const float* r1 = (ih0 + 1 >= 0 && ih0 + 1 < iH) ? xc + (ih0 + 1) * iW - pW : zeros;
                        const float* r2 = (ih0 + 2 >= 0 && ih0 + 2 < iH) ? xc + (ih0 + 2) * iW - pW : zeros;
                        const float* r3 = (ih0 + 3 >= 0 && ih0 + 3 < iH) ? xc + (ih0 + 3) * iW - pW : zeros;
                        const float* r4 = (ih0 + 4 >= 0 && ih0 + 4 < iH) ? xc + (ih0 + 4) * iW - pW : zeros;
                        for (int kw = 0; kw < 5; ++kw) {
                            __m512 w0 = _mm512_set1_ps(wc[0 + kw]);
                            __m512 w1 = _mm512_set1_ps(wc[5 + kw]);
                            __m512 w2 = _mm512_set1_ps(wc[10 + kw]);
                            __m512 w3 = _mm512_set1_ps(wc[15 + kw]);
                            __m512 w4 = _mm512_set1_ps(wc[20 + kw]);
                            __m512 v0 = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), kmasks[0 + kw], gather_s2, r0 + kw, 4);
                            __m512 v1 = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), kmasks[5 + kw], gather_s2, r1 + kw, 4);
                            __m512 v2 = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), kmasks[10 + kw], gather_s2, r2 + kw, 4);
                            __m512 v3 = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), kmasks[15 + kw], gather_s2, r3 + kw, 4);
                            __m512 v4 = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), kmasks[20 + kw], gather_s2, r4 + kw, 4);
                            a0 = _mm512_fmadd_ps(w0, v0, a0);
                            a1 = _mm512_fmadd_ps(w1, v1, a1);
                            a2 = _mm512_fmadd_ps(w2, v2, a2);
                            a3 = _mm512_fmadd_ps(w3, v3, a3);
                            a4 = _mm512_fmadd_ps(w4, v4, a4);
                        }
                        _mm512_mask_storeu_ps(yc + oh * oW, omask, _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(_mm512_add_ps(a2, a3), a4)));
                    }
                } else if (kH == 3 && kW == 3) {
                    for (int oh = 0; oh < oH; ++oh) {
                        int ih0 = oh * 2 - pH;
                        __m512 a0 = vbias, a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps();
                        alignas(64) static const float zeros[512] = {};
                        const float* r0 = (ih0 + 0 >= 0 && ih0 + 0 < iH) ? xc + (ih0 + 0) * iW - pW : zeros;
                        const float* r1 = (ih0 + 1 >= 0 && ih0 + 1 < iH) ? xc + (ih0 + 1) * iW - pW : zeros;
                        const float* r2 = (ih0 + 2 >= 0 && ih0 + 2 < iH) ? xc + (ih0 + 2) * iW - pW : zeros;
                        for (int kw = 0; kw < 3; ++kw) {
                            __m512 w0 = _mm512_set1_ps(wc[0 + kw]);
                            __m512 w1 = _mm512_set1_ps(wc[3 + kw]);
                            __m512 w2 = _mm512_set1_ps(wc[6 + kw]);
                            __m512 v0 = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), kmasks[0 + kw], gather_s2, r0 + kw, 4);
                            __m512 v1 = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), kmasks[3 + kw], gather_s2, r1 + kw, 4);
                            __m512 v2 = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), kmasks[6 + kw], gather_s2, r2 + kw, 4);
                            a0 = _mm512_fmadd_ps(w0, v0, a0);
                            a1 = _mm512_fmadd_ps(w1, v1, a1);
                            a2 = _mm512_fmadd_ps(w2, v2, a2);
                        }
                        _mm512_mask_storeu_ps(yc + oh * oW, omask, _mm512_add_ps(_mm512_add_ps(a0, a1), a2));
                    }
#endif // LOCUST
            } else {
                for (int oh = 0; oh < oH; ++oh) {
                    int ih0 = oh * 2 - pH;
                    __m512 acc = vbias;
                    for (int kh = 0; kh < kH; ++kh) {
                        int ih = ih0 + kh;
                        if (ih < 0 || ih >= iH) continue;
                        const float* xrow = xc + ih * iW - pW;
                        for (int kw = 0; kw < kW; ++kw) {
                            __mmask16 m = kmasks[kh * kW + kw];
                            __m512 v = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), m, gather_s2, xrow + kw, 4);
                            acc = _mm512_fmadd_ps(_mm512_set1_ps(wc[kh * kW + kw]), v, acc);
                        }
                    }
                    _mm512_mask_storeu_ps(yc + oh * oW, omask, acc);
                }
            }
            if (post_fn) {
                int toff = (int)(yc - yd);
                post_fn(yc, 1, oH * oW, oH * oW, fused_op, nullptr, toff);
            }
        });
    } else {
        nnr::for_static(0, oN * oC, oN * oC > 4, [&](int nc) {
            int n = nc / oC, c = nc % oC;
            int ic = (int)((size_t)c * iC / oC);
            const float* xc = xd + ((size_t)n * iC + ic) * iH * iW;
            const float* wc = wd + (size_t)c * kH * kW;
            float* yc = yd + ((size_t)n * oC + c) * oH * oW;
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
                            __m512 wv = _mm512_set1_ps(wc[kh * kW + kw]);
                            if (iw >= 0 && iw + 31 < iW) {
                                __m512 lo = _mm512_loadu_ps(xrow + iw);
                                __m512 hi = _mm512_loadu_ps(xrow + iw + 16);
                                __m512 vals = _mm512_permutex2var_ps(lo, perm_s2, hi);
                                acc = _mm512_fmadd_ps(wv, vals, acc);
                            } else {
                                int mf = std::max(0, (-iw + 1) / 2), ml = std::min(16, (iW - iw + 1) / 2);
                                __mmask16 mask = (ml > mf) ? (__mmask16)((0xFFFFu >> (16 - ml + mf)) << mf) : 0;
                                __m512 vals = _mm512_mask_i32gather_ps(
                                    _mm512_setzero_ps(), mask, gather_s2, xrow + iw, 4);
                                acc = _mm512_fmadd_ps(wv, vals, acc);
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
            if (post_fn) {
                int toff = (int)(yc - yd);
                post_fn(yc, 1, oH * oW, oH * oW, fused_op, nullptr, toff);
            }
        });
    }
    return true;
}

} // namespace nnr::avx512
