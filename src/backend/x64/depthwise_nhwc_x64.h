#pragma once
// x64 AVX-512/AVX2 NHWC depthwise convolution: per-pixel channel loop.
// Called from Conv_depthwise.h.

#ifdef NNR_ARCH_X64
#include <immintrin.h>
#include "cpu_features.h"

namespace nnr {

// Process one output pixel's channels for NHWC depthwise conv.
// xn: input base [iH, iW, C], w_repacked: [kH, kW, C], out: output [C].
// Returns the channel position after vectorized processing.
// @nnr-meta isa=[AVX512,AVX2] dtype=fp32 layout=NHWC special=DW
inline int depthwise_nhwc_pixel_x64(
    float* out, const float* xn, const float* w_repacked, const float* bias,
    int C, int kH, int kW, int iH, int iW, int dH, int dW,
    int ih0, int iw0)
{
    int c = 0;
    if (has_avx512()) {
        for (; c + 64 <= C; c += 64) {
            __m512 acc0 = bias ? _mm512_loadu_ps(bias + c)      : _mm512_setzero_ps();
            __m512 acc1 = bias ? _mm512_loadu_ps(bias + c + 16) : _mm512_setzero_ps();
            __m512 acc2 = bias ? _mm512_loadu_ps(bias + c + 32) : _mm512_setzero_ps();
            __m512 acc3 = bias ? _mm512_loadu_ps(bias + c + 48) : _mm512_setzero_ps();
            for (int kh = 0; kh < kH; kh++) {
                int ih = ih0 + kh * dH;
                if (ih < 0 || ih >= iH) continue;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = iw0 + kw * dW;
                    if (iw < 0 || iw >= iW) continue;
                    size_t xoff = (size_t)(ih * iW + iw) * C + c;
                    size_t woff = (size_t)(kh * kW + kw) * C + c;
                    acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(xn + xoff),      _mm512_loadu_ps(w_repacked + woff),      acc0);
                    acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(xn + xoff + 16), _mm512_loadu_ps(w_repacked + woff + 16), acc1);
                    acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(xn + xoff + 32), _mm512_loadu_ps(w_repacked + woff + 32), acc2);
                    acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(xn + xoff + 48), _mm512_loadu_ps(w_repacked + woff + 48), acc3);
                }
            }
            _mm512_storeu_ps(out + c,      acc0);
            _mm512_storeu_ps(out + c + 16, acc1);
            _mm512_storeu_ps(out + c + 32, acc2);
            _mm512_storeu_ps(out + c + 48, acc3);
        }
        for (; c + 16 <= C; c += 16) {
            __m512 acc = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
            for (int kh = 0; kh < kH; kh++) {
                int ih = ih0 + kh * dH;
                if (ih < 0 || ih >= iH) continue;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = iw0 + kw * dW;
                    if (iw < 0 || iw >= iW) continue;
                    acc = _mm512_fmadd_ps(
                        _mm512_loadu_ps(xn + (ih * iW + iw) * C + c),
                        _mm512_loadu_ps(w_repacked + (kh * kW + kw) * C + c),
                        acc);
                }
            }
            _mm512_storeu_ps(out + c, acc);
        }
        // Masked tail for leftover channels (< 16). Critical for mobilenet
        // shapes where C ∈ {24, 40, 72, 120, 240, ...} leaves 8 channels in
        // the scalar fallback — that scalar path dominates DW runtime.
        if (c < C) {
            int rem = C - c;
            __mmask16 m = (__mmask16)((1u << rem) - 1);
            __m512 acc = bias ? _mm512_maskz_loadu_ps(m, bias + c) : _mm512_setzero_ps();
            for (int kh = 0; kh < kH; kh++) {
                int ih = ih0 + kh * dH;
                if (ih < 0 || ih >= iH) continue;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = iw0 + kw * dW;
                    if (iw < 0 || iw >= iW) continue;
                    acc = _mm512_fmadd_ps(
                        _mm512_maskz_loadu_ps(m, xn + (ih * iW + iw) * C + c),
                        _mm512_maskz_loadu_ps(m, w_repacked + (kh * kW + kw) * C + c),
                        acc);
                }
            }
            _mm512_mask_storeu_ps(out + c, m, acc);
            c = C;
        }
    } else if (detect_isa() == isa_t::avx2) {
        for (; c + 32 <= C; c += 32) {
            __m256 acc0 = bias ? _mm256_loadu_ps(bias + c)      : _mm256_setzero_ps();
            __m256 acc1 = bias ? _mm256_loadu_ps(bias + c + 8)  : _mm256_setzero_ps();
            __m256 acc2 = bias ? _mm256_loadu_ps(bias + c + 16) : _mm256_setzero_ps();
            __m256 acc3 = bias ? _mm256_loadu_ps(bias + c + 24) : _mm256_setzero_ps();
            for (int kh = 0; kh < kH; kh++) {
                int ih = ih0 + kh * dH;
                if (ih < 0 || ih >= iH) continue;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = iw0 + kw * dW;
                    if (iw < 0 || iw >= iW) continue;
                    size_t xoff = (size_t)(ih * iW + iw) * C + c;
                    size_t woff = (size_t)(kh * kW + kw) * C + c;
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(xn + xoff),      _mm256_loadu_ps(w_repacked + woff),      acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(xn + xoff + 8),  _mm256_loadu_ps(w_repacked + woff + 8),  acc1);
                    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(xn + xoff + 16), _mm256_loadu_ps(w_repacked + woff + 16), acc2);
                    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(xn + xoff + 24), _mm256_loadu_ps(w_repacked + woff + 24), acc3);
                }
            }
            _mm256_storeu_ps(out + c,      acc0);
            _mm256_storeu_ps(out + c + 8,  acc1);
            _mm256_storeu_ps(out + c + 16, acc2);
            _mm256_storeu_ps(out + c + 24, acc3);
        }
        for (; c + 8 <= C; c += 8) {
            __m256 acc = bias ? _mm256_loadu_ps(bias + c) : _mm256_setzero_ps();
            for (int kh = 0; kh < kH; kh++) {
                int ih = ih0 + kh * dH;
                if (ih < 0 || ih >= iH) continue;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = iw0 + kw * dW;
                    if (iw < 0 || iw >= iW) continue;
                    acc = _mm256_fmadd_ps(
                        _mm256_loadu_ps(xn + (ih * iW + iw) * C + c),
                        _mm256_loadu_ps(w_repacked + (kh * kW + kw) * C + c),
                        acc);
                }
            }
            _mm256_storeu_ps(out + c, acc);
        }
    }
    return c;
}

// NCHWc depthwise: process one output pixel with block=16 (AVX-512).
// @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=[DW,NCHWc]
inline void depthwise_nchwc_pixel_avx512(
    float* out, const float* inp, const float* wt, const float* bi,
    int kH, int kW, int iH, int iW, int oW, int block,
    int oh, int ow, int sH, int sW, int pH, int pW)
{
    __m512 vbias = _mm512_loadu_ps(bi);
    __m512 acc = vbias;
    int ih0 = oh * sH - pH;
    int iw0 = ow * sW - pW;
    for (int kh = 0; kh < kH; kh++) {
        int ih = ih0 + kh;
        if (ih < 0 || ih >= iH) continue;
        for (int kw = 0; kw < kW; kw++) {
            int iw = iw0 + kw;
            if (iw < 0 || iw >= iW) continue;
            acc = _mm512_fmadd_ps(
                _mm512_loadu_ps(&inp[(ih * iW + iw) * block]),
                _mm512_loadu_ps(&wt[(kh * kW + kw) * block]),
                acc);
        }
    }
    _mm512_storeu_ps(&out[(oh * oW + ow) * block], acc);
}

} // namespace nnr

#endif // NNR_ARCH_X64
