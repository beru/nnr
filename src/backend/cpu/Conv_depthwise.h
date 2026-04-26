#pragma once
// Depthwise 2D convolution implementation, included from Conv.cpp.
// Requires: Conv_operator members (strides, dilations, cpads, post_fn, fused_op)
// Requires: backend/x64/depthwise_avx512.h included before this file (from Conv.cpp)

// Requires backend/x64/depthwise_nhwc_x64.h or backend/arm/depthwise_nhwc_neon.h
// to be included by Conv.cpp before this file.
#include "profiler.h"

template <typename T>
bool exec_depthwise_2d(tensor_t* y, const tensor_t* x, const tensor_t* w, T* bias) {
    NNR_PROFILE_SCOPE("depthwise_nchw");
    const int kH = w->dims[2], kW = w->dims[3];
    const int iH = x->dims[2], iW = x->dims[3];
    const int oN = y->dims[0], oC = y->dims[1], oH = y->dims[2], oW = y->dims[3];
    const T* xd = (const T*)x->data;
    T* yd = (T*)y->data;
    const T* wd = (const T*)w->data;
    const int sH = strides[0], sW = strides[1];
    const int dH = dilations[0], dW = dilations[1];
    const int pH = cpads[0], pW = cpads[1];

#ifdef NNR_ARCH_X64
if constexpr (std::is_same_v<T, float>) {
        if (has_avx512() && dH == 1 && dW == 1) {
            if (sH == 1 && sW == 1) {
                return avx512::depthwise_2d(y, x, w, (float*)bias, pH, pW,
                    post_fn, fused_op, ctx->arena);
            }
            if (sH == 2 && sW == 2) {
                return avx512::depthwise_2d_s2(y, x, w, (float*)bias, pH, pW,
                    post_fn, fused_op, ctx->arena);
            }
        }
        if (detect_isa() == isa_t::avx2 && dH == 1 && dW == 1) {
            if (sH == 1 && sW == 1) {
                return avx2::depthwise_2d(y, x, w, (float*)bias, pH, pW,
                    post_fn, fused_op);
            }
            if (sH == 2 && sW == 2) {
                return avx2::depthwise_2d_s2(y, x, w, (float*)bias, pH, pW,
                    post_fn, fused_op);
            }
        }
    }
#elifdef NNR_ARCH_ARM64
    if constexpr (std::is_same_v<T, float>) {
        if (dH == 1 && dW == 1) {
            if (sH == 1 && sW == 1) {
                return neon::depthwise_2d(y, x, w, (float*)bias, pH, pW,
                    post_fn, fused_op);
            }
            if (sH == 2 && sW == 2) {
                return neon::depthwise_2d_s2(y, x, w, (float*)bias, pH, pW,
                    post_fn, fused_op);
            }
        }
    }
#endif

    const int iC = x->dims[1];
    nnr::for_static(0, oN * oC, oN * oC > 4, [&](int nc) {
        int n = nc / oC, c = nc % oC;
        int ic = (int)((size_t)c * iC / oC);
        const T* xc = xd + ((size_t)n * iC + ic) * iH * iW;
        const T* wc = wd + (size_t)c * kH * kW;
        T* yc = yd + ((size_t)n * oC + c) * oH * oW;
        T bv = bias ? bias[c] : T(0);
        for (int oh = 0; oh < oH; ++oh) {
            int ih0 = oh * sH - pH;
            for (int ow = 0; ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                T sum = bv;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = ih0 + kh * dH;
                    if (ih < 0 || ih >= iH) continue;
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = iw0 + kw * dW;
                        if (iw < 0 || iw >= iW) continue;
                        sum += xc[ih * iW + iw] * wc[kh * kW + kw];
                    }
                }
                yc[oh * oW + ow] = sum;
            }
        }
        // Per-channel fused post-op (scalar path)
        if constexpr (std::is_same_v<T, float>) {
            if (post_fn) {
                int toff = (int)(yc - yd);
                post_fn(yc, 1, oH * oW, oH * oW, fused_op, nullptr, toff);
            }
        }
    });
    return true;
}

// NHWC depthwise 2D convolution.
// Input/output: [N, H, W, C]. Weights repacked to [kH, kW, C] in w_dw_nhwc.
bool exec_depthwise_2d_nhwc(tensor_t* y, const tensor_t* x,
    const float* w_repacked, float* bias)
{
    NNR_PROFILE_SCOPE("depthwise_nhwc");
    const int N = y->dims[0], C = y->dims[1], oH = y->dims[2], oW = y->dims[3];
    const int iH = x->dims[2], iW = x->dims[3];
    const int kH = inputs[1]->dims[2], kW = inputs[1]->dims[3];
    const int sH = strides[0], sW = strides[1];
    const int dH = dilations[0], dW = dilations[1];
    const int pH = cpads[0], pW = cpads[1];

    const float* xd;
    if (x->format == memory_layout_t::NHWC) {
        xd = (const float*)x->data;
    } else {
        float* ws = (float*)ctx->workspace;
        nchw_to_nhwc(ws, (const float*)x->data, N, C, iH, iW);
        xd = ws;
    }
    float* yd = (float*)y->data;

    // Parallelize over N*oH*oW output pixels (not just N*oH rows) so that
    // small-spatial tensors (e.g., 7×7) utilize all threads.
    // N*oH tasks would leave threads idle when oH < thread_count.
    auto kernel = [&](int idx) {
        int n  = idx / (oH * oW);
        int oh = (idx % (oH * oW)) / oW;
        int ow = idx % oW;
        const float* xn = xd + (size_t)n * iH * iW * C;
        float* yn = yd + (size_t)n * oH * oW * C;
        int ih0 = oh * sH - pH;
        int iw0 = ow * sW - pW;
        float* out = yn + (oh * oW + ow) * C;
        int c = 0;
#ifdef NNR_ARCH_X64
        c = nnr::depthwise_nhwc_pixel_x64(out, xn, w_repacked, bias,
            C, kH, kW, iH, iW, dH, dW, ih0, iw0);
#elifdef NNR_ARCH_ARM64
        c = nnr::depthwise_nhwc_pixel_neon(out, xn, w_repacked, bias,
            C, kH, kW, iH, iW, dH, dW, ih0, iw0);
#endif
        for (; c < C; c++) {
            float sum = bias ? bias[c] : 0.0f;
            for (int kh = 0; kh < kH; kh++) {
                int ih = ih0 + kh * dH;
                if (ih < 0 || ih >= iH) continue;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = iw0 + kw * dW;
                    if (iw < 0 || iw >= iW) continue;
                    sum += xn[(ih * iW + iw) * C + c] * w_repacked[(kh * kW + kw) * C + c];
                }
            }
            out[c] = sum;
        }
    };
    nnr::for_static(0, N * oH * oW, N * oH * oW > 4, kernel);

    if (post_fn) {
        int total = N * oH * oW * C;
        post_fn(yd, 1, total, total, fused_op, nullptr, 0);
    }

    y->format = memory_layout_t::NHWC;
    return true;
}

// ---------------------------------------------------------------------------
// NCHWc (native blocked) depthwise 2D convolution.
// Input/output: [N, Cb, H, W, block].  Weights repacked to [Cb, kH, kW, block].
// Each block-channel slice is processed independently with one SIMD
// accumulator per output position (AVX-512 ZMM, or 2 NEON qregs on ARM64).
// ---------------------------------------------------------------------------
bool exec_depthwise_2d_nchwc(tensor_t* y, const tensor_t* x,
    const float* w_repacked, const float* bias_packed)
{
    NNR_PROFILE_SCOPE("depthwise_nchwc");
    constexpr int block = NATIVE_BLOCK;  // 16 on x64, 8 on ARM64
    const int N = y->dims[0], C = y->dims[1];
    const int oH = y->dims[2], oW = y->dims[3];
    const int iH = x->dims[2], iW = x->dims[3];
    const int kH = inputs[1]->dims[2], kW = inputs[1]->dims[3];
    const int sH = strides[0], sW = strides[1];
    const int pH = cpads[0], pW = cpads[1];
    const int Cb = C / block;

    const float* xd = (const float*)x->data;
    float* yd = (float*)y->data;
    const size_t in_spatial  = (size_t)iH * iW * block;
    const size_t out_spatial = (size_t)oH * oW * block;

    const int NCb = N * Cb;
    nnr::for_static(0, NCb, NCb > 1, [&](int ncb) {
        const int cb = ncb % Cb;
        const float* inp = xd + (size_t)ncb * in_spatial;
        float* out = yd + (size_t)ncb * out_spatial;
        const float* wt = w_repacked + (size_t)cb * kH * kW * block;
        const float* bi = bias_packed + cb * block;

#ifdef NNR_ARCH_X64
        if (has_avx512()) {
            for (int oh = 0; oh < oH; ++oh)
                for (int ow = 0; ow < oW; ++ow)
                    nnr::depthwise_nchwc_pixel_avx512(out, inp, wt, bi,
                        kH, kW, iH, iW, oW, block, oh, ow, sH, sW, pH, pW);
        } else
#elifdef NNR_ARCH_ARM64
        if (true) {
            for (int oh = 0; oh < oH; ++oh)
                for (int ow = 0; ow < oW; ++ow)
                    nnr::depthwise_nchwc_pixel_neon(out, inp, wt, bi,
                        kH, kW, iH, iW, oW, block, oh, ow, sH, sW, pH, pW);
        } else
#endif
        {
            // Scalar fallback
            for (int oh = 0; oh < oH; ++oh) {
                int ih0 = oh * sH - pH;
                int kh0 = std::max(0, -ih0), kh1 = std::min(kH, iH - ih0);
                for (int ow = 0; ow < oW; ++ow) {
                    int iw0 = ow * sW - pW;
                    int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
                    float acc[block];
                    for (int c = 0; c < block; ++c) acc[c] = bi[c];
                    for (int kh = kh0; kh < kh1; ++kh)
                        for (int kw = kw0; kw < kw1; ++kw) {
                            const float* xi = &inp[((ih0 + kh) * iW + (iw0 + kw)) * block];
                            const float* wi = &wt[(kh * kW + kw) * block];
                            for (int c = 0; c < block; ++c)
                                acc[c] += xi[c] * wi[c];
                        }
                    float* dst = &out[(oh * oW + ow) * block];
                    for (int c = 0; c < block; ++c) dst[c] = acc[c];
                }
            }
        }

        // Fused post-op (Relu, Clip, etc.) on this block's output
        if (post_fn) {
            int total = oH * oW * block;
            int toff = (int)(out - yd);
            post_fn(out, 1, total, total, fused_op, nullptr, toff);
        }
    });

    y->format = NATIVE_BLOCKED_FMT;
    return true;
}

// Strip-mode depthwise NCHWc.  Iterates oh in [out_row_start, out_end).
// iH_logical is the un-ringed input height for boundary clamping; the
// ring-buffer virtual-pointer trick handles per-plane addressing.
bool exec_depthwise_2d_nchwc_strip(tensor_t* y, const tensor_t* x,
    const float* w_repacked, const float* bias_packed,
    int out_row_start, int out_end, int iH_logical)
{
    NNR_PROFILE_SCOPE("depthwise_nchwc_strip");
    constexpr int block = NATIVE_BLOCK;
    const int N = y->dims[0], C = y->dims[1];
    const int oH = y->dims[2], oW = y->dims[3];   // oH may be ring_H
    const int iH = iH_logical;                    // logical iH for bounds
    const int iW = x->dims[3];
    const int kH = inputs[1]->dims[2], kW = inputs[1]->dims[3];
    const int sH = strides[0], sW = strides[1];
    const int pH = cpads[0], pW = cpads[1];
    const int Cb = C / block;

    const float* xd = (const float*)x->data;
    float* yd = (float*)y->data;
    // Per-plane strides use the (possibly ringed) tensor heights so that
    // the virtual-pointer trick lands writes/reads in the correct ring slot.
    const size_t in_spatial  = (size_t)x->dims[2] * iW * block;
    const size_t out_spatial = (size_t)oH * oW * block;

    const int strip_rows = out_end - out_row_start;
    if (strip_rows <= 0) return true;

    const int NCb = N * Cb;
    nnr::for_static(0, NCb, NCb > 1, [&](int ncb) {
        const int cb = ncb % Cb;
        const float* inp = xd + (size_t)ncb * in_spatial;
        float* out = yd + (size_t)ncb * out_spatial;
        const float* wt = w_repacked + (size_t)cb * kH * kW * block;
        const float* bi = bias_packed + cb * block;

#ifdef NNR_ARCH_X64
        if (has_avx512()) {
            for (int oh = out_row_start; oh < out_end; ++oh)
                for (int ow = 0; ow < oW; ++ow)
                    nnr::depthwise_nchwc_pixel_avx512(out, inp, wt, bi,
                        kH, kW, iH, iW, oW, block, oh, ow, sH, sW, pH, pW);
        } else
#elifdef NNR_ARCH_ARM64
        if (true) {
            for (int oh = out_row_start; oh < out_end; ++oh)
                for (int ow = 0; ow < oW; ++ow)
                    nnr::depthwise_nchwc_pixel_neon(out, inp, wt, bi,
                        kH, kW, iH, iW, oW, block, oh, ow, sH, sW, pH, pW);
        } else
#endif
        {
            for (int oh = out_row_start; oh < out_end; ++oh) {
                int ih0 = oh * sH - pH;
                int kh0 = std::max(0, -ih0), kh1 = std::min(kH, iH - ih0);
                for (int ow = 0; ow < oW; ++ow) {
                    int iw0 = ow * sW - pW;
                    int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
                    float acc[block];
                    for (int c = 0; c < block; ++c) acc[c] = bi[c];
                    for (int kh = kh0; kh < kh1; ++kh)
                        for (int kw = kw0; kw < kw1; ++kw) {
                            const float* xi = &inp[((ih0 + kh) * iW + (iw0 + kw)) * block];
                            const float* wi = &wt[(kh * kW + kw) * block];
                            for (int c = 0; c < block; ++c)
                                acc[c] += xi[c] * wi[c];
                        }
                    float* dst = &out[(oh * oW + ow) * block];
                    for (int c = 0; c < block; ++c) dst[c] = acc[c];
                }
            }
        }

        // Per-Cb fused post-op on this strip's output rows.
        if (post_fn) {
            const int strip_total = strip_rows * oW * block;
            float* strip_out = &out[(size_t)out_row_start * oW * block];
            int toff = (int)(strip_out - yd);
            post_fn(strip_out, 1, strip_total, strip_total, fused_op, nullptr, toff);
        }
    });

    return true;
}
