#pragma once
// Standalone depthwise 2D convolution kernel for codegen and direct use.
// Weight shape: [C, 1, kH, kW], input shape: [N, C, iH, iW]

#include "nnr.h"
#include "thread_pool.h"
#include <algorithm>

#include "cpu_features.h"

#ifdef NNR_ARCH_X64
#include "backend/x64/depthwise_generic_x64.h"
#include "backend/x64/depthwise_avx2.h"
#elifdef NNR_ARCH_ARM64
#include "backend/arm/depthwise_neon.h"
#endif

namespace nnr {

// General depthwise conv2d (scalar, any stride/dilation).
// Optional post_fn is called per-channel on L1-hot output data.
inline void depthwise_conv2d(float* output, const float* input, const float* weight,
    const float* bias, int N, int C, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW,
    operator_t::post_fn_t post_fn = nullptr, const operator_t* fused_op = nullptr)
{
    int NC = N * C;
    int spatial = oH * oW;

#ifdef NNR_ARCH_X64
    if (depthwise_conv2d_x64(output, input, weight, bias, N, C, iH, iW, oH, oW,
            kH, kW, sH, sW, pH, pW, dH, dW, post_fn, fused_op))
        return;
#elifdef NNR_ARCH_ARM64
    if (has_neon() && dH == 1 && dW == 1) {
        tensor_t yt{}, xt{}, wt{};
        yt.dims[0] = N; yt.dims[1] = C; yt.dims[2] = oH; yt.dims[3] = oW;
        yt.data = output;
        xt.dims[0] = N; xt.dims[1] = C; xt.dims[2] = iH; xt.dims[3] = iW;
        xt.data = const_cast<void*>((const void*)input);
        wt.dims[0] = C; wt.dims[1] = 1; wt.dims[2] = kH; wt.dims[3] = kW;
        wt.data = const_cast<void*>((const void*)weight);
        if (sH == 1 && sW == 1) {
            neon::depthwise_2d(&yt, &xt, &wt, const_cast<float*>(bias), pH, pW, post_fn, fused_op);
            return;
        }
        if (sH == 2 && sW == 2) {
            neon::depthwise_2d_s2(&yt, &xt, &wt, const_cast<float*>(bias), pH, pW, post_fn, fused_op);
            return;
        }
    }
#endif

    // General scalar fallback
    nnr::for_static(0, NC, NC > 4, [&](int nc) {
        int c = nc % C;
        const float* xc = input + (size_t)nc * iH * iW;
        const float* wc = weight + (size_t)c * kH * kW;
        float* yc = output + (size_t)nc * spatial;
        float bv = bias ? bias[c] : 0.0f;
        for (int oh = 0; oh < oH; ++oh) {
            int ih0 = oh * sH - pH;
            for (int ow = 0; ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                float sum = bv;
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
        if (post_fn) post_fn(yc, 1, spatial, spatial, fused_op, nullptr, 0);
    });
}

// Strip variant: compute only output rows [oh_start, oh_end) per channel.
// Used by scroll tiling to keep intermediate data in L2 cache.
inline void depthwise_conv2d_strip(float* output, const float* input, const float* weight,
    const float* bias, int N, int C, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW,
    int oh_start, int oh_end,
    operator_t::post_fn_t post_fn = nullptr, const operator_t* fused_op = nullptr)
{
    int NC = N * C;
    int spatial = oH * oW;
    int strip_len = (oh_end - oh_start) * oW;

#ifdef NNR_ARCH_X64
    if (depthwise_conv2d_strip_x64(output, input, weight, bias, N, C, iH, iW, oH, oW,
            kH, kW, sH, sW, pH, pW, dH, dW, oh_start, oh_end, post_fn, fused_op))
        return;
#elifdef NNR_ARCH_ARM64
    // ARM NEON strip variant: fall through to scalar for now.
    // The full depthwise_conv2d NEON path handles complete channels;
    // strip-level NEON can be added when scroll tiling is tuned on ARM.
#endif

    // General scalar fallback
    nnr::for_static(0, NC, NC > 4, [&](int nc) {
        int c = nc % C;
        const float* xc = input + (size_t)nc * iH * iW;
        const float* wc = weight + (size_t)c * kH * kW;
        float* yc = output + (size_t)nc * spatial;
        float bv = bias ? bias[c] : 0.0f;
        for (int oh = oh_start; oh < oh_end; ++oh) {
            int ih0 = oh * sH - pH;
            for (int ow = 0; ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                float sum = bv;
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
        if (post_fn) post_fn(yc + oh_start * oW, 1, strip_len, strip_len, fused_op, nullptr, 0);
    });
}

} // namespace nnr
