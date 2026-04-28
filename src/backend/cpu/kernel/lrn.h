#pragma once
// Standalone LRN (Local Response Normalization) kernel for codegen.
// y[n,c,h,w] = x[n,c,h,w] / pow(bias + alpha * sum(x[n,ci,h,w]^2), beta)
// where ci ranges over [max(0,c-half), min(C-1,c+half)]

#include "thread_pool.h"
#include <algorithm>
#include <cmath>

#include "cpu_features.h"

#ifdef NNR_ARCH_X64
#include "backend/x64/lrn_avx512.h"
#include "backend/x64/lrn_avx2.h"
#elifdef NNR_ARCH_ARM64
#include "backend/arm/lrn_neon.h"
#endif

namespace nnr {

// Per-channel LRN computation (called from threaded wrapper)
inline void lrn_channel(const float* __restrict input, float* __restrict output,
    int nc, int C, int spatial, int c0, int c1, float alpha, float beta, float bias)
{
    int hw = 0;

#ifdef NNR_ARCH_X64
    if (has_avx512()) {
        if (beta == 0.5f)
            lrn_channel_avx512(input, output, nc, C, spatial, c0, c1, alpha, bias, pow_neg_050{}, hw);
        else if (beta == 0.75f)
            lrn_channel_avx512(input, output, nc, C, spatial, c0, c1, alpha, bias, pow_neg_075{}, hw);
        else if (beta == 1.0f)
            lrn_channel_avx512(input, output, nc, C, spatial, c0, c1, alpha, bias, pow_neg_100{}, hw);
        else
            lrn_channel_avx512(input, output, nc, C, spatial, c0, c1, alpha, bias, pow_neg_general{beta}, hw);
    } else if (detect_isa() >= isa_t::avx2) {
        if (beta == 0.5f)
            lrn_channel_avx2(input, output, nc, C, spatial, c0, c1, alpha, bias, pow_neg_050_avx2{}, hw);
        else if (beta == 0.75f)
            lrn_channel_avx2(input, output, nc, C, spatial, c0, c1, alpha, bias, pow_neg_075_avx2{}, hw);
        else if (beta == 1.0f)
            lrn_channel_avx2(input, output, nc, C, spatial, c0, c1, alpha, bias, pow_neg_100_avx2{}, hw);
        else
            lrn_channel_avx2(input, output, nc, C, spatial, c0, c1, alpha, bias, pow_neg_general_avx2{beta}, hw);
    }
#elifdef NNR_ARCH_ARM64
    lrn_channel_neon(input, output, nc, C, spatial, c0, c1, alpha, beta, bias, hw);
#endif

    const float* xc = input + (size_t)nc * spatial;
    float* yc = output + (size_t)nc * spatial;
    int n = nc / C;
    for (; hw < spatial; ++hw) {
        float sum = 0;
        for (int ci = c0; ci <= c1; ++ci) {
            float v = input[((size_t)n * C + ci) * spatial + hw];
            sum += v * v;
        }
        yc[hw] = xc[hw] / powf(bias + alpha * sum, beta);
    }
}

inline void lrn(const float* __restrict input, float* __restrict output,
    int N, int C, int H, int W, int lrn_size, float alpha, float beta, float bias)
{
    const int spatial = H * W;
    const int half = lrn_size / 2;
    const int NC = N * C;

    nnr::for_static(0, NC, NC > 4, [&](int nc) {
        int c = nc % C;
        int c0 = std::max(0, c - half);
        int c1 = std::min(C - 1, c + half);
        lrn_channel(input, output, nc, C, spatial, c0, c1, alpha, beta, bias);
    });
}

} // namespace nnr
