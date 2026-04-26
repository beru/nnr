#pragma once
#include "thread_pool.h"
#include "cpu_features.h"
#include <algorithm>
#include <limits>
#include <cfloat>

#ifdef NNR_ARCH_X64
#include "backend/x64/pool_x64.h"
#include "backend/x64/vec_ops_avx512.h"
#include "backend/x64/ops_x64.h"
#elifdef NNR_ARCH_ARM64
#include "backend/arm/pool_neon.h"
#include "backend/arm/conv_neon.h"
#endif

namespace nnr {

template <typename T>
inline void maxpool_2d(const T* input, T* output, int64_t* indices,
    int N, int C, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW)
{
    int NC = N * C;
    // Fast path: SIMD without indices
#ifdef NNR_ARCH_X64
    if constexpr (std::is_same_v<T, float>) {
        if (!indices && detect_isa() >= isa_t::avx2) {
            maxpool_2d_float_simd(input, output, NC, iH, iW, oH, oW,
                kH, kW, sH, sW, pH, pW);
            return;
        }
    }
    if constexpr (std::is_same_v<T, uint8_t>) {
        if (!indices && detect_isa() >= isa_t::avx512) {
            maxpool_2d_uint8_simd(input, output, NC, iH, iW, oH, oW,
                kH, kW, sH, sW, pH, pW);
            return;
        }
    }
#elifdef NNR_ARCH_ARM64
    if constexpr (std::is_same_v<T, float>) {
        if (!indices) {
            neon::maxpool_2d_neon(input, output, NC, iH, iW, oH, oW,
                kH, kW, sH, sW, pH, pW);
            return;
        }
    }
#endif
    nnr::for_static(0, NC, NC > 4 && oH * oW > 64, [&](int nc) {
        const T* inp = input + (size_t)nc * iH * iW;
        T* out = output + (size_t)nc * oH * oW;
        int64_t* idx_out = indices ? indices + (size_t)nc * oH * oW : nullptr;
        for (int oh = 0; oh < oH; ++oh) {
            int ih0 = oh * sH - pH;
            int kh0 = std::max(0, -ih0), kh1 = std::min(kH, iH - ih0);
            for (int ow = 0; ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
                T maxv = std::numeric_limits<T>::lowest();
                int64_t max_idx = 0;
                for (int kh = kh0; kh < kh1; ++kh)
                    for (int kw = kw0; kw < kw1; ++kw) {
                        T v = inp[(ih0 + kh) * iW + (iw0 + kw)];
                        if (v > maxv) {
                            maxv = v;
                            max_idx = (size_t)nc * iH * iW + (ih0 + kh) * iW + (iw0 + kw);
                        }
                    }
                out[oh * oW + ow] = maxv;
                if (idx_out) idx_out[oh * oW + ow] = max_idx;
            }
        }
    });
}

template <typename T>
inline void global_avgpool(const T* input, T* output, int N, int C, int spatial)
{
    T inv = T(1) / T(spatial);
    for (int n = 0; n < N; ++n) {
        const T* xn = input + (size_t)n * C * spatial;
#ifdef NNR_ARCH_X64
if constexpr (std::is_same_v<T, float>) {
            if (has_avx512()) {
                for (int c = 0; c < C; ++c)
                    output[n * C + c] = avx512::reduce_sum(xn + (size_t)c * spatial, spatial) * inv;
                continue;
            }
        }
#elifdef NNR_ARCH_ARM64
        if constexpr (std::is_same_v<T, float>) {
            neon::global_avgpool_neon(xn, output + (size_t)n * C, 1, C, spatial);
            continue;
        }
#endif
        for (int c = 0; c < C; ++c) {
            const T* xc = xn + (size_t)c * spatial;
            T sum = T(0);
            for (int s = 0; s < spatial; ++s)
                sum += xc[s];
            output[n * C + c] = sum * inv;
        }
    }
}

// NHWC global average pool: input [N, H, W, C] -> output [N, C].
// Vectorizes across channels (inner dimension) and accumulates over spatial.
inline void global_avgpool_nhwc(const float* input, float* output, int N, int C, int spatial)
{
    float inv = 1.0f / (float)spatial;
    for (int n = 0; n < N; ++n) {
        const float* xn = input + (size_t)n * C * spatial;
        float* yn = output + (size_t)n * C;
        int c = 0;
#ifdef NNR_ARCH_X64
        c = global_avgpool_nhwc_x64(xn, yn, C, spatial, inv);
#elifdef NNR_ARCH_ARM64
        c = global_avgpool_nhwc_neon(xn, yn, C, spatial, inv);
#endif
        for (; c < C; ++c) {
            float sum = 0;
            for (int s = 0; s < spatial; ++s)
                sum += xn[(size_t)s * C + c];
            yn[c] = sum * inv;
        }
    }
}

template <typename T>
inline void avgpool_2d(const T* input, T* output,
    int N, int C, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW,
    bool count_include_pad = false)
{
    int NC = N * C;
    nnr::for_static(0, NC, NC > 4 && oH * oW > 64, [&](int nc) {
        const T* inp = input + (size_t)nc * iH * iW;
        T* out = output + (size_t)nc * oH * oW;
        for (int oh = 0; oh < oH; ++oh) {
            int ih0 = oh * sH - pH;
            for (int ow = 0; ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                T sum = T(0);
                int count = 0;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = ih0 + kh;
                    if (ih < 0 || ih >= iH) {
                        if (count_include_pad) count += kW;
                        continue;
                    }
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = iw0 + kw;
                        if (iw < 0 || iw >= iW) {
                            if (count_include_pad) ++count;
                            continue;
                        }
                        sum += inp[ih * iW + iw];
                        ++count;
                    }
                }
                out[oh * oW + ow] = count > 0 ? sum / T(count) : T(0);
            }
        }
    });
}

// NHWC maxpool: input/output [N, H, W, C], channels contiguous for SIMD.
template <typename T>
// `out_ldc` (default 0 → C): output per-spatial-position stride. Set to
// parent_C for NHWC channel-axis Concat alias where this MaxPool writes a
// sub-channel stripe into a wider parent buffer.
inline void maxpool_2d_nhwc(const T* input, T* output,
    int N, int C, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW,
    int out_ldc = 0)
{
    if (out_ldc == 0) out_ldc = C;
    nnr::for_static(0, N * oH, N * oH > 4, [&](int noh) {
        int n = noh / oH, oh = noh % oH;
        const T* xn = input + (size_t)n * iH * iW * C;
        T* yn = output + (size_t)n * oH * oW * out_ldc;
        int ih0 = oh * sH - pH;
        int kh0 = std::max(0, -ih0), kh1 = std::min(kH, iH - ih0);
        for (int ow = 0; ow < oW; ow++) {
            int iw0 = ow * sW - pW;
            int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
            T* out = yn + (oh * oW + ow) * out_ldc;
            for (int c = 0; c < C; c++) {
                T maxv = std::numeric_limits<T>::lowest();
                for (int kh = kh0; kh < kh1; kh++)
                    for (int kw = kw0; kw < kw1; kw++) {
                        T v = xn[((ih0 + kh) * iW + (iw0 + kw)) * C + c];
                        if (v > maxv) maxv = v;
                    }
                out[c] = maxv;
            }
        }
    });
}

// NHWC avgpool: input/output [N, H, W, C], channels contiguous for SIMD.
// `out_ldc` (default 0 → C): NHWC channel-axis Concat alias support — output
// per-spatial-position stride is parent_C instead of local C.
template <typename T>
inline void avgpool_2d_nhwc(const T* input, T* output,
    int N, int C, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW,
    bool count_include_pad = false, int out_ldc = 0)
{
    if (out_ldc == 0) out_ldc = C;
    nnr::for_static(0, N * oH, N * oH > 4, [&](int noh) {
        int n = noh / oH, oh = noh % oH;
        const T* xn = input + (size_t)n * iH * iW * C;
        T* yn = output + (size_t)n * oH * oW * out_ldc;
        int ih0 = oh * sH - pH;
        for (int ow = 0; ow < oW; ow++) {
            int iw0 = ow * sW - pW;
            T* out = yn + (oh * oW + ow) * out_ldc;
            int valid = 0;
            for (int kh = 0; kh < kH; kh++) {
                int ih = ih0 + kh;
                if (ih < 0 || ih >= iH) continue;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = iw0 + kw;
                    if (iw < 0 || iw >= iW) continue;
                    valid++;
                }
            }
            float divisor = count_include_pad ? (float)(kH * kW) : (valid > 0 ? (float)valid : 1.0f);
            float inv = 1.0f / divisor;
            for (int c = 0; c < C; c++) {
                T sum = T(0);
                for (int kh = 0; kh < kH; kh++) {
                    int ih = ih0 + kh;
                    if (ih < 0 || ih >= iH) continue;
                    for (int kw = 0; kw < kW; kw++) {
                        int iw = iw0 + kw;
                        if (iw < 0 || iw >= iW) continue;
                        sum += xn[(ih * iW + iw) * C + c];
                    }
                }
                out[c] = (T)((float)sum * inv);
            }
        }
    });
}

// Strip variant: compute only output rows [oh_start, oh_end).
template <typename T>
inline void maxpool_2d_strip(const T* input, T* output,
    int N, int C, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW,
    int oh_start, int oh_end)
{
    int NC = N * C;
    nnr::for_static(0, NC, NC > 4, [&](int nc) {
        const T* inp = input + (size_t)nc * iH * iW;
        T* out = output + (size_t)nc * oH * oW;
        for (int oh = oh_start; oh < oh_end; ++oh) {
            int ih0 = oh * sH - pH;
            int kh0 = std::max(0, -ih0), kh1 = std::min(kH, iH - ih0);
            for (int ow = 0; ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
                T maxv = std::numeric_limits<T>::lowest();
                for (int kh = kh0; kh < kh1; ++kh)
                    for (int kw = kw0; kw < kw1; ++kw) {
                        T v = inp[(ih0 + kh) * iW + (iw0 + kw)];
                        if (v > maxv) maxv = v;
                    }
                out[oh * oW + ow] = maxv;
            }
        }
    });
}

// Strip variant: compute only output rows [oh_start, oh_end).
template <typename T>
inline void avgpool_2d_strip(const T* input, T* output,
    int N, int C, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW,
    int oh_start, int oh_end, bool count_include_pad = false)
{
    int NC = N * C;
    nnr::for_static(0, NC, NC > 4, [&](int nc) {
        const T* inp = input + (size_t)nc * iH * iW;
        T* out = output + (size_t)nc * oH * oW;
        for (int oh = oh_start; oh < oh_end; ++oh) {
            int ih0 = oh * sH - pH;
            for (int ow = 0; ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                T sum = T(0);
                int count = 0;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = ih0 + kh;
                    if (ih < 0 || ih >= iH) {
                        if (count_include_pad) count += kW;
                        continue;
                    }
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = iw0 + kw;
                        if (iw < 0 || iw >= iW) {
                            if (count_include_pad) ++count;
                            continue;
                        }
                        sum += inp[ih * iW + iw];
                        ++count;
                    }
                }
                out[oh * oW + ow] = count > 0 ? sum / T(count) : T(0);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// NCHWc (BLOCKED_16) MaxPool — scalar fallback.
// input:  [N, Cb, iH, iW, block]  where Cb = C / block
// output: [N, Cb, oH, oW, block]
// ---------------------------------------------------------------------------
inline void maxpool_2d_nchwc(const float* input, float* output,
    int N, int C, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW, int block)
{
    const int Cb = C / block;
    const size_t in_spatial  = (size_t)iH * iW * block;
    const size_t out_spatial = (size_t)oH * oW * block;

    const int NCb = N * Cb;
    nnr::for_static(0, NCb, NCb > 1, [&](int ncb) {
        const float* inp = input + (size_t)ncb * in_spatial;
        float* out = output + (size_t)ncb * out_spatial;
        for (int oh = 0; oh < oH; ++oh) {
            int ih0 = oh * sH - pH;
            int kh0 = std::max(0, -ih0), kh1 = std::min(kH, iH - ih0);
            for (int ow = 0; ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
                float* dst = out + (oh * oW + ow) * block;
                for (int c = 0; c < block; ++c) dst[c] = -FLT_MAX;
                for (int kh = kh0; kh < kh1; ++kh)
                    for (int kw = kw0; kw < kw1; ++kw) {
                        const float* src = inp + ((ih0 + kh) * iW + (iw0 + kw)) * block;
                        for (int c = 0; c < block; ++c)
                            dst[c] = std::max(dst[c], src[c]);
                    }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// NCHWc (BLOCKED_16) AveragePool — scalar fallback.
// ---------------------------------------------------------------------------
inline void avgpool_2d_nchwc(const float* input, float* output,
    int N, int C, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW,
    bool count_include_pad, int block)
{
    const int Cb = C / block;
    const size_t in_spatial  = (size_t)iH * iW * block;
    const size_t out_spatial = (size_t)oH * oW * block;

    const int NCb = N * Cb;
    nnr::for_static(0, NCb, NCb > 1, [&](int ncb) {
        const float* inp = input + (size_t)ncb * in_spatial;
        float* out = output + (size_t)ncb * out_spatial;
        for (int oh = 0; oh < oH; ++oh) {
            int ih0 = oh * sH - pH;
            int kh0 = std::max(0, -ih0), kh1 = std::min(kH, iH - ih0);
            for (int ow = 0; ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
                float* dst = out + (oh * oW + ow) * block;
                for (int c = 0; c < block; ++c) dst[c] = 0.0f;
                int valid = 0;
                for (int kh = kh0; kh < kh1; ++kh)
                    for (int kw = kw0; kw < kw1; ++kw) {
                        const float* src = inp + ((ih0 + kh) * iW + (iw0 + kw)) * block;
                        for (int c = 0; c < block; ++c)
                            dst[c] += src[c];
                        ++valid;
                    }
                float div = count_include_pad ? (float)(kH * kW) : (valid > 0 ? (float)valid : 1.0f);
                float inv = 1.0f / div;
                for (int c = 0; c < block; ++c)
                    dst[c] *= inv;
            }
        }
    });
}

// ---------------------------------------------------------------------------
// NCHWc (BLOCKED_16) GlobalAveragePool — scalar fallback.
// input:  [N, Cb, H, W, block]
// output: [N, Cb, 1, 1, block]  (= [N, Cb, block])
// ---------------------------------------------------------------------------
inline void global_avgpool_nchwc(const float* input, float* output,
    int N, int C, int spatial, int block)
{
    const int Cb = C / block;
    const int NCb = N * Cb;
    nnr::for_static(0, NCb, NCb > 1, [&](int ncb) {
        const float* inp = input + (size_t)ncb * spatial * block;
        float* out = output + (size_t)ncb * block;
        for (int c = 0; c < block; ++c) out[c] = 0.0f;
        for (int s = 0; s < spatial; ++s) {
            const float* src = inp + s * block;
            for (int c = 0; c < block; ++c)
                out[c] += src[c];
        }
        float inv = 1.0f / (float)spatial;
        for (int c = 0; c < block; ++c)
            out[c] *= inv;
    });
}

} // namespace nnr
