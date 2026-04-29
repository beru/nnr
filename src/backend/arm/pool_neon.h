// ARM NEON optimized pool kernels (MaxPool, AvgPool, GlobalAvgPool)
#pragma once
#include <arm_neon.h>
#include <cfloat>
#include <algorithm>
#include "thread_pool.h"

namespace nnr {
namespace neon {

// NCHW MaxPool 2D: vectorize across output width (stride-1 fast path)
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW tiling=spatial
inline void maxpool_2d_neon(const float* input, float* output,
    int NC, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW)
{
    // Safe oW range where full kernel fits horizontally
    int ow_safe_lo = pW > 0 ? (pW + sW - 1) / sW : 0;
    int ow_safe_hi = std::min(oW, (iW + pW - kW) / sW + 1);

    nnr::for_static(0, NC, NC > 4 && oH * oW > 64, [&](int nc) {
        const float* inp = input + (size_t)nc * iH * iW;
        float* out = output + (size_t)nc * oH * oW;

        for (int oh = 0; oh < oH; ++oh) {
            int ih0 = oh * sH - pH;
            int kh0 = std::max(0, -ih0), kh1 = std::min(kH, iH - ih0);
            float* dst = out + oh * oW;

            // Left edge (scalar)
            for (int ow = 0; ow < ow_safe_lo && ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
                float maxv = -FLT_MAX;
                for (int kh = kh0; kh < kh1; ++kh)
                    for (int kw = kw0; kw < kw1; ++kw)
                        maxv = std::max(maxv, inp[(ih0 + kh) * iW + (iw0 + kw)]);
                dst[ow] = maxv;
            }

            // Interior (NEON)
            int ow = ow_safe_lo;

            if (sW == 1) {
                // Stride-1: shift+max across 4 oW positions
                for (; ow + 4 <= ow_safe_hi; ow += 4) {
                    int iw0 = ow - pW;
                    float32x4_t vmax = vdupq_n_f32(-FLT_MAX);
                    for (int kh = kh0; kh < kh1; ++kh) {
                        const float* row = inp + (ih0 + kh) * iW + iw0;
                        float32x4_t row_max = vld1q_f32(row);
                        for (int kw = 1; kw < kW; ++kw)
                            row_max = vmaxq_f32(row_max, vld1q_f32(row + kw));
                        vmax = vmaxq_f32(vmax, row_max);
                    }
                    vst1q_f32(dst + ow, vmax);
                }
            } else if (sW == 2) {
                // Stride-2: process 4 output pixels at a time
                // Each output pixel reads from input at iw = ow*2 - pW
                // So 4 output pixels need 8 consecutive input positions + (kW-1) extra
                // vld2q_f32 reads 8 contiguous floats at row+kw, so the last block's
                // top access is row[iw0 + (kW-1) + 7]. ow_safe_hi alone permits
                // reaching iW exactly (1 element past end); cap the SIMD loop
                // tighter and let the scalar tail handle the last block.
                int simd_hi = std::min(ow_safe_hi, (iW - kW + pW + 1) / 2);
                for (; ow + 4 <= simd_hi; ow += 4) {
                    int iw0 = ow * 2 - pW;
                    float32x4_t vmax = vdupq_n_f32(-FLT_MAX);
                    for (int kh = kh0; kh < kh1; ++kh) {
                        const float* row = inp + (ih0 + kh) * iW + iw0;
                        // Load 8 values, deinterleave to get even positions
                        float32x4x2_t pairs = vld2q_f32(row);
                        float32x4_t row_max = pairs.val[0]; // elements 0,2,4,6
                        for (int kw = 1; kw < kW; ++kw) {
                            pairs = vld2q_f32(row + kw);
                            row_max = vmaxq_f32(row_max, pairs.val[0]);
                        }
                        vmax = vmaxq_f32(vmax, row_max);
                    }
                    vst1q_f32(dst + ow, vmax);
                }
            }

            // Remainder + right edge (scalar)
            for (; ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
                float maxv = -FLT_MAX;
                for (int kh = kh0; kh < kh1; ++kh)
                    for (int kw = kw0; kw < kw1; ++kw)
                        maxv = std::max(maxv, inp[(ih0 + kh) * iW + (iw0 + kw)]);
                dst[ow] = maxv;
            }
        }
    });
}

// NCHW MaxPool 2D strip (scroll path)
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=Scroll tiling=spatial
inline void maxpool_2d_strip_neon(const float* input, float* output,
    int NC, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW,
    int iH_pad, int oH_clamp, int out_row_start, int out_end)
{
    int ow_safe_lo = pW > 0 ? (pW + sW - 1) / sW : 0;
    int ow_safe_hi = std::min(oW, (iW + pW - kW) / sW + 1);

    nnr::for_static(0, NC, NC > 4, [&](int nc) {
        const float* inp = input + (size_t)nc * iH * iW;
        float* out = output + (size_t)nc * oH * oW;

        for (int oh = out_row_start; oh < out_end; ++oh) {
            int ih0 = oh * sH - pH;
            int kh0 = std::max(0, -ih0), kh1 = std::min(kH, iH_pad - ih0);
            float* dst = out + oh * oW;

            // Left edge
            for (int ow = 0; ow < ow_safe_lo && ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
                float maxv = -FLT_MAX;
                for (int kh = kh0; kh < kh1; ++kh)
                    for (int kw = kw0; kw < kw1; ++kw)
                        maxv = std::max(maxv, inp[(ih0 + kh) * iW + (iw0 + kw)]);
                dst[ow] = maxv;
            }

            int ow = ow_safe_lo;
            if (sW == 1) {
                for (; ow + 4 <= ow_safe_hi; ow += 4) {
                    int iw0 = ow - pW;
                    float32x4_t vmax = vdupq_n_f32(-FLT_MAX);
                    for (int kh = kh0; kh < kh1; ++kh) {
                        const float* row = inp + (ih0 + kh) * iW + iw0;
                        float32x4_t row_max = vld1q_f32(row);
                        for (int kw = 1; kw < kW; ++kw)
                            row_max = vmaxq_f32(row_max, vld1q_f32(row + kw));
                        vmax = vmaxq_f32(vmax, row_max);
                    }
                    vst1q_f32(dst + ow, vmax);
                }
            } else if (sW == 2) {
                // See maxpool_2d_neon: vld2q reads 8 floats; tighten bound.
                int simd_hi = std::min(ow_safe_hi, (iW - kW + pW + 1) / 2);
                for (; ow + 4 <= simd_hi; ow += 4) {
                    int iw0 = ow * 2 - pW;
                    float32x4_t vmax = vdupq_n_f32(-FLT_MAX);
                    for (int kh = kh0; kh < kh1; ++kh) {
                        const float* row = inp + (ih0 + kh) * iW + iw0;
                        float32x4x2_t pairs = vld2q_f32(row);
                        float32x4_t row_max = pairs.val[0];
                        for (int kw = 1; kw < kW; ++kw) {
                            pairs = vld2q_f32(row + kw);
                            row_max = vmaxq_f32(row_max, pairs.val[0]);
                        }
                        vmax = vmaxq_f32(vmax, row_max);
                    }
                    vst1q_f32(dst + ow, vmax);
                }
            }

            for (; ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
                float maxv = -FLT_MAX;
                for (int kh = kh0; kh < kh1; ++kh)
                    for (int kw = kw0; kw < kw1; ++kw)
                        maxv = std::max(maxv, inp[(ih0 + kh) * iW + (iw0 + kw)]);
                dst[ow] = maxv;
            }
        }
    });
}

// NCHW GlobalAvgPool: reduce_sum per channel then multiply by inv
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW
inline void global_avgpool_neon(const float* input, float* output,
    int N, int C, int spatial)
{
    float inv = 1.0f / (float)spatial;
    for (int n = 0; n < N; ++n) {
        const float* xn = input + (size_t)n * C * spatial;
        for (int c = 0; c < C; ++c) {
            const float* xc = xn + (size_t)c * spatial;
            float32x4_t vsum = vdupq_n_f32(0);
            int s = 0;
            for (; s + 4 <= spatial; s += 4)
                vsum = vaddq_f32(vsum, vld1q_f32(xc + s));
            float sum = vaddvq_f32(vsum);
            for (; s < spatial; ++s)
                sum += xc[s];
            output[n * C + c] = sum * inv;
        }
    }
}

// NHWC GlobalAvgPool: vectorize across channels
// @nnr-meta isa=NEON dtype=fp32 layout=NHWC
inline void global_avgpool_nhwc_neon(const float* input, float* output,
    int N, int C, int spatial)
{
    float inv = 1.0f / (float)spatial;
    float32x4_t vinv = vdupq_n_f32(inv);
    for (int n = 0; n < N; ++n) {
        const float* xn = input + (size_t)n * C * spatial;
        float* yn = output + (size_t)n * C;
        int c = 0;
        for (; c + 4 <= C; c += 4) {
            float32x4_t acc = vdupq_n_f32(0);
            for (int s = 0; s < spatial; ++s)
                acc = vaddq_f32(acc, vld1q_f32(xn + (size_t)s * C + c));
            vst1q_f32(yn + c, vmulq_f32(acc, vinv));
        }
        for (; c < C; ++c) {
            float sum = 0;
            for (int s = 0; s < spatial; ++s)
                sum += xn[(size_t)s * C + c];
            yn[c] = sum * inv;
        }
    }
}

// NHWC MaxPool: vectorize across channels
// @nnr-meta isa=NEON dtype=fp32 layout=NHWC tiling=spatial
inline void maxpool_2d_nhwc_neon(const float* input, float* output,
    int N, int C, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW)
{
    nnr::for_static(0, N * oH, N * oH > 4, [&](int noh) {
        int n = noh / oH, oh = noh % oH;
        int ih0 = oh * sH - pH;
        int kh0 = std::max(0, -ih0), kh1 = std::min(kH, iH - ih0);
        const float* inp = input + (size_t)n * iH * iW * C;
        float* out = output + ((size_t)n * oH + oh) * oW * C;

        for (int ow = 0; ow < oW; ++ow) {
            int iw0 = ow * sW - pW;
            int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
            float* dst = out + ow * C;

            int c = 0;
            for (; c + 4 <= C; c += 4) {
                float32x4_t vmax = vdupq_n_f32(-FLT_MAX);
                for (int kh = kh0; kh < kh1; ++kh)
                    for (int kw = kw0; kw < kw1; ++kw) {
                        const float* src = inp + ((ih0 + kh) * iW + (iw0 + kw)) * C + c;
                        vmax = vmaxq_f32(vmax, vld1q_f32(src));
                    }
                vst1q_f32(dst + c, vmax);
            }
            for (; c < C; ++c) {
                float maxv = -FLT_MAX;
                for (int kh = kh0; kh < kh1; ++kh)
                    for (int kw = kw0; kw < kw1; ++kw)
                        maxv = std::max(maxv, inp[((ih0 + kh) * iW + (iw0 + kw)) * C + c]);
                dst[c] = maxv;
            }
        }
    });
}

// NHWC AvgPool: vectorize across channels
// @nnr-meta isa=NEON dtype=fp32 layout=NHWC tiling=spatial
inline void avgpool_2d_nhwc_neon(const float* input, float* output,
    int N, int C, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW,
    bool count_include_pad)
{
    nnr::for_static(0, N * oH, N * oH > 4, [&](int noh) {
        int n = noh / oH, oh = noh % oH;
        int ih0 = oh * sH - pH;
        int kh0 = std::max(0, -ih0), kh1 = std::min(kH, iH - ih0);
        const float* inp = input + (size_t)n * iH * iW * C;
        float* out = output + ((size_t)n * oH + oh) * oW * C;

        for (int ow = 0; ow < oW; ++ow) {
            int iw0 = ow * sW - pW;
            int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
            int valid = (kh1 - kh0) * (kw1 - kw0);
            float div = count_include_pad ? (float)(kH * kW) : (valid > 0 ? (float)valid : 1.0f);
            float inv = 1.0f / div;
            float32x4_t vinv = vdupq_n_f32(inv);
            float* dst = out + ow * C;

            int c = 0;
            for (; c + 4 <= C; c += 4) {
                float32x4_t vsum = vdupq_n_f32(0);
                for (int kh = kh0; kh < kh1; ++kh)
                    for (int kw = kw0; kw < kw1; ++kw) {
                        const float* src = inp + ((ih0 + kh) * iW + (iw0 + kw)) * C + c;
                        vsum = vaddq_f32(vsum, vld1q_f32(src));
                    }
                vst1q_f32(dst + c, vmulq_f32(vsum, vinv));
            }
            for (; c < C; ++c) {
                float sum = 0;
                for (int kh = kh0; kh < kh1; ++kh)
                    for (int kw = kw0; kw < kw1; ++kw)
                        sum += inp[((ih0 + kh) * iW + (iw0 + kw)) * C + c];
                dst[c] = sum * inv;
            }
        }
    });
}

} // namespace neon
} // namespace nnr
