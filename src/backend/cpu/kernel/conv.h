#pragma once
// Convolution kernels: im2col + GEMM, 1×1 shortcuts, depthwise, NHWC.
//
// Naming convention (used throughout this file and Conv.cpp):
//   k=kernel, i=input, o=output, s=stride, p=pad, d=dilation
//   C=channels, H=height, W=width, N=batch, M=output channels
//   e.g., kH=kernel height, iW=input width, oC=output channels
//   MM=M/groups (output channels per group), CC=iC/groups (input channels per group)
//   CHW=kC*kH*kW (im2col column height), spatial=oH*oW (output spatial size)
#include "gemm.h"
#include <algorithm>
#include <cstring>
#ifdef NNR_ARCH_X64
#include "backend/x64/conv_x64.h"
#endif

namespace nnr {

// Strided im2col row: copy input at stride sW to output, with boundary zeroing.
// Eliminates per-pixel branch checking (branch-free interior).
// AVX-512 fast path for stride-2: permutex2var extracts even elements.
template <typename T>
inline void im2col_stride_row(T* __restrict drow, const T* __restrict srow,
    int oW, int iW, int sW, int iw_base)
{
    int w0 = (iw_base >= 0) ? 0 : (-iw_base + sW - 1) / sW;
    int w1 = std::min(oW, (iW - iw_base + sW - 1) / sW);
    if (w0 > 0) memset(drow, 0, w0 * sizeof(T));
    if (w1 < oW) memset(drow + w1, 0, (oW - w1) * sizeof(T));
    const T* src = srow + iw_base;
    int ow = w0;
#ifdef NNR_ARCH_X64
    if constexpr (std::is_same_v<T, float>) {
        if (sW == 2)
            ow = im2col_stride2_avx512(drow, src, ow, w0, w1, iW, iw_base);
    }
#endif
    for (; ow < w1; ow++)
        drow[ow] = src[ow * sW];
}

template <typename T>
inline void im2col(T* col, const T* input,
    int kC, int iH, int iW, int kH, int kW, int oH, int oW,
    int sH, int sW, int pH, int pW, int dH, int dW)
{
    // im2col: rearrange input patches into columns for GEMM.
    // Produces a [kC*kH*kW × oH*oW] matrix where each column is a flattened
    // receptive field. Stride-1 case uses memcpy for contiguous input rows.
    const int spatial = oH * oW;

    // Fast path for 1×K or K×1 horizontal convs: pad each input row once,
    // then offset-memcpy for each kw position. Avoids per-row boundary checks.
    if (kH == 1 && sW == 1 && dW == 1 && pW > 0) {
        const int padded_w = iW + 2 * pW;
        // Stack buffer for one padded row (small: typically 17 + 6 = 23 floats = 92 bytes)
        T* padrow = (T*)alloca(padded_w * sizeof(T));
        memset(padrow, 0, pW * sizeof(T));
        memset(padrow + pW + iW, 0, pW * sizeof(T));
        for (int c = 0; c < kC; ++c) {
            const T* xc = input + (size_t)c * iH * iW;
            for (int oh = 0; oh < oH; ++oh) {
                int ih = oh * sH - pH;
                if (ih >= 0 && ih < iH) {
                    memcpy(padrow + pW, xc + ih * iW, iW * sizeof(T));
                } else {
                    memset(padrow + pW, 0, iW * sizeof(T));
                }
                for (int kw = 0; kw < kW; ++kw) {
                    int k = c * kW + kw;
                    T* dst = col + (size_t)k * spatial + oh * oW;
                    memcpy(dst, padrow + kw, oW * sizeof(T));
                }
            }
        }
        return;
    }

    for (int c = 0; c < kC; ++c) {
        const T* xc = input + (size_t)c * iH * iW;
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                int k = (c * kH + kh) * kW + kw;
                T* dst = col + (size_t)k * spatial;
                // Horizontal bounds don't depend on oh — hoist outside the row loop.
                int iw_base = kw * dW - pW;
                int w0 = (sW == 1) ? std::max(0, -iw_base) : 0;
                int w1 = (sW == 1) ? std::min(oW, iW - iw_base) : oW;
                bool full_row = (sW == 1 && w0 == 0 && w1 == oW);
                // Vertical valid range: oh values where ih is in [0, iH).
                // ih = oh * sH - pH + kh * dH.
                int oh_first = 0, oh_last = oH;
                for (; oh_first < oH; ++oh_first) {
                    int ih = oh_first * sH - pH + kh * dH;
                    if (ih >= 0 && ih < iH) break;
                }
                for (oh_last = oH; oh_last > oh_first; --oh_last) {
                    int ih = (oh_last - 1) * sH - pH + kh * dH;
                    if (ih >= 0 && ih < iH) break;
                }
                // Zero top padding rows
                if (oh_first > 0)
                    memset(dst, 0, (size_t)oh_first * oW * sizeof(T));
                // Zero bottom padding rows
                if (oh_last < oH)
                    memset(dst + (size_t)oh_last * oW, 0, (size_t)(oH - oh_last) * oW * sizeof(T));
                // Interior rows: ih is valid for oh in [oh_first, oh_last).
                int n_rows = oh_last - oh_first;
                if (n_rows <= 0) continue;
                int ih0 = oh_first * sH - pH + kh * dH;
                if (full_row && sH == 1 && dH == 1 && iW == oW) {
                    // Bulk path: src and dst are both contiguous blocks.
                    // src = xc + ih0 * iW + iw_base, dst = dst + oh_first * oW
                    memcpy(dst + (size_t)oh_first * oW,
                           xc + (size_t)ih0 * iW + iw_base,
                           (size_t)n_rows * oW * sizeof(T));
                } else if (full_row && sH == 1 && dH == 1) {
                    // Full-width rows but iW != oW: row-by-row, no boundary checks.
                    const T* sp = xc + (size_t)ih0 * iW + iw_base;
                    T* dp = dst + (size_t)oh_first * oW;
                    for (int r = 0; r < n_rows; ++r, sp += iW, dp += oW)
                        memcpy(dp, sp, oW * sizeof(T));
                } else {
                    // General row-by-row with horizontal boundary handling.
                    for (int oh = oh_first; oh < oh_last; ++oh) {
                        int ih = oh * sH - pH + kh * dH;
                        T* drow = dst + oh * oW;
                        const T* srow = xc + ih * iW;
                        if (sW == 1) {
                            if (w0 > 0) memset(drow, 0, w0 * sizeof(T));
                            if (w1 > w0) memcpy(drow + w0, srow + iw_base + w0, (w1 - w0) * sizeof(T));
                            if (w1 < oW) memset(drow + w1, 0, (oW - w1) * sizeof(T));
                        } else {
                            im2col_stride_row(drow, srow, oW, iW, sW, iw_base);
                        }
                    }
                }
            }
        }
    }
}

// Tiled im2col: produce columns for output rows [oh0, oh0+oh_count) only.
// Output: [kC*kH*kW × oh_count*oW] matrix — same layout as im2col but for a spatial strip.
template <typename T>
inline void im2col_tiled(T* col, const T* input,
    int kC, int iH, int iW, int kH, int kW, int oW,
    int sH, int sW, int pH, int pW, int dH, int dW,
    int oh0, int oh_count)
{
    const int strip_W = oh_count * oW;
    for (int c = 0; c < kC; ++c) {
        const T* xc = input + (size_t)c * iH * iW;
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                int k = (c * kH + kh) * kW + kw;
                T* dst = col + (size_t)k * strip_W;
                int iw_base = kw * dW - pW;
                int w0 = (sW == 1) ? std::max(0, -iw_base) : 0;
                int w1 = (sW == 1) ? std::min(oW, iW - iw_base) : oW;
                bool full_row = (sW == 1 && w0 == 0 && w1 == oW);
                // Vertical valid range within the tile
                int t_first = 0, t_last = oh_count;
                for (; t_first < oh_count; ++t_first) {
                    int ih = (oh0 + t_first) * sH - pH + kh * dH;
                    if (ih >= 0 && ih < iH) break;
                }
                for (t_last = oh_count; t_last > t_first; --t_last) {
                    int ih = (oh0 + t_last - 1) * sH - pH + kh * dH;
                    if (ih >= 0 && ih < iH) break;
                }
                if (t_first > 0)
                    memset(dst, 0, (size_t)t_first * oW * sizeof(T));
                if (t_last < oh_count)
                    memset(dst + (size_t)t_last * oW, 0, (size_t)(oh_count - t_last) * oW * sizeof(T));
                int n_rows = t_last - t_first;
                if (n_rows <= 0) continue;
                int ih0_ = (oh0 + t_first) * sH - pH + kh * dH;
                if (full_row && sH == 1 && dH == 1 && iW == oW) {
                    memcpy(dst + (size_t)t_first * oW,
                           xc + (size_t)ih0_ * iW + iw_base,
                           (size_t)n_rows * oW * sizeof(T));
                } else if (full_row && sH == 1 && dH == 1) {
                    const T* sp = xc + (size_t)ih0_ * iW + iw_base;
                    T* dp = dst + (size_t)t_first * oW;
                    for (int r = 0; r < n_rows; ++r, sp += iW, dp += oW)
                        memcpy(dp, sp, oW * sizeof(T));
                } else {
                    for (int t = t_first; t < t_last; ++t) {
                        int ih = (oh0 + t) * sH - pH + kh * dH;
                        T* drow = dst + t * oW;
                        const T* srow = xc + ih * iW;
                        if (sW == 1) {
                            if (w0 > 0) memset(drow, 0, w0 * sizeof(T));
                            if (w1 > w0) memcpy(drow + w0, srow + iw_base + w0, (w1 - w0) * sizeof(T));
                            if (w1 < oW) memset(drow + w1, 0, (oW - w1) * sizeof(T));
                        } else {
                            im2col_stride_row(drow, srow, oW, iW, sW, iw_base);
                        }
                    }
                }
            }
        }
    }
}

// Tiled NHWC im2col: produce rows for output rows [oh0, oh0+oh_count) only.
// Output: [oh_count*oW × kH*kW*kC] matrix — same layout as im2col_nhwc but for a spatial strip.
template <typename T>
inline void im2col_nhwc_tiled(T* col, const T* input,
    int C, int kC, int iH, int iW, int kH, int kW, int oW,
    int sH, int sW, int pH, int pW, int dH, int dW, int c_off,
    int oh0, int oh_count)
{
    const int KHW_C = kH * kW * kC;
    for (int t = 0; t < oh_count; ++t) {
        int oh = oh0 + t;
        for (int ow = 0; ow < oW; ++ow) {
            T* dst = col + (size_t)(t * oW + ow) * KHW_C;
            for (int kh = 0; kh < kH; ++kh) {
                int ih = oh * sH - pH + kh * dH;
                for (int kw = 0; kw < kW; ++kw) {
                    int iw = ow * sW - pW + kw * dW;
                    if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
                        memcpy(dst, input + (ih * iW + iw) * C + c_off, kC * sizeof(T));
                    } else {
                        memset(dst, 0, kC * sizeof(T));
                    }
                    dst += kC;
                }
            }
        }
    }
}

// Conv2D via im2col + GEMM.  Handles 1x1/s1 shortcut and general case.
// No depthwise — that's handled separately by the Conv operator.
inline void conv2d_gemm(float* output, const float* input, const float* weight,
    const float* bias, float* workspace,
    int N, int iC, int iH, int iW,
    int M, int kC, int kH, int kW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    gemm_post_t post = {})
{
    // Output spatial dims: oH = (iH + 2*pad - effective_kernel) / stride + 1
    // where effective_kernel = (kH-1)*dH + 1 accounts for dilation.
    const int oH = (iH + 2 * pH - ((kH - 1) * dH + 1)) / sH + 1;
    const int oW = (iW + 2 * pW - ((kW - 1) * dW + 1)) / sW + 1;
    const int MM = M / groups;
    const int CC = iC / groups;
    const int kHW = kH * kW;
    const int CHW = kC * kHW;
    const int spatial = oH * oW;

    // 1x1 shortcut: direct GEMM, no im2col
    if (kH == 1 && kW == 1 && sH == 1 && sW == 1
        && dH == 1 && dW == 1 && pH == 0 && pW == 0) {
        for (int ng = 0; ng < N * groups; ++ng) {
            int n = ng / groups, g = ng % groups;
            float* yn = output + ((size_t)n * M + g * MM) * spatial;
            gemm_post_t p = post;
            if (p.bias) p.bias_off = g * MM;
            p.c_base = yn;
            p.c_base_offset = (int)((n * M + g * MM) * spatial);
            dgemm_generic(MM, spatial, CC,
                weight + (size_t)g * MM * CC,
                input + ((size_t)n * iC + g * CC) * iH * iW,
                yn, p);
        }
        return;
    }

    // General: im2col + GEMM
    float* col = workspace;
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups; ++g) {
            const float* xn = input + ((size_t)n * iC + g * CC) * iH * iW;
            im2col(col, xn, kC, iH, iW, kH, kW, oH, oW, sH, sW, pH, pW, dH, dW);
            float* yn = output + ((size_t)n * M + g * MM) * spatial;
            gemm_post_t p = post;
            if (p.bias) p.bias_off = g * MM;
            p.c_base = yn;
            p.c_base_offset = (int)((n * M + g * MM) * spatial);
            dgemm_generic(MM, spatial, CHW,
                weight + (size_t)g * MM * CHW, col, yn, p);
        }
    }
}

// Conv2D with asymmetric padding. oH, oW are passed explicitly.
// pH, pW are top-left pads used by im2col for boundary checks.
inline void conv2d_gemm_asym(float* output, const float* input, const float* weight,
    const float* bias, float* workspace,
    int N, int iC, int iH, int iW,
    int M, int kC, int kH, int kW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int oH, int oW, gemm_post_t post = {})
{
    const int MM = M / groups;
    const int CC = iC / groups;
    const int kHW = kH * kW;
    const int CHW = kC * kHW;
    const int spatial = oH * oW;

    if (kH == 1 && kW == 1 && sH == 1 && sW == 1
        && dH == 1 && dW == 1 && pH == 0 && pW == 0) {
        for (int ng = 0; ng < N * groups; ++ng) {
            int n = ng / groups, g = ng % groups;
            float* yn = output + ((size_t)n * M + g * MM) * spatial;
            gemm_post_t p = post;
            if (p.bias) p.bias_off = g * MM;
            p.c_base = yn;
            p.c_base_offset = (int)((n * M + g * MM) * spatial);
            dgemm_generic(MM, spatial, CC,
                weight + (size_t)g * MM * CC,
                input + ((size_t)n * iC + g * CC) * iH * iW,
                yn, p);
        }
        return;
    }

    float* col = workspace;
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups; ++g) {
            const float* xn = input + ((size_t)n * iC + g * CC) * iH * iW;
            im2col(col, xn, kC, iH, iW, kH, kW, oH, oW, sH, sW, pH, pW, dH, dW);
            float* yn = output + ((size_t)n * M + g * MM) * spatial;
            gemm_post_t p = post;
            if (p.bias) p.bias_off = g * MM;
            p.c_base = yn;
            p.c_base_offset = (int)((n * M + g * MM) * spatial);
            dgemm_generic(MM, spatial, CHW,
                weight + (size_t)g * MM * CHW, col, yn, p);
        }
    }
}

// 1x1 Conv strip: compute output rows [oh_start, oh_start + oh_rows) only.
// Uses gather/GEMM/scatter to avoid needing GEMM ldb/ldc support.
// workspace must hold (CC_g + MM_g) * oh_rows * oW floats.
inline void conv1x1_strip(float* output, const float* input, const float* weight,
    const float* bias, float* workspace,
    int N, int iC, int MM, int H, int W, int groups,
    int oh_start, int oh_rows,
    gemm_post_t post = {})
{
    int HW = H * W;
    int strip_W = oh_rows * W;
    int off = oh_start * W;
    int CC_g = iC / groups;
    int MM_g = MM / groups;

    float* col = workspace;
    float* tmp = col + CC_g * strip_W;

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups; ++g) {
            // Gather: copy strip rows from each input channel into contiguous buffer
            for (int c = 0; c < CC_g; ++c)
                memcpy(col + (size_t)c * strip_W,
                       input + ((size_t)n * iC + g * CC_g + c) * HW + off,
                       strip_W * sizeof(float));

            // GEMM: [MM_g × CC_g] × [CC_g × strip_W] → tmp[MM_g × strip_W]
            gemm_post_t p = post;
            if (p.bias) p.bias_off = g * MM_g;
            dgemm_generic(MM_g, strip_W, CC_g,
                weight + (size_t)g * MM_g * CC_g, col, tmp, p);

            // Scatter: copy each output channel's strip to correct tensor position
            for (int m = 0; m < MM_g; ++m)
                memcpy(output + ((size_t)n * MM + g * MM_g + m) * HW + off,
                       tmp + (size_t)m * strip_W,
                       strip_W * sizeof(float));
        }
    }
}

// NHWC im2col: rearrange NHWC input patches into rows for GEMM.
// Input: [H, W, C] (one batch), output: [spatial × (kH*kW*kC)].
// Kernel index order: (kh, kw, c) — groups of kC contiguous channels,
// enabling memcpy from NHWC input where channels are innermost.
// c_off: starting channel for this group (0 for group=1).
template <typename T>
inline void im2col_nhwc(T* col, const T* input,
    int C, int kC, int iH, int iW, int kH, int kW, int oH, int oW,
    int sH, int sW, int pH, int pW, int dH, int dW, int c_off)
{
    const int KHW_C = kH * kW * kC;
    for (int oh = 0; oh < oH; ++oh) {
        for (int ow = 0; ow < oW; ++ow) {
            T* dst = col + (size_t)(oh * oW + ow) * KHW_C;
            for (int kh = 0; kh < kH; ++kh) {
                int ih = oh * sH - pH + kh * dH;
                for (int kw = 0; kw < kW; ++kw) {
                    int iw = ow * sW - pW + kw * dW;
                    if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
                        memcpy(dst, input + (ih * iW + iw) * C + c_off, kC * sizeof(T));
                    } else {
                        memset(dst, 0, kC * sizeof(T));
                    }
                    dst += kC;
                }
            }
        }
    }
}

// im2col from NCHW input producing NHWC-ordered output [spatial × (kH*kW*kC)].
// Used at NHWC chain boundaries when the input is still in NCHW format.
// Avoids a separate full-tensor NCHW→NHWC reorder before im2col.
template <typename T>
inline void im2col_nhwc_from_nchw(T* col, const T* input,
    int kC, int iH, int iW, int kH, int kW, int oH, int oW,
    int sH, int sW, int pH, int pW, int dH, int dW, int c_off)
{
    const int KHW_C = kH * kW * kC;
    for (int oh = 0; oh < oH; ++oh) {
        for (int ow = 0; ow < oW; ++ow) {
            T* dst = col + (size_t)(oh * oW + ow) * KHW_C;
            for (int kh = 0; kh < kH; ++kh) {
                int ih = oh * sH - pH + kh * dH;
                for (int kw = 0; kw < kW; ++kw) {
                    int iw = ow * sW - pW + kw * dW;
                    if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
                        // NCHW: channel c is at input[(c_off+c)*iH*iW + ih*iW + iw]
                        for (int c = 0; c < kC; ++c)
                            dst[c] = input[(size_t)(c_off + c) * iH * iW + ih * iW + iw];
                    } else {
                        memset(dst, 0, kC * sizeof(T));
                    }
                    dst += kC;
                }
            }
        }
    }
}

// Vectorized column-wise bias for NHWC: Y[spatial × M] += bias[M]
inline void nhwc_bias_add(float* Y, const float* bias, int spatial, int M) {
#ifdef NNR_ARCH_X64
    nhwc_bias_add_x64(Y, bias, spatial, M);
#else
    for (int s = 0; s < spatial; s++)
        for (int m = 0; m < M; m++)
            Y[s * M + m] += bias[m];
#endif
}

// NHWC 1×1 Conv: Y[spatial × M] = X[spatial × C] × W^T[C × M]
// Then applies bias (per-channel) and fused post-op.
// X is [spatial × C] (NHWC), W_T is [C × M] (pre-transposed), Y is [spatial × M] (NHWC).
inline void conv1x1_nhwc(float* __restrict Y, const float* __restrict X,
    const float* __restrict W_T, const float* bias,
    int spatial, int M, int C,
    operator_t::post_fn_t post_fn, const operator_t* fused_op, int y_offset)
{
    dgemm_generic(spatial, M, C, X, W_T, Y);

    // Bias: add bias[m] to every spatial position
    if (bias) {
        for (int s = 0; s < spatial; s++) {
            float* row = Y + s * M;
            for (int m = 0; m < M; m++)
                row[m] += bias[m];
        }
    }

    // Fused post-op (elementwise, layout-agnostic)
    if (post_fn) {
        post_fn(Y, 1, spatial * M, spatial * M, fused_op, nullptr, y_offset);
    }
}

} // namespace nnr
