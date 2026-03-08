#pragma once
#include <cstring>
#include "cpu_features.h"
#include "gemm.h"
#include "winograd_transforms.h"

#ifdef NNR_ARCH_X64
#include "backend/x64/winograd_x64.h"
#elifdef NNR_ARCH_ARM64
#include "backend/arm/winograd_neon.h"
#endif

namespace nnr {

// ============================================================================
// Scalar Winograd F(4x4, 3x3) convolution -- NCHW layout (fallback)
// ============================================================================

inline void winograd_conv2d_scalar(
    float* output, const float* input, const float* w_transformed,
    const float* bias, int N, int iC, int iH, int iW,
    int M, int oH, int oW, int pH, int pW, float* workspace)
{
    const int tH = (oH + 3) / 4, tW = (oW + 3) / 4;
    const int num_tiles = tH * tW;
    float* V = workspace;
    float* M_buf = V + (size_t)36 * iC * num_tiles;

    for (int n = 0; n < N; ++n) {
        const float* x_n = input + (size_t)n * iC * iH * iW;
        float* y_n = output + (size_t)n * M * oH * oW;

        for (int c = 0; c < iC; ++c) {
            const float* x_c = x_n + (size_t)c * iH * iW;
            for (int th = 0; th < tH; ++th) {
                for (int tw = 0; tw < tW; ++tw) {
                    int tile_idx = th * tW + tw;
                    float d[36];
                    int h_start = th * 4 - pH, w_start = tw * 4 - pW;
                    for (int i = 0; i < 6; ++i)
                        for (int j = 0; j < 6; ++j) {
                            int ih = h_start + i, iw = w_start + j;
                            d[i*6+j] = (ih >= 0 && ih < iH && iw >= 0 && iw < iW)
                                ? x_c[ih*iW + iw] : 0.0f;
                        }
                    float v[36];
                    winograd_input_transform(v, d);
                    for (int pos = 0; pos < 36; ++pos)
                        V[(size_t)pos*iC*num_tiles + c*num_tiles + tile_idx] = v[pos];
                }
            }
        }

        for (int m = 0; m < M; ++m) {
            for (int pos = 0; pos < 36; ++pos) {
                const float* W_pos_m = w_transformed + (size_t)pos*M*iC + m*iC;
                for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
                    float sum = 0.0f;
                    for (int c = 0; c < iC; ++c)
                        sum += W_pos_m[c] * V[(size_t)pos*iC*num_tiles + c*num_tiles + tile_idx];
                    M_buf[pos*num_tiles + tile_idx] = sum;
                }
            }

            float bv = bias ? bias[m] : 0.0f;
            float* y_m = y_n + (size_t)m * oH * oW;
            for (int th = 0; th < tH; ++th) {
                for (int tw = 0; tw < tW; ++tw) {
                    int tile_idx = th * tW + tw;
                    float m_tile[36];
                    for (int pos = 0; pos < 36; ++pos)
                        m_tile[pos] = M_buf[pos*num_tiles + tile_idx];
                    float y_tile[16];
                    winograd_output_transform(y_tile, m_tile);
                    for (int i = 0; i < 4; ++i) {
                        int oh = th * 4 + i;
                        if (oh >= oH) break;
                        for (int j = 0; j < 4; ++j) {
                            int ow = tw * 4 + j;
                            if (ow >= oW) break;
                            y_m[oh*oW + ow] = y_tile[i*4+j] + bv;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Cache-tiled scalar Winograd with pre-packed GEMM
// ============================================================================

constexpr int WINO_GROUP_DEFAULT = 64;

inline void winograd_conv2d_tiled_scalar(
    float* output, const float* input, const float* w_transformed,
    const float* w_packed_wino, const float* bias,
    int N, int iC, int iH, int iW, int M, int oH, int oW,
    int pH, int pW, float* workspace, int wino_group,
    operator_t::post_fn_t post_fn = nullptr, const operator_t* fused_op = nullptr,
    int strip_row_start = -1, int strip_row_end = -1,
    int iH_logical = 0, int oH_logical = 0)
{
    const int iH_log = iH_logical > 0 ? iH_logical : iH;
    const int oH_log = oH_logical > 0 ? oH_logical : oH;
    const bool strip_mode = (strip_row_start >= 0);

    const int tH = (oH_log + 3) / 4, tW = (oW + 3) / 4;
    const int num_tiles = tH * tW;
    const size_t pa_sz = w_packed_wino ? pack_a_size(M, iC) : 0;

    int range_tile0 = 0, range_tile1 = num_tiles;
    if (strip_mode) {
        int th_s = strip_row_start / 4;
        int th_e = std::min((strip_row_end + 3) / 4, tH);
        range_tile0 = th_s * tW;
        range_tile1 = th_e * tW;
    }
    const int num_groups = (range_tile1 - range_tile0 + wino_group - 1) / wino_group;

    const int oh_clip_lo = strip_mode ? strip_row_start : 0;
    const int oh_clip_hi = strip_mode ? std::min(strip_row_end, oH_log) : oH_log;

    const int th_lo = (pH + 3) / 4;
    const int th_hi = std::max(th_lo, (iH_log - 6 + pH) / 4 + 1);
    const int tw_lo = (pW + 3) / 4;
    const int tw_hi = std::max(tw_lo, (iW - 6 + pW) / 4 + 1);

    for (int n = 0; n < N; ++n) {
        const float* x_n = input + (size_t)n * iC * iH * iW;
        float* y_n = output + (size_t)n * M * oH * oW;

        for (int gid = 0; gid < num_groups; ++gid) {
            int tile_start = range_tile0 + gid * wino_group;
            int tile_end = std::min(tile_start + wino_group, range_tile1);
            int gs = tile_end - tile_start;

            float* V = workspace;                      // [36][iC][gs]
            float* M_buf = V + (size_t)36 * iC * gs;   // [36][M][gs]

            // Stage 1: Input transform
            for (int c = 0; c < iC; ++c) {
                const float* x_c = x_n + (size_t)c * iH * iW;
                for (int t = 0; t < gs; ++t) {
                    int tidx = tile_start + t;
                    int th = tidx / tW, tw = tidx % tW;
                    int h0 = th * 4 - pH, w0 = tw * 4 - pW;
                    float d[36];

                    if (th >= th_lo && th < th_hi && tw >= tw_lo && tw < tw_hi) {
                        const float* src = x_c + h0 * iW + w0;
                        for (int i = 0; i < 6; ++i)
                            std::memcpy(d + i * 6, src + i * iW, 6 * sizeof(float));
                    } else {
                        for (int i = 0; i < 6; ++i)
                            for (int j = 0; j < 6; ++j) {
                                int ih = h0 + i, iw = w0 + j;
                                d[i*6+j] = (ih >= 0 && ih < iH_log && iw >= 0 && iw < iW)
                                    ? x_c[ih*iW + iw] : 0.0f;
                            }
                    }

                    float v[36];
                    winograd_input_transform(v, d);
                    for (int pos = 0; pos < 36; ++pos)
                        V[(size_t)pos*iC*gs + c*gs + t] = v[pos];
                }
            }

            // Stage 2: 36 GEMMs
            for (int pos = 0; pos < 36; ++pos) {
                const float* V_pos = V + (size_t)pos * iC * gs;
                float* M_pos = M_buf + (size_t)pos * M * gs;
                if (w_packed_wino)
                    dgemm_packed_a(M, gs, iC, w_packed_wino + pos * pa_sz, V_pos, M_pos);
                else
                    dgemm_generic(M, gs, iC, w_transformed + (size_t)pos * M * iC, V_pos, M_pos);
            }

            // Stage 3: Output transform
            for (int m = 0; m < M; ++m) {
                float bv = (!post_fn && bias) ? bias[m] : 0.0f;
                float* y_m = y_n + (size_t)m * oH * oW;
                for (int t = 0; t < gs; ++t) {
                    int tidx = tile_start + t;
                    int th = tidx / tW, tw = tidx % tW;
                    float m_tile[36];
                    for (int pos = 0; pos < 36; ++pos)
                        m_tile[pos] = M_buf[(size_t)pos*M*gs + m*gs + t];
                    float y_tile[16];
                    winograd_output_transform(y_tile, m_tile);

                    if (bv != 0.0f)
                        for (int p = 0; p < 16; ++p) y_tile[p] += bv;

                    int oh0 = th * 4, ow0 = tw * 4;
                    if (oh0 >= oh_clip_lo && oh0 + 4 <= oh_clip_hi && ow0 + 4 <= oW) {
                        for (int i = 0; i < 4; ++i)
                            std::memcpy(y_m + (oh0 + i) * oW + ow0, y_tile + i * 4, 4 * sizeof(float));
                    } else {
                        for (int i = 0; i < 4; ++i) {
                            int oh = oh0 + i;
                            if (oh < oh_clip_lo) continue;
                            if (oh >= oh_clip_hi) break;
                            for (int j = 0; j < 4; ++j) {
                                int ow = ow0 + j;
                                if (ow >= oW) break;
                                y_m[oh*oW + ow] = y_tile[i*4+j];
                            }
                        }
                    }
                }
            }
        }

        // Fused post-op: bias + activation applied to strip (or full output)
        if (post_fn) {
            int sr = strip_mode ? std::max(strip_row_start, 0) : 0;
            int er = strip_mode ? std::min(strip_row_end, oH_log) : oH_log;
            int strip_spatial = (er - sr) * oW;
            float* y_strip = y_n + (size_t)sr * oW;
            int off = (int)((size_t)n * M * oH * oW + (size_t)sr * oW);
            post_fn(y_strip, M, strip_spatial, oH * oW, fused_op, bias, off);
        }
    }
}

// ============================================================================
// Dispatch wrappers
// ============================================================================

// NCHW dispatch: AVX-512 -> AVX2 -> NEON -> scalar
inline void winograd_conv2d(
    float* output, const float* input, const float* w_transformed,
    const float* w_packed_wino, const float* bias,
    int N, int iC, int iH, int iW,
    int M, int oH, int oW, int pH, int pW, float* workspace, int wino_group,
    operator_t::post_fn_t post_fn = nullptr, const operator_t* fused_op = nullptr,
    int strip_row_start = -1, int strip_row_end = -1,
    int iH_logical = 0, int oH_logical = 0)
{
#ifdef NNR_ARCH_X64
    if (has_avx512())
        winograd_conv2d_tiled_simd<__m512>(output, input, w_transformed,
            w_packed_wino, bias, N, iC, iH, iW, M, oH, oW, pH, pW, workspace, wino_group,
            post_fn, fused_op, strip_row_start, strip_row_end, iH_logical, oH_logical);
    else if (detect_isa() == isa_t::avx2)
        winograd_conv2d_tiled_simd<__m256>(output, input, w_transformed,
            w_packed_wino, bias, N, iC, iH, iW, M, oH, oW, pH, pW, workspace, wino_group,
            post_fn, fused_op, strip_row_start, strip_row_end, iH_logical, oH_logical);
    else
        winograd_conv2d_tiled_scalar(output, input, w_transformed,
            w_packed_wino, bias, N, iC, iH, iW, M, oH, oW, pH, pW, workspace, wino_group,
            post_fn, fused_op, strip_row_start, strip_row_end, iH_logical, oH_logical);
#elifdef NNR_ARCH_ARM64
    winograd_conv2d_tiled_neon(output, input, w_transformed,
        w_packed_wino, bias, N, iC, iH, iW, M, oH, oW, pH, pW, workspace, wino_group,
        post_fn, fused_op, strip_row_start, strip_row_end, iH_logical, oH_logical);
#else
    winograd_conv2d_tiled_scalar(output, input, w_transformed,
        w_packed_wino, bias, N, iC, iH, iW, M, oH, oW, pH, pW, workspace, wino_group,
        post_fn, fused_op, strip_row_start, strip_row_end, iH_logical, oH_logical);
#endif
}

// NHWC dispatch: AVX-512 -> AVX2 -> (no scalar fallback)
inline void winograd_conv2d_nhwc(
    float* output, const float* input,
    const float* w_nhwc_transformed, const float* w_packed_b_wino,
    const float* bias,
    int N, int iC, int iH, int iW, int M, int oH, int oW,
    int pH, int pW, float* workspace, bool input_nhwc, int wino_group,
    operator_t::post_fn_t post_fn = nullptr, const operator_t* fused_op = nullptr)
{
#ifdef NNR_ARCH_X64
    if (has_avx512())
        winograd_conv2d_nhwc_simd<__m512>(output, input, w_nhwc_transformed,
            w_packed_b_wino, bias, N, iC, iH, iW, M, oH, oW, pH, pW,
            workspace, input_nhwc, wino_group, post_fn, fused_op);
    else
        winograd_conv2d_nhwc_simd<__m256>(output, input, w_nhwc_transformed,
            w_packed_b_wino, bias, N, iC, iH, iW, M, oH, oW, pH, pW,
            workspace, input_nhwc, wino_group, post_fn, fused_op);
#else
    (void)output; (void)input; (void)w_nhwc_transformed;
    (void)w_packed_b_wino; (void)bias; (void)N; (void)iC; (void)iH; (void)iW;
    (void)M; (void)oH; (void)oW; (void)pH; (void)pW; (void)workspace;
    (void)input_nhwc; (void)wino_group; (void)post_fn; (void)fused_op;
#endif
}

} // namespace nnr
