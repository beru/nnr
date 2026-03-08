#pragma once
// x64 SIMD (AVX2/AVX-512) implementations for Winograd F(4x4, 3x3) convolution.
// Templated on Vec type (__m256 for AVX2, __m512 for AVX-512).

#include <immintrin.h>
#include <cstring>
#include "cpu_features.h"
#include "thread_pool.h"
#include "backend/cpu/kernel/winograd_transforms.h"
#include "backend/cpu/kernel/gemm.h"

namespace nnr {

// ============================================================================
// SIMD traits for type-generic batch transforms
// ============================================================================

template <typename Vec> struct simd_f32;

template <> struct simd_f32<__m256> {
    static constexpr int W = 8;
    static __m256 load(const float* p) { return _mm256_loadu_ps(p); }
    static void store(float* p, __m256 v) { _mm256_storeu_ps(p, v); }
    static __m256 add(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
    static __m256 sub(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
    static __m256 set1(float v) { return _mm256_set1_ps(v); }
};

template <> struct simd_f32<__m512> {
    static constexpr int W = 16;
    // @nnr-meta isa=[AVX2,AVX512] dtype=fp32 special=Winograd
    static __m512 load(const float* p) { return _mm512_loadu_ps(p); }
    // @nnr-meta isa=[AVX2,AVX512] dtype=fp32 special=Winograd
    static void store(float* p, __m512 v) { _mm512_storeu_ps(p, v); }
    // @nnr-meta isa=[AVX2,AVX512] dtype=fp32 special=Winograd
    static __m512 add(__m512 a, __m512 b) { return _mm512_add_ps(a, b); }
    // @nnr-meta isa=[AVX2,AVX512] dtype=fp32 special=Winograd
    static __m512 sub(__m512 a, __m512 b) { return _mm512_sub_ps(a, b); }
    // @nnr-meta isa=[AVX2,AVX512] dtype=fp32 special=Winograd
    static __m512 set1(float v) { return _mm512_set1_ps(v); }
};

// SIMD row copy: avoids CRT memcpy + vzeroupper penalty inside SIMD-hot code.
// Copies exactly W floats (one full vector width).
template <typename Vec>
// @nnr-meta isa=[AVX2,AVX512] dtype=fp32 special=Winograd
inline void copy_w(float* __restrict dst, const float* __restrict src) {
    simd_f32<Vec>::store(dst, simd_f32<Vec>::load(src));
}

// ============================================================================
// SIMD batch transforms -- process W tiles in parallel (8 for AVX2, 16 for AVX-512)
// Data layout: [positions][W] (position-major, W values per position)
// ============================================================================

// BT * d * B for W tiles in parallel.
// d[36][W], v[36][W] -- position-major.
// Uses only adds/subs (integer multipliers -> repeated addition).
template <typename Vec>
// @nnr-meta isa=[AVX2,AVX512] dtype=fp32 layout=NCHW special=Winograd
inline void winograd_input_transform_simd(float* __restrict v, const float* __restrict d) {
    using S = simd_f32<Vec>;
    constexpr int W = S::W;

    // Load 6x6 tile data
    Vec dd[6][6];
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            dd[i][j] = S::load(d + (i*6+j)*W);

    // Column transform: tmp = BT * dd
    Vec tmp[6][6];
    for (int j = 0; j < 6; j++) {
        Vec r0 = dd[0][j], r1 = dd[1][j], r2 = dd[2][j];
        Vec r3 = dd[3][j], r4 = dd[4][j], r5 = dd[5][j];

        // t0 = 4*r0 - 5*r2 + r4
        Vec r0x2 = S::add(r0, r0), r0x4 = S::add(r0x2, r0x2);
        Vec r2x2 = S::add(r2, r2), r2x4 = S::add(r2x2, r2x2);
        Vec r2x5 = S::add(r2x4, r2);
        tmp[0][j] = S::add(S::sub(r0x4, r2x5), r4);

        // Common subexpressions
        Vec p12 = S::add(r1, r2), m12 = S::sub(r1, r2);
        Vec p34 = S::add(r3, r4), m34 = S::sub(r3, r4);
        Vec p12x2 = S::add(p12, p12), p12x4 = S::add(p12x2, p12x2);
        Vec m12x2 = S::add(m12, m12), m12x4 = S::add(m12x2, m12x2);

        // t1 = -4*(d1+d2) + (d3+d4)
        tmp[1][j] = S::sub(p34, p12x4);
        // t2 = 4*(d1-d2) - (d3-d4)
        tmp[2][j] = S::sub(m12x4, m34);

        // t3 = (d4-d2) + 2*(d3-d1), t4 = (d4-d2) - 2*(d3-d1)
        Vec s42 = S::sub(r4, r2), s31 = S::sub(r3, r1);
        Vec s31x2 = S::add(s31, s31);
        tmp[3][j] = S::add(s42, s31x2);
        tmp[4][j] = S::sub(s42, s31x2);
        // t5 = 4*d1 - 5*d3 + d5
        Vec r1x2 = S::add(r1, r1), r1x4 = S::add(r1x2, r1x2);
        Vec r3x2 = S::add(r3, r3), r3x4 = S::add(r3x2, r3x2);
        Vec r3x5 = S::add(r3x4, r3);
        tmp[5][j] = S::add(S::sub(r1x4, r3x5), r5);
    }

    // Row transform: v = tmp * B (same formula, applied to rows)
    for (int i = 0; i < 6; i++) {
        Vec r0 = tmp[i][0], r1 = tmp[i][1], r2 = tmp[i][2];
        Vec r3 = tmp[i][3], r4 = tmp[i][4], r5 = tmp[i][5];

        Vec r0x2 = S::add(r0, r0), r0x4 = S::add(r0x2, r0x2);
        Vec r2x2 = S::add(r2, r2), r2x4 = S::add(r2x2, r2x2);
        Vec r2x5 = S::add(r2x4, r2);
        S::store(v + (i*6+0)*W, S::add(S::sub(r0x4, r2x5), r4));

        Vec p12 = S::add(r1, r2), m12 = S::sub(r1, r2);
        Vec p34 = S::add(r3, r4), m34 = S::sub(r3, r4);
        Vec p12x2 = S::add(p12, p12), p12x4 = S::add(p12x2, p12x2);
        Vec m12x2 = S::add(m12, m12), m12x4 = S::add(m12x2, m12x2);

        S::store(v + (i*6+1)*W, S::sub(p34, p12x4));
        S::store(v + (i*6+2)*W, S::sub(m12x4, m34));

        Vec s42 = S::sub(r4, r2), s31 = S::sub(r3, r1);
        Vec s31x2 = S::add(s31, s31);
        S::store(v + (i*6+3)*W, S::add(s42, s31x2));
        S::store(v + (i*6+4)*W, S::sub(s42, s31x2));
        // v5 = 4*r1 - 5*r3 + r5
        Vec r1x2b = S::add(r1, r1), r1x4b = S::add(r1x2b, r1x2b);
        Vec r3x2b = S::add(r3, r3), r3x4b = S::add(r3x2b, r3x2b);
        Vec r3x5b = S::add(r3x4b, r3);
        S::store(v + (i*6+5)*W, S::add(S::sub(r1x4b, r3x5b), r5));
    }
}

// AT * M * A for W tiles in parallel, with fused bias addition.
// m[36][W] input, y[16][W] output -- position-major.
template <typename Vec>
// @nnr-meta isa=[AVX2,AVX512] dtype=fp32 layout=NCHW special=Winograd fusion=post_op
inline void winograd_output_transform_simd(float* __restrict y, const float* __restrict m, float bias_val) {
    using S = simd_f32<Vec>;
    constexpr int W = S::W;
    Vec vbias = S::set1(bias_val);

    // Load 6x6 products
    Vec mm[6][6];
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            mm[i][j] = S::load(m + (i*6+j)*W);

    // Column transform: AT * mm
    Vec tmp[4][6];
    for (int j = 0; j < 6; j++) {
        Vec p12 = S::add(mm[1][j], mm[2][j]);
        Vec m12 = S::sub(mm[1][j], mm[2][j]);
        Vec p34 = S::add(mm[3][j], mm[4][j]);
        Vec m34 = S::sub(mm[3][j], mm[4][j]);
        Vec m34x2 = S::add(m34, m34);
        Vec p34x2 = S::add(p34, p34), p34x4 = S::add(p34x2, p34x2);
        Vec m34x4 = S::add(m34x2, m34x2), m34x8 = S::add(m34x4, m34x4);

        tmp[0][j] = S::add(S::add(mm[0][j], p12), p34);
        tmp[1][j] = S::add(m12, m34x2);
        tmp[2][j] = S::add(p12, p34x4);
        tmp[3][j] = S::add(S::add(m12, m34x8), mm[5][j]);
    }

    // Row transform with bias fusion: y = tmp * A + bias
    for (int i = 0; i < 4; i++) {
        Vec p12 = S::add(tmp[i][1], tmp[i][2]);
        Vec m12 = S::sub(tmp[i][1], tmp[i][2]);
        Vec p34 = S::add(tmp[i][3], tmp[i][4]);
        Vec m34 = S::sub(tmp[i][3], tmp[i][4]);
        Vec m34x2 = S::add(m34, m34);
        Vec p34x2 = S::add(p34, p34), p34x4 = S::add(p34x2, p34x2);
        Vec m34x4 = S::add(m34x2, m34x2), m34x8 = S::add(m34x4, m34x4);

        S::store(y + (i*4+0)*W, S::add(S::add(S::add(tmp[i][0], p12), p34), vbias));
        S::store(y + (i*4+1)*W, S::add(S::add(m12, m34x2), vbias));
        S::store(y + (i*4+2)*W, S::add(S::add(p12, p34x4), vbias));
        S::store(y + (i*4+3)*W, S::add(S::add(S::add(m12, m34x8), tmp[i][5]), vbias));
    }
}

// ============================================================================
// Full Winograd F(4x4, 3x3) convolution -- SIMD-accelerated, NCHW layout
// ============================================================================

// SIMD-accelerated implementation: vectorized transforms, scalar GEMM.
// Vec = __m256 (AVX2, 8 tiles) or __m512 (AVX-512, 16 tiles).
template <typename Vec>
// @nnr-meta isa=[AVX2,AVX512] dtype=fp32 layout=NCHW special=[Winograd,GEMM] tiling=spatial
inline void winograd_conv2d_simd(
    float* output, const float* input, const float* w_transformed,
    const float* bias, int N, int iC, int iH, int iW,
    int M, int oH, int oW, int pH, int pW, float* workspace)
{
    constexpr int W = simd_f32<Vec>::W;
    const int tH = (oH + 3) / 4, tW = (oW + 3) / 4;
    const int num_tiles = tH * tW;
    float* V = workspace;
    float* M_buf = V + (size_t)36 * iC * num_tiles;

    for (int n = 0; n < N; ++n) {
        const float* x_n = input + (size_t)n * iC * iH * iW;
        float* y_n = output + (size_t)n * M * oH * oW;

        // Step 1: Input transform with SIMD batching
        for (int c = 0; c < iC; ++c) {
            const float* x_c = x_n + (size_t)c * iH * iW;
            int tile_base = 0;

            // SIMD: process W tiles per batch
            {
                float d_batch[36 * 16]; // max W=16 for AVX-512
                float v_batch[36 * 16];
                for (; tile_base + W <= num_tiles; tile_base += W) {
                    // Extract W tiles into position-major buffer
                    for (int k = 0; k < W; ++k) {
                        int tidx = tile_base + k;
                        int th = tidx / tW, tw = tidx % tW;
                        int h_start = th * 4 - pH, w_start = tw * 4 - pW;
                        for (int i = 0; i < 6; ++i)
                            for (int j = 0; j < 6; ++j) {
                                int ih = h_start + i, iw = w_start + j;
                                d_batch[(i*6+j)*W + k] = (ih >= 0 && ih < iH && iw >= 0 && iw < iW)
                                    ? x_c[ih*iW + iw] : 0.0f;
                            }
                    }

                    winograd_input_transform_simd<Vec>(v_batch, d_batch);

                    // Scatter to V[pos][c][tile_base..tile_base+W-1]
                    for (int pos = 0; pos < 36; ++pos)
                        copy_w<Vec>(&V[(size_t)pos*iC*num_tiles + c*num_tiles + tile_base],
                            &v_batch[pos*W]);
                }
            }

            // Scalar remainder
            for (; tile_base < num_tiles; ++tile_base) {
                int th = tile_base / tW, tw = tile_base % tW;
                int h_start = th * 4 - pH, w_start = tw * 4 - pW;
                float d[36];
                for (int i = 0; i < 6; ++i)
                    for (int j = 0; j < 6; ++j) {
                        int ih = h_start + i, iw = w_start + j;
                        d[i*6+j] = (ih >= 0 && ih < iH && iw >= 0 && iw < iW)
                            ? x_c[ih*iW + iw] : 0.0f;
                    }
                float v[36];
                winograd_input_transform(v, d);
                for (int pos = 0; pos < 36; ++pos)
                    V[(size_t)pos*iC*num_tiles + c*num_tiles + tile_base] = v[pos];
            }
        }

        // Step 2+3: GEMM (scalar) + output transform (SIMD batched)
        for (int m = 0; m < M; ++m) {
            // GEMM: dot product across input channels at each Winograd position
            for (int pos = 0; pos < 36; ++pos) {
                const float* W_pos_m = w_transformed + (size_t)pos*M*iC + m*iC;
                for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
                    float sum = 0.0f;
                    for (int c = 0; c < iC; ++c)
                        sum += W_pos_m[c] * V[(size_t)pos*iC*num_tiles + c*num_tiles + tile_idx];
                    M_buf[pos*num_tiles + tile_idx] = sum;
                }
            }

            // Output transform with SIMD batching + fused bias
            float bv = bias ? bias[m] : 0.0f;
            float* y_m = y_n + (size_t)m * oH * oW;
            int tile_base = 0;

            // SIMD: process W tiles per batch
            {
                float m_batch[36 * 16];
                float y_batch[16 * 16];
                for (; tile_base + W <= num_tiles; tile_base += W) {
                    // Gather M_buf[pos][tile_base..tile_base+W-1]
                    for (int pos = 0; pos < 36; ++pos)
                        copy_w<Vec>(&m_batch[pos*W],
                            &M_buf[pos*num_tiles + tile_base]);

                    winograd_output_transform_simd<Vec>(y_batch, m_batch, bv);

                    // Write tiles to output with bounds checking
                    for (int k = 0; k < W; ++k) {
                        int tidx = tile_base + k;
                        int th = tidx / tW, tw = tidx % tW;
                        for (int i = 0; i < 4; ++i) {
                            int oh = th * 4 + i;
                            if (oh >= oH) break;
                            for (int j = 0; j < 4; ++j) {
                                int ow = tw * 4 + j;
                                if (ow >= oW) break;
                                y_m[oh*oW + ow] = y_batch[(i*4+j)*W + k];
                            }
                        }
                    }
                }
            }

            // Scalar remainder
            for (; tile_base < num_tiles; ++tile_base) {
                int th = tile_base / tW, tw = tile_base % tW;
                float m_tile[36];
                for (int pos = 0; pos < 36; ++pos)
                    m_tile[pos] = M_buf[pos*num_tiles + tile_base];
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

// ============================================================================
// Cache-tiled SIMD Winograd with pre-packed GEMM and thread parallelism
// ============================================================================

// SIMD transforms + optimized GEMM, with thread-parallel transform stages.
// Interior/boundary tile split eliminates bounds checks for most tiles.
template <typename Vec>
// @nnr-meta isa=[AVX2,AVX512] dtype=fp32 layout=NCHW special=[Winograd,GEMM] tiling=[spatial,K] fusion=post_op
inline void winograd_conv2d_tiled_simd(
    float* output, const float* input, const float* w_transformed,
    const float* w_packed_wino, const float* bias,
    int N, int iC, int iH, int iW, int M, int oH, int oW,
    int pH, int pW, float* workspace, int wino_group,
    operator_t::post_fn_t post_fn = nullptr, const operator_t* fused_op = nullptr,
    int strip_row_start = -1, int strip_row_end = -1,
    int iH_logical = 0, int oH_logical = 0)
{
    constexpr int W = simd_f32<Vec>::W;
    // Logical heights for geometry/boundary; physical heights for channel stride
    const int iH_log = iH_logical > 0 ? iH_logical : iH;
    const int oH_log = oH_logical > 0 ? oH_logical : oH;
    const bool strip_mode = (strip_row_start >= 0);

    const int tH = (oH_log + 3) / 4, tW = (oW + 3) / 4;
    const int num_tiles = tH * tW;
    const size_t pa_sz = w_packed_wino ? pack_a_size(M, iC) : 0;

    // Strip tile range (or full range if not in strip mode)
    int range_tile0 = 0, range_tile1 = num_tiles;
    if (strip_mode) {
        int th_s = strip_row_start / 4;
        int th_e = std::min((strip_row_end + 3) / 4, tH);
        range_tile0 = th_s * tW;
        range_tile1 = th_e * tW;
    }
    const int num_groups = (range_tile1 - range_tile0 + wino_group - 1) / wino_group;

    // Output row clipping (strip boundaries or full oH)
    const int oh_clip_lo = strip_mode ? strip_row_start : 0;
    const int oh_clip_hi = strip_mode ? std::min(strip_row_end, oH_log) : oH_log;

    // Interior tile boundaries (tiles that need no padding/bounds checks)
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

            // Stage 1: Input transform -- parallel over input channels
            nnr::for_static(0, iC, iC > 4, [&](int c) {
                const float* x_c = x_n + (size_t)c * iH * iW;
                int t = 0;

                // SIMD batch: process W tiles at a time
                float d_batch[36 * 16]; // max W=16 for AVX-512
                float v_batch[36 * 16];
                for (; t + W <= gs; t += W) {
                    for (int k = 0; k < W; ++k) {
                        int tidx = tile_start + t + k;
                        int th = tidx / tW, tw = tidx % tW;
                        int h0 = th * 4 - pH, w0 = tw * 4 - pW;

                        if (th >= th_lo && th < th_hi && tw >= tw_lo && tw < tw_hi) {
                            // Interior: direct row loads, no bounds checks
                            const float* src = x_c + h0 * iW + w0;
                            for (int i = 0; i < 6; ++i) {
                                const float* row = src + i * iW;
                                int base = (i * 6) * W + k;
                                d_batch[base]         = row[0];
                                d_batch[base + W]     = row[1];
                                d_batch[base + 2*W]   = row[2];
                                d_batch[base + 3*W]   = row[3];
                                d_batch[base + 4*W]   = row[4];
                                d_batch[base + 5*W]   = row[5];
                            }
                        } else {
                            // Boundary: per-element bounds check
                            for (int i = 0; i < 6; ++i)
                                for (int j = 0; j < 6; ++j) {
                                    int ih = h0 + i, iw = w0 + j;
                                    d_batch[(i*6+j)*W + k] = (ih >= 0 && ih < iH_log && iw >= 0 && iw < iW)
                                        ? x_c[ih*iW + iw] : 0.0f;
                                }
                        }
                    }
                    winograd_input_transform_simd<Vec>(v_batch, d_batch);
                    for (int pos = 0; pos < 36; ++pos)
                        copy_w<Vec>(&V[(size_t)pos*iC*gs + c*gs + t],
                            &v_batch[pos*W]);
                }

                // Scalar remainder
                for (; t < gs; ++t) {
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
            });

            // Stage 2: 36 GEMMs -- M_buf[pos][M*gs] = W[pos][M*kC] * V[pos][kC*gs]
            if (w_packed_wino) {
                const float* A_batch[36];
                const float* B_batch[36];
                float* C_batch[36];
                for (int pos = 0; pos < 36; ++pos) {
                    A_batch[pos] = w_packed_wino + pos * pa_sz;
                    B_batch[pos] = V + (size_t)pos * iC * gs;
                    C_batch[pos] = M_buf + (size_t)pos * M * gs;
                }
                dgemm_packed_a_batch36(M, gs, iC, A_batch, B_batch, C_batch);
            } else {
                for (int pos = 0; pos < 36; ++pos)
                    dgemm_generic(M, gs, iC, w_transformed + (size_t)pos * M * iC,
                        V + (size_t)pos * iC * gs, M_buf + (size_t)pos * M * gs);
            }
            // Stage 3: Output transform -- parallel over output channels
            nnr::for_static(0, M, M > 4, [&](int m) {
                float bv = (!post_fn && bias) ? bias[m] : 0.0f;
                float* y_m = y_n + (size_t)m * oH * oW;
                int t = 0;

                // SIMD batch
                float m_batch[36 * 16];
                float y_batch[16 * 16];
                for (; t + W <= gs; t += W) {
                    for (int pos = 0; pos < 36; ++pos)
                        copy_w<Vec>(&m_batch[pos*W],
                            &M_buf[(size_t)pos*M*gs + m*gs + t]);
                    winograd_output_transform_simd<Vec>(y_batch, m_batch, bv);
                    for (int k = 0; k < W; ++k) {
                        int tidx = tile_start + t + k;
                        int th = tidx / tW, tw = tidx % tW;
                        int oh0 = th * 4, ow0 = tw * 4;
                        if (oh0 >= oh_clip_lo && oh0 + 4 <= oh_clip_hi && ow0 + 4 <= oW) {
                            // Interior: direct 4-row stores
                            for (int i = 0; i < 4; ++i) {
                                float* dst = y_m + (oh0 + i) * oW + ow0;
                                dst[0] = y_batch[(i*4+0)*W + k];
                                dst[1] = y_batch[(i*4+1)*W + k];
                                dst[2] = y_batch[(i*4+2)*W + k];
                                dst[3] = y_batch[(i*4+3)*W + k];
                            }
                        } else {
                            // Boundary: bounds-checked scatter with strip clipping
                            for (int i = 0; i < 4; ++i) {
                                int oh = oh0 + i;
                                if (oh < oh_clip_lo) continue;
                                if (oh >= oh_clip_hi) break;
                                for (int j = 0; j < 4; ++j) {
                                    int ow = ow0 + j;
                                    if (ow >= oW) break;
                                    y_m[oh*oW + ow] = y_batch[(i*4+j)*W + k];
                                }
                            }
                        }
                    }
                }

                // Scalar remainder
                for (; t < gs; ++t) {
                    int tidx = tile_start + t;
                    int th = tidx / tW, tw = tidx % tW;
                    float m_tile[36];
                    for (int pos = 0; pos < 36; ++pos)
                        m_tile[pos] = M_buf[(size_t)pos*M*gs + m*gs + t];
                    float y_tile[16];
                    winograd_output_transform(y_tile, m_tile);
                    int oh0 = th * 4, ow0 = tw * 4;
                    for (int i = 0; i < 4; ++i) {
                        int oh = oh0 + i;
                        if (oh < oh_clip_lo) continue;
                        if (oh >= oh_clip_hi) break;
                        for (int j = 0; j < 4; ++j) {
                            int ow = ow0 + j;
                            if (ow >= oW) break;
                            y_m[oh*oW + ow] = y_tile[i*4+j] + bv;
                        }
                    }
                }
            });
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
// NHWC SIMD Winograd F(4x4, 3x3) -- cache-tiled with dgemm_packed_b
// ============================================================================

// SIMD-accelerated NHWC Winograd: batched input transform + batched GEMM.
template <typename Vec>
// @nnr-meta isa=[AVX2,AVX512] dtype=fp32 layout=NHWC special=[Winograd,GEMM] tiling=spatial fusion=post_op
inline void winograd_conv2d_nhwc_simd(
    float* output, const float* input,
    const float* w_nhwc_transformed, const float* w_packed_b_wino,
    const float* bias,
    int N, int iC, int iH, int iW, int M, int oH, int oW,
    int pH, int pW, float* workspace, bool input_nhwc, int wino_group,
    operator_t::post_fn_t post_fn = nullptr, const operator_t* fused_op = nullptr)
{
    using S = simd_f32<Vec>;
    constexpr int W = S::W;
    const int tH = (oH + 3) / 4, tW = (oW + 3) / 4;
    const int num_tiles = tH * tW;
    const int num_groups = (num_tiles + wino_group - 1) / wino_group;
    const size_t pb_sz = w_packed_b_wino ? pack_b_size(iC, M) : 0;

    const int th_lo = (pH + 3) / 4;
    const int th_hi = std::max(th_lo, (iH - 6 + pH) / 4 + 1);
    const int tw_lo = (pW + 3) / 4;
    const int tw_hi = std::max(tw_lo, (iW - 6 + pW) / 4 + 1);

    for (int n = 0; n < N; ++n) {
        float* y_n = output + (size_t)n * oH * oW * M;

        for (int gid = 0; gid < num_groups; ++gid) {
            int tile_start = gid * wino_group;
            int tile_end = std::min(tile_start + wino_group, num_tiles);
            int gs = tile_end - tile_start;

            float* V = workspace;                      // [36][gs][iC]
            float* M_buf = V + (size_t)36 * gs * iC;   // [36][gs][M]

            // Stage 1: Input transform -- parallel over tiles, SIMD batched across channels
            nnr::for_static(0, gs, gs > 4, [&](int t) {
                int tidx = tile_start + t;
                int th = tidx / tW, tw = tidx % tW;
                int h0 = th * 4 - pH, w0 = tw * 4 - pW;
                bool interior = (th >= th_lo && th < th_hi && tw >= tw_lo && tw < tw_hi);

                int c = 0;
                // SIMD path: batch W channels at a time through the transform.
                // NHWC data has consecutive channels, so gather is just memcpy.
                if (input_nhwc && interior) {
                    const float* base = input + (size_t)n * iH * iW * iC;
                    float d_batch[36 * 16]; // max W=16 for AVX-512
                    float v_batch[36 * 16];
                    for (; c + W <= iC; c += W) {
                        for (int i = 0; i < 6; ++i)
                            for (int j = 0; j < 6; ++j)
                                copy_w<Vec>(&d_batch[(i*6+j)*W],
                                    &base[((h0+i)*iW + (w0+j))*iC + c]);
                        winograd_input_transform_simd<Vec>(v_batch, d_batch);
                        for (int pos = 0; pos < 36; ++pos)
                            copy_w<Vec>(&V[(size_t)pos * gs * iC + t * iC + c],
                                &v_batch[pos * W]);
                    }
                }

                // Scalar remainder (remaining channels, boundary tiles, NCHW input)
                for (; c < iC; ++c) {
                    float d[36];
                    if (interior) {
                        if (input_nhwc) {
                            const float* base = input + (size_t)n * iH * iW * iC;
                            for (int i = 0; i < 6; ++i)
                                for (int j = 0; j < 6; ++j)
                                    d[i*6+j] = base[((h0+i) * iW + (w0+j)) * iC + c];
                        } else {
                            const float* x_c = input + ((size_t)n * iC + c) * iH * iW;
                            const float* src = x_c + h0 * iW + w0;
                            for (int i = 0; i < 6; ++i)
                                std::memcpy(d + i * 6, src + i * iW, 6 * sizeof(float));
                        }
                    } else {
                        for (int i = 0; i < 6; ++i)
                            for (int j = 0; j < 6; ++j) {
                                int ih = h0 + i, iw = w0 + j;
                                if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
                                    if (input_nhwc)
                                        d[i*6+j] = input[((size_t)n * iH * iW + ih * iW + iw) * iC + c];
                                    else
                                        d[i*6+j] = input[((size_t)n * iC + c) * iH * iW + ih * iW + iw];
                                } else {
                                    d[i*6+j] = 0.0f;
                                }
                            }
                    }
                    float v[36];
                    winograd_input_transform(v, d);
                    for (int pos = 0; pos < 36; ++pos)
                        V[(size_t)pos * gs * iC + t * iC + c] = v[pos];
                }
            });

            // Stage 2: Batched 36 GEMMs -- M_buf[pos][gs*M] = V[pos][gs*kC] * W[pos][kC*M]
            if (w_packed_b_wino) {
                const float* V_batch[36];
                float* M_batch[36];
                const float* B_batch[36];
                for (int pos = 0; pos < 36; ++pos) {
                    V_batch[pos] = V + (size_t)pos * gs * iC;
                    M_batch[pos] = M_buf + (size_t)pos * gs * M;
                    B_batch[pos] = w_packed_b_wino + pos * pb_sz;
                }
                dgemm_packed_b_batch36(gs, M, iC, V_batch, B_batch, M_batch);
            } else {
                for (int pos = 0; pos < 36; ++pos)
                    dgemm_generic(gs, M, iC, V + (size_t)pos * gs * iC,
                        w_nhwc_transformed + (size_t)pos * iC * M,
                        M_buf + (size_t)pos * gs * M);
            }

            // Stage 3: Output transform -- SIMD batched over channels, parallel over tiles
            nnr::for_static(0, gs, gs > 4, [&](int t) {
                int tidx = tile_start + t;
                int th = tidx / tW, tw = tidx % tW;
                int oh0 = th * 4, ow0 = tw * 4;
                int m = 0;
                // SIMD batch: process W channels at a time
                float m_batch[36 * 16]; // 36 * max(W)
                float y_batch[16 * 16];
                for (; m + W <= M; m += W) {
                    for (int pos = 0; pos < 36; ++pos)
                        copy_w<Vec>(&m_batch[pos * W],
                            &M_buf[(size_t)pos * gs * M + t * M + m]);
                    winograd_output_transform_simd<Vec>(y_batch, m_batch, 0.0f);
                    // Scatter to NHWC output with per-channel bias
                    for (int i = 0; i < 4; ++i) {
                        int oh = oh0 + i;
                        if (oh >= oH) break;
                        for (int j = 0; j < 4; ++j) {
                            int ow = ow0 + j;
                            if (ow >= oW) break;
                            float* dst = y_n + (oh * oW + ow) * M + m;
                            const float* src = y_batch + (i * 4 + j) * W;
                            if (!post_fn && bias) {
                                S::store(dst, S::add(S::load(src), S::load(bias + m)));
                            } else {
                                copy_w<Vec>(dst, src);
                            }
                        }
                    }
                }
                // Scalar remainder
                for (; m < M; ++m) {
                    float m_tile[36];
                    for (int pos = 0; pos < 36; ++pos)
                        m_tile[pos] = M_buf[(size_t)pos * gs * M + t * M + m];
                    float y_tile[16];
                    winograd_output_transform(y_tile, m_tile);
                    float bv = (!post_fn && bias) ? bias[m] : 0.0f;
                    for (int i = 0; i < 4; ++i) {
                        int oh = oh0 + i;
                        if (oh >= oH) break;
                        for (int j = 0; j < 4; ++j) {
                            int ow = ow0 + j;
                            if (ow >= oW) break;
                            y_n[(oh * oW + ow) * M + m] = y_tile[i*4+j] + bv;
                        }
                    }
                }
            });
        }

        // Fused post-op: all spatial positions in one call (NHWC -- bias already applied)
        if (post_fn) {
            int off = (int)((size_t)n * oH * oW * M);
            post_fn(y_n, oH * oW, M, M, fused_op, nullptr, off);
        }
    }
}

} // namespace nnr
