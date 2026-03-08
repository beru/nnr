#pragma once
// NEON-accelerated Winograd F(4x4, 3x3) transforms and tiled convolution.
// Processes 4 tiles in parallel (128-bit NEON = 4 floats).
// Reuses the same algorithmic structure as the x64 SIMD Winograd.

#ifdef NNR_ARCH_ARM64

#include <arm_neon.h>
#include <cstring>
#include <algorithm>
#include "winograd_transforms.h"

namespace nnr {

// Primary template (specialized per platform)
template<typename Vec> struct simd_f32;

// NEON simd_f32 traits for Winograd transforms
template<> struct simd_f32<float32x4_t> {
    static constexpr int W = 4;
    // @nnr-meta isa=NEON dtype=fp32 special=Winograd
    static float32x4_t load(const float* p) { return vld1q_f32(p); }
    // @nnr-meta isa=NEON dtype=fp32 special=Winograd
    static void store(float* p, float32x4_t v) { vst1q_f32(p, v); }
    // @nnr-meta isa=NEON dtype=fp32 special=Winograd
    static float32x4_t add(float32x4_t a, float32x4_t b) { return vaddq_f32(a, b); }
    // @nnr-meta isa=NEON dtype=fp32 special=Winograd
    static float32x4_t sub(float32x4_t a, float32x4_t b) { return vsubq_f32(a, b); }
    // @nnr-meta isa=NEON dtype=fp32 special=Winograd
    static float32x4_t set1(float v) { return vdupq_n_f32(v); }
};

// Copy 4 floats (one NEON register) from src to dst.
// @nnr-meta isa=NEON dtype=fp32 special=Winograd
inline void copy_w_neon(float* __restrict dst, const float* __restrict src) {
    vst1q_f32(dst, vld1q_f32(src));
}

// BT * d * B for 4 tiles in parallel.
// d[36][4], v[36][4] -- position-major.
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=Winograd
inline void winograd_input_transform_neon(float* __restrict v, const float* __restrict d) {
    using S = simd_f32<float32x4_t>;
    constexpr int W = 4;

    float32x4_t dd[6][6];
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            dd[i][j] = S::load(d + (i*6+j)*W);

    // Column transform: tmp = BT * dd
    float32x4_t tmp[6][6];
    for (int j = 0; j < 6; j++) {
        float32x4_t r0 = dd[0][j], r1 = dd[1][j], r2 = dd[2][j];
        float32x4_t r3 = dd[3][j], r4 = dd[4][j], r5 = dd[5][j];

        float32x4_t r0x2 = S::add(r0, r0), r0x4 = S::add(r0x2, r0x2);
        float32x4_t r2x2 = S::add(r2, r2), r2x4 = S::add(r2x2, r2x2);
        float32x4_t r2x5 = S::add(r2x4, r2);
        tmp[0][j] = S::add(S::sub(r0x4, r2x5), r4);

        float32x4_t p12 = S::add(r1, r2), m12 = S::sub(r1, r2);
        float32x4_t p34 = S::add(r3, r4), m34 = S::sub(r3, r4);
        float32x4_t p12x2 = S::add(p12, p12), p12x4 = S::add(p12x2, p12x2);
        float32x4_t m12x2 = S::add(m12, m12), m12x4 = S::add(m12x2, m12x2);

        tmp[1][j] = S::sub(p34, p12x4);
        tmp[2][j] = S::sub(m12x4, m34);

        float32x4_t s42 = S::sub(r4, r2), s31 = S::sub(r3, r1);
        float32x4_t s31x2 = S::add(s31, s31);
        tmp[3][j] = S::add(s42, s31x2);
        tmp[4][j] = S::sub(s42, s31x2);

        float32x4_t r1x2 = S::add(r1, r1), r1x4 = S::add(r1x2, r1x2);
        float32x4_t r3x2 = S::add(r3, r3), r3x4 = S::add(r3x2, r3x2);
        float32x4_t r3x5 = S::add(r3x4, r3);
        tmp[5][j] = S::add(S::sub(r1x4, r3x5), r5);
    }

    // Row transform: v = tmp * B
    for (int i = 0; i < 6; i++) {
        float32x4_t r0 = tmp[i][0], r1 = tmp[i][1], r2 = tmp[i][2];
        float32x4_t r3 = tmp[i][3], r4 = tmp[i][4], r5 = tmp[i][5];

        float32x4_t r0x2 = S::add(r0, r0), r0x4 = S::add(r0x2, r0x2);
        float32x4_t r2x2 = S::add(r2, r2), r2x4 = S::add(r2x2, r2x2);
        float32x4_t r2x5 = S::add(r2x4, r2);
        S::store(v + (i*6+0)*W, S::add(S::sub(r0x4, r2x5), r4));

        float32x4_t p12 = S::add(r1, r2), m12 = S::sub(r1, r2);
        float32x4_t p34 = S::add(r3, r4), m34 = S::sub(r3, r4);
        float32x4_t p12x2 = S::add(p12, p12), p12x4 = S::add(p12x2, p12x2);
        float32x4_t m12x2 = S::add(m12, m12), m12x4 = S::add(m12x2, m12x2);

        S::store(v + (i*6+1)*W, S::sub(p34, p12x4));
        S::store(v + (i*6+2)*W, S::sub(m12x4, m34));

        float32x4_t s42 = S::sub(r4, r2), s31 = S::sub(r3, r1);
        float32x4_t s31x2 = S::add(s31, s31);
        S::store(v + (i*6+3)*W, S::add(s42, s31x2));
        S::store(v + (i*6+4)*W, S::sub(s42, s31x2));

        float32x4_t r1x2b = S::add(r1, r1), r1x4b = S::add(r1x2b, r1x2b);
        float32x4_t r3x2b = S::add(r3, r3), r3x4b = S::add(r3x2b, r3x2b);
        float32x4_t r3x5b = S::add(r3x4b, r3);
        S::store(v + (i*6+5)*W, S::add(S::sub(r1x4b, r3x5b), r5));
    }
}

// AT * M * A for 4 tiles in parallel, with fused bias addition.
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=Winograd
inline void winograd_output_transform_neon(float* __restrict y, const float* __restrict m, float bias_val) {
    using S = simd_f32<float32x4_t>;
    constexpr int W = 4;
    float32x4_t vbias = S::set1(bias_val);

    float32x4_t mm[6][6];
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            mm[i][j] = S::load(m + (i*6+j)*W);

    // Column transform: AT * mm
    float32x4_t tmp[4][6];
    for (int j = 0; j < 6; j++) {
        float32x4_t p12 = S::add(mm[1][j], mm[2][j]);
        float32x4_t m12 = S::sub(mm[1][j], mm[2][j]);
        float32x4_t p34 = S::add(mm[3][j], mm[4][j]);
        float32x4_t m34 = S::sub(mm[3][j], mm[4][j]);
        float32x4_t m34x2 = S::add(m34, m34);
        float32x4_t p34x2 = S::add(p34, p34), p34x4 = S::add(p34x2, p34x2);
        float32x4_t m34x4 = S::add(m34x2, m34x2), m34x8 = S::add(m34x4, m34x4);

        tmp[0][j] = S::add(S::add(mm[0][j], p12), p34);
        tmp[1][j] = S::add(m12, m34x2);
        tmp[2][j] = S::add(p12, p34x4);
        tmp[3][j] = S::add(S::add(m12, m34x8), mm[5][j]);
    }

    // Row transform with bias fusion
    for (int i = 0; i < 4; i++) {
        float32x4_t p12 = S::add(tmp[i][1], tmp[i][2]);
        float32x4_t m12 = S::sub(tmp[i][1], tmp[i][2]);
        float32x4_t p34 = S::add(tmp[i][3], tmp[i][4]);
        float32x4_t m34 = S::sub(tmp[i][3], tmp[i][4]);
        float32x4_t m34x2 = S::add(m34, m34);
        float32x4_t p34x2 = S::add(p34, p34), p34x4 = S::add(p34x2, p34x2);
        float32x4_t m34x4 = S::add(m34x2, m34x2), m34x8 = S::add(m34x4, m34x4);

        S::store(y + (i*4+0)*W, S::add(S::add(S::add(tmp[i][0], p12), p34), vbias));
        S::store(y + (i*4+1)*W, S::add(S::add(m12, m34x2), vbias));
        S::store(y + (i*4+2)*W, S::add(S::add(p12, p34x4), vbias));
        S::store(y + (i*4+3)*W, S::add(S::add(S::add(m12, m34x8), tmp[i][5]), vbias));
    }
}

// NEON-accelerated tiled Winograd F(4x4, 3x3) -- NCHW layout.
// Processes 4 tiles at a time through NEON transforms,
// uses batched NEON GEMM for the 36 pointwise multiplications.
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=Winograd tiling=[spatial,K] fusion=post_op
inline void winograd_conv2d_tiled_neon(
    float* output, const float* input, const float* w_transformed,
    const float* w_packed_wino, const float* bias,
    int N, int iC, int iH, int iW, int M, int oH, int oW,
    int pH, int pW, float* workspace, int wino_group,
    operator_t::post_fn_t post_fn = nullptr, const operator_t* fused_op = nullptr,
    int strip_row_start = -1, int strip_row_end = -1,
    int iH_logical = 0, int oH_logical = 0)
{
    constexpr int W = 4;
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

            float* V = workspace;
            float* M_buf = V + (size_t)36 * iC * gs;

            // Stage 1: Input transform -- parallel over input channels
            nnr::for_static(0, iC, iC > 4, [&](int c) {
                const float* x_c = x_n + (size_t)c * iH * iW;
                int t = 0;

                // NEON batch: 4 tiles at a time
                float d_batch[36 * 4];
                float v_batch[36 * 4];
                for (; t + W <= gs; t += W) {
                    for (int k = 0; k < W; ++k) {
                        int tidx = tile_start + t + k;
                        int th = tidx / tW, tw = tidx % tW;
                        int h0 = th * 4 - pH, w0 = tw * 4 - pW;

                        if (th >= th_lo && th < th_hi && tw >= tw_lo && tw < tw_hi) {
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
                            for (int i = 0; i < 6; ++i)
                                for (int j = 0; j < 6; ++j) {
                                    int ih = h0 + i, iw = w0 + j;
                                    d_batch[(i*6+j)*W + k] = (ih >= 0 && ih < iH_log && iw >= 0 && iw < iW)
                                        ? x_c[ih*iW + iw] : 0.0f;
                                }
                        }
                    }
                    winograd_input_transform_neon(v_batch, d_batch);
                    for (int pos = 0; pos < 36; ++pos)
                        copy_w_neon(&V[(size_t)pos*iC*gs + c*gs + t],
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

            // Stage 2: 36 batched GEMMs
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
                    dgemm_packed_a(M, gs, iC,
                        w_transformed + (size_t)pos * M * iC,
                        V + (size_t)pos * iC * gs,
                        M_buf + (size_t)pos * M * gs);
            }

            // Stage 3: Output transform -- parallel over output channels
            nnr::for_static(0, M, M > 4, [&](int m) {
                float bv = (!post_fn && bias) ? bias[m] : 0.0f;
                float* y_m = y_n + (size_t)m * oH * oW;
                int t = 0;

                // NEON batch
                float m_batch[36 * 4];
                float y_batch[16 * 4];
                for (; t + W <= gs; t += W) {
                    for (int pos = 0; pos < 36; ++pos)
                        copy_w_neon(&m_batch[pos*W],
                            &M_buf[(size_t)pos*M*gs + m*gs + t]);
                    winograd_output_transform_neon(y_batch, m_batch, bv);
                    for (int k = 0; k < W; ++k) {
                        int tidx = tile_start + t + k;
                        int th = tidx / tW, tw = tidx % tW;
                        int oh0 = th * 4, ow0 = tw * 4;
                        if (oh0 >= oh_clip_lo && oh0 + 4 <= oh_clip_hi && ow0 + 4 <= oW) {
                            for (int i = 0; i < 4; ++i) {
                                float* dst = y_m + (oh0 + i) * oW + ow0;
                                dst[0] = y_batch[(i*4+0)*W + k];
                                dst[1] = y_batch[(i*4+1)*W + k];
                                dst[2] = y_batch[(i*4+2)*W + k];
                                dst[3] = y_batch[(i*4+3)*W + k];
                            }
                        } else {
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

} // namespace nnr

#endif // NNR_ARCH_ARM64
