#pragma once
// AVX-512 NCHWc blocked-layout convolution kernels.
//
// NCHWc layout: [N, C/c, H, W, c] where c=16 (AVX-512 lane width).
// Channels within a block are contiguous — natural SIMD loads.
//
// General conv: 4x6 OC tiling — processes up to 4 OC blocks together,
// amortizing input broadcasts across multiple filter loads.
// JIT kernel for the hot interior path; intrinsics fallback for edges.
//
// Pointwise 1x1: kept as 1x14 OC tiling (separate weight layout).

#include "nnr.h"
#include "thread_pool.h"
#include "jit_conv_nchwc.h"
#include <immintrin.h>

#ifndef NNR_FORCEINLINE
#ifdef _MSC_VER
#define NNR_FORCEINLINE __forceinline
#else
#define NNR_FORCEINLINE inline __attribute__((always_inline))
#endif
#endif

namespace nnr {

// ---------------------------------------------------------------------------
// AVX-512 pointwise (1x1) NCHWc convolution — 1x14 register-tiled.
// (Kept as-is: separate weight layout [OCb, IC, 16], no KH/KW loop)
//
// input:  [N, ICb, H, W, 16]
// output: [N, OCb, H, W, 16]
// weight: [OCb, IC, 16]  packed via pack_weight_nchwc_1x1
// bias:   [OC_padded] or nullptr
// ---------------------------------------------------------------------------
// Per-tile body: WT spatial accumulators × 16 OC lanes.
// Templated on WT so compiler unrolls the accumulator loops and keeps
// acc[] entirely in zmm registers (critical: runtime wt spills to memory).
template<int WT>
// @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=NCHWc
static NNR_FORCEINLINE void conv1x1_nchwc_tile_avx512(
    float* __restrict out_row,
    const float* __restrict in_nh_base,   // input at (n, ib=0, h, 0)
    const float* __restrict w_ob,
    __m512 bv,
    int IC,
    int HW,
    int w_base)
{
    __m512 acc[WT];
    for (int t = 0; t < WT; t++) acc[t] = bv;

    for (int ic = 0; ic < IC; ic++) {
        const int ib = ic >> 4;
        const int il = ic & 15;
        __m512 wv = _mm512_loadu_ps(w_ob + (size_t)ic * 16);
        const float* in_row = in_nh_base + ((size_t)ib * HW + w_base) * 16;

        for (int t = 0; t < WT; t++) {
            __m512 av = _mm512_set1_ps(in_row[t * 16 + il]);
            acc[t] = _mm512_fmadd_ps(wv, av, acc[t]);
        }
    }

    for (int t = 0; t < WT; t++)
        _mm512_storeu_ps(out_row + (w_base + t) * 16, acc[t]);
}

// @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=NCHWc
inline void conv1x1_nchwc_avx512(
    float* __restrict output,
    const float* __restrict input,
    const float* __restrict weight,
    const float* __restrict bias,
    int N, int IC, int OC, int H, int W,
    void (*post_fn)(float*, int) = nullptr)
{
    constexpr int W_TILE = 14;

    const int HW = H * W;
    const int ICb = (IC + 15) / 16;
    const int OCb = (OC + 15) / 16;
    const size_t in_batch  = (size_t)ICb * HW * 16;
    const size_t out_batch = (size_t)OCb * HW * 16;

    const int total_work = N * OCb * H;

    nnr::for_dynamic(0, total_work, nnr::compute_threads(total_work), [&](int /*tid*/, int work_idx) {
        const int n  = work_idx / (OCb * H);
        const int rem = work_idx % (OCb * H);
        const int ob = rem / H;
        const int h  = rem % H;

        const float* in_n  = input  + n * in_batch;
        float*       out_n = output + n * out_batch;

        float* out_row = out_n + ((size_t)ob * HW + h * W) * 16;
        const float* w_ob = weight + (size_t)ob * IC * 16;
        const float* in_nh = in_n + (size_t)h * W * 16;

        __m512 bv = bias ? _mm512_loadu_ps(bias + ob * 16) : _mm512_setzero_ps();

        int w_base = 0;
        for (; w_base + W_TILE <= W; w_base += W_TILE) {
            conv1x1_nchwc_tile_avx512<W_TILE>(out_row, in_nh, w_ob, bv, IC, HW, w_base);
        }

        const int wt = W - w_base;
        switch (wt) {
            case  0: break;
            case  1: conv1x1_nchwc_tile_avx512< 1>(out_row, in_nh, w_ob, bv, IC, HW, w_base); break;
            case  2: conv1x1_nchwc_tile_avx512< 2>(out_row, in_nh, w_ob, bv, IC, HW, w_base); break;
            case  3: conv1x1_nchwc_tile_avx512< 3>(out_row, in_nh, w_ob, bv, IC, HW, w_base); break;
            case  4: conv1x1_nchwc_tile_avx512< 4>(out_row, in_nh, w_ob, bv, IC, HW, w_base); break;
            case  5: conv1x1_nchwc_tile_avx512< 5>(out_row, in_nh, w_ob, bv, IC, HW, w_base); break;
            case  6: conv1x1_nchwc_tile_avx512< 6>(out_row, in_nh, w_ob, bv, IC, HW, w_base); break;
            case  7: conv1x1_nchwc_tile_avx512< 7>(out_row, in_nh, w_ob, bv, IC, HW, w_base); break;
            case  8: conv1x1_nchwc_tile_avx512< 8>(out_row, in_nh, w_ob, bv, IC, HW, w_base); break;
            case  9: conv1x1_nchwc_tile_avx512< 9>(out_row, in_nh, w_ob, bv, IC, HW, w_base); break;
            case 10: conv1x1_nchwc_tile_avx512<10>(out_row, in_nh, w_ob, bv, IC, HW, w_base); break;
            case 11: conv1x1_nchwc_tile_avx512<11>(out_row, in_nh, w_ob, bv, IC, HW, w_base); break;
            case 12: conv1x1_nchwc_tile_avx512<12>(out_row, in_nh, w_ob, bv, IC, HW, w_base); break;
            case 13: conv1x1_nchwc_tile_avx512<13>(out_row, in_nh, w_ob, bv, IC, HW, w_base); break;
        }

        if (post_fn) {
            post_fn(out_row, W * 16);
        }
    });
}

// ---------------------------------------------------------------------------
// Intrinsics tile: 1 OC block × wt spatial (fallback for edge pixels).
// Used when JIT is not available or for bounds-checked edge positions.
// ---------------------------------------------------------------------------
template<bool safe>
// @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=NCHWc
static NNR_FORCEINLINE void conv_nchwc_tile(
    float* out, int wt,
    const float* in_n, const float* w_ob, __m512 bv,
    int ow_start, int oh, int ICb,
    int IH, int IW, int KH, int KW, int kSpatial,
    int strideH, int strideW, int padH, int padW)
{
    constexpr int MAX_TILE = 14;
    __m512 acc[MAX_TILE];
    for (int t = 0; t < wt; t++) acc[t] = bv;

    for (int icb = 0; icb < ICb; icb++) {
        const float* in_blk = in_n + (size_t)icb * IH * IW * 16;

        for (int kh = 0; kh < KH; kh++) {
            const int ih = oh * strideH + kh - padH;
            if constexpr (!safe) {
                if (ih < 0 || ih >= IH) continue;
            }

            const float* in_row = in_blk + (size_t)ih * IW * 16;

            for (int kw = 0; kw < KW; kw++) {
                const float* w_tile = w_ob +
                    ((size_t)icb * kSpatial + kh * KW + kw) * 256;

                if constexpr (safe) {
                    const float* in_ptrs[MAX_TILE];
                    for (int t = 0; t < wt; t++)
                        in_ptrs[t] = in_row + (size_t)((ow_start + t) * strideW + kw - padW) * 16;

                    for (int il = 0; il < 16; il++) {
                        __m512 wv = _mm512_loadu_ps(w_tile + il * 16);
                        for (int t = 0; t < wt; t++)
                            acc[t] = _mm512_fmadd_ps(wv,
                                _mm512_set1_ps(in_ptrs[t][il]), acc[t]);
                    }
                } else {
                    for (int il = 0; il < 16; il++) {
                        __m512 wv = _mm512_loadu_ps(w_tile + il * 16);
                        for (int t = 0; t < wt; t++) {
                            const int iw = (ow_start + t) * strideW + kw - padW;
                            if (iw < 0 || iw >= IW) continue;
                            acc[t] = _mm512_fmadd_ps(wv,
                                _mm512_set1_ps(in_row[iw * 16 + il]), acc[t]);
                        }
                    }
                }
            }
        }
    }

    for (int t = 0; t < wt; t++)
        _mm512_storeu_ps(out + (ow_start + t) * 16, acc[t]);
}

// ---------------------------------------------------------------------------
// AVX-512 general NCHWc convolution — 4x6 OC-tiled.
//
// input:  [N, ICb, IH, IW, 16]
// output: [N, OCb, OH, OW, 16]
// weight: [OCb, ICb, KH, KW, 16ic, 16oc]  packed via pack_weight_nchwc_blocked
// bias:   [OC_padded] or nullptr
//
// Threading: N × OCg × OH  where OCg = ceil(OCb / 4).
// Each work item processes F=min(4, remaining) OC blocks.
// Interior safe path: 4x6 JIT kernel.
// Edge pixels: 1x14 intrinsics fallback (called F times per OC block).
// ---------------------------------------------------------------------------
// @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=[NCHWc,JIT]
inline void conv_nchwc_avx512(
    float* __restrict output,
    const float* __restrict input,
    const float* __restrict weight,
    const float* __restrict bias,
    int N, int IC, int OC,
    int IH, int IW,
    int OH, int OW,
    int KH, int KW,
    int strideH, int strideW,
    int padH, int padW,
    void (*post_fn)(float*, int) = nullptr)
{
    constexpr int S_TILE = 6;   // spatial tile for 4x6 JIT
    constexpr int FC = 4;       // max FilterCount
    constexpr int W_TILE = 14;  // spatial tile for intrinsics fallback

    const int ICb = (IC + 15) / 16;
    const int OCb = (OC + 15) / 16;
    const int OCg = (OCb + FC - 1) / FC;  // OC groups
    const size_t in_batch  = (size_t)ICb * IH * IW * 16;
    const size_t out_batch = (size_t)OCb * OH * OW * 16;
    const int kSpatial = KH * KW;
    const size_t w_ob_stride = (size_t)ICb * kSpatial * 256;  // floats between OC blocks
    const size_t out_ob_stride = (size_t)OH * OW * 16;        // floats between OC blocks

    // Interior safe ranges
    const int ow_safe_start = padW > 0 ? (padW + strideW - 1) / strideW : 0;
    const int ow_safe_end   = (IW + padW >= KW)
        ? std::min(OW, (IW + padW - KW) / strideW + 1) : 0;
    const int oh_safe_start = padH > 0 ? (padH + strideH - 1) / strideH : 0;
    const int oh_safe_end   = (IH + padH >= KH)
        ? std::min(OH, (IH + padH - KH) / strideH + 1) : 0;

    const int64_t IW_bytes  = (int64_t)IW * 64;
    const int64_t blk_bytes = (int64_t)IH * IW * 64;
    const int64_t w_ob_stride_bytes = (int64_t)w_ob_stride * 4;
    const int64_t out_ob_stride_bytes = (int64_t)out_ob_stride * 4;

#ifdef NNR_USE_XBYAK
    // Resolve 4x6 JIT kernels for the main F value and remainder F.
    // Two spatial sizes: full tile (S_TILE) and remainder.
    const int safe_width = ow_safe_end - ow_safe_start;
    const int full_tiles = safe_width / S_TILE;
    const int s_remainder = safe_width % S_TILE;
    const int F_main = std::min(FC, OCb);
    const int F_rem  = OCb % FC;  // 0 means no partial group

    static const bool jit_disabled = (std::getenv("NNR_DISABLE_JIT_NCHWC") != nullptr);

    // JIT kernels for F_main (typically 4)
    jit_nchwc_4x6_fn_t jit_full_main = nullptr, jit_rem_main = nullptr;
    // JIT kernels for F_rem (1-3, for last OC group)
    jit_nchwc_4x6_fn_t jit_full_tail = nullptr, jit_rem_tail = nullptr;

    if (!jit_disabled) {
        if (full_tiles > 0 && jit_nchwc_4x6_eligible(KH, KW, strideW, F_main, S_TILE))
            jit_full_main = resolve_jit_nchwc_4x6(KH, KW, strideW, F_main, S_TILE);
        if (s_remainder > 0 && jit_nchwc_4x6_eligible(KH, KW, strideW, F_main, s_remainder))
            jit_rem_main = resolve_jit_nchwc_4x6(KH, KW, strideW, F_main, s_remainder);

        if (F_rem > 0 && F_rem != F_main) {
            if (full_tiles > 0 && jit_nchwc_4x6_eligible(KH, KW, strideW, F_rem, S_TILE))
                jit_full_tail = resolve_jit_nchwc_4x6(KH, KW, strideW, F_rem, S_TILE);
            if (s_remainder > 0 && jit_nchwc_4x6_eligible(KH, KW, strideW, F_rem, s_remainder))
                jit_rem_tail = resolve_jit_nchwc_4x6(KH, KW, strideW, F_rem, s_remainder);
        }
    }
#endif

    const int total_work = N * OCg * OH;

    nnr::for_dynamic(0, total_work, nnr::compute_threads(total_work),
        [&](int /*tid*/, int work_idx) {
        const int n   = work_idx / (OCg * OH);
        const int rem2 = work_idx % (OCg * OH);
        const int og  = rem2 / OH;
        const int oh  = rem2 % OH;

        const int ob_start = og * FC;
        const int F = std::min(FC, OCb - ob_start);

        const float* in_n    = input + n * in_batch;
        const float* w_base  = weight + (size_t)ob_start * ICb * kSpatial * 256;
        const float* ob_bias = bias ? bias + ob_start * 16 : nullptr;

        // Output row for first OC block
        float* out_row = output + n * out_batch
                       + (size_t)ob_start * OH * OW * 16
                       + (size_t)oh * OW * 16;

        const bool h_interior = (oh >= oh_safe_start && oh < oh_safe_end);

        if (h_interior && ow_safe_start < ow_safe_end) {
            // --- Left edge: intrinsics, 1 OC block at a time ---
            for (int ow = 0; ow < ow_safe_start; ow++) {
                for (int f = 0; f < F; f++) {
                    __m512 bv = ob_bias ? _mm512_loadu_ps(ob_bias + f * 16) : _mm512_setzero_ps();
                    conv_nchwc_tile<false>(
                        out_row + f * out_ob_stride, 1,
                        in_n, w_base + f * w_ob_stride, bv,
                        ow, oh, ICb, IH, IW, KH, KW, kSpatial,
                        strideH, strideW, padH, padW);
                }
            }

            // --- Interior: 4x6 JIT path ---
            int ow = ow_safe_start;
#ifdef NNR_USE_XBYAK
            // Select JIT kernels for this work item's F value
            jit_nchwc_4x6_fn_t jit_full = (F == F_main) ? jit_full_main : jit_full_tail;
            jit_nchwc_4x6_fn_t jit_rem  = (F == F_main) ? jit_rem_main  : jit_rem_tail;

            if (jit_full || jit_rem) {
                const float* in_h = in_n
                    + (size_t)(oh * strideH - padH) * IW * 16;

                for (int ti = 0; ti < full_tiles && jit_full; ti++, ow += S_TILE) {
                    const float* in_base = in_h + (size_t)(ow * strideW - padW) * 16;
                    jit_full(out_row + (size_t)ow * 16, in_base, w_base,
                             (int64_t)ICb, IW_bytes, blk_bytes, ob_bias,
                             w_ob_stride_bytes, out_ob_stride_bytes);
                }
                if (jit_rem) {
                    const float* in_base = in_h + (size_t)(ow * strideW - padW) * 16;
                    jit_rem(out_row + (size_t)ow * 16, in_base, w_base,
                            (int64_t)ICb, IW_bytes, blk_bytes, ob_bias,
                            w_ob_stride_bytes, out_ob_stride_bytes);
                    ow += s_remainder;
                }
            } else
#endif
            {
                // Intrinsics fallback: process each OC block separately
                for (; ow + W_TILE <= ow_safe_end; ow += W_TILE) {
                    for (int f = 0; f < F; f++) {
                        __m512 bv = ob_bias ? _mm512_loadu_ps(ob_bias + f * 16) : _mm512_setzero_ps();
                        conv_nchwc_tile<true>(
                            out_row + f * out_ob_stride, W_TILE,
                            in_n, w_base + f * w_ob_stride, bv,
                            ow, oh, ICb, IH, IW, KH, KW, kSpatial,
                            strideH, strideW, padH, padW);
                    }
                }
                if (ow < ow_safe_end) {
                    for (int f = 0; f < F; f++) {
                        __m512 bv = ob_bias ? _mm512_loadu_ps(ob_bias + f * 16) : _mm512_setzero_ps();
                        conv_nchwc_tile<true>(
                            out_row + f * out_ob_stride, ow_safe_end - ow,
                            in_n, w_base + f * w_ob_stride, bv,
                            ow, oh, ICb, IH, IW, KH, KW, kSpatial,
                            strideH, strideW, padH, padW);
                    }
                    ow = ow_safe_end;
                }
            }

            // --- Right edge: intrinsics, 1 OC block at a time ---
            for (; ow < OW; ow++) {
                for (int f = 0; f < F; f++) {
                    __m512 bv = ob_bias ? _mm512_loadu_ps(ob_bias + f * 16) : _mm512_setzero_ps();
                    conv_nchwc_tile<false>(
                        out_row + f * out_ob_stride, 1,
                        in_n, w_base + f * w_ob_stride, bv,
                        ow, oh, ICb, IH, IW, KH, KW, kSpatial,
                        strideH, strideW, padH, padW);
                }
            }
        } else {
            // H-edge rows: all with bounds checking, per OC block
            for (int ow_base = 0; ow_base < OW; ow_base += W_TILE) {
                int wt = std::min(W_TILE, OW - ow_base);
                for (int f = 0; f < F; f++) {
                    __m512 bv = ob_bias ? _mm512_loadu_ps(ob_bias + f * 16) : _mm512_setzero_ps();
                    conv_nchwc_tile<false>(
                        out_row + f * out_ob_stride, wt,
                        in_n, w_base + f * w_ob_stride, bv,
                        ow_base, oh, ICb, IH, IW, KH, KW, kSpatial,
                        strideH, strideW, padH, padW);
                }
            }
        }

        if (post_fn) {
            for (int f = 0; f < F; f++)
                post_fn(out_row + f * out_ob_stride, OW * 16);
        }
    });
}

} // namespace nnr
