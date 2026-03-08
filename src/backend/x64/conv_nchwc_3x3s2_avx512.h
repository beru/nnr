#pragma once
// AVX-512 NCHWc direct Conv specialized for KH=KW=3, strideH=strideW=2.
//
// Shape: input [N, ICb, iH, iW, 16]  output [N, OCb, oH, oW, 16]
//        weight [OCb, ICb, 3, 3, 16ic, 16oc] (IC-blocked; same layout as
//        pack_weight_nchwc_blocked used by the generic conv_nchwc_avx512).
//
// Why a dedicated kernel:
// - The generic conv_nchwc_avx512 dispatches to a JIT 4x6 tile whose inner
//   `il=0..15` is a runtime loop. At KH=KW=3 that's 9 kh/kw iterations each
//   running a 16-iter runtime loop — ~15% branch/increment overhead.
// - At stride=2, s*strideW+kw gives input offsets 0,2,4,...; the JIT doesn't
//   benefit from these being compile-time constants, while a hand-unrolled
//   intrinsics kernel lets the compiler fold them into [base+disp] addresses.
// - Conv_320 (ssd-12, iC=256 M=512 38x38->19x19) and Conv_349 (324x256 3x3/s3,
//   also ineligible for Winograd) are the two largest stride>1 3x3 Convs in
//   ssd-12 float. Together they account for ~40 ms MT of the model budget.
//
// Register map (F=4 OCb × ow_tile=6 spatial positions):
//   zmm0..23:  acc[f][s] = f*6 + s     (24 accumulators)
//   zmm24:     weight                  (1)
//   zmm25..30: broadcast input scalars (6)
//   zmm31:     spare
//
// Outer loop: for each (oh, og=OCb group of up to 4), process ow_tile=6
// output positions at a time across the safe interior range.
//
// Interior-only path: callers invoke conv_nchwc_3x3s2_tile_interior for
// output positions whose 3x3 footprint lies entirely in the padded input.
// Edge positions fall back to the generic conv_nchwc_tile<false/true>
// (reused from conv_nchwc_avx512.h).

#ifdef NNR_ARCH_X64

#include <immintrin.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include "thread_pool.h"

#ifndef NNR_FORCEINLINE
#ifdef _MSC_VER
#define NNR_FORCEINLINE __forceinline
#else
#define NNR_FORCEINLINE inline __attribute__((always_inline))
#endif
#endif

namespace nnr {

// ---------------------------------------------------------------------------
// Interior tile: F OC blocks × ow_tile spatial positions, templated on F and
// ow_tile so the compiler keeps accumulators in registers and unrolls the
// broadcast/FMA bodies. KH=KW=3, strideW=2 hard-coded — no bounds checks.
//
//   out_row:                 output at (ob_start, oh, ow_start)
//   in_base:                 input at (icb=0, ih=oh*2-padH, iw=ow_start*2-padW)
//   w_base:                  weight for first OC block, (icb=0, kh=0, kw=0)
//   ob_bias:                 pointer to F*16 bias floats, or nullptr
//   ICb:                     IC block count
//   in_row_stride_floats:    = IW * 16
//   in_blk_stride_floats:    = IH * IW * 16 (floats per ICb slab)
//   w_ob_stride_floats:      floats between successive OC blocks in weights
//   out_ob_stride_floats:    = oH * oW * 16
// Interior contract: the 3x3 footprint at every (ow_start+s, oh) is fully
// inside the padded input — no bounds checks.
// ---------------------------------------------------------------------------
template <int F, int ow_tile>
// @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=[NCHWc,Direct]
static NNR_FORCEINLINE void conv_nchwc_3x3s2_tile(
    float* __restrict out_row,
    const float* __restrict in_base,     // input at (icb=0, ih=oh*2-padH, iw=ow_start*2-padW)
    const float* __restrict w_base,      // weight for ob_start, (icb=0, kh=0, kw=0)
    const float* __restrict ob_bias,
    int ICb,
    int64_t in_row_stride_floats,        // = IW * 16
    int64_t in_blk_stride_floats,        // = IH * IW * 16
    int64_t w_ob_stride_floats,          // floats per OC block (= ICb*9*256)
    int64_t out_ob_stride_floats)        // = oH * oW * 16
{
    static_assert(F >= 1 && F <= 4, "F must be 1..4");
    static_assert(ow_tile >= 1 && ow_tile <= 6, "ow_tile must be 1..6");

    __m512 acc[F][ow_tile];
    if (ob_bias) {
        for (int f = 0; f < F; ++f) {
            __m512 b = _mm512_loadu_ps(ob_bias + (size_t)f * 16);
            for (int s = 0; s < ow_tile; ++s) acc[f][s] = b;
        }
    } else {
        for (int f = 0; f < F; ++f)
            for (int s = 0; s < ow_tile; ++s) acc[f][s] = _mm512_setzero_ps();
    }

    constexpr int STRIDE_W = 2;
    constexpr int BLOCK = 16;
    constexpr size_t W_ICB_FLOATS = 9ull * 16 * 16;  // 3*3*16*16 = 2304

    for (int icb = 0; icb < ICb; ++icb) {
        const float* in_icb = in_base + (size_t)icb * in_blk_stride_floats;
        const float* w_icb  = w_base  + (size_t)icb * W_ICB_FLOATS;

        // Fully unroll 3x3 kh/kw.
        for (int kh = 0; kh < 3; ++kh) {
            const float* in_row = in_icb + (size_t)kh * in_row_stride_floats;
            for (int kw = 0; kw < 3; ++kw) {
                // Weight tile at this (kh, kw): [16ic, 16oc] per OC block,
                // stride w_ob_stride_floats between OC blocks.
                const float* w_k = w_icb + (size_t)(kh * 3 + kw) * 256;

                // Unroll il=0..15 (16 input channels within one ICb segment).
                // Per il: bcast ow_tile input scalars, load F weight rows,
                // ow_tile*F FMAs.
                #pragma unroll
                for (int il = 0; il < 16; ++il) {
                    __m512 bcast[ow_tile];
                    for (int s = 0; s < ow_tile; ++s) {
                        const int iw_s = s * STRIDE_W + kw;  // 2s + kw
                        bcast[s] = _mm512_set1_ps(in_row[(size_t)iw_s * BLOCK + il]);
                    }
                    for (int f = 0; f < F; ++f) {
                        __m512 wv = _mm512_loadu_ps(w_k
                            + (size_t)f * w_ob_stride_floats
                            + (size_t)il * 16);
                        for (int s = 0; s < ow_tile; ++s)
                            acc[f][s] = _mm512_fmadd_ps(wv, bcast[s], acc[f][s]);
                    }
                }
            }
        }
    }

    // Store accumulators.
    for (int f = 0; f < F; ++f) {
        float* out_f = out_row + (size_t)f * out_ob_stride_floats;
        for (int s = 0; s < ow_tile; ++s)
            _mm512_storeu_ps(out_f + (size_t)s * 16, acc[f][s]);
    }
}

// ---------------------------------------------------------------------------
// Edge tile: same contract as the interior kernel but with bounds-checked
// input reads. Used on the first few columns where iw=ow*2-padW could be
// negative. Templated on F=1..4, ow_tile=1..6.
// ---------------------------------------------------------------------------
template <int F, int ow_tile>
// @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=[NCHWc,Direct]
static NNR_FORCEINLINE void conv_nchwc_3x3s2_tile_safe(
    float* __restrict out_row,
    const float* __restrict in_base_icb0_pad,  // input at (icb=0, unpadded)
    const float* __restrict w_base,
    const float* __restrict ob_bias,
    int ICb,
    int ow_start,     // first output column in this tile
    int oh,
    int IH, int IW,
    int padH, int padW,
    int64_t in_blk_stride_floats,
    int64_t w_ob_stride_floats,
    int64_t out_ob_stride_floats)
{
    static_assert(F >= 1 && F <= 4, "F must be 1..4");
    static_assert(ow_tile >= 1 && ow_tile <= 6, "ow_tile must be 1..6");

    __m512 acc[F][ow_tile];
    if (ob_bias) {
        for (int f = 0; f < F; ++f) {
            __m512 b = _mm512_loadu_ps(ob_bias + (size_t)f * 16);
            for (int s = 0; s < ow_tile; ++s) acc[f][s] = b;
        }
    } else {
        for (int f = 0; f < F; ++f)
            for (int s = 0; s < ow_tile; ++s) acc[f][s] = _mm512_setzero_ps();
    }

    constexpr int STRIDE_W = 2;
    constexpr int BLOCK = 16;
    constexpr size_t W_ICB_FLOATS = 9ull * 16 * 16;
    const size_t in_row_stride_floats = (size_t)IW * BLOCK;

    for (int icb = 0; icb < ICb; ++icb) {
        const float* in_icb = in_base_icb0_pad + (size_t)icb * in_blk_stride_floats;
        const float* w_icb  = w_base            + (size_t)icb * W_ICB_FLOATS;

        for (int kh = 0; kh < 3; ++kh) {
            const int ih = oh * 2 - padH + kh;
            if (ih < 0 || ih >= IH) continue;
            const float* in_row = in_icb + (size_t)ih * in_row_stride_floats;

            for (int kw = 0; kw < 3; ++kw) {
                const float* w_k = w_icb + (size_t)(kh * 3 + kw) * 256;

                for (int il = 0; il < 16; ++il) {
                    __m512 bcast[ow_tile];
                    for (int s = 0; s < ow_tile; ++s) {
                        const int iw = (ow_start + s) * STRIDE_W - padW + kw;
                        if (iw >= 0 && iw < IW)
                            bcast[s] = _mm512_set1_ps(in_row[(size_t)iw * BLOCK + il]);
                        else
                            bcast[s] = _mm512_setzero_ps();
                    }
                    for (int f = 0; f < F; ++f) {
                        __m512 wv = _mm512_loadu_ps(w_k
                            + (size_t)f * w_ob_stride_floats
                            + (size_t)il * 16);
                        for (int s = 0; s < ow_tile; ++s)
                            acc[f][s] = _mm512_fmadd_ps(wv, bcast[s], acc[f][s]);
                    }
                }
            }
        }
    }

    for (int f = 0; f < F; ++f) {
        float* out_f = out_row + (size_t)f * out_ob_stride_floats;
        for (int s = 0; s < ow_tile; ++s)
            _mm512_storeu_ps(out_f + (size_t)s * 16, acc[f][s]);
    }
}

// ---------------------------------------------------------------------------
// Driver: full NCHWc 3x3 stride=2 Conv.
//
// input:  [N, ICb, IH, IW, 16]
// output: [N, OCb, OH, OW, 16]
// weight: [OCb, ICb, 3, 3, 16ic, 16oc] (packed via pack_weight_nchwc_blocked)
// bias:   [OC_padded] or nullptr
//
// Threading responsibility is OUTSIDE this function — pass a lambda dispatch
// if you want MT. This version is the serial reference; the dispatched
// `conv_nchwc_3x3s2_driver` below uses nnr::for_dynamic over (n, og, oh).
// ---------------------------------------------------------------------------
// @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=[NCHWc,Direct]
inline void conv_nchwc_3x3s2_avx512(
    float* __restrict output,
    const float* __restrict input,
    const float* __restrict weight,
    const float* __restrict bias,
    int N, int IC, int OC,
    int IH, int IW,
    int OH, int OW,
    int padH, int padW)
{
    constexpr int S_TILE = 6;
    constexpr int FC = 4;

    const int ICb = (IC + 15) / 16;
    const int OCb = (OC + 15) / 16;
    const int OCg = (OCb + FC - 1) / FC;
    const size_t in_batch  = (size_t)ICb * IH * IW * 16;
    const size_t out_batch = (size_t)OCb * OH * OW * 16;
    const size_t w_ob_stride_floats = (size_t)ICb * 9 * 256;
    const size_t out_ob_stride_floats = (size_t)OH * OW * 16;
    const size_t in_row_stride_floats = (size_t)IW * 16;
    const size_t in_blk_stride_floats = (size_t)IH * IW * 16;

    // Interior safe range: output positions whose entire 3x3 footprint at
    // stride=2 lies inside the unpadded image.
    //   Column safe:  ow_start*2 - padW >= 0  AND  ow_end*2 - padW + 2 < IW
    //   → ow_safe_start = ceil(padW / 2)
    //   → ow_safe_end   = floor((IW - 3 + padW) / 2) + 1
    //   Row safe:  oh*2 - padH >= 0  AND  oh*2 - padH + 2 < IH
    //   → oh_safe_start = ceil(padH / 2)
    //   → oh_safe_end   = floor((IH - 3 + padH) / 2) + 1
    const int ow_safe_start = (padW + 1) / 2;
    const int ow_safe_end   = (IW + padW >= 3)
        ? std::min(OW, (IW - 3 + padW) / 2 + 1) : 0;
    const int oh_safe_start = (padH + 1) / 2;
    const int oh_safe_end   = (IH + padH >= 3)
        ? std::min(OH, (IH - 3 + padH) / 2 + 1) : 0;

    const int total_work = N * OCg * OH;
    nnr::for_dynamic(0, total_work, nnr::compute_threads(total_work),
        [&](int /*tid*/, int work_idx) {
        const int n    = work_idx / (OCg * OH);
        const int rem2 = work_idx % (OCg * OH);
        const int og   = rem2 / OH;
        const int oh   = rem2 % OH;

        const float* in_n  = input  + (size_t)n * in_batch;
        float*       out_n = output + (size_t)n * out_batch;

        {
            const int ob_start = og * FC;
            const int F = std::min(FC, OCb - ob_start);

            const float* w_base  = weight + (size_t)ob_start * w_ob_stride_floats;
            const float* ob_bias = bias ? bias + (size_t)ob_start * 16 : nullptr;
            float*       out_row_base = out_n + (size_t)ob_start * out_ob_stride_floats;

            {
                float* out_row = out_row_base + (size_t)oh * OW * 16;
                const bool h_interior = (oh >= oh_safe_start && oh < oh_safe_end);

                // -- Left edge: bounds-checked, ow_tile=1 per column --
                const int left_end = h_interior ? ow_safe_start : OW;
                for (int ow = 0; ow < left_end && ow < OW; ++ow) {
                    // Use ow_tile=1 safe tile, F up to 4.
                    const float* in_base_icb0 = in_n;  // safe kernel computes ih/iw internally from oh/ow_start
                    if (F == 4) conv_nchwc_3x3s2_tile_safe<4, 1>(out_row + (size_t)ow * 16, in_base_icb0, w_base, ob_bias, ICb, ow, oh, IH, IW, padH, padW, in_blk_stride_floats, w_ob_stride_floats, out_ob_stride_floats);
                    else if (F == 3) conv_nchwc_3x3s2_tile_safe<3, 1>(out_row + (size_t)ow * 16, in_base_icb0, w_base, ob_bias, ICb, ow, oh, IH, IW, padH, padW, in_blk_stride_floats, w_ob_stride_floats, out_ob_stride_floats);
                    else if (F == 2) conv_nchwc_3x3s2_tile_safe<2, 1>(out_row + (size_t)ow * 16, in_base_icb0, w_base, ob_bias, ICb, ow, oh, IH, IW, padH, padW, in_blk_stride_floats, w_ob_stride_floats, out_ob_stride_floats);
                    else             conv_nchwc_3x3s2_tile_safe<1, 1>(out_row + (size_t)ow * 16, in_base_icb0, w_base, ob_bias, ICb, ow, oh, IH, IW, padH, padW, in_blk_stride_floats, w_ob_stride_floats, out_ob_stride_floats);
                }

                if (!h_interior) return;

                // -- Safe interior: use S_TILE=6 bulk + ow_tile remainder --
                const int ow_end = ow_safe_end;
                int ow = ow_safe_start;
                for (; ow + S_TILE <= ow_end; ow += S_TILE) {
                    const int ih0 = oh * 2 - padH;
                    const int iw0 = ow * 2 - padW;
                    const float* in_base = in_n + (size_t)ih0 * in_row_stride_floats
                                                + (size_t)iw0 * 16;
                    if (F == 4) conv_nchwc_3x3s2_tile<4, S_TILE>(out_row + (size_t)ow * 16, in_base, w_base, ob_bias, ICb, in_row_stride_floats, in_blk_stride_floats, w_ob_stride_floats, out_ob_stride_floats);
                    else if (F == 3) conv_nchwc_3x3s2_tile<3, S_TILE>(out_row + (size_t)ow * 16, in_base, w_base, ob_bias, ICb, in_row_stride_floats, in_blk_stride_floats, w_ob_stride_floats, out_ob_stride_floats);
                    else if (F == 2) conv_nchwc_3x3s2_tile<2, S_TILE>(out_row + (size_t)ow * 16, in_base, w_base, ob_bias, ICb, in_row_stride_floats, in_blk_stride_floats, w_ob_stride_floats, out_ob_stride_floats);
                    else             conv_nchwc_3x3s2_tile<1, S_TILE>(out_row + (size_t)ow * 16, in_base, w_base, ob_bias, ICb, in_row_stride_floats, in_blk_stride_floats, w_ob_stride_floats, out_ob_stride_floats);
                }
                // Interior remainder (ow_tile < S_TILE): fall back to the safe
                // path for simplicity — at ow_tile<6 the arithmetic saving
                // from dropping bounds checks is small.
                for (; ow < ow_end; ++ow) {
                    if (F == 4) conv_nchwc_3x3s2_tile_safe<4, 1>(out_row + (size_t)ow * 16, in_n, w_base, ob_bias, ICb, ow, oh, IH, IW, padH, padW, in_blk_stride_floats, w_ob_stride_floats, out_ob_stride_floats);
                    else if (F == 3) conv_nchwc_3x3s2_tile_safe<3, 1>(out_row + (size_t)ow * 16, in_n, w_base, ob_bias, ICb, ow, oh, IH, IW, padH, padW, in_blk_stride_floats, w_ob_stride_floats, out_ob_stride_floats);
                    else if (F == 2) conv_nchwc_3x3s2_tile_safe<2, 1>(out_row + (size_t)ow * 16, in_n, w_base, ob_bias, ICb, ow, oh, IH, IW, padH, padW, in_blk_stride_floats, w_ob_stride_floats, out_ob_stride_floats);
                    else             conv_nchwc_3x3s2_tile_safe<1, 1>(out_row + (size_t)ow * 16, in_n, w_base, ob_bias, ICb, ow, oh, IH, IW, padH, padW, in_blk_stride_floats, w_ob_stride_floats, out_ob_stride_floats);
                }

                // -- Right edge: bounds-checked --
                for (int owr = ow_safe_end; owr < OW; ++owr) {
                    if (F == 4) conv_nchwc_3x3s2_tile_safe<4, 1>(out_row + (size_t)owr * 16, in_n, w_base, ob_bias, ICb, owr, oh, IH, IW, padH, padW, in_blk_stride_floats, w_ob_stride_floats, out_ob_stride_floats);
                    else if (F == 3) conv_nchwc_3x3s2_tile_safe<3, 1>(out_row + (size_t)owr * 16, in_n, w_base, ob_bias, ICb, owr, oh, IH, IW, padH, padW, in_blk_stride_floats, w_ob_stride_floats, out_ob_stride_floats);
                    else if (F == 2) conv_nchwc_3x3s2_tile_safe<2, 1>(out_row + (size_t)owr * 16, in_n, w_base, ob_bias, ICb, owr, oh, IH, IW, padH, padW, in_blk_stride_floats, w_ob_stride_floats, out_ob_stride_floats);
                    else             conv_nchwc_3x3s2_tile_safe<1, 1>(out_row + (size_t)owr * 16, in_n, w_base, ob_bias, ICb, owr, oh, IH, IW, padH, padW, in_blk_stride_floats, w_ob_stride_floats, out_ob_stride_floats);
                }
            }
        }
    });
}

} // namespace nnr

#endif // NNR_ARCH_X64
