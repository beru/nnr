#pragma once
// AVX-512 Winograd F(4x4, 3x3) convolution for NCHWc (NCHW16c) layout.
//
// Input:  [N, ICb, iH, iW, 16]
// Output: [N, OCb, oH, oW, 16]
// Weight (pre-transformed): [36, OCb, iC, 16_oc]  — NCHWc-native layout
//
// Design notes (v3):
// - V layout is NCHWc-native: [36][ICb][num_tiles][16_ic]. The inner 16 is
//   contiguous channel lanes, so the input transform scatters nothing —
//   all stores are unit-stride 16-float writes (same as NCHW Winograd).
// - U layout is [36][OCb][iC][16_oc]. Inside the matmul, walking the flat
//   ic dimension gives one 16-float weight load per ic step. This is the
//   exact shape used by conv1x1_nchwc_tile_avx512: 14 tile accumulators ×
//   16_oc lanes, with one broadcast + one load + 14 fmadds per ic step.
// - Central step is 36 independent (M × num_tiles × iC) matmuls, one per
//   Winograd position. Each matmul uses a `W_TILE=14` register-tiled
//   micro-kernel over `num_tiles`, with `16_oc` as the SIMD axis.
// - Both the input/output transforms and the central matmul stay in NCHWc
//   layout end-to-end. No layout translation, no scatter stores.
//
// This kernel's purpose is to measure whether an NCHWc-native Winograd can
// match the production NCHW Winograd (winograd_conv2d_tiled_simd) on raw
// throughput, so we know whether shipping one makes sense once the
// boundary-reorder savings from NCHWc chain integration are factored in.
//
// AVX-512 only. AVX2 fallback is TODO.

#ifdef NNR_ARCH_X64

#include <immintrin.h>
#include <cstring>
#include <algorithm>
#include "thread_pool.h"
#include "backend/cpu/kernel/winograd_transforms.h"
#include "backend/x64/winograd_x64.h"  // winograd_input/output_transform_simd

#ifndef NNR_FORCEINLINE
#ifdef _MSC_VER
#define NNR_FORCEINLINE __forceinline
#else
#define NNR_FORCEINLINE inline __attribute__((always_inline))
#endif
#endif

namespace nnr {

// ---------------------------------------------------------------------------
// Weight pre-transform: NCHW weight [M, iC, 3, 3] -> [36, OCb, iC, 16_oc]
//
// Each (oc, ic) 3x3 kernel becomes a 6x6 Winograd tile u[36] via
// winograd_filter_transform. Then we scatter u[pos] to
// U[pos][ocb=oc/16][ic][ol=oc%16].
//
// Walking U at a fixed (pos, ocb) by ic gives the natural `load one
// 16-float weight row per ic step` pattern for the matmul micro-kernel,
// with unit stride between ic rows.
// ---------------------------------------------------------------------------
// @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=[Winograd,NCHWc]
inline void winograd_nchwc_weight_transform(
    float* __restrict U,                 // [36][OCb][iC][16_oc]
    const float* __restrict weight,      // [M][iC][3][3]
    int M, int iC)
{
    const int OCb = M / 16;
    std::memset(U, 0, (size_t)36 * OCb * iC * 16 * sizeof(float));

    for (int oc = 0; oc < M; ++oc) {
        const int ocb = oc / 16;
        const int ol  = oc % 16;
        for (int ic = 0; ic < iC; ++ic) {
            float u[36];
            winograd_filter_transform(u, weight + ((size_t)oc * iC + ic) * 9);
            for (int pos = 0; pos < 36; ++pos) {
                // U[pos][ocb][ic][ol] = u[pos]
                size_t idx = (((size_t)pos * OCb + ocb) * (size_t)iC + ic) * 16 + ol;
                U[idx] = u[pos];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Input transform for NCHWc: loads 6x6 spatial patches of 16 channels
// (one unit-stride 16-wide load per cell) and writes V[36][ICb][gs][16]
// with unit-stride 16-wide stores — no scatter.
//
// Reuses winograd_input_transform_simd<__m512> from winograd_x64.h, which is
// lane-independent. Here the 16 lanes represent 16 channels of one tile
// instead of 16 spatial tiles of one channel.
//
// Processes tiles [tile_start, tile_start+gs) of the overall num_tiles domain.
// V is sized for `gs` tiles, not `num_tiles` — the outer group loop in the
// driver reuses the same V buffer across groups so the workspace fits in L3.
// ---------------------------------------------------------------------------
// @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=[Winograd,NCHWc]
inline void winograd_nchwc_input_transform_group(
    float* __restrict V,                // [36][ICb][gs][16]
    const float* __restrict x_n,        // [ICb][iH][iW][16]
    int iH, int iW, int pH, int pW,
    int ICb, int gs, int tile_start, int tW,
    int nthreads)
{
    // Interior bounds — tiles whose 6×6 patch is fully inside the unpadded
    // image. Outside this range we per-cell bounds-check.
    const int th_lo = (pH + 3) / 4;
    const int th_hi = std::max(th_lo, (iH - 6 + pH) / 4 + 1);
    const int tw_lo = (pW + 3) / 4;
    const int tw_hi = std::max(tw_lo, (iW - 6 + pW) / 4 + 1);

    const size_t V_pos_stride = (size_t)ICb * gs * 16;
    const int total = ICb * gs;

    // Parallelize over (icb × gs). Each (icb, t_local) step streams a 6×6
    // patch of 16 channels through a Winograd input transform and writes 36
    // unit-stride 16-wide slots in V. Tasks are independent: V slices at
    // different (icb, t_local) never alias. d_batch/v_batch are per-thread
    // automatic storage. `nthreads` set by the driver's cap policy.
    nnr::thread_pool_t::get().for_static(0, total, nthreads, [&](int idx) {
        const int icb     = idx / gs;
        const int t_local = idx % gs;
        const int t       = tile_start + t_local;

        alignas(64) float d_batch[36 * 16];
        alignas(64) float v_batch[36 * 16];

        const float* x_icb      = x_n + (size_t)icb * iH * iW * 16;
        float*       V_icb_base = V + (size_t)icb * gs * 16;

        const int th = t / tW;
        const int tw = t % tW;
        const int h0 = th * 4 - pH;
        const int w0 = tw * 4 - pW;

        const bool interior =
            (th >= th_lo && th < th_hi && tw >= tw_lo && tw < tw_hi);

        if (interior) {
            // 6 rows × 6 columns of 16-wide unit-stride loads.
            const float* base = x_icb + (size_t)h0 * iW * 16 + (size_t)w0 * 16;
            for (int i = 0; i < 6; ++i) {
                const float* row = base + (size_t)i * iW * 16;
                // Use SIMD loads + stores into d_batch so the compiler
                // keeps these in zmm registers without going through
                // memcpy/CRT machinery.
                _mm512_store_ps(&d_batch[(i * 6 + 0) * 16], _mm512_loadu_ps(row +  0));
                _mm512_store_ps(&d_batch[(i * 6 + 1) * 16], _mm512_loadu_ps(row + 16));
                _mm512_store_ps(&d_batch[(i * 6 + 2) * 16], _mm512_loadu_ps(row + 32));
                _mm512_store_ps(&d_batch[(i * 6 + 3) * 16], _mm512_loadu_ps(row + 48));
                _mm512_store_ps(&d_batch[(i * 6 + 4) * 16], _mm512_loadu_ps(row + 64));
                _mm512_store_ps(&d_batch[(i * 6 + 5) * 16], _mm512_loadu_ps(row + 80));
            }
        } else {
            // Boundary: per-cell bounds check, load or zero 16 channels.
            for (int i = 0; i < 6; ++i) {
                const int ih = h0 + i;
                for (int j = 0; j < 6; ++j) {
                    const int iw = w0 + j;
                    float* dst = &d_batch[(i * 6 + j) * 16];
                    if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
                        _mm512_store_ps(dst,
                            _mm512_loadu_ps(x_icb + ((size_t)ih * iW + iw) * 16));
                    } else {
                        _mm512_store_ps(dst, _mm512_setzero_ps());
                    }
                }
            }
        }

        // Lane-independent Winograd transform: treats the 16 lanes as
        // 16 channels of one tile.
        winograd_input_transform_simd<__m512>(v_batch, d_batch);

        // Store 36 positions × 16 channels as 36 unit-stride 16-wide writes.
        // Use t_local (not t) — V is sized for gs tiles, not num_tiles.
        float* dst_tile = V_icb_base + (size_t)t_local * 16;
        for (int pos = 0; pos < 36; ++pos) {
            _mm512_storeu_ps(dst_tile + pos * V_pos_stride,
                _mm512_load_ps(&v_batch[pos * 16]));
        }
    });
}

// ---------------------------------------------------------------------------
// Matmul micro-kernel: 14 tile accumulators × 16_oc SIMD lanes.
//
// Per (pos, ocb), accumulate
//   M[pos][ocb][t0..t0+WT-1][16_oc] +=
//     sum over ic of U[pos][ocb][ic][16_oc] * V[pos][icb=ic/16][t][il=ic%16]
//
// The inner loop walks a flat ic dimension. Each ic step does:
//   - 1 weight load (16 floats, contiguous — U stride 16 per ic)
//   - WT broadcasts (one per tile)
//   - WT fmadds
//
// WT=14 keeps all accumulators + 1 weight + 1 broadcast = 16 zmm, well
// within the 32 zmm budget, matching conv1x1_nchwc_tile_avx512.
//
// Templated on WT so the compiler unrolls the per-tile loop and keeps
// acc[] entirely in registers. (Runtime WT spills to memory — see
// feedback_runtime_tile_spill.)
// ---------------------------------------------------------------------------
template <int WT>
// @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=[Winograd,NCHWc,GEMM] tiling=[K,MR,NR]
static NNR_FORCEINLINE void winograd_nchwc_matmul_tile(
    float* __restrict M_pos_ocb,           // M[pos][ocb][t0*16]
    const float* __restrict U_pos_ocb,     // U[pos][ocb][0 .. iC*16]
    const float* __restrict V_pos_base,    // V[pos][0][0]
    int iC, int gs, int t0)
{
    __m512 acc[WT];
    for (int t = 0; t < WT; ++t) acc[t] = _mm512_setzero_ps();

    // Flat ic walk — compiler knows iC is a compile-time multiple of 16 via
    // the caller contract, but the loop is plain ic to keep things simple.
    for (int ic = 0; ic < iC; ++ic) {
        const int icb = ic >> 4;
        const int il  = ic & 15;
        const __m512 wv = _mm512_loadu_ps(U_pos_ocb + (size_t)ic * 16);
        // V[pos][icb][t0 + t][il] — for fixed icb and varying t, this is
        // unit-stride across t (stride 16 between tiles). `gs` is the
        // tile dimension of V within this group.
        const float* v_row = V_pos_base + (size_t)icb * gs * 16 + (size_t)t0 * 16;
        for (int t = 0; t < WT; ++t) {
            const __m512 av = _mm512_set1_ps(v_row[t * 16 + il]);
            acc[t] = _mm512_fmadd_ps(wv, av, acc[t]);
        }
    }

    for (int t = 0; t < WT; ++t)
        _mm512_storeu_ps(M_pos_ocb + (t0 + t) * 16, acc[t]);
}

// @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=[Winograd,NCHWc,GEMM] tiling=[K,MR,NR]
inline void winograd_nchwc_matmul_group(
    float* __restrict M,              // [36][OCb][gs][16]
    const float* __restrict U,        // [36][OCb][iC][16]
    const float* __restrict V,        // [36][ICb][gs][16]
    int OCb, int iC, int gs,
    int nthreads)
{
    constexpr int W_TILE = 14;
    const size_t M_pos_stride = (size_t)OCb * gs * 16;
    const size_t U_pos_stride = (size_t)OCb * iC * 16;
    const size_t V_pos_stride = (size_t)(iC / 16) * gs * 16;

    // Parallelize over (pos × ocb). Each (pos, ocb) is an independent slice
    // of U, V, and M — zero cross-task data dependencies. Splitting over
    // ocb in addition to pos raises the task count from 36 to 36*OCb, which
    // is what small-channel shapes (OCb=8) need to saturate a ≥16-thread
    // pool without leaving cores idle at the tail of the 36-task schedule.
    // `nthreads` set by the driver's cap policy.
    const int total = 36 * OCb;
    nnr::thread_pool_t::get().for_static(0, total, nthreads, [&](int idx) {
        const int pos = idx / OCb;
        const int ocb = idx % OCb;

        const float* U_pos     = U + (size_t)pos * U_pos_stride;
        const float* V_pos     = V + (size_t)pos * V_pos_stride;
        float*       M_pos     = M + (size_t)pos * M_pos_stride;
        const float* U_pos_ocb = U_pos + (size_t)ocb * iC * 16;
        float*       M_pos_ocb = M_pos + (size_t)ocb * gs * 16;

        int t0 = 0;
        for (; t0 + W_TILE <= gs; t0 += W_TILE) {
            winograd_nchwc_matmul_tile<W_TILE>(M_pos_ocb, U_pos_ocb, V_pos,
                iC, gs, t0);
        }
        const int wt = gs - t0;
        switch (wt) {
            case  0: break;
            case  1: winograd_nchwc_matmul_tile< 1>(M_pos_ocb, U_pos_ocb, V_pos, iC, gs, t0); break;
            case  2: winograd_nchwc_matmul_tile< 2>(M_pos_ocb, U_pos_ocb, V_pos, iC, gs, t0); break;
            case  3: winograd_nchwc_matmul_tile< 3>(M_pos_ocb, U_pos_ocb, V_pos, iC, gs, t0); break;
            case  4: winograd_nchwc_matmul_tile< 4>(M_pos_ocb, U_pos_ocb, V_pos, iC, gs, t0); break;
            case  5: winograd_nchwc_matmul_tile< 5>(M_pos_ocb, U_pos_ocb, V_pos, iC, gs, t0); break;
            case  6: winograd_nchwc_matmul_tile< 6>(M_pos_ocb, U_pos_ocb, V_pos, iC, gs, t0); break;
            case  7: winograd_nchwc_matmul_tile< 7>(M_pos_ocb, U_pos_ocb, V_pos, iC, gs, t0); break;
            case  8: winograd_nchwc_matmul_tile< 8>(M_pos_ocb, U_pos_ocb, V_pos, iC, gs, t0); break;
            case  9: winograd_nchwc_matmul_tile< 9>(M_pos_ocb, U_pos_ocb, V_pos, iC, gs, t0); break;
            case 10: winograd_nchwc_matmul_tile<10>(M_pos_ocb, U_pos_ocb, V_pos, iC, gs, t0); break;
            case 11: winograd_nchwc_matmul_tile<11>(M_pos_ocb, U_pos_ocb, V_pos, iC, gs, t0); break;
            case 12: winograd_nchwc_matmul_tile<12>(M_pos_ocb, U_pos_ocb, V_pos, iC, gs, t0); break;
            case 13: winograd_nchwc_matmul_tile<13>(M_pos_ocb, U_pos_ocb, V_pos, iC, gs, t0); break;
        }
    });
}

// ---------------------------------------------------------------------------
// Output transform for NCHWc: reads M[36][OCb][num_tiles][16_oc] with
// unit-stride 16-wide loads, applies AT·M·A with 16-wide SIMD, writes 4×4
// output tiles into NCHW16c output with per-channel bias fused at store.
// ---------------------------------------------------------------------------
// @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=[Winograd,NCHWc] fusion=post_op
inline void winograd_nchwc_output_transform_group(
    float* __restrict y_n,              // [OCb][oH][oW][16]
    const float* __restrict M,          // [36][OCb][gs][16]
    const float* __restrict bias,       // [M] or nullptr
    int oH, int oW, int OCb, int gs, int tile_start, int tW,
    int nthreads,
    int oh_clip_lo = 0, int oh_clip_hi = 0)
{
    if (oh_clip_hi <= 0) oh_clip_hi = oH;
    const size_t M_pos_stride = (size_t)OCb * gs * 16;
    const int total = OCb * gs;

    // Parallelize over (ocb × gs). Each task gathers 36 positions
    // × 16 floats from M, applies the lane-independent output transform,
    // and scatters 16 floats to 4×4 output positions in y. Tasks are
    // independent: distinct (ocb, t_local) write distinct output regions.
    // `nthreads` set by the driver's cap policy.
    nnr::thread_pool_t::get().for_static(0, total, nthreads, [&](int idx) {
        const int ocb     = idx / gs;
        const int t_local = idx % gs;
        const int t_abs   = tile_start + t_local;

        const int th = t_abs / tW;
        const int tw = t_abs % tW;
        const int oh0 = th * 4;
        const int ow0 = tw * 4;

        // Skip tiles entirely outside the clip range
        if (oh0 + 4 <= oh_clip_lo || oh0 >= oh_clip_hi) return;

        alignas(64) float m_batch[36 * 16];
        alignas(64) float y_batch[16 * 16];

        const __m512 vbias = bias
            ? _mm512_loadu_ps(bias + (size_t)ocb * 16)
            : _mm512_setzero_ps();
        float*       y_ocb      = y_n + (size_t)ocb * oH * oW * 16;
        const float* M_ocb_base = M + (size_t)ocb * gs * 16;

        // Gather 36 positions — each is a 16-wide unit-stride load.
        // Use t_local (not t_abs) — M is sized for gs tiles, not num_tiles.
        const float* src_tile = M_ocb_base + (size_t)t_local * 16;
        for (int pos = 0; pos < 36; ++pos) {
            _mm512_store_ps(&m_batch[pos * 16],
                _mm512_loadu_ps(src_tile + (size_t)pos * M_pos_stride));
        }

        // Lane-independent output transform. Per-lane bias is fused in
        // the scatter below because each of 16 lanes has its own bias.
        winograd_output_transform_simd<__m512>(y_batch, m_batch, 0.0f);

        if (oh0 >= oh_clip_lo && oh0 + 4 <= oh_clip_hi && ow0 + 4 <= oW) {
            for (int i = 0; i < 4; ++i) {
                float* row = y_ocb + (size_t)(oh0 + i) * oW * 16 + (size_t)ow0 * 16;
                _mm512_storeu_ps(row +  0,
                    _mm512_add_ps(_mm512_load_ps(&y_batch[(i * 4 + 0) * 16]), vbias));
                _mm512_storeu_ps(row + 16,
                    _mm512_add_ps(_mm512_load_ps(&y_batch[(i * 4 + 1) * 16]), vbias));
                _mm512_storeu_ps(row + 32,
                    _mm512_add_ps(_mm512_load_ps(&y_batch[(i * 4 + 2) * 16]), vbias));
                _mm512_storeu_ps(row + 48,
                    _mm512_add_ps(_mm512_load_ps(&y_batch[(i * 4 + 3) * 16]), vbias));
            }
        } else {
            for (int i = 0; i < 4; ++i) {
                const int oh = oh0 + i;
                if (oh < oh_clip_lo || oh >= oh_clip_hi) continue;
                float* row = y_ocb + (size_t)oh * oW * 16;
                for (int j = 0; j < 4; ++j) {
                    const int ow = ow0 + j;
                    if (ow >= oW) break;
                    __m512 yv = _mm512_load_ps(&y_batch[(i * 4 + j) * 16]);
                    _mm512_storeu_ps(row + (size_t)ow * 16, _mm512_add_ps(yv, vbias));
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Per-group tile count: chosen so V+M fits in L3 with headroom for the
// input/output tensors and for transient state from adjacent ops in the
// graph. The un-grouped workspace is 36*(iC+M)*num_tiles*4 bytes, which
// for ssd-12 Block 1 is 103 MB — 3× the Zen 4 L3 capacity. That forces
// stage 2 to read V from DRAM after stage 1's barrier, which was the
// root cause of the 2026-04-16 MT regression (see
// project_nchwc_winograd_shipped.md, feedback_playground_vs_production.md).
//
// Target budget: one-quarter of a single L3 domain (8 MB on a 32 MB
// Zen 4 CCD, 4 MB on a 16 MB Zen 2 CCX). That leaves ~3/4 of L3 for the
// input activation, model weights, and whatever adjacent ops are doing.
// Floor at 32 tiles so the matmul's W_TILE=14 main body still fires —
// below that the remainder switch handles every tile.
//
// Must be deterministic from (iC, M, num_tiles) so that Conv_reshape.h
// and the exec driver agree on the workspace size.
// ---------------------------------------------------------------------------
// @nnr-meta isa=scalar dtype=fp32 special=[Winograd,NCHWc]
inline int winograd_nchwc_group_size(int iC, int M, int num_tiles)
{
    const size_t BUDGET_BYTES =
        (size_t)cpu_features().l3_kb_per_domain * 1024 / 4;
    constexpr int    MIN_GS       = 32;
    const size_t denom = (size_t)36 * (size_t)(iC + M) * sizeof(float);
    int gs = (int)(BUDGET_BYTES / denom);
    if (gs < MIN_GS) gs = MIN_GS;
    if (gs > num_tiles) gs = num_tiles;
    return gs;
}

// ---------------------------------------------------------------------------
// Workspace sizing: V[36][ICb][gs][16] + M[36][OCb][gs][16].
// Sized for ONE group — the outer group loop in winograd_conv2d_nchwc_avx512
// reuses the same V/M buffer across groups.
// Requires iC and M to be multiples of 16 (caller enforces).
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Cap policies for the three Winograd stages. Two distinct concerns:
//
//  - Stages 1/3 (input/output transform) dispatch tiny tasks
//    (~200/400 ns each). Per-task work is independent of shape; the
//    only lever is total task count. Cross-CCD wake costs ~20 µs; a
//    thread needs ≳30 tasks (6 µs) to amortize it. At 12 threads that
//    means total ≳ 1500 for full MT to pay. Below that, cap at one
//    L3 domain. ssd-12 256→256 stage 1 (total≈1800) sits above the
//    threshold, yolov2 1024→1024 stage 1 (total=1024) below.
//
//  - Stage 2 (36 position matmuls) is bandwidth-bound on large shapes
//    but compute-bound on small ones. Cap only when U (36·iC·M·4 B)
//    exceeds one L3 domain's effective capacity — beyond that, the
//    second domain's cores also hit DRAM for U reads so cross-domain
//    dispatch buys nothing. When U fits, each domain caches U locally
//    and extra cores deliver proportional speedup (ssd-12 256→256
//    U=9.4 MB loses ~2 ms/conv at the cap).
//
// Stage-2 threshold = half an L3 domain (16 MB on a 32 MB Zen 4 CCD,
// 8 MB on a 16 MB Zen 2 CCX), leaving room for V+M (group workspace)
// plus input/output tensors and adjacent-op state. Raising this ratio
// pessimizes ssd-12's 512→256 on Zen 4 (U=18.9 MB fits numerically but
// L3 thrashes once co-resident tensors are factored).
//
// On monolithic parts num_l3_physical()==num_physical() so every gate
// reduces to num_physical().
// ---------------------------------------------------------------------------
// @nnr-meta isa=scalar dtype=fp32 special=[Winograd,NCHWc]
inline int winograd_nchwc_transform_threads(int total_tasks)
{
    constexpr int TRANSFORM_MT_MIN_TASKS = 1500;
    auto& pool = nnr::thread_pool_t::get();
    return (total_tasks >= TRANSFORM_MT_MIN_TASKS)
        ? pool.num_physical()
        : pool.num_l3_physical();
}

// @nnr-meta isa=scalar dtype=fp32 special=[Winograd,NCHWc]
inline int winograd_nchwc_matmul_threads(int iC, int M)
{
    const size_t U_CAP_THRESHOLD =
        (size_t)cpu_features().l3_kb_per_domain * 1024 / 2;
    const size_t U_bytes = (size_t)36 * (size_t)iC * (size_t)M * sizeof(float);
    auto& pool = nnr::thread_pool_t::get();
    return (U_bytes > U_CAP_THRESHOLD) ? pool.num_l3_physical() : pool.num_physical();
}

// @nnr-meta isa=scalar dtype=fp32 special=[Winograd,NCHWc]
inline size_t winograd_nchwc_workspace_size(int iC, int M, int oH, int oW)
{
    const int tH = (oH + 3) / 4, tW = (oW + 3) / 4;
    const int num_tiles = tH * tW;
    const int gs = winograd_nchwc_group_size(iC, M, num_tiles);
    return (size_t)36 * ((size_t)iC + (size_t)M) * (size_t)gs * sizeof(float);
}

// ---------------------------------------------------------------------------
// Full NCHWc Winograd F(4x4, 3x3) convolution. Single-threaded reference
// driver for the playground bench.
//
// input:  [N][ICb][iH][iW][16]
// output: [N][OCb][oH][oW][16]
// U:      [36][OCb][iC][16]   (pre-transformed via winograd_nchwc_weight_transform)
// bias:   [M] or nullptr
// ---------------------------------------------------------------------------
// @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=[Winograd,NCHWc] tiling=spatial fusion=post_op
inline void winograd_conv2d_nchwc_avx512(
    float* __restrict output,
    const float* __restrict input,
    const float* __restrict U,
    const float* __restrict bias,
    int N, int iC, int iH, int iW,
    int M, int oH, int oW,
    int pH, int pW,
    void* workspace,
    int strip_row_start = -1, int strip_row_end = -1,
    int oH_logical = 0)
{
    // Logical oH for tile grid geometry; physical oH for channel strides.
    // In strip mode with ring buffers, oH may be ring_H (small) while
    // oH_logical is the full image height for correct tiling.
    const int oH_log = oH_logical > 0 ? oH_logical : oH;
    const bool strip_mode = (strip_row_start >= 0);

    const int tH = (oH_log + 3) / 4, tW = (oW + 3) / 4;
    const int num_tiles = tH * tW;
    const int ICb = iC / 16;
    const int OCb = M / 16;

    // Strip tile range (or full range if not in strip mode)
    int range_tile0 = 0, range_tile1 = num_tiles;
    if (strip_mode) {
        int th_s = strip_row_start / 4;
        int th_e = std::min((strip_row_end + 3) / 4, tH);
        range_tile0 = th_s * tW;
        range_tile1 = th_e * tW;
    }
    const int range_tiles = range_tile1 - range_tile0;
    if (range_tiles <= 0) return;

    // Output row clipping (strip boundaries or full oH)
    const int oh_clip_lo = strip_mode ? strip_row_start : 0;
    const int oh_clip_hi = strip_mode ? std::min(strip_row_end, oH_log) : oH_log;

    // Outer group tiling: process tiles in chunks of `gs` so V+M fits in L3.
    // Must match winograd_nchwc_group_size() used by Conv_reshape to size
    // the workspace — do NOT compute gs differently here.
    const int gs         = winograd_nchwc_group_size(iC, M, range_tiles);
    const int num_groups = (range_tiles + gs - 1) / gs;

    // V and M are sized for ONE group, reused across groups.
    float* V     = reinterpret_cast<float*>(workspace);           // [36][ICb][gs][16]
    float* M_buf = V + (size_t)36 * ICb * gs * 16;                 // [36][OCb][gs][16]

    const size_t in_batch_elems  = (size_t)ICb * iH * iW * 16;
    const size_t out_batch_elems = (size_t)OCb * oH * oW * 16;

    // Cap decisions per stage. See the winograd_nchwc_*_threads()
    // comments for the policy. Stages 1/3 gate on their own task counts
    // (ICb*gs, OCb*gs); stage 2 on U size.
    const int nth_mm = winograd_nchwc_matmul_threads(iC, M);

    for (int n = 0; n < N; ++n) {
        const float* x_n = input  + (size_t)n * in_batch_elems;
        float*       y_n = output + (size_t)n * out_batch_elems;

        for (int g = 0; g < num_groups; ++g) {
            const int tile_start = range_tile0 + g * gs;
            const int tile_end   = std::min(tile_start + gs, range_tile1);
            const int gs_eff     = tile_end - tile_start;

            const int nth_s1 = winograd_nchwc_transform_threads(ICb * gs_eff);
            const int nth_s3 = winograd_nchwc_transform_threads(OCb * gs_eff);

            // Stage 1: Input transform — NCHWc-native, unit-stride stores.
            // Writes V[36][ICb][gs_eff][16] for tiles [tile_start, tile_end).
            winograd_nchwc_input_transform_group(V, x_n,
                iH, iW, pH, pW, ICb, gs_eff, tile_start, tW, nth_s1);

            // Stage 2: 36 position matmuls, 14×16 register-tiled.
            // Reads V, writes M[36][OCb][gs_eff][16].
            winograd_nchwc_matmul_group(M_buf, U, V, OCb, iC, gs_eff, nth_mm);

            // Stage 3: Output transform — NCHWc-native, unit-stride loads.
            // Reads M, writes the tile_start..tile_end slice of y_n.
            winograd_nchwc_output_transform_group(y_n, M_buf, bias,
                oH, oW, OCb, gs_eff, tile_start, tW, nth_s3,
                oh_clip_lo, oh_clip_hi);
        }
    }
}

} // namespace nnr

#endif // NNR_ARCH_X64
