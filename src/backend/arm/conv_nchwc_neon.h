#pragma once
// ARM NEON NCHWc (NCHW8c) convolution kernels.
//
// NCHWc layout on ARM: [N, C/8, H, W, 8] — 8 channels contiguous per block.
// 8 = 2 NEON qregs = pair of 128-bit vectors. This matches the existing
// MR=8 gemm_ukernel_neon.h micro-kernel's OC tiling and amortizes one input
// broadcast across 8 OC lanes (vs 4 for NCHW4c).
//
// M1 of the ARM NCHWc plan ships the 1×1 pointwise kernel only. General
// K×K and depthwise come in M2/M3 — see plan file
//
// Weight layout (pack_weight_nchwc_1x1 in kernel/conv_nchwc.h with block=8):
//   [OC/8, IC, 8]  — each OC block is 8 output lanes × IC input channels.

#if defined(__aarch64__) || defined(_M_ARM64)

#ifdef _MSC_VER
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif

#include "nnr.h"
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
// NEON pointwise (1x1) NCHWc convolution — WT spatial × 8 OC register tile.
//
// Per-tile body: WT spatial accumulators, each a 2×float32x4_t pair holding
// 8 OC lanes. Templated on WT so the compiler unrolls the accumulator loops
// and keeps acc[] in Q-registers (critical: runtime wt spills to memory —
// see kb/feedback_runtime_tile_spill.md).
//
// Register budget at WT=8: 16 accumulator regs + 2 weight regs = 18
// (ARM A64 has 32 V-regs; ~14 free for bias/scratch).
// ---------------------------------------------------------------------------
template <int WT>
// @nnr-meta isa=NEON dtype=fp32 layout=BLOCKED_8 special=NCHWc tiling=NR
static NNR_FORCEINLINE void conv1x1_nchwc_tile_neon(
    float* __restrict out_row,
    const float* __restrict in_nh_base,   // input at (n, ib=0, h, 0)
    const float* __restrict w_ob,
    float32x4_t bv_lo, float32x4_t bv_hi,
    int IC,
    int HW,
    int w_base)
{
    float32x4_t acc_lo[WT];
    float32x4_t acc_hi[WT];
    for (int t = 0; t < WT; t++) {
        acc_lo[t] = bv_lo;
        acc_hi[t] = bv_hi;
    }

    for (int ic = 0; ic < IC; ic++) {
        const int ib = ic >> 3;        // IC block index (ic / 8)
        const int il = ic & 7;         // IC lane within block
        // Load 8 OC weights as 2 qregs (lanes 0..3, 4..7).
        const float32x4_t wv_lo = vld1q_f32(w_ob + (size_t)ic * 8);
        const float32x4_t wv_hi = vld1q_f32(w_ob + (size_t)ic * 8 + 4);
        const float* in_row = in_nh_base + ((size_t)ib * HW + w_base) * 8;

        for (int t = 0; t < WT; t++) {
            const float a = in_row[t * 8 + il];
            // vfmaq_n_f32(acc, v, s): acc += v * broadcast(s)
            acc_lo[t] = vfmaq_n_f32(acc_lo[t], wv_lo, a);
            acc_hi[t] = vfmaq_n_f32(acc_hi[t], wv_hi, a);
        }
    }

    for (int t = 0; t < WT; t++) {
        vst1q_f32(out_row + (size_t)(w_base + t) * 8,     acc_lo[t]);
        vst1q_f32(out_row + (size_t)(w_base + t) * 8 + 4, acc_hi[t]);
    }
}

// ---------------------------------------------------------------------------
// NEON NCHWc 1x1 Conv dispatcher.
//
// input:  [N, ICb, H, W, 8]  — pre-reordered by nchw_to_nchwc
// output: [N, OCb, H, W, 8]
// weight: [OCb, IC, 8]       — packed via pack_weight_nchwc_1x1(block=8)
// bias:   [OC_padded] or nullptr
// ---------------------------------------------------------------------------
// @nnr-meta isa=NEON dtype=fp32 layout=BLOCKED_8 special=NCHWc tiling=spatial
inline void conv1x1_nchwc_neon(
    float* __restrict output,
    const float* __restrict input,
    const float* __restrict weight,
    const float* __restrict bias,
    int N, int IC, int OC, int H, int W)
{
    constexpr int block = 8;
    constexpr int W_TILE = 8;

    const int HW = H * W;
    const int ICb = (IC + block - 1) / block;
    const int OCb = (OC + block - 1) / block;
    const size_t in_batch  = (size_t)ICb * HW * block;
    const size_t out_batch = (size_t)OCb * HW * block;

    const int total_work = N * OCb * H;

    nnr::for_static(0, total_work, total_work > 1, [&](int work_idx) {
        const int n   = work_idx / (OCb * H);
        const int rem = work_idx % (OCb * H);
        const int ob  = rem / H;
        const int h   = rem % H;

        const float* in_n  = input  + n * in_batch;
        float*       out_n = output + n * out_batch;

        float*       out_row = out_n + ((size_t)ob * HW + h * W) * block;
        const float* w_ob    = weight + (size_t)ob * IC * block;
        const float* in_nh   = in_n + (size_t)h * W * block;

        const float* bias_ob = bias ? (bias + ob * block) : nullptr;
        float32x4_t bv_lo = bias_ob ? vld1q_f32(bias_ob)     : vdupq_n_f32(0);
        float32x4_t bv_hi = bias_ob ? vld1q_f32(bias_ob + 4) : vdupq_n_f32(0);

        int w_base = 0;
        for (; w_base + W_TILE <= W; w_base += W_TILE) {
            conv1x1_nchwc_tile_neon<W_TILE>(out_row, in_nh, w_ob, bv_lo, bv_hi, IC, HW, w_base);
        }

        const int wt = W - w_base;
        switch (wt) {
            case 0: break;
            case 1: conv1x1_nchwc_tile_neon<1>(out_row, in_nh, w_ob, bv_lo, bv_hi, IC, HW, w_base); break;
            case 2: conv1x1_nchwc_tile_neon<2>(out_row, in_nh, w_ob, bv_lo, bv_hi, IC, HW, w_base); break;
            case 3: conv1x1_nchwc_tile_neon<3>(out_row, in_nh, w_ob, bv_lo, bv_hi, IC, HW, w_base); break;
            case 4: conv1x1_nchwc_tile_neon<4>(out_row, in_nh, w_ob, bv_lo, bv_hi, IC, HW, w_base); break;
            case 5: conv1x1_nchwc_tile_neon<5>(out_row, in_nh, w_ob, bv_lo, bv_hi, IC, HW, w_base); break;
            case 6: conv1x1_nchwc_tile_neon<6>(out_row, in_nh, w_ob, bv_lo, bv_hi, IC, HW, w_base); break;
            case 7: conv1x1_nchwc_tile_neon<7>(out_row, in_nh, w_ob, bv_lo, bv_hi, IC, HW, w_base); break;
        }
    });
}

} // namespace nnr

#endif // __aarch64__ || _M_ARM64
