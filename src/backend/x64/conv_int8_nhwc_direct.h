#pragma once
// Memcpy-free NHWC-direct int8 Conv kernel (NR=16, MR=6).
//
// Wraps `jit_nhwc_direct_gemm_nr16_avx512_t` for production dispatch from
// QLinearConv. Operates directly on raw (un-padded) NHWC input: OOB kernel
// pixels point at a small stack-local `x_zp_row[IC]` buffer, so no per-
// inference pre-pad copy is needed.
//
// The JIT GEMM reads input pixels via a per-M-block kp-major indirection
// table — no pack_a memcpy. The in-JIT requantize epilogue writes uint8
// directly via rq->y_rows[]. v1 constraints:
//   - kH==3, kW==3, sH==sW==1, dH==dW==1, pH==pW==1   (SSD-12 envelope)
//   - OC % 16 == 0
//   - w_zp == 0  (row_corr is zero — symmetric int8 weights)
// These are enforced by the dispatch predicate, not asserted here.
//
// Reuses the existing NR=48 sub-stride packed weight buffer unchanged
// (each NR=48 sub-panel is already an NR=16 panel).

#include "cpu_features.h"

#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)

#include "jit_conv_int8_ukernel.h"
#include "thread_pool.h"
#include "gemm_int8_avx512.h"   // for conv_rq_params_t (and int8_compute_threads)

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace nnr::int8 {

// @nnr-meta isa=[AVX512,AVX512_VNNI] dtype=[int8,uint8] layout=NHWC special=[JIT,Direct] tiling=NR fusion=post_op
inline void conv_int8_nhwc_direct_nr16(
    int OC, int oH, int oW, int IC, int kH, int kW,
    int sH, int sW, int dH, int dW,
    const uint8_t* __restrict x_nhwc,       // raw NHWC, iH × iW × IC (un-padded)
    int iH, int iW, int pH, int pW,
    int x_zp,
    const int8_t* __restrict packed_W,      // sub-stride NR=48 layout (reused)
    const int32_t* __restrict w_col_sums,   // [OC]
    const conv_rq_params_t* rq)
{
    (void)dH; (void)dW;  // v1 predicate guarantees dH=dW=1

    constexpr int MR    = 6;
    constexpr int NR16  = 16;

    const int spatial   = oH * oW;
    const int K         = IC * kH * kW;
    const int K4        = (K + 3) & ~3;
    const int Kgroups   = K4 / 4;
    const int IC4       = IC / 4;
    const int kHW       = kH * kW;
    const size_t sub_stride = (size_t)Kgroups * 64;
    const int Nblocks16 = OC / NR16;        // OC%16==0 by predicate

    // JIT cache (one kernel per MR_actual 1..6).
    static jit_cache_t<int, jit_nhwc_direct_gemm_nr16_avx512_t> kcache;
    jit_nhwc_direct_gemm_nr16_fn_t fn[MR + 1] = {};
    for (int mr = 1; mr <= MR; mr++) {
        fn[mr] = kcache.get_or_create(mr, mr)
                     ->fn<jit_nhwc_direct_gemm_nr16_fn_t>();
    }

    // Zero-pad indirection: OOB kernel pixels read from this IC-byte buffer
    // of x_zp, in-bounds pixels read directly from x_nhwc. Kernel reads
    // exactly IC bytes (IC4 × 4) per pointer via vpbroadcastd dword loads
    // at offsets 0, 4, ..., IC-4.
    constexpr int IC_MAX = 2048;  // v1 shapes have IC ≤ 512; conservative cap
    alignas(64) uint8_t x_zp_row[IC_MAX];
    const int ic_fill = IC;
    for (int i = 0; i < ic_fill; i++) x_zp_row[i] = (uint8_t)x_zp;

    const int nblocks_m = (spatial + MR - 1) / MR;
    const int64_t total_ops = (int64_t)spatial * OC * K;
    int nt = (nblocks_m > 1)
           ? nnr::int8_compute_threads(nblocks_m, total_ops) : 1;

    // jit_rq_params_t scalar fields are constant across M-blocks; pointer
    // fields (output_scales/bias_int32/w_col_sums) advance per N-block;
    // row_corr/y_rows fill per M-block. Mirrors the canonical translation
    // in conv_int8_packed_nr48 (gemm_int8_avx512.h:1431-1448).
    const int32_t x_zp_neg_v = -(int32_t)x_zp;
    const bool x_zp_nonzero  = (x_zp != 0);

    uint8_t* const Y_base = rq->Y_out;
    const int y_stride    = rq->y_out_stride;

    nnr::for_static(0, nblocks_m, nt, [&](int mb) {
        const int ib = mb * MR;
        const int mr = std::min(MR, spatial - ib);

        // kp-major indirection table on the stack: [kHW × MR] pointers.
        // Pad tail rows (m >= mr) with row 0 — the JIT issues all MR
        // broadcasts unconditionally and we don't want OOB reads. OOB
        // kernel pixels (ih or iw outside [0,iH)/[0,iW)) point at
        // x_zp_row so the broadcast lane picks up x_zp bytes.
        constexpr int KHW_MAX = 49;
        const uint8_t* ptrs_kp_major[KHW_MAX * MR];

        const int oh0 = ib / oW;
        const int ow0 = ib % oW;
        // Fast path: the M-block is fully interior iff
        //   (1) it doesn't straddle a row boundary, and
        //   (2) for every (kh, kw) and every m in [0,mr), the (ih, iw) is
        //       inside [0, iH) × [0, iW).
        // For the current v1 envelope (sH=sW=1, pH=pW=1, kH=kW=3) this
        // collapses to oh0 ∈ [1, iH-2] and [ow0, ow0+mr-1] ⊂ [1, iW-2].
        // ~91% of M-blocks for Block 3 (150²) satisfy this, and we skip
        // 9×6 = 54 branches per block.
        const int oh_end = (ib + mr - 1) / oW;
        const bool same_row = (oh0 == oh_end);
        const int ih_lo    = oh0 * sH - pH;                    // kh = 0
        const int ih_hi    = oh0 * sH - pH + (kH - 1);         // kh = kH-1
        const int iw_lo    = ow0 * sW - pW;                    // m=0, kw=0
        const int iw_hi    = (ow0 + mr - 1) * sW - pW + (kW - 1); // m=mr-1, kw=kW-1
        const bool interior = same_row
                           && ih_lo >= 0 && ih_hi < iH
                           && iw_lo >= 0 && iw_hi < iW;

        if (interior) {
            // Branchless: all pointers land in x_nhwc. Tail rows (m >= mr)
            // reuse m=0's base — identical to the slow path's clamp.
            for (int kp = 0; kp < kHW; kp++) {
                int kh = kp / kW;
                int kw = kp % kW;
                const int ih = oh0 * sH - pH + kh;
                const uint8_t* row_base = x_nhwc + (size_t)ih * iW * IC;
                const int iw_base = ow0 * sW - pW + kw;
                // m = 0 … mr-1
                for (int m = 0; m < mr; m++) {
                    ptrs_kp_major[(size_t)kp * MR + m] =
                        row_base + (size_t)(iw_base + m * sW) * IC;
                }
                // m = mr … MR-1: replicate row 0
                for (int m = mr; m < MR; m++) {
                    ptrs_kp_major[(size_t)kp * MR + m] =
                        row_base + (size_t)iw_base * IC;
                }
            }
        } else {
            // Slow path with per-(kp, m) OOB check.
            for (int kp = 0; kp < kHW; kp++) {
                int kh = kp / kW;
                int kw = kp % kW;
                for (int m = 0; m < MR; m++) {
                    int sp = ib + (m < mr ? m : 0);
                    int oh = sp / oW, ow = sp % oW;
                    int ih = oh * sH - pH + kh;
                    int iw = ow * sW - pW + kw;
                    const uint8_t* p;
                    if ((unsigned)ih < (unsigned)iH && (unsigned)iw < (unsigned)iW) {
                        p = x_nhwc + ((size_t)ih * iW + iw) * IC;
                    } else {
                        p = x_zp_row;
                    }
                    ptrs_kp_major[(size_t)kp * MR + m] = p;
                }
            }
        }

        // Per-M-block constant rq fields (row_corr is zero for w_zp==0).
        jit_rq_params_t rq_local{};
        rq_local.x_zp_neg = x_zp_neg_v;
        rq_local.y_zp_int = rq->y_zp_int;
        rq_local.rq_qmin  = rq->rq_qmin;
        rq_local.rq_qmax  = rq->rq_qmax;
        for (int r = 0; r < MR; r++) rq_local.row_corr[r] = 0;

        // C is unused when rq != nullptr but the JIT signature requires
        // a valid pointer and ldc; provide a small stack scratch.
        alignas(64) int32_t panel_unused[MR * NR16];

        for (int nb = 0; nb < Nblocks16; nb++) {
            const uint8_t* B = (const uint8_t*)packed_W + (size_t)nb * sub_stride;
            const int n_start = nb * NR16;

            // Advance the per-N-block scalar arrays to this 16-channel slice.
            // Mirrors gemm_int8_avx512.h:1435-1437.
            rq_local.output_scales = rq->output_scales + n_start;
            rq_local.bias_int32    = rq->bias_int32 ? rq->bias_int32 + n_start : nullptr;
            rq_local.w_col_sums    = x_zp_nonzero   ? w_col_sums   + n_start : nullptr;

            // y_rows for this (M-block, N-block).
            for (int r = 0; r < MR; r++) {
                int sp = ib + (r < mr ? r : 0);
                int oh = sp / oW, ow = sp % oW;
                rq_local.y_rows[r] = Y_base
                    + ((size_t)oh * oW + ow) * (size_t)y_stride
                    + n_start;
            }

            fn[mr](ptrs_kp_major, kHW, IC4, B, panel_unused, NR16, &rq_local);
        }
    });
}

} // namespace nnr::int8

#endif // NNR_ARCH_X64 && NNR_USE_XBYAK
