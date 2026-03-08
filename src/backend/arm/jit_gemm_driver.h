#pragma once
// JIT GEMM driver: cache-blocked tiling matching the intrinsics path structure.
//
// Uses the same IBLK=48/JBLK=64/KC=256 blocking as the intrinsics GEMM,
// with a per-thread B scratch buffer for L1 locality.
//
// Key difference from intrinsics: LHS (A) is NOT packed. The JIT 6×16
// micro-kernel reads A directly from row-major memory.
//
// Loop order per tile (M-block × N-block):
//   for each K-block (KC=256):
//     copy B sub-panel [KC, JBLK] into per-thread L1 scratch
//     for each MR=6 strip within M-block:
//       for each NR=16 tile within N-block:
//         call JIT micro-kernel(A[i, k], scratch[k, j], C[i, j])

#ifdef NNR_USE_XBYAK_AARCH64

#include "jit_gemm_neon.h"
#include "thread_pool.h"
#include <algorithm>
#include <cstring>
#include <cfloat>
#include <arm_neon.h>

namespace nnr {
namespace neon_jit {

static constexpr int JIT_KC = 256;
static constexpr int JIT_IBLK = 48;   // M-block: 8 × MR=6
static constexpr int JIT_JBLK = 64;   // N-block: 4 × NR=16

// ============================================================================
// RHS packing: B [K, N] row-major → [N/NR, K, NR] panels
// ============================================================================

// @nnr-meta isa=scalar dtype=fp32 layout=NCHW special=[GEMM,JIT]
inline size_t pack_b_jit_size(int K, int N) {
    int n_tiles = (N + JIT_NR - 1) / JIT_NR;
    return (size_t)n_tiles * K * JIT_NR;
}

// @nnr-meta isa=scalar dtype=fp32 layout=NCHW special=[GEMM,JIT] tiling=K
inline void pack_b_jit(float* __restrict dst, const float* __restrict B,
                        int K, int N)
{
    const int n_full = N / JIT_NR;
    const int n_rem = N % JIT_NR;

    for (int jt = 0; jt < n_full; jt++) {
        float* panel = dst + (size_t)jt * K * JIT_NR;
        const float* src = B + jt * JIT_NR;
        for (int k = 0; k < K; k++)
            memcpy(panel + k * JIT_NR, src + (size_t)k * N, JIT_NR * sizeof(float));
    }

    if (n_rem > 0) {
        float* panel = dst + (size_t)n_full * K * JIT_NR;
        memset(panel, 0, (size_t)K * JIT_NR * sizeof(float));
        const float* src = B + n_full * JIT_NR;
        for (int k = 0; k < K; k++)
            memcpy(panel + k * JIT_NR, src + (size_t)k * N, n_rem * sizeof(float));
    }
}

// ============================================================================
// GEMM driver: C[M,N] = A[M,K] * B_packed[N/16, K, 16]
//
// A is unpacked row-major with stride lda.
// B_packed has layout [N_tile, K, NR=16].
// C is row-major with stride ldc.
// ============================================================================

// @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=[GEMM,JIT] tiling=[K,MR,NR]
inline void dgemm_jit(int M, int N, int K,
                       const float* __restrict A, int lda,
                       const float* __restrict B_packed,
                       float* __restrict C, int ldc)
{
    auto uk_beta0 = get_jit_gemm_6x16_beta0();
    auto uk_beta1 = get_jit_gemm_6x16_beta1();

    const int ni = (M + JIT_IBLK - 1) / JIT_IBLK;
    const int nj = (N + JIT_JBLK - 1) / JIT_JBLK;
    const int nk = (K + JIT_KC - 1) / JIT_KC;
    const int ntiles = ni * nj;
    const bool par = ntiles > 1 && (int64_t)M * N * K > (1 << 21);

    // Per-thread scratch: B sub-panel [KC, JBLK] repacked for L1 locality
    NNR_POOL_ENSURE_SCRATCH((size_t)JIT_KC * JIT_JBLK * sizeof(float));

    nnr::for_dynamic(0, ntiles, par, [&](int tid, int tile) {
        float* pb_scratch = (float*)NNR_POOL_SCRATCH(tid);
        int it = tile / nj;
        int jt = tile % nj;
        int i0 = it * JIT_IBLK;
        int j0 = jt * JIT_JBLK;
        int ie = std::min(i0 + JIT_IBLK, M);
        int je = std::min(j0 + JIT_JBLK, N);
        int iw = ie - i0;
        int jw = je - j0;

        // N-tiles within this N-block
        int nt_start = j0 / JIT_NR;
        int nt_end = (je + JIT_NR - 1) / JIT_NR;

        for (int kb = 0; kb < nk; kb++) {
            int k0 = kb * JIT_KC;
            int kc = std::min(JIT_KC, K - k0);
            bool first_k = (kb == 0);

            auto uk = first_k ? uk_beta0 : uk_beta1;

            // Copy B panels for this (N-block, K-block) into scratch
            // Layout: [kc, jw] packed contiguously
            // But the JIT kernel expects [kc, NR=16] per N-tile.
            // So we copy each N-tile's KC slice into scratch sequentially.
            float* pb_dst = pb_scratch;
            for (int nt = nt_start; nt < nt_end; nt++) {
                const float* bp_src = B_packed + (size_t)nt * K * JIT_NR + (size_t)k0 * JIT_NR;
                memcpy(pb_dst, bp_src, (size_t)kc * JIT_NR * sizeof(float));
                pb_dst += kc * JIT_NR;
            }

            // Only call JIT kernel for full NR=16 tiles (avoids 16-wide write
            // past valid N columns which would corrupt adjacent rows)
            int nt_jit_end = nt_end;
            if (nt_end > 0 && (nt_end - 1) * JIT_NR + JIT_NR > N)
                nt_jit_end = nt_end - 1;  // last tile is partial, skip JIT

            int i = 0;
            for (; i + JIT_MR <= iw; i += JIT_MR) {
                const float* ap = A + (size_t)(i0 + i) * lda + k0;
                float* cp = C + (size_t)(i0 + i) * ldc;

                // Full N-tiles: JIT micro-kernel
                float* pb_ptr = pb_scratch;
                for (int nt = nt_start; nt < nt_jit_end; nt++) {
                    int jj = nt * JIT_NR;
                    uk(ap, pb_ptr, cp + jj, kc, lda, ldc);
                    pb_ptr += kc * JIT_NR;
                }

                // Partial N-tile: JIT into temp buffer, copy valid cols
                for (int nt = nt_jit_end; nt < nt_end; nt++) {
                    int jj = nt * JIT_NR;
                    int jjw = std::min(JIT_NR, N - jj);

                    // Use a temp 6×16 buffer for the JIT kernel output
                    float tmp_c[JIT_MR * JIT_NR];
                    if (!first_k) {
                        // Load existing partial C into temp
                        for (int r = 0; r < JIT_MR; r++)
                            memcpy(tmp_c + r * JIT_NR, cp + r * ldc + jj,
                                   jjw * sizeof(float));
                    }
                    // JIT kernel writes to tmp_c with ldc=NR=16
                    uk(ap, pb_ptr, tmp_c, kc, lda, JIT_NR);
                    // Copy valid columns back
                    for (int r = 0; r < JIT_MR; r++)
                        memcpy(cp + r * ldc + jj, tmp_c + r * JIT_NR,
                               jjw * sizeof(float));
                    pb_ptr += kc * JIT_NR;
                }
            }

            // M remainder: NEON 1-row fallback
            for (; i < iw; i++) {
                const float* a_row = A + (size_t)(i0 + i) * lda + k0;
                float* c_row = C + (size_t)(i0 + i) * ldc;

                float* pb_ptr = pb_scratch;
                for (int nt = nt_start; nt < nt_end; nt++) {
                    int jj = nt * JIT_NR;
                    int jjw = std::min(JIT_NR, N - jj);

                    // NEON 1×16 GEMV
                    if (jjw == JIT_NR) {
                        float32x4_t acc0, acc1, acc2, acc3;
                        if (first_k) {
                            acc0 = acc1 = acc2 = acc3 = vdupq_n_f32(0);
                        } else {
                            acc0 = vld1q_f32(c_row + jj);
                            acc1 = vld1q_f32(c_row + jj + 4);
                            acc2 = vld1q_f32(c_row + jj + 8);
                            acc3 = vld1q_f32(c_row + jj + 12);
                        }
                        for (int k = 0; k < kc; k++) {
                            float32x4_t va = vdupq_n_f32(a_row[k]);
                            acc0 = vfmaq_f32(acc0, va, vld1q_f32(pb_ptr + k * JIT_NR));
                            acc1 = vfmaq_f32(acc1, va, vld1q_f32(pb_ptr + k * JIT_NR + 4));
                            acc2 = vfmaq_f32(acc2, va, vld1q_f32(pb_ptr + k * JIT_NR + 8));
                            acc3 = vfmaq_f32(acc3, va, vld1q_f32(pb_ptr + k * JIT_NR + 12));
                        }
                        vst1q_f32(c_row + jj, acc0);
                        vst1q_f32(c_row + jj + 4, acc1);
                        vst1q_f32(c_row + jj + 8, acc2);
                        vst1q_f32(c_row + jj + 12, acc3);
                    } else {
                        // Partial N-tile: scalar
                        for (int j = 0; j < jjw; j++) {
                            float acc = first_k ? 0.0f : c_row[jj + j];
                            for (int k = 0; k < kc; k++)
                                acc += a_row[k] * pb_ptr[k * JIT_NR + j];
                            c_row[jj + j] = acc;
                        }
                    }
                    pb_ptr += kc * JIT_NR;
                }
            }
        }
    });
}

} // namespace neon_jit
} // namespace nnr

#endif // NNR_USE_XBYAK_AARCH64
