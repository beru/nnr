#pragma once
// ARM NEON int8 GEMM (MVP). C[M×N] = (A - a_zp) × (B - b_zp), accumulated to int32.
//
// Uses SDOT (ARMv8.4-A dotprod): 4×(int8×int8)→int32 per lane, 16 ops per instruction.
//
// A: uint8 [M×K] row-major. Shifted to int8 on-the-fly via XOR 0x80 (i.e. A_u - 128).
// B: int8  [K×N] row-major, must be pre-packed via pack_b_int8_neon_and_col_sums().
// C: int32 [M×N16] row-major, with N16 = (N+15)&~15 padding.
//
// Zero-point compensation (derivation):
//   Let Au = A_uint8, Bs = B_int8. Define A_s = Au XOR 0x80 = Au - 128.
//   SDOT over K: sdot = Σ_k (A_s * Bs) = Σ_k (Au - 128) * Bs = Σ_k(Au·Bs) - 128·col_sum[j]
//   → Σ_k(Au·Bs) = sdot + 128·col_sum[j]
//   Target: C[i,j] = Σ_k ((Au - a_zp)(Bs - b_zp))
//                  = Σ_k(Au·Bs) - a_zp·col_sum[j] - b_zp·row_sum[i] + K·a_zp·b_zp
//                  = sdot + (128 - a_zp)·col_sum[j] - b_zp·row_sum[i] + K·a_zp·b_zp
//
// Packed B layout: dst[(k/4)·4·N16 + j·4 + (k%4)] = B[k, j].
//   Each 16-byte SIMD load covers 4 cols × 4 K-values (one SDOT tile slice).
//   K padded to K4 = (K+3)&~3, N padded to N16 = (N+15)&~15 with zeros.
//
// Counterpart of x64/gemm_int8_avx512.h. MR=4×NR=16 main tile + MR=1 M-tail.
// i8mm SMMLA path is a further follow-up.

#if defined(__aarch64__) || defined(_M_ARM64)

#ifdef _MSC_VER
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif
#include <cstdint>
#include <cstring>

#ifdef NNR_USE_XBYAK_AARCH64
// Must be included at file scope — xbyak_aarch64 pulls in <deque> etc. which
// cannot live inside the nnr::int8::neon namespace.
#include "jit_gemm_int8_neon.h"
#endif

namespace nnr::int8::neon {

inline size_t pack_b_int8_neon_size(int K, int N)
{
    int K4 = (K + 3) & ~3;
    int N16 = (N + 15) & ~15;
    return (size_t)K4 * N16;
}

// Pack B[K×N] (row-major int8) into SDOT-friendly groups of 4 K-rows × 16-col tiles,
// and fuse column-sum computation in one pass.
// Layout: packed[(k/4)·4·N16 + j·4 + (k%4)] = B[k, j].  Zero-padded for K%4, N%16.
inline void pack_b_int8_neon_and_col_sums(int8_t* __restrict dst, int32_t* __restrict col_sums,
    const int8_t* __restrict B, int K, int N)
{
    int K4 = (K + 3) & ~3;
    int N16 = (N + 15) & ~15;
    std::memset(dst, 0, (size_t)K4 * N16);
    std::memset(col_sums, 0, (size_t)N16 * sizeof(int32_t));

    int k = 0;
    for (; k + 4 <= K; k += 4) {
        const int8_t* r0 = B + (size_t)(k + 0) * N;
        const int8_t* r1 = B + (size_t)(k + 1) * N;
        const int8_t* r2 = B + (size_t)(k + 2) * N;
        const int8_t* r3 = B + (size_t)(k + 3) * N;
        int8_t* out = dst + (size_t)(k / 4) * 4 * N16;
        for (int j = 0; j < N; j++) {
            int8_t v0 = r0[j], v1 = r1[j], v2 = r2[j], v3 = r3[j];
            out[j * 4 + 0] = v0;
            out[j * 4 + 1] = v1;
            out[j * 4 + 2] = v2;
            out[j * 4 + 3] = v3;
            col_sums[j] += (int32_t)v0 + (int32_t)v1 + (int32_t)v2 + (int32_t)v3;
        }
    }
    // K tail (< 4 rows): fill remaining slots in the final K-group, padding stays zero.
    for (; k < K; k++) {
        int kk = k % 4;
        int8_t* out = dst + (size_t)(k / 4) * 4 * N16;
        const int8_t* row = B + (size_t)k * N;
        for (int j = 0; j < N; j++) {
            out[j * 4 + kk] = row[j];
            col_sums[j] += (int32_t)row[j];
        }
    }
}

// Row sums of A (uint8). row_sums[i] = Σ_k A[i, k].
inline void compute_row_sums_neon(int32_t* __restrict row_sums, const uint8_t* __restrict A,
    int M, int K)
{
    for (int i = 0; i < M; i++) {
        const uint8_t* row = A + (size_t)i * K;
        uint32x4_t acc = vdupq_n_u32(0);
        int k = 0;
        for (; k + 16 <= K; k += 16) {
            uint8x16_t v = vld1q_u8(row + k);
            acc = vpadalq_u16(acc, vpaddlq_u8(v));
        }
        int32_t s = (int32_t)vaddvq_u32(acc);
        for (; k < K; k++) s += (int32_t)row[k];
        row_sums[i] = s;
    }
}

// Main GEMM. Expects packed_B from pack_b_int8_neon_and_col_sums, row_sums from
// compute_row_sums_neon. Output ldc is N16 = (N+15)&~15.
// Tile: MR=4 × NR=16 on the main body, falls through to MR=1 for the M-tail.
// K processed in groups of 4 via SDOT. Register budget for MR=4:
//   16 int32x4 accumulators + 4 A splat vectors + 4 B vectors = 24 vregs (NEON has 32).
inline void gemm_int8_neon(int M, int N, int K,
    const uint8_t* __restrict A, int a_zp,
    const int8_t* __restrict packed_B, int b_zp,
    const int32_t* __restrict col_sums, const int32_t* __restrict row_sums,
    int32_t* __restrict C)
{
    int N16 = (N + 15) & ~15;
    int32_t a_off_k = 128 - a_zp;                      // (128 - a_zp) * col_sums[j]
    int32_t kakzbzp = (int32_t)K * a_zp * b_zp;        // + K·a_zp·b_zp constant
    int32x4_t a_off_v = vdupq_n_s32(a_off_k);

    int kk_full = K & ~3;  // largest multiple of 4 ≤ K

    // Helper: splat 4 bytes of A starting at `ptr`, XOR-shifted to int8 representation.
    // Returns the 16-byte vector that SDOT consumes (4 K-values broadcast 4×).
    auto load_a_splat = [](const uint8_t* ptr) -> int8x16_t {
        uint32_t a4u;
        std::memcpy(&a4u, ptr, 4);
        int32_t a4s = (int32_t)(a4u ^ 0x80808080u);
        return vreinterpretq_s8_s32(vdupq_n_s32(a4s));
    };
    // Same but pads past K with zeros (for the K-tail step).
    auto load_a_splat_tail = [](const uint8_t* ptr, int n_valid) -> int8x16_t {
        uint8_t a_pad[4] = {0, 0, 0, 0};
        for (int kx = 0; kx < n_valid; kx++) a_pad[kx] = ptr[kx];
        uint32_t a4u;
        std::memcpy(&a4u, a_pad, 4);
        int32_t a4s = (int32_t)(a4u ^ 0x80808080u);
        return vreinterpretq_s8_s32(vdupq_n_s32(a4s));
    };

    int i = 0;

    // Per-i-block A pre-pack buffer for the MR=16 body.
    // Layout: a_pack16[kg*64 + r4*16 + r*4 + k] = (A[i+r4*4+r, kg*4+k]) XOR 0x80
    //   i.e. 16 rows grouped as 4 groups of 4 rows; each 16-byte chunk
    //   holds [row_a k0..k3 | row_b k0..k3 | row_c k0..k3 | row_d k0..k3].
    // A single vld1q_s8 then feeds four `vdotq_laneq_s32` consumers via the
    // 4-byte lane selector — one B load per K=4 step amortises across 16
    // SDOTs, matching KleidiAI's MR=16×NR=4 tile shape.
    // Cap at K ≤ 4096 (64 KB stack per function call).
    constexpr int kMaxK_MR16 = 4096;
    alignas(16) int8_t a_pack16[16 * kMaxK_MR16];
    const int k4_groups = (K + 3) / 4;
    const int K4_padded = k4_groups * 4;
    const bool mr16_ok = (K <= kMaxK_MR16);

    // MR=16 body: process 16 rows per outer iteration. MR=4 below handles
    // M%16 in [4,15]; MR=1 handles the 0-3 trailing rows.
    if (mr16_ok) for (; i + 16 <= M; i += 16) {
        // Pack 16 rows into a_pack16 with XOR-0x80 fused in.
        // Tail K-bytes beyond K are zero-padded (contribute 0 to SDOT).
        std::memset(a_pack16, 0, (size_t)k4_groups * 64);
        const uint8x8_t v_xor = vdup_n_u8(0x80);
        for (int r4 = 0; r4 < 4; r4++) {
            for (int rr = 0; rr < 4; rr++) {
                const uint8_t* arow = A + (size_t)(i + r4 * 4 + rr) * K;
                int kg = 0;
                for (; kg + 1 <= K / 4; kg++) {
                    // 4 bytes of A → XOR'd 4 bytes into a_pack16
                    uint32_t av;
                    std::memcpy(&av, arow + kg * 4, 4);
                    av ^= 0x80808080u;
                    std::memcpy(a_pack16 + kg * 64 + r4 * 16 + rr * 4, &av, 4);
                }
                // K-tail (< 4 remaining): XOR valid bytes, leave the rest zero
                int n_tail = K - kg * 4;
                if (n_tail > 0) {
                    uint8_t tmp[4] = {0x80, 0x80, 0x80, 0x80};
                    for (int kx = 0; kx < n_tail; kx++) tmp[kx] = arow[kg * 4 + kx];
                    for (int kx = 0; kx < 4; kx++) tmp[kx] ^= 0x80;
                    // Only the first n_tail bytes are "real"; the padding bytes
                    // must contribute 0 to SDOT. After XOR-0x80, 0 contributes 0 ✓.
                    for (int kx = n_tail; kx < 4; kx++) tmp[kx] = 0;
                    std::memcpy(a_pack16 + kg * 64 + r4 * 16 + rr * 4, tmp, 4);
                }
            }
            (void)v_xor; // silence unused-warning on the scalar path above
        }

        // Row biases: -b_zp * row_sums[i+r] + kakzbzp.
        int32_t rb[16];
        for (int r = 0; r < 16; r++)
            rb[r] = -b_zp * row_sums[i + r] + kakzbzp;

        for (int j = 0; j < N; j += 4) {
            // 16 named accumulators — avoid an int32x4_t[16] array because
            // MSVC's register allocator spills it to stack when the epilogue
            // indexes it with a variable (measured 40% regression vs GCC).
            int32x4_t a00 = vdupq_n_s32(0), a01 = vdupq_n_s32(0);
            int32x4_t a02 = vdupq_n_s32(0), a03 = vdupq_n_s32(0);
            int32x4_t a04 = vdupq_n_s32(0), a05 = vdupq_n_s32(0);
            int32x4_t a06 = vdupq_n_s32(0), a07 = vdupq_n_s32(0);
            int32x4_t a08 = vdupq_n_s32(0), a09 = vdupq_n_s32(0);
            int32x4_t a10 = vdupq_n_s32(0), a11 = vdupq_n_s32(0);
            int32x4_t a12 = vdupq_n_s32(0), a13 = vdupq_n_s32(0);
            int32x4_t a14 = vdupq_n_s32(0), a15 = vdupq_n_s32(0);

            for (int kg = 0; kg < k4_groups; kg++) {
                int8x16_t av0 = vld1q_s8(a_pack16 + kg * 64 +  0);  // rows 0-3
                int8x16_t av1 = vld1q_s8(a_pack16 + kg * 64 + 16);  // rows 4-7
                int8x16_t av2 = vld1q_s8(a_pack16 + kg * 64 + 32);  // rows 8-11
                int8x16_t av3 = vld1q_s8(a_pack16 + kg * 64 + 48);  // rows 12-15

                const int8_t* bp = packed_B + (size_t)kg * 4 * N16 + (size_t)j * 4;
                int8x16_t b = vld1q_s8(bp);                          // 4 cols × 4 K

                a00 = vdotq_laneq_s32(a00, b, av0, 0);
                a01 = vdotq_laneq_s32(a01, b, av0, 1);
                a02 = vdotq_laneq_s32(a02, b, av0, 2);
                a03 = vdotq_laneq_s32(a03, b, av0, 3);
                a04 = vdotq_laneq_s32(a04, b, av1, 0);
                a05 = vdotq_laneq_s32(a05, b, av1, 1);
                a06 = vdotq_laneq_s32(a06, b, av1, 2);
                a07 = vdotq_laneq_s32(a07, b, av1, 3);
                a08 = vdotq_laneq_s32(a08, b, av2, 0);
                a09 = vdotq_laneq_s32(a09, b, av2, 1);
                a10 = vdotq_laneq_s32(a10, b, av2, 2);
                a11 = vdotq_laneq_s32(a11, b, av2, 3);
                a12 = vdotq_laneq_s32(a12, b, av3, 0);
                a13 = vdotq_laneq_s32(a13, b, av3, 1);
                a14 = vdotq_laneq_s32(a14, b, av3, 2);
                a15 = vdotq_laneq_s32(a15, b, av3, 3);
            }

            // Epilogue: acc[r] + rb[r] + a_off * col_sums[j..j+3] → store.
            int32x4_t cs = vld1q_s32(col_sums + j);
            #define NNR_SDOT_EPI(R, ACC) do { \
                int32x4_t bv = vdupq_n_s32(rb[R]); \
                ACC = vmlaq_s32(vaddq_s32(ACC, bv), a_off_v, cs); \
                vst1q_s32(C + (size_t)(i + (R)) * N16 + j, ACC); \
            } while (0)
            NNR_SDOT_EPI( 0, a00); NNR_SDOT_EPI( 1, a01);
            NNR_SDOT_EPI( 2, a02); NNR_SDOT_EPI( 3, a03);
            NNR_SDOT_EPI( 4, a04); NNR_SDOT_EPI( 5, a05);
            NNR_SDOT_EPI( 6, a06); NNR_SDOT_EPI( 7, a07);
            NNR_SDOT_EPI( 8, a08); NNR_SDOT_EPI( 9, a09);
            NNR_SDOT_EPI(10, a10); NNR_SDOT_EPI(11, a11);
            NNR_SDOT_EPI(12, a12); NNR_SDOT_EPI(13, a13);
            NNR_SDOT_EPI(14, a14); NNR_SDOT_EPI(15, a15);
            #undef NNR_SDOT_EPI
        }
        (void)K4_padded;
    }

    // MR=4 body: process 4 rows per outer iteration.
    for (; i + 4 <= M; i += 4) {
        const uint8_t* a_row0 = A + (size_t)(i + 0) * K;
        const uint8_t* a_row1 = A + (size_t)(i + 1) * K;
        const uint8_t* a_row2 = A + (size_t)(i + 2) * K;
        const uint8_t* a_row3 = A + (size_t)(i + 3) * K;
        int32x4_t bias_v0 = vdupq_n_s32(-b_zp * row_sums[i + 0] + kakzbzp);
        int32x4_t bias_v1 = vdupq_n_s32(-b_zp * row_sums[i + 1] + kakzbzp);
        int32x4_t bias_v2 = vdupq_n_s32(-b_zp * row_sums[i + 2] + kakzbzp);
        int32x4_t bias_v3 = vdupq_n_s32(-b_zp * row_sums[i + 3] + kakzbzp);

        for (int j = 0; j < N; j += 16) {
            // 16 accumulators: [row 0..3][col-group 0..3].
            int32x4_t acc00 = vdupq_n_s32(0), acc01 = vdupq_n_s32(0);
            int32x4_t acc02 = vdupq_n_s32(0), acc03 = vdupq_n_s32(0);
            int32x4_t acc10 = vdupq_n_s32(0), acc11 = vdupq_n_s32(0);
            int32x4_t acc12 = vdupq_n_s32(0), acc13 = vdupq_n_s32(0);
            int32x4_t acc20 = vdupq_n_s32(0), acc21 = vdupq_n_s32(0);
            int32x4_t acc22 = vdupq_n_s32(0), acc23 = vdupq_n_s32(0);
            int32x4_t acc30 = vdupq_n_s32(0), acc31 = vdupq_n_s32(0);
            int32x4_t acc32 = vdupq_n_s32(0), acc33 = vdupq_n_s32(0);

            // Hot K loop (full K-groups of 4).
            for (int kk = 0; kk < kk_full; kk += 4) {
                int8x16_t av0 = load_a_splat(a_row0 + kk);
                int8x16_t av1 = load_a_splat(a_row1 + kk);
                int8x16_t av2 = load_a_splat(a_row2 + kk);
                int8x16_t av3 = load_a_splat(a_row3 + kk);

                const int8_t* bp = packed_B + (size_t)(kk / 4) * 4 * N16 + (size_t)j * 4;
                int8x16_t b0 = vld1q_s8(bp + 0);
                int8x16_t b1 = vld1q_s8(bp + 16);
                int8x16_t b2 = vld1q_s8(bp + 32);
                int8x16_t b3 = vld1q_s8(bp + 48);

                acc00 = vdotq_s32(acc00, av0, b0); acc01 = vdotq_s32(acc01, av0, b1);
                acc02 = vdotq_s32(acc02, av0, b2); acc03 = vdotq_s32(acc03, av0, b3);
                acc10 = vdotq_s32(acc10, av1, b0); acc11 = vdotq_s32(acc11, av1, b1);
                acc12 = vdotq_s32(acc12, av1, b2); acc13 = vdotq_s32(acc13, av1, b3);
                acc20 = vdotq_s32(acc20, av2, b0); acc21 = vdotq_s32(acc21, av2, b1);
                acc22 = vdotq_s32(acc22, av2, b2); acc23 = vdotq_s32(acc23, av2, b3);
                acc30 = vdotq_s32(acc30, av3, b0); acc31 = vdotq_s32(acc31, av3, b1);
                acc32 = vdotq_s32(acc32, av3, b2); acc33 = vdotq_s32(acc33, av3, b3);
            }
            // K tail (< 4 remaining). A is zero-padded, B is already zero-padded in packing.
            if (kk_full < K) {
                int n_tail = K - kk_full;
                int8x16_t av0 = load_a_splat_tail(a_row0 + kk_full, n_tail);
                int8x16_t av1 = load_a_splat_tail(a_row1 + kk_full, n_tail);
                int8x16_t av2 = load_a_splat_tail(a_row2 + kk_full, n_tail);
                int8x16_t av3 = load_a_splat_tail(a_row3 + kk_full, n_tail);

                const int8_t* bp = packed_B + (size_t)(kk_full / 4) * 4 * N16 + (size_t)j * 4;
                int8x16_t b0 = vld1q_s8(bp + 0);
                int8x16_t b1 = vld1q_s8(bp + 16);
                int8x16_t b2 = vld1q_s8(bp + 32);
                int8x16_t b3 = vld1q_s8(bp + 48);

                acc00 = vdotq_s32(acc00, av0, b0); acc01 = vdotq_s32(acc01, av0, b1);
                acc02 = vdotq_s32(acc02, av0, b2); acc03 = vdotq_s32(acc03, av0, b3);
                acc10 = vdotq_s32(acc10, av1, b0); acc11 = vdotq_s32(acc11, av1, b1);
                acc12 = vdotq_s32(acc12, av1, b2); acc13 = vdotq_s32(acc13, av1, b3);
                acc20 = vdotq_s32(acc20, av2, b0); acc21 = vdotq_s32(acc21, av2, b1);
                acc22 = vdotq_s32(acc22, av2, b2); acc23 = vdotq_s32(acc23, av2, b3);
                acc30 = vdotq_s32(acc30, av3, b0); acc31 = vdotq_s32(acc31, av3, b1);
                acc32 = vdotq_s32(acc32, av3, b2); acc33 = vdotq_s32(acc33, av3, b3);
            }

            // Shared column-sum compensation.
            int32x4_t cs0 = vld1q_s32(col_sums + j + 0);
            int32x4_t cs1 = vld1q_s32(col_sums + j + 4);
            int32x4_t cs2 = vld1q_s32(col_sums + j + 8);
            int32x4_t cs3 = vld1q_s32(col_sums + j + 12);

            acc00 = vmlaq_s32(vaddq_s32(acc00, bias_v0), a_off_v, cs0);
            acc01 = vmlaq_s32(vaddq_s32(acc01, bias_v0), a_off_v, cs1);
            acc02 = vmlaq_s32(vaddq_s32(acc02, bias_v0), a_off_v, cs2);
            acc03 = vmlaq_s32(vaddq_s32(acc03, bias_v0), a_off_v, cs3);
            acc10 = vmlaq_s32(vaddq_s32(acc10, bias_v1), a_off_v, cs0);
            acc11 = vmlaq_s32(vaddq_s32(acc11, bias_v1), a_off_v, cs1);
            acc12 = vmlaq_s32(vaddq_s32(acc12, bias_v1), a_off_v, cs2);
            acc13 = vmlaq_s32(vaddq_s32(acc13, bias_v1), a_off_v, cs3);
            acc20 = vmlaq_s32(vaddq_s32(acc20, bias_v2), a_off_v, cs0);
            acc21 = vmlaq_s32(vaddq_s32(acc21, bias_v2), a_off_v, cs1);
            acc22 = vmlaq_s32(vaddq_s32(acc22, bias_v2), a_off_v, cs2);
            acc23 = vmlaq_s32(vaddq_s32(acc23, bias_v2), a_off_v, cs3);
            acc30 = vmlaq_s32(vaddq_s32(acc30, bias_v3), a_off_v, cs0);
            acc31 = vmlaq_s32(vaddq_s32(acc31, bias_v3), a_off_v, cs1);
            acc32 = vmlaq_s32(vaddq_s32(acc32, bias_v3), a_off_v, cs2);
            acc33 = vmlaq_s32(vaddq_s32(acc33, bias_v3), a_off_v, cs3);

            int32_t* c0 = C + (size_t)(i + 0) * N16 + j;
            int32_t* c1 = C + (size_t)(i + 1) * N16 + j;
            int32_t* c2 = C + (size_t)(i + 2) * N16 + j;
            int32_t* c3 = C + (size_t)(i + 3) * N16 + j;
            vst1q_s32(c0 + 0, acc00); vst1q_s32(c0 + 4, acc01);
            vst1q_s32(c0 + 8, acc02); vst1q_s32(c0 + 12, acc03);
            vst1q_s32(c1 + 0, acc10); vst1q_s32(c1 + 4, acc11);
            vst1q_s32(c1 + 8, acc12); vst1q_s32(c1 + 12, acc13);
            vst1q_s32(c2 + 0, acc20); vst1q_s32(c2 + 4, acc21);
            vst1q_s32(c2 + 8, acc22); vst1q_s32(c2 + 12, acc23);
            vst1q_s32(c3 + 0, acc30); vst1q_s32(c3 + 4, acc31);
            vst1q_s32(c3 + 8, acc32); vst1q_s32(c3 + 12, acc33);
        }
    }

    // M-tail: remaining 0..3 rows, process with MR=1.
    for (; i < M; i++) {
        const uint8_t* a_row = A + (size_t)i * K;
        int32_t row_bias = -b_zp * row_sums[i] + kakzbzp;
        int32x4_t bias_v = vdupq_n_s32(row_bias);

        for (int j = 0; j < N; j += 16) {
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);

            for (int kk = 0; kk < kk_full; kk += 4) {
                int8x16_t a_vec = load_a_splat(a_row + kk);
                const int8_t* bp = packed_B + (size_t)(kk / 4) * 4 * N16 + (size_t)j * 4;
                int8x16_t b0 = vld1q_s8(bp + 0);
                int8x16_t b1 = vld1q_s8(bp + 16);
                int8x16_t b2 = vld1q_s8(bp + 32);
                int8x16_t b3 = vld1q_s8(bp + 48);
                acc0 = vdotq_s32(acc0, a_vec, b0);
                acc1 = vdotq_s32(acc1, a_vec, b1);
                acc2 = vdotq_s32(acc2, a_vec, b2);
                acc3 = vdotq_s32(acc3, a_vec, b3);
            }
            if (kk_full < K) {
                int8x16_t a_vec = load_a_splat_tail(a_row + kk_full, K - kk_full);
                const int8_t* bp = packed_B + (size_t)(kk_full / 4) * 4 * N16 + (size_t)j * 4;
                int8x16_t b0 = vld1q_s8(bp + 0);
                int8x16_t b1 = vld1q_s8(bp + 16);
                int8x16_t b2 = vld1q_s8(bp + 32);
                int8x16_t b3 = vld1q_s8(bp + 48);
                acc0 = vdotq_s32(acc0, a_vec, b0);
                acc1 = vdotq_s32(acc1, a_vec, b1);
                acc2 = vdotq_s32(acc2, a_vec, b2);
                acc3 = vdotq_s32(acc3, a_vec, b3);
            }

            int32x4_t cs0 = vld1q_s32(col_sums + j + 0);
            int32x4_t cs1 = vld1q_s32(col_sums + j + 4);
            int32x4_t cs2 = vld1q_s32(col_sums + j + 8);
            int32x4_t cs3 = vld1q_s32(col_sums + j + 12);
            acc0 = vmlaq_s32(vaddq_s32(acc0, bias_v), a_off_v, cs0);
            acc1 = vmlaq_s32(vaddq_s32(acc1, bias_v), a_off_v, cs1);
            acc2 = vmlaq_s32(vaddq_s32(acc2, bias_v), a_off_v, cs2);
            acc3 = vmlaq_s32(vaddq_s32(acc3, bias_v), a_off_v, cs3);

            vst1q_s32(C + (size_t)i * N16 + j + 0,  acc0);
            vst1q_s32(C + (size_t)i * N16 + j + 4,  acc1);
            vst1q_s32(C + (size_t)i * N16 + j + 8,  acc2);
            vst1q_s32(C + (size_t)i * N16 + j + 12, acc3);
        }
    }
}

// =============================================================================
// i8mm SMMLA variant — 2× throughput over SDOT on chips with ARMv8.6 +i8mm.
// Runtime-gated by the caller via `has_neon_i8mm()`. Packing layout differs
// from the SDOT path (8 K-values per slice, 2-col-pair groups), so callers
// that pick this path also use the matching `pack_b_int8_smmla_*` pair.
// =============================================================================
#if defined(__ARM_FEATURE_MATMUL_INT8) || (defined(_MSC_VER) && defined(_M_ARM64))

// Portable L1-keep prefetch. MSVC ARM64 has __prefetch; GCC/Clang use the builtin.
#if defined(_MSC_VER) && defined(_M_ARM64)
#define NNR_PREFETCH_L1(addr) __prefetch((const void*)(addr))
#elif defined(__GNUC__) || defined(__clang__)
#define NNR_PREFETCH_L1(addr) __builtin_prefetch((const void*)(addr), 0, 3)
#else
#define NNR_PREFETCH_L1(addr) ((void)0)
#endif


// Size of packed B for the SMMLA path: K padded to K8, N padded to N16.
inline size_t pack_b_int8_smmla_size(int K, int N)
{
    int K8 = (K + 7) & ~7;
    int N16 = (N + 15) & ~15;
    return (size_t)K8 * N16;
}

// Pack B[K×N] (row-major int8) into SMMLA-friendly 2-col pairs × 8-K groups,
// and compute column sums in the same pass. Zero-padded for K%8 and N%16.
//
// Layout: packed[(k/8)·8·N16 + jpair·16 + col·8 + (k%8)] = B[k, jpair*2 + col]
// where jpair ∈ [0, N16/2), col ∈ {0, 1}, k ∈ [0, K).
// Each 16-byte SIMD load covers one 2-col pair's 8 K-values (one SMMLA B operand).
inline void pack_b_int8_smmla_and_col_sums(int8_t* __restrict dst, int32_t* __restrict col_sums,
    const int8_t* __restrict B, int K, int N)
{
    int K8 = (K + 7) & ~7;
    int N16 = (N + 15) & ~15;
    std::memset(dst, 0, (size_t)K8 * N16);
    std::memset(col_sums, 0, (size_t)N16 * sizeof(int32_t));

    // Main K-groups of 8.
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        int8_t* group_base = dst + (size_t)(k / 8) * 8 * N16;
        for (int j = 0; j < N; j++) {
            int jpair = j >> 1;
            int col   = j & 1;
            int8_t* out = group_base + (size_t)jpair * 16 + (size_t)col * 8;
            int32_t s = 0;
            for (int kk = 0; kk < 8; kk++) {
                int8_t v = B[(size_t)(k + kk) * N + j];
                out[kk] = v;
                s += (int32_t)v;
            }
            col_sums[j] += s;
        }
    }
    // K tail (< 8 rows): fill remaining slots in the final K-group; padding stays zero.
    if (k < K) {
        int rem = K - k;
        int8_t* group_base = dst + (size_t)(k / 8) * 8 * N16;
        for (int j = 0; j < N; j++) {
            int jpair = j >> 1;
            int col   = j & 1;
            int8_t* out = group_base + (size_t)jpair * 16 + (size_t)col * 8;
            int32_t s = 0;
            for (int kk = 0; kk < rem; kk++) {
                int8_t v = B[(size_t)(k + kk) * N + j];
                out[kk] = v;
                s += (int32_t)v;
            }
            col_sums[j] += s;
        }
    }
}

// SMMLA-based GEMM. Same zero-point math as the SDOT path
// (acc + (128 - a_zp)·col_sums[j] - b_zp·row_sums[i] + K·a_zp·b_zp),
// but SMMLA accumulators hold 2×2 C blocks rather than 1×4, so the epilogue
// interleaves lane-halves to reconstitute row-major output strips of 4 cols.
//
// Tile: MR=4×NR=16 main body (2 row-pairs × 8 col-pairs, 16 SMMLAs per K=8),
//       MR=2 row-pair is the fallback for the M-tail when only 2 or 3 rows remain,
//       and a pure-scalar fallback for the last 0 or 1 row. Keeping the MR=2 case
//       inline rather than bouncing to the SDOT path avoids carrying two pack
//       layouts simultaneously.
inline void gemm_int8_smmla(int M, int N, int K,
    const uint8_t* __restrict A, int a_zp,
    const int8_t* __restrict packed_B, int b_zp,
    const int32_t* __restrict col_sums, const int32_t* __restrict row_sums,
    int32_t* __restrict C)
{
    int N16 = (N + 15) & ~15;
    int32_t a_off_k = 128 - a_zp;
    int32_t kakzbzp = (int32_t)K * a_zp * b_zp;
    int32x4_t a_off_v = vdupq_n_s32(a_off_k);

    int kk_full = K & ~7;  // largest multiple of 8 ≤ K
    int n_tail  = K - kk_full;

    // Load 8 A bytes from row `ptr`, XOR 0x80 (u8→s8 = u8-128) into an int8x8_t.
    auto load_a8_xor = [](const uint8_t* ptr) -> int8x8_t {
        uint8x8_t u = vld1_u8(ptr);
        return vreinterpret_s8_u8(veor_u8(u, vdup_n_u8(0x80)));
    };
    auto load_a8_xor_tail = [](const uint8_t* ptr, int n_valid) -> int8x8_t {
        uint8_t buf[8] = {0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80};
        // Pre-fill with 0x80 so XOR-0x80 gives 0 (the int8 value that contributes nothing
        // to the SMMLA sum, matching the packed-B tail being zero-padded).
        for (int kx = 0; kx < n_valid; kx++) buf[kx] = ptr[kx];
        uint8x8_t u = vld1_u8(buf);
        return vreinterpret_s8_u8(veor_u8(u, vdup_n_u8(0x80)));
    };

    int i = 0;

    // Per-i-block A pre-pack.
    //   Layout: a_pack[kg*32 + 0..7]   = (A[i+0, kg*8..kg*8+7]) XOR 0x80
    //           a_pack[kg*32 + 8..15]  = (A[i+1, ...)           XOR 0x80
    //           a_pack[kg*32 + 16..23] = (A[i+2, ...)           XOR 0x80
    //           a_pack[kg*32 + 24..31] = (A[i+3, ...)           XOR 0x80
    //   Tail k-group is zero-padded (which is the int8 value that contributes
    //   0 to the SMMLA sum, matching zero-padded B in the SMMLA pack).
    //   XOR-0x80 is fused in at pack time so the inner loop can use two raw
    //   vld1q_s8 instead of four vld1_u8 + four veor + two vcombine per K-step.
    //   Cap K at 4096 per block (16 KB stack); typical K is 64-1024.
    constexpr int kMaxK = 4096;
    alignas(16) int8_t a_pack[4 * kMaxK];
    const int k_groups = (K + 7) / 8;
    const uint8x8_t v_xor_mask = vdup_n_u8(0x80);
    const bool apack_ok = (K <= kMaxK);

    // MR=4 main body: process 4 rows (two 2-row SMMLA pairs) per outer iteration.
    for (; i + 4 <= M; i += 4) {
        const uint8_t* a_row0 = A + (size_t)(i + 0) * K;
        const uint8_t* a_row1 = A + (size_t)(i + 1) * K;
        const uint8_t* a_row2 = A + (size_t)(i + 2) * K;
        const uint8_t* a_row3 = A + (size_t)(i + 3) * K;

        int32_t rb0 = -b_zp * row_sums[i + 0] + kakzbzp;
        int32_t rb1 = -b_zp * row_sums[i + 1] + kakzbzp;
        int32_t rb2 = -b_zp * row_sums[i + 2] + kakzbzp;
        int32_t rb3 = -b_zp * row_sums[i + 3] + kakzbzp;

        if (apack_ok) {
            // Full K-groups: 8-byte XOR-and-store per row.
            int kg = 0;
            for (; kg < kk_full / 8; kg++) {
                int kk = kg * 8;
                vst1_s8(a_pack + kg * 32 + 0,
                    vreinterpret_s8_u8(veor_u8(vld1_u8(a_row0 + kk), v_xor_mask)));
                vst1_s8(a_pack + kg * 32 + 8,
                    vreinterpret_s8_u8(veor_u8(vld1_u8(a_row1 + kk), v_xor_mask)));
                vst1_s8(a_pack + kg * 32 + 16,
                    vreinterpret_s8_u8(veor_u8(vld1_u8(a_row2 + kk), v_xor_mask)));
                vst1_s8(a_pack + kg * 32 + 24,
                    vreinterpret_s8_u8(veor_u8(vld1_u8(a_row3 + kk), v_xor_mask)));
            }
            // Partial K-tail group: zero slots beyond K contribute nothing to SMMLA.
            if (n_tail > 0) {
                std::memset(a_pack + kg * 32, 0, 32);
                for (int kx = 0; kx < n_tail; kx++) {
                    a_pack[kg * 32 + 0  + kx] = (int8_t)(a_row0[kk_full + kx] ^ 0x80);
                    a_pack[kg * 32 + 8  + kx] = (int8_t)(a_row1[kk_full + kx] ^ 0x80);
                    a_pack[kg * 32 + 16 + kx] = (int8_t)(a_row2[kk_full + kx] ^ 0x80);
                    a_pack[kg * 32 + 24 + kx] = (int8_t)(a_row3[kk_full + kx] ^ 0x80);
                }
            }
        }

#ifdef NNR_USE_XBYAK_AARCH64
        // Fast path: single JIT call handles all N/16 j-tiles for this i-block,
        // including SMMLA accumulation, ZIP extraction, row-bias + col-sum
        // compensation, and direct writes to final C. Eliminates per-j-tile
        // function-call overhead and the intermediate c_raw memory round-trip.
        if (apack_ok) {
            static const auto jit_uk = nnr::int8::neon_jit::get_jit_smmla_4xN();
            nnr::int8::neon_jit::JitSmmlaAux aux = { a_off_k, { rb0, rb1, rb2, rb3 } };
            int n_iters = (N + 15) / 16;
            jit_uk(a_pack, packed_B, col_sums, C + (size_t)i * N16,
                   k_groups, n_iters, N16, &aux);
            continue;
        }
#endif

        for (int j = 0; j < N; j += 16) {
            // 16 accumulators: acc[row_pair][col_pair]. Each holds [C[r*2, c*2],
            // C[r*2, c*2+1], C[r*2+1, c*2], C[r*2+1, c*2+1]] after all K-groups.
            int32x4_t acc0_0 = vdupq_n_s32(0), acc0_1 = vdupq_n_s32(0);
            int32x4_t acc0_2 = vdupq_n_s32(0), acc0_3 = vdupq_n_s32(0);
            int32x4_t acc0_4 = vdupq_n_s32(0), acc0_5 = vdupq_n_s32(0);
            int32x4_t acc0_6 = vdupq_n_s32(0), acc0_7 = vdupq_n_s32(0);
            int32x4_t acc1_0 = vdupq_n_s32(0), acc1_1 = vdupq_n_s32(0);
            int32x4_t acc1_2 = vdupq_n_s32(0), acc1_3 = vdupq_n_s32(0);
            int32x4_t acc1_4 = vdupq_n_s32(0), acc1_5 = vdupq_n_s32(0);
            int32x4_t acc1_6 = vdupq_n_s32(0), acc1_7 = vdupq_n_s32(0);

            if (apack_ok) {
                // Unified K loop (main + tail): pre-packed A already has XOR-0x80
                // fused in and K-tail group zero-padded. Two vld1q_s8 replace the
                // per-K 4×vld1_u8 + 4×veor + 2×vcombine.
                for (int kg = 0; kg < k_groups; kg++) {
                    int8x16_t av01 = vld1q_s8(a_pack + kg * 32);
                    int8x16_t av23 = vld1q_s8(a_pack + kg * 32 + 16);

                    const int8_t* bp = packed_B + (size_t)kg * 8 * N16 + (size_t)j * 8;
                    // Prefetch B for kg+2 (two K-groups = 256 B = 4 cache lines) —
                    // hides L2 latency when packed_B doesn't fully stay resident in L1.
                    if (kg + 2 < k_groups) {
                        const int8_t* bp_next = packed_B + (size_t)(kg + 2) * 8 * N16 + (size_t)j * 8;
                        NNR_PREFETCH_L1(bp_next +   0);
                        NNR_PREFETCH_L1(bp_next +  64);
                    }
                    int8x16_t b0 = vld1q_s8(bp +   0);
                    int8x16_t b1 = vld1q_s8(bp +  16);
                    int8x16_t b2 = vld1q_s8(bp +  32);
                    int8x16_t b3 = vld1q_s8(bp +  48);
                    int8x16_t b4 = vld1q_s8(bp +  64);
                    int8x16_t b5 = vld1q_s8(bp +  80);
                    int8x16_t b6 = vld1q_s8(bp +  96);
                    int8x16_t b7 = vld1q_s8(bp + 112);

                    acc0_0 = vmmlaq_s32(acc0_0, av01, b0); acc1_0 = vmmlaq_s32(acc1_0, av23, b0);
                    acc0_1 = vmmlaq_s32(acc0_1, av01, b1); acc1_1 = vmmlaq_s32(acc1_1, av23, b1);
                    acc0_2 = vmmlaq_s32(acc0_2, av01, b2); acc1_2 = vmmlaq_s32(acc1_2, av23, b2);
                    acc0_3 = vmmlaq_s32(acc0_3, av01, b3); acc1_3 = vmmlaq_s32(acc1_3, av23, b3);
                    acc0_4 = vmmlaq_s32(acc0_4, av01, b4); acc1_4 = vmmlaq_s32(acc1_4, av23, b4);
                    acc0_5 = vmmlaq_s32(acc0_5, av01, b5); acc1_5 = vmmlaq_s32(acc1_5, av23, b5);
                    acc0_6 = vmmlaq_s32(acc0_6, av01, b6); acc1_6 = vmmlaq_s32(acc1_6, av23, b6);
                    acc0_7 = vmmlaq_s32(acc0_7, av01, b7); acc1_7 = vmmlaq_s32(acc1_7, av23, b7);
                }
            } else {
                // Fallback for K > kMaxK (larger than stack A-pack budget).
                // Same math, slower per-K A build.
                for (int kk = 0; kk < kk_full; kk += 8) {
                    int8x8_t a0 = load_a8_xor(a_row0 + kk);
                    int8x8_t a1 = load_a8_xor(a_row1 + kk);
                    int8x8_t a2 = load_a8_xor(a_row2 + kk);
                    int8x8_t a3 = load_a8_xor(a_row3 + kk);
                    int8x16_t av01 = vcombine_s8(a0, a1);
                    int8x16_t av23 = vcombine_s8(a2, a3);

                    const int8_t* bp = packed_B + (size_t)(kk / 8) * 8 * N16 + (size_t)j * 8;
                    int8x16_t b0 = vld1q_s8(bp +   0);
                    int8x16_t b1 = vld1q_s8(bp +  16);
                    int8x16_t b2 = vld1q_s8(bp +  32);
                    int8x16_t b3 = vld1q_s8(bp +  48);
                    int8x16_t b4 = vld1q_s8(bp +  64);
                    int8x16_t b5 = vld1q_s8(bp +  80);
                    int8x16_t b6 = vld1q_s8(bp +  96);
                    int8x16_t b7 = vld1q_s8(bp + 112);

                    acc0_0 = vmmlaq_s32(acc0_0, av01, b0); acc1_0 = vmmlaq_s32(acc1_0, av23, b0);
                    acc0_1 = vmmlaq_s32(acc0_1, av01, b1); acc1_1 = vmmlaq_s32(acc1_1, av23, b1);
                    acc0_2 = vmmlaq_s32(acc0_2, av01, b2); acc1_2 = vmmlaq_s32(acc1_2, av23, b2);
                    acc0_3 = vmmlaq_s32(acc0_3, av01, b3); acc1_3 = vmmlaq_s32(acc1_3, av23, b3);
                    acc0_4 = vmmlaq_s32(acc0_4, av01, b4); acc1_4 = vmmlaq_s32(acc1_4, av23, b4);
                    acc0_5 = vmmlaq_s32(acc0_5, av01, b5); acc1_5 = vmmlaq_s32(acc1_5, av23, b5);
                    acc0_6 = vmmlaq_s32(acc0_6, av01, b6); acc1_6 = vmmlaq_s32(acc1_6, av23, b6);
                    acc0_7 = vmmlaq_s32(acc0_7, av01, b7); acc1_7 = vmmlaq_s32(acc1_7, av23, b7);
                }
                if (n_tail > 0) {
                    int8x8_t a0 = load_a8_xor_tail(a_row0 + kk_full, n_tail);
                    int8x8_t a1 = load_a8_xor_tail(a_row1 + kk_full, n_tail);
                    int8x8_t a2 = load_a8_xor_tail(a_row2 + kk_full, n_tail);
                    int8x8_t a3 = load_a8_xor_tail(a_row3 + kk_full, n_tail);
                    int8x16_t av01 = vcombine_s8(a0, a1);
                    int8x16_t av23 = vcombine_s8(a2, a3);

                    const int8_t* bp = packed_B + (size_t)(kk_full / 8) * 8 * N16 + (size_t)j * 8;
                    int8x16_t b0 = vld1q_s8(bp +   0), b1 = vld1q_s8(bp +  16);
                    int8x16_t b2 = vld1q_s8(bp +  32), b3 = vld1q_s8(bp +  48);
                    int8x16_t b4 = vld1q_s8(bp +  64), b5 = vld1q_s8(bp +  80);
                    int8x16_t b6 = vld1q_s8(bp +  96), b7 = vld1q_s8(bp + 112);

                    acc0_0 = vmmlaq_s32(acc0_0, av01, b0); acc1_0 = vmmlaq_s32(acc1_0, av23, b0);
                    acc0_1 = vmmlaq_s32(acc0_1, av01, b1); acc1_1 = vmmlaq_s32(acc1_1, av23, b1);
                    acc0_2 = vmmlaq_s32(acc0_2, av01, b2); acc1_2 = vmmlaq_s32(acc1_2, av23, b2);
                    acc0_3 = vmmlaq_s32(acc0_3, av01, b3); acc1_3 = vmmlaq_s32(acc1_3, av23, b3);
                    acc0_4 = vmmlaq_s32(acc0_4, av01, b4); acc1_4 = vmmlaq_s32(acc1_4, av23, b4);
                    acc0_5 = vmmlaq_s32(acc0_5, av01, b5); acc1_5 = vmmlaq_s32(acc1_5, av23, b5);
                    acc0_6 = vmmlaq_s32(acc0_6, av01, b6); acc1_6 = vmmlaq_s32(acc1_6, av23, b6);
                    acc0_7 = vmmlaq_s32(acc0_7, av01, b7); acc1_7 = vmmlaq_s32(acc1_7, av23, b7);
                }
            }

            // Epilogue: SMMLA 2×2-block layout → row-major output strips of 4 cols.
            // Output row r, cols j+cp*2 and j+cp*2+1 come from acc[r/2][cp] lanes
            // (r%2)*2 + 0 and (r%2)*2 + 1.  Combining low halves of adjacent col-pair
            // accumulators gives the top row's 4-col strip; combining high halves gives
            // the bottom row's.  For the 4-col strips covering cols j+0..3, j+4..7, ...:
            auto row_lo4 = [](int32x4_t a, int32x4_t b) {
                return vcombine_s32(vget_low_s32(a), vget_low_s32(b));
            };
            auto row_hi4 = [](int32x4_t a, int32x4_t b) {
                return vcombine_s32(vget_high_s32(a), vget_high_s32(b));
            };
            int32x4_t r0c03 = row_lo4(acc0_0, acc0_1);
            int32x4_t r0c47 = row_lo4(acc0_2, acc0_3);
            int32x4_t r0c8B = row_lo4(acc0_4, acc0_5);
            int32x4_t r0cCF = row_lo4(acc0_6, acc0_7);
            int32x4_t r1c03 = row_hi4(acc0_0, acc0_1);
            int32x4_t r1c47 = row_hi4(acc0_2, acc0_3);
            int32x4_t r1c8B = row_hi4(acc0_4, acc0_5);
            int32x4_t r1cCF = row_hi4(acc0_6, acc0_7);
            int32x4_t r2c03 = row_lo4(acc1_0, acc1_1);
            int32x4_t r2c47 = row_lo4(acc1_2, acc1_3);
            int32x4_t r2c8B = row_lo4(acc1_4, acc1_5);
            int32x4_t r2cCF = row_lo4(acc1_6, acc1_7);
            int32x4_t r3c03 = row_hi4(acc1_0, acc1_1);
            int32x4_t r3c47 = row_hi4(acc1_2, acc1_3);
            int32x4_t r3c8B = row_hi4(acc1_4, acc1_5);
            int32x4_t r3cCF = row_hi4(acc1_6, acc1_7);

            // Zero-point compensation: + row_bias, + (128-a_zp) * col_sums[j..j+15].
            int32x4_t cs03 = vld1q_s32(col_sums + j +  0);
            int32x4_t cs47 = vld1q_s32(col_sums + j +  4);
            int32x4_t cs8B = vld1q_s32(col_sums + j +  8);
            int32x4_t csCF = vld1q_s32(col_sums + j + 12);
            int32x4_t bv0 = vdupq_n_s32(rb0), bv1 = vdupq_n_s32(rb1);
            int32x4_t bv2 = vdupq_n_s32(rb2), bv3 = vdupq_n_s32(rb3);

            r0c03 = vmlaq_s32(vaddq_s32(r0c03, bv0), a_off_v, cs03);
            r0c47 = vmlaq_s32(vaddq_s32(r0c47, bv0), a_off_v, cs47);
            r0c8B = vmlaq_s32(vaddq_s32(r0c8B, bv0), a_off_v, cs8B);
            r0cCF = vmlaq_s32(vaddq_s32(r0cCF, bv0), a_off_v, csCF);
            r1c03 = vmlaq_s32(vaddq_s32(r1c03, bv1), a_off_v, cs03);
            r1c47 = vmlaq_s32(vaddq_s32(r1c47, bv1), a_off_v, cs47);
            r1c8B = vmlaq_s32(vaddq_s32(r1c8B, bv1), a_off_v, cs8B);
            r1cCF = vmlaq_s32(vaddq_s32(r1cCF, bv1), a_off_v, csCF);
            r2c03 = vmlaq_s32(vaddq_s32(r2c03, bv2), a_off_v, cs03);
            r2c47 = vmlaq_s32(vaddq_s32(r2c47, bv2), a_off_v, cs47);
            r2c8B = vmlaq_s32(vaddq_s32(r2c8B, bv2), a_off_v, cs8B);
            r2cCF = vmlaq_s32(vaddq_s32(r2cCF, bv2), a_off_v, csCF);
            r3c03 = vmlaq_s32(vaddq_s32(r3c03, bv3), a_off_v, cs03);
            r3c47 = vmlaq_s32(vaddq_s32(r3c47, bv3), a_off_v, cs47);
            r3c8B = vmlaq_s32(vaddq_s32(r3c8B, bv3), a_off_v, cs8B);
            r3cCF = vmlaq_s32(vaddq_s32(r3cCF, bv3), a_off_v, csCF);

            int32_t* c0 = C + (size_t)(i + 0) * N16 + j;
            int32_t* c1 = C + (size_t)(i + 1) * N16 + j;
            int32_t* c2 = C + (size_t)(i + 2) * N16 + j;
            int32_t* c3 = C + (size_t)(i + 3) * N16 + j;
            vst1q_s32(c0 +  0, r0c03); vst1q_s32(c0 +  4, r0c47);
            vst1q_s32(c0 +  8, r0c8B); vst1q_s32(c0 + 12, r0cCF);
            vst1q_s32(c1 +  0, r1c03); vst1q_s32(c1 +  4, r1c47);
            vst1q_s32(c1 +  8, r1c8B); vst1q_s32(c1 + 12, r1cCF);
            vst1q_s32(c2 +  0, r2c03); vst1q_s32(c2 +  4, r2c47);
            vst1q_s32(c2 +  8, r2c8B); vst1q_s32(c2 + 12, r2cCF);
            vst1q_s32(c3 +  0, r3c03); vst1q_s32(c3 +  4, r3c47);
            vst1q_s32(c3 +  8, r3c8B); vst1q_s32(c3 + 12, r3cCF);
        }
    }

    // MR=2 M-tail: process one row-pair at a time. Covers 2 rows if 2 or 3 remain.
    for (; i + 2 <= M; i += 2) {
        const uint8_t* a_row0 = A + (size_t)(i + 0) * K;
        const uint8_t* a_row1 = A + (size_t)(i + 1) * K;
        int32_t rb0 = -b_zp * row_sums[i + 0] + kakzbzp;
        int32_t rb1 = -b_zp * row_sums[i + 1] + kakzbzp;

        for (int j = 0; j < N; j += 16) {
            int32x4_t acc_0 = vdupq_n_s32(0), acc_1 = vdupq_n_s32(0);
            int32x4_t acc_2 = vdupq_n_s32(0), acc_3 = vdupq_n_s32(0);
            int32x4_t acc_4 = vdupq_n_s32(0), acc_5 = vdupq_n_s32(0);
            int32x4_t acc_6 = vdupq_n_s32(0), acc_7 = vdupq_n_s32(0);

            for (int kk = 0; kk < kk_full; kk += 8) {
                int8x8_t a0 = load_a8_xor(a_row0 + kk);
                int8x8_t a1 = load_a8_xor(a_row1 + kk);
                int8x16_t av = vcombine_s8(a0, a1);
                const int8_t* bp = packed_B + (size_t)(kk / 8) * 8 * N16 + (size_t)j * 8;
                acc_0 = vmmlaq_s32(acc_0, av, vld1q_s8(bp +   0));
                acc_1 = vmmlaq_s32(acc_1, av, vld1q_s8(bp +  16));
                acc_2 = vmmlaq_s32(acc_2, av, vld1q_s8(bp +  32));
                acc_3 = vmmlaq_s32(acc_3, av, vld1q_s8(bp +  48));
                acc_4 = vmmlaq_s32(acc_4, av, vld1q_s8(bp +  64));
                acc_5 = vmmlaq_s32(acc_5, av, vld1q_s8(bp +  80));
                acc_6 = vmmlaq_s32(acc_6, av, vld1q_s8(bp +  96));
                acc_7 = vmmlaq_s32(acc_7, av, vld1q_s8(bp + 112));
            }
            if (n_tail > 0) {
                int8x8_t a0 = load_a8_xor_tail(a_row0 + kk_full, n_tail);
                int8x8_t a1 = load_a8_xor_tail(a_row1 + kk_full, n_tail);
                int8x16_t av = vcombine_s8(a0, a1);
                const int8_t* bp = packed_B + (size_t)(kk_full / 8) * 8 * N16 + (size_t)j * 8;
                acc_0 = vmmlaq_s32(acc_0, av, vld1q_s8(bp +   0));
                acc_1 = vmmlaq_s32(acc_1, av, vld1q_s8(bp +  16));
                acc_2 = vmmlaq_s32(acc_2, av, vld1q_s8(bp +  32));
                acc_3 = vmmlaq_s32(acc_3, av, vld1q_s8(bp +  48));
                acc_4 = vmmlaq_s32(acc_4, av, vld1q_s8(bp +  64));
                acc_5 = vmmlaq_s32(acc_5, av, vld1q_s8(bp +  80));
                acc_6 = vmmlaq_s32(acc_6, av, vld1q_s8(bp +  96));
                acc_7 = vmmlaq_s32(acc_7, av, vld1q_s8(bp + 112));
            }

            int32x4_t r0c03 = vcombine_s32(vget_low_s32(acc_0),  vget_low_s32(acc_1));
            int32x4_t r0c47 = vcombine_s32(vget_low_s32(acc_2),  vget_low_s32(acc_3));
            int32x4_t r0c8B = vcombine_s32(vget_low_s32(acc_4),  vget_low_s32(acc_5));
            int32x4_t r0cCF = vcombine_s32(vget_low_s32(acc_6),  vget_low_s32(acc_7));
            int32x4_t r1c03 = vcombine_s32(vget_high_s32(acc_0), vget_high_s32(acc_1));
            int32x4_t r1c47 = vcombine_s32(vget_high_s32(acc_2), vget_high_s32(acc_3));
            int32x4_t r1c8B = vcombine_s32(vget_high_s32(acc_4), vget_high_s32(acc_5));
            int32x4_t r1cCF = vcombine_s32(vget_high_s32(acc_6), vget_high_s32(acc_7));

            int32x4_t cs03 = vld1q_s32(col_sums + j +  0);
            int32x4_t cs47 = vld1q_s32(col_sums + j +  4);
            int32x4_t cs8B = vld1q_s32(col_sums + j +  8);
            int32x4_t csCF = vld1q_s32(col_sums + j + 12);
            int32x4_t bv0 = vdupq_n_s32(rb0), bv1 = vdupq_n_s32(rb1);

            r0c03 = vmlaq_s32(vaddq_s32(r0c03, bv0), a_off_v, cs03);
            r0c47 = vmlaq_s32(vaddq_s32(r0c47, bv0), a_off_v, cs47);
            r0c8B = vmlaq_s32(vaddq_s32(r0c8B, bv0), a_off_v, cs8B);
            r0cCF = vmlaq_s32(vaddq_s32(r0cCF, bv0), a_off_v, csCF);
            r1c03 = vmlaq_s32(vaddq_s32(r1c03, bv1), a_off_v, cs03);
            r1c47 = vmlaq_s32(vaddq_s32(r1c47, bv1), a_off_v, cs47);
            r1c8B = vmlaq_s32(vaddq_s32(r1c8B, bv1), a_off_v, cs8B);
            r1cCF = vmlaq_s32(vaddq_s32(r1cCF, bv1), a_off_v, csCF);

            int32_t* c0 = C + (size_t)(i + 0) * N16 + j;
            int32_t* c1 = C + (size_t)(i + 1) * N16 + j;
            vst1q_s32(c0 +  0, r0c03); vst1q_s32(c0 +  4, r0c47);
            vst1q_s32(c0 +  8, r0c8B); vst1q_s32(c0 + 12, r0cCF);
            vst1q_s32(c1 +  0, r1c03); vst1q_s32(c1 +  4, r1c47);
            vst1q_s32(c1 +  8, r1c8B); vst1q_s32(c1 + 12, r1cCF);
        }
    }

    // Last row (if M is odd): scalar fallback. Fine — at most 1 row at the end.
    for (; i < M; i++) {
        const uint8_t* a_row = A + (size_t)i * K;
        int32_t rb = -b_zp * row_sums[i] + kakzbzp;
        for (int j = 0; j < N; j++) {
            int32_t s = 0;
            for (int kk = 0; kk < K; kk++) {
                int32_t av = (int32_t)((int8_t)(a_row[kk] ^ 0x80));  // u8 -> s8 via XOR 0x80
                // Retrieve B[kk, j] from the SMMLA packed layout.
                int jpair = j >> 1;
                int col   = j & 1;
                int k_group = kk >> 3;
                int k_in    = kk & 7;
                int32_t bv = (int32_t)packed_B[(size_t)k_group * 8 * N16
                                              + (size_t)jpair * 16
                                              + (size_t)col * 8
                                              + k_in];
                s += av * bv;
            }
            C[(size_t)i * N16 + j] = s + rb + a_off_k * col_sums[j];
        }
    }
}

#endif // __ARM_FEATURE_MATMUL_INT8

} // namespace nnr::int8::neon

#endif // aarch64
