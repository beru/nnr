#pragma once
// FP16 GEMM — NEON widening FMA (FP16 × FP16 → FP32 accumulator → FP32 output).
//
// Uses the ARMv8.2 FP16FML (`FHM`) intrinsic family `vfmlalq_{low,high}_f16` via
// the by-lane form `vfmlalq_lane_{low,high}_f16`. The low/high pair covers the
// 8 lanes of a float16x8_t B vector; the lane form lets one A value broadcast
// to 4 multiplication slots while the compiler keeps the A vector live across
// the row-group of FMAs.
//
// Accumulator is FP32: FP16 mantissa (11 bits) is insufficient for K > ~256.
//
// Tile: MR=4 × NR=16 main body + MR=1 M-tail. Register budget (main body):
//   16 × float32x4_t  accs   (v16..v31 conventionally)
//    2 × float16x8_t  B vecs
//    1 × float16x4_t  A gather vec
//  + a few scratch   — well within the 32-vreg AArch64 budget.
//
// External element type is `uint16_t` (FP16 bit pattern) following the NNR
// convention in `f16_convert_neon.h`; we reinterpret to `float16_t*` inside
// the kernel. This keeps the API portable across MSVC (where `float16_t` is
// the software struct) and GCC/Clang (where it aliases `__fp16`).
//
// Runtime gated on `has_neon_fp16()`. All three NNR ARM targets (Pi 5 A76,
// Oryon Gen 1, Oryon Gen 3) include FP16FML alongside FP16 arithmetic.

#if defined(__aarch64__) || defined(_M_ARM64)

#ifdef _MSC_VER
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif
#include <algorithm>
#include <cstdint>
#include <cstring>
#include "cpu_features.h"
#include "thread_pool.h"

namespace nnr::fp16::neon {

// Pack B[K×N] row-major FP16 into K-major [K][N16] with N-tail zero-pad so the
// kernel can always issue a full vld1q_u16 for 8-column groups.
//   N16 = round_up(N, 16).
inline size_t pack_b_fp16_neon_size(int K, int N)
{
    int N16 = (N + 15) & ~15;
    return (size_t)K * N16 * sizeof(uint16_t);
}

inline void pack_b_fp16_neon(uint16_t* __restrict dst,
                             const uint16_t* __restrict B,
                             int K, int N)
{
    int N16 = (N + 15) & ~15;
    std::memset(dst, 0, (size_t)K * N16 * sizeof(uint16_t));
    for (int k = 0; k < K; k++)
        std::memcpy(dst + (size_t)k * N16, B + (size_t)k * N,
                    (size_t)N * sizeof(uint16_t));
}

// C[M×N] = A[M×K] · B[K×N] in FP32. FP16 inputs, FP32 output.
//   A:        [M][K]   row-major   uint16_t (FP16 bit pattern)
//   B_packed: [K][N16] from pack_b_fp16_neon
//   C:        [M][N]   row-major   float
//
// Returns false if FP16 arithmetic isn't available at runtime.
inline bool gemm_fp16_neon(
    int M, int N, int K,
    const uint16_t* __restrict A,
    const uint16_t* __restrict B_packed,
    float* __restrict C)
{
    if (!has_neon_fp16()) return false;
    const int N16 = (N + 15) & ~15;
    constexpr int NR = 16;

    // Parallelize the outer M-block loop. Each block owns 4 rows of C with
    // no inter-block dependency; static scheduling hands each worker a
    // contiguous chunk (range/nthreads MR blocks) in a single dispatch.
    const int n_mblock = M / 4;
    const bool par = n_mblock > 1 && (int64_t)M * N * K > (1 << 20);
    nnr::for_static(0, n_mblock, par, [&](int ib) {
        const int i = ib * 4;
        const uint16_t* a0 = A + (size_t)(i + 0) * K;
        const uint16_t* a1 = A + (size_t)(i + 1) * K;
        const uint16_t* a2 = A + (size_t)(i + 2) * K;
        const uint16_t* a3 = A + (size_t)(i + 3) * K;

        for (int j = 0; j < N; j += NR) {
            float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
            float32x4_t c02 = vdupq_n_f32(0), c03 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
            float32x4_t c12 = vdupq_n_f32(0), c13 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
            float32x4_t c22 = vdupq_n_f32(0), c23 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
            float32x4_t c32 = vdupq_n_f32(0), c33 = vdupq_n_f32(0);

            const uint16_t* b_row = B_packed + (size_t)j;
            for (int k = 0; k < K; k++) {
                float16x8_t b_lo = vreinterpretq_f16_u16(
                    vld1q_u16(b_row + (size_t)k * N16));
                float16x8_t b_hi = vreinterpretq_f16_u16(
                    vld1q_u16(b_row + (size_t)k * N16 + 8));

                // Per-row broadcast: each A scalar loaded into all 8 lanes of
                // its own float16x8. Avoids the stack round-trip that a 4-entry
                // `uint16_t tmp[4] = {a0[k],a1[k],...}` gather would force for
                // the by-lane intrinsic. Compiler keeps a_* in vregs.
                float16x8_t a0v = vreinterpretq_f16_u16(vdupq_n_u16(a0[k]));
                float16x8_t a1v = vreinterpretq_f16_u16(vdupq_n_u16(a1[k]));
                float16x8_t a2v = vreinterpretq_f16_u16(vdupq_n_u16(a2[k]));
                float16x8_t a3v = vreinterpretq_f16_u16(vdupq_n_u16(a3[k]));

                c00 = vfmlalq_low_f16 (c00, a0v, b_lo);
                c01 = vfmlalq_high_f16(c01, a0v, b_lo);
                c02 = vfmlalq_low_f16 (c02, a0v, b_hi);
                c03 = vfmlalq_high_f16(c03, a0v, b_hi);
                c10 = vfmlalq_low_f16 (c10, a1v, b_lo);
                c11 = vfmlalq_high_f16(c11, a1v, b_lo);
                c12 = vfmlalq_low_f16 (c12, a1v, b_hi);
                c13 = vfmlalq_high_f16(c13, a1v, b_hi);
                c20 = vfmlalq_low_f16 (c20, a2v, b_lo);
                c21 = vfmlalq_high_f16(c21, a2v, b_lo);
                c22 = vfmlalq_low_f16 (c22, a2v, b_hi);
                c23 = vfmlalq_high_f16(c23, a2v, b_hi);
                c30 = vfmlalq_low_f16 (c30, a3v, b_lo);
                c31 = vfmlalq_high_f16(c31, a3v, b_lo);
                c32 = vfmlalq_low_f16 (c32, a3v, b_hi);
                c33 = vfmlalq_high_f16(c33, a3v, b_hi);
            }

            const int nstore = std::min(NR, N - j);
            float* c_base = C + (size_t)i * N + j;
            if (nstore == NR) {
                vst1q_f32(c_base + 0 * N + 0 , c00);
                vst1q_f32(c_base + 0 * N + 4 , c01);
                vst1q_f32(c_base + 0 * N + 8 , c02);
                vst1q_f32(c_base + 0 * N + 12, c03);
                vst1q_f32(c_base + 1 * N + 0 , c10);
                vst1q_f32(c_base + 1 * N + 4 , c11);
                vst1q_f32(c_base + 1 * N + 8 , c12);
                vst1q_f32(c_base + 1 * N + 12, c13);
                vst1q_f32(c_base + 2 * N + 0 , c20);
                vst1q_f32(c_base + 2 * N + 4 , c21);
                vst1q_f32(c_base + 2 * N + 8 , c22);
                vst1q_f32(c_base + 2 * N + 12, c23);
                vst1q_f32(c_base + 3 * N + 0 , c30);
                vst1q_f32(c_base + 3 * N + 4 , c31);
                vst1q_f32(c_base + 3 * N + 8 , c32);
                vst1q_f32(c_base + 3 * N + 12, c33);
            } else {
                // N-tail: spill to scratch then memcpy the nstore lanes.
                alignas(16) float tmp[4][NR];
                vst1q_f32(tmp[0] +  0, c00); vst1q_f32(tmp[0] +  4, c01);
                vst1q_f32(tmp[0] +  8, c02); vst1q_f32(tmp[0] + 12, c03);
                vst1q_f32(tmp[1] +  0, c10); vst1q_f32(tmp[1] +  4, c11);
                vst1q_f32(tmp[1] +  8, c12); vst1q_f32(tmp[1] + 12, c13);
                vst1q_f32(tmp[2] +  0, c20); vst1q_f32(tmp[2] +  4, c21);
                vst1q_f32(tmp[2] +  8, c22); vst1q_f32(tmp[2] + 12, c23);
                vst1q_f32(tmp[3] +  0, c30); vst1q_f32(tmp[3] +  4, c31);
                vst1q_f32(tmp[3] +  8, c32); vst1q_f32(tmp[3] + 12, c33);
                for (int r = 0; r < 4; r++)
                    std::memcpy(c_base + (size_t)r * N, tmp[r],
                                (size_t)nstore * sizeof(float));
            }
        }
    });

    // MR=1 M-tail (covers 0..3 remaining rows). Serial — at most 3 rows.
    for (int i = n_mblock * 4; i < M; i++) {
        const uint16_t* a_row = A + (size_t)i * K;
        for (int j = 0; j < N; j += NR) {
            float32x4_t c0 = vdupq_n_f32(0), c1 = vdupq_n_f32(0);
            float32x4_t c2 = vdupq_n_f32(0), c3 = vdupq_n_f32(0);
            const uint16_t* b_row = B_packed + (size_t)j;
            for (int k = 0; k < K; k++) {
                float16x8_t b_lo = vreinterpretq_f16_u16(
                    vld1q_u16(b_row + (size_t)k * N16));
                float16x8_t b_hi = vreinterpretq_f16_u16(
                    vld1q_u16(b_row + (size_t)k * N16 + 8));
                float16x8_t av = vreinterpretq_f16_u16(vdupq_n_u16(a_row[k]));

                c0 = vfmlalq_low_f16 (c0, av, b_lo);
                c1 = vfmlalq_high_f16(c1, av, b_lo);
                c2 = vfmlalq_low_f16 (c2, av, b_hi);
                c3 = vfmlalq_high_f16(c3, av, b_hi);
            }
            const int nstore = std::min(NR, N - j);
            float* c_row = C + (size_t)i * N + j;
            if (nstore == NR) {
                vst1q_f32(c_row +  0, c0);
                vst1q_f32(c_row +  4, c1);
                vst1q_f32(c_row +  8, c2);
                vst1q_f32(c_row + 12, c3);
            } else {
                alignas(16) float tmp[NR];
                vst1q_f32(tmp +  0, c0); vst1q_f32(tmp +  4, c1);
                vst1q_f32(tmp +  8, c2); vst1q_f32(tmp + 12, c3);
                std::memcpy(c_row, tmp, (size_t)nstore * sizeof(float));
            }
        }
    }
    return true;
}

} // namespace nnr::fp16::neon

#endif // aarch64
