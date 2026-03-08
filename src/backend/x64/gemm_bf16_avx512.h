#pragma once
// AVX512_BF16 GEMM: C[M×N] = A[M×K] × B[K×N]
// A and B are BF16 (uint16_t), C is accumulated in FP32.
// Uses VDPBF16PS: 2 BF16×BF16→FP32 per lane per instruction (2× throughput vs FP32 FMA).
//
// B must be pre-packed into VNNI format via pack_b_bf16():
//   packed[k_pair * N + j] = { B[2k, j] as low16, B[2k+1, j] as high16 }
// A is read row-major — consecutive K pairs are naturally adjacent.

#include <immintrin.h>
#include <cstring>
#include <algorithm>
#include "cpu_features.h"
#include "thread_pool.h"
#include "backend/cpu/kernel/post_ops.h"

namespace nnr::bf16 {

// Packed B size in uint16_t elements. K is rounded up to even.
// @nnr-meta isa=AVX512 dtype=bf16 layout=NCHW special=GEMM
inline size_t pack_b_bf16_size(int K, int N)
{
    int K2 = (K + 1) & ~1;  // round up to even
    // Each pair of K rows → N uint32_t (= 2N uint16_t)
    return (size_t)(K2 / 2) * N * 2;
}

// Pack B[K×N] (row-major BF16) into VNNI pairs for VDPBF16PS.
// Output: packed[k_pair][j] = { B[2k, j], B[2k+1, j] } as uint32_t.
// Stored as uint16_t array for pointer arithmetic convenience.
// @nnr-meta isa=AVX512 dtype=bf16 layout=NCHW special=GEMM
inline void pack_b_bf16(uint16_t* __restrict dst, const uint16_t* __restrict B, int K, int N)
{
    int K2 = (K + 1) & ~1;
    memset(dst, 0, pack_b_bf16_size(K, N) * sizeof(uint16_t));
    uint32_t* out = reinterpret_cast<uint32_t*>(dst);
    for (int k = 0; k < K - 1; k += 2) {
        const uint16_t* row0 = B + (size_t)k * N;
        const uint16_t* row1 = B + (size_t)(k + 1) * N;
        uint32_t* panel = out + (size_t)(k / 2) * N;
        for (int j = 0; j < N; j++)
            panel[j] = (uint32_t)row0[j] | ((uint32_t)row1[j] << 16);
    }
    // Odd K: last row paired with zero
    if (K & 1) {
        const uint16_t* row0 = B + (size_t)(K - 1) * N;
        uint32_t* panel = out + (size_t)((K - 1) / 2) * N;
        for (int j = 0; j < N; j++)
            panel[j] = (uint32_t)row0[j];
    }
}

// BF16 GEMM with pre-packed B.
// C[M×N] = A[M×K] × packed_B, accumulated in FP32.
// A: row-major BF16 [M×K], packed_B: VNNI-packed by pack_b_bf16().
// C: row-major FP32 [M×N] — caller converts to BF16 if needed.
template <typename PostFn>
// @nnr-meta isa=AVX512 dtype=bf16 layout=NCHW special=GEMM tiling=[K,NR] fusion=post_op
inline void dgemm_bf16(int M, int N, int K,
    const uint16_t* __restrict A, const uint16_t* __restrict packed_B,
    float* __restrict C, const PostFn& post_fn = PostFn{})
{
    int K2 = (K + 1) & ~1;
    int Kpairs = K2 / 2;
    // packed_B stride: N uint32_t per K-pair = 2*N uint16_t
    const uint32_t* B32 = reinterpret_cast<const uint32_t*>(packed_B);

    bool par = M > 4 && (int64_t)M * N * K > (1 << 18);

    nnr::for_static(0, M, par, [&](int i) {
        const uint16_t* a_row = A + (size_t)i * K;
        float* c_row = C + (size_t)i * N;

        int j = 0;
        // Main loop: 32 columns (2 ZMM accumulators)
        for (; j + 32 <= N; j += 32) {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();

            for (int kp = 0; kp < Kpairs; kp++) {
                // Broadcast A[i, 2k:2k+2] as paired BF16 to all 16 lanes
                __m512bh va = (__m512bh)_mm512_set1_epi32(
                    *(const uint32_t*)(a_row + kp * 2));
                // Load B packed pairs for columns j..j+15 and j+16..j+31
                __m512bh vb0 = (__m512bh)_mm512_loadu_si512(B32 + (size_t)kp * N + j);
                __m512bh vb1 = (__m512bh)_mm512_loadu_si512(B32 + (size_t)kp * N + j + 16);
                acc0 = _mm512_dpbf16_ps(acc0, va, vb0);
                acc1 = _mm512_dpbf16_ps(acc1, va, vb1);
            }

            _mm512_storeu_ps(c_row + j, acc0);
            _mm512_storeu_ps(c_row + j + 16, acc1);
        }
        // 16-column remainder
        for (; j + 16 <= N; j += 16) {
            __m512 acc0 = _mm512_setzero_ps();

            for (int kp = 0; kp < Kpairs; kp++) {
                __m512bh va = (__m512bh)_mm512_set1_epi32(
                    *(const uint32_t*)(a_row + kp * 2));
                __m512bh vb0 = (__m512bh)_mm512_loadu_si512(B32 + (size_t)kp * N + j);
                acc0 = _mm512_dpbf16_ps(acc0, va, vb0);
            }

            _mm512_storeu_ps(c_row + j, acc0);
        }
        // Masked tail (< 16 columns)
        if (j < N) {
            __mmask16 mask = (__mmask16)((1u << (N - j)) - 1);
            __m512 acc0 = _mm512_setzero_ps();

            for (int kp = 0; kp < Kpairs; kp++) {
                __m512bh va = (__m512bh)_mm512_set1_epi32(
                    *(const uint32_t*)(a_row + kp * 2));
                __m512bh vb0 = (__m512bh)_mm512_maskz_loadu_epi32(mask,
                    B32 + (size_t)kp * N + j);
                acc0 = _mm512_dpbf16_ps(acc0, va, vb0);
            }

            _mm512_mask_storeu_ps(c_row + j, mask, acc0);
        }

        post_fn.apply(i, c_row, N);
    });
}

// Simpler scalar tail: BF16 GEMM without packing (for small N remainder).
// Directly reads unpacked A[M×K] and B[K×N] in BF16.
template <typename PostFn>
// @nnr-meta isa=scalar dtype=bf16 layout=NCHW special=GEMM fusion=post_op
inline void dgemm_bf16_unpacked(int M, int N, int K,
    const uint16_t* __restrict A, const uint16_t* __restrict B,
    float* __restrict C, const PostFn& post_fn = PostFn{})
{
    int K2 = (K + 1) & ~1;
    int Kpairs = K2 / 2;

    bool par = M > 4 && (int64_t)M * N * K > (1 << 18);

    nnr::for_static(0, M, par, [&](int i) {
        const uint16_t* a_row = A + (size_t)i * K;
        float* c_row = C + (size_t)i * N;

        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += bfloat16_to_float32(a_row[k])
                     * bfloat16_to_float32(B[(size_t)k * N + j]);
            c_row[j] = sum;
        }

        post_fn.apply(i, c_row, N);
    });
}

} // namespace nnr::bf16
