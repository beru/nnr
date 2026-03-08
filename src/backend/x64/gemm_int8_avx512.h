#pragma once
// AVX-512 VNNI int8 GEMM: C[M×N] = (A - a_zp) × (B - b_zp)
// Uses VPDPBUSD: 4×(uint8×int8)→int32 per lane, 64 multiply-adds per instruction.
//
// A: uint8 [M×K] row-major (or int8, shifted to unsigned internally)
// B: int8 [K×N] row-major, must be pre-packed via pack_b_int8() into VNNI groups of 4.
// C: int32 [M×N] row-major output.
//
// Zero-point compensation:
//   (a - a_zp) * (b - b_zp) = a*b - a_zp*col_sum_B - b_zp*row_sum_A + K*a_zp*b_zp
// col_sum_B and row_sum_A are precomputed, so the inner VPDPBUSD loop uses raw values.

#include <immintrin.h>
#include <cstring>
#include <algorithm>
#include "cpu_features.h"
#include "thread_pool.h"
#include "jit_conv_int8_ukernel.h"

namespace nnr::int8 {

// Packed B size in bytes. K is rounded up to multiple of 4.
// @nnr-meta isa=scalar dtype=[int8,uint8] layout=NCHW special=GEMM
inline size_t pack_b_int8_size(int K, int N)
{
    int K4 = (K + 3) & ~3;
    return (size_t)K4 * N;
}

// Pack B[K×N] (row-major int8) into VNNI groups of 4.
// Output: for each group of 4 K rows and each column j, store 4 int8 values
// as {B[4k,j], B[4k+1,j], B[4k+2,j], B[4k+3,j]} in one uint32.
// Layout: packed[k_group * N + j] as uint32 containing 4 int8 values.
// @nnr-meta isa=SSE dtype=[int8,uint8] layout=NCHW special=GEMM
inline void pack_b_int8(int8_t* __restrict dst, const int8_t* __restrict B, int K, int N)
{
    int K4 = (K + 3) & ~3;
    memset(dst, 0, (size_t)K4 * N);
    uint32_t* out = reinterpret_cast<uint32_t*>(dst);

    // Process 4 rows at a time (one VNNI group) with SSE interleave
    int k = 0;
    for (; k + 4 <= K; k += 4) {
        const int8_t* r0 = B + (size_t)k * N;
        const int8_t* r1 = r0 + N;
        const int8_t* r2 = r1 + N;
        const int8_t* r3 = r2 + N;
        uint32_t* panel = out + (size_t)(k / 4) * N;
        int j = 0;
        for (; j + 16 <= N; j += 16) {
            __m128i v0 = _mm_loadu_si128((const __m128i*)(r0 + j));
            __m128i v1 = _mm_loadu_si128((const __m128i*)(r1 + j));
            __m128i v2 = _mm_loadu_si128((const __m128i*)(r2 + j));
            __m128i v3 = _mm_loadu_si128((const __m128i*)(r3 + j));
            __m128i lo01 = _mm_unpacklo_epi8(v0, v1);
            __m128i hi01 = _mm_unpackhi_epi8(v0, v1);
            __m128i lo23 = _mm_unpacklo_epi8(v2, v3);
            __m128i hi23 = _mm_unpackhi_epi8(v2, v3);
            _mm_storeu_si128((__m128i*)(panel + j + 0),  _mm_unpacklo_epi16(lo01, lo23));
            _mm_storeu_si128((__m128i*)(panel + j + 4),  _mm_unpackhi_epi16(lo01, lo23));
            _mm_storeu_si128((__m128i*)(panel + j + 8),  _mm_unpacklo_epi16(hi01, hi23));
            _mm_storeu_si128((__m128i*)(panel + j + 12), _mm_unpackhi_epi16(hi01, hi23));
        }
        for (; j < N; j++)
            panel[j] = (uint32_t)(uint8_t)r0[j] | ((uint32_t)(uint8_t)r1[j] << 8)
                | ((uint32_t)(uint8_t)r2[j] << 16) | ((uint32_t)(uint8_t)r3[j] << 24);
    }
    // Remainder K rows (< 4)
    if (k < K) {
        int8_t (*out4)[4] = reinterpret_cast<int8_t(*)[4]>(dst);
        for (; k < K; k++) {
            int kl = k % 4;
            const int8_t* row = B + (size_t)k * N;
            for (int j = 0; j < N; j++)
                out4[(k / 4) * N + j][kl] = row[j];
        }
    }
}

// Compute column sums of B: col_sum[j] = sum_k(B[k,j])
// Used for zero-point compensation.
// @nnr-meta isa=AVX512 dtype=int8 layout=NCHW special=GEMM
inline void compute_col_sums(int32_t* __restrict col_sum, const int8_t* __restrict B,
    int K, int N)
{
    int j = 0;
    for (; j + 16 <= N; j += 16)
        _mm512_storeu_si512(col_sum + j, _mm512_setzero_si512());
    for (; j < N; j++) col_sum[j] = 0;

    for (int k = 0; k < K; k++) {
        const int8_t* row = B + (size_t)k * N;
        j = 0;
        for (; j + 16 <= N; j += 16) {
            __m512i acc = _mm512_loadu_si512(col_sum + j);
            __m128i bytes = _mm_loadu_si128((const __m128i*)(row + j));
            _mm512_storeu_si512(col_sum + j, _mm512_add_epi32(acc, _mm512_cvtepi8_epi32(bytes)));
        }
        for (; j < N; j++)
            col_sum[j] += (int32_t)row[j];
    }
}

// Fused pack + column sums: pack B into VNNI groups AND compute column sums in one pass.
// @nnr-meta isa=[AVX512,SSE] dtype=[int8,uint8] layout=NCHW special=GEMM
inline void pack_b_int8_and_col_sums(int8_t* __restrict dst, int32_t* __restrict col_sum,
    const int8_t* __restrict B, int K, int N)
{
    int K4 = (K + 3) & ~3;
    memset(dst, 0, (size_t)K4 * N);
    // Zero col_sum
    int j = 0;
    for (; j + 16 <= N; j += 16)
        _mm512_storeu_si512(col_sum + j, _mm512_setzero_si512());
    for (; j < N; j++) col_sum[j] = 0;

    uint32_t* out = reinterpret_cast<uint32_t*>(dst);

    // Process 4 rows at a time (one VNNI group), accumulate col sums simultaneously
    int k = 0;
    for (; k + 4 <= K; k += 4) {
        const int8_t* r0 = B + (size_t)k * N;
        const int8_t* r1 = r0 + N;
        const int8_t* r2 = r1 + N;
        const int8_t* r3 = r2 + N;
        uint32_t* panel = out + (size_t)(k / 4) * N;
        j = 0;
        for (; j + 16 <= N; j += 16) {
            __m128i v0 = _mm_loadu_si128((const __m128i*)(r0 + j));
            __m128i v1 = _mm_loadu_si128((const __m128i*)(r1 + j));
            __m128i v2 = _mm_loadu_si128((const __m128i*)(r2 + j));
            __m128i v3 = _mm_loadu_si128((const __m128i*)(r3 + j));
            // Interleave into VNNI groups
            __m128i lo01 = _mm_unpacklo_epi8(v0, v1);
            __m128i hi01 = _mm_unpackhi_epi8(v0, v1);
            __m128i lo23 = _mm_unpacklo_epi8(v2, v3);
            __m128i hi23 = _mm_unpackhi_epi8(v2, v3);
            _mm_storeu_si128((__m128i*)(panel + j + 0),  _mm_unpacklo_epi16(lo01, lo23));
            _mm_storeu_si128((__m128i*)(panel + j + 4),  _mm_unpackhi_epi16(lo01, lo23));
            _mm_storeu_si128((__m128i*)(panel + j + 8),  _mm_unpacklo_epi16(hi01, hi23));
            _mm_storeu_si128((__m128i*)(panel + j + 12), _mm_unpackhi_epi16(hi01, hi23));
            // Accumulate column sums (sign-extend and add all 4 rows)
            __m512i acc = _mm512_loadu_si512(col_sum + j);
            acc = _mm512_add_epi32(acc, _mm512_cvtepi8_epi32(v0));
            acc = _mm512_add_epi32(acc, _mm512_cvtepi8_epi32(v1));
            acc = _mm512_add_epi32(acc, _mm512_cvtepi8_epi32(v2));
            acc = _mm512_add_epi32(acc, _mm512_cvtepi8_epi32(v3));
            _mm512_storeu_si512(col_sum + j, acc);
        }
        for (; j < N; j++) {
            panel[j] = (uint32_t)(uint8_t)r0[j] | ((uint32_t)(uint8_t)r1[j] << 8)
                | ((uint32_t)(uint8_t)r2[j] << 16) | ((uint32_t)(uint8_t)r3[j] << 24);
            col_sum[j] += (int32_t)r0[j] + r1[j] + r2[j] + r3[j];
        }
    }
    // Remainder K rows (< 4)
    if (k < K) {
        int8_t (*out4)[4] = reinterpret_cast<int8_t(*)[4]>(dst);
        for (; k < K; k++) {
            int kl = k % 4;
            const int8_t* row = B + (size_t)k * N;
            j = 0;
            for (; j + 16 <= N; j += 16) {
                __m512i acc = _mm512_loadu_si512(col_sum + j);
                __m128i bytes = _mm_loadu_si128((const __m128i*)(row + j));
                _mm512_storeu_si512(col_sum + j, _mm512_add_epi32(acc, _mm512_cvtepi8_epi32(bytes)));
            }
            for (; j < N; j++) {
                out4[(k / 4) * N + j][kl] = row[j];
                col_sum[j] += (int32_t)row[j];
            }
            // Pack the row (must do after col_sum loop for scalar tail)
            for (j = 0; j + 16 <= N; j += 16) {
                // Already packed by interleave above for full groups;
                // for remainder, scatter individually
            }
            // Scalar pack for remainder K rows
            for (j = 0; j < N; j++)
                out4[(k / 4) * N + j][kl] = row[j];
        }
    }
}

// Compute row sums of A: row_sum[i] = sum_k(A[i,k])
// @nnr-meta isa=scalar dtype=uint8 layout=NCHW special=GEMM
inline void compute_row_sums(int32_t* __restrict row_sum, const uint8_t* __restrict A,
    int M, int K)
{
    for (int i = 0; i < M; i++) {
        const uint8_t* row = A + (size_t)i * K;
        int32_t sum = 0;
        for (int k = 0; k < K; k++)
            sum += (int32_t)row[k];
        row_sum[i] = sum;
    }
}

// Int8 GEMM with pre-packed B.
// C[i,j] = sum_k(A[i,k] * packed_B[k,j])  (raw unsigned×signed dot product)
//         - a_zp * col_sum_B[j]             (zero-point compensation)
//         - b_zp * row_sum_A[i]
//         + K * a_zp * b_zp
//
// A: uint8 [M×K], packed_B: VNNI-packed by pack_b_int8(),
// col_sum_B: [N] int32, row_sum_A: [M] int32, C: [M×N] int32.
// GEMV-tiled packing: [N/16][Kgroups][16] layout for sequential B access in GEMV.
// Each N-block of 16 columns has all its Kgroups contiguous → 64-byte sequential reads.
// Size is same as standard packing: K4 * N bytes.
// @nnr-meta isa=scalar dtype=[int8,uint8] layout=NCHW special=GEMM
inline void pack_b_int8_gemv(int8_t* __restrict dst, int32_t* __restrict col_sum,
    const int8_t* __restrict B, int K, int N)
{
    int K4 = (K + 3) & ~3;
    int Kgroups = K4 / 4;
    int Nblocks = (N + 15) / 16;
    uint32_t* out = reinterpret_cast<uint32_t*>(dst);
    memset(dst, 0, (size_t)K4 * N);
    memset(col_sum, 0, (size_t)N * sizeof(int32_t));

    for (int nb = 0; nb < Nblocks; nb++) {
        int j0 = nb * 16;
        int je = std::min(j0 + 16, N);
        int ncols = je - j0;
        uint32_t* block = out + (size_t)nb * Kgroups * 16;

        int k = 0;
        for (; k + 4 <= K; k += 4) {
            int kg = k / 4;
            const int8_t* r0 = B + (size_t)k * N + j0;
            const int8_t* r1 = B + (size_t)(k + 1) * N + j0;
            const int8_t* r2 = B + (size_t)(k + 2) * N + j0;
            const int8_t* r3 = B + (size_t)(k + 3) * N + j0;
            uint32_t* panel = block + (size_t)kg * 16;
            for (int j = 0; j < ncols; j++) {
                panel[j] = (uint32_t)(uint8_t)r0[j] | ((uint32_t)(uint8_t)r1[j] << 8)
                    | ((uint32_t)(uint8_t)r2[j] << 16) | ((uint32_t)(uint8_t)r3[j] << 24);
                col_sum[j0 + j] += (int32_t)r0[j] + r1[j] + r2[j] + r3[j];
            }
        }
        for (; k < K; k++) {
            int kg = k / 4;
            int kl = k % 4;
            const int8_t* row = B + (size_t)k * N + j0;
            int8_t (*panel4)[4] = reinterpret_cast<int8_t(*)[4]>(block + (size_t)kg * 16);
            for (int j = 0; j < ncols; j++) {
                panel4[j][kl] = row[j];
                col_sum[j0 + j] += (int32_t)row[j];
            }
        }
    }
}

// @nnr-meta isa=scalar dtype=[int8,uint8] layout=NCHW special=GEMM
inline size_t pack_b_int8_gemv_size(int K, int N)
{
    int K4 = (K + 3) & ~3;
    int Nblocks = (N + 15) / 16;
    return (size_t)Nblocks * (K4 / 4) * 16 * sizeof(uint32_t);
}

// GEMV kernel using tiled packing. Sequential B reads within each N-block.
// @nnr-meta isa=[AVX512,AVX512_VNNI] dtype=[int8,uint8] layout=NCHW special=GEMM tiling=NR
inline void gemm_int8_gemv(int M, int N, int K,
    const uint8_t* __restrict A, int a_zp,
    const int8_t* __restrict packed_B_gemv, int b_zp,
    const int32_t* __restrict col_sum_B,
    const int32_t* __restrict row_sum_A,
    int32_t* __restrict C)
{
    int K4 = (K + 3) & ~3;
    int Kgroups = K4 / 4;
    int Nblocks = (N + 15) / 16;
    const uint32_t* B32 = reinterpret_cast<const uint32_t*>(packed_B_gemv);
    int32_t zp_product = (int32_t)K * a_zp * b_zp;

    int nt = (Nblocks > 1)
           ? nnr::int8_compute_threads(Nblocks, (int64_t)M * N * K) : 1;

    for (int row = 0; row < M; row++) {
        const uint8_t* arow = A + (size_t)row * K;
        int32_t* crow = C + (size_t)row * N;

        nnr::for_static(0, Nblocks, nt, [&](int nb) {
            int j0 = nb * 16;
            int ncols = std::min(16, N - j0);
            __mmask16 mask = (ncols == 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << ncols) - 1);

            __m512i acc = _mm512_setzero_si512();
            const uint32_t* bblock = B32 + (size_t)nb * Kgroups * 16;

            // K-outer loop: sequential 64-byte reads through this N-block's data
            for (int kg = 0; kg < Kgroups; kg++) {
                __m512i va = _mm512_set1_epi32(*(const uint32_t*)(arow + kg * 4));
                __m512i vb = _mm512_maskz_loadu_epi32(mask, bblock + (size_t)kg * 16);
                acc = _mm512_dpbusd_epi32(acc, va, vb);
            }

            // Zero-point compensation and store
            __m512i col_comp = _mm512_mullo_epi32(_mm512_set1_epi32(a_zp),
                _mm512_maskz_loadu_epi32(mask, col_sum_B + j0));
            __m512i result = _mm512_sub_epi32(acc, col_comp);
            result = _mm512_sub_epi32(result, _mm512_set1_epi32(b_zp * row_sum_A[row]));
            result = _mm512_add_epi32(result, _mm512_set1_epi32(zp_product));
            _mm512_mask_storeu_epi32(crow + j0, mask, result);
        });
    }
}

// @nnr-meta isa=[AVX512,AVX512_VNNI] dtype=[int8,uint8] layout=NCHW special=[GEMM,JIT] tiling=[MR,NR] fusion=qdq
inline void gemm_int8(int M, int N, int K,
    const uint8_t* __restrict A, int a_zp,
    const int8_t* __restrict packed_B, int b_zp,
    const int32_t* __restrict col_sum_B,
    const int32_t* __restrict row_sum_A,
    int32_t* __restrict C)
{
    // Note: for M <= 4, caller should use gemm_int8_gemv with GEMV-tiled packing instead.

    int K4 = (K + 3) & ~3;
    int Kgroups = K4 / 4;
    const uint32_t* B32 = reinterpret_cast<const uint32_t*>(packed_B);
    int32_t zp_product = (int32_t)K * a_zp * b_zp;

    constexpr int MR = 6;

#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)
    // JIT NR=32 path: 2× VNNI output per K-step vs intrinsics NR=16.
    // Uses standard packed_B directly (stride N per K-group, no re-packing).
    // Zero-point: adj_col[j] = -azp * col_sum_B[j]
    //             adj_row[r] = -bzp * row_sum_A[r] + K*azp*bzp
    // adj arrays are computed per-tile on the stack — no heap allocation.
    static jit_cache_t<int, jit_packed_gemm_nr32_avx512_t> nr32_cache;
    jit_packed_gemm_nr32_fn_t nr32_fns[MR + 1] = {};
    for (int mr = 1; mr <= MR; mr++)
        nr32_fns[mr] = nr32_cache.get_or_create(mr, mr)->fn<jit_packed_gemm_nr32_fn_t>();

    constexpr int NR32 = 32;
    int nblocks_m  = (M + MR   - 1) / MR;
    int nblocks_n  = (N + NR32 - 1) / NR32;
    int ntiles = nblocks_m * nblocks_n;
    int nt = (ntiles > 1)
           ? nnr::int8_compute_threads(ntiles, (int64_t)M * N * K) : 1;
    int ldb = N * 4;   // packed B K-group stride in bytes

    nnr::for_dynamic(0, ntiles, nt, [&](int /*tid*/, int tile) {
        int mb = tile / nblocks_n;
        int nb = tile % nblocks_n;
        int i0 = mb * MR;
        int j0 = nb * NR32;
        int ie = std::min(i0 + MR, M);
        int je = std::min(j0 + NR32, N);
        int nr = ie - i0;
        int jw = je - j0;

        // Per-tile zero-point correction (stack, no heap).
        // adj_col[k] = -azp * col_sum_B[j0+k], padded to 32 for safe ZMM load.
        // adj_row[r] = -bzp * row_sum_A[i0+r] + K*azp*bzp
        int32_t adj_col[NR32] = {};
        int32_t adj_row[MR] = {};
        for (int k = 0; k < jw; k++) adj_col[k] = -(int32_t)a_zp * col_sum_B[j0 + k];
        for (int r = 0; r < nr; r++) adj_row[r] = -(int32_t)b_zp * row_sum_A[i0 + r] + zp_product;

        uint16_t mask_lo = (uint16_t)((jw >= 16) ? 0xFFFFu : ((1u << jw) - 1));
        uint16_t mask_hi = (uint16_t)((jw <= 16) ? 0u     : ((1u << (jw - 16)) - 1));

        nr32_fns[nr](
            A + (size_t)i0 * K,       // A row i0
            B32 + j0,                 // packed B: K-group 0, col j0 (stride ldb per K-group)
            C + (size_t)i0 * N + j0,  // C tile
            Kgroups,
            K,                        // lda: A row stride in bytes
            N,                        // ldc: C row stride in int32 elements
            adj_row,
            adj_col,
            mask_lo, mask_hi,
            ldb);
    });

#else
    // Intrinsics fallback: 2D tiling over M-blocks × 64-col blocks.
    __m512i v_azp = _mm512_set1_epi32(a_zp);
    __m512i v_zpp = _mm512_set1_epi32(zp_product);

    constexpr int NBLK = 64;
    int nblocks_m = (M + MR   - 1) / MR;
    int nblocks_n = (N + NBLK - 1) / NBLK;
    int ntiles = nblocks_m * nblocks_n;
    int nt = (ntiles > 1)
           ? nnr::int8_compute_threads(ntiles, (int64_t)M * N * K) : 1;

    nnr::for_dynamic(0, ntiles, nt, [&](int /*tid*/, int tile) {
        int mb = tile / nblocks_n;
        int nb = tile % nblocks_n;
        int i0 = mb * MR;
        int j0 = nb * NBLK;
        int ie = std::min(i0 + MR, M);
        int je = std::min(j0 + NBLK, N);
        int nr = ie - i0;

        const uint8_t* aptr[6];
        for (int r = 0; r < nr; r++) aptr[r] = A + (size_t)(i0 + r) * K;

        for (int j = j0; j < je; j += 16) {
            int jw = std::min(16, je - j);
            __mmask16 mask = (jw == 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << jw) - 1);

            __m512i acc[6];
            for (int r = 0; r < nr; r++) acc[r] = _mm512_setzero_si512();

            for (int kg = 0; kg < Kgroups; kg++) {
                __m512i vb = _mm512_maskz_loadu_epi32(mask, B32 + (size_t)kg * N + j);
                for (int r = 0; r < nr; r++) {
                    __m512i va = _mm512_set1_epi32(*(const uint32_t*)(aptr[r] + kg * 4));
                    acc[r] = _mm512_dpbusd_epi32(acc[r], va, vb);
                }
            }

            __m512i col_comp = _mm512_mullo_epi32(v_azp, _mm512_maskz_loadu_epi32(mask, col_sum_B + j));
            for (int r = 0; r < nr; r++) {
                __m512i result = _mm512_sub_epi32(acc[r], col_comp);
                result = _mm512_sub_epi32(result, _mm512_set1_epi32(b_zp * row_sum_A[i0 + r]));
                result = _mm512_add_epi32(result, v_zpp);
                _mm512_mask_storeu_epi32(C + (size_t)(i0 + r) * N + j, mask, result);
            }
        }
    });
#endif
}

// Optional requantize epilogue: fuse int32→float→uint8 into the tile body.
// When non-null, the caller's separate requantize loop is not needed.
//
// Two field sets coexist:
//   ORT-style (packed NR=48): output_scales, bias_int32, qmin/qmax (adjusted), y_zp_int
//   Legacy (gather-GEMM, fused im2col): combined_scales, bias_vals, inv_y_scale, y_zp
// Caller fills both. Migrate legacy paths incrementally.
struct conv_rq_params_t {
    // ORT-style: acc+bias_i32 → float → *output_scale → clamp[−zp,255−zp] → int32 → +zp → vpmovusdb
    const float* output_scales;    // [MM] per-channel: x_scale * w_scale / y_scale
    const int32_t* bias_int32;     // [MM] raw integer bias, or nullptr
    float rq_qmin;                 // 0.0f - y_zp (adjusted for pre-zp clamp)
    float rq_qmax;                 // 255.0f - y_zp
    int32_t y_zp_int;             // output zero point (added after cvtps2dq)
    // Legacy (gather-GEMM, fused im2col — same struct, different fields)
    const float* combined_scales;  // [MM] x_scale * w_scale (without /y_scale)
    const float* bias_vals;        // [MM] pre-scaled float bias, or nullptr
    float inv_y_scale;
    float y_zp;                    // float cast of y zero point
    float qmin, qmax;             // [0, 255] for legacy paths
    // Common
    uint8_t* Y_out;
    int y_out_stride;
};

// ── Gather-GEMM: fused transposed int8 conv with pre-packed weights ──
// M=spatial(ow), N=OC(channels), K=CHW.
// JIT kernel: pre-biased accumulators (adj_col), gather K-loop, fused requantize → uint8 NHWC.
// No intermediate y_i32 buffer — everything done in-register.
// @nnr-meta isa=[AVX512,AVX512_VNNI] dtype=[int8,uint8] layout=NHWC special=[GEMM,JIT,Direct] tiling=[MR,NR] fusion=qdq
inline void conv_int8_gather_gemm(
    int OC, int oW,
    int oh_start, int oh_count,
    int K,
    const int8_t* __restrict W_packed,
    const int32_t* __restrict w_col_sums,
    const uint8_t* __restrict x_pad,
    int x_zp,
    const size_t* __restrict k_off_oh_all,
    const conv_rq_params_t* rq)
{
    int K4 = (K + 3) & ~3;
    int Kgroups = K4 / 4;
    constexpr int MR = GATHER_GEMM_MR;
    constexpr int NR = GATHER_GEMM_NR;
    int Nblocks = (OC + NR - 1) / NR;

    // Pre-compute adj_col per NR-block: -x_zp * col_sum_w, padded to 48
    // Each NR-block gets its own 48-element adj_col array.
    std::vector<int32_t> adj_col_all((size_t)Nblocks * 48, 0);
    for (int oc = 0; oc < OC; oc++)
        adj_col_all[oc / 48 * 48 + oc % 48] = -(int32_t)x_zp * w_col_sums[oc];

#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)
    static jit_cache_t<int, jit_gather_gemm_nr48_avx512_t> gather_cache;
    jit_gather_gemm_nr48_fn_t jit_fns[MR + 1] = {};
    for (int mr = 1; mr <= MR; mr++)
        jit_fns[mr] = gather_cache.get_or_create(mr, mr)->fn<jit_gather_gemm_nr48_fn_t>();
#endif

    int nblocks_ow = (oW + MR - 1) / MR;
    int ntiles = nblocks_ow * oh_count;
    int nt = (ntiles > 1)
           ? nnr::int8_compute_threads(ntiles,
                 (int64_t)OC * oW * oh_count * K) : 1;

    nnr::for_dynamic(0, ntiles, nt, [&](int /*tid*/, int tile) {
        int mb = tile / oh_count;
        int oh_rel = tile % oh_count;
        int ow0 = mb * MR;
        int mr = std::min(MR, oW - ow0);
        int oh = oh_start + oh_rel;
        const size_t* k_off = k_off_oh_all + (size_t)oh * K4;

        for (int nb = 0; nb < Nblocks; nb++) {
            int oc0 = nb * NR;
            int jw = std::min(NR, OC - oc0);
            uint16_t m0 = jw >= 16 ? 0xFFFF : (uint16_t)((1u << jw) - 1);
            uint16_t m1 = jw >= 32 ? 0xFFFF : jw > 16 ? (uint16_t)((1u << (jw-16)) - 1) : 0;
            uint16_t m2 = jw > 32 ? (jw >= 48 ? 0xFFFF : (uint16_t)((1u << (jw-32)) - 1)) : 0;

            // Build per-block rq params pointing to the right channel slice
            gather_rq_t rq_blk;
            rq_blk.scale = rq->combined_scales + oc0;
            rq_blk.bias  = rq->bias_vals ? rq->bias_vals + oc0 : nullptr;
            rq_blk.inv_y_scale = rq->inv_y_scale;
            rq_blk.y_zp        = rq->y_zp;
            rq_blk.qmin         = rq->qmin;
            rq_blk.qmax         = rq->qmax;
            // NHWC output: base at [oh*oW + ow0][oc0]
            rq_blk.Y_out    = rq->Y_out + ((size_t)oh * oW + ow0) * rq->y_out_stride + oc0;
            rq_blk.y_stride = rq->y_out_stride;  // OC (full row stride between spatial positions)

#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)
            const int8_t* Wb = W_packed + (size_t)nb * Kgroups * 192;
            const int32_t* adj = adj_col_all.data() + nb * 48;
            jit_fns[mr](x_pad + ow0, Wb, adj, Kgroups, k_off, &rq_blk, m0, m1, m2);
#endif
        }
    });
}

// Fused im2col + int8 GEMM for stride-1 convolution.
// Reads input pixels directly during the K-loop — no im2col buffer, no B-packing.
// x_pad: pre-padded input [C × padH × padW] uint8 (NOT shifted; -128 applied on-the-fly).
//        Must have ≥16 bytes slack past last valid pixel for unmasked JIT loads.
// W: weight buffer, zero-padded at positions [CHW, K4) per row.
// Y: scratch [MM × oh_count * oW] int32 (written by JIT, read back for ZP compensation).
// rq: if non-null, requantize in-tile and write uint8 to rq->Y_out (no separate pass needed).
// @nnr-meta isa=[AVX512,AVX512_VNNI] dtype=[int8,uint8] layout=NHWC special=[GEMM,JIT,IM2COL] tiling=[MR,NR] fusion=qdq
inline void conv_int8_fused_gemm(
    int MM, int oW,
    int oh_start, int oh_count,
    int C, int kH, int kW, int dH, int dW,
    const uint8_t* __restrict W, int a_zp,
    const int32_t* __restrict row_sum_W,
    const uint8_t* __restrict x_pad, int padH, int padW,
    int b_zp,
    int32_t* __restrict Y,
    const size_t* __restrict k_off_base = nullptr,   // pre-computed base [K4]
    const size_t* __restrict k_off_oh_all = nullptr, // pre-computed [oH × K4]
    const conv_rq_params_t* rq = nullptr)
{
    int CHW = C * kH * kW;
    int K4 = (CHW + 3) & ~3;
    int Kgroups = K4 / 4;
    int chunk_sp = oh_count * oW;
    size_t plane = (size_t)padH * padW;

    int32_t zp_product = (int32_t)CHW * a_zp * b_zp;
    __m512i v_azp = _mm512_set1_epi32(a_zp);
    __m512i v_zpp = _mm512_set1_epi32(zp_product);

    // Use pre-computed k_off if provided, else compute on the fly
    std::vector<size_t> k_off_local;
    const size_t* k_off;
    if (k_off_base) {
        k_off = k_off_base;
    } else {
        k_off_local.resize(K4);
        for (int k = 0; k < CHW; k++) {
            int c = k / (kH * kW);
            int rem = k % (kH * kW);
            int kh = rem / kW;
            int kw = rem % kW;
            k_off_local[k] = (size_t)c * plane + (size_t)(kh * dH) * padW + (size_t)(kw * dW);
        }
        for (int k = CHW; k < K4; k++)
            k_off_local[k] = 0;
        k_off = k_off_local.data();
    }

#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)
    static jit_cache_t<jit_conv_int8_key_t, jit_conv_int8_ukernel_avx512_t,
                       jit_conv_int8_hash_t> jit_cache;
    static jit_cache_t<jit_conv_int8_key_t, jit_conv_int8_nr32_ukernel_avx512_t,
                       jit_conv_int8_hash_t> jit_nr32_cache;
    bool use_jit = (CHW % 4 == 0);
    bool use_jit_nr32 = use_jit && (oW > 16);
#else
    constexpr bool use_jit = false;
    constexpr bool use_jit_nr32 = false;
#endif

    const int MR = use_jit_nr32 ? CONV_INT8_MR_JIT_NR32
                 : use_jit      ? CONV_INT8_MR_JIT
                                : CONV_INT8_MR;
    int nblocks_m = (MM + MR - 1) / MR;
    int ntiles = nblocks_m * oh_count;
    int nt = (ntiles > 1)
           ? nnr::int8_compute_threads(ntiles,
                 (int64_t)MM * chunk_sp * CHW) : 1;

    // Pre-build per-oh k_off tables (one per unique oh value).
    // Use pre-computed k_off_oh_all from reshape if available, else build locally.
    std::vector<size_t> k_off_oh_local;
    const size_t* k_off_oh_ptr = nullptr;  // pointer to [oh × K4] table
#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)
    jit_conv_int8_ukernel_fn_t    jit_fns[CONV_INT8_MR_JIT + 1]      = {};
    jit_conv_int8_nr32_ukernel_fn_t jit_nr32_fns[CONV_INT8_MR_JIT_NR32 + 1] = {};
    if (use_jit) {
        for (int mr = 1; mr <= MR; mr++)
            jit_fns[mr] = jit_cache.get_or_create({mr}, mr)->fn<jit_conv_int8_ukernel_fn_t>();
        if (use_jit_nr32)
            for (int mr = 1; mr <= MR; mr++)
                jit_nr32_fns[mr] = jit_nr32_cache.get_or_create({mr}, mr)->fn<jit_conv_int8_nr32_ukernel_fn_t>();

        if (k_off_oh_all) {
            // Use pre-computed table from reshape (indexed by absolute oh)
            k_off_oh_ptr = k_off_oh_all;
        } else {
            // Fallback: build locally for this tile range
            k_off_oh_local.resize((size_t)oh_count * K4);
            for (int oh_rel = 0; oh_rel < oh_count; oh_rel++) {
                size_t oh_off = (size_t)(oh_start + oh_rel) * padW;
                size_t* dst = k_off_oh_local.data() + (size_t)oh_rel * K4;
                for (int k = 0; k < K4; k++)
                    dst[k] = k_off[k] + oh_off;
            }
            k_off_oh_ptr = k_off_oh_local.data() - (size_t)oh_start * K4;  // offset so [oh_start] maps to [0]
        }
    }
#endif

    constexpr int KC = 384;  // K-blocking tile size (matches ORT)
    bool use_kb = (CHW > KC);

    nnr::for_dynamic(0, ntiles, nt, [&](int, int tile) {
        int mb = tile / oh_count;
        int oh_rel = tile % oh_count;
        int i0 = mb * MR;
        int ie = std::min(i0 + MR, MM);
        int nr = ie - i0;
        size_t oh_off = (size_t)(oh_start + oh_rel) * padW;

#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)
        if (use_jit) {
            auto jit_fn = jit_fns[nr];
            const size_t* k_off_oh_base = k_off_oh_ptr + (size_t)(oh_start + oh_rel) * K4;

            alignas(64) int32_t col_sum_buf[32];  // [0..15] col0, [16..31] col1

            // Hoist per-row requantize vectors outside ow0 loop (tile-invariant).
            __m512 vcs_r[CONV_INT8_MR_JIT], vbias_r[CONV_INT8_MR_JIT];
            __m512 vis, vzp_v, vqmin_v, vqmax_v;
            if (rq) {
                vis    = _mm512_set1_ps(rq->inv_y_scale);
                vzp_v  = _mm512_set1_ps(rq->y_zp);
                vqmin_v = _mm512_set1_ps(rq->qmin);
                vqmax_v = _mm512_set1_ps(rq->qmax);
                for (int r = 0; r < nr; r++) {
                    vcs_r[r]   = _mm512_set1_ps(rq->combined_scales[i0 + r]);
                    vbias_r[r] = _mm512_set1_ps(rq->bias_vals ? rq->bias_vals[i0 + r] : 0.0f);
                }
            }

            // Per-tile accumulated column sums for K-blocking (one per ow-chunk)
            // Max ow-chunks: ceil(oW/16) for NR=16, ceil(oW/32) for NR=32
            constexpr int MAX_OW_CHUNKS = 16;  // up to 256 ow positions
            alignas(64) int32_t total_col_sums[MAX_OW_CHUNKS * 32];

            // K-blocking outer loop
            for (int kb = 0; kb < CHW; kb += (use_kb ? KC : CHW)) {
                int kc = use_kb ? std::min(KC, CHW - kb) : CHW;
                int kc4 = (kc + 3) & ~3;
                int kgroups_kc = kc4 / 4;
                int zm = (kb == 0) ? 1 : 0;
                bool last_k = (kb + kc >= CHW);

                const size_t* k_off_oh = k_off_oh_base + kb;
                int ow_chunk_idx = 0;

                for (int ow0 = 0; ow0 < oW; ) {
                    int jw = oW - ow0;
                    if (use_jit_nr32 && jw > 16) {
                        int jw32 = std::min(32, jw);
                        uint16_t m0 = 0xFFFF;
                        uint16_t m1 = (jw32 == 32) ? 0xFFFF : (uint16_t)((1u << (jw32-16))-1);

                        jit_nr32_fns[nr](W + (size_t)i0*CHW + kb, x_pad + ow0,
                            Y + (size_t)i0*chunk_sp + (size_t)oh_rel*oW + ow0,
                            k_off_oh, kgroups_kc, CHW, nr, chunk_sp,
                            col_sum_buf, m0, m1, zm);

                        // Accumulate column sums across KC blocks
                        if (use_kb) {
                            int32_t* ts = total_col_sums + ow_chunk_idx * 32;
                            if (kb == 0) {
                                memcpy(ts, col_sum_buf, 32 * sizeof(int32_t));
                            } else {
                                __m512i a0 = _mm512_load_si512(ts);
                                __m512i a1 = _mm512_load_si512(ts + 16);
                                a0 = _mm512_add_epi32(a0, _mm512_load_si512(col_sum_buf));
                                a1 = _mm512_add_epi32(a1, _mm512_load_si512(col_sum_buf + 16));
                                _mm512_store_si512(ts, a0);
                                _mm512_store_si512(ts + 16, a1);
                            }
                        }

                        if (last_k) {
                            int32_t* cs = use_kb ? (total_col_sums + ow_chunk_idx * 32)
                                                 : col_sum_buf;
                            __m512i cc0 = _mm512_mullo_epi32(v_azp, _mm512_load_si512(cs));
                            __m512i cc1 = _mm512_mullo_epi32(v_azp, _mm512_load_si512(cs + 16));
                            size_t out_j = (size_t)oh_rel*oW + ow0;
                            for (int r = 0; r < nr; r++) {
                                int32_t* yr0 = Y + (size_t)(i0+r)*chunk_sp + out_j;
                                int32_t* yr1 = yr0 + 16;
                                for (int ci = 0; ci < 2; ci++) {
                                    __m512i cc = (ci == 0) ? cc0 : cc1;
                                    __mmask16 sm = (ci == 0) ? (__mmask16)m0 : (__mmask16)m1;
                                    int abs_ow = ow0 + ci*16;
                                    int32_t* yr = (ci == 0) ? yr0 : yr1;
                                    __m512i result = _mm512_maskz_loadu_epi32(sm, yr);
                                    result = _mm512_sub_epi32(result, cc);
                                    result = _mm512_sub_epi32(result, _mm512_set1_epi32(b_zp * row_sum_W[i0+r]));
                                    result = _mm512_add_epi32(result, v_zpp);
                                    if (rq) {
                                        __m512 fv = _mm512_cvtepi32_ps(result);
                                        fv = _mm512_fmadd_ps(fv, vcs_r[r], vbias_r[r]);
                                        fv = _mm512_add_ps(_mm512_roundscale_ps(
                                            _mm512_mul_ps(fv, vis), _MM_FROUND_TO_NEAREST_INT), vzp_v);
                                        fv = _mm512_max_ps(_mm512_min_ps(fv, vqmax_v), vqmin_v);
                                        uint8_t* dst = rq->Y_out + (size_t)(i0+r)*rq->y_out_stride
                                                     + (size_t)(oh_start+oh_rel)*oW + abs_ow;
                                        _mm_mask_storeu_epi8(dst, sm,
                                            _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(fv)));
                                    } else {
                                        _mm512_mask_storeu_epi32(yr, sm, result);
                                    }
                                }
                            }
                        }
                        ow_chunk_idx++;
                        ow0 += 32;
                    } else {
                        int jw16 = std::min(16, jw);
                        uint16_t mask = (jw16 == 16) ? 0xFFFF : (uint16_t)((1u << jw16)-1);
                        jit_fn(W + (size_t)i0*CHW + kb, x_pad + ow0,
                            Y + (size_t)i0*chunk_sp + (size_t)oh_rel*oW + ow0,
                            k_off_oh, kgroups_kc, CHW, nr, chunk_sp, col_sum_buf, mask,
                            zm);

                        // Accumulate column sums across KC blocks
                        if (use_kb) {
                            int32_t* ts = total_col_sums + ow_chunk_idx * 32;
                            if (kb == 0) {
                                memcpy(ts, col_sum_buf, 16 * sizeof(int32_t));
                            } else {
                                __m512i a0 = _mm512_load_si512(ts);
                                a0 = _mm512_add_epi32(a0, _mm512_load_si512(col_sum_buf));
                                _mm512_store_si512(ts, a0);
                            }
                        }

                        if (last_k) {
                            int32_t* cs = use_kb ? (total_col_sums + ow_chunk_idx * 32)
                                                 : col_sum_buf;
                            __m512i col_comp = _mm512_mullo_epi32(v_azp, _mm512_load_si512(cs));
                            __mmask16 store_mask = mask;
                            size_t out_j = (size_t)oh_rel*oW + ow0;
                            for (int r = 0; r < nr; r++) {
                                int32_t* yrow = Y + (size_t)(i0+r)*chunk_sp + out_j;
                                __m512i result = _mm512_maskz_loadu_epi32(store_mask, yrow);
                                result = _mm512_sub_epi32(result, col_comp);
                                result = _mm512_sub_epi32(result, _mm512_set1_epi32(b_zp * row_sum_W[i0+r]));
                                result = _mm512_add_epi32(result, v_zpp);
                                if (rq) {
                                    __m512 fv = _mm512_cvtepi32_ps(result);
                                    fv = _mm512_fmadd_ps(fv, vcs_r[r], vbias_r[r]);
                                    fv = _mm512_add_ps(_mm512_roundscale_ps(
                                        _mm512_mul_ps(fv, vis), _MM_FROUND_TO_NEAREST_INT), vzp_v);
                                    fv = _mm512_max_ps(_mm512_min_ps(fv, vqmax_v), vqmin_v);
                                    uint8_t* dst = rq->Y_out + (size_t)(i0+r)*rq->y_out_stride
                                                 + (size_t)(oh_start+oh_rel)*oW + ow0;
                                    _mm_mask_storeu_epi8(dst, store_mask,
                                        _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(fv)));
                                } else {
                                    _mm512_mask_storeu_epi32(yrow, store_mask, result);
                                }
                            }
                        }
                        ow_chunk_idx++;
                        ow0 += 16;
                    }
                }
            } // end K-blocking loop
            return;
        }
#endif
        // Intrinsics fallback (non-x64 or no Xbyak)
        const uint8_t* aptr[6];
        for (int r = 0; r < nr; r++) aptr[r] = W + (size_t)(i0 + r) * CHW;

        const __m128i v128 = _mm_set1_epi8((char)128);
        __m512i v_ones8 = _mm512_set1_epi8(1);
        __m512i v_ones16 = _mm512_set1_epi16(1);

        // Hoist per-row requantize vectors outside ow0 loop (same as JIT path)
        __m512 vcs_r[CONV_INT8_MR], vbias_r[CONV_INT8_MR];
        __m512 vis_fb, vzp_vfb, vqmin_vfb, vqmax_vfb;
        if (rq) {
            vis_fb    = _mm512_set1_ps(rq->inv_y_scale);
            vzp_vfb   = _mm512_set1_ps(rq->y_zp);
            vqmin_vfb = _mm512_set1_ps(rq->qmin);
            vqmax_vfb = _mm512_set1_ps(rq->qmax);
            for (int r = 0; r < nr; r++) {
                vcs_r[r]   = _mm512_set1_ps(rq->combined_scales[i0 + r]);
                vbias_r[r] = _mm512_set1_ps(rq->bias_vals ? rq->bias_vals[i0 + r] : 0.0f);
            }
        }

        for (int ow0 = 0; ow0 < oW; ow0 += 16) {
            int jw = std::min(16, oW - ow0);
            __mmask16 mask = (jw == 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << jw) - 1);

            __m512i acc[6];
            for (int r = 0; r < nr; r++) acc[r] = _mm512_setzero_si512();
            __m512i col_sum = _mm512_setzero_si512();

            for (int kg = 0; kg < Kgroups; kg++) {
                int k0 = kg * 4;

                __m128i r0 = _mm_maskz_loadu_epi8(mask, x_pad + k_off[k0  ] + oh_off + ow0);
                __m128i r1 = _mm_maskz_loadu_epi8(mask, x_pad + k_off[k0+1] + oh_off + ow0);
                __m128i r2 = _mm_maskz_loadu_epi8(mask, x_pad + k_off[k0+2] + oh_off + ow0);
                __m128i r3 = _mm_maskz_loadu_epi8(mask, x_pad + k_off[k0+3] + oh_off + ow0);

                if (k0 + 4 <= CHW) {
                    r0 = _mm_sub_epi8(r0, v128);
                    r1 = _mm_sub_epi8(r1, v128);
                    r2 = _mm_sub_epi8(r2, v128);
                    r3 = _mm_sub_epi8(r3, v128);
                } else {
                    r0 = (k0   < CHW) ? _mm_sub_epi8(r0, v128) : _mm_setzero_si128();
                    r1 = (k0+1 < CHW) ? _mm_sub_epi8(r1, v128) : _mm_setzero_si128();
                    r2 = (k0+2 < CHW) ? _mm_sub_epi8(r2, v128) : _mm_setzero_si128();
                    r3 = (k0+3 < CHW) ? _mm_sub_epi8(r3, v128) : _mm_setzero_si128();
                }

                __m128i lo01 = _mm_unpacklo_epi8(r0, r1);
                __m128i hi01 = _mm_unpackhi_epi8(r0, r1);
                __m128i lo23 = _mm_unpacklo_epi8(r2, r3);
                __m128i hi23 = _mm_unpackhi_epi8(r2, r3);
                __m128i q0 = _mm_unpacklo_epi16(lo01, lo23);
                __m128i q1 = _mm_unpackhi_epi16(lo01, lo23);
                __m128i q2 = _mm_unpacklo_epi16(hi01, hi23);
                __m128i q3 = _mm_unpackhi_epi16(hi01, hi23);
                __m512i vnni = _mm512_inserti32x4(
                    _mm512_inserti32x4(
                        _mm512_inserti32x4(_mm512_castsi128_si512(q0), q1, 1),
                        q2, 2),
                    q3, 3);

                col_sum = _mm512_add_epi32(col_sum,
                    _mm512_madd_epi16(_mm512_maddubs_epi16(v_ones8, vnni), v_ones16));

                for (int r = 0; r < nr; r++) {
                    __m512i va = _mm512_set1_epi32(*(const uint32_t*)(aptr[r] + k0));
                    acc[r] = _mm512_dpbusd_epi32(acc[r], va, vnni);
                }
            }

            __m512i col_comp = _mm512_mullo_epi32(v_azp, col_sum);
            size_t out_j = (size_t)oh_rel * oW + ow0;
            for (int r = 0; r < nr; r++) {
                __m512i result = _mm512_sub_epi32(acc[r], col_comp);
                result = _mm512_sub_epi32(result, _mm512_set1_epi32(b_zp * row_sum_W[i0 + r]));
                result = _mm512_add_epi32(result, v_zpp);
                if (rq) {
                    __m512 fv = _mm512_cvtepi32_ps(result);
                    fv = _mm512_fmadd_ps(fv, vcs_r[r], vbias_r[r]);
                    fv = _mm512_add_ps(_mm512_roundscale_ps(
                        _mm512_mul_ps(fv, vis_fb), _MM_FROUND_TO_NEAREST_INT), vzp_vfb);
                    fv = _mm512_max_ps(_mm512_min_ps(fv, vqmax_vfb), vqmin_vfb);
                    uint8_t* dst = rq->Y_out
                        + (size_t)(i0 + r) * rq->y_out_stride
                        + (size_t)(oh_start + oh_rel) * oW + ow0;
                    _mm_mask_storeu_epi8(dst, mask, _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(fv)));
                } else {
                    _mm512_mask_storeu_epi32(Y + (size_t)(i0 + r) * chunk_sp + out_j, mask, result);
                }
            }
        }
    });
}

// ── Gather-GEMM: pre-packed weight functions ─────────────────────────
// Pack weights W[OC × K] (int8, row-major) into NR=48-blocked VNNI format.
// Layout: packed[nb * Kgroups * 192 + kg * 192 + ch * 4 + sub]
//        = W[(nb*48 + ch)][kg*4 + sub]
// ldb = 192 bytes per K-group (constant, sequential access within NR-block).

// @nnr-meta isa=scalar dtype=int8 layout=NCHW special=GEMM
inline size_t pack_weights_gather_nr48_size(int OC, int K) {
    int K4 = (K + 3) & ~3;
    int Kgroups = K4 / 4;
    int Nblocks = (OC + 47) / 48;
    return (size_t)Nblocks * Kgroups * 192;
}

// @nnr-meta isa=scalar dtype=int8 layout=NCHW special=GEMM
inline void pack_weights_gather_nr48(
    int8_t* __restrict dst, int32_t* __restrict col_sums,
    const int8_t* __restrict W, int OC, int K)
{
    int K4 = (K + 3) & ~3;
    int Kgroups = K4 / 4;
    constexpr int NR = 48;
    int Nblocks = (OC + NR - 1) / NR;

    memset(dst, 0, (size_t)Nblocks * Kgroups * 192);

    for (int nb = 0; nb < Nblocks; nb++) {
        for (int kg = 0; kg < Kgroups; kg++) {
            int8_t* blk = dst + (size_t)nb * Kgroups * 192 + kg * 192;
            for (int c = 0; c < NR; c++) {
                int oc = nb * NR + c;
                if (oc >= OC) break;
                for (int sub = 0; sub < 4; sub++) {
                    int k = kg * 4 + sub;
                    blk[c * 4 + sub] = (k < K) ? W[(size_t)oc * K + k] : 0;
                }
            }
        }
    }

    // Column sums: signed sum of int8 weights per output channel
    for (int oc = 0; oc < OC; oc++) {
        int32_t s = 0;
        for (int k = 0; k < K; k++)
            s += (int32_t)W[(size_t)oc * K + k];
        col_sums[oc] = s;
    }
}

// ---- NR=48 packed GEMM (3-sub-panel format) ----
// B is packed into NR=48 blocks, each with 3 sub-panels of 16 columns.
// Sub-panel layout: [Kgroups × 64] bytes (16 cols × 4 bytes per K-group).
// NR-block layout: [sub0][sub1][sub2], each sub_stride bytes apart.
// JIT micro-kernels process 3 zmm registers (48 cols) per VPDPBUSD.

// @nnr-meta isa=scalar dtype=[int8,uint8] layout=NCHW special=GEMM
inline size_t pack_b_int8_nr48_sub_stride(int K) {
    return (size_t)((K + 3) / 4) * 64;
}

// @nnr-meta isa=scalar dtype=[int8,uint8] layout=NCHW special=GEMM
inline size_t pack_b_int8_nr48_size(int K, int N) {
    return (size_t)((N + 47) / 48) * 3 * pack_b_int8_nr48_sub_stride(K);
}

// @nnr-meta isa=scalar dtype=[int8,uint8] layout=NCHW special=GEMM
inline void pack_b_int8_nr48_and_col_sums(
    int8_t* __restrict dst, int32_t* __restrict col_sum,
    const int8_t* __restrict B, int K, int N)
{
    constexpr int NR = 48;
    int K4 = (K + 3) & ~3, Kgroups = K4 / 4;
    int nb = (N + NR - 1) / NR;
    size_t sub_stride = (size_t)Kgroups * 64;

    memset(dst, 0, (size_t)nb * 3 * sub_stride);
    memset(col_sum, 0, (size_t)N * sizeof(int32_t));

    for (int jb = 0; jb < nb; jb++) {
        int j0 = jb * NR;
        uint8_t* panel_base = (uint8_t*)dst + (size_t)jb * 3 * sub_stride;
        for (int s = 0; s < 3; s++) {
            uint8_t* sub = panel_base + s * sub_stride;
            int col_start = j0 + s * 16;
            int col_count = std::min(16, N - col_start);
            if (col_count <= 0) continue;
            for (int kg = 0; kg < Kgroups; kg++)
                for (int c = 0; c < col_count; c++)
                    for (int sb = 0; sb < 4; sb++) {
                        int k = kg * 4 + sb;
                        int8_t val = (k < K) ? B[(size_t)k * N + col_start + c] : 0;
                        sub[kg * 64 + c * 4 + sb] = (uint8_t)val;
                        if (k < K) col_sum[col_start + c] += val;
                    }
        }
    }
}

// NR=48 tiled GEMM with JIT micro-kernels.
// packed_B must be in NR=48 3-sub-panel format (from pack_b_int8_nr48_and_col_sums).
// ldc: C row stride (must be >= N, rounded up to 16 for safe zmm tail writes).
// Uses MC/KC tiling for A to keep working set in L1/L2.
// @nnr-meta isa=[AVX512,AVX512_VNNI] dtype=[int8,uint8] layout=NCHW special=[GEMM,JIT] tiling=[K,MR,NR] fusion=qdq
inline void gemm_int8_nr48(int M, int N, int K,
    const uint8_t* __restrict A, int a_zp,
    const int8_t* __restrict packed_B_nr48, int b_zp,
    const int32_t* __restrict col_sum_B,
    const int32_t* __restrict row_sum_A,
    int32_t* __restrict C, int ldc = 0)
{
    if (ldc == 0) ldc = (N + 15) & ~15;  // default: pad to 16-column boundary
#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)
    constexpr int MR = 6, NR = 48, MC = 48, KC = 384;
    int K4 = (K + 3) & ~3;
    size_t full_sub_stride = pack_b_int8_nr48_sub_stride(K);

    // JIT kernel caches (thread-safe, created once)
    static jit_cache_t<int, jit_packed_gemm_nr48_avx512_t> cache48;
    static jit_cache_t<int, jit_packed_gemm_partial16_avx512_t> cache16;
    static jit_cache_t<int, jit_packed_gemm_partial32_avx512_t> cache32;

    jit_packed_gemm_nr48_fn_t fn48[MR + 1] = {};
    jit_packed_gemm_partial_fn_t fn16[MR + 1] = {};
    jit_packed_gemm_partial_fn_t fn32[MR + 1] = {};
    for (int mr = 1; mr <= MR; mr++) {
        fn48[mr] = cache48.get_or_create(mr, mr)->fn<jit_packed_gemm_nr48_fn_t>();
        fn16[mr] = cache16.get_or_create(mr, mr)->fn<jit_packed_gemm_partial_fn_t>();
        fn32[mr] = cache32.get_or_create(mr, mr)->fn<jit_packed_gemm_partial_fn_t>();
    }

    int num_full_nr = N / NR;
    int nr_rem = N % NR;
    int nr_tail = (nr_rem > 16) ? 2 : (nr_rem > 0) ? 1 : 0;

    int nblocks_m = (M + MC - 1) / MC;
    int nt = (nblocks_m > 1)
           ? nnr::int8_compute_threads(nblocks_m, (int64_t)M * N * K) : 1;
    int32_t zp_product = (int32_t)K * a_zp * b_zp;

    for (int kb = 0; kb < K; kb += KC) {
        int kc = std::min(KC, K - kb);
        int kc4 = ((kc + 3) & ~3);
        int kgroups = kc4 / 4;
        int kb_group = kb / 4;
        int zm = (kb == 0) ? 1 : 0;
        bool last_k = (kb + KC >= K);

        // B tile: offset into packed B at the right K-group
        const uint8_t* b_tile = (const uint8_t*)packed_B_nr48
            + (size_t)kb_group * 64;

        nnr::for_dynamic(0, nblocks_m, nt, [&](int /*tid*/, int mb) {
            // Per-thread pack_a buffer on the stack
            alignas(64) uint8_t packed_A[MC * ((KC + 3) & ~3)];

            int ib = mb * MC;
            int mc = std::min(MC, M - ib);

            // Pack A: copy rows [ib..ib+mc), columns [kb..kb+kc), pad to kc4
            for (int m = 0; m < mc; m++) {
                uint8_t* out = packed_A + (size_t)m * kc4;
                memcpy(out, A + (size_t)(ib + m) * K + kb, kc);
                if (kc4 > kc) memset(out + kc, 0, kc4 - kc);
            }

            for (int i = 0; i < mc; i += MR) {
                int mr = std::min(MR, mc - i);
                int32_t* c_ptr = C + (size_t)(ib + i) * ldc;

                if (num_full_nr > 0 || nr_tail > 0) {
                    fn48[mr](packed_A + (size_t)i * kc4, b_tile, c_ptr,
                        kgroups, kc4, ldc, full_sub_stride, zm,
                        num_full_nr, nr_tail, nullptr);
                }
                // Handle cols 33-47 of remainder (nr_rem > 32)
                if (nr_rem > 32) {
                    const uint8_t* b_last = b_tile
                        + (size_t)(num_full_nr * 3 + 2) * full_sub_stride;
                    fn16[mr](packed_A + (size_t)i * kc4, b_last,
                        c_ptr + num_full_nr * NR + 32,
                        kgroups, kc4, ldc, full_sub_stride, zm);
                }
            }

            // Zero-point correction after last K iteration (per M-block)
            if (last_k) {
                __m512i v_azp = _mm512_set1_epi32(-(int32_t)a_zp);
                for (int i = ib; i < ib + mc; i++) {
                    int32_t row_corr = -(int32_t)b_zp * row_sum_A[i] + zp_product;
                    __m512i v_rc = _mm512_set1_epi32(row_corr);
                    int32_t* crow = C + (size_t)i * ldc;
                    int j = 0;
                    for (; j + 16 <= N; j += 16) {
                        __m512i cv = _mm512_loadu_si512(crow + j);
                        __m512i cs = _mm512_loadu_si512(col_sum_B + j);
                        cv = _mm512_add_epi32(cv, _mm512_mullo_epi32(v_azp, cs));
                        cv = _mm512_add_epi32(cv, v_rc);
                        _mm512_storeu_si512(crow + j, cv);
                    }
                    for (; j < N; j++)
                        crow[j] += -(int32_t)a_zp * col_sum_B[j] + row_corr;
                }
            }
        });
    }
#else
    // Fallback to NR=32 path when Xbyak is not available
    gemm_int8(M, N, K, A, a_zp, packed_B_nr48, b_zp, col_sum_B, row_sum_A, C);
#endif
}

// ---- Conv NR=48 weight packing with (kh,kw,ic) K-order for NHWC input ----
// Reorders K dimension from (ic,kh,kw) to (kh,kw,ic) so pack_a can do contiguous
// IC-byte copies from NHWC-padded input instead of byte-by-byte gather.
// W is [OC, IC*kH*kW] in original (ic,kh,kw) order.

// @nnr-meta isa=scalar dtype=[int8,uint8] layout=NHWC special=GEMM
inline void pack_weights_nr48_nhwc(
    int8_t* __restrict dst, int32_t* __restrict col_sums,
    const int8_t* __restrict W, int OC, int IC, int kH, int kW)
{
    int K = IC * kH * kW;
    constexpr int NR = 48;
    int K4 = (K + 3) & ~3;
    int Kgroups = K4 / 4;
    int Nblocks = (OC + NR - 1) / NR;
    size_t sub_stride = (size_t)Kgroups * 64;

    memset(dst, 0, (size_t)Nblocks * 3 * sub_stride);

    // Build permutation: k_new = kh*kW*IC + kw*IC + c, k_old = c*kH*kW + kh*kW + kw
    std::vector<int> perm(K);
    for (int kh = 0; kh < kH; kh++)
        for (int kw = 0; kw < kW; kw++)
            for (int c = 0; c < IC; c++)
                perm[kh * kW * IC + kw * IC + c] = c * kH * kW + kh * kW + kw;

    for (int nb = 0; nb < Nblocks; nb++) {
        for (int s = 0; s < 3; s++) {
            uint8_t* sub = (uint8_t*)dst + (size_t)nb * 3 * sub_stride + s * sub_stride;
            int col_start = nb * NR + s * 16;
            int col_count = std::min(16, OC - col_start);
            if (col_count <= 0) continue;
            for (int kg = 0; kg < Kgroups; kg++) {
                for (int c = 0; c < col_count; c++) {
                    int oc = col_start + c;
                    for (int sb = 0; sb < 4; sb++) {
                        int k_new = kg * 4 + sb;
                        if (k_new < K) {
                            int k_old = perm[k_new];
                            sub[kg * 64 + c * 4 + sb] = (uint8_t)W[(size_t)oc * K + k_old];
                        }
                    }
                }
            }
        }
    }

    // Column sums (order-independent — just sum all K weights per OC)
    for (int oc = 0; oc < OC; oc++) {
        int32_t s = 0;
        for (int k = 0; k < K; k++)
            s += (int32_t)W[(size_t)oc * K + k];
        col_sums[oc] = s;
    }
}

// ---- Conv using packed NR=48 GEMM (ORT-style: A=input, B=weights) ----
// Weights pre-packed in sub-stride NR=48 format via pack_weights_nr48_nhwc().
// Input is raw NHWC [iH × iW × IC] — padding handled inline in pack_a via
// bounds checking (eliminates the serial NHWC pad copy).
// M=spatial (oH*oW), N=OC, K=IC*kH*kW.

// @nnr-meta isa=scalar dtype=[int8,uint8] layout=NCHW special=GEMM
inline size_t pack_weights_nr48_substride_size(int OC, int K) {
    return pack_b_int8_nr48_size(K, OC);  // same total size, just different addressing
}

// Pack weights [OC × K] row-major (int8) into sub-stride NR=48 format.
// W[oc, k] → B panel for NR=48 JIT kernel. col_sums[oc] = sum_k(W[oc,k]).
// This is the B-side packing: N=OC dimension is blocked into NR=48.
// @nnr-meta isa=scalar dtype=[int8,uint8] layout=NHWC special=GEMM
inline void pack_weights_nr48_substride(
    int8_t* __restrict dst, int32_t* __restrict col_sums,
    const int8_t* __restrict W, int OC, int K)
{
    constexpr int NR = 48;
    int K4 = (K + 3) & ~3;
    int Kgroups = K4 / 4;
    int Nblocks = (OC + NR - 1) / NR;
    size_t sub_stride = (size_t)Kgroups * 64;

    memset(dst, 0, (size_t)Nblocks * 3 * sub_stride);

    for (int nb = 0; nb < Nblocks; nb++) {
        for (int s = 0; s < 3; s++) {
            uint8_t* sub = (uint8_t*)dst + (size_t)nb * 3 * sub_stride + s * sub_stride;
            int col_start = nb * NR + s * 16;
            int col_count = std::min(16, OC - col_start);
            if (col_count <= 0) continue;
            for (int kg = 0; kg < Kgroups; kg++) {
                for (int c = 0; c < col_count; c++) {
                    int oc = col_start + c;
                    for (int sb = 0; sb < 4; sb++) {
                        int k = kg * 4 + sb;
                        sub[kg * 64 + c * 4 + sb] = (k < K) ? (uint8_t)W[(size_t)oc * K + k] : 0;
                    }
                }
            }
        }
    }

    // Column sums
    for (int oc = 0; oc < OC; oc++) {
        int32_t s = 0;
        for (int k = 0; k < K; k++)
            s += (int32_t)W[(size_t)oc * K + k];
        col_sums[oc] = s;
    }
}

// Conv using pre-packed NR=48 GEMM with NHWC input (ORT-style).
// x_pad_nhwc: pre-padded input [padH × padW × IC_pad], uint8, filled with x_zp in padding.
//   IC_pad = (IC+3)&~3 for VNNI alignment.
// packed_W: weights in sub-stride NR=48 format with (kh,kw,ic) K-order
//   (from pack_weights_nr48_nhwc).
// w_col_sums: per-OC column sums of weights (for zero-point correction).
// rq: requantization params (if non-null, write uint8 output directly).
//
// SkipMemcpyStores=true: PROBE-ONLY (incorrect output). Skips the byte stores
// in pack_a to measure the upper bound of a hypothetical memcpy-free kernel
// where the JIT GEMM reads x_nhwc directly via offsets. Default false → no
// behavior change for production callers.
template <bool SkipMemcpyStores = false>
// @nnr-meta isa=AVX512 dtype=[int8,uint8] layout=NHWC special=[GEMM,JIT] tiling=[K,MR,NR] fusion=qdq
inline void conv_int8_packed_nr48(
    int OC, int oH, int oW, int IC, int kH, int kW,
    int pH, int pW, int sH, int sW, int dH, int dW,
    const uint8_t* __restrict x_nhwc,
    int iH, int iW,
    int x_zp,
    const int8_t* __restrict packed_W,
    const int32_t* __restrict w_col_sums,
    int w_zp,
    const conv_rq_params_t* rq)
{
#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)
    constexpr int MR = 6, NR = 48, MC = 48, KC = 384;
    int K = IC * kH * kW;
    int K4 = (K + 3) & ~3;
    int spatial = oH * oW;
    size_t full_sub_stride = pack_b_int8_nr48_sub_stride(K);

    // JIT kernel caches
    static jit_cache_t<int, jit_packed_gemm_nr48_avx512_t> cache48;
    static jit_cache_t<int, jit_packed_gemm_partial16_avx512_t> cache16;
    static jit_cache_t<int, jit_packed_gemm_partial32_avx512_t> cache32;

    jit_packed_gemm_nr48_fn_t fn48[MR + 1] = {};
    jit_packed_gemm_partial_fn_t fn16[MR + 1] = {};
    jit_packed_gemm_partial_fn_t fn32[MR + 1] = {};
    for (int mr = 1; mr <= MR; mr++) {
        fn48[mr] = cache48.get_or_create(mr, mr)->fn<jit_packed_gemm_nr48_fn_t>();
        fn16[mr] = cache16.get_or_create(mr, mr)->fn<jit_packed_gemm_partial_fn_t>();
        fn32[mr] = cache32.get_or_create(mr, mr)->fn<jit_packed_gemm_partial_fn_t>();
    }

    int nblocks_m = (spatial + MC - 1) / MC;
    int32_t zp_product = (int32_t)K * x_zp * w_zp;
    // row_sums are only consumed via row_corr = -w_zp * row_sums + zp_product.
    // When w_zp == 0 (all symmetric-int8 weight models), row_corr is 0
    // regardless of x_zp, so we can skip computing row_sums in pack_a entirely.
    // Saves ~33% of pack_a cost on Block 1 shapes (sad_epu8 + reduce path).
    const bool need_row_sums = (w_zp != 0);

    // M×N partitioning: add N-blocking when M-blocks can't saturate threads
    int nthreads = nnr::thread_pool_t::get().num_threads();
    int64_t total_ops = (int64_t)spatial * OC * K;
    int NC = OC, nblocks_n = 1;
    // N-blocking: partition output channels when M-blocks can't saturate threads.
    // Guards: need enough M-blocks (>=3) and enough total work (>16M ops) to
    // justify the overhead of redundant pack_A across N-slices.
    if (nblocks_m < nthreads && nblocks_m >= 3 && OC > NR && total_ops > (1 << 24)) {
        int need_n = (nthreads + nblocks_m - 1) / nblocks_m;
        NC = (OC / (need_n * NR)) * NR;
        if (NC < NR) NC = NR;
        nblocks_n = (OC + NC - 1) / NC;
    }
    int total_blocks = nblocks_m * nblocks_n;
    int nt = (total_blocks > 1)
           ? nnr::int8_compute_threads(total_blocks, total_ops) : 1;

    // Precompute kernel-pixel dilation offsets
    int kPixels = kH * kW;
    int kp_dh[49], kp_dw[49]; // max 7×7 kernel, no heap alloc
    for (int kh = 0; kh < kH; kh++)
        for (int kw = 0; kw < kW; kw++) {
            kp_dh[kh * kW + kw] = kh * dH;
            kp_dw[kh * kW + kw] = kw * dW;
        }

    // Threading over M×N blocks with K-loop inside each thread.
    // When M-blocks < nthreads, N-blocking adds tiles for better utilization.
    // for_static: pre-assigned blocks, no atomic contention, better cache locality.
    nnr::for_static(0, total_blocks, nt, [&](int block) {
        int mb = block / nblocks_n;  // mb-major: group N-slices of same M-block
        int nb = block % nblocks_n;

        // N-slice parameters
        int n_start = nb * NC;
        int n_count = std::min(NC, OC - n_start);
        int n_padded = (n_count + 15) & ~15;
        int slice_full_nr = n_count / NR;
        int slice_nr_rem = n_count % NR;
        int slice_nr_tail = (slice_nr_rem > 16) ? 2 : (slice_nr_rem > 0) ? 1 : 0;

        // Per-thread pack_a buffer + row sums on stack
        alignas(64) uint8_t packed_A[MC * ((KC + 3) & ~3)];
        alignas(64) int32_t row_sums[MC];
        // Per-thread C accumulator sized for N-slice
        alignas(64) int32_t C_stack[MC * ((512 + 15) & ~15)];
        int32_t* C_local = (n_padded <= 528) ? C_stack
            : (int32_t*)_aligned_malloc((size_t)MC * n_padded * sizeof(int32_t), 64);

        int ib = mb * MC;
        int mc = std::min(MC, spatial - ib);
        int ldc_local = n_padded;

        for (int kb = 0; kb < K; kb += KC) {
            int kc = std::min(KC, K - kb);
            int kc4 = ((kc + 3) & ~3);
            int kgroups = kc4 / 4;
            int kb_group = kb / 4;
            int zm = (kb == 0) ? 1 : 0;
            bool last_k = (kb + KC >= K);

            const uint8_t* b_tile = (const uint8_t*)packed_W
                + (size_t)(n_start / NR) * 3 * full_sub_stride
                + (size_t)kb_group * 64;

            // Decompose kb into kernel-pixel and IC offset:
            // K is ordered (kh, kw, ic), so kb maps to (kp_start, ic_start)
            int kp_start = kb / IC;
            int ic_start = kb % IC;
            // Single-pixel fast path: entire KC block from one kernel pixel
            bool single_pixel = (ic_start + kc <= IC);

            // Pack A directly from raw NHWC input with inline padding.
            // For each spatial position (oh,ow) and kernel pixel (kh,kw),
            // compute the source coordinate and check bounds against input dims.
            // Interior positions (>99% for typical pad=1) always pass; the branch
            // is near-perfectly predicted. Eliminates the serial NHWC pad copy.
            int oh = ib / oW, ow = ib % oW;
            for (int m = 0; m < mc; m++) {
                uint8_t* out = packed_A + (size_t)m * kc4;

                if (single_pixel) {
                    int src_h = oh * sH + kp_dh[kp_start] - pH;
                    int src_w = ow * sW + kp_dw[kp_start] - pW;

                    if ((unsigned)src_h < (unsigned)iH && (unsigned)src_w < (unsigned)iW) {
                        // In-bounds: read from raw input + (optional) fused row sum
                        const uint8_t* src = x_nhwc + ((size_t)src_h * iW + src_w) * IC + ic_start;
                        int32_t rsum = 0;
                        int k = 0;
                        __m512i vzero = _mm512_setzero_si512();
                        __m512i vsum = _mm512_setzero_si512();
                        for (; k + 64 <= kc; k += 64) {
                            __m512i v = _mm512_loadu_si512(src + k);
                            if constexpr (!SkipMemcpyStores)
                                _mm512_storeu_si512(out + k, v);
                            if (need_row_sums)
                                vsum = _mm512_add_epi64(vsum, _mm512_sad_epu8(v, vzero));
                        }
                        if (k < kc) {
                            if constexpr (!SkipMemcpyStores)
                                memcpy(out + k, src + k, kc - k);
                            if (need_row_sums)
                                for (int t = k; t < kc; t++)
                                    rsum += (int32_t)src[t];
                        }
                        if constexpr (!SkipMemcpyStores)
                            for (int t = kc; t < kc4; t++)
                                out[t] = 0;
                        if (need_row_sums) {
                            rsum += (int32_t)_mm512_reduce_add_epi64(vsum);
                            if (kb == 0) row_sums[m] = rsum;
                            else         row_sums[m] += rsum;
                        }
                    } else {
                        // Padding region: fill with x_zp
                        if constexpr (!SkipMemcpyStores) {
                            memset(out, (uint8_t)x_zp, kc);
                            for (int t = kc; t < kc4; t++)
                                out[t] = 0;
                        }
                        if (need_row_sums) {
                            int32_t rsum = kc * x_zp;
                            if (kb == 0) row_sums[m] = rsum;
                            else         row_sums[m] += rsum;
                        }
                    }
                } else {
                    // Multi-pixel path with inline bounds checking
                    int written = 0;
                    int kp = kp_start;
                    int ic = ic_start;
                    while (written < kc && kp < kPixels) {
                        int avail = IC - ic;
                        int need = kc - written;
                        int n = (avail < need) ? avail : need;

                        int src_h = oh * sH + kp_dh[kp] - pH;
                        int src_w = ow * sW + kp_dw[kp] - pW;

                        if constexpr (!SkipMemcpyStores) {
                            if ((unsigned)src_h < (unsigned)iH && (unsigned)src_w < (unsigned)iW) {
                                const uint8_t* src = x_nhwc + ((size_t)src_h * iW + src_w) * IC + ic;
                                memcpy(out + written, src, n);
                            } else {
                                memset(out + written, (uint8_t)x_zp, n);
                            }
                        }
                        written += n;
                        ic = 0;
                        kp++;
                    }
                    if constexpr (!SkipMemcpyStores)
                        for (int t = written; t < kc4; t++)
                            out[t] = 0;
                    // Row sum (separate pass) — skip when w_zp == 0
                    if (need_row_sums) {
                        int32_t rsum = 0;
                        int k = 0;
                        __m512i vzero = _mm512_setzero_si512();
                        __m512i vsum = _mm512_setzero_si512();
                        for (; k + 64 <= kc4; k += 64) {
                            __m512i v = _mm512_loadu_si512(out + k);
                            vsum = _mm512_add_epi64(vsum, _mm512_sad_epu8(v, vzero));
                        }
                        rsum = (int32_t)_mm512_reduce_add_epi64(vsum);
                        for (; k < kc; k++)
                            rsum += (int32_t)out[k];
                        if (kb == 0) row_sums[m] = rsum;
                        else         row_sums[m] += rsum;
                    }
                }

                if (++ow >= oW) { ow = 0; oh++; }
            }
            for (int i = 0; i < mc; i += MR) {
                int mr = std::min(MR, mc - i);
                int32_t* c_ptr = C_local + (size_t)i * ldc_local;

                // Fill JIT requantize params for last K-block
                jit_rq_params_t rq_jit;
                jit_rq_params_t* rq_ptr = nullptr;
                if (last_k && rq) {
                    bool has_zp = (x_zp != 0 || w_zp != 0);
                    rq_jit.output_scales = rq->output_scales + n_start;
                    rq_jit.bias_int32    = rq->bias_int32 ? rq->bias_int32 + n_start : nullptr;
                    rq_jit.w_col_sums    = (x_zp != 0) ? w_col_sums + n_start : nullptr;
                    rq_jit.x_zp_neg      = -(int32_t)x_zp;
                    rq_jit.y_zp_int      = rq->y_zp_int;
                    rq_jit.rq_qmin       = rq->rq_qmin;
                    rq_jit.rq_qmax       = rq->rq_qmax;
                    for (int r = 0; r < mr; r++) {
                        int m = i + r;
                        int sp = ib + m;
                        rq_jit.row_corr[r] = has_zp
                            ? -(int32_t)w_zp * row_sums[m] + zp_product : 0;
                        rq_jit.y_rows[r] = rq->Y_out + (size_t)sp * rq->y_out_stride + n_start;
                    }
                    rq_ptr = &rq_jit;
                }

                if (slice_full_nr > 0 || slice_nr_tail > 0) {
                    fn48[mr](packed_A + (size_t)i * kc4, b_tile, c_ptr,
                        kgroups, kc4, ldc_local, full_sub_stride, zm,
                        slice_full_nr, slice_nr_tail, rq_ptr);
                }
                if (slice_nr_rem > 32) {
                    const uint8_t* b_last = b_tile
                        + (size_t)(slice_full_nr * 3 + 2) * full_sub_stride;
                    fn16[mr](packed_A + (size_t)i * kc4, b_last,
                        c_ptr + slice_full_nr * NR + 32,
                        kgroups, kc4, ldc_local, full_sub_stride, zm);
                }

                // Scalar fallback for fn16 columns (not handled by JIT requantize)
                if (last_k && rq && slice_nr_rem > 32) {
                    int fn48_cols = slice_full_nr * NR + slice_nr_tail * 16;
                    for (int r = 0; r < mr; r++) {
                        int m = i + r;
                        int sp = ib + m;
                        bool has_zp = (x_zp != 0 || w_zp != 0);
                        int32_t row_corr = has_zp
                            ? -(int32_t)w_zp * row_sums[m] + zp_product : 0;
                        int32_t* crow = c_ptr + (size_t)r * ldc_local;
                        uint8_t* y_row = rq->Y_out + (size_t)sp * rq->y_out_stride + n_start;
                        for (int j = fn48_cols; j < n_count; j++) {
                            int32_t v = crow[j];
                            if (x_zp != 0) v += -(int32_t)x_zp * w_col_sums[n_start + j];
                            if (has_zp) v += row_corr;
                            if (rq->bias_int32) v += rq->bias_int32[n_start + j];
                            float fv = (float)v * rq->output_scales[n_start + j];
                            fv = std::max(fv, rq->rq_qmin);
                            fv = std::min(fv, rq->rq_qmax);
                            int32_t ri = (int32_t)std::roundf(fv) + rq->y_zp_int;
                            y_row[j] = (uint8_t)std::min(255, std::max(0, ri));
                        }
                    }
                }
            }

        } // for kb (KC blocks)

        if (C_local != C_stack)
            _aligned_free(C_local);
    });
#endif
}

} // namespace nnr::int8
