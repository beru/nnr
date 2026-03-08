#pragma once
// Fused Scaled Dot-Product Attention (SDPA) — AVX-512.
//
// Computes: Output = Softmax(Q × K^T) × V
// where Q, K, V are [batch, seq_len, head_dim] (head_dim typically 64).
//
// Tiled implementation: processes BR query rows at a time to keep the
// attention score tile [BR × seq_len] in L2 cache instead of materializing
// the full [seq_len × seq_len] matrix (~9MB for seq_len=1500).

#ifdef NNR_ARCH_X64
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include "simd_math_avx512.h"
#include "backend/cpu/kernel/post_ops.h"
#include "backend/cpu/kernel/gemm.h"  // gemm_post_t
#include "gemm_avx512.h"

namespace nnr {

// Process one attention head: Q[S,D] × K[S,D]^T → Softmax → × V[S,D] → O[S,D]
// Q is pre-scaled (already multiplied by 1/sqrt(dk)).
// K is NOT transposed — the kernel handles the transpose internally via GEMM.
// @nnr-meta isa=AVX512 dtype=fp32 tiling=[K,spatial]
inline void sdpa_head_avx512(
    const float* Q,   // [seq_len, head_dim]
    const float* K,   // [seq_len, head_dim]  (NOT transposed)
    const float* V,   // [seq_len, head_dim]
    float* O,         // [seq_len, head_dim]
    float* scratch,   // workspace: at least seq_len * head_dim + BR * seq_len floats
    int seq_len,
    int head_dim)
{
    constexpr int BR = 64;  // query tile size

    // Transpose K → K^T [head_dim, seq_len] in scratch
    float* KT = scratch;
    for (int j = 0; j < seq_len; j++)
        for (int d = 0; d < head_dim; d++)
            KT[d * seq_len + j] = K[j * head_dim + d];

    float* S = scratch + (size_t)head_dim * seq_len; // [BR × seq_len]

    for (int i0 = 0; i0 < seq_len; i0 += BR) {
        int br = std::min(BR, seq_len - i0);

        // --- Phase 1: S[br,S] = Q_tile[br,D] × K^T[D,S] ---
        // Use avx512::dgemm directly (not dgemm_generic) to avoid nested threading.
        avx512::dgemm(br, seq_len, head_dim,
                      Q + (size_t)i0 * head_dim, KT, S, gemm_post_t{});

        // --- Phase 2: Softmax each row of S ---
        for (int qi = 0; qi < br; qi++) {
            float* s_row = S + (size_t)qi * seq_len;
            // Find max
            __m512 vmax = _mm512_set1_ps(-1e30f);
            int j = 0;
            for (; j + 16 <= seq_len; j += 16)
                vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(s_row + j));
            float row_max = _mm512_reduce_max_ps(vmax);
            for (; j < seq_len; j++) row_max = (s_row[j] > row_max) ? s_row[j] : row_max;

            // exp and sum
            __m512 vmx = _mm512_set1_ps(row_max);
            __m512 vsum = _mm512_setzero_ps();
            j = 0;
            for (; j + 16 <= seq_len; j += 16) {
                __m512 e = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(s_row + j), vmx));
                _mm512_storeu_ps(s_row + j, e);
                vsum = _mm512_add_ps(vsum, e);
            }
            float sum = _mm512_reduce_add_ps(vsum);
            for (; j < seq_len; j++) {
                float e = expf(s_row[j] - row_max);
                s_row[j] = e;
                sum += e;
            }

            // Normalize
            float inv_sum = 1.0f / sum;
            __m512 vinv = _mm512_set1_ps(inv_sum);
            j = 0;
            for (; j + 16 <= seq_len; j += 16)
                _mm512_storeu_ps(s_row + j, _mm512_mul_ps(_mm512_loadu_ps(s_row + j), vinv));
            for (; j < seq_len; j++) s_row[j] *= inv_sum;
        }

        // --- Phase 3: O_tile[br,D] = S[br,S] × V[S,D] ---
        avx512::dgemm(br, head_dim, seq_len,
                      S, V, O + (size_t)i0 * head_dim, gemm_post_t{});
    }
}

// Multi-head SDPA: Q, K, V are [batch, seq_len, head_dim].
// Output is [batch, seq_len, head_dim].
// Parallelizes across batch (= num_heads).
// @nnr-meta isa=AVX512 dtype=fp32
void sdpa_multihead_avx512(
    const float* Q, const float* K, const float* V, float* O,
    int batch, int seq_len, int head_dim);

} // namespace nnr

#endif // NNR_ARCH_X64
