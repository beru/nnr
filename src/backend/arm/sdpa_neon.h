#pragma once
// Fused Scaled Dot-Product Attention — NEON (FP32).
// Counterpart of x64/sdpa_avx512.h.
//
// Computes: Output = Softmax(Q × K^T) × V
// where Q, K, V are [batch, seq_len, head_dim].
//
// Tiled: BR=32 query rows at a time to keep the attention score tile in L2.
// Q is pre-scaled by the caller (multiplied by 1/sqrt(d_k)).

#if defined(__aarch64__) || defined(_M_ARM64)

#ifdef _MSC_VER
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "backend/cpu/kernel/gemm.h"
#include "backend/arm/gemm_neon.h"
#include "backend/arm/simd_math_neon.h"
#include "thread_pool.h"

namespace nnr {

// Serial GEMM: C[M×N] = A[M×K] × B[K×N], row-major, single-threaded.
// Uses an 8×4 packed-A micro-kernel with `vfmaq_laneq_f32` lane-based A-broadcast.
// Pattern copied from `gemm_neon.h`'s default tiled path (lines ~460-513) but
// without the tiled K-blocking / scratch pool / threading. `neon::dgemm` can't be
// used inside SDPA because its internal `for_static` thrashes the thread pool:
// the fused SDPA wants OUTER head-parallelism, and the NNR thread pool has no
// nested-dispatch support (calling from a worker deadlocks). So the inner GEMM
// must be serial. `pa_scratch` is a caller-supplied buffer ≥ K * 8 floats.
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=SDPA_inner
inline void sdpa_gemm_serial(int M, int N, int K,
    const float* __restrict A, const float* __restrict B, float* __restrict C,
    float* __restrict pa_scratch)
{
    int i = 0;
    for (; i + 8 <= M; i += 8) {
        // Pack A: 8 rows × K → K × 8 interleaved for vfmaq_laneq_f32.
        for (int k = 0; k < K; k++) {
            pa_scratch[k * 8 + 0] = A[(size_t)(i + 0) * K + k];
            pa_scratch[k * 8 + 1] = A[(size_t)(i + 1) * K + k];
            pa_scratch[k * 8 + 2] = A[(size_t)(i + 2) * K + k];
            pa_scratch[k * 8 + 3] = A[(size_t)(i + 3) * K + k];
            pa_scratch[k * 8 + 4] = A[(size_t)(i + 4) * K + k];
            pa_scratch[k * 8 + 5] = A[(size_t)(i + 5) * K + k];
            pa_scratch[k * 8 + 6] = A[(size_t)(i + 6) * K + k];
            pa_scratch[k * 8 + 7] = A[(size_t)(i + 7) * K + k];
        }
        int j = 0;
        for (; j + 4 <= N; j += 4) {
            float32x4_t c0 = vdupq_n_f32(0.f), c1 = vdupq_n_f32(0.f);
            float32x4_t c2 = vdupq_n_f32(0.f), c3 = vdupq_n_f32(0.f);
            float32x4_t c4 = vdupq_n_f32(0.f), c5 = vdupq_n_f32(0.f);
            float32x4_t c6 = vdupq_n_f32(0.f), c7 = vdupq_n_f32(0.f);
            for (int k = 0; k < K; k++) {
                float32x4_t bv   = vld1q_f32(B + (size_t)k * N + j);
                float32x4_t a_lo = vld1q_f32(pa_scratch + k * 8);
                float32x4_t a_hi = vld1q_f32(pa_scratch + k * 8 + 4);
                c0 = vfmaq_laneq_f32(c0, bv, a_lo, 0);
                c1 = vfmaq_laneq_f32(c1, bv, a_lo, 1);
                c2 = vfmaq_laneq_f32(c2, bv, a_lo, 2);
                c3 = vfmaq_laneq_f32(c3, bv, a_lo, 3);
                c4 = vfmaq_laneq_f32(c4, bv, a_hi, 0);
                c5 = vfmaq_laneq_f32(c5, bv, a_hi, 1);
                c6 = vfmaq_laneq_f32(c6, bv, a_hi, 2);
                c7 = vfmaq_laneq_f32(c7, bv, a_hi, 3);
            }
            vst1q_f32(C + (size_t)(i + 0) * N + j, c0);
            vst1q_f32(C + (size_t)(i + 1) * N + j, c1);
            vst1q_f32(C + (size_t)(i + 2) * N + j, c2);
            vst1q_f32(C + (size_t)(i + 3) * N + j, c3);
            vst1q_f32(C + (size_t)(i + 4) * N + j, c4);
            vst1q_f32(C + (size_t)(i + 5) * N + j, c5);
            vst1q_f32(C + (size_t)(i + 6) * N + j, c6);
            vst1q_f32(C + (size_t)(i + 7) * N + j, c7);
        }
        // N-tail: 1..3 cols (scalar across 8 rows — rare for our shapes)
        for (; j < N; j++) {
            float s[8] = {};
            for (int k = 0; k < K; k++) {
                float bv = B[(size_t)k * N + j];
                s[0] += pa_scratch[k * 8 + 0] * bv;
                s[1] += pa_scratch[k * 8 + 1] * bv;
                s[2] += pa_scratch[k * 8 + 2] * bv;
                s[3] += pa_scratch[k * 8 + 3] * bv;
                s[4] += pa_scratch[k * 8 + 4] * bv;
                s[5] += pa_scratch[k * 8 + 5] * bv;
                s[6] += pa_scratch[k * 8 + 6] * bv;
                s[7] += pa_scratch[k * 8 + 7] * bv;
            }
            for (int r = 0; r < 8; r++) C[(size_t)(i + r) * N + j] = s[r];
        }
    }
    // M-tail (< 8 rows): scalar 1-row kernel.
    for (; i < M; i++) {
        int j = 0;
        for (; j + 4 <= N; j += 4) {
            float32x4_t c = vdupq_n_f32(0.f);
            for (int k = 0; k < K; k++)
                c = vfmaq_f32(c, vdupq_n_f32(A[(size_t)i * K + k]),
                                 vld1q_f32(B + (size_t)k * N + j));
            vst1q_f32(C + (size_t)i * N + j, c);
        }
        for (; j < N; j++) {
            float s = 0.f;
            for (int k = 0; k < K; k++) s += A[(size_t)i * K + k] * B[(size_t)k * N + j];
            C[(size_t)i * N + j] = s;
        }
    }
}

// Scratch layout for sdpa_head_neon:
//   [ K^T : head_dim * seq_len ]
//   [ S   : BR * seq_len       ]
//   [ PA  : 8 * max(head_dim, seq_len) ]   (A-pack for sdpa_gemm_serial)
inline size_t sdpa_head_scratch_floats(int seq_len, int head_dim) {
    return (size_t)head_dim * seq_len
         + (size_t)32 * seq_len
         + (size_t)8 * (size_t)std::max(head_dim, seq_len);
}

inline void sdpa_head_neon(
    const float* Q,   // [seq_len, head_dim]
    const float* K,   // [seq_len, head_dim] (not transposed)
    const float* V,   // [seq_len, head_dim]
    float* O,         // [seq_len, head_dim]
    float* scratch,   // ≥ sdpa_head_scratch_floats(seq_len, head_dim)
    int seq_len,
    int head_dim)
{
    constexpr int BR = 32;

    // Transpose K → K^T in scratch.
    float* KT = scratch;
    for (int j = 0; j < seq_len; j++)
        for (int d = 0; d < head_dim; d++)
            KT[(size_t)d * seq_len + j] = K[(size_t)j * head_dim + d];

    float* S = scratch + (size_t)head_dim * seq_len;
    float* PA = S + (size_t)BR * seq_len;

    for (int i0 = 0; i0 < seq_len; i0 += BR) {
        int br = std::min(BR, seq_len - i0);

        // Phase 1: S[br, seq_len] = Q_tile × K^T (serial inner GEMM)
        sdpa_gemm_serial(br, seq_len, head_dim,
                         Q + (size_t)i0 * head_dim, KT, S, PA);

        // Phase 2: Softmax each row of S (max → exp(x - max) → normalize)
        for (int qi = 0; qi < br; qi++) {
            float* s_row = S + (size_t)qi * seq_len;

            // Row max
            float32x4_t vmax = vdupq_n_f32(-1e30f);
            int j = 0;
            for (; j + 4 <= seq_len; j += 4)
                vmax = vmaxq_f32(vmax, vld1q_f32(s_row + j));
            float row_max = vmaxvq_f32(vmax);
            for (; j < seq_len; j++) row_max = std::max(row_max, s_row[j]);

            // exp(x - max), accumulate sum
            float32x4_t vmx = vdupq_n_f32(row_max);
            float32x4_t vsum = vdupq_n_f32(0.0f);
            j = 0;
            for (; j + 4 <= seq_len; j += 4) {
                float32x4_t e = exp_neon_ps(vsubq_f32(vld1q_f32(s_row + j), vmx));
                vst1q_f32(s_row + j, e);
                vsum = vaddq_f32(vsum, e);
            }
            float sum = vaddvq_f32(vsum);
            for (; j < seq_len; j++) {
                float e = std::exp(s_row[j] - row_max);
                s_row[j] = e;
                sum += e;
            }

            // Normalize
            float32x4_t vinv = vdupq_n_f32(1.0f / sum);
            j = 0;
            for (; j + 4 <= seq_len; j += 4)
                vst1q_f32(s_row + j, vmulq_f32(vld1q_f32(s_row + j), vinv));
            float inv = 1.0f / sum;
            for (; j < seq_len; j++) s_row[j] *= inv;
        }

        // Phase 3: O_tile[br, head_dim] = S × V (serial inner GEMM)
        sdpa_gemm_serial(br, head_dim, seq_len,
                         S, V, O + (size_t)i0 * head_dim, PA);
    }
}

// Multi-head SDPA: Q, K, V, O are [batch(=num_heads), seq_len, head_dim].
//
// Parallelizes across heads via `nnr::for_static` and uses `sdpa_gemm_serial`
// (not `neon::dgemm`) inside each head — the NNR thread pool has no nested
// dispatch support, so the inner GEMMs must be serial for the outer parallel
// region to work. Each worker gets its own scratch slice. See
// `docs/decisions.md` 2026-04-25 for the full rationale.
inline void sdpa_multihead_neon(
    const float* Q, const float* K, const float* V, float* O,
    int batch, int seq_len, int head_dim)
{
    size_t head_stride = (size_t)seq_len * head_dim;
    size_t scratch_floats = sdpa_head_scratch_floats(seq_len, head_dim);
    std::vector<float> scratch_all((size_t)batch * scratch_floats);

    nnr::for_static(0, batch, batch > 1, [&](int h) {
        sdpa_head_neon(
            Q + h * head_stride,
            K + h * head_stride,
            V + h * head_stride,
            O + h * head_stride,
            scratch_all.data() + (size_t)h * scratch_floats,
            seq_len, head_dim);
    });
}

} // namespace nnr

#endif // aarch64
