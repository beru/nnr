#pragma once
// ARM NEON GEMM backend.
// 128-bit (8×8 micro-kernel) counterpart of x64/gemm_avx2.h.
// Included from kernel/gemm.h — requires gemm_post_t to be defined before inclusion.

#if defined(__aarch64__) || defined(_M_ARM64)

#ifdef _MSC_VER
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif
#include <cfloat>
#include <cstring>
#include <algorithm>
#include "thread_pool.h"
#include "backend/arm/vec_ops_neon.h"
#include "backend/arm/gemm_ukernel_neon.h"

namespace nnr {

// Zero bias array used when post-op is active but no explicit bias.
// On ARM this serves the same role as the x64 fused_zero_bias.
#if !defined(NNR_FUSED_ZERO_BIAS_DEFINED)
#define NNR_FUSED_ZERO_BIAS_DEFINED
alignas(16) inline const float fused_zero_bias[16] = {};
#endif

namespace neon {

// ============================================================================
// Fused post-op helpers: bias + clamp on accumulators.
// ============================================================================

#if 0 // LOCUST
;MR = 8
;args = ", ".join(f"float32x4_t& c{r}" for r in range(MR))
// NCHW 8-row: per-row bias broadcast + clamp
inline void fuse_nchw_@MR@(@args@,
                        const float* bp, float fmin, float fmax) {
;for r in range(MR):
    c@r@ = vaddq_f32(c@r@, vdupq_n_f32(bp[@r@]));
;    pass
    float32x4_t vmin = vdupq_n_f32(fmin);
    float32x4_t vmax = vdupq_n_f32(fmax);
;for r in range(MR):
    c@r@ = vmaxq_f32(c@r@, vmin); c@r@ = vminq_f32(c@r@, vmax);
;    pass
}
#else // LOCUST
// NCHW 8-row: per-row bias broadcast + clamp
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW tiling=MR fusion=post_op
inline void fuse_nchw_8(float32x4_t& c0, float32x4_t& c1, float32x4_t& c2, float32x4_t& c3, float32x4_t& c4, float32x4_t& c5, float32x4_t& c6, float32x4_t& c7,
                        const float* bp, float fmin, float fmax) {
    c0 = vaddq_f32(c0, vdupq_n_f32(bp[0]));
    c1 = vaddq_f32(c1, vdupq_n_f32(bp[1]));
    c2 = vaddq_f32(c2, vdupq_n_f32(bp[2]));
    c3 = vaddq_f32(c3, vdupq_n_f32(bp[3]));
    c4 = vaddq_f32(c4, vdupq_n_f32(bp[4]));
    c5 = vaddq_f32(c5, vdupq_n_f32(bp[5]));
    c6 = vaddq_f32(c6, vdupq_n_f32(bp[6]));
    c7 = vaddq_f32(c7, vdupq_n_f32(bp[7]));
    float32x4_t vmin = vdupq_n_f32(fmin);
    float32x4_t vmax = vdupq_n_f32(fmax);
    c0 = vmaxq_f32(c0, vmin); c0 = vminq_f32(c0, vmax);
    c1 = vmaxq_f32(c1, vmin); c1 = vminq_f32(c1, vmax);
    c2 = vmaxq_f32(c2, vmin); c2 = vminq_f32(c2, vmax);
    c3 = vmaxq_f32(c3, vmin); c3 = vminq_f32(c3, vmax);
    c4 = vmaxq_f32(c4, vmin); c4 = vminq_f32(c4, vmax);
    c5 = vmaxq_f32(c5, vmin); c5 = vminq_f32(c5, vmax);
    c6 = vmaxq_f32(c6, vmin); c6 = vminq_f32(c6, vmax);
    c7 = vmaxq_f32(c7, vmin); c7 = vminq_f32(c7, vmax);
}
#endif // LOCUST

// NCHW 1-row: single-row bias broadcast + clamp
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW fusion=post_op
inline void fuse_nchw_1(float32x4_t& acc, float bias, float fmin, float fmax) {
    acc = vaddq_f32(acc, vdupq_n_f32(bias));
    acc = vmaxq_f32(acc, vdupq_n_f32(fmin));
    acc = vminq_f32(acc, vdupq_n_f32(fmax));
}

// NCHW multi-vec array: bias + clamp over array of accumulators (small-M path)
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW fusion=post_op
inline void fuse_nchw_arr(float32x4_t* acc, int nvec, float bias, float fmin, float fmax) {
    float32x4_t vb = vdupq_n_f32(bias);
    float32x4_t vmin = vdupq_n_f32(fmin);
    float32x4_t vmax = vdupq_n_f32(fmax);
    for (int j = 0; j < nvec; j++) {
        acc[j] = vaddq_f32(acc[j], vb);
        acc[j] = vmaxq_f32(acc[j], vmin);
        acc[j] = vminq_f32(acc[j], vmax);
    }
}

// Scalar bias + clamp
// @nnr-meta isa=scalar dtype=fp32 layout=NCHW fusion=post_op
inline void fuse_scalar(float& s, float bias, float fmin, float fmax) {
    s += bias;
    s = std::max(s, fmin);
    s = std::min(s, fmax);
}

#if 0 // LOCUST
;args = ", ".join(f"float32x4_t& c{r}" for r in range(MR))
// NHWC 8-row: add pre-loaded bias vector + clamp
inline void fuse_nhwc_@MR@(@args@,
                        float32x4_t vb, float fmin, float fmax) {
;for r in range(0, MR, 2):
    c@r@ = vaddq_f32(c@r@, vb); c@r+1@ = vaddq_f32(c@r+1@, vb);
;    pass
    float32x4_t vmin = vdupq_n_f32(fmin);
    float32x4_t vmax = vdupq_n_f32(fmax);
;for r in range(MR):
    c@r@ = vmaxq_f32(c@r@, vmin); c@r@ = vminq_f32(c@r@, vmax);
;    pass
}
#else // LOCUST
// NHWC 8-row: add pre-loaded bias vector + clamp
// @nnr-meta isa=NEON dtype=fp32 layout=NHWC tiling=MR fusion=post_op
inline void fuse_nhwc_8(float32x4_t& c0, float32x4_t& c1, float32x4_t& c2, float32x4_t& c3, float32x4_t& c4, float32x4_t& c5, float32x4_t& c6, float32x4_t& c7,
                        float32x4_t vb, float fmin, float fmax) {
    c0 = vaddq_f32(c0, vb); c1 = vaddq_f32(c1, vb);
    c2 = vaddq_f32(c2, vb); c3 = vaddq_f32(c3, vb);
    c4 = vaddq_f32(c4, vb); c5 = vaddq_f32(c5, vb);
    c6 = vaddq_f32(c6, vb); c7 = vaddq_f32(c7, vb);
    float32x4_t vmin = vdupq_n_f32(fmin);
    float32x4_t vmax = vdupq_n_f32(fmax);
    c0 = vmaxq_f32(c0, vmin); c0 = vminq_f32(c0, vmax);
    c1 = vmaxq_f32(c1, vmin); c1 = vminq_f32(c1, vmax);
    c2 = vmaxq_f32(c2, vmin); c2 = vminq_f32(c2, vmax);
    c3 = vmaxq_f32(c3, vmin); c3 = vminq_f32(c3, vmax);
    c4 = vmaxq_f32(c4, vmin); c4 = vminq_f32(c4, vmax);
    c5 = vmaxq_f32(c5, vmin); c5 = vminq_f32(c5, vmax);
    c6 = vmaxq_f32(c6, vmin); c6 = vminq_f32(c6, vmax);
    c7 = vmaxq_f32(c7, vmin); c7 = vminq_f32(c7, vmax);
}
#endif // LOCUST

// NHWC 1-row: add pre-loaded bias vector + clamp
// @nnr-meta isa=NEON dtype=fp32 layout=NHWC fusion=post_op
inline void fuse_nhwc_1(float32x4_t& acc, float32x4_t vb, float fmin, float fmax) {
    acc = vaddq_f32(acc, vb);
    acc = vmaxq_f32(acc, vdupq_n_f32(fmin));
    acc = vminq_f32(acc, vdupq_n_f32(fmax));
}

// ============================================================================
// NEON GEMM: C[n×m] = A[n×o] × B[o×m] with optional post-processing.
// 8×8 micro-kernel using vfmaq_laneq_f32 for efficient A-broadcast.
// ============================================================================
template <typename PostFn>
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=GEMM tiling=[K,MR,NR] fusion=post_op
inline void dgemm(int n, int m, int o, const float* __restrict A, const float* __restrict B, float* __restrict C, const PostFn& post_fn)
{
    constexpr bool can_fuse = PostFn::per_row_bias;

    // GEMV fast path: M=1 (row vector × matrix)
    if (n == 1) {
        int j = 0;
        for (; j + 4 <= m; j += 4)
            vst1q_f32(C + j, vdupq_n_f32(0.0f));
        for (; j < m; ++j)
            C[j] = 0.0f;
        for (int k = 0; k < o; ++k) {
            float a_k = A[k];
            const float* brow = B + (size_t)k * m;
            float32x4_t va = vdupq_n_f32(a_k);
            j = 0;
            for (; j + 4 <= m; j += 4)
                vst1q_f32(C + j,
                    vfmaq_f32(vld1q_f32(C + j), va, vld1q_f32(brow + j)));
            for (; j < m; ++j)
                C[j] += a_k * brow[j];
        }
        post_fn.apply(0, C, m);
        return;
    }

    // GEMV fast path: N=1 (matrix × column vector)
    // Uses 4 independent accumulators to hide FMA latency.
    if (m == 1) {
        nnr::for_static(0, n, n > 256, [&](int i) {
            const float* pa_row = A + (size_t)i * o;
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 16 <= o; k += 16) {
                acc0 = vfmaq_f32(acc0, vld1q_f32(pa_row + k),      vld1q_f32(B + k));
                acc1 = vfmaq_f32(acc1, vld1q_f32(pa_row + k + 4),  vld1q_f32(B + k + 4));
                acc2 = vfmaq_f32(acc2, vld1q_f32(pa_row + k + 8),  vld1q_f32(B + k + 8));
                acc3 = vfmaq_f32(acc3, vld1q_f32(pa_row + k + 12), vld1q_f32(B + k + 12));
            }
            acc0 = vaddq_f32(acc0, acc1);
            acc2 = vaddq_f32(acc2, acc3);
            acc0 = vaddq_f32(acc0, acc2);
            for (; k + 4 <= o; k += 4)
                acc0 = vfmaq_f32(acc0, vld1q_f32(pa_row + k), vld1q_f32(B + k));
            float s = vaddvq_f32(acc0);
            for (; k < o; ++k)
                s += pa_row[k] * B[k];
            C[i] = s;
            post_fn.apply(i, C + i, 1);
        });
        return;
    }

    // Small-K path: A[N,K] is small (fits in L1), B[K,M] streams through L2.
    // 8-row register blocking gives 8 independent FMA chains.
    // Threshold o<=48 is empirically better than o<=64: at o=64 with large m
    // (e.g. multi-head attention M=1500 K=64 N=1500), the non-packed inner loop
    // re-reads B (n/8)×(m/4) times which thrashes L2 — the default packed-tile
    // path wins there. Don't raise this above 48 without a per-shape sweep.
    if (o <= 48 && m >= 32) {
        int mchunks = (m + 3) / 4;
        nnr::for_static(0, mchunks, mchunks >= 32 && (int64_t)n * m * o > (1 << 22), [&](int jc) {
            int j = jc * 4;
            int jw = std::min(4, m - j);
            float fmin = -FLT_MAX;
            float fmax = FLT_MAX;
            if constexpr (can_fuse) { if (post_fn.kind != post_op_kind::none) {
                fmin = post_fn.clip_min; fmax = post_fn.clip_max;
            }}
            int i = 0;
            for (; i + 8 <= n; i += 8) {
                float32x4_t c0 = vdupq_n_f32(0.0f);
                float32x4_t c1 = vdupq_n_f32(0.0f);
                float32x4_t c2 = vdupq_n_f32(0.0f);
                float32x4_t c3 = vdupq_n_f32(0.0f);
                float32x4_t c4 = vdupq_n_f32(0.0f);
                float32x4_t c5 = vdupq_n_f32(0.0f);
                float32x4_t c6 = vdupq_n_f32(0.0f);
                float32x4_t c7 = vdupq_n_f32(0.0f);
                for (int k = 0; k < o; k++) {
                    float32x4_t bv;
                    if (jw == 4) {
                        bv = vld1q_f32(B + (size_t)k * m + j);
                    } else {
                        alignas(16) float tmp[4] = {};
                        for (int p = 0; p < jw; p++) tmp[p] = B[(size_t)k * m + j + p];
                        bv = vld1q_f32(tmp);
                    }
                    c0 = vfmaq_f32(c0, vdupq_n_f32(A[(size_t)(i+0) * o + k]), bv);
                    c1 = vfmaq_f32(c1, vdupq_n_f32(A[(size_t)(i+1) * o + k]), bv);
                    c2 = vfmaq_f32(c2, vdupq_n_f32(A[(size_t)(i+2) * o + k]), bv);
                    c3 = vfmaq_f32(c3, vdupq_n_f32(A[(size_t)(i+3) * o + k]), bv);
                    c4 = vfmaq_f32(c4, vdupq_n_f32(A[(size_t)(i+4) * o + k]), bv);
                    c5 = vfmaq_f32(c5, vdupq_n_f32(A[(size_t)(i+5) * o + k]), bv);
                    c6 = vfmaq_f32(c6, vdupq_n_f32(A[(size_t)(i+6) * o + k]), bv);
                    c7 = vfmaq_f32(c7, vdupq_n_f32(A[(size_t)(i+7) * o + k]), bv);
                }
                if constexpr (can_fuse) {
                    const float* bp = (post_fn.kind != post_op_kind::none && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                    fuse_nchw_8(c0, c1, c2, c3, c4, c5, c6, c7, bp, fmin, fmax);
                }
                if (jw == 4) {
                    vst1q_f32(C + (size_t)(i+0) * m + j, c0);
                    vst1q_f32(C + (size_t)(i+1) * m + j, c1);
                    vst1q_f32(C + (size_t)(i+2) * m + j, c2);
                    vst1q_f32(C + (size_t)(i+3) * m + j, c3);
                    vst1q_f32(C + (size_t)(i+4) * m + j, c4);
                    vst1q_f32(C + (size_t)(i+5) * m + j, c5);
                    vst1q_f32(C + (size_t)(i+6) * m + j, c6);
                    vst1q_f32(C + (size_t)(i+7) * m + j, c7);
                } else {
                    alignas(16) float tmp[4];
                    float32x4_t* rows[] = {&c0,&c1,&c2,&c3,&c4,&c5,&c6,&c7};
                    for (int r = 0; r < 8; r++) {
                        vst1q_f32(tmp, *rows[r]);
                        for (int p = 0; p < jw; p++) C[(size_t)(i+r) * m + j + p] = tmp[p];
                    }
                }
                if (!(can_fuse && post_fn.kind != post_op_kind::none))
                    post_fn.apply_rows(i, i + 8, C, m, j, jw);
            }
            for (; i < n; i++) {
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (int k = 0; k < o; k++) {
                    float32x4_t bv;
                    if (jw == 4) {
                        bv = vld1q_f32(B + (size_t)k * m + j);
                    } else {
                        alignas(16) float tmp[4] = {};
                        for (int p = 0; p < jw; p++) tmp[p] = B[(size_t)k * m + j + p];
                        bv = vld1q_f32(tmp);
                    }
                    acc = vfmaq_f32(acc, vdupq_n_f32(A[(size_t)i * o + k]), bv);
                }
                if constexpr (can_fuse) {
                    const float* bp = (post_fn.kind != post_op_kind::none && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                    fuse_nchw_1(acc, bp[0], fmin, fmax);
                }
                if (jw == 4) {
                    vst1q_f32(C + (size_t)i * m + j, acc);
                } else {
                    alignas(16) float tmp[4];
                    vst1q_f32(tmp, acc);
                    for (int p = 0; p < jw; p++) C[(size_t)i * m + j + p] = tmp[p];
                }
                if (!(can_fuse && post_fn.kind != post_op_kind::none))
                    post_fn.apply(i, C + (size_t)i * m + j, jw);
            }
        });
        return;
    }

    // Small-M path: 8-row register blocking shares B loads across rows.
    if (m < 32) {
        int mfull = m / 4, mtail = m & 3;
        int ngroups = (n + 7) / 8;
        nnr::for_static(0, ngroups, ngroups > 8 && (int64_t)n * m * o > (1 << 18), [&](int ig) {
            int i0 = ig * 8;
            int ie = std::min(i0 + 8, n);
            int nr = ie - i0;
            float fmin = -FLT_MAX;
            float fmax = FLT_MAX;
            if constexpr (can_fuse) { if (post_fn.kind != post_op_kind::none) {
                fmin = post_fn.clip_min; fmax = post_fn.clip_max;
            }}
            if (nr == 8) {
                float32x4_t c0[8]={}, c1[8]={}, c2[8]={}, c3[8]={};
                float32x4_t c4[8]={}, c5[8]={}, c6[8]={}, c7[8]={};
                const float* pa[8];
                for (int r = 0; r < 8; r++)
                    pa[r] = A + (size_t)(i0 + r) * o;
                for (int k = 0; k < o; k++) {
                    const float* br = B + (size_t)k * m;
                    float32x4_t av[8];
                    for (int r = 0; r < 8; r++)
                        av[r] = vdupq_n_f32(pa[r][k]);
                    for (int j = 0; j < mfull; j++) {
                        float32x4_t bv = vld1q_f32(br + j * 4);
                        c0[j] = vfmaq_f32(c0[j], av[0], bv);
                        c1[j] = vfmaq_f32(c1[j], av[1], bv);
                        c2[j] = vfmaq_f32(c2[j], av[2], bv);
                        c3[j] = vfmaq_f32(c3[j], av[3], bv);
                        c4[j] = vfmaq_f32(c4[j], av[4], bv);
                        c5[j] = vfmaq_f32(c5[j], av[5], bv);
                        c6[j] = vfmaq_f32(c6[j], av[6], bv);
                        c7[j] = vfmaq_f32(c7[j], av[7], bv);
                    }
                    if (mtail) {
                        alignas(16) float tmp[4] = {};
                        for (int p = 0; p < mtail; p++) tmp[p] = br[mfull * 4 + p];
                        float32x4_t bv = vld1q_f32(tmp);
                        c0[mfull] = vfmaq_f32(c0[mfull], av[0], bv);
                        c1[mfull] = vfmaq_f32(c1[mfull], av[1], bv);
                        c2[mfull] = vfmaq_f32(c2[mfull], av[2], bv);
                        c3[mfull] = vfmaq_f32(c3[mfull], av[3], bv);
                        c4[mfull] = vfmaq_f32(c4[mfull], av[4], bv);
                        c5[mfull] = vfmaq_f32(c5[mfull], av[5], bv);
                        c6[mfull] = vfmaq_f32(c6[mfull], av[6], bv);
                        c7[mfull] = vfmaq_f32(c7[mfull], av[7], bv);
                    }
                }
                if constexpr (can_fuse) {
                    const float* bp = (post_fn.kind != post_op_kind::none && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i0 : fused_zero_bias;
                    float32x4_t* rows[] = {c0, c1, c2, c3, c4, c5, c6, c7};
                    int nvec = mtail ? mfull + 1 : mfull;
                    for (int r = 0; r < 8; r++)
                        fuse_nchw_arr(rows[r], nvec, bp[r], fmin, fmax);
                }
                float32x4_t* all[] = {c0, c1, c2, c3, c4, c5, c6, c7};
                for (int r = 0; r < 8; r++) {
                    float* pc = C + (size_t)(i0 + r) * m;
                    for (int j = 0; j < mfull; j++)
                        vst1q_f32(pc + j * 4, all[r][j]);
                    if (mtail) {
                        alignas(16) float tmp[4];
                        vst1q_f32(tmp, all[r][mfull]);
                        for (int p = 0; p < mtail; p++) pc[mfull * 4 + p] = tmp[p];
                    }
                }
                if (!(can_fuse && post_fn.kind != post_op_kind::none))
                    post_fn.apply_rows(i0, i0 + 8, C, m, 0, m);
            } else {
                // Remainder: 1-7 rows
                for (int i = i0; i < ie; i++) {
                    float* pc = C + (size_t)i * m;
                    const float* pa_r = A + (size_t)i * o;
                    float32x4_t acc[8] = {};
                    for (int k = 0; k < o; k++) {
                        const float* br = B + (size_t)k * m;
                        float32x4_t av = vdupq_n_f32(pa_r[k]);
                        for (int j = 0; j < mfull; j++)
                            acc[j] = vfmaq_f32(acc[j], av, vld1q_f32(br + j * 4));
                        if (mtail) {
                            alignas(16) float tmp[4] = {};
                            for (int p = 0; p < mtail; p++) tmp[p] = br[mfull * 4 + p];
                            acc[mfull] = vfmaq_f32(acc[mfull], av, vld1q_f32(tmp));
                        }
                    }
                    if constexpr (can_fuse) {
                        const float* bp = (post_fn.kind != post_op_kind::none && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                        int nvec = mtail ? mfull + 1 : mfull;
                        fuse_nchw_arr(acc, nvec, bp[0], fmin, fmax);
                    }
                    for (int j = 0; j < mfull; j++)
                        vst1q_f32(pc + j * 4, acc[j]);
                    if (mtail) {
                        alignas(16) float tmp[4];
                        vst1q_f32(tmp, acc[mfull]);
                        for (int p = 0; p < mtail; p++) pc[mfull * 4 + p] = tmp[p];
                    }
                    if (!(can_fuse && post_fn.kind != post_op_kind::none))
                        post_fn.apply(i, pc, m);
                }
            }
        });
        return;
    }

    // Tiled path: 8×8 micro-kernel with A-packing, B-packing, K-blocking.
    // Uses vfmaq_laneq_f32 for efficient A-broadcast from packed layout.
    constexpr int JBLK = 64;
    constexpr int KC = 256;
    int ni = (n + JBLK - 1) / JBLK;
    int nj = (m + JBLK - 1) / JBLK;
    int ntiles = ni * nj;
    bool par = ntiles > 1 && (int64_t)n * m * o > (1 << 21);

    NNR_POOL_ENSURE_SCRATCH(((size_t)KC * JBLK + (size_t)KC * 8) * sizeof(float));

    nnr::for_dynamic(0, ntiles, par, [&](int tid, int tile) {
        float* scratch = (float*)NNR_POOL_SCRATCH(tid);
        float* pb = scratch;
        float* pa_pack = scratch + (size_t)KC * JBLK;
        int i0 = (tile / nj) * JBLK;
        int j0 = (tile % nj) * JBLK;
        int ie = std::min(i0 + JBLK, n);
        int je = std::min(j0 + JBLK, m);
        int jw = je - j0;

        for (int k0 = 0; k0 < o; k0 += KC) {
            int kc = std::min(KC, o - k0);
            bool last_k = (k0 + kc == o);

            float fmin = -FLT_MAX;
            float fmax = FLT_MAX;
            bool fuse_nchw = false;
            if constexpr (can_fuse) { if (last_k && post_fn.kind != post_op_kind::none) {
                fuse_nchw = true; fmin = post_fn.clip_min; fmax = post_fn.clip_max;
            }}

            // Pack B sub-panel: pb[kc × JBLK]
            for (int k = 0; k < kc; ++k)
                memcpy(pb + (size_t)k * JBLK, B + (size_t)(k0 + k) * m + j0, jw * sizeof(float));

            // 8-row groups
            int i = i0;
            for (; i + 8 <= ie; i += 8) {
                const float* pa[8];
                float* pc[8];
                for (int r = 0; r < 8; r++) {
                    pa[r] = A + (size_t)(i + r) * o + k0;
                    pc[r] = C + (size_t)(i + r) * m;
                }

                // Pack A: 8 rows × kc cols → pa_pack[kc × 8] (interleaved)
                // Layout enables vld1q_f32 to load 4 row values for vfmaq_laneq_f32.
                for (int k = 0; k < kc; k++) {
                    pa_pack[k * 8 + 0] = pa[0][k];
                    pa_pack[k * 8 + 1] = pa[1][k];
                    pa_pack[k * 8 + 2] = pa[2][k];
                    pa_pack[k * 8 + 3] = pa[3][k];
                    pa_pack[k * 8 + 4] = pa[4][k];
                    pa_pack[k * 8 + 5] = pa[5][k];
                    pa_pack[k * 8 + 6] = pa[6][k];
                    pa_pack[k * 8 + 7] = pa[7][k];
                }

                int v = j0;
                for (; v + UK_NR <= je; v += UK_NR) {
                    const float* bp_uk = fused_zero_bias;
                    if constexpr (can_fuse) { if (fuse_nchw && post_fn.bias) bp_uk = post_fn.bias + post_fn.bias_off + i; }
                    ukernel_nchw(kc, pa_pack, pb + (v - j0), JBLK, pc, v,
                        k0 == 0, fuse_nchw, bp_uk, fmin, fmax);
                }
                for (; v + 4 <= je; v += 4) {
                    const float* pp = pb + (v - j0);
                    float32x4_t c0, c1, c2, c3, c4, c5, c6, c7;
                    if (k0 == 0) {
                        c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = vdupq_n_f32(0.0f);
                    } else {
                        c0 = vld1q_f32(pc[0] + v); c1 = vld1q_f32(pc[1] + v);
                        c2 = vld1q_f32(pc[2] + v); c3 = vld1q_f32(pc[3] + v);
                        c4 = vld1q_f32(pc[4] + v); c5 = vld1q_f32(pc[5] + v);
                        c6 = vld1q_f32(pc[6] + v); c7 = vld1q_f32(pc[7] + v);
                    }
                    for (int k = 0; k < kc; k++) {
                        float32x4_t bv = vld1q_f32(pp + (size_t)k * JBLK);
                        float32x4_t a_lo = vld1q_f32(pa_pack + k * 8);
                        float32x4_t a_hi = vld1q_f32(pa_pack + k * 8 + 4);
                        c0 = vfmaq_laneq_f32(c0, bv, a_lo, 0);
                        c1 = vfmaq_laneq_f32(c1, bv, a_lo, 1);
                        c2 = vfmaq_laneq_f32(c2, bv, a_lo, 2);
                        c3 = vfmaq_laneq_f32(c3, bv, a_lo, 3);
                        c4 = vfmaq_laneq_f32(c4, bv, a_hi, 0);
                        c5 = vfmaq_laneq_f32(c5, bv, a_hi, 1);
                        c6 = vfmaq_laneq_f32(c6, bv, a_hi, 2);
                        c7 = vfmaq_laneq_f32(c7, bv, a_hi, 3);
                    }
                    if constexpr (can_fuse) {
                        const float* bp = (fuse_nchw && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                        fuse_nchw_8(c0, c1, c2, c3, c4, c5, c6, c7, bp, fmin, fmax);
                    }
                    vst1q_f32(pc[0] + v, c0); vst1q_f32(pc[1] + v, c1);
                    vst1q_f32(pc[2] + v, c2); vst1q_f32(pc[3] + v, c3);
                    vst1q_f32(pc[4] + v, c4); vst1q_f32(pc[5] + v, c5);
                    vst1q_f32(pc[6] + v, c6); vst1q_f32(pc[7] + v, c7);
                }
                // Scalar remainder columns
                for (; v < je; ++v) {
                    const float* pp = pb + (v - j0);
                    for (int r = 0; r < 8; r++) {
                        float s = (k0 == 0) ? 0.0f : pc[r][v];
                        for (int k = 0; k < kc; ++k)
                            s += pa_pack[k * 8 + r] * pp[(size_t)k * JBLK];
                        if constexpr (can_fuse) {
                            const float* bp = (fuse_nchw && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                            fuse_scalar(s, bp[r], fmin, fmax);
                        }
                        pc[r][v] = s;
                    }
                }
            }
            // Remainder rows (< 8): 1-row kernel
            for (; i < ie; i++) {
                float* pci = C + (size_t)i * m;
                const float* pai = A + (size_t)i * o + k0;
                int v = j0;
                for (; v + 4 <= je; v += 4) {
                    const float* pp = pb + (v - j0);
                    float32x4_t acc = (k0 == 0) ? vdupq_n_f32(0.0f) : vld1q_f32(pci + v);
                    for (int k = 0; k < kc; ++k)
                        acc = vfmaq_f32(acc, vdupq_n_f32(pai[k]),
                            vld1q_f32(pp + (size_t)k * JBLK));
                    if constexpr (can_fuse) {
                        const float* bp = (fuse_nchw && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                        fuse_nchw_1(acc, bp[0], fmin, fmax);
                    }
                    vst1q_f32(pci + v, acc);
                }
                for (; v < je; ++v) {
                    float s = (k0 == 0) ? 0.0f : pci[v];
                    const float* pp = pb + (v - j0);
                    for (int k = 0; k < kc; ++k)
                        s += pai[k] * pp[(size_t)k * JBLK];
                    if constexpr (can_fuse) {
                        const float* bp = (fuse_nchw && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                        fuse_scalar(s, bp[0], fmin, fmax);
                    }
                    pci[v] = s;
                }
            }
        }
        // Post-process after all K-blocks complete
        if (!(can_fuse && post_fn.kind != post_op_kind::none))
            post_fn.apply_rows(i0, ie, C, m, j0, jw);
    });
}

// ============================================================================
// Weight pre-packing
// ============================================================================

// Pack A: pre-arrange A panels for dgemm_packed_a().
// Format: panels of IBLK×KC with 8-row interleaving for vfmaq_laneq_f32.
// @nnr-meta isa=scalar dtype=fp32 layout=NCHW special=GEMM
inline size_t pack_a_size(int n, int o) {
    constexpr int IBLK = 64, KC = 256;
    int ni = (n + IBLK - 1) / IBLK;
    int nk = (o + KC - 1) / KC;
    return (size_t)ni * nk * KC * IBLK;
}

// @nnr-meta isa=scalar dtype=fp32 layout=NCHW special=GEMM tiling=K
inline void pack_a(float* __restrict dst, const float* __restrict A, int n, int o) {
    constexpr int IBLK = 64, KC = 256;
    int ni = (n + IBLK - 1) / IBLK;
    int nk = (o + KC - 1) / KC;
    memset(dst, 0, (size_t)ni * nk * KC * IBLK * sizeof(float));
    for (int it = 0; it < ni; it++) {
        int i0 = it * IBLK;
        int iw = std::min(IBLK, n - i0);
        for (int kt = 0; kt < nk; kt++) {
            int k0 = kt * KC;
            int kc = std::min(KC, o - k0);
            float* panel = dst + ((size_t)it * nk + kt) * KC * IBLK;
            // Pack in 8-row groups for lane-based FMA
            int i = 0;
            for (; i + 8 <= iw; i += 8) {
                for (int k = 0; k < kc; k++) {
                    for (int r = 0; r < 8; r++)
                        panel[(size_t)i * KC + k * 8 + r] = A[(size_t)(i0 + i + r) * o + k0 + k];
                }
            }
            // Remainder rows: pack individually
            for (; i < iw; i++) {
                for (int k = 0; k < kc; k++)
                    panel[(size_t)i * KC + k] = A[(size_t)(i0 + i) * o + k0 + k];
            }
        }
    }
}

// Pack B: pre-arrange B panels for dgemm_packed_b().
// Same layout as x64: JBLK contiguous per K row.
// @nnr-meta isa=scalar dtype=fp32 layout=NCHW special=GEMM
inline size_t pack_b_size(int o, int m) {
    constexpr int JBLK = 64, KC = 256;
    int nj = (m + JBLK - 1) / JBLK;
    int nk = (o + KC - 1) / KC;
    return (size_t)nj * nk * KC * JBLK;
}

// @nnr-meta isa=scalar dtype=fp32 layout=NCHW special=GEMM tiling=K
inline void pack_b(float* __restrict dst, const float* __restrict B, int o, int m) {
    constexpr int JBLK = 64, KC = 256;
    int nj = (m + JBLK - 1) / JBLK;
    int nk = (o + KC - 1) / KC;
    memset(dst, 0, (size_t)nj * nk * KC * JBLK * sizeof(float));
    for (int jt = 0; jt < nj; jt++) {
        int j0 = jt * JBLK;
        int jw = std::min(JBLK, m - j0);
        for (int kt = 0; kt < nk; kt++) {
            int k0 = kt * KC;
            int kc = std::min(KC, o - k0);
            float* panel = dst + ((size_t)jt * nk + kt) * KC * JBLK;
            for (int k = 0; k < kc; k++)
                memcpy(panel + k * JBLK, B + (size_t)(k0 + k) * m + j0, jw * sizeof(float));
        }
    }
}

// ============================================================================
// GEMM with pre-packed A
// ============================================================================
template <typename PostFn>
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=GEMM tiling=[K,MR] fusion=post_op
inline void dgemm_packed_a(int n, int m, int o, const float* __restrict packed_A,
    const float* __restrict B, float* __restrict C, const PostFn& post_fn)
{
    constexpr bool can_fuse = PostFn::per_row_bias;
    constexpr int IBLK = 64, JBLK = 64, KC = 256;
    int ni = (n + IBLK - 1) / IBLK;
    int nj = (m + JBLK - 1) / JBLK;
    int nk = (o + KC - 1) / KC;
    int ntiles = ni * nj;
    bool par = ntiles > 1 && (int64_t)n * m * o > (1 << 21);

    NNR_POOL_ENSURE_SCRATCH((size_t)KC * JBLK * sizeof(float));

    nnr::for_dynamic(0, ntiles, par, [&](int tid, int tile) {
        float* pb = (float*)NNR_POOL_SCRATCH(tid);
        int it = tile / nj;
        int jt = tile % nj;
        int i0 = it * IBLK;
        int j0 = jt * JBLK;
        int ie = std::min(i0 + IBLK, n);
        int je = std::min(j0 + JBLK, m);
        int iw = ie - i0;
        int jw = je - j0;

        for (int kt = 0; kt < nk; kt++) {
            int k0 = kt * KC;
            int kc = std::min(KC, o - k0);
            bool last_k = (k0 + kc == o);

            float fmin = -FLT_MAX;
            float fmax = FLT_MAX;
            bool do_fuse = false;
            if constexpr (can_fuse) { if (last_k && post_fn.kind != post_op_kind::none) {
                do_fuse = true; fmin = post_fn.clip_min; fmax = post_fn.clip_max;
            }}

            // Pack B sub-panel
            for (int k = 0; k < kc; ++k)
                memcpy(pb + (size_t)k * JBLK, B + (size_t)(k0 + k) * m + j0, jw * sizeof(float));

            const float* pa_panel = packed_A + ((size_t)it * nk + kt) * KC * IBLK;

            int i = 0;
            for (; i + 8 <= iw; i += 8) {
                const float* pa = pa_panel + (size_t)i * KC;
                float* pc[8];
                for (int r = 0; r < 8; r++)
                    pc[r] = C + (size_t)(i0 + i + r) * m;

                int v = j0;
                for (; v + UK_NR <= je; v += UK_NR) {
                    const float* bp_uk = fused_zero_bias;
                    if constexpr (can_fuse) { if (do_fuse && post_fn.bias) bp_uk = post_fn.bias + post_fn.bias_off + i0 + i; }
                    ukernel_nchw(kc, pa, pb + (v - j0), JBLK, pc, v,
                        k0 == 0, do_fuse, bp_uk, fmin, fmax);
                }
                for (; v + 4 <= je; v += 4) {
                    const float* pp = pb + (v - j0);
                    float32x4_t c0, c1, c2, c3, c4, c5, c6, c7;
                    if (k0 == 0) {
                        c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = vdupq_n_f32(0.0f);
                    } else {
                        c0 = vld1q_f32(pc[0] + v); c1 = vld1q_f32(pc[1] + v);
                        c2 = vld1q_f32(pc[2] + v); c3 = vld1q_f32(pc[3] + v);
                        c4 = vld1q_f32(pc[4] + v); c5 = vld1q_f32(pc[5] + v);
                        c6 = vld1q_f32(pc[6] + v); c7 = vld1q_f32(pc[7] + v);
                    }
                    for (int k = 0; k < kc; k++) {
                        float32x4_t bv = vld1q_f32(pp + (size_t)k * JBLK);
                        float32x4_t a_lo = vld1q_f32(pa + k * 8);
                        float32x4_t a_hi = vld1q_f32(pa + k * 8 + 4);
                        c0 = vfmaq_laneq_f32(c0, bv, a_lo, 0);
                        c1 = vfmaq_laneq_f32(c1, bv, a_lo, 1);
                        c2 = vfmaq_laneq_f32(c2, bv, a_lo, 2);
                        c3 = vfmaq_laneq_f32(c3, bv, a_lo, 3);
                        c4 = vfmaq_laneq_f32(c4, bv, a_hi, 0);
                        c5 = vfmaq_laneq_f32(c5, bv, a_hi, 1);
                        c6 = vfmaq_laneq_f32(c6, bv, a_hi, 2);
                        c7 = vfmaq_laneq_f32(c7, bv, a_hi, 3);
                    }
                    if constexpr (can_fuse) {
                        const float* bp = (do_fuse && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i0 + i : fused_zero_bias;
                        fuse_nchw_8(c0, c1, c2, c3, c4, c5, c6, c7, bp, fmin, fmax);
                    }
                    vst1q_f32(pc[0] + v, c0); vst1q_f32(pc[1] + v, c1);
                    vst1q_f32(pc[2] + v, c2); vst1q_f32(pc[3] + v, c3);
                    vst1q_f32(pc[4] + v, c4); vst1q_f32(pc[5] + v, c5);
                    vst1q_f32(pc[6] + v, c6); vst1q_f32(pc[7] + v, c7);
                }
                for (; v < je; ++v) {
                    const float* pp = pb + (v - j0);
                    for (int r = 0; r < 8; r++) {
                        float s = (k0 == 0) ? 0.0f : pc[r][v];
                        for (int k = 0; k < kc; ++k)
                            s += pa[k * 8 + r] * pp[(size_t)k * JBLK];
                        if constexpr (can_fuse) {
                            const float* bp = (do_fuse && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i0 + i : fused_zero_bias;
                            fuse_scalar(s, bp[r], fmin, fmax);
                        }
                        pc[r][v] = s;
                    }
                }
            }
            // Remainder rows
            for (; i < iw; i++) {
                float* pci = C + (size_t)(i0 + i) * m;
                const float* pai = pa_panel + (size_t)i * KC;
                int v = j0;
                for (; v + 4 <= je; v += 4) {
                    const float* pp = pb + (v - j0);
                    float32x4_t acc = (k0 == 0) ? vdupq_n_f32(0.0f) : vld1q_f32(pci + v);
                    for (int k = 0; k < kc; ++k)
                        acc = vfmaq_f32(acc, vdupq_n_f32(pai[k]),
                            vld1q_f32(pp + (size_t)k * JBLK));
                    if constexpr (can_fuse) {
                        const float* bp = (do_fuse && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i0 + i : fused_zero_bias;
                        fuse_nchw_1(acc, bp[0], fmin, fmax);
                    }
                    vst1q_f32(pci + v, acc);
                }
                for (; v < je; ++v) {
                    float s = (k0 == 0) ? 0.0f : pci[v];
                    const float* pp = pb + (v - j0);
                    for (int k = 0; k < kc; ++k)
                        s += pai[k] * pp[(size_t)k * JBLK];
                    if constexpr (can_fuse) {
                        const float* bp = (do_fuse && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i0 + i : fused_zero_bias;
                        fuse_scalar(s, bp[0], fmin, fmax);
                    }
                    pci[v] = s;
                }
            }
        }
        if (!(can_fuse && post_fn.kind != post_op_kind::none))
            post_fn.apply_rows(i0, ie, C, m, j0, jw);
    });
}

// ============================================================================
// GEMM with pre-packed B
// ============================================================================
template <typename PostFn>
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=GEMM tiling=[K,MR] fusion=post_op
inline void dgemm_packed_b(int n, int m, int o, const float* __restrict A,
    const float* __restrict packed_B, float* __restrict C, const PostFn& post_fn)
{
    constexpr bool can_fuse = PostFn::per_row_bias;
    constexpr int JBLK = 64;
    constexpr int KC = 256;
    int nj = (m + JBLK - 1) / JBLK;
    int nk = (o + KC - 1) / KC;
    int ni = (n + JBLK - 1) / JBLK;
    int ntiles = ni * nj;
    bool par = ntiles > 1 && (int64_t)n * m * o > (1 << 21);

    NNR_POOL_ENSURE_SCRATCH((size_t)KC * 8 * sizeof(float));

    nnr::for_dynamic(0, ntiles, par, [&](int tid, int tile) {
        float* pa_pack = (float*)NNR_POOL_SCRATCH(tid);
        int i0 = (tile / nj) * JBLK;
        int j0 = (tile % nj) * JBLK;
        int jt = tile % nj;
        int ie = std::min(i0 + JBLK, n);
        int je = std::min(j0 + JBLK, m);
        int jw = je - j0;

        for (int kt = 0; kt < nk; kt++) {
            int k0 = kt * KC;
            int kc = std::min(KC, o - k0);
            bool last_k = (k0 + kc == o);

            const float* pb = packed_B + ((size_t)jt * nk + kt) * KC * JBLK;

            float fmin = -FLT_MAX;
            float fmax = FLT_MAX;
            bool fuse_nchw = false;
            if constexpr (can_fuse) { if (last_k && post_fn.kind != post_op_kind::none) {
                fuse_nchw = true; fmin = post_fn.clip_min; fmax = post_fn.clip_max;
            }}

            int i = i0;
            for (; i + 8 <= ie; i += 8) {
                const float* pa[8];
                float* pc[8];
                for (int r = 0; r < 8; r++) {
                    pa[r] = A + (size_t)(i + r) * o + k0;
                    pc[r] = C + (size_t)(i + r) * m;
                }
                for (int k = 0; k < kc; k++) {
                    pa_pack[k * 8 + 0] = pa[0][k];
                    pa_pack[k * 8 + 1] = pa[1][k];
                    pa_pack[k * 8 + 2] = pa[2][k];
                    pa_pack[k * 8 + 3] = pa[3][k];
                    pa_pack[k * 8 + 4] = pa[4][k];
                    pa_pack[k * 8 + 5] = pa[5][k];
                    pa_pack[k * 8 + 6] = pa[6][k];
                    pa_pack[k * 8 + 7] = pa[7][k];
                }
                int v = j0;
                for (; v + UK_NR <= je; v += UK_NR) {
                    const float* bp_uk = fused_zero_bias;
                    if constexpr (can_fuse) { if (fuse_nchw && post_fn.bias) bp_uk = post_fn.bias + post_fn.bias_off + i; }
                    ukernel_nchw(kc, pa_pack, pb + (v - j0), JBLK, pc, v,
                        k0 == 0, fuse_nchw, bp_uk, fmin, fmax);
                }
                for (; v + 4 <= je; v += 4) {
                    const float* pp = pb + (v - j0);
                    float32x4_t c0, c1, c2, c3, c4, c5, c6, c7;
                    if (k0 == 0) {
                        c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = vdupq_n_f32(0.0f);
                    } else {
                        c0 = vld1q_f32(pc[0] + v); c1 = vld1q_f32(pc[1] + v);
                        c2 = vld1q_f32(pc[2] + v); c3 = vld1q_f32(pc[3] + v);
                        c4 = vld1q_f32(pc[4] + v); c5 = vld1q_f32(pc[5] + v);
                        c6 = vld1q_f32(pc[6] + v); c7 = vld1q_f32(pc[7] + v);
                    }
                    for (int k = 0; k < kc; k++) {
                        float32x4_t bv = vld1q_f32(pp + (size_t)k * JBLK);
                        float32x4_t a_lo = vld1q_f32(pa_pack + k * 8);
                        float32x4_t a_hi = vld1q_f32(pa_pack + k * 8 + 4);
                        c0 = vfmaq_laneq_f32(c0, bv, a_lo, 0);
                        c1 = vfmaq_laneq_f32(c1, bv, a_lo, 1);
                        c2 = vfmaq_laneq_f32(c2, bv, a_lo, 2);
                        c3 = vfmaq_laneq_f32(c3, bv, a_lo, 3);
                        c4 = vfmaq_laneq_f32(c4, bv, a_hi, 0);
                        c5 = vfmaq_laneq_f32(c5, bv, a_hi, 1);
                        c6 = vfmaq_laneq_f32(c6, bv, a_hi, 2);
                        c7 = vfmaq_laneq_f32(c7, bv, a_hi, 3);
                    }
                    if constexpr (can_fuse) {
                        const float* bp = (fuse_nchw && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                        fuse_nchw_8(c0, c1, c2, c3, c4, c5, c6, c7, bp, fmin, fmax);
                    }
                    vst1q_f32(pc[0] + v, c0); vst1q_f32(pc[1] + v, c1);
                    vst1q_f32(pc[2] + v, c2); vst1q_f32(pc[3] + v, c3);
                    vst1q_f32(pc[4] + v, c4); vst1q_f32(pc[5] + v, c5);
                    vst1q_f32(pc[6] + v, c6); vst1q_f32(pc[7] + v, c7);
                }
                for (; v < je; ++v) {
                    const float* pp = pb + (v - j0);
                    for (int r = 0; r < 8; r++) {
                        float s = (k0 == 0) ? 0.0f : pc[r][v];
                        for (int k = 0; k < kc; ++k)
                            s += pa_pack[k * 8 + r] * pp[(size_t)k * JBLK];
                        if constexpr (can_fuse) {
                            const float* bp = (fuse_nchw && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                            fuse_scalar(s, bp[r], fmin, fmax);
                        }
                        pc[r][v] = s;
                    }
                }
            }
            for (; i < ie; i++) {
                float* pci = C + (size_t)i * m;
                const float* pai = A + (size_t)i * o + k0;
                int v = j0;
                for (; v + 4 <= je; v += 4) {
                    const float* pp = pb + (v - j0);
                    float32x4_t acc = (k0 == 0) ? vdupq_n_f32(0.0f) : vld1q_f32(pci + v);
                    for (int k = 0; k < kc; ++k)
                        acc = vfmaq_f32(acc, vdupq_n_f32(pai[k]),
                            vld1q_f32(pp + (size_t)k * JBLK));
                    if constexpr (can_fuse) {
                        const float* bp = (fuse_nchw && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                        fuse_nchw_1(acc, bp[0], fmin, fmax);
                    }
                    vst1q_f32(pci + v, acc);
                }
                for (; v < je; ++v) {
                    float s = (k0 == 0) ? 0.0f : pci[v];
                    const float* pp = pb + (v - j0);
                    for (int k = 0; k < kc; ++k)
                        s += pai[k] * pp[(size_t)k * JBLK];
                    if constexpr (can_fuse) {
                        const float* bp = (fuse_nchw && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                        fuse_scalar(s, bp[0], fmin, fmax);
                    }
                    pci[v] = s;
                }
            }
        }
        if (!(can_fuse && post_fn.kind != post_op_kind::none))
            post_fn.apply_rows(i0, ie, C, m, j0, jw);
    });
}

// ============================================================================
// NHWC-native GEMM with pre-packed B
// Optimized loop ordering for NHWC Conv: tiles spatial only, B stays L1-hot.
// ============================================================================
template <typename PostFn>
// @nnr-meta isa=NEON dtype=fp32 layout=NHWC special=GEMM tiling=[K,NR] fusion=post_op
inline void dgemm_nhwc(int n, int m, int o, const float* __restrict A,
    const float* __restrict packed_B, float* __restrict C, const PostFn& post_fn)
{
    constexpr bool can_fuse = PostFn::per_row_bias;
    constexpr int JBLK = 64, KC = 256;
    int nj = (m + JBLK - 1) / JBLK;
    int nk = (o + KC - 1) / KC;

    float fmin = -FLT_MAX, fmax = FLT_MAX;
    if constexpr (!can_fuse) {
        if (post_fn.kind == post_op_kind::relu || post_fn.kind == post_op_kind::clip) {
            fmin = post_fn.clip_min; fmax = post_fn.clip_max;
        }
    }

    for (int jt = 0; jt < nj; jt++) {
        int j0 = jt * JBLK;
        int je = std::min(j0 + JBLK, m);
        int jw = je - j0;

        for (int kt = 0; kt < nk; kt++) {
            int k0 = kt * KC;
            int kc = std::min(KC, o - k0);
            bool last_k = (k0 + kc == o);

            const float* pb = packed_B + ((size_t)jt * nk + kt) * KC * JBLK;

            // NHWC: bias is per-column, fuse on last K tile
            bool do_fuse = false;
            float f_min = -FLT_MAX, f_max = FLT_MAX;
            const float* bias_col = nullptr;
            if constexpr (!can_fuse) {
                if (last_k && post_fn.kind != post_op_kind::none) {
                    do_fuse = true;
                    f_min = post_fn.clip_min;
                    f_max = post_fn.clip_max;
                    bias_col = post_fn.bias;
                }
            }

            // Multi-row NHWC: process UK_MR rows at once to reuse B loads
            int n_full = (n / UK_MR) * UK_MR;
            bool fuse_bias = do_fuse && bias_col != nullptr;
            nnr::for_static(0, n / UK_MR, n > UK_MR, [&](int ig) {
                int i0 = ig * UK_MR;
                float* pc0 = C + (size_t)i0 * m;
                const float* pa0 = A + (size_t)i0 * o + k0;
                int v = j0;
                for (; v + UK_NR <= je; v += UK_NR)
                    ukernel_nhwc_mr(kc, pa0, o, pb + (v - j0), JBLK,
                        pc0, m, v, k0 == 0, fuse_bias, bias_col, f_min, f_max);
            });
            // Remainder rows (< UK_MR)
            for (int i = n_full; i < n; i++) {
                float* pci = C + (size_t)i * m;
                const float* pai = A + (size_t)i * o + k0;
                int v = j0;
                for (; v + UK_NR <= je; v += UK_NR)
                    ukernel_nhwc(kc, pai, pb + (v - j0), JBLK, pci, v,
                        k0 == 0, fuse_bias, bias_col, f_min, f_max);
                for (; v + 4 <= je; v += 4) {
                    const float* pp = pb + (v - j0);
                    float32x4_t acc = (k0 == 0) ? vdupq_n_f32(0.0f) : vld1q_f32(pci + v);
                    for (int k = 0; k < kc; ++k)
                        acc = vfmaq_f32(acc, vdupq_n_f32(pai[k]),
                            vld1q_f32(pp + (size_t)k * JBLK));
                    if constexpr (!can_fuse) {
                        if (fuse_bias) {
                            acc = vaddq_f32(acc, vld1q_f32(bias_col + v));
                            acc = vmaxq_f32(acc, vdupq_n_f32(f_min));
                            acc = vminq_f32(acc, vdupq_n_f32(f_max));
                        }
                    }
                    vst1q_f32(pci + v, acc);
                }
                for (; v < je; ++v) {
                    float s = (k0 == 0) ? 0.0f : pci[v];
                    const float* pp = pb + (v - j0);
                    for (int k = 0; k < kc; ++k)
                        s += pai[k] * pp[(size_t)k * JBLK];
                    if constexpr (!can_fuse) {
                        if (fuse_bias) {
                            s += bias_col[v];
                            s = std::max(s, f_min);
                            s = std::min(s, f_max);
                        }
                    }
                    pci[v] = s;
                }
            }

            // Apply non-fusible post-ops after last K tile
            if (last_k && !(do_fuse)) {
                for (int i = 0; i < n; i++)
                    post_fn.apply(i, C + (size_t)i * m + j0, jw, j0);
            }
        }
    }
}

// ============================================================================
// Batched GEMM for Winograd: 36 independent GEMMs
// ============================================================================
template <typename PostFn>
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=[GEMM,Winograd] tiling=K
inline void dgemm_packed_a_batch36(int n, int m, int o,
    const float* const packed_A_batch[36],
    const float* const B_batch[36],
    float* const C_batch[36],
    const PostFn& post_fn)
{
    for (int p = 0; p < 36; ++p)
        neon::dgemm_packed_a(n, m, o, packed_A_batch[p], B_batch[p], C_batch[p], post_fn);
}

template <typename PostFn>
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=[GEMM,Winograd] tiling=K
inline void dgemm_packed_b_batch36(int n, int m, int o,
    const float* const A_batch[36],
    const float* const packed_B_batch[36],
    float* const C_batch[36],
    const PostFn& post_fn)
{
    for (int p = 0; p < 36; ++p)
        neon::dgemm_packed_b(n, m, o, A_batch[p], packed_B_batch[p], C_batch[p], post_fn);
}

} // namespace neon
} // namespace nnr

#endif // __aarch64__ || _M_ARM64
