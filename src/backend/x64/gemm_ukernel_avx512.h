#pragma once
// AVX-512 micro-kernels for GEMM.
// Configuration: ISA=avx512, MR=8, NR=16

#ifdef NNR_ARCH_X64

#include <immintrin.h>
#ifdef NNR_USE_XBYAK
#include "jit_gemm_ukernel.h"
#endif

#ifndef NNR_FORCEINLINE
#ifdef _MSC_VER
#define NNR_FORCEINLINE __forceinline
#else
#define NNR_FORCEINLINE inline __attribute__((always_inline))
#endif
#endif

namespace nnr { namespace avx512 {

static constexpr int UK_MR = 8;
static constexpr int UK_NR = 16;

// NCHW micro-kernel: 8 rows x 16 cols, 8 FMA per K step
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=GEMM tiling=[MR,NR]
NNR_FORCEINLINE void ukernel_nchw(
    int kc,
    const float* __restrict pa,
    const float* __restrict pb,
    int pb_stride,
    float* const __restrict pc[8],
    int v,
    bool zero_init,
    bool do_fuse,
    const float* bp,
    float fmin, float fmax)
{
#if 0 // LOCUST
;MR = 8
    __m512 c0, c1, c2, c3, c4, c5, c6, c7;
    if (zero_init) {
        c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = _mm512_setzero_ps();
    } else {
;for r in range(MR):
        c@r@ = _mm512_loadu_ps(pc[@r@] + v);
;    pass
    }
#else // LOCUST
    __m512 c0, c1, c2, c3, c4, c5, c6, c7;
    if (zero_init) {
        c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = _mm512_setzero_ps();
    } else {
        c0 = _mm512_loadu_ps(pc[0] + v);
        c1 = _mm512_loadu_ps(pc[1] + v);
        c2 = _mm512_loadu_ps(pc[2] + v);
        c3 = _mm512_loadu_ps(pc[3] + v);
        c4 = _mm512_loadu_ps(pc[4] + v);
        c5 = _mm512_loadu_ps(pc[5] + v);
        c6 = _mm512_loadu_ps(pc[6] + v);
        c7 = _mm512_loadu_ps(pc[7] + v);
    }
#endif // LOCUST
    {
        int k = 0;
        // K-4x unrolled main loop: 32 FMAs per iteration, reduces loop overhead
        for (; k + 4 <= kc; k += 4) {
            if (k + 8 < kc) _mm_prefetch((const char*)(pb + (size_t)(k+8) * pb_stride), _MM_HINT_T0);
            __m512 bv0 = _mm512_loadu_ps(pb + (size_t)k * pb_stride);
            __m512 bv1 = _mm512_loadu_ps(pb + (size_t)(k+1) * pb_stride);
            __m512 bv2 = _mm512_loadu_ps(pb + (size_t)(k+2) * pb_stride);
            __m512 bv3 = _mm512_loadu_ps(pb + (size_t)(k+3) * pb_stride);
            const float* ap0 = pa + k * 8;
            const float* ap1 = ap0 + 8;
            const float* ap2 = ap1 + 8;
            const float* ap3 = ap2 + 8;
            c0 = _mm512_fmadd_ps(_mm512_set1_ps(ap0[0]), bv0, c0);
            c1 = _mm512_fmadd_ps(_mm512_set1_ps(ap0[1]), bv0, c1);
            c2 = _mm512_fmadd_ps(_mm512_set1_ps(ap0[2]), bv0, c2);
            c3 = _mm512_fmadd_ps(_mm512_set1_ps(ap0[3]), bv0, c3);
            c4 = _mm512_fmadd_ps(_mm512_set1_ps(ap0[4]), bv0, c4);
            c5 = _mm512_fmadd_ps(_mm512_set1_ps(ap0[5]), bv0, c5);
            c6 = _mm512_fmadd_ps(_mm512_set1_ps(ap0[6]), bv0, c6);
            c7 = _mm512_fmadd_ps(_mm512_set1_ps(ap0[7]), bv0, c7);
            c0 = _mm512_fmadd_ps(_mm512_set1_ps(ap1[0]), bv1, c0);
            c1 = _mm512_fmadd_ps(_mm512_set1_ps(ap1[1]), bv1, c1);
            c2 = _mm512_fmadd_ps(_mm512_set1_ps(ap1[2]), bv1, c2);
            c3 = _mm512_fmadd_ps(_mm512_set1_ps(ap1[3]), bv1, c3);
            c4 = _mm512_fmadd_ps(_mm512_set1_ps(ap1[4]), bv1, c4);
            c5 = _mm512_fmadd_ps(_mm512_set1_ps(ap1[5]), bv1, c5);
            c6 = _mm512_fmadd_ps(_mm512_set1_ps(ap1[6]), bv1, c6);
            c7 = _mm512_fmadd_ps(_mm512_set1_ps(ap1[7]), bv1, c7);
            c0 = _mm512_fmadd_ps(_mm512_set1_ps(ap2[0]), bv2, c0);
            c1 = _mm512_fmadd_ps(_mm512_set1_ps(ap2[1]), bv2, c1);
            c2 = _mm512_fmadd_ps(_mm512_set1_ps(ap2[2]), bv2, c2);
            c3 = _mm512_fmadd_ps(_mm512_set1_ps(ap2[3]), bv2, c3);
            c4 = _mm512_fmadd_ps(_mm512_set1_ps(ap2[4]), bv2, c4);
            c5 = _mm512_fmadd_ps(_mm512_set1_ps(ap2[5]), bv2, c5);
            c6 = _mm512_fmadd_ps(_mm512_set1_ps(ap2[6]), bv2, c6);
            c7 = _mm512_fmadd_ps(_mm512_set1_ps(ap2[7]), bv2, c7);
            c0 = _mm512_fmadd_ps(_mm512_set1_ps(ap3[0]), bv3, c0);
            c1 = _mm512_fmadd_ps(_mm512_set1_ps(ap3[1]), bv3, c1);
            c2 = _mm512_fmadd_ps(_mm512_set1_ps(ap3[2]), bv3, c2);
            c3 = _mm512_fmadd_ps(_mm512_set1_ps(ap3[3]), bv3, c3);
            c4 = _mm512_fmadd_ps(_mm512_set1_ps(ap3[4]), bv3, c4);
            c5 = _mm512_fmadd_ps(_mm512_set1_ps(ap3[5]), bv3, c5);
            c6 = _mm512_fmadd_ps(_mm512_set1_ps(ap3[6]), bv3, c6);
            c7 = _mm512_fmadd_ps(_mm512_set1_ps(ap3[7]), bv3, c7);
        }
        // K-1x remainder (handles 0-3 remaining)
        for (; k < kc; k++) {
            __m512 bv = _mm512_loadu_ps(pb + (size_t)k * pb_stride);
            const float* ap = pa + k * 8;
            c0 = _mm512_fmadd_ps(_mm512_set1_ps(ap[0]), bv, c0);
            c1 = _mm512_fmadd_ps(_mm512_set1_ps(ap[1]), bv, c1);
            c2 = _mm512_fmadd_ps(_mm512_set1_ps(ap[2]), bv, c2);
            c3 = _mm512_fmadd_ps(_mm512_set1_ps(ap[3]), bv, c3);
            c4 = _mm512_fmadd_ps(_mm512_set1_ps(ap[4]), bv, c4);
            c5 = _mm512_fmadd_ps(_mm512_set1_ps(ap[5]), bv, c5);
            c6 = _mm512_fmadd_ps(_mm512_set1_ps(ap[6]), bv, c6);
            c7 = _mm512_fmadd_ps(_mm512_set1_ps(ap[7]), bv, c7);
        }
    }
#if 0 // LOCUST
    if (do_fuse) {
        __m512 vmin = _mm512_set1_ps(fmin);
        __m512 vmax = _mm512_set1_ps(fmax);
;for r in range(MR):
        c@r@ = _mm512_add_ps(c@r@, _mm512_set1_ps(bp[@r@]));
;    pass
;for r in range(MR):
        c@r@ = _mm512_max_ps(c@r@, vmin); c@r@ = _mm512_min_ps(c@r@, vmax);
;    pass
    }
;for r in range(MR):
    _mm512_storeu_ps(pc[@r@] + v, c@r@);
;    pass
#else // LOCUST
    if (do_fuse) {
        __m512 vmin = _mm512_set1_ps(fmin);
        __m512 vmax = _mm512_set1_ps(fmax);
        c0 = _mm512_add_ps(c0, _mm512_set1_ps(bp[0]));
        c1 = _mm512_add_ps(c1, _mm512_set1_ps(bp[1]));
        c2 = _mm512_add_ps(c2, _mm512_set1_ps(bp[2]));
        c3 = _mm512_add_ps(c3, _mm512_set1_ps(bp[3]));
        c4 = _mm512_add_ps(c4, _mm512_set1_ps(bp[4]));
        c5 = _mm512_add_ps(c5, _mm512_set1_ps(bp[5]));
        c6 = _mm512_add_ps(c6, _mm512_set1_ps(bp[6]));
        c7 = _mm512_add_ps(c7, _mm512_set1_ps(bp[7]));
        c0 = _mm512_max_ps(c0, vmin);
        c1 = _mm512_max_ps(c1, vmin);
        c2 = _mm512_max_ps(c2, vmin);
        c3 = _mm512_max_ps(c3, vmin);
        c4 = _mm512_max_ps(c4, vmin);
        c5 = _mm512_max_ps(c5, vmin);
        c6 = _mm512_max_ps(c6, vmin);
        c7 = _mm512_max_ps(c7, vmin);
        c0 = _mm512_min_ps(c0, vmax);
        c1 = _mm512_min_ps(c1, vmax);
        c2 = _mm512_min_ps(c2, vmax);
        c3 = _mm512_min_ps(c3, vmax);
        c4 = _mm512_min_ps(c4, vmax);
        c5 = _mm512_min_ps(c5, vmax);
        c6 = _mm512_min_ps(c6, vmax);
        c7 = _mm512_min_ps(c7, vmax);
    }
    _mm512_storeu_ps(pc[0] + v, c0);
    _mm512_storeu_ps(pc[1] + v, c1);
    _mm512_storeu_ps(pc[2] + v, c2);
    _mm512_storeu_ps(pc[3] + v, c3);
    _mm512_storeu_ps(pc[4] + v, c4);
    _mm512_storeu_ps(pc[5] + v, c5);
    _mm512_storeu_ps(pc[6] + v, c6);
    _mm512_storeu_ps(pc[7] + v, c7);
#endif // LOCUST
}

// NCHW double-wide micro-kernel: 8 rows × 32 cols (2 NR blocks).
// Processes two adjacent B panels per K-step, doubling FMA throughput
// to match the broadcast rate (8 broadcasts → 16 FMAs → 2 FMA/broadcast).
// Register usage: 16 accumulators (c0L..c7L, c0R..c7R) + 2 B + 1 A = 19 zmm.
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=GEMM tiling=[MR,NR]
NNR_FORCEINLINE void ukernel_nchw_2x(
    int kc,
    const float* __restrict pa,
    const float* __restrict pb_left,
    const float* __restrict pb_right,
    int pb_stride,
    float* const __restrict pc[8],
    int v_left, int v_right,
    bool zero_init,
    bool do_fuse,
    const float* bp,
    float fmin, float fmax)
{
    __m512 cL0, cL1, cL2, cL3, cL4, cL5, cL6, cL7;
    __m512 cR0, cR1, cR2, cR3, cR4, cR5, cR6, cR7;
    if (zero_init) {
        cL0 = cL1 = cL2 = cL3 = cL4 = cL5 = cL6 = cL7 = _mm512_setzero_ps();
        cR0 = cR1 = cR2 = cR3 = cR4 = cR5 = cR6 = cR7 = _mm512_setzero_ps();
    } else {
        cL0 = _mm512_loadu_ps(pc[0] + v_left);  cR0 = _mm512_loadu_ps(pc[0] + v_right);
        cL1 = _mm512_loadu_ps(pc[1] + v_left);  cR1 = _mm512_loadu_ps(pc[1] + v_right);
        cL2 = _mm512_loadu_ps(pc[2] + v_left);  cR2 = _mm512_loadu_ps(pc[2] + v_right);
        cL3 = _mm512_loadu_ps(pc[3] + v_left);  cR3 = _mm512_loadu_ps(pc[3] + v_right);
        cL4 = _mm512_loadu_ps(pc[4] + v_left);  cR4 = _mm512_loadu_ps(pc[4] + v_right);
        cL5 = _mm512_loadu_ps(pc[5] + v_left);  cR5 = _mm512_loadu_ps(pc[5] + v_right);
        cL6 = _mm512_loadu_ps(pc[6] + v_left);  cR6 = _mm512_loadu_ps(pc[6] + v_right);
        cL7 = _mm512_loadu_ps(pc[7] + v_left);  cR7 = _mm512_loadu_ps(pc[7] + v_right);
    }
    {
        int k = 0;
        for (; k + 2 <= kc; k += 2) {
            __m512 bL0 = _mm512_loadu_ps(pb_left  + (size_t)k * pb_stride);
            __m512 bR0 = _mm512_loadu_ps(pb_right + (size_t)k * pb_stride);
            __m512 bL1 = _mm512_loadu_ps(pb_left  + (size_t)(k+1) * pb_stride);
            __m512 bR1 = _mm512_loadu_ps(pb_right + (size_t)(k+1) * pb_stride);
            const float* ap0 = pa + k * 8;
            const float* ap1 = ap0 + 8;
            __m512 a0;
            a0 = _mm512_set1_ps(ap0[0]); cL0 = _mm512_fmadd_ps(a0, bL0, cL0); cR0 = _mm512_fmadd_ps(a0, bR0, cR0);
            a0 = _mm512_set1_ps(ap0[1]); cL1 = _mm512_fmadd_ps(a0, bL0, cL1); cR1 = _mm512_fmadd_ps(a0, bR0, cR1);
            a0 = _mm512_set1_ps(ap0[2]); cL2 = _mm512_fmadd_ps(a0, bL0, cL2); cR2 = _mm512_fmadd_ps(a0, bR0, cR2);
            a0 = _mm512_set1_ps(ap0[3]); cL3 = _mm512_fmadd_ps(a0, bL0, cL3); cR3 = _mm512_fmadd_ps(a0, bR0, cR3);
            a0 = _mm512_set1_ps(ap0[4]); cL4 = _mm512_fmadd_ps(a0, bL0, cL4); cR4 = _mm512_fmadd_ps(a0, bR0, cR4);
            a0 = _mm512_set1_ps(ap0[5]); cL5 = _mm512_fmadd_ps(a0, bL0, cL5); cR5 = _mm512_fmadd_ps(a0, bR0, cR5);
            a0 = _mm512_set1_ps(ap0[6]); cL6 = _mm512_fmadd_ps(a0, bL0, cL6); cR6 = _mm512_fmadd_ps(a0, bR0, cR6);
            a0 = _mm512_set1_ps(ap0[7]); cL7 = _mm512_fmadd_ps(a0, bL0, cL7); cR7 = _mm512_fmadd_ps(a0, bR0, cR7);
            a0 = _mm512_set1_ps(ap1[0]); cL0 = _mm512_fmadd_ps(a0, bL1, cL0); cR0 = _mm512_fmadd_ps(a0, bR1, cR0);
            a0 = _mm512_set1_ps(ap1[1]); cL1 = _mm512_fmadd_ps(a0, bL1, cL1); cR1 = _mm512_fmadd_ps(a0, bR1, cR1);
            a0 = _mm512_set1_ps(ap1[2]); cL2 = _mm512_fmadd_ps(a0, bL1, cL2); cR2 = _mm512_fmadd_ps(a0, bR1, cR2);
            a0 = _mm512_set1_ps(ap1[3]); cL3 = _mm512_fmadd_ps(a0, bL1, cL3); cR3 = _mm512_fmadd_ps(a0, bR1, cR3);
            a0 = _mm512_set1_ps(ap1[4]); cL4 = _mm512_fmadd_ps(a0, bL1, cL4); cR4 = _mm512_fmadd_ps(a0, bR1, cR4);
            a0 = _mm512_set1_ps(ap1[5]); cL5 = _mm512_fmadd_ps(a0, bL1, cL5); cR5 = _mm512_fmadd_ps(a0, bR1, cR5);
            a0 = _mm512_set1_ps(ap1[6]); cL6 = _mm512_fmadd_ps(a0, bL1, cL6); cR6 = _mm512_fmadd_ps(a0, bR1, cR6);
            a0 = _mm512_set1_ps(ap1[7]); cL7 = _mm512_fmadd_ps(a0, bL1, cL7); cR7 = _mm512_fmadd_ps(a0, bR1, cR7);
        }
        if (k < kc) {
            __m512 bL = _mm512_loadu_ps(pb_left  + (size_t)k * pb_stride);
            __m512 bR = _mm512_loadu_ps(pb_right + (size_t)k * pb_stride);
            const float* ap = pa + k * 8;
            __m512 a0;
            a0 = _mm512_set1_ps(ap[0]); cL0 = _mm512_fmadd_ps(a0, bL, cL0); cR0 = _mm512_fmadd_ps(a0, bR, cR0);
            a0 = _mm512_set1_ps(ap[1]); cL1 = _mm512_fmadd_ps(a0, bL, cL1); cR1 = _mm512_fmadd_ps(a0, bR, cR1);
            a0 = _mm512_set1_ps(ap[2]); cL2 = _mm512_fmadd_ps(a0, bL, cL2); cR2 = _mm512_fmadd_ps(a0, bR, cR2);
            a0 = _mm512_set1_ps(ap[3]); cL3 = _mm512_fmadd_ps(a0, bL, cL3); cR3 = _mm512_fmadd_ps(a0, bR, cR3);
            a0 = _mm512_set1_ps(ap[4]); cL4 = _mm512_fmadd_ps(a0, bL, cL4); cR4 = _mm512_fmadd_ps(a0, bR, cR4);
            a0 = _mm512_set1_ps(ap[5]); cL5 = _mm512_fmadd_ps(a0, bL, cL5); cR5 = _mm512_fmadd_ps(a0, bR, cR5);
            a0 = _mm512_set1_ps(ap[6]); cL6 = _mm512_fmadd_ps(a0, bL, cL6); cR6 = _mm512_fmadd_ps(a0, bR, cR6);
            a0 = _mm512_set1_ps(ap[7]); cL7 = _mm512_fmadd_ps(a0, bL, cL7); cR7 = _mm512_fmadd_ps(a0, bR, cR7);
        }
    }
    if (do_fuse) {
        __m512 vmin = _mm512_set1_ps(fmin);
        __m512 vmax = _mm512_set1_ps(fmax);
        cL0 = _mm512_add_ps(cL0, _mm512_set1_ps(bp[0])); cR0 = _mm512_add_ps(cR0, _mm512_set1_ps(bp[0]));
        cL1 = _mm512_add_ps(cL1, _mm512_set1_ps(bp[1])); cR1 = _mm512_add_ps(cR1, _mm512_set1_ps(bp[1]));
        cL2 = _mm512_add_ps(cL2, _mm512_set1_ps(bp[2])); cR2 = _mm512_add_ps(cR2, _mm512_set1_ps(bp[2]));
        cL3 = _mm512_add_ps(cL3, _mm512_set1_ps(bp[3])); cR3 = _mm512_add_ps(cR3, _mm512_set1_ps(bp[3]));
        cL4 = _mm512_add_ps(cL4, _mm512_set1_ps(bp[4])); cR4 = _mm512_add_ps(cR4, _mm512_set1_ps(bp[4]));
        cL5 = _mm512_add_ps(cL5, _mm512_set1_ps(bp[5])); cR5 = _mm512_add_ps(cR5, _mm512_set1_ps(bp[5]));
        cL6 = _mm512_add_ps(cL6, _mm512_set1_ps(bp[6])); cR6 = _mm512_add_ps(cR6, _mm512_set1_ps(bp[6]));
        cL7 = _mm512_add_ps(cL7, _mm512_set1_ps(bp[7])); cR7 = _mm512_add_ps(cR7, _mm512_set1_ps(bp[7]));
        cL0 = _mm512_max_ps(cL0, vmin); cL0 = _mm512_min_ps(cL0, vmax);
        cL1 = _mm512_max_ps(cL1, vmin); cL1 = _mm512_min_ps(cL1, vmax);
        cL2 = _mm512_max_ps(cL2, vmin); cL2 = _mm512_min_ps(cL2, vmax);
        cL3 = _mm512_max_ps(cL3, vmin); cL3 = _mm512_min_ps(cL3, vmax);
        cL4 = _mm512_max_ps(cL4, vmin); cL4 = _mm512_min_ps(cL4, vmax);
        cL5 = _mm512_max_ps(cL5, vmin); cL5 = _mm512_min_ps(cL5, vmax);
        cL6 = _mm512_max_ps(cL6, vmin); cL6 = _mm512_min_ps(cL6, vmax);
        cL7 = _mm512_max_ps(cL7, vmin); cL7 = _mm512_min_ps(cL7, vmax);
        cR0 = _mm512_max_ps(cR0, vmin); cR0 = _mm512_min_ps(cR0, vmax);
        cR1 = _mm512_max_ps(cR1, vmin); cR1 = _mm512_min_ps(cR1, vmax);
        cR2 = _mm512_max_ps(cR2, vmin); cR2 = _mm512_min_ps(cR2, vmax);
        cR3 = _mm512_max_ps(cR3, vmin); cR3 = _mm512_min_ps(cR3, vmax);
        cR4 = _mm512_max_ps(cR4, vmin); cR4 = _mm512_min_ps(cR4, vmax);
        cR5 = _mm512_max_ps(cR5, vmin); cR5 = _mm512_min_ps(cR5, vmax);
        cR6 = _mm512_max_ps(cR6, vmin); cR6 = _mm512_min_ps(cR6, vmax);
        cR7 = _mm512_max_ps(cR7, vmin); cR7 = _mm512_min_ps(cR7, vmax);
    }
    _mm512_storeu_ps(pc[0] + v_left, cL0);  _mm512_storeu_ps(pc[0] + v_right, cR0);
    _mm512_storeu_ps(pc[1] + v_left, cL1);  _mm512_storeu_ps(pc[1] + v_right, cR1);
    _mm512_storeu_ps(pc[2] + v_left, cL2);  _mm512_storeu_ps(pc[2] + v_right, cR2);
    _mm512_storeu_ps(pc[3] + v_left, cL3);  _mm512_storeu_ps(pc[3] + v_right, cR3);
    _mm512_storeu_ps(pc[4] + v_left, cL4);  _mm512_storeu_ps(pc[4] + v_right, cR4);
    _mm512_storeu_ps(pc[5] + v_left, cL5);  _mm512_storeu_ps(pc[5] + v_right, cR5);
    _mm512_storeu_ps(pc[6] + v_left, cL6);  _mm512_storeu_ps(pc[6] + v_right, cR6);
    _mm512_storeu_ps(pc[7] + v_left, cL7);  _mm512_storeu_ps(pc[7] + v_right, cR7);
}

// NHWC micro-kernel: 1 row x 16 cols
// @nnr-meta isa=AVX512 dtype=fp32 layout=NHWC special=GEMM tiling=[MR,NR]
NNR_FORCEINLINE void ukernel_nhwc(
    int kc,
    const float* __restrict pa,
    const float* __restrict pb,
    int pb_stride,
    float* __restrict pc,
    int v,
    bool zero_init,
    bool do_fuse,
    const float* bias_col,
    float fmin, float fmax)
{
    __m512 a;
    if (zero_init) {
        a = _mm512_setzero_ps();
    } else {
        a = _mm512_loadu_ps(pc + v);
    }
    for (int k = 0; k < kc; ++k) {
        a = _mm512_fmadd_ps(_mm512_set1_ps(pa[k]),
            _mm512_loadu_ps(pb + (size_t)k * pb_stride), a);
    }
    if (do_fuse) {
        a = _mm512_add_ps(a, _mm512_loadu_ps(bias_col + v));
        a = _mm512_max_ps(a, _mm512_set1_ps(fmin));
        a = _mm512_min_ps(a, _mm512_set1_ps(fmax));
    }
    _mm512_storeu_ps(pc + v, a);
}

#ifdef NNR_USE_XBYAK
// Resolve JIT micro-kernel function pointer for (zero_init, do_fuse).
// 4 entries max (2 bools). Thread-safe, zero overhead after first call.
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=[GEMM,JIT]
inline jit_ukernel_fn_t resolve_jit_ukernel(bool zero_init, bool do_fuse,
                                              float fmin, float fmax) {
    static jit_dispatch_t<jit_ukernel_key_t, jit_ukernel_avx512_t,
                          jit_ukernel_fn_t, jit_ukernel_hash_t, 4> dispatch;
    return dispatch.resolve({zero_init, do_fuse}, zero_init, do_fuse, fmin, fmax);
}
#endif

// JIT micro-kernel dispatch: uses JIT when NNR_USE_XBYAK is defined,
// falls back to intrinsics ukernel_nchw() otherwise.
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=[GEMM,JIT] tiling=[MR,NR]
NNR_FORCEINLINE void ukernel_nchw_jit(
    int kc,
    const float* __restrict pa,
    const float* __restrict pb,
    int pb_stride,
    float* const __restrict pc[8],
    int v,
    bool zero_init,
    bool do_fuse,
    const float* bp,
    float fmin, float fmax)
{
#ifdef NNR_USE_XBYAK
    auto fn = resolve_jit_ukernel(zero_init, do_fuse, fmin, fmax);
    fn(kc, pa, pb, pb_stride, pc, v, bp);
#else
    ukernel_nchw(kc, pa, pb, pb_stride, pc, v, zero_init, do_fuse, bp, fmin, fmax);
#endif
}

} // namespace avx512
} // namespace nnr

#endif // NNR_ARCH_X64
