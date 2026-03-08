#pragma once
// Configuration: ISA=avx2, MR=6, NR=8

#ifdef NNR_ARCH_X64

#include <immintrin.h>

#ifndef NNR_FORCEINLINE
#ifdef _MSC_VER
#define NNR_FORCEINLINE __forceinline
#else
#define NNR_FORCEINLINE inline __attribute__((always_inline))
#endif
#endif

namespace nnr { namespace avx2 {

static constexpr int UK_MR = 6;
static constexpr int UK_NR = 8;

// NCHW micro-kernel: 6 rows x 8 cols, 6 FMA per K step
// @nnr-meta isa=AVX2 dtype=fp32 layout=NCHW special=GEMM tiling=[MR,NR]
NNR_FORCEINLINE void ukernel_nchw(
    int kc,
    const float* __restrict pa,
    const float* __restrict pb,
    int pb_stride,
    float* const __restrict pc[6],
    int v,
    bool zero_init,
    bool do_fuse,
    const float* bp,
    float fmin, float fmax)
{
#if 0 // LOCUST
;MR = 6
;decl = ", ".join(f"c{r}" for r in range(MR))
     __m256 @decl@;
     if (zero_init) {
;chain = " = ".join(f"c{r}" for r in range(MR))
         @chain@ = _mm256_setzero_ps();
     } else {
;for r in range(MR):
         c@r@ = _mm256_loadu_ps(pc[@r@] + v);
;    pass
     }
     for (int k = 0; k < kc; k++) {
         __m256 bv = _mm256_loadu_ps(pb + (size_t)k * pb_stride);
         const float* ap = pa + k * @MR@;
;for r in range(MR):
         c@r@ = _mm256_fmadd_ps(_mm256_set1_ps(ap[@r@]), bv, c@r@);
;    pass
     }
     if (do_fuse) {
         __m256 vmin = _mm256_set1_ps(fmin);
         __m256 vmax = _mm256_set1_ps(fmax);
;for r in range(MR):
         c@r@ = _mm256_add_ps(c@r@, _mm256_set1_ps(bp[@r@]));
;    pass
;for r in range(MR):
         c@r@ = _mm256_max_ps(c@r@, vmin); c@r@ = _mm256_min_ps(c@r@, vmax);
;    pass
     }
;for r in range(MR):
     _mm256_storeu_ps(pc[@r@] + v, c@r@);
;    pass
#else // LOCUST
     __m256 c0, c1, c2, c3, c4, c5;
     if (zero_init) {
         c0 = c1 = c2 = c3 = c4 = c5 = _mm256_setzero_ps();
     } else {
         c0 = _mm256_loadu_ps(pc[0] + v);
         c1 = _mm256_loadu_ps(pc[1] + v);
         c2 = _mm256_loadu_ps(pc[2] + v);
         c3 = _mm256_loadu_ps(pc[3] + v);
         c4 = _mm256_loadu_ps(pc[4] + v);
         c5 = _mm256_loadu_ps(pc[5] + v);
     }
     for (int k = 0; k < kc; k++) {
         __m256 bv = _mm256_loadu_ps(pb + (size_t)k * pb_stride);
         const float* ap = pa + k * 6;
         c0 = _mm256_fmadd_ps(_mm256_set1_ps(ap[0]), bv, c0);
         c1 = _mm256_fmadd_ps(_mm256_set1_ps(ap[1]), bv, c1);
         c2 = _mm256_fmadd_ps(_mm256_set1_ps(ap[2]), bv, c2);
         c3 = _mm256_fmadd_ps(_mm256_set1_ps(ap[3]), bv, c3);
         c4 = _mm256_fmadd_ps(_mm256_set1_ps(ap[4]), bv, c4);
         c5 = _mm256_fmadd_ps(_mm256_set1_ps(ap[5]), bv, c5);
     }
     if (do_fuse) {
         __m256 vmin = _mm256_set1_ps(fmin);
         __m256 vmax = _mm256_set1_ps(fmax);
         c0 = _mm256_add_ps(c0, _mm256_set1_ps(bp[0]));
         c1 = _mm256_add_ps(c1, _mm256_set1_ps(bp[1]));
         c2 = _mm256_add_ps(c2, _mm256_set1_ps(bp[2]));
         c3 = _mm256_add_ps(c3, _mm256_set1_ps(bp[3]));
         c4 = _mm256_add_ps(c4, _mm256_set1_ps(bp[4]));
         c5 = _mm256_add_ps(c5, _mm256_set1_ps(bp[5]));
         c0 = _mm256_max_ps(c0, vmin); c0 = _mm256_min_ps(c0, vmax);
         c1 = _mm256_max_ps(c1, vmin); c1 = _mm256_min_ps(c1, vmax);
         c2 = _mm256_max_ps(c2, vmin); c2 = _mm256_min_ps(c2, vmax);
         c3 = _mm256_max_ps(c3, vmin); c3 = _mm256_min_ps(c3, vmax);
         c4 = _mm256_max_ps(c4, vmin); c4 = _mm256_min_ps(c4, vmax);
         c5 = _mm256_max_ps(c5, vmin); c5 = _mm256_min_ps(c5, vmax);
     }
     _mm256_storeu_ps(pc[0] + v, c0);
     _mm256_storeu_ps(pc[1] + v, c1);
     _mm256_storeu_ps(pc[2] + v, c2);
     _mm256_storeu_ps(pc[3] + v, c3);
     _mm256_storeu_ps(pc[4] + v, c4);
     _mm256_storeu_ps(pc[5] + v, c5);
#endif // LOCUST
}

// NHWC micro-kernel: 1 row x 8 cols
// @nnr-meta isa=AVX2 dtype=fp32 layout=NHWC special=GEMM tiling=[MR,NR]
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
    __m256 a;
    if (zero_init) {
        a = _mm256_setzero_ps();
    } else {
        a = _mm256_loadu_ps(pc + v);
    }
    for (int k = 0; k < kc; ++k) {
        a = _mm256_fmadd_ps(_mm256_set1_ps(pa[k]),
            _mm256_loadu_ps(pb + (size_t)k * pb_stride), a);
    }
    if (do_fuse) {
        a = _mm256_add_ps(a, _mm256_loadu_ps(bias_col + v));
        a = _mm256_max_ps(a, _mm256_set1_ps(fmin));
        a = _mm256_min_ps(a, _mm256_set1_ps(fmax));
    }
    _mm256_storeu_ps(pc + v, a);
}

} // namespace avx2
} // namespace nnr

#endif // NNR_ARCH_X64
