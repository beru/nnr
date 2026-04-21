#pragma once
// AVX-512 GEMM backend.
// Included from gemm_kernel.h — requires gemm_post_t to be defined before inclusion.

#include <immintrin.h>
#include <cfloat>
#include "thread_pool.h"
#include "backend/x64/gemm_ukernel_avx512.h"
#include "profiler.h"

namespace nnr {
// Zero buffer for register-fused post-op: bias pointer redirected here when
// no bias is needed, so fusion code can load unconditionally (no branch in v-loop).
alignas(64) inline constexpr float fused_zero_bias[64] = {};
namespace avx512 {

// Fused post-op helpers: bias + clamp on accumulators.
// Extracted from repeated epilogue code across all GEMM variants.

#if 0 // LOCUST
;def gen_fuse_nchw(nr):
;    args = ", ".join(f"__m512& c{r}" for r in range(nr))
inline void fuse_nchw_@nr@(@args@,
                        const float* bp, float fmin, float fmax) {
;    for r in range(nr):
    c@r@ = _mm512_add_ps(c@r@, _mm512_set1_ps(bp[@r@]));
;        pass
    __m512 vmin = _mm512_set1_ps(fmin);
    __m512 vmax = _mm512_set1_ps(fmax);
;    for r in range(nr):
    c@r@ = _mm512_max_ps(c@r@, vmin); c@r@ = _mm512_min_ps(c@r@, vmax);
;        pass
}
;    pass
;
;gen_fuse_nchw(8)

;gen_fuse_nchw(6)
#else // LOCUST
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW fusion=post_op
inline void fuse_nchw_8(__m512& c0, __m512& c1, __m512& c2, __m512& c3, __m512& c4, __m512& c5, __m512& c6, __m512& c7,
                        const float* bp, float fmin, float fmax) {
    c0 = _mm512_add_ps(c0, _mm512_set1_ps(bp[0]));
    c1 = _mm512_add_ps(c1, _mm512_set1_ps(bp[1]));
    c2 = _mm512_add_ps(c2, _mm512_set1_ps(bp[2]));
    c3 = _mm512_add_ps(c3, _mm512_set1_ps(bp[3]));
    c4 = _mm512_add_ps(c4, _mm512_set1_ps(bp[4]));
    c5 = _mm512_add_ps(c5, _mm512_set1_ps(bp[5]));
    c6 = _mm512_add_ps(c6, _mm512_set1_ps(bp[6]));
    c7 = _mm512_add_ps(c7, _mm512_set1_ps(bp[7]));
    __m512 vmin = _mm512_set1_ps(fmin);
    __m512 vmax = _mm512_set1_ps(fmax);
    c0 = _mm512_max_ps(c0, vmin); c0 = _mm512_min_ps(c0, vmax);
    c1 = _mm512_max_ps(c1, vmin); c1 = _mm512_min_ps(c1, vmax);
    c2 = _mm512_max_ps(c2, vmin); c2 = _mm512_min_ps(c2, vmax);
    c3 = _mm512_max_ps(c3, vmin); c3 = _mm512_min_ps(c3, vmax);
    c4 = _mm512_max_ps(c4, vmin); c4 = _mm512_min_ps(c4, vmax);
    c5 = _mm512_max_ps(c5, vmin); c5 = _mm512_min_ps(c5, vmax);
    c6 = _mm512_max_ps(c6, vmin); c6 = _mm512_min_ps(c6, vmax);
    c7 = _mm512_max_ps(c7, vmin); c7 = _mm512_min_ps(c7, vmax);
}

// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW fusion=post_op
inline void fuse_nchw_6(__m512& c0, __m512& c1, __m512& c2, __m512& c3, __m512& c4, __m512& c5,
                        const float* bp, float fmin, float fmax) {
    c0 = _mm512_add_ps(c0, _mm512_set1_ps(bp[0]));
    c1 = _mm512_add_ps(c1, _mm512_set1_ps(bp[1]));
    c2 = _mm512_add_ps(c2, _mm512_set1_ps(bp[2]));
    c3 = _mm512_add_ps(c3, _mm512_set1_ps(bp[3]));
    c4 = _mm512_add_ps(c4, _mm512_set1_ps(bp[4]));
    c5 = _mm512_add_ps(c5, _mm512_set1_ps(bp[5]));
    __m512 vmin = _mm512_set1_ps(fmin);
    __m512 vmax = _mm512_set1_ps(fmax);
    c0 = _mm512_max_ps(c0, vmin); c0 = _mm512_min_ps(c0, vmax);
    c1 = _mm512_max_ps(c1, vmin); c1 = _mm512_min_ps(c1, vmax);
    c2 = _mm512_max_ps(c2, vmin); c2 = _mm512_min_ps(c2, vmax);
    c3 = _mm512_max_ps(c3, vmin); c3 = _mm512_min_ps(c3, vmax);
    c4 = _mm512_max_ps(c4, vmin); c4 = _mm512_min_ps(c4, vmax);
    c5 = _mm512_max_ps(c5, vmin); c5 = _mm512_min_ps(c5, vmax);
}
#endif // LOCUST

// NCHW 1-row: single-row bias broadcast + clamp
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW fusion=post_op
inline void fuse_nchw_1(__m512& acc, float bias, float fmin, float fmax) {
    acc = _mm512_add_ps(acc, _mm512_set1_ps(bias));
    acc = _mm512_max_ps(acc, _mm512_set1_ps(fmin));
    acc = _mm512_min_ps(acc, _mm512_set1_ps(fmax));
}

// NCHW multi-vec array: bias + clamp over array of accumulators (small-M path)
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW fusion=post_op
inline void fuse_nchw_arr(__m512* acc, int nvec, float bias, float fmin, float fmax) {
    __m512 vb = _mm512_set1_ps(bias);
    __m512 vmin = _mm512_set1_ps(fmin);
    __m512 vmax = _mm512_set1_ps(fmax);
    for (int j = 0; j < nvec; j++) {
        acc[j] = _mm512_add_ps(acc[j], vb);
        acc[j] = _mm512_max_ps(acc[j], vmin);
        acc[j] = _mm512_min_ps(acc[j], vmax);
    }
}

// Scalar bias + clamp
// @nnr-meta isa=scalar dtype=fp32 fusion=post_op
inline void fuse_scalar(float& s, float bias, float fmin, float fmax) {
    s += bias;
    s = std::max(s, fmin);
    s = std::min(s, fmax);
}

// AVX-512 row copy for B-packing: avoids CRT memcpy call + vzeroupper penalty.
// Copies n floats. Full-vector loop + masked tail for remainder.
// @nnr-meta isa=AVX512 dtype=fp32
inline void copy_row_avx512(float* __restrict dst, const float* __restrict src, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(dst + i, _mm512_loadu_ps(src + i));
    if (i < n) {
        __mmask16 mask = (__mmask16)((1u << (n - i)) - 1u);
        _mm512_mask_storeu_ps(dst + i, mask, _mm512_maskz_loadu_ps(mask, src + i));
    }
}

// Copy exactly JBLK (64) floats — no branch, 4 full zmm stores.
// @nnr-meta isa=AVX512 dtype=fp32
inline void copy_row_avx512_full(float* __restrict dst, const float* __restrict src) {
    _mm512_storeu_ps(dst,      _mm512_loadu_ps(src));
    _mm512_storeu_ps(dst + 16, _mm512_loadu_ps(src + 16));
    _mm512_storeu_ps(dst + 32, _mm512_loadu_ps(src + 32));
    _mm512_storeu_ps(dst + 48, _mm512_loadu_ps(src + 48));
}

// 2D B-panel copy: kc rows of jw floats from src (stride src_stride) into
// dst (stride JBLK). Branch on full vs partial row is hoisted outside the loop.
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=GEMM
NNR_NOINLINE inline void pack_b_panel_avx512(float* __restrict dst, const float* __restrict src,
                                int kc, int jw, int src_stride) {
    constexpr int JBLK = 64;
    if (jw == JBLK) {
        for (int k = 0; k < kc; ++k)
            copy_row_avx512_full(dst + (size_t)k * JBLK, src + (size_t)k * src_stride);
    } else {
        for (int k = 0; k < kc; ++k)
            copy_row_avx512(dst + (size_t)k * JBLK, src + (size_t)k * src_stride, jw);
    }
}

#if 0 // LOCUST
;def gen_fuse_nhwc(nr):
;    args = ", ".join(f"__m512& c{r}" for r in range(nr))
inline void fuse_nhwc_@nr@(@args@,
                        __m512 vb, float fmin, float fmax) {
;    for r in range(0, nr, 2):
;        if r + 1 < nr:
    c@r@ = _mm512_add_ps(c@r@, vb); c@r+1@ = _mm512_add_ps(c@r+1@, vb);
;        else:
    c@r@ = _mm512_add_ps(c@r@, vb);
;        pass
;    pass
    __m512 vmin = _mm512_set1_ps(fmin);
    __m512 vmax = _mm512_set1_ps(fmax);
;    for r in range(nr):
    c@r@ = _mm512_max_ps(c@r@, vmin); c@r@ = _mm512_min_ps(c@r@, vmax);
;        pass
}
;    pass
;
;gen_fuse_nhwc(8)

;gen_fuse_nhwc(6)
#else // LOCUST
// @nnr-meta isa=AVX512 dtype=fp32 layout=NHWC fusion=post_op
inline void fuse_nhwc_8(__m512& c0, __m512& c1, __m512& c2, __m512& c3, __m512& c4, __m512& c5, __m512& c6, __m512& c7,
                        __m512 vb, float fmin, float fmax) {
    c0 = _mm512_add_ps(c0, vb); c1 = _mm512_add_ps(c1, vb);
    c2 = _mm512_add_ps(c2, vb); c3 = _mm512_add_ps(c3, vb);
    c4 = _mm512_add_ps(c4, vb); c5 = _mm512_add_ps(c5, vb);
    c6 = _mm512_add_ps(c6, vb); c7 = _mm512_add_ps(c7, vb);
    __m512 vmin = _mm512_set1_ps(fmin);
    __m512 vmax = _mm512_set1_ps(fmax);
    c0 = _mm512_max_ps(c0, vmin); c0 = _mm512_min_ps(c0, vmax);
    c1 = _mm512_max_ps(c1, vmin); c1 = _mm512_min_ps(c1, vmax);
    c2 = _mm512_max_ps(c2, vmin); c2 = _mm512_min_ps(c2, vmax);
    c3 = _mm512_max_ps(c3, vmin); c3 = _mm512_min_ps(c3, vmax);
    c4 = _mm512_max_ps(c4, vmin); c4 = _mm512_min_ps(c4, vmax);
    c5 = _mm512_max_ps(c5, vmin); c5 = _mm512_min_ps(c5, vmax);
    c6 = _mm512_max_ps(c6, vmin); c6 = _mm512_min_ps(c6, vmax);
    c7 = _mm512_max_ps(c7, vmin); c7 = _mm512_min_ps(c7, vmax);
}

// @nnr-meta isa=AVX512 dtype=fp32 layout=NHWC fusion=post_op
inline void fuse_nhwc_6(__m512& c0, __m512& c1, __m512& c2, __m512& c3, __m512& c4, __m512& c5,
                        __m512 vb, float fmin, float fmax) {
    c0 = _mm512_add_ps(c0, vb); c1 = _mm512_add_ps(c1, vb);
    c2 = _mm512_add_ps(c2, vb); c3 = _mm512_add_ps(c3, vb);
    c4 = _mm512_add_ps(c4, vb); c5 = _mm512_add_ps(c5, vb);
    __m512 vmin = _mm512_set1_ps(fmin);
    __m512 vmax = _mm512_set1_ps(fmax);
    c0 = _mm512_max_ps(c0, vmin); c0 = _mm512_min_ps(c0, vmax);
    c1 = _mm512_max_ps(c1, vmin); c1 = _mm512_min_ps(c1, vmax);
    c2 = _mm512_max_ps(c2, vmin); c2 = _mm512_min_ps(c2, vmax);
    c3 = _mm512_max_ps(c3, vmin); c3 = _mm512_min_ps(c3, vmax);
    c4 = _mm512_max_ps(c4, vmin); c4 = _mm512_min_ps(c4, vmax);
    c5 = _mm512_max_ps(c5, vmin); c5 = _mm512_min_ps(c5, vmax);
}
#endif // LOCUST

#if 0 // LOCUST-POD
;# -- Reusable Locust generators for NHWC FMA blocks -----------------------
;# These are called from multiple locations in the file.
;# All generators take indent level `d` (number of 4-space indents).
;# Use emit(d, text) which prepends d*4 spaces.
;
;def gen_pa_decls(d, nr, base="i", stride="o", offset="k0", src="A"):
;    for r in range(nr):
const float* pa@r@ = @src@ + (size_t)(@base@+@r@) * @stride@ + @offset@;
;        pass
;
;def gen_nhwc_8row_fma_k2x(d, pa_src="direct", k_var="k0"):
;    idx0 = f"{k_var}+k" if pa_src == "direct" else "k"
;    idx1 = f"{k_var}+k+1" if pa_src == "direct" else "k+1"
int k = 0;
for (; k + 2 <= kc; k += 2) {
    __m512 bv0 = _mm512_loadu_ps(pp + (size_t)k * JBLK);
    __m512 bv1 = _mm512_loadu_ps(pp + (size_t)(k+1) * JBLK);
;    for r in range(8):
    c@r@ = _mm512_fmadd_ps(_mm512_set1_ps(pa@r@[@idx0@]), bv0, c@r@);
;        pass
;    for r in range(8):
    c@r@ = _mm512_fmadd_ps(_mm512_set1_ps(pa@r@[@idx1@]), bv1, c@r@);
;        pass
}
if (k < kc) {
    __m512 bv = _mm512_loadu_ps(pp + (size_t)k * JBLK);
;    for r in range(8):
    c@r@ = _mm512_fmadd_ps(_mm512_set1_ps(pa@r@[@idx0@]), bv, c@r@);
;        pass
}
;    pass
;
;def gen_nhwc_nrow_c_zero_or_load(d, nr, masked=False, src="C"):
;    load = "_mm512_maskz_loadu_ps(mask, " if masked else "_mm512_loadu_ps("
;    cs = " = ".join(f"c{r}" for r in range(nr))
if (k0 == 0) {
    @cs@ = _mm512_setzero_ps();
} else {
;    for r in range(nr):
    c@r@ = @load@@src@ + (size_t)(i+@r@) * m + v);
;        pass
}
;    pass
;
;def gen_nhwc_nrow_store(d, nr, masked=False, src="C"):
;    for r in range(nr):
;        if masked:
_mm512_mask_storeu_ps(@src@ + (size_t)(i+@r@) * m + v, mask, c@r@);
;            pass
;        else:
_mm512_storeu_ps(@src@ + (size_t)(i+@r@) * m + v, c@r@);
;            pass
;def gen_nhwc_nrow_fma(d, nr):
for (int k = 0; k < kc; k++) {
    __m512 bv = _mm512_loadu_ps(pp + (size_t)k * JBLK);
;    for r in range(nr):
    c@r@ = _mm512_fmadd_ps(_mm512_set1_ps(pa@r@[k]), bv, c@r@);
;        pass
}
;    pass
;
;def gen_nhwc_nrow_vloop(d, nr, fma_fn, brace_wrap=False):
;    """Emit full + masked v-loop for nr rows."""
;    cs = ", ".join(f"c{r}" for r in range(nr))
;    for is_tail in (False, True):
;        if is_tail:
if (v < je) {
    __mmask16 mask = (__mmask16)((1u << (je - v)) - 1);
;            pass
;        else:
for (; v + 16 <= je; v += 16) {
;            pass
    const float* pp = pb + (v - j0);
    __m512 @cs@;
;        gen_nhwc_nrow_c_zero_or_load(d+1, nr, masked=is_tail)
;        if brace_wrap:
    {
;            fma_fn(d+2)
    }
;            pass
;        else:
;            fma_fn(d+1)
;        fb_load = "_mm512_maskz_loadu_ps(mask, fb + v)" if is_tail else "_mm512_loadu_ps(fb + v)"
    fuse_nhwc_@nr@(@cs@, @fb_load@, fmin, fmax);
;        gen_nhwc_nrow_store(d+1, nr, masked=is_tail)
}
;        pass
;
;def gen_nhwc_nrow_block(d, nr, offset="k0", src="A", fma_fn=None, brace_wrap=False):
;    """Emit for (; i + nr <= ie; i += nr) { pa_decls; v-loop } block."""
;    if fma_fn is None:
;        fma_fn = lambda dd: gen_nhwc_nrow_fma(dd, nr)
for (; i + @nr@ <= ie; i += @nr@) {
;    gen_pa_decls(d+1, nr, "i", "o", offset, src)
    int v = j0;
;    gen_nhwc_nrow_vloop(d+1, nr, fma_fn, brace_wrap)
}
;    pass
;
;def gen_pack_a_8(d):
for (int k = 0; k < kc; k++) {
;    for r in range(0, 8, 2):
    pa_pack[k * 8 + @r@] = pa[@r@][k]; pa_pack[k * 8 + @r+1@] = pa[@r+1@][k];
;        pass
}
;    pass
#endif // LOCUST-POD

// NHWC 1-row: add pre-loaded bias vector + clamp
// @nnr-meta isa=AVX512 dtype=fp32 layout=NHWC fusion=post_op
inline void fuse_nhwc_1(__m512& acc, __m512 vb, float fmin, float fmax) {
    acc = _mm512_add_ps(acc, vb);
    acc = _mm512_max_ps(acc, _mm512_set1_ps(fmin));
    acc = _mm512_min_ps(acc, _mm512_set1_ps(fmax));
}

// [COLD: 0 hits in packed_a, 32–80 hits in packed_b across 5 models]
// Row remainder for tiled GEMM variants: processes <8 remaining rows
// with a 1-row SIMD+scalar kernel. Extracted with NNR_NOINLINE to keep the hot
// 8-row ukernel loop compact in icache.
template <typename PostFn>
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=GEMM tiling=[MR,NR] fusion=post_op
static NNR_NOINLINE void dgemm_row_remainder(
    int i_start, int ie, int j0, int je, int k0, int kc,
    const float* __restrict row_a,  // per-row A data, first remainder row
    int a_stride,                   // stride between A rows (KC for packed_a, o for packed_b)
    const float* __restrict pb,     // packed B sub-panel
    float* __restrict C, int m,
    bool fuse_nchw, float fmin, float fmax,
    const PostFn& post_fn)
{
    constexpr bool can_fuse = PostFn::per_row_bias;
    constexpr int JBLK = 64;
    for (int rem = 0, i = i_start; i < ie; i++, rem++) {
        float* pci = C + (size_t)i * m;
        const float* pai = row_a + (size_t)rem * a_stride;
        int v = j0;
        for (; v + 16 <= je; v += 16) {
            const float* pp = pb + (v - j0);
            __m512 acc = (k0 == 0) ? _mm512_setzero_ps() : _mm512_loadu_ps(pci + v);
            for (int k = 0; k < kc; ++k)
                acc = _mm512_fmadd_ps(_mm512_set1_ps(pai[k]),
                    _mm512_loadu_ps(pp + (size_t)k * JBLK), acc);
            if constexpr (can_fuse) {
                const float* bp = (fuse_nchw && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                fuse_nchw_1(acc, bp[0], fmin, fmax);
            }
            _mm512_storeu_ps(pci + v, acc);
        }
        if (v < je) {
            __mmask16 mask = (__mmask16)((1u << (je - v)) - 1);
            const float* pp = pb + (v - j0);
            __m512 acc = (k0 == 0) ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(mask, pci + v);
            for (int k = 0; k < kc; ++k)
                acc = _mm512_fmadd_ps(_mm512_set1_ps(pai[k]),
                    _mm512_loadu_ps(pp + (size_t)k * JBLK), acc);
            if constexpr (can_fuse) {
                const float* bp = (fuse_nchw && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                fuse_nchw_1(acc, bp[0], fmin, fmax);
            }
            _mm512_mask_storeu_ps(pci + v, mask, acc);
        }
    }
}

// AVX-512 GEMM: C[n×m] = A[n×o] × B[o×m] with optional post-processing.
// Covers all float fast paths: GEMV M=1, GEMV N=1, small-K, small-M, tiled 6×16.
template <typename PostFn>
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=GEMM tiling=[K,MR,NR] fusion=post_op
inline void dgemm(int n, int m, int o, const float* __restrict A, const float* __restrict B, float* __restrict C, const PostFn& post_fn)
{
    // Register-fused post-op only for NCHW (per-row bias). NHWC is fused in dgemm_nhwc.
    constexpr bool can_fuse = PostFn::per_row_bias;
    // GEMV fast path: M=1 (row vector × matrix)
    // [COLD: 0 hits across 5 models — dgemm itself is rarely called; Conv uses packed variants]
    if (n == 1) {
        NNR_PROFILE_COUNT("dgemm:gemv_n1");
        int j;
        for (j = 0; j + 16 <= m; j += 16)
            _mm512_storeu_ps(C + j, _mm512_setzero_ps());
        for (; j < m; ++j)
            C[j] = 0.0f;
        for (int k = 0; k < o; ++k) {
            float a_k = A[k];
            const float* brow = B + (size_t)k * m;
            __m512 va = _mm512_set1_ps(a_k);
            j = 0;
            for (; j + 16 <= m; j += 16)
                _mm512_storeu_ps(C + j,
                    _mm512_fmadd_ps(va, _mm512_loadu_ps(brow + j),
                        _mm512_loadu_ps(C + j)));
            for (; j < m; ++j)
                C[j] += a_k * brow[j];
        }
        post_fn.apply(0, C, m);
        return;
    }

    // GEMV fast path: N=1 (matrix × column vector)
    // Uses 4 independent accumulators to hide FMA latency.
    // [COLD: 1 hit (MobileNetV2 final FC) — only model with M=1 GEMM via dgemm_generic]
    if (m == 1) {
        NNR_PROFILE_COUNT("dgemm:gemv_m1");
        nnr::for_static(0, n, nnr::compute_threads(n), [&](int i) {
            const float* pa_row = A + (size_t)i * o;
#if 0 // LOCUST
;            N = 4  # number of accumulators
;            for i in range(N):
            __m512 acc@i@ = _mm512_setzero_ps();
;                pass
            int k = 0;
            for (; k + @N*16@ <= o; k += @N*16@) {
;            for i in range(N):
;                off = f" + {i*16}" if i else ""
                acc@i@ = _mm512_fmadd_ps(_mm512_loadu_ps(pa_row + k@off@), _mm512_loadu_ps(B + k@off@), acc@i@);
;                pass
            }
;            # tree reduction
;            step = 1
;            while step < N:
;                for i in range(0, N, step * 2):
            acc@i@ = _mm512_add_ps(acc@i@, acc@i+step@);
;                    pass
;                step *= 2
#else // LOCUST
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();
            int k = 0;
            for (; k + 64 <= o; k += 64) {
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(pa_row + k), _mm512_loadu_ps(B + k), acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(pa_row + k + 16), _mm512_loadu_ps(B + k + 16), acc1);
                acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(pa_row + k + 32), _mm512_loadu_ps(B + k + 32), acc2);
                acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(pa_row + k + 48), _mm512_loadu_ps(B + k + 48), acc3);
            }
            acc0 = _mm512_add_ps(acc0, acc1);
            acc2 = _mm512_add_ps(acc2, acc3);
            acc0 = _mm512_add_ps(acc0, acc2);
#endif // LOCUST
            for (; k + 16 <= o; k += 16)
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(pa_row + k),
                    _mm512_loadu_ps(B + k), acc0);
            float s = _mm512_reduce_add_ps(acc0);
            for (; k < o; ++k)
                s += pa_row[k] * B[k];
            C[i] = s;
            post_fn.apply(i, C + i, 1);
        });
        return;
    }

    // Small-K path: A[N,K] is small (fits in L1), B[K,M] streams through L2.
    // 6-row register blocking gives 6 independent FMA chains (> 4-cycle latency).
    // No B-packing needed. Parallelize over M (column chunks).
    // [COLD: 0 hits — Conv uses dgemm_packed_a, not raw dgemm]
    if (o <= 48 && m >= 64) {
        NNR_PROFILE_COUNT("dgemm:small_k");
        int mchunks = (m + 15) / 16;
        nnr::for_static(0, mchunks, ((int64_t)n * m * o > (1 << 22)) ? nnr::compute_threads(mchunks) : 1, [&](int jc) {
            int j = jc * 16;
            __mmask16 mask = (j + 16 <= m) ? (__mmask16)0xFFFF
                : (__mmask16)((1u << (m - j)) - 1);
            float fmin = -FLT_MAX;
            float fmax = FLT_MAX;
            if constexpr (can_fuse) { if (post_fn.kind != post_op_kind::none) {
                fmin = post_fn.clip_min; fmax = post_fn.clip_max;
            }}
            int i = 0;
#if 0 // LOCUST
;            MR = 6
            for (; i + @MR@ <= n; i += @MR@) {
;            for r in range(MR):
                __m512 c@r@ = _mm512_setzero_ps();
;                pass
                const float* pa0 = A + (size_t)i * o;
;            for r in range(1, MR):
                const float* pa@r@ = pa@r-1@ + o;
;                pass
                for (int k = 0; k < o; k++) {
                    __m512 bv = _mm512_maskz_loadu_ps(mask, B + (size_t)k * m + j);
;            for r in range(MR):
                    c@r@ = _mm512_fmadd_ps(_mm512_set1_ps(pa@r@[k]), bv, c@r@);
;                pass
                }
                if constexpr (can_fuse) {
;            cs = ", ".join(f"c{r}" for r in range(MR))
                    const float* bp = (post_fn.kind != post_op_kind::none && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                    fuse_nchw_@MR@(@cs@, bp, fmin, fmax);
                }
;            for r in range(MR):
                _mm512_mask_storeu_ps(C + (size_t)(i+@r@) * m + j, mask, c@r@);
;                pass
                if (!(can_fuse && post_fn.kind != post_op_kind::none)) {
                    int plen = (j + 16 <= m) ? 16 : (m - j);
                    post_fn.apply_rows(i, i + @MR@, C, m, j, plen);
                }
#else // LOCUST
            for (; i + 6 <= n; i += 6) {
                __m512 c0 = _mm512_setzero_ps();
                __m512 c1 = _mm512_setzero_ps();
                __m512 c2 = _mm512_setzero_ps();
                __m512 c3 = _mm512_setzero_ps();
                __m512 c4 = _mm512_setzero_ps();
                __m512 c5 = _mm512_setzero_ps();
                const float* pa0 = A + (size_t)i * o;
                const float* pa1 = pa0 + o;
                const float* pa2 = pa1 + o;
                const float* pa3 = pa2 + o;
                const float* pa4 = pa3 + o;
                const float* pa5 = pa4 + o;
                for (int k = 0; k < o; k++) {
                    __m512 bv = _mm512_maskz_loadu_ps(mask, B + (size_t)k * m + j);
                    c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k]), bv, c0);
                    c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k]), bv, c1);
                    c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k]), bv, c2);
                    c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k]), bv, c3);
                    c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k]), bv, c4);
                    c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k]), bv, c5);
                }
                if constexpr (can_fuse) {
                    const float* bp = (post_fn.kind != post_op_kind::none && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                    fuse_nchw_6(c0, c1, c2, c3, c4, c5, bp, fmin, fmax);
                }
                _mm512_mask_storeu_ps(C + (size_t)(i+0) * m + j, mask, c0);
                _mm512_mask_storeu_ps(C + (size_t)(i+1) * m + j, mask, c1);
                _mm512_mask_storeu_ps(C + (size_t)(i+2) * m + j, mask, c2);
                _mm512_mask_storeu_ps(C + (size_t)(i+3) * m + j, mask, c3);
                _mm512_mask_storeu_ps(C + (size_t)(i+4) * m + j, mask, c4);
                _mm512_mask_storeu_ps(C + (size_t)(i+5) * m + j, mask, c5);
                if (!(can_fuse && post_fn.kind != post_op_kind::none)) {
                    int plen = (j + 16 <= m) ? 16 : (m - j);
                    post_fn.apply_rows(i, i + 6, C, m, j, plen);
                }
#endif // LOCUST
            }
            for (; i < n; i++) {
                __m512 acc = _mm512_setzero_ps();
                const float* pa = A + (size_t)i * o;
                for (int k = 0; k < o; k++)
                    acc = _mm512_fmadd_ps(_mm512_set1_ps(pa[k]),
                        _mm512_maskz_loadu_ps(mask, B + (size_t)k * m + j), acc);
                if constexpr (can_fuse) {
                    const float* bp = (post_fn.kind != post_op_kind::none && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                    fuse_nchw_1(acc, bp[0], fmin, fmax);
                }
                _mm512_mask_storeu_ps(C + (size_t)i * m + j, mask, acc);
                if (!(can_fuse && post_fn.kind != post_op_kind::none))
                    post_fn.apply(i, C + (size_t)i * m + j, (j + 16 <= m) ? 16 : (m - j));
            }
        });
        return;
    }

    // Small-M path: 6-row register blocking shares B loads across rows.
    // B is loaded once per K-step and broadcast to 6 FMA chains (6x fewer B reads).
    // Accumulators stay in ZMM registers across the entire K loop (no C traffic).
    // [COLD: 0 hits — Conv uses dgemm_packed_a, not raw dgemm]
    if (m < 64) {
        NNR_PROFILE_COUNT("dgemm:small_m");
        __mmask16 tail_mask = (m & 15) ? ((__mmask16)1 << (m & 15)) - 1 : (__mmask16)0xFFFF;
        int mfull = m / 16, mtail = m & 15;
        int ngroups = (n + 5) / 6;
        nnr::for_static(0, ngroups, ((int64_t)n * m * o > (1 << 18)) ? nnr::compute_threads(ngroups) : 1, [&](int ig) {
            int i0 = ig * 6;
            int ie = std::min(i0 + 6, n);
            int nr = ie - i0;
            float fmin = -FLT_MAX;
            float fmax = FLT_MAX;
            if constexpr (can_fuse) { if (post_fn.kind != post_op_kind::none) {
                fmin = post_fn.clip_min; fmax = post_fn.clip_max;
            }}
#if 0 // LOCUST
;            MR = 6
;            arrs = " ".join(f"__m512 c{r}[4]={{}};" for r in range(MR))
            if (nr == @MR@) {
;            for r in range(MR):
                const float* pa@r@ = A + (size_t)(i0+@r@) * o;
;                pass
                @arrs@
                for (int k = 0; k < o; k++) {
                    const float* br = B + (size_t)k * m;
;            for r in range(MR):
                    __m512 a@r@ = _mm512_set1_ps(pa@r@[k]);
;                pass
                    for (int j = 0; j < mfull; j++) {
                        __m512 bv = _mm512_loadu_ps(br + j * 16);
;            for r in range(MR):
                        c@r@[j] = _mm512_fmadd_ps(a@r@, bv, c@r@[j]);
;                pass
                    }
                    if (mtail) {
                        __m512 bv = _mm512_maskz_loadu_ps(tail_mask, br + mfull * 16);
;            for r in range(MR):
                        c@r@[mfull] = _mm512_fmadd_ps(a@r@, bv, c@r@[mfull]);
;                pass
                    }
                }
                if constexpr (can_fuse) {
                    const float* bp = (post_fn.kind != post_op_kind::none && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i0 : fused_zero_bias;
;            rows_list = ", ".join(f"c{r}" for r in range(MR))
                    __m512* rows[] = {@rows_list@};
                    int nvec = mtail ? mfull + 1 : mfull;
                    for (int r = 0; r < @MR@; r++)
                        fuse_nchw_arr(rows[r], nvec, bp[r], fmin, fmax);
                }
;            for r in range(0, MR, 2):
                float *pc@r@ = C + (size_t)(i0+@r@) * m, *pc@r+1@ = C + (size_t)(i0+@r+1@) * m;
;                pass
                for (int j = 0; j < mfull; j++) {
;            for r in range(0, MR, 2):
                    _mm512_storeu_ps(pc@r@ + j*16, c@r@[j]); _mm512_storeu_ps(pc@r+1@ + j*16, c@r+1@[j]);
;                pass
                }
                if (mtail) {
;            for r in range(MR):
                    _mm512_mask_storeu_ps(pc@r@ + mfull*16, tail_mask, c@r@[mfull]);
;                pass
                }
                if (!(can_fuse && post_fn.kind != post_op_kind::none))
                    post_fn.apply_rows(i0, i0 + @MR@, C, m, 0, m);
#else // LOCUST
            if (nr == 6) {
                const float* pa0 = A + (size_t)(i0+0) * o;
                const float* pa1 = A + (size_t)(i0+1) * o;
                const float* pa2 = A + (size_t)(i0+2) * o;
                const float* pa3 = A + (size_t)(i0+3) * o;
                const float* pa4 = A + (size_t)(i0+4) * o;
                const float* pa5 = A + (size_t)(i0+5) * o;
                __m512 c0[4]={}; __m512 c1[4]={}; __m512 c2[4]={}; __m512 c3[4]={}; __m512 c4[4]={}; __m512 c5[4]={};
                for (int k = 0; k < o; k++) {
                    const float* br = B + (size_t)k * m;
                    __m512 a0 = _mm512_set1_ps(pa0[k]);
                    __m512 a1 = _mm512_set1_ps(pa1[k]);
                    __m512 a2 = _mm512_set1_ps(pa2[k]);
                    __m512 a3 = _mm512_set1_ps(pa3[k]);
                    __m512 a4 = _mm512_set1_ps(pa4[k]);
                    __m512 a5 = _mm512_set1_ps(pa5[k]);
                    for (int j = 0; j < mfull; j++) {
                        __m512 bv = _mm512_loadu_ps(br + j * 16);
                        c0[j] = _mm512_fmadd_ps(a0, bv, c0[j]);
                        c1[j] = _mm512_fmadd_ps(a1, bv, c1[j]);
                        c2[j] = _mm512_fmadd_ps(a2, bv, c2[j]);
                        c3[j] = _mm512_fmadd_ps(a3, bv, c3[j]);
                        c4[j] = _mm512_fmadd_ps(a4, bv, c4[j]);
                        c5[j] = _mm512_fmadd_ps(a5, bv, c5[j]);
                    }
                    if (mtail) {
                        __m512 bv = _mm512_maskz_loadu_ps(tail_mask, br + mfull * 16);
                        c0[mfull] = _mm512_fmadd_ps(a0, bv, c0[mfull]);
                        c1[mfull] = _mm512_fmadd_ps(a1, bv, c1[mfull]);
                        c2[mfull] = _mm512_fmadd_ps(a2, bv, c2[mfull]);
                        c3[mfull] = _mm512_fmadd_ps(a3, bv, c3[mfull]);
                        c4[mfull] = _mm512_fmadd_ps(a4, bv, c4[mfull]);
                        c5[mfull] = _mm512_fmadd_ps(a5, bv, c5[mfull]);
                    }
                }
                if constexpr (can_fuse) {
                    const float* bp = (post_fn.kind != post_op_kind::none && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i0 : fused_zero_bias;
                    __m512* rows[] = {c0, c1, c2, c3, c4, c5};
                    int nvec = mtail ? mfull + 1 : mfull;
                    for (int r = 0; r < 6; r++)
                        fuse_nchw_arr(rows[r], nvec, bp[r], fmin, fmax);
                }
                float *pc0 = C + (size_t)(i0+0) * m, *pc1 = C + (size_t)(i0+1) * m;
                float *pc2 = C + (size_t)(i0+2) * m, *pc3 = C + (size_t)(i0+3) * m;
                float *pc4 = C + (size_t)(i0+4) * m, *pc5 = C + (size_t)(i0+5) * m;
                for (int j = 0; j < mfull; j++) {
                    _mm512_storeu_ps(pc0 + j*16, c0[j]); _mm512_storeu_ps(pc1 + j*16, c1[j]);
                    _mm512_storeu_ps(pc2 + j*16, c2[j]); _mm512_storeu_ps(pc3 + j*16, c3[j]);
                    _mm512_storeu_ps(pc4 + j*16, c4[j]); _mm512_storeu_ps(pc5 + j*16, c5[j]);
                }
                if (mtail) {
                    _mm512_mask_storeu_ps(pc0 + mfull*16, tail_mask, c0[mfull]);
                    _mm512_mask_storeu_ps(pc1 + mfull*16, tail_mask, c1[mfull]);
                    _mm512_mask_storeu_ps(pc2 + mfull*16, tail_mask, c2[mfull]);
                    _mm512_mask_storeu_ps(pc3 + mfull*16, tail_mask, c3[mfull]);
                    _mm512_mask_storeu_ps(pc4 + mfull*16, tail_mask, c4[mfull]);
                    _mm512_mask_storeu_ps(pc5 + mfull*16, tail_mask, c5[mfull]);
                }
                if (!(can_fuse && post_fn.kind != post_op_kind::none))
                    post_fn.apply_rows(i0, i0 + 6, C, m, 0, m);
#endif // LOCUST
            } else {
                // Remainder: 1-5 rows
                for (int i = i0; i < ie; i++) {
                    float* pc = C + (size_t)i * m;
                    const float* pa = A + (size_t)i * o;
                    __m512 acc[4] = {};
                    for (int k = 0; k < o; k++) {
                        const float* br = B + (size_t)k * m;
                        __m512 av = _mm512_set1_ps(pa[k]);
                        for (int j = 0; j < mfull; j++)
                            acc[j] = _mm512_fmadd_ps(av, _mm512_loadu_ps(br + j*16), acc[j]);
                        if (mtail)
                            acc[mfull] = _mm512_fmadd_ps(av,
                                _mm512_maskz_loadu_ps(tail_mask, br + mfull*16), acc[mfull]);
                    }
                    if constexpr (can_fuse) {
                        const float* bp = (post_fn.kind != post_op_kind::none && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                        int nvec = mtail ? mfull + 1 : mfull;
                        fuse_nchw_arr(acc, nvec, bp[0], fmin, fmax);
                    }
                    for (int j = 0; j < mfull; j++)
                        _mm512_storeu_ps(pc + j*16, acc[j]);
                    if (mtail)
                        _mm512_mask_storeu_ps(pc + mfull*16, tail_mask, acc[mfull]);
                    if (!(can_fuse && post_fn.kind != post_op_kind::none))
                        post_fn.apply(i, pc, m);
                }
            }
        });
        return;
    }

    // Tiled path: 8×16 micro-kernel with A-packing, B-packing, K-blocking
    // A-packing: 8 rows packed into contiguous [K×8] panel → sequential L1 reads
    // K-blocking: B panel KC×JBLK=64 KB (fits any modern L2, the smallest
    //             currently shipping is ~256 KB on Zen 1/2); A panel KC×8=8 KB
    //             (fits any modern L1d, ≥32 KB universally).
    // 8 accumulators × 0.5-cycle FMA throughput = 4 cycles/step = exactly Zen4
    // FMA latency (4 cycles) → no pipeline stalls, full 2 FMAs/cycle throughput.
    // [COLD: 0 hits — Conv uses dgemm_packed_a, not raw dgemm]
    NNR_PROFILE_COUNT("dgemm:tiled");
    constexpr int JBLK = 64;
    constexpr int KC = 256;
    int ni = (n + JBLK - 1) / JBLK;
    int nj = (m + JBLK - 1) / JBLK;
    int ntiles = ni * nj;
    int nt = ((int64_t)n * m * o > (1 << 21)) ? nnr::compute_threads(ntiles) : 1;

    // Scratch per thread: B panel [KC×JBLK] + A panel [KC×8]
    NNR_POOL_ENSURE_SCRATCH(((size_t)KC * JBLK + (size_t)KC * 8) * sizeof(float));

    nnr::for_dynamic(0, ntiles, nt, [&](int tid, int tile) {
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

            // Pack B sub-panel: pb[kc × JBLK]
            pack_b_panel_avx512(pb, B + (size_t)k0 * m + j0, kc, jw, m);

            float fmin = -FLT_MAX;
            float fmax = FLT_MAX;
            bool fuse_nchw = false;
            if constexpr (can_fuse) { if (last_k && post_fn.kind != post_op_kind::none) {
                fuse_nchw = true; fmin = post_fn.clip_min; fmax = post_fn.clip_max;
            }}

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
#if 0 // LOCUST
;                gen_pack_a_8(4)
#else // LOCUST
                for (int k = 0; k < kc; k++) {
                    pa_pack[k * 8 + 0] = pa[0][k]; pa_pack[k * 8 + 1] = pa[1][k];
                    pa_pack[k * 8 + 2] = pa[2][k]; pa_pack[k * 8 + 3] = pa[3][k];
                    pa_pack[k * 8 + 4] = pa[4][k]; pa_pack[k * 8 + 5] = pa[5][k];
                    pa_pack[k * 8 + 6] = pa[6][k]; pa_pack[k * 8 + 7] = pa[7][k];
                }
#endif // LOCUST

                int v = j0;
                for (; v + 16 <= je; v += 16) {
                    const float* pp = pb + (v - j0);
                    const float* bp_uk = fused_zero_bias;
                    if constexpr (can_fuse) { if (fuse_nchw && post_fn.bias) bp_uk = post_fn.bias + post_fn.bias_off + i; }
                    ukernel_nchw_jit(kc, pa_pack, pp, JBLK, pc, v,
                        k0 == 0, fuse_nchw, bp_uk, fmin, fmax);
                }
                // [COLD: 0 hits — tiled path itself is never reached in Conv workloads]
                // Scalar remainder columns (at most 15 elements)
                if (v < je) { NNR_PROFILE_COUNT("tiled:scalar_cols");
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
                }}
            }
            // [COLD: 0 hits — tiled path itself is never reached in Conv workloads]
            // Remainder rows (< 8): extracted to NNR_NOINLINE
            if (i < ie) {
                NNR_PROFILE_COUNT("tiled:row_remainder");
                dgemm_row_remainder(i, ie, j0, je, k0, kc,
                    A + (size_t)i * o + k0, o, pb, C, m,
                    fuse_nchw, fmin, fmax, post_fn);
            }
        }
        // Post-process after all K-blocks complete (bias + activation on L1-hot tile)
        // Skip if already fused into micro-kernel stores above
        if (!(can_fuse && post_fn.kind != post_op_kind::none))
            post_fn.apply_rows(i0, ie, C, m, j0, jw);
    });
}

// Pre-pack B matrix into panel format for dgemm_packed_b().
// Packed layout: [nj × nk × KC × JBLK] where nj=ceil(m/JBLK), nk=ceil(o/KC).
// Eliminates per-tile B-copy from the GEMM hot loop — critical for NHWC Conv
// where B is the (constant) weight matrix reused across all spatial tiles.
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=GEMM
inline size_t pack_b_size(int o, int m) {
    constexpr int JBLK = 64, KC = 256;
    int nj = (m + JBLK - 1) / JBLK;
    int nk = (o + KC - 1) / KC;
    return (size_t)nj * nk * KC * JBLK;
}

// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=GEMM
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
            pack_b_panel_avx512(panel, B + (size_t)k0 * m + j0, kc, jw, m);
        }
    }
}

// GEMM with pre-packed B: C[n×m] = A[n×o] × packed_B
// packed_B must be created by pack_b(). Same micro-kernel as dgemm(),
// but skips B-packing per tile — only A-packing remains in the hot loop.
// Scratch per thread drops from KC*JBLK+KC*6 (70KB) to KC*6 (6KB).
template <typename PostFn>
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=GEMM tiling=[K,MR,NR] fusion=post_op
inline void dgemm_packed_b(int n, int m, int o, const float* __restrict A,
    const float* __restrict packed_B, float* __restrict C, const PostFn& post_fn)
{
    constexpr bool can_fuse = PostFn::per_row_bias;
    constexpr int JBLK = 64;
    constexpr int KC = 256;
    int nj = (m + JBLK - 1) / JBLK;
    int nk = (o + KC - 1) / KC;
    int ni = (n + JBLK - 1) / JBLK;

    // Thin-M: M≤64 (ni==1) is memory-bandwidth-bound. Threading provides
    // zero benefit — workers just compete for DRAM BW. Run single-threaded
    // to avoid thread pool wake/sync overhead.
    int ntiles = ni * nj;
    int nt = (ni > 1 && (int64_t)n * m * o > (1 << 21)) ? nnr::compute_threads(ntiles) : 1;

    NNR_POOL_ENSURE_SCRATCH((size_t)KC * 8 * sizeof(float));

    nnr::for_dynamic(0, ntiles, nt, [&](int tid, int tile) {
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
#if 0 // LOCUST
;                gen_pack_a_8(4)
#else // LOCUST
                for (int k = 0; k < kc; k++) {
                    pa_pack[k * 8 + 0] = pa[0][k]; pa_pack[k * 8 + 1] = pa[1][k];
                    pa_pack[k * 8 + 2] = pa[2][k]; pa_pack[k * 8 + 3] = pa[3][k];
                    pa_pack[k * 8 + 4] = pa[4][k]; pa_pack[k * 8 + 5] = pa[5][k];
                    pa_pack[k * 8 + 6] = pa[6][k]; pa_pack[k * 8 + 7] = pa[7][k];
                }
#endif // LOCUST
                int v = j0;
                const float* bp_uk = fused_zero_bias;
                if constexpr (can_fuse) { if (fuse_nchw && post_fn.bias) bp_uk = post_fn.bias + post_fn.bias_off + i; }
                for (; v + 32 <= je; v += 32) {
                    const float* ppL = pb + (v - j0);
                    const float* ppR = pb + (v - j0) + 16;
                    ukernel_nchw_2x(kc, pa_pack, ppL, ppR, JBLK, pc, v, v + 16,
                        k0 == 0, fuse_nchw, bp_uk, fmin, fmax);
                }
                for (; v + 16 <= je; v += 16) {
                    const float* pp = pb + (v - j0);
                    ukernel_nchw_jit(kc, pa_pack, pp, JBLK, pc, v,
                        k0 == 0, fuse_nchw, bp_uk, fmin, fmax);
                }
                // [COLD: 0 hits — packed_b output dims always align to 16]
                if (v < je) { NNR_PROFILE_COUNT("packed_b:scalar_cols");
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
                }}
            }
            // [WARM: 32–80 hits — spatial dims not always divisible by 8]
            // Remainder rows (< 8): extracted to NNR_NOINLINE to keep hot loop compact
            if (i < ie) {
                NNR_PROFILE_COUNT("packed_b:row_remainder");
                dgemm_row_remainder(i, ie, j0, je, k0, kc,
                    A + (size_t)i * o + k0, o, pb, C, m,
                    fuse_nchw, fmin, fmax, post_fn);
            }
        }
        if (!(can_fuse && post_fn.kind != post_op_kind::none))
            post_fn.apply_rows(i0, ie, C, m, j0, jw);
    });
}

// Pre-pack A matrix into panel format for dgemm_packed_a().
// Packed layout: [ni × nk × JBLK × KC] where ni=ceil(n/JBLK), nk=ceil(o/KC).
// Within each (ib, kb) panel:
//   - Full 8-row groups: interleaved [kc × 8] (micro-kernel format), stride 8*KC
//   - Remainder rows (< 8): contiguous [kc] per row, stride KC
// Eliminates per-tile A-packing from the GEMM hot loop — critical for NCHW Conv
// where A is the (constant) weight matrix reused across all J-tiles.
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=GEMM
inline size_t pack_a_size(int n, int o) {
    constexpr int JBLK = 64, KC = 256;
    int ni = (n + JBLK - 1) / JBLK;
    int nk = (o + KC - 1) / KC;
    return (size_t)ni * nk * JBLK * KC;
}

// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=GEMM
inline void pack_a(float* __restrict dst, const float* __restrict A, int n, int o) {
    constexpr int JBLK = 64, KC = 256;
    int ni = (n + JBLK - 1) / JBLK;
    int nk = (o + KC - 1) / KC;
    memset(dst, 0, (size_t)ni * nk * JBLK * KC * sizeof(float));
    for (int ib = 0; ib < ni; ib++) {
        int i0 = ib * JBLK;
        int iw = std::min(JBLK, n - i0);
        int nfull = iw / 8;
        int nrem = iw % 8;
        for (int kb = 0; kb < nk; kb++) {
            int k0 = kb * KC;
            int kc = std::min(KC, o - k0);
            float* panel = dst + ((size_t)ib * nk + kb) * JBLK * KC;
            // Pack 8-row interleaved groups
            for (int g = 0; g < nfull; g++) {
                float* grp = panel + (size_t)g * 8 * KC;
                const float* pa[8];
                for (int r = 0; r < 8; r++)
                    pa[r] = A + (size_t)(i0 + g * 8 + r) * o + k0;
#if 0 // LOCUST
                for (int k = 0; k < kc; k++) {
;                for r in range(0, 8, 2):
                    grp[k * 8 + @r@] = pa[@r@][k]; grp[k * 8 + @r+1@] = pa[@r+1@][k];
;                    pass
                }
#else // LOCUST
                for (int k = 0; k < kc; k++) {
                    grp[k * 8 + 0] = pa[0][k]; grp[k * 8 + 1] = pa[1][k];
                    grp[k * 8 + 2] = pa[2][k]; grp[k * 8 + 3] = pa[3][k];
                    grp[k * 8 + 4] = pa[4][k]; grp[k * 8 + 5] = pa[5][k];
                    grp[k * 8 + 6] = pa[6][k]; grp[k * 8 + 7] = pa[7][k];
                }
#endif // LOCUST
            }
            // Pack remainder rows (contiguous per row)
            for (int r = 0; r < nrem; r++) {
                float* rem = panel + (size_t)nfull * 8 * KC + (size_t)r * KC;
                copy_row_avx512(rem, A + (size_t)(i0 + nfull * 8 + r) * o + k0, kc);
            }
        }
    }
}

// GEMM with pre-packed A: C[n×m] = packed_A × B[o×m]
// packed_A must be created by pack_a(). Same micro-kernel as dgemm(),
// but skips A-packing per tile — only B-packing remains in the hot loop.
// Scratch per thread drops from KC*JBLK+KC*8 (72KB) to KC*JBLK (64KB).
template <typename PostFn>
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=GEMM tiling=[K,MR,NR] fusion=post_op
inline void dgemm_packed_a(int n, int m, int o, const float* __restrict packed_A,
    const float* __restrict B, float* __restrict C, const PostFn& post_fn)
{
    constexpr bool can_fuse = PostFn::per_row_bias;
    constexpr int JBLK = 64;
    constexpr int KC = 256;
    int nk = (o + KC - 1) / KC;
    int ni = (n + JBLK - 1) / JBLK;
    int nj = (m + JBLK - 1) / JBLK;
    int ntiles = ni * nj;
    int nt = ((int64_t)n * m * o > (1 << 21)) ? nnr::compute_threads(ntiles) : 1;

    // Single J-tile optimization: when nj==1, all I-tiles would redundantly
    // pack the same B sub-panel. Pre-pack once or skip entirely.
    //   m == JBLK: B layout matches packed layout (stride == JBLK), zero-copy.
    //   m < JBLK:  Pre-pack all K-blocks once into a shared read-only buffer.
    float* shared_pb = nullptr;
    const bool skip_pack = (nj == 1 && m == JBLK);
    const bool prepack_b = (nj == 1 && m < JBLK);
    if (prepack_b) {
        size_t total = (size_t)nk * KC * JBLK;
        shared_pb = (float*)nnr_aligned_alloc(total * sizeof(float), 64);
        if (shared_pb) {
            int jw = m;  // nj==1 means je-j0 == m
            for (int kt = 0; kt < nk; kt++) {
                int k0 = kt * KC;
                int kc = std::min(KC, o - k0);
                float* dst = shared_pb + (size_t)kt * KC * JBLK;
                memset(dst, 0, (size_t)KC * JBLK * sizeof(float));
                pack_b_panel_avx512(dst, B + (size_t)k0 * m, kc, jw, m);
            }
        }
    }

    if (!skip_pack && !prepack_b)
        NNR_POOL_ENSURE_SCRATCH((size_t)KC * JBLK * sizeof(float));

    nnr::for_dynamic(0, ntiles, nt, [&](int tid, int tile) {
        float* pb = (!skip_pack && !prepack_b) ? (float*)NNR_POOL_SCRATCH(tid) : nullptr;
        int ib = tile / nj;
        int i0 = ib * JBLK;
        int j0 = (tile % nj) * JBLK;
        int ie = std::min(i0 + JBLK, n);
        int je = std::min(j0 + JBLK, m);
        int jw = je - j0;
        int iw = ie - i0;
        int nfull = iw / 8;

        for (int kt = 0; kt < nk; kt++) {
            int k0 = kt * KC;
            int kc = std::min(KC, o - k0);
            bool last_k = (k0 + kc == o);

            // Resolve B panel pointer:
            //   skip_pack: B is already [kc × JBLK] contiguous — use directly
            //   prepack_b: read from shared pre-packed buffer
            //   default:   pack per-tile as before
            const float* b_panel;
            if (skip_pack) {
                b_panel = B + (size_t)k0 * m + j0;
            } else if (prepack_b && shared_pb) {
                b_panel = shared_pb + (size_t)kt * KC * JBLK;
            } else {
                NNR_PROFILE_SCOPE("packed_a:pack_b");
                pack_b_panel_avx512(pb, B + (size_t)k0 * m + j0, kc, jw, m);
                b_panel = pb;
            }

            const float* a_panel = packed_A + ((size_t)ib * nk + kt) * JBLK * KC;

            float fmin = -FLT_MAX;
            float fmax = FLT_MAX;
            bool fuse_nchw = false;
            if constexpr (can_fuse) { if (last_k && post_fn.kind != post_op_kind::none) {
                fuse_nchw = true; fmin = post_fn.clip_min; fmax = post_fn.clip_max;
            }}

            // 8-row groups using pre-packed interleaved A
            int i = i0;
            int grp = 0;
            for (; i + 8 <= ie; i += 8, grp++) {
                float* pc[8];
                for (int r = 0; r < 8; r++)
                    pc[r] = C + (size_t)(i + r) * m;

                const float* pa_pack = a_panel + (size_t)grp * 8 * KC;

                int v = j0;
                // Double-wide: process 32 columns per call (2 NR blocks).
                // Amortizes broadcast cost: 8 broadcasts feed 16 FMAs instead of 8.
                const float* bp_uk = fused_zero_bias;
                if constexpr (can_fuse) { if (fuse_nchw && post_fn.bias) bp_uk = post_fn.bias + post_fn.bias_off + i; }
                for (; v + 32 <= je; v += 32) {
                    const float* ppL = b_panel + (v - j0);
                    const float* ppR = b_panel + (v - j0) + 16;
                    ukernel_nchw_2x(kc, pa_pack, ppL, ppR, JBLK, pc, v, v + 16,
                        k0 == 0, fuse_nchw, bp_uk, fmin, fmax);
                }
                // Remainder: single 16-wide kernel
                for (; v + 16 <= je; v += 16) {
                    const float* pp = b_panel + (v - j0);
                    ukernel_nchw_jit(kc, pa_pack, pp, JBLK, pc, v,
                        k0 == 0, fuse_nchw, bp_uk, fmin, fmax);
                }
                // Masked vector remainder columns (at most 15 elements).
                // AVX-512 masking replaces scalar — same 8-accumulator
                // structure as ukernel_nchw, K-2x unrolled.
                if (v < je) { NNR_PROFILE_COUNT("packed_a:masked_cols");
                    __mmask16 mask = (__mmask16)((1u << (je - v)) - 1);
                    const float* pp = b_panel + (v - j0);
                    __m512 c0, c1, c2, c3, c4, c5, c6, c7;
                    if (k0 == 0) {
                        c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = _mm512_setzero_ps();
                    } else {
                        c0 = _mm512_maskz_loadu_ps(mask, pc[0] + v);
                        c1 = _mm512_maskz_loadu_ps(mask, pc[1] + v);
                        c2 = _mm512_maskz_loadu_ps(mask, pc[2] + v);
                        c3 = _mm512_maskz_loadu_ps(mask, pc[3] + v);
                        c4 = _mm512_maskz_loadu_ps(mask, pc[4] + v);
                        c5 = _mm512_maskz_loadu_ps(mask, pc[5] + v);
                        c6 = _mm512_maskz_loadu_ps(mask, pc[6] + v);
                        c7 = _mm512_maskz_loadu_ps(mask, pc[7] + v);
                    }
                    int k = 0;
                    for (; k + 2 <= kc; k += 2) {
                        __m512 bv0 = _mm512_loadu_ps(pp + (size_t)k * JBLK);
                        __m512 bv1 = _mm512_loadu_ps(pp + (size_t)(k+1) * JBLK);
                        const float* ap0 = pa_pack + k * 8;
                        const float* ap1 = ap0 + 8;
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
                    }
                    if (k < kc) {
                        __m512 bv = _mm512_loadu_ps(pp + (size_t)k * JBLK);
                        const float* ap = pa_pack + k * 8;
                        c0 = _mm512_fmadd_ps(_mm512_set1_ps(ap[0]), bv, c0);
                        c1 = _mm512_fmadd_ps(_mm512_set1_ps(ap[1]), bv, c1);
                        c2 = _mm512_fmadd_ps(_mm512_set1_ps(ap[2]), bv, c2);
                        c3 = _mm512_fmadd_ps(_mm512_set1_ps(ap[3]), bv, c3);
                        c4 = _mm512_fmadd_ps(_mm512_set1_ps(ap[4]), bv, c4);
                        c5 = _mm512_fmadd_ps(_mm512_set1_ps(ap[5]), bv, c5);
                        c6 = _mm512_fmadd_ps(_mm512_set1_ps(ap[6]), bv, c6);
                        c7 = _mm512_fmadd_ps(_mm512_set1_ps(ap[7]), bv, c7);
                    }
                    if constexpr (can_fuse) { if (fuse_nchw) {
                        const float* bp = post_fn.bias ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
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
                        c0 = _mm512_max_ps(c0, vmin); c0 = _mm512_min_ps(c0, vmax);
                        c1 = _mm512_max_ps(c1, vmin); c1 = _mm512_min_ps(c1, vmax);
                        c2 = _mm512_max_ps(c2, vmin); c2 = _mm512_min_ps(c2, vmax);
                        c3 = _mm512_max_ps(c3, vmin); c3 = _mm512_min_ps(c3, vmax);
                        c4 = _mm512_max_ps(c4, vmin); c4 = _mm512_min_ps(c4, vmax);
                        c5 = _mm512_max_ps(c5, vmin); c5 = _mm512_min_ps(c5, vmax);
                        c6 = _mm512_max_ps(c6, vmin); c6 = _mm512_min_ps(c6, vmax);
                        c7 = _mm512_max_ps(c7, vmin); c7 = _mm512_min_ps(c7, vmax);
                    }}
                    _mm512_mask_storeu_ps(pc[0] + v, mask, c0);
                    _mm512_mask_storeu_ps(pc[1] + v, mask, c1);
                    _mm512_mask_storeu_ps(pc[2] + v, mask, c2);
                    _mm512_mask_storeu_ps(pc[3] + v, mask, c3);
                    _mm512_mask_storeu_ps(pc[4] + v, mask, c4);
                    _mm512_mask_storeu_ps(pc[5] + v, mask, c5);
                    _mm512_mask_storeu_ps(pc[6] + v, mask, c6);
                    _mm512_mask_storeu_ps(pc[7] + v, mask, c7);
                }
            }
            // [COLD: 0 hits — weight dims are always multiples of 8 in typical models]
            // Remainder rows (< 8): extracted to NNR_NOINLINE to keep hot loop compact
            if (i < ie) {
                NNR_PROFILE_COUNT("packed_a:row_remainder");
                const float* row_a = a_panel + (size_t)nfull * 8 * KC;
                dgemm_row_remainder(i, ie, j0, je, k0, kc,
                    row_a, KC, b_panel, C, m, fuse_nchw, fmin, fmax, post_fn);
                i = ie;
            }
        }
        if (!(can_fuse && post_fn.kind != post_op_kind::none))
            post_fn.apply_rows(i0, ie, C, m, j0, jw);
    });

    if (shared_pb) nnr_aligned_free(shared_pb);
}

// Batched GEMM for Winograd: performs 36 independent GEMMs sharing dispatch/threading.
// C_batch[p][n×m] = packed_A_batch[p] × B_batch[p][o×m]  for p in [0..36)
// All 36 GEMMs have the same (n, m, o) dimensions.
// Threading: distributes ntiles×36 work items across threads for better
// utilization when individual GEMMs are small (e.g., 64×196 → 4 tiles).
template <typename PostFn>
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=[GEMM,Winograd] tiling=[K,MR,NR] fusion=post_op
inline void dgemm_packed_a_batch36(
    int n, int m, int o,
    const float* const packed_A_batch[36],
    const float* const B_batch[36],
    float* const C_batch[36],
    const PostFn& post_fn)
{
    constexpr bool can_fuse = PostFn::per_row_bias;
    constexpr int JBLK = 64;
    constexpr int KC = 256;
    int nk = (o + KC - 1) / KC;
    int ni = (n + JBLK - 1) / JBLK;
    int nj = (m + JBLK - 1) / JBLK;
    int ntiles = ni * nj;
    int total_work = ntiles * 36;
    int nt = ((int64_t)n * m * o * 36 > (1 << 21)) ? nnr::compute_threads(total_work) : 1;

    NNR_POOL_ENSURE_SCRATCH((size_t)KC * JBLK * sizeof(float));

    nnr::for_dynamic(0, total_work, nt, [&](int tid, int work_idx) {
        int tile = work_idx / 36;
        int p = work_idx % 36;

        float* pb = (float*)NNR_POOL_SCRATCH(tid);
        int ib = tile / nj;
        int i0 = ib * JBLK;
        int j0 = (tile % nj) * JBLK;
        int ie = std::min(i0 + JBLK, n);
        int je = std::min(j0 + JBLK, m);
        int jw = je - j0;
        int iw = ie - i0;
        int nfull = iw / 8;

        {
            const float* packed_A = packed_A_batch[p];
            const float* B = B_batch[p];
            float* C = C_batch[p];

            for (int kt = 0; kt < nk; kt++) {
                int k0 = kt * KC;
                int kc = std::min(KC, o - k0);
                bool last_k = (k0 + kc == o);

                pack_b_panel_avx512(pb, B + (size_t)k0 * m + j0, kc, jw, m);

                const float* a_panel = packed_A + ((size_t)ib * nk + kt) * JBLK * KC;

                float fmin = -FLT_MAX;
                float fmax = FLT_MAX;
                bool fuse_nchw = false;
                if constexpr (can_fuse) { if (last_k && post_fn.kind != post_op_kind::none) {
                    fuse_nchw = true; fmin = post_fn.clip_min; fmax = post_fn.clip_max;
                }}

                int i = i0;
                int grp = 0;
                for (; i + 8 <= ie; i += 8, grp++) {
                    float* pc[8];
                    for (int r = 0; r < 8; r++)
                        pc[r] = C + (size_t)(i + r) * m;

                    const float* pa_pack = a_panel + (size_t)grp * 8 * KC;

                    int v = j0;
                    const float* bp_uk = fused_zero_bias;
                    if constexpr (can_fuse) { if (fuse_nchw && post_fn.bias) bp_uk = post_fn.bias + post_fn.bias_off + i; }
                    for (; v + 32 <= je; v += 32) {
                        const float* ppL = pb + (v - j0);
                        const float* ppR = pb + (v - j0) + 16;
                        ukernel_nchw_2x(kc, pa_pack, ppL, ppR, JBLK, pc, v, v + 16,
                            k0 == 0, fuse_nchw, bp_uk, fmin, fmax);
                    }
                    for (; v + 16 <= je; v += 16) {
                        const float* pp = pb + (v - j0);
                        ukernel_nchw_jit(kc, pa_pack, pp, JBLK, pc, v,
                            k0 == 0, fuse_nchw, bp_uk, fmin, fmax);
                    }
                    if (v < je) {
                        __mmask16 mask = (__mmask16)((1u << (je - v)) - 1);
                        const float* pp = pb + (v - j0);
                        __m512 c0, c1, c2, c3, c4, c5, c6, c7;
                        if (k0 == 0) {
                            c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = _mm512_setzero_ps();
                        } else {
                            c0 = _mm512_maskz_loadu_ps(mask, pc[0] + v);
                            c1 = _mm512_maskz_loadu_ps(mask, pc[1] + v);
                            c2 = _mm512_maskz_loadu_ps(mask, pc[2] + v);
                            c3 = _mm512_maskz_loadu_ps(mask, pc[3] + v);
                            c4 = _mm512_maskz_loadu_ps(mask, pc[4] + v);
                            c5 = _mm512_maskz_loadu_ps(mask, pc[5] + v);
                            c6 = _mm512_maskz_loadu_ps(mask, pc[6] + v);
                            c7 = _mm512_maskz_loadu_ps(mask, pc[7] + v);
                        }
                        int k = 0;
                        for (; k + 2 <= kc; k += 2) {
                            __m512 bv0 = _mm512_loadu_ps(pp + (size_t)k * JBLK);
                            __m512 bv1 = _mm512_loadu_ps(pp + (size_t)(k+1) * JBLK);
                            const float* ap0 = pa_pack + k * 8;
                            const float* ap1 = ap0 + 8;
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
                        }
                        if (k < kc) {
                            __m512 bv = _mm512_loadu_ps(pp + (size_t)k * JBLK);
                            const float* ap = pa_pack + k * 8;
                            c0 = _mm512_fmadd_ps(_mm512_set1_ps(ap[0]), bv, c0);
                            c1 = _mm512_fmadd_ps(_mm512_set1_ps(ap[1]), bv, c1);
                            c2 = _mm512_fmadd_ps(_mm512_set1_ps(ap[2]), bv, c2);
                            c3 = _mm512_fmadd_ps(_mm512_set1_ps(ap[3]), bv, c3);
                            c4 = _mm512_fmadd_ps(_mm512_set1_ps(ap[4]), bv, c4);
                            c5 = _mm512_fmadd_ps(_mm512_set1_ps(ap[5]), bv, c5);
                            c6 = _mm512_fmadd_ps(_mm512_set1_ps(ap[6]), bv, c6);
                            c7 = _mm512_fmadd_ps(_mm512_set1_ps(ap[7]), bv, c7);
                        }
                        if constexpr (can_fuse) { if (fuse_nchw) {
                            const float* bp = post_fn.bias ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
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
                            c0 = _mm512_max_ps(c0, vmin); c0 = _mm512_min_ps(c0, vmax);
                            c1 = _mm512_max_ps(c1, vmin); c1 = _mm512_min_ps(c1, vmax);
                            c2 = _mm512_max_ps(c2, vmin); c2 = _mm512_min_ps(c2, vmax);
                            c3 = _mm512_max_ps(c3, vmin); c3 = _mm512_min_ps(c3, vmax);
                            c4 = _mm512_max_ps(c4, vmin); c4 = _mm512_min_ps(c4, vmax);
                            c5 = _mm512_max_ps(c5, vmin); c5 = _mm512_min_ps(c5, vmax);
                            c6 = _mm512_max_ps(c6, vmin); c6 = _mm512_min_ps(c6, vmax);
                            c7 = _mm512_max_ps(c7, vmin); c7 = _mm512_min_ps(c7, vmax);
                        }}
                        _mm512_mask_storeu_ps(pc[0] + v, mask, c0);
                        _mm512_mask_storeu_ps(pc[1] + v, mask, c1);
                        _mm512_mask_storeu_ps(pc[2] + v, mask, c2);
                        _mm512_mask_storeu_ps(pc[3] + v, mask, c3);
                        _mm512_mask_storeu_ps(pc[4] + v, mask, c4);
                        _mm512_mask_storeu_ps(pc[5] + v, mask, c5);
                        _mm512_mask_storeu_ps(pc[6] + v, mask, c6);
                        _mm512_mask_storeu_ps(pc[7] + v, mask, c7);
                    }
                }
                for (int rem = 0; i < ie; i++, rem++) {
                    float* pci = C + (size_t)i * m;
                    const float* pai = a_panel + (size_t)nfull * 8 * KC + (size_t)rem * KC;
                    int v = j0;
                    for (; v + 16 <= je; v += 16) {
                        const float* pp = pb + (v - j0);
                        __m512 acc = (k0 == 0) ? _mm512_setzero_ps() : _mm512_loadu_ps(pci + v);
                        for (int k = 0; k < kc; ++k)
                            acc = _mm512_fmadd_ps(_mm512_set1_ps(pai[k]),
                                _mm512_loadu_ps(pp + (size_t)k * JBLK), acc);
                        if constexpr (can_fuse) {
                            const float* bp = (fuse_nchw && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                            fuse_nchw_1(acc, bp[0], fmin, fmax);
                        }
                        _mm512_storeu_ps(pci + v, acc);
                    }
                    if (v < je) {
                        __mmask16 mask = (__mmask16)((1u << (je - v)) - 1);
                        const float* pp = pb + (v - j0);
                        __m512 acc = (k0 == 0) ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(mask, pci + v);
                        for (int k = 0; k < kc; ++k)
                            acc = _mm512_fmadd_ps(_mm512_set1_ps(pai[k]),
                                _mm512_loadu_ps(pp + (size_t)k * JBLK), acc);
                        if constexpr (can_fuse) {
                            const float* bp = (fuse_nchw && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                            fuse_nchw_1(acc, bp[0], fmin, fmax);
                        }
                        _mm512_mask_storeu_ps(pci + v, mask, acc);
                    }
                }
            }
            if (!(can_fuse && post_fn.kind != post_op_kind::none))
                post_fn.apply_rows(i0, ie, C_batch[p], m, j0, jw);
        }
    });

}

// NHWC-native GEMM: C[n × m] = A[n × o] × packed_B[o × m]
// Optimized for NHWC Conv where n=spatial (large), m=output channels (moderate),
// o=K (moderate), and B=weights (constant, pre-packed by pack_b()).
// Key difference from dgemm_packed_b: tiles over spatial only (not M), keeping
// B-panels L1-hot across all 6-row groups within a K-block. No A-packing —
// A rows are read directly with sequential stride.
template <typename PostFn>
// @nnr-meta isa=AVX512 dtype=fp32 layout=NHWC special=GEMM tiling=[K,MR,NR] fusion=post_op
inline void dgemm_nhwc(int n, int m, int o, const float* __restrict A,
    const float* __restrict packed_B, float* __restrict C, const PostFn& post_fn)
{
    constexpr int JBLK = 64;
    constexpr int KC = 256;
    constexpr int IBLK = 64;
    int nj = (m + JBLK - 1) / JBLK;
    int nk = (o + KC - 1) / KC;
    int ni = (n + IBLK - 1) / IBLK;
    int nt_nhwc = ((int64_t)n * m * o > (1 << 21)) ? nnr::compute_threads(ni) : 1;

    // Fallback to dgemm_packed_b when B panels exceed L2 or A rows exceed L1.
    // B-hot/C-hot paths keep accumulators in registers across K-blocks,
    // avoiding C reload overhead. This works well when:
    //   - Packed B fits in half of L2 (leaves room for A panel + C lines)
    //   - A row group (8 rows × K) fits in L1
    // For large OC or large K, packed_b tiles both M and N for better locality.
    {
        size_t packed_b_total = (size_t)nj * nk * KC * JBLK * sizeof(float);
        size_t a_rows_bytes = (size_t)8 * o * sizeof(float);
        const size_t L2_HALF = (size_t)cpu_features().l2_kb * 1024 / 2;
        const size_t L1_BYTES = (size_t)cpu_features().l1d_kb * 1024;
        if (packed_b_total > L2_HALF || a_rows_bytes > L1_BYTES
            || (nk > 4 && ni <= 4))
        {
            NNR_PROFILE_COUNT("dgemm_nhwc:packed_b_fallback");
            NNR_PROFILE_SCOPE("nhwc->packed_b");
            avx512::dgemm_packed_b(n, m, o, A, packed_B, C, post_fn);
            return;
        }
    }

    // C-hot path: when K spans many blocks, keep C in registers across k-blocks.
    // Loop order: i-chunks → j-blocks → 6-row groups → v-slices → k-blocks.
    // Eliminates C reload traffic (nk loads → 1) at the cost of re-streaming A
    // per v-slice, which is cheap since A stays in L2.
    // [COLD: 0 hits in NCHW models. Would fire for deep-K NHWC Conv.]
    if (nk > 2) {
        NNR_PROFILE_COUNT("dgemm_nhwc:c_hot");
        nnr::for_dynamic(0, ni, nt_nhwc, [&](int /*tid*/, int ichunk) {
            int i0 = ichunk * IBLK;
            int ie = std::min(i0 + IBLK, n);
            for (int jt = 0; jt < nj; jt++) {
                int j0 = jt * JBLK;
                int je = std::min(j0 + JBLK, m);
                bool fuse = post_fn.kind != post_op_kind::none;
                const float* fb = fuse ? post_fn.bias : fused_zero_bias;
                float fmin = fuse ? post_fn.clip_min : -FLT_MAX;
                float fmax = fuse ? post_fn.clip_max : FLT_MAX;
                int i = i0;
#if 0 // LOCUST
;                # C-hot 8-row: pa decls + full + masked FMA with K-2x unroll
                for (; i + 8 <= ie; i += 8) {
;                gen_pa_decls(5, 8, "i", "o", "0", "A")
                    int v = j0;
;                for is_tail in (False, True):
;                    if is_tail:
                    if (v < je) {
                        __mmask16 mask = (__mmask16)((1u << (je - v)) - 1);
;                        pass
;                    else:
                    for (; v + 16 <= je; v += 16) {
;                        pass
;                    for r in range(0, 8, 2):
                        __m512 c@r@ = _mm512_setzero_ps(), c@r+1@ = _mm512_setzero_ps();
;                        pass
                        for (int kt = 0; kt < nk; kt++) {
                            int k0 = kt * KC;
                            int kc = std::min(KC, o - k0);
                            const float* pp = packed_B + ((size_t)jt * nk + kt) * KC * JBLK + (v - j0);
                            {
;                    gen_nhwc_8row_fma_k2x(7, "direct", "k0")
                        }
                        }
;                    fb_load = "_mm512_maskz_loadu_ps(mask, fb + v)" if is_tail else "_mm512_loadu_ps(fb + v)"
                        fuse_nhwc_8(c0, c1, c2, c3, c4, c5, c6, c7, @fb_load@, fmin, fmax);
;                    gen_nhwc_nrow_store(6, 8, masked=is_tail)
                    }
;                    pass
                }
#else // LOCUST
                for (; i + 8 <= ie; i += 8) {
                    const float* pa0 = A + (size_t)(i+0) * o + 0;
                    const float* pa1 = A + (size_t)(i+1) * o + 0;
                    const float* pa2 = A + (size_t)(i+2) * o + 0;
                    const float* pa3 = A + (size_t)(i+3) * o + 0;
                    const float* pa4 = A + (size_t)(i+4) * o + 0;
                    const float* pa5 = A + (size_t)(i+5) * o + 0;
                    const float* pa6 = A + (size_t)(i+6) * o + 0;
                    const float* pa7 = A + (size_t)(i+7) * o + 0;
                    int v = j0;
                    for (; v + 16 <= je; v += 16) {
                        __m512 c0 = _mm512_setzero_ps(), c1 = _mm512_setzero_ps();
                        __m512 c2 = _mm512_setzero_ps(), c3 = _mm512_setzero_ps();
                        __m512 c4 = _mm512_setzero_ps(), c5 = _mm512_setzero_ps();
                        __m512 c6 = _mm512_setzero_ps(), c7 = _mm512_setzero_ps();
                        for (int kt = 0; kt < nk; kt++) {
                            int k0 = kt * KC;
                            int kc = std::min(KC, o - k0);
                            const float* pp = packed_B + ((size_t)jt * nk + kt) * KC * JBLK + (v - j0);
                            {
                            int k = 0;
                            for (; k + 4 <= kc; k += 4) {
                                __m512 bv0 = _mm512_loadu_ps(pp + (size_t)k * JBLK);
                                __m512 bv1 = _mm512_loadu_ps(pp + (size_t)(k+1) * JBLK);
                                __m512 bv2 = _mm512_loadu_ps(pp + (size_t)(k+2) * JBLK);
                                __m512 bv3 = _mm512_loadu_ps(pp + (size_t)(k+3) * JBLK);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k0+k]), bv0, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k0+k]), bv0, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k0+k]), bv0, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k0+k]), bv0, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k0+k]), bv0, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k0+k]), bv0, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa6[k0+k]), bv0, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa7[k0+k]), bv0, c7);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k0+k+1]), bv1, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k0+k+1]), bv1, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k0+k+1]), bv1, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k0+k+1]), bv1, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k0+k+1]), bv1, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k0+k+1]), bv1, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa6[k0+k+1]), bv1, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa7[k0+k+1]), bv1, c7);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k0+k+2]), bv2, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k0+k+2]), bv2, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k0+k+2]), bv2, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k0+k+2]), bv2, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k0+k+2]), bv2, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k0+k+2]), bv2, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa6[k0+k+2]), bv2, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa7[k0+k+2]), bv2, c7);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k0+k+3]), bv3, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k0+k+3]), bv3, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k0+k+3]), bv3, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k0+k+3]), bv3, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k0+k+3]), bv3, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k0+k+3]), bv3, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa6[k0+k+3]), bv3, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa7[k0+k+3]), bv3, c7);
                            }
                            for (; k < kc; k++) {
                                __m512 bv = _mm512_loadu_ps(pp + (size_t)k * JBLK);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k0+k]), bv, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k0+k]), bv, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k0+k]), bv, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k0+k]), bv, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k0+k]), bv, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k0+k]), bv, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa6[k0+k]), bv, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa7[k0+k]), bv, c7);
                            }
                        }
                        }
                        fuse_nhwc_8(c0, c1, c2, c3, c4, c5, c6, c7, _mm512_loadu_ps(fb + v), fmin, fmax);
                        _mm512_storeu_ps(C + (size_t)(i+0) * m + v, c0);
                        _mm512_storeu_ps(C + (size_t)(i+1) * m + v, c1);
                        _mm512_storeu_ps(C + (size_t)(i+2) * m + v, c2);
                        _mm512_storeu_ps(C + (size_t)(i+3) * m + v, c3);
                        _mm512_storeu_ps(C + (size_t)(i+4) * m + v, c4);
                        _mm512_storeu_ps(C + (size_t)(i+5) * m + v, c5);
                        _mm512_storeu_ps(C + (size_t)(i+6) * m + v, c6);
                        _mm512_storeu_ps(C + (size_t)(i+7) * m + v, c7);
                    }
                    if (v < je) {
                        __mmask16 mask = (__mmask16)((1u << (je - v)) - 1);
                        __m512 c0 = _mm512_setzero_ps(), c1 = _mm512_setzero_ps();
                        __m512 c2 = _mm512_setzero_ps(), c3 = _mm512_setzero_ps();
                        __m512 c4 = _mm512_setzero_ps(), c5 = _mm512_setzero_ps();
                        __m512 c6 = _mm512_setzero_ps(), c7 = _mm512_setzero_ps();
                        for (int kt = 0; kt < nk; kt++) {
                            int k0 = kt * KC;
                            int kc = std::min(KC, o - k0);
                            const float* pp = packed_B + ((size_t)jt * nk + kt) * KC * JBLK + (v - j0);
                            {
                            int k = 0;
                            for (; k + 4 <= kc; k += 4) {
                                __m512 bv0 = _mm512_loadu_ps(pp + (size_t)k * JBLK);
                                __m512 bv1 = _mm512_loadu_ps(pp + (size_t)(k+1) * JBLK);
                                __m512 bv2 = _mm512_loadu_ps(pp + (size_t)(k+2) * JBLK);
                                __m512 bv3 = _mm512_loadu_ps(pp + (size_t)(k+3) * JBLK);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k0+k]), bv0, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k0+k]), bv0, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k0+k]), bv0, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k0+k]), bv0, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k0+k]), bv0, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k0+k]), bv0, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa6[k0+k]), bv0, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa7[k0+k]), bv0, c7);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k0+k+1]), bv1, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k0+k+1]), bv1, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k0+k+1]), bv1, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k0+k+1]), bv1, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k0+k+1]), bv1, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k0+k+1]), bv1, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa6[k0+k+1]), bv1, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa7[k0+k+1]), bv1, c7);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k0+k+2]), bv2, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k0+k+2]), bv2, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k0+k+2]), bv2, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k0+k+2]), bv2, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k0+k+2]), bv2, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k0+k+2]), bv2, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa6[k0+k+2]), bv2, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa7[k0+k+2]), bv2, c7);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k0+k+3]), bv3, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k0+k+3]), bv3, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k0+k+3]), bv3, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k0+k+3]), bv3, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k0+k+3]), bv3, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k0+k+3]), bv3, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa6[k0+k+3]), bv3, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa7[k0+k+3]), bv3, c7);
                            }
                            for (; k < kc; k++) {
                                __m512 bv = _mm512_loadu_ps(pp + (size_t)k * JBLK);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k0+k]), bv, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k0+k]), bv, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k0+k]), bv, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k0+k]), bv, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k0+k]), bv, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k0+k]), bv, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa6[k0+k]), bv, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa7[k0+k]), bv, c7);
                            }
                        }
                        }
                        fuse_nhwc_8(c0, c1, c2, c3, c4, c5, c6, c7, _mm512_maskz_loadu_ps(mask, fb + v), fmin, fmax);
                        _mm512_mask_storeu_ps(C + (size_t)(i+0) * m + v, mask, c0);
                        _mm512_mask_storeu_ps(C + (size_t)(i+1) * m + v, mask, c1);
                        _mm512_mask_storeu_ps(C + (size_t)(i+2) * m + v, mask, c2);
                        _mm512_mask_storeu_ps(C + (size_t)(i+3) * m + v, mask, c3);
                        _mm512_mask_storeu_ps(C + (size_t)(i+4) * m + v, mask, c4);
                        _mm512_mask_storeu_ps(C + (size_t)(i+5) * m + v, mask, c5);
                        _mm512_mask_storeu_ps(C + (size_t)(i+6) * m + v, mask, c6);
                        _mm512_mask_storeu_ps(C + (size_t)(i+7) * m + v, mask, c7);
                    }
                }
#endif // LOCUST
                for (; i < ie; i++) {
                    const float* pai = A + (size_t)i * o;
                    int v = j0;
                    for (; v + 16 <= je; v += 16) {
                        __m512 acc = _mm512_setzero_ps();
                        for (int kt = 0; kt < nk; kt++) {
                            int k0 = kt * KC;
                            int kc = std::min(KC, o - k0);
                            const float* pp = packed_B + ((size_t)jt * nk + kt) * KC * JBLK + (v - j0);
                            for (int k = 0; k < kc; ++k)
                                acc = _mm512_fmadd_ps(_mm512_set1_ps(pai[k0 + k]),
                                    _mm512_loadu_ps(pp + (size_t)k * JBLK), acc);
                        }
                        fuse_nhwc_1(acc, _mm512_loadu_ps(fb + v), fmin, fmax);
                        _mm512_storeu_ps(C + (size_t)i * m + v, acc);
                    }
                    if (v < je) {
                        __mmask16 mask = (__mmask16)((1u << (je - v)) - 1);
                        __m512 acc = _mm512_setzero_ps();
                        for (int kt = 0; kt < nk; kt++) {
                            int k0 = kt * KC;
                            int kc = std::min(KC, o - k0);
                            const float* pp = packed_B + ((size_t)jt * nk + kt) * KC * JBLK + (v - j0);
                            for (int k = 0; k < kc; ++k)
                                acc = _mm512_fmadd_ps(_mm512_set1_ps(pai[k0 + k]),
                                    _mm512_loadu_ps(pp + (size_t)k * JBLK), acc);
                        }
                        fuse_nhwc_1(acc, _mm512_maskz_loadu_ps(mask, fb + v), fmin, fmax);
                        _mm512_mask_storeu_ps(C + (size_t)i * m + v, mask, acc);
                    }
                }
            }
            if (post_fn.kind == post_op_kind::none) {
                for (int jt = 0; jt < nj; jt++) {
                    int j0 = jt * JBLK;
                    int jw = std::min(JBLK, m - j0);
                    post_fn.apply_rows(i0, ie, C, m, j0, jw);
                }
            }
        });
        return;
    }

    // B-hot path: few k-blocks, B-panel shared across 8-row groups.
    // Loop order: i-chunks → k-blocks → j-blocks → 6-row groups.
    // [COLD: 0 hits in NCHW models. Primary NHWC path for typical Conv shapes.]
    NNR_PROFILE_COUNT("dgemm_nhwc:b_hot");
    nnr::for_dynamic(0, ni, nt_nhwc, [&](int /*tid*/, int ichunk) {
        int i0 = ichunk * IBLK;
        int ie = std::min(i0 + IBLK, n);

        for (int kt = 0; kt < nk; kt++) {
            int k0 = kt * KC;
            int kc = std::min(KC, o - k0);
            bool last_k = (k0 + kc == o);

            bool fuse = last_k && post_fn.kind != post_op_kind::none;
            const float* fb = fuse ? post_fn.bias : fused_zero_bias;
            float fmin = fuse ? post_fn.clip_min : -FLT_MAX;
            float fmax = fuse ? post_fn.clip_max : FLT_MAX;

            for (int jt = 0; jt < nj; jt++) {
                int j0 = jt * JBLK;
                int je = std::min(j0 + JBLK, m);
                const float* pb = packed_B + ((size_t)jt * nk + kt) * KC * JBLK;

                // 8-row groups
                int i = i0;
#if 0 // LOCUST
;                gen_nhwc_nrow_block(4, 8, fma_fn=lambda d: gen_nhwc_8row_fma_k2x(d, "packed"), brace_wrap=True)
#else // LOCUST
                for (; i + 8 <= ie; i += 8) {
                    const float* pa0 = A + (size_t)(i+0) * o + k0;
                    const float* pa1 = A + (size_t)(i+1) * o + k0;
                    const float* pa2 = A + (size_t)(i+2) * o + k0;
                    const float* pa3 = A + (size_t)(i+3) * o + k0;
                    const float* pa4 = A + (size_t)(i+4) * o + k0;
                    const float* pa5 = A + (size_t)(i+5) * o + k0;
                    const float* pa6 = A + (size_t)(i+6) * o + k0;
                    const float* pa7 = A + (size_t)(i+7) * o + k0;
                    int v = j0;
                    for (; v + 16 <= je; v += 16) {
                        const float* pp = pb + (v - j0);
                        __m512 c0, c1, c2, c3, c4, c5, c6, c7;
                        if (k0 == 0) {
                            c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = _mm512_setzero_ps();
                        } else {
                            c0 = _mm512_loadu_ps(C + (size_t)(i+0) * m + v);
                            c1 = _mm512_loadu_ps(C + (size_t)(i+1) * m + v);
                            c2 = _mm512_loadu_ps(C + (size_t)(i+2) * m + v);
                            c3 = _mm512_loadu_ps(C + (size_t)(i+3) * m + v);
                            c4 = _mm512_loadu_ps(C + (size_t)(i+4) * m + v);
                            c5 = _mm512_loadu_ps(C + (size_t)(i+5) * m + v);
                            c6 = _mm512_loadu_ps(C + (size_t)(i+6) * m + v);
                            c7 = _mm512_loadu_ps(C + (size_t)(i+7) * m + v);
                        }
                        {
                            int k = 0;
                            for (; k + 4 <= kc; k += 4) {
                                __m512 bv0 = _mm512_loadu_ps(pp + (size_t)k * JBLK);
                                __m512 bv1 = _mm512_loadu_ps(pp + (size_t)(k+1) * JBLK);
                                __m512 bv2 = _mm512_loadu_ps(pp + (size_t)(k+2) * JBLK);
                                __m512 bv3 = _mm512_loadu_ps(pp + (size_t)(k+3) * JBLK);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k]), bv0, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k]), bv0, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k]), bv0, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k]), bv0, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k]), bv0, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k]), bv0, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa6[k]), bv0, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa7[k]), bv0, c7);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k+1]), bv1, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k+1]), bv1, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k+1]), bv1, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k+1]), bv1, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k+1]), bv1, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k+1]), bv1, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa6[k+1]), bv1, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa7[k+1]), bv1, c7);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k+2]), bv2, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k+2]), bv2, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k+2]), bv2, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k+2]), bv2, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k+2]), bv2, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k+2]), bv2, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa6[k+2]), bv2, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa7[k+2]), bv2, c7);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k+3]), bv3, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k+3]), bv3, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k+3]), bv3, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k+3]), bv3, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k+3]), bv3, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k+3]), bv3, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa6[k+3]), bv3, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa7[k+3]), bv3, c7);
                            }
                            for (; k < kc; k++) {
                                __m512 bv = _mm512_loadu_ps(pp + (size_t)k * JBLK);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k]), bv, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k]), bv, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k]), bv, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k]), bv, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k]), bv, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k]), bv, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa6[k]), bv, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa7[k]), bv, c7);
                            }
                        }
                        fuse_nhwc_8(c0, c1, c2, c3, c4, c5, c6, c7, _mm512_loadu_ps(fb + v), fmin, fmax);
                        _mm512_storeu_ps(C + (size_t)(i+0) * m + v, c0);
                        _mm512_storeu_ps(C + (size_t)(i+1) * m + v, c1);
                        _mm512_storeu_ps(C + (size_t)(i+2) * m + v, c2);
                        _mm512_storeu_ps(C + (size_t)(i+3) * m + v, c3);
                        _mm512_storeu_ps(C + (size_t)(i+4) * m + v, c4);
                        _mm512_storeu_ps(C + (size_t)(i+5) * m + v, c5);
                        _mm512_storeu_ps(C + (size_t)(i+6) * m + v, c6);
                        _mm512_storeu_ps(C + (size_t)(i+7) * m + v, c7);
                    }
                    if (v < je) {
                        __mmask16 mask = (__mmask16)((1u << (je - v)) - 1);
                        const float* pp = pb + (v - j0);
                        __m512 c0, c1, c2, c3, c4, c5, c6, c7;
                        if (k0 == 0) {
                            c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = _mm512_setzero_ps();
                        } else {
                            c0 = _mm512_maskz_loadu_ps(mask, C + (size_t)(i+0) * m + v);
                            c1 = _mm512_maskz_loadu_ps(mask, C + (size_t)(i+1) * m + v);
                            c2 = _mm512_maskz_loadu_ps(mask, C + (size_t)(i+2) * m + v);
                            c3 = _mm512_maskz_loadu_ps(mask, C + (size_t)(i+3) * m + v);
                            c4 = _mm512_maskz_loadu_ps(mask, C + (size_t)(i+4) * m + v);
                            c5 = _mm512_maskz_loadu_ps(mask, C + (size_t)(i+5) * m + v);
                            c6 = _mm512_maskz_loadu_ps(mask, C + (size_t)(i+6) * m + v);
                            c7 = _mm512_maskz_loadu_ps(mask, C + (size_t)(i+7) * m + v);
                        }
                        {
                            int k = 0;
                            for (; k + 4 <= kc; k += 4) {
                                __m512 bv0 = _mm512_loadu_ps(pp + (size_t)k * JBLK);
                                __m512 bv1 = _mm512_loadu_ps(pp + (size_t)(k+1) * JBLK);
                                __m512 bv2 = _mm512_loadu_ps(pp + (size_t)(k+2) * JBLK);
                                __m512 bv3 = _mm512_loadu_ps(pp + (size_t)(k+3) * JBLK);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k]), bv0, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k]), bv0, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k]), bv0, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k]), bv0, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k]), bv0, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k]), bv0, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa6[k]), bv0, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa7[k]), bv0, c7);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k+1]), bv1, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k+1]), bv1, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k+1]), bv1, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k+1]), bv1, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k+1]), bv1, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k+1]), bv1, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa6[k+1]), bv1, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa7[k+1]), bv1, c7);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k+2]), bv2, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k+2]), bv2, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k+2]), bv2, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k+2]), bv2, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k+2]), bv2, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k+2]), bv2, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa6[k+2]), bv2, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa7[k+2]), bv2, c7);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k+3]), bv3, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k+3]), bv3, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k+3]), bv3, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k+3]), bv3, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k+3]), bv3, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k+3]), bv3, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa6[k+3]), bv3, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa7[k+3]), bv3, c7);
                            }
                            for (; k < kc; k++) {
                                __m512 bv = _mm512_loadu_ps(pp + (size_t)k * JBLK);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k]), bv, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k]), bv, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k]), bv, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k]), bv, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k]), bv, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k]), bv, c5);
                                c6 = _mm512_fmadd_ps(_mm512_set1_ps(pa6[k]), bv, c6);
                                c7 = _mm512_fmadd_ps(_mm512_set1_ps(pa7[k]), bv, c7);
                            }
                        }
                        fuse_nhwc_8(c0, c1, c2, c3, c4, c5, c6, c7, _mm512_maskz_loadu_ps(mask, fb + v), fmin, fmax);
                        _mm512_mask_storeu_ps(C + (size_t)(i+0) * m + v, mask, c0);
                        _mm512_mask_storeu_ps(C + (size_t)(i+1) * m + v, mask, c1);
                        _mm512_mask_storeu_ps(C + (size_t)(i+2) * m + v, mask, c2);
                        _mm512_mask_storeu_ps(C + (size_t)(i+3) * m + v, mask, c3);
                        _mm512_mask_storeu_ps(C + (size_t)(i+4) * m + v, mask, c4);
                        _mm512_mask_storeu_ps(C + (size_t)(i+5) * m + v, mask, c5);
                        _mm512_mask_storeu_ps(C + (size_t)(i+6) * m + v, mask, c6);
                        _mm512_mask_storeu_ps(C + (size_t)(i+7) * m + v, mask, c7);
                    }
                }
#endif // LOCUST
                // Remainder rows (< 8)
                for (; i < ie; i++) {
                    const float* pai = A + (size_t)i * o + k0;
                    int v = j0;
                    for (; v + 16 <= je; v += 16) {
                        const float* pp = pb + (v - j0);
                        __m512 acc = (k0 == 0) ? _mm512_setzero_ps()
                            : _mm512_loadu_ps(C + (size_t)i * m + v);
                        for (int k = 0; k < kc; ++k)
                            acc = _mm512_fmadd_ps(_mm512_set1_ps(pai[k]),
                                _mm512_loadu_ps(pp + (size_t)k * JBLK), acc);
                        fuse_nhwc_1(acc, _mm512_loadu_ps(fb + v), fmin, fmax);
                        _mm512_storeu_ps(C + (size_t)i * m + v, acc);
                    }
                    if (v < je) {
                        __mmask16 mask = (__mmask16)((1u << (je - v)) - 1);
                        const float* pp = pb + (v - j0);
                        __m512 acc = (k0 == 0) ? _mm512_setzero_ps()
                            : _mm512_maskz_loadu_ps(mask, C + (size_t)i * m + v);
                        for (int k = 0; k < kc; ++k)
                            acc = _mm512_fmadd_ps(_mm512_set1_ps(pai[k]),
                                _mm512_loadu_ps(pp + (size_t)k * JBLK), acc);
                        fuse_nhwc_1(acc, _mm512_maskz_loadu_ps(mask, fb + v), fmin, fmax);
                        _mm512_mask_storeu_ps(C + (size_t)i * m + v, mask, acc);
                    }
                }
            }
        }
        // Post-process after all K-blocks complete (bias + activation on L1-hot data)
        if (post_fn.kind == post_op_kind::none) {
            for (int jt = 0; jt < nj; jt++) {
                int j0 = jt * JBLK;
                int jw = std::min(JBLK, m - j0);
                post_fn.apply_rows(i0, ie, C, m, j0, jw);
            }
        }
    });
}

// Batched GEMM for NHWC Winograd: 36 independent GEMMs sharing dispatch.
// C_batch[p][n×m] = A_batch[p][n×o] × packed_B_batch[p][o×m]  for p in [0..36)
template <typename PostFn>
// @nnr-meta isa=AVX512 dtype=fp32 layout=NHWC special=[GEMM,Winograd] tiling=[K,MR,NR] fusion=post_op
inline void dgemm_packed_b_batch36(
    int n, int m, int o,
    const float* const A_batch[36],
    const float* const packed_B_batch[36],
    float* const C_batch[36],
    const PostFn& post_fn)
{
    constexpr int JBLK = 64;
    constexpr int KC = 256;
    constexpr int IBLK = 64;
    int nj = (m + JBLK - 1) / JBLK;
    int nk = (o + KC - 1) / KC;
    int ni = (n + IBLK - 1) / IBLK;

    // Small-spatial fallback: when n fits in 1 i-block and m spans many j-blocks,
    // use dgemm_packed_b tiling (both i and j dimensions with A-packing) but
    // parallelize across 36 positions × ntiles — critical because individual
    // dgemm_packed_b calls may be too small to trigger threading on their own.
    if (ni <= 1 && nj > 2) {
        int ntiles = std::max(ni, 1) * nj;
        int total = 36 * ntiles;
        int nt = ((int64_t)n * m * o * 36 > (1 << 21)) ? nnr::compute_threads(total) : 1;
        NNR_POOL_ENSURE_SCRATCH((size_t)KC * 8 * sizeof(float));
        nnr::for_dynamic(0, total, nt, [&](int tid, int task) {
            float* pa_pack = (float*)NNR_POOL_SCRATCH(tid);
            int p = task / ntiles;
            int tile = task % ntiles;
            const float* A = A_batch[p];
            const float* packed_B = packed_B_batch[p];
            float* C = C_batch[p];
            int i0 = (tile / nj) * JBLK;
            int j0 = (tile % nj) * JBLK;
            int jt = tile % nj;
            int ie = std::min(i0 + JBLK, n);
            int je = std::min(j0 + JBLK, m);
            for (int kt = 0; kt < nk; kt++) {
                int k0 = kt * KC;
                int kc = std::min(KC, o - k0);
                bool last_k = (k0 + kc == o);
                const float* pb = packed_B + ((size_t)jt * nk + kt) * KC * JBLK;

                bool fuse = last_k && post_fn.kind != post_op_kind::none;
                const float* fb = fuse ? post_fn.bias : fused_zero_bias;
                float fmin = fuse ? post_fn.clip_min : -FLT_MAX;
                float fmax = fuse ? post_fn.clip_max : FLT_MAX;

                int i = i0;
                for (; i + 8 <= ie; i += 8) {
                    const float* pa[8];
                    float* pc[8];
                    for (int r = 0; r < 8; r++) {
                        pa[r] = A + (size_t)(i + r) * o + k0;
                        pc[r] = C + (size_t)(i + r) * m;
                    }
#if 0 // LOCUST
;                    gen_pack_a_8(5)
#else // LOCUST
                    for (int k = 0; k < kc; k++) {
                        pa_pack[k * 8 + 0] = pa[0][k]; pa_pack[k * 8 + 1] = pa[1][k];
                        pa_pack[k * 8 + 2] = pa[2][k]; pa_pack[k * 8 + 3] = pa[3][k];
                        pa_pack[k * 8 + 4] = pa[4][k]; pa_pack[k * 8 + 5] = pa[5][k];
                        pa_pack[k * 8 + 6] = pa[6][k]; pa_pack[k * 8 + 7] = pa[7][k];
                    }
#endif // LOCUST
#if 0 // LOCUST
;                    def gen_packed_a_8row_vloop(d, pc_fmt="pc[{r}]"):
;                        NR = 8
;                        cs = ", ".join(f"c{r}" for r in range(NR))
;                        cs_chain = " = ".join(f"c{r}" for r in range(NR))
int v = j0;
;                        for is_tail in (False, True):
;                            if is_tail:
if (v < je) {
     __mmask16 mask = (__mmask16)((1u << (je - v)) - 1);
;                                pass
;                            else:
for (; v + 16 <= je; v += 16) {
;                                pass
     const float* pp = pb + (v - j0);
     __m512 @cs@;
     if (k0 == 0) { @cs_chain@ = _mm512_setzero_ps(); }
     else {
;                            load = "_mm512_maskz_loadu_ps(mask, " if is_tail else "_mm512_loadu_ps("
;                            for r in range(0, NR, 2):
;                                pc0 = pc_fmt.format(r=r)
;                                pc1 = pc_fmt.format(r=r+1)
         c@r@ = @load@@pc0@ + v); c@r+1@ = @load@@pc1@ + v);
;                                pass
     }
     for (int k = 0; k < kc; k++) {
         if (k + 8 < kc) _mm_prefetch((const char*)(pp + (size_t)(k+8) * JBLK), _MM_HINT_T0);
         __m512 bv = _mm512_loadu_ps(pp + (size_t)k * JBLK);
         const float* ap = pa_pack + k * 8;
;                            for r in range(NR):
         c@r@ = _mm512_fmadd_ps(_mm512_set1_ps(ap[@r@]), bv, c@r@);
;                                pass
     }
;                            fb_load = "_mm512_maskz_loadu_ps(mask, fb + v)" if is_tail else "_mm512_loadu_ps(fb + v)"
     fuse_nhwc_8(@cs@, @fb_load@, fmin, fmax);
;                            store = "_mm512_mask_storeu_ps(" if is_tail else "_mm512_storeu_ps("
;                            mask_arg = "mask, " if is_tail else ""
;                            for r in range(0, NR, 2):
;                                pc0 = pc_fmt.format(r=r)
;                                pc1 = pc_fmt.format(r=r+1)
     @store@@pc0@ + v, @mask_arg@c@r@); @store@@pc1@ + v, @mask_arg@c@r+1@);
;                                pass
}
;                            pass
;
;                    gen_packed_a_8row_vloop(5)
#else // LOCUST
int v = j0;
for (; v + 16 <= je; v += 16) {
     const float* pp = pb + (v - j0);
     __m512 c0, c1, c2, c3, c4, c5, c6, c7;
     if (k0 == 0) { c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = _mm512_setzero_ps(); }
     else {
         c0 = _mm512_loadu_ps(pc[0] + v); c1 = _mm512_loadu_ps(pc[1] + v);
         c2 = _mm512_loadu_ps(pc[2] + v); c3 = _mm512_loadu_ps(pc[3] + v);
         c4 = _mm512_loadu_ps(pc[4] + v); c5 = _mm512_loadu_ps(pc[5] + v);
         c6 = _mm512_loadu_ps(pc[6] + v); c7 = _mm512_loadu_ps(pc[7] + v);
     }
     for (int k = 0; k < kc; k++) {
         if (k + 8 < kc) _mm_prefetch((const char*)(pp + (size_t)(k+8) * JBLK), _MM_HINT_T0);
         __m512 bv = _mm512_loadu_ps(pp + (size_t)k * JBLK);
         const float* ap = pa_pack + k * 8;
         c0 = _mm512_fmadd_ps(_mm512_set1_ps(ap[0]), bv, c0);
         c1 = _mm512_fmadd_ps(_mm512_set1_ps(ap[1]), bv, c1);
         c2 = _mm512_fmadd_ps(_mm512_set1_ps(ap[2]), bv, c2);
         c3 = _mm512_fmadd_ps(_mm512_set1_ps(ap[3]), bv, c3);
         c4 = _mm512_fmadd_ps(_mm512_set1_ps(ap[4]), bv, c4);
         c5 = _mm512_fmadd_ps(_mm512_set1_ps(ap[5]), bv, c5);
         c6 = _mm512_fmadd_ps(_mm512_set1_ps(ap[6]), bv, c6);
         c7 = _mm512_fmadd_ps(_mm512_set1_ps(ap[7]), bv, c7);
     }
     fuse_nhwc_8(c0, c1, c2, c3, c4, c5, c6, c7, _mm512_loadu_ps(fb + v), fmin, fmax);
     _mm512_storeu_ps(pc[0] + v, c0); _mm512_storeu_ps(pc[1] + v, c1);
     _mm512_storeu_ps(pc[2] + v, c2); _mm512_storeu_ps(pc[3] + v, c3);
     _mm512_storeu_ps(pc[4] + v, c4); _mm512_storeu_ps(pc[5] + v, c5);
     _mm512_storeu_ps(pc[6] + v, c6); _mm512_storeu_ps(pc[7] + v, c7);
}
if (v < je) {
     __mmask16 mask = (__mmask16)((1u << (je - v)) - 1);
     const float* pp = pb + (v - j0);
     __m512 c0, c1, c2, c3, c4, c5, c6, c7;
     if (k0 == 0) { c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = _mm512_setzero_ps(); }
     else {
         c0 = _mm512_maskz_loadu_ps(mask, pc[0] + v); c1 = _mm512_maskz_loadu_ps(mask, pc[1] + v);
         c2 = _mm512_maskz_loadu_ps(mask, pc[2] + v); c3 = _mm512_maskz_loadu_ps(mask, pc[3] + v);
         c4 = _mm512_maskz_loadu_ps(mask, pc[4] + v); c5 = _mm512_maskz_loadu_ps(mask, pc[5] + v);
         c6 = _mm512_maskz_loadu_ps(mask, pc[6] + v); c7 = _mm512_maskz_loadu_ps(mask, pc[7] + v);
     }
     for (int k = 0; k < kc; k++) {
         if (k + 8 < kc) _mm_prefetch((const char*)(pp + (size_t)(k+8) * JBLK), _MM_HINT_T0);
         __m512 bv = _mm512_loadu_ps(pp + (size_t)k * JBLK);
         const float* ap = pa_pack + k * 8;
         c0 = _mm512_fmadd_ps(_mm512_set1_ps(ap[0]), bv, c0);
         c1 = _mm512_fmadd_ps(_mm512_set1_ps(ap[1]), bv, c1);
         c2 = _mm512_fmadd_ps(_mm512_set1_ps(ap[2]), bv, c2);
         c3 = _mm512_fmadd_ps(_mm512_set1_ps(ap[3]), bv, c3);
         c4 = _mm512_fmadd_ps(_mm512_set1_ps(ap[4]), bv, c4);
         c5 = _mm512_fmadd_ps(_mm512_set1_ps(ap[5]), bv, c5);
         c6 = _mm512_fmadd_ps(_mm512_set1_ps(ap[6]), bv, c6);
         c7 = _mm512_fmadd_ps(_mm512_set1_ps(ap[7]), bv, c7);
     }
     fuse_nhwc_8(c0, c1, c2, c3, c4, c5, c6, c7, _mm512_maskz_loadu_ps(mask, fb + v), fmin, fmax);
     _mm512_mask_storeu_ps(pc[0] + v, mask, c0); _mm512_mask_storeu_ps(pc[1] + v, mask, c1);
     _mm512_mask_storeu_ps(pc[2] + v, mask, c2); _mm512_mask_storeu_ps(pc[3] + v, mask, c3);
     _mm512_mask_storeu_ps(pc[4] + v, mask, c4); _mm512_mask_storeu_ps(pc[5] + v, mask, c5);
     _mm512_mask_storeu_ps(pc[6] + v, mask, c6); _mm512_mask_storeu_ps(pc[7] + v, mask, c7);
}
#endif // LOCUST
                }
                for (; i < ie; i++) {
                    float* pci = C + (size_t)i * m;
                    const float* pai = A + (size_t)i * o + k0;
                    int v = j0;
                    for (; v + 16 <= je; v += 16) {
                        const float* pp = pb + (v - j0);
                        __m512 acc = (k0 == 0) ? _mm512_setzero_ps() : _mm512_loadu_ps(pci + v);
                        for (int k = 0; k < kc; ++k)
                            acc = _mm512_fmadd_ps(_mm512_set1_ps(pai[k]),
                                _mm512_loadu_ps(pp + (size_t)k * JBLK), acc);
                        fuse_nhwc_1(acc, _mm512_loadu_ps(fb + v), fmin, fmax);
                        _mm512_storeu_ps(pci + v, acc);
                    }
                    if (v < je) {
                        __mmask16 mask = (__mmask16)((1u << (je - v)) - 1);
                        const float* pp = pb + (v - j0);
                        __m512 acc = (k0 == 0) ? _mm512_setzero_ps()
                            : _mm512_maskz_loadu_ps(mask, pci + v);
                        for (int k = 0; k < kc; ++k)
                            acc = _mm512_fmadd_ps(_mm512_set1_ps(pai[k]),
                                _mm512_loadu_ps(pp + (size_t)k * JBLK), acc);
                        fuse_nhwc_1(acc, _mm512_maskz_loadu_ps(mask, fb + v), fmin, fmax);
                        _mm512_mask_storeu_ps(pci + v, mask, acc);
                    }
                }
            }
            if (post_fn.kind == post_op_kind::none) {
                int jw = std::min(JBLK, m - j0);
                post_fn.apply_rows(i0, ie, C, m, j0, jw);
            }
        });
        return;
    }

    int total_work = ni * 36;
    int nt = ((int64_t)n * m * o * 36 > (1 << 21)) ? nnr::compute_threads(total_work) : 1;

    nnr::for_dynamic(0, total_work, nt, [&](int /*tid*/, int work_idx) {
        int ichunk = work_idx / 36;
        int p = work_idx % 36;
        int i0 = ichunk * IBLK;
        int ie = std::min(i0 + IBLK, n);

        {
            const float* A = A_batch[p];
            float* C = C_batch[p];

            for (int kt = 0; kt < nk; kt++) {
                int k0 = kt * KC;
                int kc = std::min(KC, o - k0);
                bool last_k = (k0 + kc == o);

                for (int jt = 0; jt < nj; jt++) {
                    int j0 = jt * JBLK;
                    int je = std::min(j0 + JBLK, m);
                    const float* pb = packed_B_batch[p] + ((size_t)jt * nk + kt) * KC * JBLK;

                    bool fuse = last_k && post_fn.kind != post_op_kind::none;
                    const float* fb = fuse ? post_fn.bias : fused_zero_bias;
                    float fmin = fuse ? post_fn.clip_min : -FLT_MAX;
                    float fmax = fuse ? post_fn.clip_max : FLT_MAX;

                    int i = i0;
#if 0 // LOCUST
;                    gen_nhwc_nrow_block(5, 6)
#else // LOCUST
                    for (; i + 6 <= ie; i += 6) {
                        const float* pa0 = A + (size_t)(i+0) * o + k0;
                        const float* pa1 = A + (size_t)(i+1) * o + k0;
                        const float* pa2 = A + (size_t)(i+2) * o + k0;
                        const float* pa3 = A + (size_t)(i+3) * o + k0;
                        const float* pa4 = A + (size_t)(i+4) * o + k0;
                        const float* pa5 = A + (size_t)(i+5) * o + k0;
                        int v = j0;
                        for (; v + 16 <= je; v += 16) {
                            const float* pp = pb + (v - j0);
                            __m512 c0, c1, c2, c3, c4, c5;
                            if (k0 == 0) {
                                c0 = c1 = c2 = c3 = c4 = c5 = _mm512_setzero_ps();
                            } else {
                                c0 = _mm512_loadu_ps(C + (size_t)(i+0) * m + v);
                                c1 = _mm512_loadu_ps(C + (size_t)(i+1) * m + v);
                                c2 = _mm512_loadu_ps(C + (size_t)(i+2) * m + v);
                                c3 = _mm512_loadu_ps(C + (size_t)(i+3) * m + v);
                                c4 = _mm512_loadu_ps(C + (size_t)(i+4) * m + v);
                                c5 = _mm512_loadu_ps(C + (size_t)(i+5) * m + v);
                            }
                            for (int k = 0; k < kc; k++) {
                                __m512 bv = _mm512_loadu_ps(pp + (size_t)k * JBLK);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k]), bv, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k]), bv, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k]), bv, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k]), bv, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k]), bv, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k]), bv, c5);
                            }
                            fuse_nhwc_6(c0, c1, c2, c3, c4, c5, _mm512_loadu_ps(fb + v), fmin, fmax);
                            _mm512_storeu_ps(C + (size_t)(i+0) * m + v, c0);
                            _mm512_storeu_ps(C + (size_t)(i+1) * m + v, c1);
                            _mm512_storeu_ps(C + (size_t)(i+2) * m + v, c2);
                            _mm512_storeu_ps(C + (size_t)(i+3) * m + v, c3);
                            _mm512_storeu_ps(C + (size_t)(i+4) * m + v, c4);
                            _mm512_storeu_ps(C + (size_t)(i+5) * m + v, c5);
                        }
                        if (v < je) {
                            __mmask16 mask = (__mmask16)((1u << (je - v)) - 1);
                            const float* pp = pb + (v - j0);
                            __m512 c0, c1, c2, c3, c4, c5;
                            if (k0 == 0) {
                                c0 = c1 = c2 = c3 = c4 = c5 = _mm512_setzero_ps();
                            } else {
                                c0 = _mm512_maskz_loadu_ps(mask, C + (size_t)(i+0) * m + v);
                                c1 = _mm512_maskz_loadu_ps(mask, C + (size_t)(i+1) * m + v);
                                c2 = _mm512_maskz_loadu_ps(mask, C + (size_t)(i+2) * m + v);
                                c3 = _mm512_maskz_loadu_ps(mask, C + (size_t)(i+3) * m + v);
                                c4 = _mm512_maskz_loadu_ps(mask, C + (size_t)(i+4) * m + v);
                                c5 = _mm512_maskz_loadu_ps(mask, C + (size_t)(i+5) * m + v);
                            }
                            for (int k = 0; k < kc; k++) {
                                __m512 bv = _mm512_loadu_ps(pp + (size_t)k * JBLK);
                                c0 = _mm512_fmadd_ps(_mm512_set1_ps(pa0[k]), bv, c0);
                                c1 = _mm512_fmadd_ps(_mm512_set1_ps(pa1[k]), bv, c1);
                                c2 = _mm512_fmadd_ps(_mm512_set1_ps(pa2[k]), bv, c2);
                                c3 = _mm512_fmadd_ps(_mm512_set1_ps(pa3[k]), bv, c3);
                                c4 = _mm512_fmadd_ps(_mm512_set1_ps(pa4[k]), bv, c4);
                                c5 = _mm512_fmadd_ps(_mm512_set1_ps(pa5[k]), bv, c5);
                            }
                            fuse_nhwc_6(c0, c1, c2, c3, c4, c5, _mm512_maskz_loadu_ps(mask, fb + v), fmin, fmax);
                            _mm512_mask_storeu_ps(C + (size_t)(i+0) * m + v, mask, c0);
                            _mm512_mask_storeu_ps(C + (size_t)(i+1) * m + v, mask, c1);
                            _mm512_mask_storeu_ps(C + (size_t)(i+2) * m + v, mask, c2);
                            _mm512_mask_storeu_ps(C + (size_t)(i+3) * m + v, mask, c3);
                            _mm512_mask_storeu_ps(C + (size_t)(i+4) * m + v, mask, c4);
                            _mm512_mask_storeu_ps(C + (size_t)(i+5) * m + v, mask, c5);
                        }
                    }
#endif // LOCUST
                    for (; i < ie; i++) {
                        const float* pai = A + (size_t)i * o + k0;
                        int v = j0;
                        for (; v + 16 <= je; v += 16) {
                            const float* pp = pb + (v - j0);
                            __m512 acc = (k0 == 0) ? _mm512_setzero_ps()
                                : _mm512_loadu_ps(C + (size_t)i * m + v);
                            for (int k = 0; k < kc; ++k)
                                acc = _mm512_fmadd_ps(_mm512_set1_ps(pai[k]),
                                    _mm512_loadu_ps(pp + (size_t)k * JBLK), acc);
                            fuse_nhwc_1(acc, _mm512_loadu_ps(fb + v), fmin, fmax);
                            _mm512_storeu_ps(C + (size_t)i * m + v, acc);
                        }
                        if (v < je) {
                            __mmask16 mask = (__mmask16)((1u << (je - v)) - 1);
                            const float* pp = pb + (v - j0);
                            __m512 acc = (k0 == 0) ? _mm512_setzero_ps()
                                : _mm512_maskz_loadu_ps(mask, C + (size_t)i * m + v);
                            for (int k = 0; k < kc; ++k)
                                acc = _mm512_fmadd_ps(_mm512_set1_ps(pai[k]),
                                    _mm512_loadu_ps(pp + (size_t)k * JBLK), acc);
                            fuse_nhwc_1(acc, _mm512_maskz_loadu_ps(mask, fb + v), fmin, fmax);
                            _mm512_mask_storeu_ps(C + (size_t)i * m + v, mask, acc);
                        }
                    }
                }
            }
            if (post_fn.kind == post_op_kind::none) {
                for (int jt = 0; jt < nj; jt++) {
                    int j0 = jt * JBLK;
                    int jw = std::min(JBLK, m - j0);
                    post_fn.apply_rows(i0, ie, C, m, j0, jw);
                }
            }
        }
    });
}

}} // namespace nnr::avx512
