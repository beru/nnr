#pragma once
// AVX2+FMA GEMM backend.
// 256-bit (6x8 micro-kernel) counterpart of avx512/gemm.h.
// Included from gemm_kernel.h — requires gemm_post_t to be defined before inclusion.

#include <immintrin.h>
#include <cfloat>
#include "thread_pool.h"
#include "backend/x64/vec_ops_avx2.h"
#include "backend/x64/gemm_ukernel_avx2.h"

namespace nnr {
// fused_zero_bias declared in gemm_avx512.h (included before this header).
namespace avx2 {

// Tail mask for AVX2: returns __m256i with lanes 0..n-1 set to -1, rest 0.
// @nnr-meta isa=AVX2 dtype=fp32
inline __m256i tail_mask_epi32(int n)
{
    alignas(32) static const int32_t table[16] = {
        -1, -1, -1, -1, -1, -1, -1, -1,
         0,  0,  0,  0,  0,  0,  0,  0
    };
    return _mm256_loadu_si256((const __m256i*)(table + 8 - n));
}

// Fused post-op helpers: bias + clamp on accumulators.
// Extracted from repeated epilogue code across all GEMM variants.

#if 0 // LOCUST
;def gen_fuse_nchw(nr):
;    args = ", ".join(f"__m256& c{r}" for r in range(nr))
inline void fuse_nchw_@nr@(@args@,
                        const float* bp, float fmin, float fmax) {
;    for r in range(nr):
    c@r@ = _mm256_add_ps(c@r@, _mm256_set1_ps(bp[@r@]));
;        pass
    __m256 vmin = _mm256_set1_ps(fmin);
    __m256 vmax = _mm256_set1_ps(fmax);
;    for r in range(nr):
    c@r@ = _mm256_max_ps(c@r@, vmin); c@r@ = _mm256_min_ps(c@r@, vmax);
;        pass
}
;    pass
;
;gen_fuse_nchw(6)
#else // LOCUST
// @nnr-meta isa=AVX2 dtype=fp32
inline void fuse_nchw_6(__m256& c0, __m256& c1, __m256& c2, __m256& c3, __m256& c4, __m256& c5,
                        const float* bp, float fmin, float fmax) {
    c0 = _mm256_add_ps(c0, _mm256_set1_ps(bp[0]));
    c1 = _mm256_add_ps(c1, _mm256_set1_ps(bp[1]));
    c2 = _mm256_add_ps(c2, _mm256_set1_ps(bp[2]));
    c3 = _mm256_add_ps(c3, _mm256_set1_ps(bp[3]));
    c4 = _mm256_add_ps(c4, _mm256_set1_ps(bp[4]));
    c5 = _mm256_add_ps(c5, _mm256_set1_ps(bp[5]));
    __m256 vmin = _mm256_set1_ps(fmin);
    __m256 vmax = _mm256_set1_ps(fmax);
    c0 = _mm256_max_ps(c0, vmin); c0 = _mm256_min_ps(c0, vmax);
    c1 = _mm256_max_ps(c1, vmin); c1 = _mm256_min_ps(c1, vmax);
    c2 = _mm256_max_ps(c2, vmin); c2 = _mm256_min_ps(c2, vmax);
    c3 = _mm256_max_ps(c3, vmin); c3 = _mm256_min_ps(c3, vmax);
    c4 = _mm256_max_ps(c4, vmin); c4 = _mm256_min_ps(c4, vmax);
    c5 = _mm256_max_ps(c5, vmin); c5 = _mm256_min_ps(c5, vmax);
}
#endif // LOCUST

// NCHW 1-row: single-row bias broadcast + clamp
// @nnr-meta isa=AVX2 dtype=fp32
inline void fuse_nchw_1(__m256& acc, float bias, float fmin, float fmax) {
    acc = _mm256_add_ps(acc, _mm256_set1_ps(bias));
    acc = _mm256_max_ps(acc, _mm256_set1_ps(fmin));
    acc = _mm256_min_ps(acc, _mm256_set1_ps(fmax));
}

// NCHW multi-vec array: bias + clamp over array of accumulators (small-M path)
// @nnr-meta isa=AVX2 dtype=fp32
inline void fuse_nchw_arr(__m256* acc, int nvec, float bias, float fmin, float fmax) {
    __m256 vb = _mm256_set1_ps(bias);
    __m256 vmin = _mm256_set1_ps(fmin);
    __m256 vmax = _mm256_set1_ps(fmax);
    for (int j = 0; j < nvec; j++) {
        acc[j] = _mm256_add_ps(acc[j], vb);
        acc[j] = _mm256_max_ps(acc[j], vmin);
        acc[j] = _mm256_min_ps(acc[j], vmax);
    }
}

// Scalar bias + clamp
// @nnr-meta isa=AVX2 dtype=fp32
inline void fuse_scalar(float& s, float bias, float fmin, float fmax) {
    s += bias;
    s = std::max(s, fmin);
    s = std::min(s, fmax);
}

#if 0 // LOCUST
;def gen_fuse_nhwc(nr):
;    args = ", ".join(f"__m256& c{r}" for r in range(nr))
inline void fuse_nhwc_@nr@(@args@,
                        __m256 vb, float fmin, float fmax) {
;    for r in range(0, nr, 2):
;        if r + 1 < nr:
    c@r@ = _mm256_add_ps(c@r@, vb); c@r+1@ = _mm256_add_ps(c@r+1@, vb);
;        else:
    c@r@ = _mm256_add_ps(c@r@, vb);
;        pass
;    pass
    __m256 vmin = _mm256_set1_ps(fmin);
    __m256 vmax = _mm256_set1_ps(fmax);
;    for r in range(nr):
    c@r@ = _mm256_max_ps(c@r@, vmin); c@r@ = _mm256_min_ps(c@r@, vmax);
;        pass
}
;    pass
;
;gen_fuse_nhwc(6)
#else // LOCUST
// @nnr-meta isa=AVX2 dtype=fp32
inline void fuse_nhwc_6(__m256& c0, __m256& c1, __m256& c2, __m256& c3, __m256& c4, __m256& c5,
                        __m256 vb, float fmin, float fmax) {
    c0 = _mm256_add_ps(c0, vb); c1 = _mm256_add_ps(c1, vb);
    c2 = _mm256_add_ps(c2, vb); c3 = _mm256_add_ps(c3, vb);
    c4 = _mm256_add_ps(c4, vb); c5 = _mm256_add_ps(c5, vb);
    __m256 vmin = _mm256_set1_ps(fmin);
    __m256 vmax = _mm256_set1_ps(fmax);
    c0 = _mm256_max_ps(c0, vmin); c0 = _mm256_min_ps(c0, vmax);
    c1 = _mm256_max_ps(c1, vmin); c1 = _mm256_min_ps(c1, vmax);
    c2 = _mm256_max_ps(c2, vmin); c2 = _mm256_min_ps(c2, vmax);
    c3 = _mm256_max_ps(c3, vmin); c3 = _mm256_min_ps(c3, vmax);
    c4 = _mm256_max_ps(c4, vmin); c4 = _mm256_min_ps(c4, vmax);
    c5 = _mm256_max_ps(c5, vmin); c5 = _mm256_min_ps(c5, vmax);
}
#endif // LOCUST

// NHWC 1-row: add pre-loaded bias vector + clamp
// @nnr-meta isa=AVX2 dtype=fp32
inline void fuse_nhwc_1(__m256& acc, __m256 vb, float fmin, float fmax) {
    acc = _mm256_add_ps(acc, vb);
    acc = _mm256_max_ps(acc, _mm256_set1_ps(fmin));
    acc = _mm256_min_ps(acc, _mm256_set1_ps(fmax));
}

// AVX2 row copy: avoids CRT memcpy call + vzeroupper penalty.
// Copies n floats. Full-vector loop + masked tail for remainder.
// @nnr-meta isa=AVX2 dtype=fp32
inline void copy_row_avx2(float* __restrict dst, const float* __restrict src, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_loadu_ps(src + i));
    if (i < n) {
        __m256i mask = tail_mask_epi32(n - i);
        _mm256_maskstore_ps(dst + i, mask, _mm256_maskload_ps(src + i, mask));
    }
}

// Copy exactly JBLK (32) floats — no branch, 4 full ymm stores.
// @nnr-meta isa=AVX2 dtype=fp32
inline void copy_row_avx2_full(float* __restrict dst, const float* __restrict src) {
    _mm256_storeu_ps(dst,      _mm256_loadu_ps(src));
    _mm256_storeu_ps(dst + 8,  _mm256_loadu_ps(src + 8));
    _mm256_storeu_ps(dst + 16, _mm256_loadu_ps(src + 16));
    _mm256_storeu_ps(dst + 24, _mm256_loadu_ps(src + 24));
}

// 2D B-panel copy: kc rows of jw floats from src (stride src_stride) into
// dst (stride JBLK). Branch on full vs partial row is hoisted outside the loop.
// @nnr-meta isa=AVX2 dtype=fp32
NNR_NOINLINE inline void pack_b_panel_avx2(float* __restrict dst, const float* __restrict src,
                              int kc, int jw, int src_stride) {
    constexpr int JBLK = 32;
    if (jw == JBLK) {
        for (int k = 0; k < kc; ++k)
            copy_row_avx2_full(dst + (size_t)k * JBLK, src + (size_t)k * src_stride);
    } else {
        for (int k = 0; k < kc; ++k)
            copy_row_avx2(dst + (size_t)k * JBLK, src + (size_t)k * src_stride, jw);
    }
}

#if 0 // LOCUST
;# -- Reusable Locust generators for AVX2 GEMM blocks -----------------------
;# All generators take indent level `d` (number of 4-space indents).
;# Use emit(d, text) which prepends d*4 spaces.
;
;def gen_pack_a_6(d, dst="pa_pack", src="pa"):
;    emit(d,   "for (int k = 0; k < kc; k++) {")
;    for r in range(0, 6, 2):
;        emit(d+1, f"{dst}[k * 6 + {r}] = {src}[{r}][k]; {dst}[k * 6 + {r+1}] = {src}[{r+1}][k];")
;    emit(d,   "}")
;
;def gen_pa_decls(d, nr, base="i", stride="o", offset="k0", src="A"):
;    suf = f" + {offset}" if offset and offset != "0" else ""
;    for r in range(nr):
;        emit(d, f"const float* pa{r} = {src} + (size_t)({base}+{r}) * {stride}{suf};")
;
;def gen_nhwc_6row_fma(d, pa_src="direct", k_var="k0"):
;    idx = f"{k_var}+k" if pa_src == "direct" else "k"
;    emit(d,   "for (int k = 0; k < kc; k++) {")
;    emit(d+1,   "__m256 bv = _mm256_loadu_ps(pp + (size_t)k * JBLK);")
;    for r in range(6):
;        emit(d+1, f"c{r} = _mm256_fmadd_ps(_mm256_set1_ps(pa{r}[{idx}]), bv, c{r});")
;    emit(d,   "}")
;
;def gen_nhwc_6row_c_zero_or_load(d, masked=False, src="C"):
;    cs = " = ".join(f"c{r}" for r in range(6))
;    load_fn = "_mm256_maskload_ps(" if masked else "_mm256_loadu_ps("
;    emit(d,   "if (k0 == 0) {")
;    emit(d+1, f"{cs} = _mm256_setzero_ps();")
;    emit(d,   "} else {")
;    for r in range(6):
;        emit(d+1, f'c{r} = {load_fn}{src} + (size_t)(i+{r}) * m + v{", mask" if masked else ""});')
;    emit(d,   "}")
;
;def gen_nhwc_6row_store(d, masked=False, src="C"):
;    if masked:
;        for r in range(6):
;            emit(d, f"_mm256_maskstore_ps({src} + (size_t)(i+{r}) * m + v, mask, c{r});")
;    else:
;        for r in range(6):
;            emit(d, f"_mm256_storeu_ps({src} + (size_t)(i+{r}) * m + v, c{r});")
;
;def gen_nhwc_6row_vloop(d, fma_fn):
;    """Emit full + masked v-loop for 6 rows."""
;    cs = ", ".join(f"c{r}" for r in range(6))
;    for is_tail in (False, True):
;        if is_tail:
;            emit(d,   "if (v < je) {")
;            emit(d+1,   "__m256i mask = tail_mask_epi32(je - v);")
;        else:
;            emit(d,   "for (; v + 8 <= je; v += 8) {")
;        emit(d+1, f"const float* pp = pb + (v - j0);")
;        emit(d+1, f"__m256 {cs};")
;        gen_nhwc_6row_c_zero_or_load(d+1, masked=is_tail)
;        fma_fn(d+1)
;        fb_load = "_mm256_maskload_ps(fb + v, mask)" if is_tail else "_mm256_loadu_ps(fb + v)"
;        emit(d+1, f"fuse_nhwc_6({cs}, {fb_load}, fmin, fmax);")
;        gen_nhwc_6row_store(d+1, masked=is_tail)
;        emit(d,   "}")
;
;def gen_nhwc_6row_block(d, offset="k0", src="A", fma_fn=None):
;    """Emit for (; i + 6 <= ie; i += 6) { pa_decls; v-loop } block."""
;    if fma_fn is None:
;        fma_fn = lambda dd: gen_nhwc_6row_fma(dd)
;    emit(d,   "for (; i + 6 <= ie; i += 6) {")
;    gen_pa_decls(d+1, 6, "i", "o", offset, src)
;    emit(d+1,   "int v = j0;")
;    gen_nhwc_6row_vloop(d+1, fma_fn)
;    emit(d,   "}")
;
;def gen_packed_fma(d):
;    """FMA loop using pa_pack (packed A) indexing."""
;    emit(d,   "for (int k = 0; k < kc; k++) {")
;    emit(d+1,   "__m256 bv = _mm256_loadu_ps(pp + (size_t)k * JBLK);")
;    emit(d+1,   "const float* ap = pa_pack + k * 6;")
;    for r in range(6):
;        emit(d+1, f"c{r} = _mm256_fmadd_ps(_mm256_set1_ps(ap[{r}]), bv, c{r});")
;    emit(d,   "}")
;
;def gen_packed_vloop(d):
;    """v-loop using pc[r] pointers and packed A."""
;    cs = ", ".join(f"c{r}" for r in range(6))
;    cs_eq = " = ".join(f"c{r}" for r in range(6))
;    for is_tail in (False, True):
;        if is_tail:
;            emit(d,   "if (v < je) {")
;            emit(d+1,   "__m256i mask = tail_mask_epi32(je - v);")
;        else:
;            emit(d,   "for (; v + 8 <= je; v += 8) {")
;        emit(d+1, f"const float* pp = pb + (v - j0);")
;        emit(d+1, f"__m256 {cs};")
;        emit(d+1, f"if (k0 == 0) {{ {cs_eq} = _mm256_setzero_ps(); }}")
;        emit(d+1,   "else {")
;        for r in range(0, 6, 2):
;            if is_tail:
;                emit(d+2, f"c{r} = _mm256_maskload_ps(pc[{r}] + v, mask); c{r+1} = _mm256_maskload_ps(pc[{r+1}] + v, mask);")
;            else:
;                emit(d+2, f"c{r} = _mm256_loadu_ps(pc[{r}] + v); c{r+1} = _mm256_loadu_ps(pc[{r+1}] + v);")
;        emit(d+1, "}")
;        gen_packed_fma(d+1)
;        fb = "_mm256_maskload_ps(fb + v, mask)" if is_tail else "_mm256_loadu_ps(fb + v)"
;        emit(d+1, f"fuse_nhwc_6({cs}, {fb}, fmin, fmax);")
;        for r in range(0, 6, 2):
;            if is_tail:
;                emit(d+1, f"_mm256_maskstore_ps(pc[{r}] + v, mask, c{r}); _mm256_maskstore_ps(pc[{r+1}] + v, mask, c{r+1});")
;            else:
;                emit(d+1, f"_mm256_storeu_ps(pc[{r}] + v, c{r}); _mm256_storeu_ps(pc[{r+1}] + v, c{r+1});")
;        emit(d,   "}")
#else // LOCUST
#endif // LOCUST

// AVX2 GEMM: C[n×m] = A[n×o] × B[o×m] with optional post-processing.
// Covers all float fast paths: GEMV M=1, GEMV N=1, small-K, small-M, tiled 6×8.
template <typename PostFn>
// @nnr-meta isa=AVX2 dtype=fp32 special=GEMM tiling=[K,MR,NR] fusion=post_op
inline void dgemm(int n, int m, int o, const float* __restrict A, const float* __restrict B, float* __restrict C, const PostFn& post_fn)
{
    constexpr bool can_fuse = PostFn::per_row_bias;
    // GEMV fast path: M=1 (row vector × matrix)
    if (n == 1) {
        int j;
        for (j = 0; j + 8 <= m; j += 8)
            _mm256_storeu_ps(C + j, _mm256_setzero_ps());
        for (; j < m; ++j)
            C[j] = 0.0f;
        for (int k = 0; k < o; ++k) {
            float a_k = A[k];
            const float* brow = B + (size_t)k * m;
            __m256 va = _mm256_set1_ps(a_k);
            j = 0;
            for (; j + 8 <= m; j += 8)
                _mm256_storeu_ps(C + j,
                    _mm256_fmadd_ps(va, _mm256_loadu_ps(brow + j),
                        _mm256_loadu_ps(C + j)));
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
            #if 0 // LOCUST
;            N = 4  # number of accumulators
;            for i in range(N):
            __m256 acc@i@ = _mm256_setzero_ps();
;                pass
            int k = 0;
            for (; k + @N*8@ <= o; k += @N*8@) {
;            for i in range(N):
;                off = f" + {i*8}" if i else ""
                acc@i@ = _mm256_fmadd_ps(_mm256_loadu_ps(pa_row + k@off@), _mm256_loadu_ps(B + k@off@), acc@i@);
;                pass
            }
;            # tree reduction
;            step = 1
;            while step < N:
;                for i in range(0, N, step * 2):
            acc@i@ = _mm256_add_ps(acc@i@, acc@i+step@);
;                    pass
;                step *= 2
#else // LOCUST
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            int k = 0;
            for (; k + 32 <= o; k += 32) {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa_row + k), _mm256_loadu_ps(B + k), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa_row + k + 8), _mm256_loadu_ps(B + k + 8), acc1);
                acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(pa_row + k + 16), _mm256_loadu_ps(B + k + 16), acc2);
                acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(pa_row + k + 24), _mm256_loadu_ps(B + k + 24), acc3);
            }
            acc0 = _mm256_add_ps(acc0, acc1);
            acc2 = _mm256_add_ps(acc2, acc3);
            acc0 = _mm256_add_ps(acc0, acc2);
            #endif // LOCUST
            for (; k + 8 <= o; k += 8)
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa_row + k),
                    _mm256_loadu_ps(B + k), acc0);
            float s = avx2::hsum(acc0);
            for (; k < o; ++k)
                s += pa_row[k] * B[k];
            C[i] = s;
            post_fn.apply(i, C + i, 1);
        });
        return;
    }

    // Small-K path: A[N,K] is small (fits in L1), B[K,M] streams through L2.
    // 6-row register blocking gives 6 independent FMA chains (> 4-cycle latency).
    if (o <= 48 && m >= 64) {
        int mchunks = (m + 7) / 8;
        nnr::for_static(0, mchunks, mchunks >= 32 && (int64_t)n * m * o > (1 << 22), [&](int jc) {
            int j = jc * 8;
            __m256i mask = (j + 8 <= m) ? _mm256_set1_epi32(-1)
                : tail_mask_epi32(m - j);
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
                __m256 c@r@ = _mm256_setzero_ps();
;                pass
                const float* pa0 = A + (size_t)i * o;
;            for r in range(1, MR):
                const float* pa@r@ = pa@r-1@ + o;
;                pass
                for (int k = 0; k < o; k++) {
                    __m256 bv = _mm256_maskload_ps(B + (size_t)k * m + j, mask);
;            for r in range(MR):
                    c@r@ = _mm256_fmadd_ps(_mm256_set1_ps(pa@r@[k]), bv, c@r@);
;                pass
                }
                if constexpr (can_fuse) {
;            cs = ", ".join(f"c{r}" for r in range(MR))
                    const float* bp = (post_fn.kind != post_op_kind::none && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                    fuse_nchw_@MR@(@cs@, bp, fmin, fmax);
                }
;            for r in range(MR):
                _mm256_maskstore_ps(C + (size_t)(i+@r@) * m + j, mask, c@r@);
;                pass
                if (!(can_fuse && post_fn.kind != post_op_kind::none)) {
                    int plen = (j + 8 <= m) ? 8 : (m - j);
                    post_fn.apply_rows(i, i + @MR@, C, m, j, plen);
                }
#else // LOCUST
            for (; i + 6 <= n; i += 6) {
                __m256 c0 = _mm256_setzero_ps();
                __m256 c1 = _mm256_setzero_ps();
                __m256 c2 = _mm256_setzero_ps();
                __m256 c3 = _mm256_setzero_ps();
                __m256 c4 = _mm256_setzero_ps();
                __m256 c5 = _mm256_setzero_ps();
                const float* pa0 = A + (size_t)i * o;
                const float* pa1 = pa0 + o;
                const float* pa2 = pa1 + o;
                const float* pa3 = pa2 + o;
                const float* pa4 = pa3 + o;
                const float* pa5 = pa4 + o;
                for (int k = 0; k < o; k++) {
                    __m256 bv = _mm256_maskload_ps(B + (size_t)k * m + j, mask);
                    c0 = _mm256_fmadd_ps(_mm256_set1_ps(pa0[k]), bv, c0);
                    c1 = _mm256_fmadd_ps(_mm256_set1_ps(pa1[k]), bv, c1);
                    c2 = _mm256_fmadd_ps(_mm256_set1_ps(pa2[k]), bv, c2);
                    c3 = _mm256_fmadd_ps(_mm256_set1_ps(pa3[k]), bv, c3);
                    c4 = _mm256_fmadd_ps(_mm256_set1_ps(pa4[k]), bv, c4);
                    c5 = _mm256_fmadd_ps(_mm256_set1_ps(pa5[k]), bv, c5);
                }
                if constexpr (can_fuse) {
                    const float* bp = (post_fn.kind != post_op_kind::none && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                    fuse_nchw_6(c0, c1, c2, c3, c4, c5, bp, fmin, fmax);
                }
                _mm256_maskstore_ps(C + (size_t)(i+0) * m + j, mask, c0);
                _mm256_maskstore_ps(C + (size_t)(i+1) * m + j, mask, c1);
                _mm256_maskstore_ps(C + (size_t)(i+2) * m + j, mask, c2);
                _mm256_maskstore_ps(C + (size_t)(i+3) * m + j, mask, c3);
                _mm256_maskstore_ps(C + (size_t)(i+4) * m + j, mask, c4);
                _mm256_maskstore_ps(C + (size_t)(i+5) * m + j, mask, c5);
                if (!(can_fuse && post_fn.kind != post_op_kind::none)) {
                    int plen = (j + 8 <= m) ? 8 : (m - j);
                    post_fn.apply_rows(i, i + 6, C, m, j, plen);
                }
            #endif // LOCUST
            }
            for (; i < n; i++) {
                __m256 acc = _mm256_setzero_ps();
                const float* pa = A + (size_t)i * o;
                for (int k = 0; k < o; k++)
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(pa[k]),
                        _mm256_maskload_ps(B + (size_t)k * m + j, mask), acc);
                if constexpr (can_fuse) {
                    const float* bp = (post_fn.kind != post_op_kind::none && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                    fuse_nchw_1(acc, bp[0], fmin, fmax);
                }
                _mm256_maskstore_ps(C + (size_t)i * m + j, mask, acc);
                if (!(can_fuse && post_fn.kind != post_op_kind::none))
                    post_fn.apply(i, C + (size_t)i * m + j, (j + 8 <= m) ? 8 : (m - j));
            }
        });
        return;
    }

    // Small-M path: 6-row register blocking shares B loads across rows.
    if (m < 64) {
        __m256i tail_mask_v = (m & 7) ? tail_mask_epi32(m & 7) : _mm256_set1_epi32(-1);
        int mfull = m / 8, mtail = m & 7;
        int ngroups = (n + 5) / 6;
        nnr::for_static(0, ngroups, ngroups > 8 && (int64_t)n * m * o > (1 << 18), [&](int ig) {
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
            if (nr == @MR@) {
;            for r in range(MR):
                const float* pa@r@ = A + (size_t)(i0+@r@) * o;
;                pass
;            arrs = ", ".join(f"c{r}[8]={{}}" for r in range(MR))
                __m256 @arrs@;
                for (int k = 0; k < o; k++) {
                    const float* br = B + (size_t)k * m;
;            for r in range(MR):
                    __m256 a@r@ = _mm256_set1_ps(pa@r@[k]);
;                pass
                    for (int j = 0; j < mfull; j++) {
                        __m256 bv = _mm256_loadu_ps(br + j * 8);
;            for r in range(MR):
                        c@r@[j] = _mm256_fmadd_ps(a@r@, bv, c@r@[j]);
;                pass
                    }
                    if (mtail) {
                        __m256 bv = _mm256_maskload_ps(br + mfull * 8, tail_mask_v);
;            for r in range(MR):
                        c@r@[mfull] = _mm256_fmadd_ps(a@r@, bv, c@r@[mfull]);
;                pass
                    }
                }
                if constexpr (can_fuse) {
                    const float* bp = (post_fn.kind != post_op_kind::none && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i0 : fused_zero_bias;
;            rows = ", ".join(f"c{r}" for r in range(MR))
                    __m256* rows[] = {@rows@};
                    int nvec = mtail ? mfull + 1 : mfull;
                    for (int r = 0; r < @MR@; r++)
                        fuse_nchw_arr(rows[r], nvec, bp[r], fmin, fmax);
                }
;            # pc pointers, 2 per line
;            for r in range(0, MR, 2):
;                if r + 1 < MR:
                float *pc@r@ = C + (size_t)(i0+@r@) * m, *pc@r+1@ = C + (size_t)(i0+@r+1@) * m;
;                else:
                float *pc@r@ = C + (size_t)(i0+@r@) * m;
;                pass
;            pass
                for (int j = 0; j < mfull; j++) {
;            for r in range(0, MR, 2):
;                if r + 1 < MR:
                    _mm256_storeu_ps(pc@r@ + j*8, c@r@[j]); _mm256_storeu_ps(pc@r+1@ + j*8, c@r+1@[j]);
;                else:
                    _mm256_storeu_ps(pc@r@ + j*8, c@r@[j]);
;                pass
;            pass
                }
                if (mtail) {
;            for r in range(MR):
                    _mm256_maskstore_ps(pc@r@ + mfull*8, tail_mask_v, c@r@[mfull]);
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
                __m256 c0[8]={}, c1[8]={}, c2[8]={}, c3[8]={}, c4[8]={}, c5[8]={};
                for (int k = 0; k < o; k++) {
                    const float* br = B + (size_t)k * m;
                    __m256 a0 = _mm256_set1_ps(pa0[k]);
                    __m256 a1 = _mm256_set1_ps(pa1[k]);
                    __m256 a2 = _mm256_set1_ps(pa2[k]);
                    __m256 a3 = _mm256_set1_ps(pa3[k]);
                    __m256 a4 = _mm256_set1_ps(pa4[k]);
                    __m256 a5 = _mm256_set1_ps(pa5[k]);
                    for (int j = 0; j < mfull; j++) {
                        __m256 bv = _mm256_loadu_ps(br + j * 8);
                        c0[j] = _mm256_fmadd_ps(a0, bv, c0[j]);
                        c1[j] = _mm256_fmadd_ps(a1, bv, c1[j]);
                        c2[j] = _mm256_fmadd_ps(a2, bv, c2[j]);
                        c3[j] = _mm256_fmadd_ps(a3, bv, c3[j]);
                        c4[j] = _mm256_fmadd_ps(a4, bv, c4[j]);
                        c5[j] = _mm256_fmadd_ps(a5, bv, c5[j]);
                    }
                    if (mtail) {
                        __m256 bv = _mm256_maskload_ps(br + mfull * 8, tail_mask_v);
                        c0[mfull] = _mm256_fmadd_ps(a0, bv, c0[mfull]);
                        c1[mfull] = _mm256_fmadd_ps(a1, bv, c1[mfull]);
                        c2[mfull] = _mm256_fmadd_ps(a2, bv, c2[mfull]);
                        c3[mfull] = _mm256_fmadd_ps(a3, bv, c3[mfull]);
                        c4[mfull] = _mm256_fmadd_ps(a4, bv, c4[mfull]);
                        c5[mfull] = _mm256_fmadd_ps(a5, bv, c5[mfull]);
                    }
                }
                if constexpr (can_fuse) {
                    const float* bp = (post_fn.kind != post_op_kind::none && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i0 : fused_zero_bias;
                    __m256* rows[] = {c0, c1, c2, c3, c4, c5};
                    int nvec = mtail ? mfull + 1 : mfull;
                    for (int r = 0; r < 6; r++)
                        fuse_nchw_arr(rows[r], nvec, bp[r], fmin, fmax);
                }
                float *pc0 = C + (size_t)(i0+0) * m, *pc1 = C + (size_t)(i0+1) * m;
                float *pc2 = C + (size_t)(i0+2) * m, *pc3 = C + (size_t)(i0+3) * m;
                float *pc4 = C + (size_t)(i0+4) * m, *pc5 = C + (size_t)(i0+5) * m;
                for (int j = 0; j < mfull; j++) {
                    _mm256_storeu_ps(pc0 + j*8, c0[j]); _mm256_storeu_ps(pc1 + j*8, c1[j]);
                    _mm256_storeu_ps(pc2 + j*8, c2[j]); _mm256_storeu_ps(pc3 + j*8, c3[j]);
                    _mm256_storeu_ps(pc4 + j*8, c4[j]); _mm256_storeu_ps(pc5 + j*8, c5[j]);
                }
                if (mtail) {
                    _mm256_maskstore_ps(pc0 + mfull*8, tail_mask_v, c0[mfull]);
                    _mm256_maskstore_ps(pc1 + mfull*8, tail_mask_v, c1[mfull]);
                    _mm256_maskstore_ps(pc2 + mfull*8, tail_mask_v, c2[mfull]);
                    _mm256_maskstore_ps(pc3 + mfull*8, tail_mask_v, c3[mfull]);
                    _mm256_maskstore_ps(pc4 + mfull*8, tail_mask_v, c4[mfull]);
                    _mm256_maskstore_ps(pc5 + mfull*8, tail_mask_v, c5[mfull]);
                }
                if (!(can_fuse && post_fn.kind != post_op_kind::none))
                    post_fn.apply_rows(i0, i0 + 6, C, m, 0, m);
            #endif // LOCUST
            } else {
                // Remainder: 1-5 rows
                for (int i = i0; i < ie; i++) {
                    float* pc = C + (size_t)i * m;
                    const float* pa = A + (size_t)i * o;
                    __m256 acc[8] = {};
                    for (int k = 0; k < o; k++) {
                        const float* br = B + (size_t)k * m;
                        __m256 av = _mm256_set1_ps(pa[k]);
                        for (int j = 0; j < mfull; j++)
                            acc[j] = _mm256_fmadd_ps(av, _mm256_loadu_ps(br + j*8), acc[j]);
                        if (mtail)
                            acc[mfull] = _mm256_fmadd_ps(av,
                                _mm256_maskload_ps(br + mfull*8, tail_mask_v), acc[mfull]);
                    }
                    if constexpr (can_fuse) {
                        const float* bp = (post_fn.kind != post_op_kind::none && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                        int nvec = mtail ? mfull + 1 : mfull;
                        fuse_nchw_arr(acc, nvec, bp[0], fmin, fmax);
                    }
                    for (int j = 0; j < mfull; j++)
                        _mm256_storeu_ps(pc + j*8, acc[j]);
                    if (mtail)
                        _mm256_maskstore_ps(pc + mfull*8, tail_mask_v, acc[mfull]);
                    if (!(can_fuse && post_fn.kind != post_op_kind::none))
                        post_fn.apply(i, pc, m);
                }
            }
        });
        return;
    }

    // Tiled path: 6×8 micro-kernel with A-packing, B-packing, K-blocking
    constexpr int JBLK = 64;
    constexpr int KC = 256;
    int ni = (n + JBLK - 1) / JBLK;
    int nj = (m + JBLK - 1) / JBLK;
    int ntiles = ni * nj;
    int nt = ((int64_t)n * m * o > (1 << 21)) ? nnr::compute_threads(ntiles) : 1;

    NNR_POOL_ENSURE_SCRATCH(((size_t)KC * JBLK + (size_t)KC * 6) * sizeof(float));

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

            float fmin = -FLT_MAX;
            float fmax = FLT_MAX;
            bool fuse_nchw = false;
            if constexpr (can_fuse) { if (last_k && post_fn.kind != post_op_kind::none) {
                fuse_nchw = true; fmin = post_fn.clip_min; fmax = post_fn.clip_max;
            }}

            // Pack B sub-panel: pb[kc × JBLK]
            pack_b_panel_avx2(pb, B + (size_t)k0 * m + j0, kc, jw, m);

            // 6-row groups
            int i = i0;
            for (; i + 6 <= ie; i += 6) {
                const float* pa[6];
                float* pc[6];
                for (int r = 0; r < 6; r++) {
                    pa[r] = A + (size_t)(i + r) * o + k0;
                    pc[r] = C + (size_t)(i + r) * m;
                }

                // Pack A: 6 rows × kc cols → pa_pack[kc × 6] (interleaved)
                #if 0 // LOCUST
;                gen_pack_a_6(4)
#else // LOCUST
                for (int k = 0; k < kc; k++) {
                    pa_pack[k * 6 + 0] = pa[0][k]; pa_pack[k * 6 + 1] = pa[1][k];
                    pa_pack[k * 6 + 2] = pa[2][k]; pa_pack[k * 6 + 3] = pa[3][k];
                    pa_pack[k * 6 + 4] = pa[4][k]; pa_pack[k * 6 + 5] = pa[5][k];
                }
                #endif // LOCUST

                int v = j0;
                for (; v + 8 <= je; v += 8) {
                    const float* pp = pb + (v - j0);
                    const float* bp_uk = fused_zero_bias;
                    if constexpr (can_fuse) { if (fuse_nchw && post_fn.bias) bp_uk = post_fn.bias + post_fn.bias_off + i; }
                    ukernel_nchw(kc, pa_pack, pp, JBLK, pc, v,
                        k0 == 0, fuse_nchw, bp_uk, fmin, fmax);
                }
                // Scalar remainder columns
                for (; v < je; ++v) {
                    const float* pp = pb + (v - j0);
                    for (int r = 0; r < 6; r++) {
                        float s = (k0 == 0) ? 0.0f : pc[r][v];
                        for (int k = 0; k < kc; ++k)
                            s += pa_pack[k * 6 + r] * pp[(size_t)k * JBLK];
                        if constexpr (can_fuse) {
                            const float* bp = (fuse_nchw && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                            fuse_scalar(s, bp[r], fmin, fmax);
                        }
                        pc[r][v] = s;
                    }
                }
            }
            // Remainder rows (< 6): 1-row kernel
            for (; i < ie; i++) {
                float* pci = C + (size_t)i * m;
                const float* pai = A + (size_t)i * o + k0;
                int v = j0;
                for (; v + 8 <= je; v += 8) {
                    const float* pp = pb + (v - j0);
                    __m256 acc = (k0 == 0) ? _mm256_setzero_ps() : _mm256_loadu_ps(pci + v);
                    for (int k = 0; k < kc; ++k)
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(pai[k]),
                            _mm256_loadu_ps(pp + (size_t)k * JBLK), acc);
                    if constexpr (can_fuse) {
                        const float* bp = (fuse_nchw && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                        fuse_nchw_1(acc, bp[0], fmin, fmax);
                    }
                    _mm256_storeu_ps(pci + v, acc);
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

// Pre-pack B matrix into panel format for dgemm_packed_b().
// @nnr-meta isa=AVX2 dtype=fp32
inline size_t pack_b_size(int o, int m) {
    constexpr int JBLK = 64, KC = 256;
    int nj = (m + JBLK - 1) / JBLK;
    int nk = (o + KC - 1) / KC;
    return (size_t)nj * nk * KC * JBLK;
}

// @nnr-meta isa=AVX2 dtype=fp32
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
            pack_b_panel_avx2(panel, B + (size_t)k0 * m + j0, kc, jw, m);
        }
    }
}

// GEMM with pre-packed B: C[n×m] = A[n×o] × packed_B
template <typename PostFn>
// @nnr-meta isa=AVX2 dtype=fp32 special=GEMM tiling=[K,MR,NR] fusion=post_op
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
    int nt = ((int64_t)n * m * o > (1 << 21)) ? nnr::compute_threads(ntiles) : 1;

    NNR_POOL_ENSURE_SCRATCH((size_t)KC * 6 * sizeof(float));

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
            for (; i + 6 <= ie; i += 6) {
                const float* pa[6];
                float* pc[6];
                for (int r = 0; r < 6; r++) {
                    pa[r] = A + (size_t)(i + r) * o + k0;
                    pc[r] = C + (size_t)(i + r) * m;
                }
                #if 0 // LOCUST
;                gen_pack_a_6(4)
#else // LOCUST
                for (int k = 0; k < kc; k++) {
                    pa_pack[k * 6 + 0] = pa[0][k]; pa_pack[k * 6 + 1] = pa[1][k];
                    pa_pack[k * 6 + 2] = pa[2][k]; pa_pack[k * 6 + 3] = pa[3][k];
                    pa_pack[k * 6 + 4] = pa[4][k]; pa_pack[k * 6 + 5] = pa[5][k];
                }
                #endif // LOCUST
                int v = j0;
                for (; v + 8 <= je; v += 8) {
                    const float* pp = pb + (v - j0);
                    const float* bp_uk = fused_zero_bias;
                    if constexpr (can_fuse) { if (fuse_nchw && post_fn.bias) bp_uk = post_fn.bias + post_fn.bias_off + i; }
                    ukernel_nchw(kc, pa_pack, pp, JBLK, pc, v,
                        k0 == 0, fuse_nchw, bp_uk, fmin, fmax);
                }
                for (; v < je; ++v) {
                    const float* pp = pb + (v - j0);
                    for (int r = 0; r < 6; r++) {
                        float s = (k0 == 0) ? 0.0f : pc[r][v];
                        for (int k = 0; k < kc; ++k)
                            s += pa_pack[k * 6 + r] * pp[(size_t)k * JBLK];
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
                for (; v + 8 <= je; v += 8) {
                    const float* pp = pb + (v - j0);
                    __m256 acc = (k0 == 0) ? _mm256_setzero_ps() : _mm256_loadu_ps(pci + v);
                    for (int k = 0; k < kc; ++k)
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(pai[k]),
                            _mm256_loadu_ps(pp + (size_t)k * JBLK), acc);
                    if constexpr (can_fuse) {
                        const float* bp = (fuse_nchw && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                        fuse_nchw_1(acc, bp[0], fmin, fmax);
                    }
                    _mm256_storeu_ps(pci + v, acc);
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

// Pre-pack A matrix into panel format for dgemm_packed_a().
// @nnr-meta isa=AVX2 dtype=fp32
inline size_t pack_a_size(int n, int o) {
    constexpr int JBLK = 64, KC = 256;
    int ni = (n + JBLK - 1) / JBLK;
    int nk = (o + KC - 1) / KC;
    return (size_t)ni * nk * JBLK * KC;
}

// @nnr-meta isa=AVX2 dtype=fp32
inline void pack_a(float* __restrict dst, const float* __restrict A, int n, int o) {
    constexpr int JBLK = 64, KC = 256;
    int ni = (n + JBLK - 1) / JBLK;
    int nk = (o + KC - 1) / KC;
    memset(dst, 0, (size_t)ni * nk * JBLK * KC * sizeof(float));
    for (int ib = 0; ib < ni; ib++) {
        int i0 = ib * JBLK;
        int iw = std::min(JBLK, n - i0);
        int nfull = iw / 6;
        int nrem = iw % 6;
        for (int kb = 0; kb < nk; kb++) {
            int k0 = kb * KC;
            int kc = std::min(KC, o - k0);
            float* panel = dst + ((size_t)ib * nk + kb) * JBLK * KC;
            for (int g = 0; g < nfull; g++) {
                float* grp = panel + (size_t)g * 6 * KC;
                const float* pa[6];
                for (int r = 0; r < 6; r++)
                    pa[r] = A + (size_t)(i0 + g * 6 + r) * o + k0;
                #if 0 // LOCUST
;                gen_pack_a_6(4, "grp")
#else // LOCUST
                for (int k = 0; k < kc; k++) {
                    grp[k * 6 + 0] = pa[0][k]; grp[k * 6 + 1] = pa[1][k];
                    grp[k * 6 + 2] = pa[2][k]; grp[k * 6 + 3] = pa[3][k];
                    grp[k * 6 + 4] = pa[4][k]; grp[k * 6 + 5] = pa[5][k];
                }
                #endif // LOCUST
            }
            for (int r = 0; r < nrem; r++) {
                float* rem = panel + (size_t)nfull * 6 * KC + (size_t)r * KC;
                copy_row_avx2(rem, A + (size_t)(i0 + nfull * 6 + r) * o + k0, kc);
            }
        }
    }
}

// GEMM with pre-packed A: C[n×m] = packed_A × B[o×m]
template <typename PostFn>
// @nnr-meta isa=AVX2 dtype=fp32 special=GEMM tiling=[K,MR,NR] fusion=post_op
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
    float* shared_pb = nullptr;
    const bool skip_pack = (nj == 1 && m == JBLK);
    const bool prepack_b = (nj == 1 && m < JBLK);
    if (prepack_b) {
        size_t total = (size_t)nk * KC * JBLK;
        shared_pb = (float*)nnr_aligned_alloc(total * sizeof(float), 64);
        if (shared_pb) {
            int jw = m;
            for (int kt = 0; kt < nk; kt++) {
                int k0 = kt * KC;
                int kc = std::min(KC, o - k0);
                float* dst = shared_pb + (size_t)kt * KC * JBLK;
                memset(dst, 0, (size_t)KC * JBLK * sizeof(float));
                pack_b_panel_avx2(dst, B + (size_t)k0 * m, kc, jw, m);
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
        int nfull = iw / 6;

        for (int kt = 0; kt < nk; kt++) {
            int k0 = kt * KC;
            int kc = std::min(KC, o - k0);
            bool last_k = (k0 + kc == o);

            const float* b_panel;
            if (skip_pack)
                b_panel = B + (size_t)k0 * m + j0;
            else if (prepack_b && shared_pb)
                b_panel = shared_pb + (size_t)kt * KC * JBLK;
            else {
                pack_b_panel_avx2(pb, B + (size_t)k0 * m + j0, kc, jw, m);
                b_panel = pb;
            }

            const float* a_panel = packed_A + ((size_t)ib * nk + kt) * JBLK * KC;

            float fmin = -FLT_MAX;
            float fmax = FLT_MAX;
            bool fuse_nchw = false;
            if constexpr (can_fuse) { if (last_k && post_fn.kind != post_op_kind::none) {
                fuse_nchw = true; fmin = post_fn.clip_min; fmax = post_fn.clip_max;
            }}

            int i = i0;
            int grp = 0;
            for (; i + 6 <= ie; i += 6, grp++) {
                float* pc[6];
                for (int r = 0; r < 6; r++)
                    pc[r] = C + (size_t)(i + r) * m;

                const float* pa_pack = a_panel + (size_t)grp * 6 * KC;

                int v = j0;
                for (; v + 8 <= je; v += 8) {
                    const float* pp = b_panel + (v - j0);
                    const float* bp_uk = fused_zero_bias;
                    if constexpr (can_fuse) { if (fuse_nchw && post_fn.bias) bp_uk = post_fn.bias + post_fn.bias_off + i; }
                    ukernel_nchw(kc, pa_pack, pp, JBLK, pc, v,
                        k0 == 0, fuse_nchw, bp_uk, fmin, fmax);
                }
                if (v < je) {
                    __m256i vmask = tail_mask_epi32(je - v);
                    const float* pp = b_panel + (v - j0);
                    __m256 c0, c1, c2, c3, c4, c5;
                    if (k0 == 0) {
                        c0 = c1 = c2 = c3 = c4 = c5 = _mm256_setzero_ps();
                    } else {
                        c0 = _mm256_maskload_ps(pc[0] + v, vmask);
                        c1 = _mm256_maskload_ps(pc[1] + v, vmask);
                        c2 = _mm256_maskload_ps(pc[2] + v, vmask);
                        c3 = _mm256_maskload_ps(pc[3] + v, vmask);
                        c4 = _mm256_maskload_ps(pc[4] + v, vmask);
                        c5 = _mm256_maskload_ps(pc[5] + v, vmask);
                    }
                    for (int k = 0; k < kc; k++) {
                        __m256 bv = _mm256_loadu_ps(pp + (size_t)k * JBLK);
                        const float* ap = pa_pack + k * 6;
                        c0 = _mm256_fmadd_ps(_mm256_set1_ps(ap[0]), bv, c0);
                        c1 = _mm256_fmadd_ps(_mm256_set1_ps(ap[1]), bv, c1);
                        c2 = _mm256_fmadd_ps(_mm256_set1_ps(ap[2]), bv, c2);
                        c3 = _mm256_fmadd_ps(_mm256_set1_ps(ap[3]), bv, c3);
                        c4 = _mm256_fmadd_ps(_mm256_set1_ps(ap[4]), bv, c4);
                        c5 = _mm256_fmadd_ps(_mm256_set1_ps(ap[5]), bv, c5);
                    }
                    if constexpr (can_fuse) { if (fuse_nchw) {
                        const float* bp = post_fn.bias ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
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
                    }}
                    _mm256_maskstore_ps(pc[0] + v, vmask, c0);
                    _mm256_maskstore_ps(pc[1] + v, vmask, c1);
                    _mm256_maskstore_ps(pc[2] + v, vmask, c2);
                    _mm256_maskstore_ps(pc[3] + v, vmask, c3);
                    _mm256_maskstore_ps(pc[4] + v, vmask, c4);
                    _mm256_maskstore_ps(pc[5] + v, vmask, c5);
                }
            }
            for (int rem = 0; i < ie; i++, rem++) {
                float* pci = C + (size_t)i * m;
                const float* pai = a_panel + (size_t)nfull * 6 * KC + (size_t)rem * KC;
                int v = j0;
                for (; v + 8 <= je; v += 8) {
                    const float* pp = b_panel + (v - j0);
                    __m256 acc = (k0 == 0) ? _mm256_setzero_ps() : _mm256_loadu_ps(pci + v);
                    for (int k = 0; k < kc; ++k)
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(pai[k]),
                            _mm256_loadu_ps(pp + (size_t)k * JBLK), acc);
                    if constexpr (can_fuse) {
                        const float* bp = (fuse_nchw && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                        fuse_nchw_1(acc, bp[0], fmin, fmax);
                    }
                    _mm256_storeu_ps(pci + v, acc);
                }
                if (v < je) {
                    __m256i vmask = tail_mask_epi32(je - v);
                    const float* pp = b_panel + (v - j0);
                    __m256 acc = (k0 == 0) ? _mm256_setzero_ps() : _mm256_maskload_ps(pci + v, vmask);
                    for (int k = 0; k < kc; ++k)
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(pai[k]),
                            _mm256_loadu_ps(pp + (size_t)k * JBLK), acc);
                    if constexpr (can_fuse) {
                        const float* bp = (fuse_nchw && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                        fuse_nchw_1(acc, bp[0], fmin, fmax);
                    }
                    _mm256_maskstore_ps(pci + v, vmask, acc);
                }
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
template <typename PostFn>
// @nnr-meta isa=AVX2 dtype=fp32 special=GEMM tiling=[K,MR,NR] fusion=post_op
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
        int nfull = iw / 6;

        {
            const float* packed_A = packed_A_batch[p];
            const float* B = B_batch[p];
            float* C = C_batch[p];

            for (int kt = 0; kt < nk; kt++) {
                int k0 = kt * KC;
                int kc = std::min(KC, o - k0);
                bool last_k = (k0 + kc == o);

                pack_b_panel_avx2(pb, B + (size_t)k0 * m + j0, kc, jw, m);

                const float* a_panel = packed_A + ((size_t)ib * nk + kt) * JBLK * KC;

                float fmin = -FLT_MAX;
                float fmax = FLT_MAX;
                bool fuse_nchw = false;
                if constexpr (can_fuse) { if (last_k && post_fn.kind != post_op_kind::none) {
                    fuse_nchw = true; fmin = post_fn.clip_min; fmax = post_fn.clip_max;
                }}

                int i = i0;
                int grp = 0;
                for (; i + 6 <= ie; i += 6, grp++) {
                    float* pc[6];
                    for (int r = 0; r < 6; r++)
                        pc[r] = C + (size_t)(i + r) * m;

                    const float* pa_pack = a_panel + (size_t)grp * 6 * KC;

                    int v = j0;
                    for (; v + 8 <= je; v += 8) {
                        const float* pp = pb + (v - j0);
                        const float* bp_uk = fused_zero_bias;
                        if constexpr (can_fuse) { if (fuse_nchw && post_fn.bias) bp_uk = post_fn.bias + post_fn.bias_off + i; }
                        ukernel_nchw(kc, pa_pack, pp, JBLK, pc, v,
                            k0 == 0, fuse_nchw, bp_uk, fmin, fmax);
                    }
                    if (v < je) {
                        __m256i vmask = tail_mask_epi32(je - v);
                        const float* pp = pb + (v - j0);
                        __m256 c0, c1, c2, c3, c4, c5;
                        if (k0 == 0) {
                            c0 = c1 = c2 = c3 = c4 = c5 = _mm256_setzero_ps();
                        } else {
                            c0 = _mm256_maskload_ps(pc[0] + v, vmask);
                            c1 = _mm256_maskload_ps(pc[1] + v, vmask);
                            c2 = _mm256_maskload_ps(pc[2] + v, vmask);
                            c3 = _mm256_maskload_ps(pc[3] + v, vmask);
                            c4 = _mm256_maskload_ps(pc[4] + v, vmask);
                            c5 = _mm256_maskload_ps(pc[5] + v, vmask);
                        }
                        for (int k = 0; k < kc; k++) {
                            __m256 bv = _mm256_loadu_ps(pp + (size_t)k * JBLK);
                            const float* ap = pa_pack + k * 6;
                            c0 = _mm256_fmadd_ps(_mm256_set1_ps(ap[0]), bv, c0);
                            c1 = _mm256_fmadd_ps(_mm256_set1_ps(ap[1]), bv, c1);
                            c2 = _mm256_fmadd_ps(_mm256_set1_ps(ap[2]), bv, c2);
                            c3 = _mm256_fmadd_ps(_mm256_set1_ps(ap[3]), bv, c3);
                            c4 = _mm256_fmadd_ps(_mm256_set1_ps(ap[4]), bv, c4);
                            c5 = _mm256_fmadd_ps(_mm256_set1_ps(ap[5]), bv, c5);
                        }
                        if constexpr (can_fuse) { if (fuse_nchw) {
                            const float* bp = post_fn.bias ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
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
                        }}
                        _mm256_maskstore_ps(pc[0] + v, vmask, c0);
                        _mm256_maskstore_ps(pc[1] + v, vmask, c1);
                        _mm256_maskstore_ps(pc[2] + v, vmask, c2);
                        _mm256_maskstore_ps(pc[3] + v, vmask, c3);
                        _mm256_maskstore_ps(pc[4] + v, vmask, c4);
                        _mm256_maskstore_ps(pc[5] + v, vmask, c5);
                    }
                }
                for (int rem = 0; i < ie; i++, rem++) {
                    float* pci = C + (size_t)i * m;
                    const float* pai = a_panel + (size_t)nfull * 6 * KC + (size_t)rem * KC;
                    int v = j0;
                    for (; v + 8 <= je; v += 8) {
                        const float* pp = pb + (v - j0);
                        __m256 acc = (k0 == 0) ? _mm256_setzero_ps() : _mm256_loadu_ps(pci + v);
                        for (int k = 0; k < kc; ++k)
                            acc = _mm256_fmadd_ps(_mm256_set1_ps(pai[k]),
                                _mm256_loadu_ps(pp + (size_t)k * JBLK), acc);
                        if constexpr (can_fuse) {
                            const float* bp = (fuse_nchw && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                            fuse_nchw_1(acc, bp[0], fmin, fmax);
                        }
                        _mm256_storeu_ps(pci + v, acc);
                    }
                    if (v < je) {
                        __m256i vmask = tail_mask_epi32(je - v);
                        const float* pp = pb + (v - j0);
                        __m256 acc = (k0 == 0) ? _mm256_setzero_ps() : _mm256_maskload_ps(pci + v, vmask);
                        for (int k = 0; k < kc; ++k)
                            acc = _mm256_fmadd_ps(_mm256_set1_ps(pai[k]),
                                _mm256_loadu_ps(pp + (size_t)k * JBLK), acc);
                        if constexpr (can_fuse) {
                            const float* bp = (fuse_nchw && post_fn.bias) ? post_fn.bias + post_fn.bias_off + i : fused_zero_bias;
                            fuse_nchw_1(acc, bp[0], fmin, fmax);
                        }
                        _mm256_maskstore_ps(pci + v, vmask, acc);
                    }
                }
            }
            if (!(can_fuse && post_fn.kind != post_op_kind::none))
                post_fn.apply_rows(i0, ie, C, m, j0, jw);
        }
    });
}

// NHWC-native GEMM: C[n × m] = A[n × o] × packed_B[o × m]
// AVX2 variant of avx512::dgemm_nhwc — same loop structure, 8-wide vectors.
template <typename PostFn>
// @nnr-meta isa=AVX2 dtype=fp32 layout=NHWC special=GEMM tiling=[K,MR,NR] fusion=post_op
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

    // Large-M or large-K fallback: dgemm_packed_b (tiles both dimensions with
    // A-packing) gives better locality when M needs many j-blocks, or when K
    // needs many k-blocks with few i-blocks (strided A access dominates).
    if (nj > 2 || (nk > 4 && ni <= 4)) {
        avx2::dgemm_packed_b(n, m, o, A, packed_B, C, post_fn);
        return;
    }

    // C-hot path: when K spans many blocks, keep C in registers across k-blocks.
    // Loop order: i-chunks → j-blocks → 6-row groups → v-slices → k-blocks.
    if (nk > 2) {
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
;                def gen_chot_fma(d):
;                    emit(d,   "for (int kt = 0; kt < nk; kt++) {")
;                    emit(d+1,   "int k0 = kt * KC;")
;                    emit(d+1,   "int kc = std::min(KC, o - k0);")
;                    emit(d+1,   "const float* pp = packed_B + ((size_t)jt * nk + kt) * KC * JBLK + (v - j0);")
;                    emit(d+1,   "for (int k = 0; k < kc; k++) {")
;                    emit(d+2,     "__m256 bv = _mm256_loadu_ps(pp + (size_t)k * JBLK);")
;                    for r in range(6):
;                        emit(d+2, f"c{r} = _mm256_fmadd_ps(_mm256_set1_ps(pa{r}[k0 + k]), bv, c{r});")
;                    emit(d+1, "}")
;                    emit(d,   "}")
;
;                def gen_chot_vloop(d):
;                    cs = ", ".join(f"c{r}" for r in range(6))
;                    for is_tail in (False, True):
;                        if is_tail:
;                            emit(d,   "if (v < je) {")
;                            emit(d+1,   "__m256i mask = tail_mask_epi32(je - v);")
;                        else:
;                            emit(d,   "for (; v + 8 <= je; v += 8) {")
;                        for r in range(6):
;                            emit(d+1, f"__m256 c{r} = _mm256_setzero_ps();")
;                        gen_chot_fma(d+1)
;                        fb = "_mm256_maskload_ps(fb + v, mask)" if is_tail else "_mm256_loadu_ps(fb + v)"
;                        emit(d+1, f"fuse_nhwc_6({cs}, {fb}, fmin, fmax);")
;                        if is_tail:
;                            for r in range(6):
;                                emit(d+1, f"_mm256_maskstore_ps(C + (size_t)(i+{r}) * m + v, mask, c{r});")
;                        else:
;                            for r in range(6):
;                                emit(d+1, f"_mm256_storeu_ps(C + (size_t)(i+{r}) * m + v, c{r});")
;                        emit(d, "}")
;
                for (; i + 6 <= ie; i += 6) {
;                gen_pa_decls(5, 6, "i", "o", "0", "A")
                    int v = j0;
;                gen_chot_vloop(5)
                }
#else // LOCUST
                for (; i + 6 <= ie; i += 6) {
                    const float* pa0 = A + (size_t)(i+0) * o;
                    const float* pa1 = A + (size_t)(i+1) * o;
                    const float* pa2 = A + (size_t)(i+2) * o;
                    const float* pa3 = A + (size_t)(i+3) * o;
                    const float* pa4 = A + (size_t)(i+4) * o;
                    const float* pa5 = A + (size_t)(i+5) * o;
                    int v = j0;
                    for (; v + 8 <= je; v += 8) {
                        __m256 c0 = _mm256_setzero_ps();
                        __m256 c1 = _mm256_setzero_ps();
                        __m256 c2 = _mm256_setzero_ps();
                        __m256 c3 = _mm256_setzero_ps();
                        __m256 c4 = _mm256_setzero_ps();
                        __m256 c5 = _mm256_setzero_ps();
                        for (int kt = 0; kt < nk; kt++) {
                            int k0 = kt * KC;
                            int kc = std::min(KC, o - k0);
                            const float* pp = packed_B + ((size_t)jt * nk + kt) * KC * JBLK + (v - j0);
                            for (int k = 0; k < kc; k++) {
                                __m256 bv = _mm256_loadu_ps(pp + (size_t)k * JBLK);
                                c0 = _mm256_fmadd_ps(_mm256_set1_ps(pa0[k0 + k]), bv, c0);
                                c1 = _mm256_fmadd_ps(_mm256_set1_ps(pa1[k0 + k]), bv, c1);
                                c2 = _mm256_fmadd_ps(_mm256_set1_ps(pa2[k0 + k]), bv, c2);
                                c3 = _mm256_fmadd_ps(_mm256_set1_ps(pa3[k0 + k]), bv, c3);
                                c4 = _mm256_fmadd_ps(_mm256_set1_ps(pa4[k0 + k]), bv, c4);
                                c5 = _mm256_fmadd_ps(_mm256_set1_ps(pa5[k0 + k]), bv, c5);
                            }
                        }
                        fuse_nhwc_6(c0, c1, c2, c3, c4, c5, _mm256_loadu_ps(fb + v), fmin, fmax);
                        _mm256_storeu_ps(C + (size_t)(i+0) * m + v, c0);
                        _mm256_storeu_ps(C + (size_t)(i+1) * m + v, c1);
                        _mm256_storeu_ps(C + (size_t)(i+2) * m + v, c2);
                        _mm256_storeu_ps(C + (size_t)(i+3) * m + v, c3);
                        _mm256_storeu_ps(C + (size_t)(i+4) * m + v, c4);
                        _mm256_storeu_ps(C + (size_t)(i+5) * m + v, c5);
                    }
                    if (v < je) {
                        __m256i mask = tail_mask_epi32(je - v);
                        __m256 c0 = _mm256_setzero_ps();
                        __m256 c1 = _mm256_setzero_ps();
                        __m256 c2 = _mm256_setzero_ps();
                        __m256 c3 = _mm256_setzero_ps();
                        __m256 c4 = _mm256_setzero_ps();
                        __m256 c5 = _mm256_setzero_ps();
                        for (int kt = 0; kt < nk; kt++) {
                            int k0 = kt * KC;
                            int kc = std::min(KC, o - k0);
                            const float* pp = packed_B + ((size_t)jt * nk + kt) * KC * JBLK + (v - j0);
                            for (int k = 0; k < kc; k++) {
                                __m256 bv = _mm256_loadu_ps(pp + (size_t)k * JBLK);
                                c0 = _mm256_fmadd_ps(_mm256_set1_ps(pa0[k0 + k]), bv, c0);
                                c1 = _mm256_fmadd_ps(_mm256_set1_ps(pa1[k0 + k]), bv, c1);
                                c2 = _mm256_fmadd_ps(_mm256_set1_ps(pa2[k0 + k]), bv, c2);
                                c3 = _mm256_fmadd_ps(_mm256_set1_ps(pa3[k0 + k]), bv, c3);
                                c4 = _mm256_fmadd_ps(_mm256_set1_ps(pa4[k0 + k]), bv, c4);
                                c5 = _mm256_fmadd_ps(_mm256_set1_ps(pa5[k0 + k]), bv, c5);
                            }
                        }
                        fuse_nhwc_6(c0, c1, c2, c3, c4, c5, _mm256_maskload_ps(fb + v, mask), fmin, fmax);
                        _mm256_maskstore_ps(C + (size_t)(i+0) * m + v, mask, c0);
                        _mm256_maskstore_ps(C + (size_t)(i+1) * m + v, mask, c1);
                        _mm256_maskstore_ps(C + (size_t)(i+2) * m + v, mask, c2);
                        _mm256_maskstore_ps(C + (size_t)(i+3) * m + v, mask, c3);
                        _mm256_maskstore_ps(C + (size_t)(i+4) * m + v, mask, c4);
                        _mm256_maskstore_ps(C + (size_t)(i+5) * m + v, mask, c5);
                    }
                }
                #endif // LOCUST
                for (; i < ie; i++) {
                    const float* pai = A + (size_t)i * o;
                    int v = j0;
                    for (; v + 8 <= je; v += 8) {
                        __m256 acc = _mm256_setzero_ps();
                        for (int kt = 0; kt < nk; kt++) {
                            int k0 = kt * KC;
                            int kc = std::min(KC, o - k0);
                            const float* pp = packed_B + ((size_t)jt * nk + kt) * KC * JBLK + (v - j0);
                            for (int k = 0; k < kc; ++k)
                                acc = _mm256_fmadd_ps(_mm256_set1_ps(pai[k0 + k]),
                                    _mm256_loadu_ps(pp + (size_t)k * JBLK), acc);
                        }
                        fuse_nhwc_1(acc, _mm256_loadu_ps(fb + v), fmin, fmax);
                        _mm256_storeu_ps(C + (size_t)i * m + v, acc);
                    }
                    if (v < je) {
                        __m256i mask = tail_mask_epi32(je - v);
                        __m256 acc = _mm256_setzero_ps();
                        for (int kt = 0; kt < nk; kt++) {
                            int k0 = kt * KC;
                            int kc = std::min(KC, o - k0);
                            const float* pp = packed_B + ((size_t)jt * nk + kt) * KC * JBLK + (v - j0);
                            for (int k = 0; k < kc; ++k)
                                acc = _mm256_fmadd_ps(_mm256_set1_ps(pai[k0 + k]),
                                    _mm256_loadu_ps(pp + (size_t)k * JBLK), acc);
                        }
                        fuse_nhwc_1(acc, _mm256_maskload_ps(fb + v, mask), fmin, fmax);
                        _mm256_maskstore_ps(C + (size_t)i * m + v, mask, acc);
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

    // B-hot path: few k-blocks, B-panel shared across 6-row groups.
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

                int i = i0;
                #if 0 // LOCUST
;                gen_nhwc_6row_block(4)
#else // LOCUST
                for (; i + 6 <= ie; i += 6) {
                    const float* pa0 = A + (size_t)(i+0) * o + k0;
                    const float* pa1 = A + (size_t)(i+1) * o + k0;
                    const float* pa2 = A + (size_t)(i+2) * o + k0;
                    const float* pa3 = A + (size_t)(i+3) * o + k0;
                    const float* pa4 = A + (size_t)(i+4) * o + k0;
                    const float* pa5 = A + (size_t)(i+5) * o + k0;
                    int v = j0;
                    for (; v + 8 <= je; v += 8) {
                        const float* pp = pb + (v - j0);
                        __m256 c0, c1, c2, c3, c4, c5;
                        if (k0 == 0) {
                            c0 = c1 = c2 = c3 = c4 = c5 = _mm256_setzero_ps();
                        } else {
                            c0 = _mm256_loadu_ps(C + (size_t)(i+0) * m + v);
                            c1 = _mm256_loadu_ps(C + (size_t)(i+1) * m + v);
                            c2 = _mm256_loadu_ps(C + (size_t)(i+2) * m + v);
                            c3 = _mm256_loadu_ps(C + (size_t)(i+3) * m + v);
                            c4 = _mm256_loadu_ps(C + (size_t)(i+4) * m + v);
                            c5 = _mm256_loadu_ps(C + (size_t)(i+5) * m + v);
                        }
                        for (int k = 0; k < kc; k++) {
                            __m256 bv = _mm256_loadu_ps(pp + (size_t)k * JBLK);
                            c0 = _mm256_fmadd_ps(_mm256_set1_ps(pa0[k0+k]), bv, c0);
                            c1 = _mm256_fmadd_ps(_mm256_set1_ps(pa1[k0+k]), bv, c1);
                            c2 = _mm256_fmadd_ps(_mm256_set1_ps(pa2[k0+k]), bv, c2);
                            c3 = _mm256_fmadd_ps(_mm256_set1_ps(pa3[k0+k]), bv, c3);
                            c4 = _mm256_fmadd_ps(_mm256_set1_ps(pa4[k0+k]), bv, c4);
                            c5 = _mm256_fmadd_ps(_mm256_set1_ps(pa5[k0+k]), bv, c5);
                        }
                        fuse_nhwc_6(c0, c1, c2, c3, c4, c5, _mm256_loadu_ps(fb + v), fmin, fmax);
                        _mm256_storeu_ps(C + (size_t)(i+0) * m + v, c0);
                        _mm256_storeu_ps(C + (size_t)(i+1) * m + v, c1);
                        _mm256_storeu_ps(C + (size_t)(i+2) * m + v, c2);
                        _mm256_storeu_ps(C + (size_t)(i+3) * m + v, c3);
                        _mm256_storeu_ps(C + (size_t)(i+4) * m + v, c4);
                        _mm256_storeu_ps(C + (size_t)(i+5) * m + v, c5);
                    }
                    if (v < je) {
                        __m256i mask = tail_mask_epi32(je - v);
                        const float* pp = pb + (v - j0);
                        __m256 c0, c1, c2, c3, c4, c5;
                        if (k0 == 0) {
                            c0 = c1 = c2 = c3 = c4 = c5 = _mm256_setzero_ps();
                        } else {
                            c0 = _mm256_maskload_ps(C + (size_t)(i+0) * m + v, mask);
                            c1 = _mm256_maskload_ps(C + (size_t)(i+1) * m + v, mask);
                            c2 = _mm256_maskload_ps(C + (size_t)(i+2) * m + v, mask);
                            c3 = _mm256_maskload_ps(C + (size_t)(i+3) * m + v, mask);
                            c4 = _mm256_maskload_ps(C + (size_t)(i+4) * m + v, mask);
                            c5 = _mm256_maskload_ps(C + (size_t)(i+5) * m + v, mask);
                        }
                        for (int k = 0; k < kc; k++) {
                            __m256 bv = _mm256_loadu_ps(pp + (size_t)k * JBLK);
                            c0 = _mm256_fmadd_ps(_mm256_set1_ps(pa0[k0+k]), bv, c0);
                            c1 = _mm256_fmadd_ps(_mm256_set1_ps(pa1[k0+k]), bv, c1);
                            c2 = _mm256_fmadd_ps(_mm256_set1_ps(pa2[k0+k]), bv, c2);
                            c3 = _mm256_fmadd_ps(_mm256_set1_ps(pa3[k0+k]), bv, c3);
                            c4 = _mm256_fmadd_ps(_mm256_set1_ps(pa4[k0+k]), bv, c4);
                            c5 = _mm256_fmadd_ps(_mm256_set1_ps(pa5[k0+k]), bv, c5);
                        }
                        fuse_nhwc_6(c0, c1, c2, c3, c4, c5, _mm256_maskload_ps(fb + v, mask), fmin, fmax);
                        _mm256_maskstore_ps(C + (size_t)(i+0) * m + v, mask, c0);
                        _mm256_maskstore_ps(C + (size_t)(i+1) * m + v, mask, c1);
                        _mm256_maskstore_ps(C + (size_t)(i+2) * m + v, mask, c2);
                        _mm256_maskstore_ps(C + (size_t)(i+3) * m + v, mask, c3);
                        _mm256_maskstore_ps(C + (size_t)(i+4) * m + v, mask, c4);
                        _mm256_maskstore_ps(C + (size_t)(i+5) * m + v, mask, c5);
                    }
                }
                #endif // LOCUST
                for (; i < ie; i++) {
                    const float* pai = A + (size_t)i * o + k0;
                    int v = j0;
                    for (; v + 8 <= je; v += 8) {
                        const float* pp = pb + (v - j0);
                        __m256 acc = (k0 == 0) ? _mm256_setzero_ps()
                            : _mm256_loadu_ps(C + (size_t)i * m + v);
                        for (int k = 0; k < kc; ++k) {
                            acc = _mm256_fmadd_ps(_mm256_set1_ps(pai[k]),
                                _mm256_loadu_ps(pp + (size_t)k * JBLK), acc);
                        }
                        fuse_nhwc_1(acc, _mm256_loadu_ps(fb + v), fmin, fmax);
                        _mm256_storeu_ps(C + (size_t)i * m + v, acc);
                    }
                    if (v < je) {
                        __m256i mask = tail_mask_epi32(je - v);
                        const float* pp = pb + (v - j0);
                        __m256 acc = (k0 == 0) ? _mm256_setzero_ps()
                            : _mm256_maskload_ps(C + (size_t)i * m + v, mask);
                        for (int k = 0; k < kc; ++k) {
                            acc = _mm256_fmadd_ps(_mm256_set1_ps(pai[k]),
                                _mm256_loadu_ps(pp + (size_t)k * JBLK), acc);
                        }
                        fuse_nhwc_1(acc, _mm256_maskload_ps(fb + v, mask), fmin, fmax);
                        _mm256_maskstore_ps(C + (size_t)i * m + v, mask, acc);
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
    });
}

// Batched GEMM for NHWC Winograd: 36 independent GEMMs sharing dispatch.
// C_batch[p][n×m] = A_batch[p][n×o] × packed_B_batch[p][o×m]  for p in [0..36)
template <typename PostFn>
// @nnr-meta isa=AVX2 dtype=fp32 layout=NHWC special=GEMM tiling=[K,MR,NR] fusion=post_op
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
    // parallelize across 36 positions × ntiles.
    if (ni <= 1 && nj > 2) {
        int ntiles = std::max(ni, 1) * nj;
        int total = 36 * ntiles;
        int nt = ((int64_t)n * m * o * 36 > (1 << 21)) ? nnr::compute_threads(total) : 1;
        NNR_POOL_ENSURE_SCRATCH((size_t)KC * 6 * sizeof(float));
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
                for (; i + 6 <= ie; i += 6) {
                    const float* pa[6];
                    float* pc[6];
                    for (int r = 0; r < 6; r++) {
                        pa[r] = A + (size_t)(i + r) * o + k0;
                        pc[r] = C + (size_t)(i + r) * m;
                    }
                    #if 0 // LOCUST
;                    gen_pack_a_6(5)
#else // LOCUST
                    for (int k = 0; k < kc; k++) {
                        pa_pack[k * 6 + 0] = pa[0][k]; pa_pack[k * 6 + 1] = pa[1][k];
                        pa_pack[k * 6 + 2] = pa[2][k]; pa_pack[k * 6 + 3] = pa[3][k];
                        pa_pack[k * 6 + 4] = pa[4][k]; pa_pack[k * 6 + 5] = pa[5][k];
                    }
                    #endif // LOCUST
                    #if 0 // LOCUST
                    int v = j0;
;                    gen_packed_vloop(5)
#else // LOCUST
                    int v = j0;
                    for (; v + 8 <= je; v += 8) {
                        const float* pp = pb + (v - j0);
                        __m256 c0, c1, c2, c3, c4, c5;
                        if (k0 == 0) { c0 = c1 = c2 = c3 = c4 = c5 = _mm256_setzero_ps(); }
                        else {
                            c0 = _mm256_loadu_ps(pc[0] + v); c1 = _mm256_loadu_ps(pc[1] + v);
                            c2 = _mm256_loadu_ps(pc[2] + v); c3 = _mm256_loadu_ps(pc[3] + v);
                            c4 = _mm256_loadu_ps(pc[4] + v); c5 = _mm256_loadu_ps(pc[5] + v);
                        }
                        for (int k = 0; k < kc; k++) {
                            __m256 bv = _mm256_loadu_ps(pp + (size_t)k * JBLK);
                            const float* ap = pa_pack + k * 6;
                            c0 = _mm256_fmadd_ps(_mm256_set1_ps(ap[0]), bv, c0);
                            c1 = _mm256_fmadd_ps(_mm256_set1_ps(ap[1]), bv, c1);
                            c2 = _mm256_fmadd_ps(_mm256_set1_ps(ap[2]), bv, c2);
                            c3 = _mm256_fmadd_ps(_mm256_set1_ps(ap[3]), bv, c3);
                            c4 = _mm256_fmadd_ps(_mm256_set1_ps(ap[4]), bv, c4);
                            c5 = _mm256_fmadd_ps(_mm256_set1_ps(ap[5]), bv, c5);
                        }
                        fuse_nhwc_6(c0, c1, c2, c3, c4, c5, _mm256_loadu_ps(fb + v), fmin, fmax);
                        _mm256_storeu_ps(pc[0] + v, c0); _mm256_storeu_ps(pc[1] + v, c1);
                        _mm256_storeu_ps(pc[2] + v, c2); _mm256_storeu_ps(pc[3] + v, c3);
                        _mm256_storeu_ps(pc[4] + v, c4); _mm256_storeu_ps(pc[5] + v, c5);
                    }
                    if (v < je) {
                        __m256i mask = tail_mask_epi32(je - v);
                        const float* pp = pb + (v - j0);
                        __m256 c0, c1, c2, c3, c4, c5;
                        if (k0 == 0) { c0 = c1 = c2 = c3 = c4 = c5 = _mm256_setzero_ps(); }
                        else {
                            c0 = _mm256_maskload_ps(pc[0] + v, mask); c1 = _mm256_maskload_ps(pc[1] + v, mask);
                            c2 = _mm256_maskload_ps(pc[2] + v, mask); c3 = _mm256_maskload_ps(pc[3] + v, mask);
                            c4 = _mm256_maskload_ps(pc[4] + v, mask); c5 = _mm256_maskload_ps(pc[5] + v, mask);
                        }
                        for (int k = 0; k < kc; k++) {
                            __m256 bv = _mm256_loadu_ps(pp + (size_t)k * JBLK);
                            const float* ap = pa_pack + k * 6;
                            c0 = _mm256_fmadd_ps(_mm256_set1_ps(ap[0]), bv, c0);
                            c1 = _mm256_fmadd_ps(_mm256_set1_ps(ap[1]), bv, c1);
                            c2 = _mm256_fmadd_ps(_mm256_set1_ps(ap[2]), bv, c2);
                            c3 = _mm256_fmadd_ps(_mm256_set1_ps(ap[3]), bv, c3);
                            c4 = _mm256_fmadd_ps(_mm256_set1_ps(ap[4]), bv, c4);
                            c5 = _mm256_fmadd_ps(_mm256_set1_ps(ap[5]), bv, c5);
                        }
                        fuse_nhwc_6(c0, c1, c2, c3, c4, c5, _mm256_maskload_ps(fb + v, mask), fmin, fmax);
                        _mm256_maskstore_ps(pc[0] + v, mask, c0); _mm256_maskstore_ps(pc[1] + v, mask, c1);
                        _mm256_maskstore_ps(pc[2] + v, mask, c2); _mm256_maskstore_ps(pc[3] + v, mask, c3);
                        _mm256_maskstore_ps(pc[4] + v, mask, c4); _mm256_maskstore_ps(pc[5] + v, mask, c5);
                    }
                    #endif // LOCUST
                }
                for (; i < ie; i++) {
                    float* pci = C + (size_t)i * m;
                    const float* pai = A + (size_t)i * o + k0;
                    int v = j0;
                    for (; v + 8 <= je; v += 8) {
                        const float* pp = pb + (v - j0);
                        __m256 acc = (k0 == 0) ? _mm256_setzero_ps() : _mm256_loadu_ps(pci + v);
                        for (int k = 0; k < kc; ++k) {
                            acc = _mm256_fmadd_ps(_mm256_set1_ps(pai[k]),
                                _mm256_loadu_ps(pp + (size_t)k * JBLK), acc);
                        }
                        fuse_nhwc_1(acc, _mm256_loadu_ps(fb + v), fmin, fmax);
                        _mm256_storeu_ps(pci + v, acc);
                    }
                    if (v < je) {
                        __m256i mask = tail_mask_epi32(je - v);
                        const float* pp = pb + (v - j0);
                        __m256 acc = (k0 == 0) ? _mm256_setzero_ps()
                            : _mm256_maskload_ps(pci + v, mask);
                        for (int k = 0; k < kc; ++k) {
                            acc = _mm256_fmadd_ps(_mm256_set1_ps(pai[k]),
                                _mm256_loadu_ps(pp + (size_t)k * JBLK), acc);
                        }
                        fuse_nhwc_1(acc, _mm256_maskload_ps(fb + v, mask), fmin, fmax);
                        _mm256_maskstore_ps(pci + v, mask, acc);
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
;                    gen_nhwc_6row_block(5)
#else // LOCUST
                    for (; i + 6 <= ie; i += 6) {
                        const float* pa0 = A + (size_t)(i+0) * o + k0;
                        const float* pa1 = A + (size_t)(i+1) * o + k0;
                        const float* pa2 = A + (size_t)(i+2) * o + k0;
                        const float* pa3 = A + (size_t)(i+3) * o + k0;
                        const float* pa4 = A + (size_t)(i+4) * o + k0;
                        const float* pa5 = A + (size_t)(i+5) * o + k0;
                        int v = j0;
                        for (; v + 8 <= je; v += 8) {
                            const float* pp = pb + (v - j0);
                            __m256 c0, c1, c2, c3, c4, c5;
                            if (k0 == 0) {
                                c0 = c1 = c2 = c3 = c4 = c5 = _mm256_setzero_ps();
                            } else {
                                c0 = _mm256_loadu_ps(C + (size_t)(i+0) * m + v);
                                c1 = _mm256_loadu_ps(C + (size_t)(i+1) * m + v);
                                c2 = _mm256_loadu_ps(C + (size_t)(i+2) * m + v);
                                c3 = _mm256_loadu_ps(C + (size_t)(i+3) * m + v);
                                c4 = _mm256_loadu_ps(C + (size_t)(i+4) * m + v);
                                c5 = _mm256_loadu_ps(C + (size_t)(i+5) * m + v);
                            }
                            for (int k = 0; k < kc; k++) {
                                __m256 bv = _mm256_loadu_ps(pp + (size_t)k * JBLK);
                                c0 = _mm256_fmadd_ps(_mm256_set1_ps(pa0[k0+k]), bv, c0);
                                c1 = _mm256_fmadd_ps(_mm256_set1_ps(pa1[k0+k]), bv, c1);
                                c2 = _mm256_fmadd_ps(_mm256_set1_ps(pa2[k0+k]), bv, c2);
                                c3 = _mm256_fmadd_ps(_mm256_set1_ps(pa3[k0+k]), bv, c3);
                                c4 = _mm256_fmadd_ps(_mm256_set1_ps(pa4[k0+k]), bv, c4);
                                c5 = _mm256_fmadd_ps(_mm256_set1_ps(pa5[k0+k]), bv, c5);
                            }
                            fuse_nhwc_6(c0, c1, c2, c3, c4, c5, _mm256_loadu_ps(fb + v), fmin, fmax);
                            _mm256_storeu_ps(C + (size_t)(i+0) * m + v, c0);
                            _mm256_storeu_ps(C + (size_t)(i+1) * m + v, c1);
                            _mm256_storeu_ps(C + (size_t)(i+2) * m + v, c2);
                            _mm256_storeu_ps(C + (size_t)(i+3) * m + v, c3);
                            _mm256_storeu_ps(C + (size_t)(i+4) * m + v, c4);
                            _mm256_storeu_ps(C + (size_t)(i+5) * m + v, c5);
                        }
                        if (v < je) {
                            __m256i mask = tail_mask_epi32(je - v);
                            const float* pp = pb + (v - j0);
                            __m256 c0, c1, c2, c3, c4, c5;
                            if (k0 == 0) {
                                c0 = c1 = c2 = c3 = c4 = c5 = _mm256_setzero_ps();
                            } else {
                                c0 = _mm256_maskload_ps(C + (size_t)(i+0) * m + v, mask);
                                c1 = _mm256_maskload_ps(C + (size_t)(i+1) * m + v, mask);
                                c2 = _mm256_maskload_ps(C + (size_t)(i+2) * m + v, mask);
                                c3 = _mm256_maskload_ps(C + (size_t)(i+3) * m + v, mask);
                                c4 = _mm256_maskload_ps(C + (size_t)(i+4) * m + v, mask);
                                c5 = _mm256_maskload_ps(C + (size_t)(i+5) * m + v, mask);
                            }
                            for (int k = 0; k < kc; k++) {
                                __m256 bv = _mm256_loadu_ps(pp + (size_t)k * JBLK);
                                c0 = _mm256_fmadd_ps(_mm256_set1_ps(pa0[k0+k]), bv, c0);
                                c1 = _mm256_fmadd_ps(_mm256_set1_ps(pa1[k0+k]), bv, c1);
                                c2 = _mm256_fmadd_ps(_mm256_set1_ps(pa2[k0+k]), bv, c2);
                                c3 = _mm256_fmadd_ps(_mm256_set1_ps(pa3[k0+k]), bv, c3);
                                c4 = _mm256_fmadd_ps(_mm256_set1_ps(pa4[k0+k]), bv, c4);
                                c5 = _mm256_fmadd_ps(_mm256_set1_ps(pa5[k0+k]), bv, c5);
                            }
                            fuse_nhwc_6(c0, c1, c2, c3, c4, c5, _mm256_maskload_ps(fb + v, mask), fmin, fmax);
                            _mm256_maskstore_ps(C + (size_t)(i+0) * m + v, mask, c0);
                            _mm256_maskstore_ps(C + (size_t)(i+1) * m + v, mask, c1);
                            _mm256_maskstore_ps(C + (size_t)(i+2) * m + v, mask, c2);
                            _mm256_maskstore_ps(C + (size_t)(i+3) * m + v, mask, c3);
                            _mm256_maskstore_ps(C + (size_t)(i+4) * m + v, mask, c4);
                            _mm256_maskstore_ps(C + (size_t)(i+5) * m + v, mask, c5);
                        }
                    }
                    #endif // LOCUST
                    for (; i < ie; i++) {
                        const float* pai = A + (size_t)i * o + k0;
                        int v = j0;
                        for (; v + 8 <= je; v += 8) {
                            const float* pp = pb + (v - j0);
                            __m256 acc = (k0 == 0) ? _mm256_setzero_ps()
                                : _mm256_loadu_ps(C + (size_t)i * m + v);
                            for (int k = 0; k < kc; ++k)
                                acc = _mm256_fmadd_ps(_mm256_set1_ps(pai[k]),
                                    _mm256_loadu_ps(pp + (size_t)k * JBLK), acc);
                            fuse_nhwc_1(acc, _mm256_loadu_ps(fb + v), fmin, fmax);
                            _mm256_storeu_ps(C + (size_t)i * m + v, acc);
                        }
                        if (v < je) {
                            __m256i mask = tail_mask_epi32(je - v);
                            const float* pp = pb + (v - j0);
                            __m256 acc = (k0 == 0) ? _mm256_setzero_ps()
                                : _mm256_maskload_ps(C + (size_t)i * m + v, mask);
                            for (int k = 0; k < kc; ++k)
                                acc = _mm256_fmadd_ps(_mm256_set1_ps(pai[k]),
                                    _mm256_loadu_ps(pp + (size_t)k * JBLK), acc);
                            fuse_nhwc_1(acc, _mm256_maskload_ps(fb + v, mask), fmin, fmax);
                            _mm256_maskstore_ps(C + (size_t)i * m + v, mask, acc);
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

}} // namespace nnr::avx2
