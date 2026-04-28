#include "cpu_features.h"
#ifdef NNR_ARCH_X64
#include "simd_math_avx512.h"
#include "thread_pool.h"

namespace nnr {

void sigmoid_avx512(float* data, size_t n) {
    constexpr size_t CHUNK = 16384;
    int nchunks = (int)((n + CHUNK - 1) / CHUNK);
    nnr::for_static(0, nchunks, nchunks > 1, [&](int c) {
        size_t start = (size_t)c * CHUNK;
        size_t end = std::min(start + CHUNK, n);
        size_t i = start;
        for (; i + 16 <= end; i += 16) {
            __m512 v = _mm512_loadu_ps(data + i);
            _mm512_storeu_ps(data + i, sigmoid512_ps(v));
        }
        // Scalar tail
        for (; i < end; i++)
            data[i] = 1.0f / (1.0f + expf(-data[i]));
    });
}

void silu_avx512(const float* src, float* dst, size_t n) {
    constexpr size_t CHUNK = 16384;
    int nchunks = (int)((n + CHUNK - 1) / CHUNK);
    nnr::for_static(0, nchunks, nchunks > 1, [&](int c) {
        size_t start = (size_t)c * CHUNK;
        size_t end = std::min(start + CHUNK, n);
        size_t i = start;
        for (; i + 16 <= end; i += 16) {
            __m512 v = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, silu512_ps(v));
        }
        for (; i < end; i++)
            dst[i] = src[i] / (1.0f + expf(-src[i]));
    });
}

void mul_avx512(const float* a, const float* b, float* dst, size_t n) {
    constexpr size_t CHUNK = 16384;
    int nchunks = (int)((n + CHUNK - 1) / CHUNK);
    nnr::for_static(0, nchunks, nchunks > 1, [&](int c) {
        size_t start = (size_t)c * CHUNK;
        size_t end = std::min(start + CHUNK, n);
        size_t i = start;
        for (; i + 16 <= end; i += 16) {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            _mm512_storeu_ps(dst + i, _mm512_mul_ps(va, vb));
        }
        for (; i < end; i++)
            dst[i] = a[i] * b[i];
    });
}

void sub_avx512(const float* a, const float* b, float* dst, size_t n) {
    constexpr size_t CHUNK = 16384;
    int nchunks = (int)((n + CHUNK - 1) / CHUNK);
    nnr::for_static(0, nchunks, nchunks > 1, [&](int c) {
        size_t start = (size_t)c * CHUNK;
        size_t end = std::min(start + CHUNK, n);
        size_t i = start;
        for (; i + 16 <= end; i += 16) {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            _mm512_storeu_ps(dst + i, _mm512_sub_ps(va, vb));
        }
        if (i < end) {
            __mmask16 mask = (__mmask16)((1u << (end - i)) - 1);
            _mm512_mask_storeu_ps(dst + i, mask, _mm512_sub_ps(
                _mm512_maskz_loadu_ps(mask, a + i),
                _mm512_maskz_loadu_ps(mask, b + i)));
        }
    });
}

void div_avx512(const float* a, const float* b, float* dst, size_t n) {
    constexpr size_t CHUNK = 16384;
    int nchunks = (int)((n + CHUNK - 1) / CHUNK);
    nnr::for_static(0, nchunks, nchunks > 1, [&](int c) {
        size_t start = (size_t)c * CHUNK;
        size_t end = std::min(start + CHUNK, n);
        size_t i = start;
        for (; i + 16 <= end; i += 16) {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            _mm512_storeu_ps(dst + i, _mm512_div_ps(va, vb));
        }
        if (i < end) {
            __mmask16 mask = (__mmask16)((1u << (end - i)) - 1);
            _mm512_mask_storeu_ps(dst + i, mask, _mm512_div_ps(
                _mm512_maskz_loadu_ps(mask, a + i),
                _mm512_maskz_loadu_ps(mask, b + i)));
        }
    });
}

// ---------------------------------------------------------------------------
// Generic AVX-512 fp32 broadcast walker.
//
// Strategy: find the largest "inner" contiguous run of trailing dims that both
// inputs traverse with the natural row-major stride (no broadcast) — collapse
// those into a single SIMD inner loop. Iterate the remaining outer dims, each
// of which may broadcast independently on a or b. Thread the outer iteration.
//
// Innermost-dim handling covers four broadcast kinds detected from the last
// dim of the output:
//   VV: both contiguous (stride 1 on both)        → vec * vec
//   VS: a contiguous, b broadcast (stride 0 on b) → vec * splat(b[base])
//   SV: a broadcast, b contiguous                 → splat(a[base]) * vec
//   SS: both broadcast                            → splat(a[base] * b[base])
//
// Greedy collapse only extends past the innermost dim for VV (collapsing
// further requires both inputs to remain stride-contig). For VS/SV/SS we
// stop at the innermost dim and let outer-loop broadcast walking handle
// the rest.
// ---------------------------------------------------------------------------

enum class binary_kind_t : uint8_t { Mul, Add, Sub, Div };

template <binary_kind_t Kind>
static inline __m512 simd_apply_avx512(__m512 a, __m512 b) {
    if constexpr (Kind == binary_kind_t::Mul) return _mm512_mul_ps(a, b);
    if constexpr (Kind == binary_kind_t::Add) return _mm512_add_ps(a, b);
    if constexpr (Kind == binary_kind_t::Sub) return _mm512_sub_ps(a, b);
    if constexpr (Kind == binary_kind_t::Div) return _mm512_div_ps(a, b);
}

template <binary_kind_t Kind>
static inline float scalar_apply(float a, float b) {
    if constexpr (Kind == binary_kind_t::Mul) return a * b;
    if constexpr (Kind == binary_kind_t::Add) return a + b;
    if constexpr (Kind == binary_kind_t::Sub) return a - b;
    if constexpr (Kind == binary_kind_t::Div) return a / b;
}

template <binary_kind_t Kind>
static void binary_broadcast_avx512_impl(
    const float* pa, const float* pb, float* py,
    const int* dims, const int* a_bstr, const int* b_bstr, int ndim)
{
    if (ndim <= 0) return;

    // Classify innermost dim (drives inner SIMD kernel selection).
    int as_in = a_bstr[ndim - 1];
    int bs_in = b_bstr[ndim - 1];
    int inner_n = dims[ndim - 1];
    int inner_dim_count = 1;

    // Greedy collapse only for VV (both contig). Mixed broadcast patterns can't
    // be folded into a single linear SIMD chunk.
    if (as_in == 1 && bs_in == 1) {
        size_t exp = (size_t)inner_n;
        for (int d = ndim - 2; d >= 0; d--) {
            if ((size_t)a_bstr[d] == exp && (size_t)b_bstr[d] == exp) {
                inner_n *= dims[d];
                exp *= (size_t)dims[d];
                inner_dim_count++;
            } else {
                break;
            }
        }
    }

    int outer_ndim = ndim - inner_dim_count;
    if (outer_ndim == 0) {
        // Whole tensor collapses to one SIMD run (effectively SAME_SHAPE).
        size_t i = 0;
        for (; i + 16 <= (size_t)inner_n; i += 16) {
            __m512 va = _mm512_loadu_ps(pa + i);
            __m512 vb = _mm512_loadu_ps(pb + i);
            _mm512_storeu_ps(py + i, simd_apply_avx512<Kind>(va, vb));
        }
        for (; i < (size_t)inner_n; ++i)
            py[i] = scalar_apply<Kind>(pa[i], pb[i]);
        return;
    }

    size_t outer_count = 1;
    for (int d = 0; d < outer_ndim; d++) outer_count *= (size_t)dims[d];

    // Capture stride/dim arrays by copying into local vectors so the lambda
    // doesn't carry pointer aliasing concerns across threads.
    nnr::for_static(0, (int)outer_count, outer_count > 16, [&](int idx) {
        // Decode linear outer index → multi-dim index → input offsets.
        int a_off = 0, b_off = 0;
        size_t rem = (size_t)idx;
        for (int d = outer_ndim - 1; d >= 0; d--) {
            int i = (int)(rem % (size_t)dims[d]);
            rem /= (size_t)dims[d];
            a_off += i * a_bstr[d];
            b_off += i * b_bstr[d];
        }
        size_t y_off = (size_t)idx * (size_t)inner_n;

        const float* la = pa + a_off;
        const float* lb = pb + b_off;
        float* ly = py + y_off;
        size_t i = 0;

        if (as_in == 1 && bs_in == 1) {
            for (; i + 16 <= (size_t)inner_n; i += 16) {
                __m512 va = _mm512_loadu_ps(la + i);
                __m512 vb = _mm512_loadu_ps(lb + i);
                _mm512_storeu_ps(ly + i, simd_apply_avx512<Kind>(va, vb));
            }
            for (; i < (size_t)inner_n; ++i)
                ly[i] = scalar_apply<Kind>(la[i], lb[i]);
        } else if (as_in == 1 && bs_in == 0) {
            __m512 vb = _mm512_set1_ps(*lb);
            float sb = *lb;
            for (; i + 16 <= (size_t)inner_n; i += 16) {
                __m512 va = _mm512_loadu_ps(la + i);
                _mm512_storeu_ps(ly + i, simd_apply_avx512<Kind>(va, vb));
            }
            for (; i < (size_t)inner_n; ++i)
                ly[i] = scalar_apply<Kind>(la[i], sb);
        } else if (as_in == 0 && bs_in == 1) {
            __m512 va = _mm512_set1_ps(*la);
            float sa = *la;
            for (; i + 16 <= (size_t)inner_n; i += 16) {
                __m512 vb = _mm512_loadu_ps(lb + i);
                _mm512_storeu_ps(ly + i, simd_apply_avx512<Kind>(va, vb));
            }
            for (; i < (size_t)inner_n; ++i)
                ly[i] = scalar_apply<Kind>(sa, lb[i]);
        } else {
            // Both broadcast: result is a constant splat over inner_n.
            float v = scalar_apply<Kind>(*la, *lb);
            __m512 vv = _mm512_set1_ps(v);
            for (; i + 16 <= (size_t)inner_n; i += 16)
                _mm512_storeu_ps(ly + i, vv);
            for (; i < (size_t)inner_n; ++i) ly[i] = v;
        }
    });
}

void mul_broadcast_avx512(const float* a, const float* b, float* y,
    const int* dims, const int* a_bstr, const int* b_bstr, int ndim) {
    binary_broadcast_avx512_impl<binary_kind_t::Mul>(a, b, y, dims, a_bstr, b_bstr, ndim);
}
void add_broadcast_avx512(const float* a, const float* b, float* y,
    const int* dims, const int* a_bstr, const int* b_bstr, int ndim) {
    binary_broadcast_avx512_impl<binary_kind_t::Add>(a, b, y, dims, a_bstr, b_bstr, ndim);
}
void sub_broadcast_avx512(const float* a, const float* b, float* y,
    const int* dims, const int* a_bstr, const int* b_bstr, int ndim) {
    binary_broadcast_avx512_impl<binary_kind_t::Sub>(a, b, y, dims, a_bstr, b_bstr, ndim);
}
void div_broadcast_avx512(const float* a, const float* b, float* y,
    const int* dims, const int* a_bstr, const int* b_bstr, int ndim) {
    binary_broadcast_avx512_impl<binary_kind_t::Div>(a, b, y, dims, a_bstr, b_bstr, ndim);
}

void gelu_avx512(const float* src, float* dst, size_t n) {
    const __m512 inv_sqrt2 = _mm512_set1_ps(0.7071067811865476f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one = _mm512_set1_ps(1.0f);

    constexpr size_t CHUNK = 16384;
    int nchunks = (int)((n + CHUNK - 1) / CHUNK);
    nnr::for_static(0, nchunks, nchunks > 1, [&](int c) {
        size_t start = (size_t)c * CHUNK;
        size_t end = std::min(start + CHUNK, n);
        size_t i = start;
        for (; i + 16 <= end; i += 16) {
            __m512 x = _mm512_loadu_ps(src + i);
            __m512 e = erf512_ps(_mm512_mul_ps(x, inv_sqrt2));
            __m512 g = _mm512_mul_ps(_mm512_mul_ps(half, x), _mm512_add_ps(one, e));
            _mm512_storeu_ps(dst + i, g);
        }
        if (i < end) {
            __mmask16 mask = (__mmask16)((1u << (end - i)) - 1);
            __m512 x = _mm512_maskz_loadu_ps(mask, src + i);
            __m512 e = erf512_ps(_mm512_mul_ps(x, inv_sqrt2));
            __m512 g = _mm512_mul_ps(_mm512_mul_ps(half, x), _mm512_add_ps(one, e));
            _mm512_mask_storeu_ps(dst + i, mask, g);
        }
    });
}

} // namespace nnr
#endif // NNR_ARCH_X64
