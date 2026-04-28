#include "cpu_features.h"
#ifdef NNR_ARCH_X64
#include "simd_math_avx2.h"
#include "thread_pool.h"

namespace nnr {

void sigmoid_avx2(float* data, size_t n) {
    constexpr size_t CHUNK = 16384;
    int nchunks = (int)((n + CHUNK - 1) / CHUNK);
    nnr::for_static(0, nchunks, nchunks > 1, [&](int c) {
        size_t start = (size_t)c * CHUNK;
        size_t end = std::min(start + CHUNK, n);
        size_t i = start;
        for (; i + 8 <= end; i += 8) {
            __m256 v = _mm256_loadu_ps(data + i);
            _mm256_storeu_ps(data + i, sigmoid256_ps(v));
        }
        for (; i < end; i++)
            data[i] = 1.0f / (1.0f + expf(-data[i]));
    });
}

void silu_avx2(const float* src, float* dst, size_t n) {
    constexpr size_t CHUNK = 16384;
    int nchunks = (int)((n + CHUNK - 1) / CHUNK);
    nnr::for_static(0, nchunks, nchunks > 1, [&](int c) {
        size_t start = (size_t)c * CHUNK;
        size_t end = std::min(start + CHUNK, n);
        size_t i = start;
        for (; i + 8 <= end; i += 8) {
            __m256 v = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, silu256_ps(v));
        }
        for (; i < end; i++)
            dst[i] = src[i] / (1.0f + expf(-src[i]));
    });
}

void mul_avx2(const float* a, const float* b, float* dst, size_t n) {
    constexpr size_t CHUNK = 16384;
    int nchunks = (int)((n + CHUNK - 1) / CHUNK);
    nnr::for_static(0, nchunks, nchunks > 1, [&](int c) {
        size_t start = (size_t)c * CHUNK;
        size_t end = std::min(start + CHUNK, n);
        size_t i = start;
        for (; i + 8 <= end; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(dst + i, _mm256_mul_ps(va, vb));
        }
        for (; i < end; i++)
            dst[i] = a[i] * b[i];
    });
}

void sub_avx2(const float* a, const float* b, float* dst, size_t n) {
    constexpr size_t CHUNK = 16384;
    int nchunks = (int)((n + CHUNK - 1) / CHUNK);
    nnr::for_static(0, nchunks, nchunks > 1, [&](int c) {
        size_t start = (size_t)c * CHUNK;
        size_t end = std::min(start + CHUNK, n);
        size_t i = start;
        for (; i + 8 <= end; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(dst + i, _mm256_sub_ps(va, vb));
        }
        for (; i < end; i++)
            dst[i] = a[i] - b[i];
    });
}

void div_avx2(const float* a, const float* b, float* dst, size_t n) {
    constexpr size_t CHUNK = 16384;
    int nchunks = (int)((n + CHUNK - 1) / CHUNK);
    nnr::for_static(0, nchunks, nchunks > 1, [&](int c) {
        size_t start = (size_t)c * CHUNK;
        size_t end = std::min(start + CHUNK, n);
        size_t i = start;
        for (; i + 8 <= end; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(dst + i, _mm256_div_ps(va, vb));
        }
        for (; i < end; i++)
            dst[i] = a[i] / b[i];
    });
}

void gelu_avx2(const float* src, float* dst, size_t n) {
    const __m256 inv_sqrt2 = _mm256_set1_ps(0.7071067811865476f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);

    constexpr size_t CHUNK = 16384;
    int nchunks = (int)((n + CHUNK - 1) / CHUNK);
    nnr::for_static(0, nchunks, nchunks > 1, [&](int c) {
        size_t start = (size_t)c * CHUNK;
        size_t end = std::min(start + CHUNK, n);
        size_t i = start;
        for (; i + 8 <= end; i += 8) {
            __m256 x = _mm256_loadu_ps(src + i);
            __m256 e = erf256_ps(_mm256_mul_ps(x, inv_sqrt2));
            __m256 g = _mm256_mul_ps(_mm256_mul_ps(half, x), _mm256_add_ps(one, e));
            _mm256_storeu_ps(dst + i, g);
        }
        for (; i < end; i++) {
            float xv = src[i];
            dst[i] = 0.5f * xv * (1.0f + erff(xv * 0.7071067811865476f));
        }
    });
}

// ---------------------------------------------------------------------------
// Generic AVX-2 fp32 broadcast walker.
// Mirrors the AVX-512 implementation; inner SIMD width is 8 floats.
// ---------------------------------------------------------------------------

enum class binary_kind_avx2_t : uint8_t { Mul, Add, Sub, Div };

template <binary_kind_avx2_t Kind>
static inline __m256 simd_apply_avx2(__m256 a, __m256 b) {
    if constexpr (Kind == binary_kind_avx2_t::Mul) return _mm256_mul_ps(a, b);
    if constexpr (Kind == binary_kind_avx2_t::Add) return _mm256_add_ps(a, b);
    if constexpr (Kind == binary_kind_avx2_t::Sub) return _mm256_sub_ps(a, b);
    if constexpr (Kind == binary_kind_avx2_t::Div) return _mm256_div_ps(a, b);
}

template <binary_kind_avx2_t Kind>
static inline float scalar_apply_avx2(float a, float b) {
    if constexpr (Kind == binary_kind_avx2_t::Mul) return a * b;
    if constexpr (Kind == binary_kind_avx2_t::Add) return a + b;
    if constexpr (Kind == binary_kind_avx2_t::Sub) return a - b;
    if constexpr (Kind == binary_kind_avx2_t::Div) return a / b;
}

template <binary_kind_avx2_t Kind>
static void binary_broadcast_avx2_impl(
    const float* pa, const float* pb, float* py,
    const int* dims, const int* a_bstr, const int* b_bstr, int ndim)
{
    if (ndim <= 0) return;

    int as_in = a_bstr[ndim - 1];
    int bs_in = b_bstr[ndim - 1];
    int inner_n = dims[ndim - 1];
    int inner_dim_count = 1;

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
        size_t i = 0;
        for (; i + 8 <= (size_t)inner_n; i += 8) {
            __m256 va = _mm256_loadu_ps(pa + i);
            __m256 vb = _mm256_loadu_ps(pb + i);
            _mm256_storeu_ps(py + i, simd_apply_avx2<Kind>(va, vb));
        }
        for (; i < (size_t)inner_n; ++i)
            py[i] = scalar_apply_avx2<Kind>(pa[i], pb[i]);
        return;
    }

    size_t outer_count = 1;
    for (int d = 0; d < outer_ndim; d++) outer_count *= (size_t)dims[d];

    nnr::for_static(0, (int)outer_count, outer_count > 16, [&](int idx) {
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
            for (; i + 8 <= (size_t)inner_n; i += 8) {
                __m256 va = _mm256_loadu_ps(la + i);
                __m256 vb = _mm256_loadu_ps(lb + i);
                _mm256_storeu_ps(ly + i, simd_apply_avx2<Kind>(va, vb));
            }
            for (; i < (size_t)inner_n; ++i)
                ly[i] = scalar_apply_avx2<Kind>(la[i], lb[i]);
        } else if (as_in == 1 && bs_in == 0) {
            __m256 vb = _mm256_set1_ps(*lb);
            float sb = *lb;
            for (; i + 8 <= (size_t)inner_n; i += 8) {
                __m256 va = _mm256_loadu_ps(la + i);
                _mm256_storeu_ps(ly + i, simd_apply_avx2<Kind>(va, vb));
            }
            for (; i < (size_t)inner_n; ++i)
                ly[i] = scalar_apply_avx2<Kind>(la[i], sb);
        } else if (as_in == 0 && bs_in == 1) {
            __m256 va = _mm256_set1_ps(*la);
            float sa = *la;
            for (; i + 8 <= (size_t)inner_n; i += 8) {
                __m256 vb = _mm256_loadu_ps(lb + i);
                _mm256_storeu_ps(ly + i, simd_apply_avx2<Kind>(va, vb));
            }
            for (; i < (size_t)inner_n; ++i)
                ly[i] = scalar_apply_avx2<Kind>(sa, lb[i]);
        } else {
            float v = scalar_apply_avx2<Kind>(*la, *lb);
            __m256 vv = _mm256_set1_ps(v);
            for (; i + 8 <= (size_t)inner_n; i += 8)
                _mm256_storeu_ps(ly + i, vv);
            for (; i < (size_t)inner_n; ++i) ly[i] = v;
        }
    });
}

void mul_broadcast_avx2(const float* a, const float* b, float* y,
    const int* dims, const int* a_bstr, const int* b_bstr, int ndim) {
    binary_broadcast_avx2_impl<binary_kind_avx2_t::Mul>(a, b, y, dims, a_bstr, b_bstr, ndim);
}
void add_broadcast_avx2(const float* a, const float* b, float* y,
    const int* dims, const int* a_bstr, const int* b_bstr, int ndim) {
    binary_broadcast_avx2_impl<binary_kind_avx2_t::Add>(a, b, y, dims, a_bstr, b_bstr, ndim);
}
void sub_broadcast_avx2(const float* a, const float* b, float* y,
    const int* dims, const int* a_bstr, const int* b_bstr, int ndim) {
    binary_broadcast_avx2_impl<binary_kind_avx2_t::Sub>(a, b, y, dims, a_bstr, b_bstr, ndim);
}
void div_broadcast_avx2(const float* a, const float* b, float* y,
    const int* dims, const int* a_bstr, const int* b_bstr, int ndim) {
    binary_broadcast_avx2_impl<binary_kind_avx2_t::Div>(a, b, y, dims, a_bstr, b_bstr, ndim);
}

} // namespace nnr
#endif // NNR_ARCH_X64
