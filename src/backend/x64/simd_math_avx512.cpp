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
