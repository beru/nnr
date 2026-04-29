#pragma once
// AVX2+FMA vectorized operations: clip, reduction, dot product.
// 256-bit (8 floats per vector) counterpart of avx512/vec_ops.h.

#include <immintrin.h>

namespace nnr::avx2 {

// Horizontal sum of 8 floats in a YMM register.
// @nnr-meta isa=AVX2 dtype=fp32
inline float hsum(__m256 v)
{
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

// Vectorized clamp: py[i] = clamp(px[i], minv, maxv)
// @nnr-meta isa=AVX2 dtype=fp32
inline void clip(const float* __restrict px, float* __restrict py, size_t len, float minv, float maxv)
{
    __m256 vmin = _mm256_set1_ps(minv);
    __m256 vmax = _mm256_set1_ps(maxv);
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 v = _mm256_loadu_ps(px + i);
        v = _mm256_max_ps(v, vmin);
        v = _mm256_min_ps(v, vmax);
        _mm256_storeu_ps(py + i, v);
    }
    for (; i < len; ++i) {
        float v = px[i];
        v = std::max(v, minv);
        v = std::min(v, maxv);
        py[i] = v;
    }
}

// Fused bias-add + clamp: data[i] = clamp(data[i] + bias, minv, maxv)
// @nnr-meta isa=AVX2 dtype=fp32 fusion=post_op
inline void bias_clip(float* __restrict data, int len, float bias, float minv, float maxv)
{
    __m256 vb = _mm256_set1_ps(bias);
    __m256 vmin = _mm256_set1_ps(minv);
    __m256 vmax = _mm256_set1_ps(maxv);
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 v = _mm256_add_ps(_mm256_loadu_ps(data + i), vb);
        _mm256_storeu_ps(data + i, _mm256_min_ps(_mm256_max_ps(v, vmin), vmax));
    }
    for (; i < len; ++i)
        data[i] = std::clamp(data[i] + bias, minv, maxv);
}

// Fused bias-add + relu: data[i] = max(data[i] + bias, 0)
// @nnr-meta isa=AVX2 dtype=fp32 fusion=post_op
inline void bias_relu(float* __restrict data, int len, float bias)
{
    __m256 vb = _mm256_set1_ps(bias);
    __m256 vz = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 v = _mm256_add_ps(_mm256_loadu_ps(data + i), vb);
        _mm256_storeu_ps(data + i, _mm256_max_ps(v, vz));
    }
    for (; i < len; ++i)
        data[i] = std::max(data[i] + bias, 0.0f);
}

// Bias-add only: data[i] += bias
// @nnr-meta isa=AVX2 dtype=fp32 fusion=post_op
inline void bias_add(float* __restrict data, int len, float bias)
{
    if (bias == 0.0f) return;
    __m256 vb = _mm256_set1_ps(bias);
    int i = 0;
    for (; i + 8 <= len; i += 8)
        _mm256_storeu_ps(data + i, _mm256_add_ps(_mm256_loadu_ps(data + i), vb));
    for (; i < len; ++i)
        data[i] += bias;
}

// Fused binary Add: data[i] += skip[i] + bias.
// @nnr-meta isa=AVX2 dtype=fp32 fusion=binary
inline void add_skip_bias(float* __restrict data, const float* __restrict skip,
                          int len, float bias)
{
    __m256 vb = _mm256_set1_ps(bias);
    int i = 0;
    if (bias == 0.0f) {
        for (; i + 8 <= len; i += 8)
            _mm256_storeu_ps(data + i, _mm256_add_ps(_mm256_loadu_ps(data + i),
                                                    _mm256_loadu_ps(skip + i)));
        for (; i < len; ++i) data[i] += skip[i];
    } else {
        for (; i + 8 <= len; i += 8) {
            __m256 d = _mm256_loadu_ps(data + i);
            __m256 s = _mm256_loadu_ps(skip + i);
            _mm256_storeu_ps(data + i, _mm256_add_ps(d, _mm256_add_ps(s, vb)));
        }
        for (; i < len; ++i) data[i] += skip[i] + bias;
    }
}

// HardSwish: data[i] = (data[i] + bias) * clamp((data[i] + bias) / 6 + 0.5, 0, 1)
// @nnr-meta isa=AVX2 dtype=fp32 fusion=post_op
inline void bias_hardswish(float* __restrict data, int len, float bias)
{
    __m256 vb   = _mm256_set1_ps(bias);
    __m256 v6   = _mm256_set1_ps(1.0f / 6.0f);
    __m256 vhalf = _mm256_set1_ps(0.5f);
    __m256 vzero = _mm256_setzero_ps();
    __m256 vone  = _mm256_set1_ps(1.0f);
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 x = _mm256_add_ps(_mm256_loadu_ps(data + i), vb);
        __m256 t = _mm256_min_ps(_mm256_max_ps(
            _mm256_add_ps(_mm256_mul_ps(x, v6), vhalf), vzero), vone);
        _mm256_storeu_ps(data + i, _mm256_mul_ps(x, t));
    }
    for (; i < len; ++i) {
        float x = data[i] + bias;
        data[i] = x * std::max(0.0f, std::min(1.0f, x / 6.0f + 0.5f));
    }
}

// HardSwish (no bias): py[i] = px[i] * clamp(px[i] / 6 + 0.5, 0, 1)
// @nnr-meta isa=AVX2 dtype=fp32 fusion=post_op
inline void hardswish(const float* __restrict px, float* __restrict py, size_t len)
{
    __m256 v6   = _mm256_set1_ps(1.0f / 6.0f);
    __m256 vhalf = _mm256_set1_ps(0.5f);
    __m256 vzero = _mm256_setzero_ps();
    __m256 vone  = _mm256_set1_ps(1.0f);
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 x = _mm256_loadu_ps(px + i);
        __m256 t = _mm256_min_ps(_mm256_max_ps(
            _mm256_add_ps(_mm256_mul_ps(x, v6), vhalf), vzero), vone);
        _mm256_storeu_ps(py + i, _mm256_mul_ps(x, t));
    }
    for (; i < len; ++i) {
        float x = px[i];
        py[i] = x * std::max(0.0f, std::min(1.0f, x / 6.0f + 0.5f));
    }
}

// Vectorized leaky relu: py[i] = px[i] >= 0 ? px[i] : px[i] * alpha
// @nnr-meta isa=AVX2 dtype=fp32 fusion=post_op
inline void leaky_relu(const float* __restrict px, float* __restrict py, int len, float alpha)
{
    __m256 va = _mm256_set1_ps(alpha);
    __m256 vz = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 v = _mm256_loadu_ps(px + i);
        __m256 neg = _mm256_mul_ps(v, va);
        __m256 mask = _mm256_cmp_ps(v, vz, _CMP_LT_OS);
        _mm256_storeu_ps(py + i, _mm256_blendv_ps(v, neg, mask));
    }
    for (; i < len; ++i)
        py[i] = px[i] >= 0 ? px[i] : px[i] * alpha;
}

// Fused bias-add + leaky relu: data[i] = (data[i]+bias) >= 0 ? (data[i]+bias) : (data[i]+bias)*alpha
// @nnr-meta isa=AVX2 dtype=fp32 fusion=post_op
inline void bias_leaky_relu(float* __restrict data, int len, float bias, float alpha)
{
    __m256 vb = _mm256_set1_ps(bias);
    __m256 va = _mm256_set1_ps(alpha);
    __m256 vz = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 v = _mm256_add_ps(_mm256_loadu_ps(data + i), vb);
        __m256 neg = _mm256_mul_ps(v, va);
        __m256 mask = _mm256_cmp_ps(v, vz, _CMP_LT_OS);
        _mm256_storeu_ps(data + i, _mm256_blendv_ps(v, neg, mask));
    }
    for (; i < len; ++i) {
        float v = data[i] + bias;
        data[i] = v >= 0 ? v : v * alpha;
    }
}

// Vectorized horizontal sum of a float array with 4 independent accumulators.
// @nnr-meta isa=AVX2 dtype=fp32
inline float reduce_sum(const float* data, int len)
{
#if 0 // LOCUST
;N = 4
;decl = " ".join(f"__m256 s{i} = _mm256_setzero_ps();" for i in range(N))
    @decl@
    int i = 0;
    for (; i + @N*8@ <= len; i += @N*8@) {
;for i in range(N):
        s@i@ = _mm256_add_ps(s@i@, _mm256_loadu_ps(data + i + @i*8@));
;    pass
    }
;step = 1
;while step < N:
;    for i in range(0, N, step * 2):
;        if i + step < N:
    s@i@ = _mm256_add_ps(s@i@, s@i+step@);
;            pass
;        pass
;    step *= 2
#else // LOCUST
    __m256 s0 = _mm256_setzero_ps(); __m256 s1 = _mm256_setzero_ps(); __m256 s2 = _mm256_setzero_ps(); __m256 s3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= len; i += 32) {
        s0 = _mm256_add_ps(s0, _mm256_loadu_ps(data + i + 0));
        s1 = _mm256_add_ps(s1, _mm256_loadu_ps(data + i + 8));
        s2 = _mm256_add_ps(s2, _mm256_loadu_ps(data + i + 16));
        s3 = _mm256_add_ps(s3, _mm256_loadu_ps(data + i + 24));
    }
    s0 = _mm256_add_ps(s0, s1);
    s2 = _mm256_add_ps(s2, s3);
    s0 = _mm256_add_ps(s0, s2);
#endif // LOCUST
    for (; i + 8 <= len; i += 8)
        s0 = _mm256_add_ps(s0, _mm256_loadu_ps(data + i));
    float total = hsum(s0);
    for (; i < len; i++)
        total += data[i];
    return total;
}

// Vectorized dot product with 4 independent accumulators to hide FMA latency.
// @nnr-meta isa=AVX2 dtype=fp32
inline float dot_product(const float* __restrict a, const float* __restrict b, int len)
{
#if 0 // LOCUST
;N = 4
;decl = " ".join(f"__m256 acc{i} = _mm256_setzero_ps();" for i in range(N))
    @decl@
    int i = 0;
    for (; i + @N*8@ <= len; i += @N*8@) {
;for i in range(N):
        acc@i@ = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+@i*8@), _mm256_loadu_ps(b+i+@i*8@), acc@i@);
;    pass
    }
    // Reduce @N@ accumulators to 1
;step = 1
;while step < N:
;    for i in range(0, N, step * 2):
;        if i + step < N:
    acc@i@ = _mm256_add_ps(acc@i@, acc@i+step@);
;            pass
;        pass
;    step *= 2
#else // LOCUST
    __m256 acc0 = _mm256_setzero_ps(); __m256 acc1 = _mm256_setzero_ps(); __m256 acc2 = _mm256_setzero_ps(); __m256 acc3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= len; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+0), _mm256_loadu_ps(b+i+0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+8), _mm256_loadu_ps(b+i+8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24), acc3);
    }
    // Reduce 4 accumulators to 1
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
#endif // LOCUST
    for (; i + 8 <= len; i += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc0);
    float s = hsum(acc0);
    for (; i < len; ++i)
        s += a[i] * b[i];
    return s;
}

} // namespace nnr::avx2
