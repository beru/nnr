#pragma once
// AVX-512 vectorized operations: clip, reduction, dot product.
// Included from individual operator .cpp files.

#include <immintrin.h>

namespace nnr::avx512 {

// Vectorized clamp: py[i] = clamp(px[i], minv, maxv)
// @nnr-meta isa=AVX512 dtype=fp32
inline void clip(const float* __restrict px, float* __restrict py, size_t len, float minv, float maxv)
{
    __m512 vmin = _mm512_set1_ps(minv);
    __m512 vmax = _mm512_set1_ps(maxv);
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        __m512 v = _mm512_loadu_ps(px + i);
        v = _mm512_max_ps(v, vmin);
        v = _mm512_min_ps(v, vmax);
        _mm512_storeu_ps(py + i, v);
    }
    for (; i < len; ++i) {
        float v = px[i];
        v = std::max(v, minv);
        v = std::min(v, maxv);
        py[i] = v;
    }
}

// Fused bias-add + clamp: data[i] = clamp(data[i] + bias, minv, maxv)
// @nnr-meta isa=AVX512 dtype=fp32 fusion=post_op
inline void bias_clip(float* __restrict data, int len, float bias, float minv, float maxv)
{
    __m512 vb = _mm512_set1_ps(bias);
    __m512 vmin = _mm512_set1_ps(minv);
    __m512 vmax = _mm512_set1_ps(maxv);
    int i = 0;
    for (; i + 16 <= len; i += 16) {
        __m512 v = _mm512_add_ps(_mm512_loadu_ps(data + i), vb);
        _mm512_storeu_ps(data + i, _mm512_min_ps(_mm512_max_ps(v, vmin), vmax));
    }
    for (; i < len; ++i)
        data[i] = std::clamp(data[i] + bias, minv, maxv);
}

// Fused bias-add + relu: data[i] = max(data[i] + bias, 0)
// @nnr-meta isa=AVX512 dtype=fp32 fusion=post_op
inline void bias_relu(float* __restrict data, int len, float bias)
{
    __m512 vb = _mm512_set1_ps(bias);
    __m512 vz = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= len; i += 16) {
        __m512 v = _mm512_add_ps(_mm512_loadu_ps(data + i), vb);
        _mm512_storeu_ps(data + i, _mm512_max_ps(v, vz));
    }
    for (; i < len; ++i)
        data[i] = std::max(data[i] + bias, 0.0f);
}

// Bias-add only: data[i] += bias
// @nnr-meta isa=AVX512 dtype=fp32 fusion=post_op
inline void bias_add(float* __restrict data, int len, float bias)
{
    if (bias == 0.0f) return;
    __m512 vb = _mm512_set1_ps(bias);
    int i = 0;
    for (; i + 16 <= len; i += 16)
        _mm512_storeu_ps(data + i, _mm512_add_ps(_mm512_loadu_ps(data + i), vb));
    for (; i < len; ++i)
        data[i] += bias;
}

// Fused binary Add: data[i] += skip[i] + bias. The hot inner loop for
// Conv+Add(+Relu) residual fusion (ResNet/MobileNet/EfficientNet bottlenecks).
// @nnr-meta isa=AVX512 dtype=fp32 fusion=binary
inline void add_skip_bias(float* __restrict data, const float* __restrict skip,
                          int len, float bias)
{
    __m512 vb = _mm512_set1_ps(bias);
    int i = 0;
    if (bias == 0.0f) {
        for (; i + 16 <= len; i += 16)
            _mm512_storeu_ps(data + i, _mm512_add_ps(_mm512_loadu_ps(data + i),
                                                    _mm512_loadu_ps(skip + i)));
        for (; i < len; ++i)
            data[i] += skip[i];
    } else {
        for (; i + 16 <= len; i += 16) {
            __m512 d = _mm512_loadu_ps(data + i);
            __m512 s = _mm512_loadu_ps(skip + i);
            _mm512_storeu_ps(data + i, _mm512_add_ps(d, _mm512_add_ps(s, vb)));
        }
        for (; i < len; ++i)
            data[i] += skip[i] + bias;
    }
}

// HardSwish: data[i] = (data[i] + bias) * clamp((data[i] + bias) / 6 + 0.5, 0, 1)
// @nnr-meta isa=AVX512 dtype=fp32 fusion=post_op
inline void bias_hardswish(float* __restrict data, int len, float bias)
{
    __m512 vb   = _mm512_set1_ps(bias);
    __m512 v6   = _mm512_set1_ps(1.0f / 6.0f);
    __m512 vhalf = _mm512_set1_ps(0.5f);
    __m512 vzero = _mm512_setzero_ps();
    __m512 vone  = _mm512_set1_ps(1.0f);
    int i = 0;
    for (; i + 16 <= len; i += 16) {
        __m512 x = _mm512_add_ps(_mm512_loadu_ps(data + i), vb);
        __m512 t = _mm512_min_ps(_mm512_max_ps(
            _mm512_add_ps(_mm512_mul_ps(x, v6), vhalf), vzero), vone);
        _mm512_storeu_ps(data + i, _mm512_mul_ps(x, t));
    }
    for (; i < len; ++i) {
        float x = data[i] + bias;
        data[i] = x * std::max(0.0f, std::min(1.0f, x / 6.0f + 0.5f));
    }
}

// HardSwish (no bias): py[i] = px[i] * clamp(px[i] / 6 + 0.5, 0, 1)
// @nnr-meta isa=AVX512 dtype=fp32
inline void hardswish(const float* __restrict px, float* __restrict py, size_t len)
{
    __m512 v6   = _mm512_set1_ps(1.0f / 6.0f);
    __m512 vhalf = _mm512_set1_ps(0.5f);
    __m512 vzero = _mm512_setzero_ps();
    __m512 vone  = _mm512_set1_ps(1.0f);
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        __m512 x = _mm512_loadu_ps(px + i);
        __m512 t = _mm512_min_ps(_mm512_max_ps(
            _mm512_add_ps(_mm512_mul_ps(x, v6), vhalf), vzero), vone);
        _mm512_storeu_ps(py + i, _mm512_mul_ps(x, t));
    }
    for (; i < len; ++i) {
        float x = px[i];
        py[i] = x * std::max(0.0f, std::min(1.0f, x / 6.0f + 0.5f));
    }
}

// Vectorized leaky relu: py[i] = px[i] >= 0 ? px[i] : px[i] * alpha
// @nnr-meta isa=AVX512 dtype=fp32
inline void leaky_relu(const float* __restrict px, float* __restrict py, int len, float alpha)
{
    __m512 va = _mm512_set1_ps(alpha);
    __m512 vz = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= len; i += 16) {
        __m512 v = _mm512_loadu_ps(px + i);
        __mmask16 mask = _mm512_cmp_ps_mask(v, vz, _CMP_LT_OS);
        __m512 neg = _mm512_mul_ps(v, va);
        _mm512_storeu_ps(py + i, _mm512_mask_blend_ps(mask, v, neg));
    }
    for (; i < len; ++i)
        py[i] = px[i] >= 0 ? px[i] : px[i] * alpha;
}

// Fused bias-add + leaky relu: data[i] = (data[i]+bias) >= 0 ? (data[i]+bias) : (data[i]+bias)*alpha
// @nnr-meta isa=AVX512 dtype=fp32 fusion=post_op
inline void bias_leaky_relu(float* __restrict data, int len, float bias, float alpha)
{
    __m512 vb = _mm512_set1_ps(bias);
    __m512 va = _mm512_set1_ps(alpha);
    __m512 vz = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= len; i += 16) {
        __m512 v = _mm512_add_ps(_mm512_loadu_ps(data + i), vb);
        __mmask16 mask = _mm512_cmp_ps_mask(v, vz, _CMP_LT_OS);
        __m512 neg = _mm512_mul_ps(v, va);
        _mm512_storeu_ps(data + i, _mm512_mask_blend_ps(mask, v, neg));
    }
    for (; i < len; ++i) {
        float v = data[i] + bias;
        data[i] = v >= 0 ? v : v * alpha;
    }
}

// Vectorized horizontal sum of a float array.
// @nnr-meta isa=AVX512 dtype=fp32
inline float reduce_sum(const float* data, int len)
{
#if 0 // LOCUST
;N = 4
;STRIDE = 16 * N
;for j in range(N):
    __m512 s@j@ = _mm512_setzero_ps();
;    pass
    int i = 0;
    for (; i + @STRIDE@ <= len; i += @STRIDE@) {
;for j in range(N):
;    off = 16 * j
        s@j@ = _mm512_add_ps(s@j@, _mm512_loadu_ps(data + i + @off@));
;    pass
    }
    s0 = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
#else // LOCUST
    __m512 s0 = _mm512_setzero_ps();
    __m512 s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps();
    __m512 s3 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 64 <= len; i += 64) {
        s0 = _mm512_add_ps(s0, _mm512_loadu_ps(data + i + 0));
        s1 = _mm512_add_ps(s1, _mm512_loadu_ps(data + i + 16));
        s2 = _mm512_add_ps(s2, _mm512_loadu_ps(data + i + 32));
        s3 = _mm512_add_ps(s3, _mm512_loadu_ps(data + i + 48));
    }
    s0 = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
#endif // LOCUST
    for (; i + 16 <= len; i += 16)
        s0 = _mm512_add_ps(s0, _mm512_loadu_ps(data + i));
    float total = _mm512_reduce_add_ps(s0);
    for (; i < len; i++)
        total += data[i];
    return total;
}

// Vectorized dot product with 4 independent accumulators to hide FMA latency.
// @nnr-meta isa=AVX512 dtype=fp32
inline float dot_product(const float* __restrict a, const float* __restrict b, int len)
{
#if 0 // LOCUST
;for j in range(N):
    __m512 acc@j@ = _mm512_setzero_ps();
;    pass
    int i = 0;
    for (; i + @STRIDE@ <= len; i += @STRIDE@) {
;for j in range(N):
;    off = 16 * j
        acc@j@ = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+@off@), _mm512_loadu_ps(b+i+@off@), acc@j@);
;    pass
    }
    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc0 = _mm512_add_ps(acc0, acc2);
#else // LOCUST
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 64 <= len; i += 64) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+0), _mm512_loadu_ps(b+i+0), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+16), _mm512_loadu_ps(b+i+16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+32), _mm512_loadu_ps(b+i+32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+48), _mm512_loadu_ps(b+i+48), acc3);
    }
    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc0 = _mm512_add_ps(acc0, acc2);
#endif // LOCUST
    for (; i + 16 <= len; i += 16)
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i), _mm512_loadu_ps(b+i), acc0);
    float s = _mm512_reduce_add_ps(acc0);
    for (; i < len; ++i)
        s += a[i] * b[i];
    return s;
}

} // namespace nnr::avx512
