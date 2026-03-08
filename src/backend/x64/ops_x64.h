#pragma once
// x64 AVX-512 fast paths for operator exec() methods.
// Small vectorized loops extracted from operator .cpp files.

#ifdef NNR_ARCH_X64
#include <immintrin.h>
#include <cmath>

namespace nnr {

// --- ReduceSum ---

// Horizontal sum of contiguous float array (vectorized).
// @nnr-meta isa=AVX512 dtype=fp32
inline float reduce_sum_avx512(const float* p, int len) {
    __m512 vsum = _mm512_setzero_ps();
    int r = 0;
    for (; r + 16 <= len; r += 16)
        vsum = _mm512_add_ps(vsum, _mm512_loadu_ps(p + r));
    float sum = _mm512_reduce_add_ps(vsum);
    for (; r < len; r++) sum += p[r];
    return sum;
}

// Reduce over red dimension, vectorize over tail (tail >= 16).
// @nnr-meta isa=AVX512 dtype=fp32
inline void reduce_sum_tail_avx512(float* dst, const float* src, int red, int tail) {
    int t = 0;
    for (; t + 16 <= tail; t += 16) {
        __m512 vsum = _mm512_setzero_ps();
        for (int r = 0; r < red; r++)
            vsum = _mm512_add_ps(vsum, _mm512_loadu_ps(src + (size_t)r * tail + t));
        _mm512_storeu_ps(dst + t, vsum);
    }
    for (; t < tail; t++) {
        float sum = 0;
        for (int r = 0; r < red; r++)
            sum += src[(size_t)r * tail + t];
        dst[t] = sum;
    }
}

// --- Softmax ---
// --- InstanceNormalization ---

// AVX-512F horizontal reduction (avoid _mm512_reduce_add_ps which needs AVX-512DQ).
// @nnr-meta isa=AVX512 dtype=fp32
inline float hsum512f(__m512 v) {
    __m128 a = _mm512_castps512_ps128(v);
    __m128 b = _mm512_extractf32x4_ps(v, 1);
    __m128 c = _mm512_extractf32x4_ps(v, 2);
    __m128 d = _mm512_extractf32x4_ps(v, 3);
    __m128 s = _mm_add_ps(_mm_add_ps(a, b), _mm_add_ps(c, d));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

// Compute mean and variance of a contiguous float array (vectorized).
// @nnr-meta isa=AVX512 dtype=fp32
inline void compute_mean_var_avx512(const float* src, int len, float& mean, float& var) {
    __m512 vsum = _mm512_setzero_ps();
    __m512 vsum2 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= len; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        vsum = _mm512_add_ps(vsum, v);
        vsum2 = _mm512_fmadd_ps(v, v, vsum2);
    }
    float s = hsum512f(vsum), s2 = hsum512f(vsum2);
    for (; i < len; i++) { s += src[i]; s2 += src[i] * src[i]; }
    mean = s / len;
    var = s2 / len - mean * mean;
}

// Apply affine: dst[i] = a * src[i] + b (vectorized).
// @nnr-meta isa=AVX512 dtype=fp32
inline void affine_avx512(float* dst, const float* src, int len, float a, float b) {
    __m512 va = _mm512_set1_ps(a);
    __m512 vb = _mm512_set1_ps(b);
    int i = 0;
    for (; i + 16 <= len; i += 16)
        _mm512_storeu_ps(dst + i, _mm512_fmadd_ps(va, _mm512_loadu_ps(src + i), vb));
    for (; i < len; i++)
        dst[i] = a * src[i] + b;
}

// --- BatchNormalization ---

// Vectorized channel affine: dst[c] = alpha[c] * src[c] + beta[c], 16 channels at a time.
// @nnr-meta isa=AVX512 dtype=fp32 fusion=bn
inline void channel_affine_avx512(float* dst, const float* src,
    const float* alpha, const float* beta, int C)
{
    int c = 0;
    for (; c + 16 <= C; c += 16) {
        __m512 va = _mm512_loadu_ps(alpha + c);
        __m512 vb = _mm512_loadu_ps(beta + c);
        __m512 vx = _mm512_loadu_ps(src + c);
        _mm512_storeu_ps(dst + c, _mm512_fmadd_ps(va, vx, vb));
    }
    for (; c < C; c++)
        dst[c] = alpha[c] * src[c] + beta[c];
}

// Uint8 NHWC channel affine: dst[c] = clamp(round(alpha[c] * src[c] + beta[c]), 0, 255),
// 16 channels at a time. alpha/beta are per-channel fp32; src/dst are uint8.
// Used by the fused DQ→BN→Q NHWC path.
// @nnr-meta isa=AVX512 dtype=[uint8,fp32] layout=NHWC fusion=[bn,qdq]
inline void channel_affine_u8_avx512(uint8_t* dst, const uint8_t* src,
    const float* alpha, const float* beta, int C)
{
    int c = 0;
    for (; c + 16 <= C; c += 16) {
        __m128i xu8 = _mm_loadu_si128((const __m128i*)(src + c));
        __m512i xi32 = _mm512_cvtepu8_epi32(xu8);
        __m512 xf = _mm512_cvtepi32_ps(xi32);
        __m512 va = _mm512_loadu_ps(alpha + c);
        __m512 vb = _mm512_loadu_ps(beta + c);
        xf = _mm512_fmadd_ps(xf, va, vb);
        __m512i yi32 = _mm512_cvtps_epi32(xf);  // round-to-nearest
        _mm_storeu_si128((__m128i*)(dst + c), _mm512_cvtusepi32_epi8(yi32));
    }
    for (; c < C; c++) {
        float v = alpha[c] * (float)src[c] + beta[c];
        v = std::max(0.0f, std::min(255.0f, v));
        dst[c] = (uint8_t)std::lrintf(v);
    }
}

// Fp32→uint8 NHWC channel affine: dst[c] = clamp(round(alpha[c] * src[c] + beta[c]), 0, 255),
// 16 channels at a time. Used by the fused (Concat→)BN→Q NHWC path where BN
// input stays fp32 but the fused Q output is uint8.
// @nnr-meta isa=AVX512 dtype=[fp32,uint8] layout=NHWC fusion=[bn,qdq]
inline void channel_affine_f32_u8_avx512(uint8_t* dst, const float* src,
    const float* alpha, const float* beta, int C)
{
    int c = 0;
    __m512 v0 = _mm512_setzero_ps();
    __m512 v255 = _mm512_set1_ps(255.0f);
    for (; c + 16 <= C; c += 16) {
        __m512 xf = _mm512_loadu_ps(src + c);
        __m512 va = _mm512_loadu_ps(alpha + c);
        __m512 vb = _mm512_loadu_ps(beta + c);
        xf = _mm512_fmadd_ps(xf, va, vb);
        xf = _mm512_max_ps(_mm512_min_ps(xf, v255), v0);
        __m512i yi32 = _mm512_cvtps_epi32(xf);
        _mm_storeu_si128((__m128i*)(dst + c), _mm512_cvtusepi32_epi8(yi32));
    }
    for (; c < C; c++) {
        float v = alpha[c] * src[c] + beta[c];
        v = std::max(0.0f, std::min(255.0f, v));
        dst[c] = (uint8_t)std::lrintf(v);
    }
}

// --- Elementwise Add (threaded) ---

// Vectorized a + b -> dst
// @nnr-meta isa=AVX512 dtype=fp32
inline void add_vec_avx512(float* dst, const float* a, const float* b, size_t start, size_t end) {
    size_t i = start;
    for (; i + 16 <= end; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        _mm512_storeu_ps(dst + i, _mm512_add_ps(va, vb));
    }
    for (; i < end; ++i) dst[i] = a[i] + b[i];
}

// --- Bias broadcast Add: dst[r*N+i] = src[r*N+i] + bias[i] ---

// @nnr-meta isa=AVX512 dtype=fp32 fusion=post_op
inline void add_bias_broadcast_avx512(float* dst, const float* src, const float* bias,
                                       int rows, int N) {
    for (int r = 0; r < rows; r++) {
        const float* s = src + (size_t)r * N;
        float* d = dst + (size_t)r * N;
        int i = 0;
        for (; i + 16 <= N; i += 16)
            _mm512_storeu_ps(d + i, _mm512_add_ps(
                _mm512_loadu_ps(s + i), _mm512_loadu_ps(bias + i)));
        if (i < N) {
            __mmask16 mask = (__mmask16)((1u << (N - i)) - 1);
            _mm512_mask_storeu_ps(d + i, mask, _mm512_add_ps(
                _mm512_maskz_loadu_ps(mask, s + i),
                _mm512_maskz_loadu_ps(mask, bias + i)));
        }
    }
}

// --- Global average pool (NHWC) ---

// Vectorized accumulate over spatial, per-channel output.
// Returns channel position after vectorized processing.
// @nnr-meta isa=[AVX512,AVX2] dtype=fp32 layout=NHWC
inline int global_avgpool_nhwc_x64(const float* xn, float* yn, int C, int spatial, float inv) {
    int c = 0;
    if (has_avx512()) {
        for (; c + 16 <= C; c += 16) {
            __m512 acc = _mm512_setzero_ps();
            for (int s = 0; s < spatial; ++s)
                acc = _mm512_add_ps(acc, _mm512_loadu_ps(xn + (size_t)s * C + c));
            _mm512_storeu_ps(yn + c, _mm512_mul_ps(acc, _mm512_set1_ps(inv)));
        }
    } else if (detect_isa() >= isa_t::avx2) {
        for (; c + 8 <= C; c += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int s = 0; s < spatial; ++s)
                acc = _mm256_add_ps(acc, _mm256_loadu_ps(xn + (size_t)s * C + c));
            _mm256_storeu_ps(yn + c, _mm256_mul_ps(acc, _mm256_set1_ps(inv)));
        }
    }
    return c;
}

// --- LayerNormalization: single-row AVX-512 kernel ---
// 3-pass: mean → variance → normalize+scale+bias (all SIMD).

// @nnr-meta isa=AVX512 dtype=fp32
inline void layer_norm_row_avx512(const float* row, float* out,
                                   const float* scale, const float* bias,
                                   int inner, float eps) {
    // Pass 1: mean
    __m512 vsum = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= inner; i += 16)
        vsum = _mm512_add_ps(vsum, _mm512_loadu_ps(row + i));
    float sum = _mm512_reduce_add_ps(vsum);
    for (; i < inner; i++) sum += row[i];
    float mean = sum / inner;
    __m512 vmean = _mm512_set1_ps(mean);

    // Pass 2: variance
    __m512 vvar = _mm512_setzero_ps();
    i = 0;
    for (; i + 16 <= inner; i += 16) {
        __m512 d = _mm512_sub_ps(_mm512_loadu_ps(row + i), vmean);
        vvar = _mm512_fmadd_ps(d, d, vvar);
    }
    float var = _mm512_reduce_add_ps(vvar);
    for (; i < inner; i++) { float d = row[i] - mean; var += d * d; }
    float invstd = 1.0f / sqrtf(var / inner + eps);
    __m512 vinvstd = _mm512_set1_ps(invstd);

    // Pass 3: normalize + scale + bias
    i = 0;
    if (bias) {
        for (; i + 16 <= inner; i += 16) {
            __m512 norm = _mm512_mul_ps(
                _mm512_sub_ps(_mm512_loadu_ps(row + i), vmean), vinvstd);
            _mm512_storeu_ps(out + i,
                _mm512_fmadd_ps(norm, _mm512_loadu_ps(scale + i),
                                _mm512_loadu_ps(bias + i)));
        }
        if (i < inner) {
            __mmask16 mask = (__mmask16)((1u << (inner - i)) - 1);
            __m512 norm = _mm512_mul_ps(
                _mm512_sub_ps(_mm512_maskz_loadu_ps(mask, row + i), vmean), vinvstd);
            _mm512_mask_storeu_ps(out + i, mask,
                _mm512_fmadd_ps(norm, _mm512_maskz_loadu_ps(mask, scale + i),
                                _mm512_maskz_loadu_ps(mask, bias + i)));
        }
    } else {
        for (; i + 16 <= inner; i += 16) {
            __m512 norm = _mm512_mul_ps(
                _mm512_sub_ps(_mm512_loadu_ps(row + i), vmean), vinvstd);
            _mm512_storeu_ps(out + i, _mm512_mul_ps(norm, _mm512_loadu_ps(scale + i)));
        }
        if (i < inner) {
            __mmask16 mask = (__mmask16)((1u << (inner - i)) - 1);
            __m512 norm = _mm512_mul_ps(
                _mm512_sub_ps(_mm512_maskz_loadu_ps(mask, row + i), vmean), vinvstd);
            _mm512_mask_storeu_ps(out + i, mask,
                _mm512_mul_ps(norm, _mm512_maskz_loadu_ps(mask, scale + i)));
        }
    }
}

} // namespace nnr

#endif // NNR_ARCH_X64
