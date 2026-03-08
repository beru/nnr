#pragma once
// ARM NEON helpers for conv kernels.
// Called from kernel/gemm.h via #ifdef NNR_ARCH_ARM64 dispatch.

#ifdef NNR_ARCH_ARM64
#include <arm_neon.h>

namespace nnr {

// Column-wise bias add: data[j] += bias[j] for j in [0, len).
// Returns the position after vectorized processing.
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW
inline int col_bias_add_neon(float* data, const float* bias, int len) {
    int j = 0;
    for (; j + 4 <= len; j += 4)
        vst1q_f32(data + j, vaddq_f32(vld1q_f32(data + j), vld1q_f32(bias + j)));
    return j;
}

// Vectorized a + b -> dst
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW
inline void add_vec_neon(float* dst, const float* a, const float* b, size_t start, size_t end) {
    size_t i = start;
    for (; i + 4 <= end; i += 4)
        vst1q_f32(dst + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < end; ++i) dst[i] = a[i] + b[i];
}

// Fused Add inplace: row[i] += skip[off+i] + bias (NEON vectorized)
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW fusion=binary
inline void add_skip_bias_neon(float* row, const float* skip, int cols, int row_off, float bv) {
    float32x4_t vbv = vdupq_n_f32(bv);
    int i = 0;
    for (; i + 4 <= cols; i += 4)
        vst1q_f32(row + i, vaddq_f32(vld1q_f32(row + i),
            vaddq_f32(vld1q_f32(skip + row_off + i), vbv)));
    for (; i < cols; ++i)
        row[i] += skip[row_off + i] + bv;
}

// Global average pool NHWC: vectorize over channels, accumulate over spatial.
// Returns channel position after vectorized processing.
// @nnr-meta isa=NEON dtype=fp32 layout=NHWC
inline int global_avgpool_nhwc_neon(const float* xn, float* yn, int C, int spatial, float inv) {
    int c = 0;
    float32x4_t vinv = vdupq_n_f32(inv);
    for (; c + 4 <= C; c += 4) {
        float32x4_t acc = vdupq_n_f32(0);
        for (int s = 0; s < spatial; ++s)
            acc = vaddq_f32(acc, vld1q_f32(xn + (size_t)s * C + c));
        vst1q_f32(yn + c, vmulq_f32(acc, vinv));
    }
    return c;
}

// Horizontal sum of contiguous float array (NEON vectorized).
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW
inline float reduce_sum_neon(const float* p, int len) {
    float32x4_t vsum = vdupq_n_f32(0);
    int r = 0;
    for (; r + 4 <= len; r += 4)
        vsum = vaddq_f32(vsum, vld1q_f32(p + r));
    float sum = vaddvq_f32(vsum);
    for (; r < len; r++) sum += p[r];
    return sum;
}

// Reduce over red dimension, vectorize over tail (tail >= 4).
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW
inline void reduce_sum_tail_neon(float* dst, const float* src, int red, int tail) {
    int t = 0;
    for (; t + 4 <= tail; t += 4) {
        float32x4_t vsum = vdupq_n_f32(0);
        for (int r = 0; r < red; r++)
            vsum = vaddq_f32(vsum, vld1q_f32(src + (size_t)r * tail + t));
        vst1q_f32(dst + t, vsum);
    }
    for (; t < tail; t++) {
        float sum = 0;
        for (int r = 0; r < red; r++)
            sum += src[(size_t)r * tail + t];
        dst[t] = sum;
    }
}

// Compute mean and variance of a contiguous float array (NEON vectorized).
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW
inline void compute_mean_var_neon(const float* src, int len, float& mean, float& var) {
    float32x4_t vsum = vdupq_n_f32(0);
    float32x4_t vsum2 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        vsum = vaddq_f32(vsum, v);
        vsum2 = vfmaq_f32(vsum2, v, v);
    }
    float s = vaddvq_f32(vsum), s2 = vaddvq_f32(vsum2);
    for (; i < len; i++) { s += src[i]; s2 += src[i] * src[i]; }
    mean = s / len;
    var = s2 / len - mean * mean;
}

// Apply affine: dst[i] = a * src[i] + b (NEON vectorized).
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW
inline void affine_neon(float* dst, const float* src, int len, float a, float b) {
    float32x4_t va = vdupq_n_f32(a);
    float32x4_t vb = vdupq_n_f32(b);
    int i = 0;
    for (; i + 4 <= len; i += 4)
        vst1q_f32(dst + i, vfmaq_f32(vb, va, vld1q_f32(src + i)));
    for (; i < len; i++)
        dst[i] = a * src[i] + b;
}

} // namespace nnr

#endif // NNR_ARCH_ARM64
