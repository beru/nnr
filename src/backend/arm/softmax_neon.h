#pragma once
// ARM NEON softmax row kernel.
// Uses polynomial exp approximation from simd_math_neon.h.

#ifdef NNR_ARCH_ARM64
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include "backend/arm/simd_math_neon.h"

namespace nnr {

// NEON softmax for one row: find max, compute exp(x-max), normalize.
// @nnr-meta isa=NEON dtype=fp32
inline void softmax_row_neon(const float* row, float* out, int len) {
    // 1. Find max
    float32x4_t vmax = vdupq_n_f32(-1e30f);
    int j = 0;
    for (; j + 4 <= len; j += 4)
        vmax = vmaxq_f32(vmax, vld1q_f32(row + j));
    float maxv = vmaxvq_f32(vmax);
    for (; j < len; j++)
        maxv = std::max(maxv, row[j]);

    // 2. exp(x - max) and sum
    float32x4_t vsum = vdupq_n_f32(0);
    float32x4_t vmx = vdupq_n_f32(maxv);
    j = 0;
    for (; j + 4 <= len; j += 4) {
        float32x4_t v = vsubq_f32(vld1q_f32(row + j), vmx);
        v = exp_neon_ps(v);
        vst1q_f32(out + j, v);
        vsum = vaddq_f32(vsum, v);
    }
    float sum = vaddvq_f32(vsum);
    for (; j < len; j++) {
        out[j] = expf(row[j] - maxv);
        sum += out[j];
    }

    // 3. Normalize
    if (sum != 0) {
        float inv = 1.0f / sum;
        float32x4_t vinv = vdupq_n_f32(inv);
        j = 0;
        for (; j + 4 <= len; j += 4)
            vst1q_f32(out + j, vmulq_f32(vld1q_f32(out + j), vinv));
        for (; j < len; j++)
            out[j] *= inv;
    }
}

} // namespace nnr

#endif // NNR_ARCH_ARM64
