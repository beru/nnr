#pragma once
// ARM NEON LRN (Local Response Normalization) kernel.
// Called from kernel/lrn.h.

#ifdef NNR_ARCH_ARM64
#include <arm_neon.h>
#include <cmath>

namespace nnr {

// NEON-vectorized LRN per-channel: processes 4 spatial positions at a time.
// Updates hw to the position after vectorized processing.
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW
inline void lrn_channel_neon(const float* __restrict input, float* __restrict output,
    int nc, int C, int spatial, int c0, int c1, float alpha, float beta, float bias, int& hw)
{
    const float* xc = input + (size_t)nc * spatial;
    float* yc = output + (size_t)nc * spatial;
    int n = nc / C;
    float32x4_t valpha = vdupq_n_f32(alpha);
    float32x4_t vbias = vdupq_n_f32(bias);
    for (; hw + 4 <= spatial; hw += 4) {
        float32x4_t vsum = vdupq_n_f32(0);
        for (int ci = c0; ci <= c1; ++ci) {
            float32x4_t v = vld1q_f32(input + ((size_t)n * C + ci) * spatial + hw);
            vsum = vfmaq_f32(vsum, v, v);
        }
        float32x4_t denom = vfmaq_f32(vbias, valpha, vsum);
        float32x4_t vx = vld1q_f32(xc + hw);
        float d[4], x[4];
        vst1q_f32(d, denom);
        vst1q_f32(x, vx);
        float r[4];
        r[0] = x[0] / powf(d[0], beta);
        r[1] = x[1] / powf(d[1], beta);
        r[2] = x[2] / powf(d[2], beta);
        r[3] = x[3] / powf(d[3], beta);
        vst1q_f32(yc + hw, vld1q_f32(r));
    }
}

} // namespace nnr

#endif // NNR_ARCH_ARM64
