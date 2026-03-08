#pragma once
// ARM NEON layout conversion helpers.
// Called from kernel/layout.h.

#ifdef NNR_ARCH_ARM64
#include <arm_neon.h>

namespace nnr {

// 4x4 float32 transpose using NEON.
// Reads 4 rows of 4 from src (stride = src_stride), writes 4 rows of 4 to dst (stride = dst_stride).
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW
inline void transpose_4x4_neon(
    const float* src, int src_stride,
    float* dst, int dst_stride)
{
    float32x4_t r0 = vld1q_f32(src);
    float32x4_t r1 = vld1q_f32(src + src_stride);
    float32x4_t r2 = vld1q_f32(src + 2 * src_stride);
    float32x4_t r3 = vld1q_f32(src + 3 * src_stride);

    float32x4_t t0 = vtrn1q_f32(r0, r1);
    float32x4_t t1 = vtrn2q_f32(r0, r1);
    float32x4_t t2 = vtrn1q_f32(r2, r3);
    float32x4_t t3 = vtrn2q_f32(r2, r3);

    float64x2_t d0 = vtrn1q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2));
    float64x2_t d1 = vtrn1q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3));
    float64x2_t d2 = vtrn2q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2));
    float64x2_t d3 = vtrn2q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3));

    vst1q_f32(dst,                  vreinterpretq_f32_f64(d0));
    vst1q_f32(dst + dst_stride,     vreinterpretq_f32_f64(d1));
    vst1q_f32(dst + 2 * dst_stride, vreinterpretq_f32_f64(d2));
    vst1q_f32(dst + 3 * dst_stride, vreinterpretq_f32_f64(d3));
}

// Transpose batch [C x HW] -> [HW x C] using 4x4 NEON blocks.
// src_n: [C][HW], dst_n: [HW][C], processes hw_begin..hw_end range.
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW
inline void nchw_nhwc_batch_neon(const float* src_n, float* dst_n,
    int C, int HW, int hw_begin, int hw_end)
{
    const int hw_range = hw_end - hw_begin;
    if (C >= 4 && hw_range >= 4) {
        int hw, c;
        for (hw = 0; hw + 4 <= hw_range; hw += 4) {
            for (c = 0; c + 4 <= C; c += 4) {
                transpose_4x4_neon(
                    src_n + (size_t)c * HW + hw_begin + hw, HW,
                    dst_n + (size_t)(hw_begin + hw) * C + c, C);
            }
            for (; c < C; c++) {
                const float* sc = src_n + (size_t)c * HW + hw_begin + hw;
                for (int h2 = 0; h2 < 4; h2++)
                    dst_n[(hw_begin + hw + h2) * C + c] = sc[h2];
            }
        }
        for (; hw < hw_range; hw++) {
            for (c = 0; c < C; c++)
                dst_n[(hw_begin + hw) * C + c] = src_n[(size_t)c * HW + hw_begin + hw];
        }
    } else {
        for (int c = 0; c < C; c++) {
            const float* sc = src_n + (size_t)c * HW + hw_begin;
            for (int hw = 0; hw < hw_range; hw++)
                dst_n[(hw_begin + hw) * C + c] = sc[hw];
        }
    }
}

// Transpose batch [HW x C] -> [C x HW] using 4x4 NEON blocks.
// @nnr-meta isa=NEON dtype=fp32 layout=NHWC
inline void nhwc_nchw_batch_neon(const float* src_n, float* dst_n,
    int C, int HW, int hw_begin, int hw_end)
{
    const int hw_range = hw_end - hw_begin;
    if (C >= 4 && hw_range >= 4) {
        int hw, c;
        for (hw = 0; hw + 4 <= hw_range; hw += 4) {
            for (c = 0; c + 4 <= C; c += 4) {
                transpose_4x4_neon(
                    src_n + (size_t)(hw_begin + hw) * C + c, C,
                    dst_n + (size_t)c * HW + hw_begin + hw, HW);
            }
            for (; c < C; c++) {
                float* dc = dst_n + (size_t)c * HW + hw_begin + hw;
                for (int h2 = 0; h2 < 4; h2++)
                    dc[h2] = src_n[(hw_begin + hw + h2) * C + c];
            }
        }
        for (; hw < hw_range; hw++) {
            for (c = 0; c < C; c++)
                dst_n[(size_t)c * HW + hw_begin + hw] = src_n[(hw_begin + hw) * C + c];
        }
    } else {
        for (int c = 0; c < C; c++) {
            float* dc = dst_n + (size_t)c * HW + hw_begin;
            for (int hw = 0; hw < hw_range; hw++)
                dc[hw] = src_n[(hw_begin + hw) * C + c];
        }
    }
}

// Block alignment for spatial tiling.
// @nnr-meta isa=scalar dtype=fp32
inline int layout_block_align_neon() { return 4; }

} // namespace nnr

#endif // NNR_ARCH_ARM64
