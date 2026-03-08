#pragma once
// ARM NEON GEMM micro-kernels: MR=8, NR=8.
// 16 accumulators + 2 A + 2 B = 20 of 32 registers.

#if defined(__aarch64__) || defined(_M_ARM64)

#ifdef _MSC_VER
#define NNR_FORCEINLINE __forceinline
#else
#define NNR_FORCEINLINE inline __attribute__((always_inline))
#endif

namespace nnr { namespace neon {

static constexpr int UK_MR = 8;
static constexpr int UK_NR = 8;

// NCHW micro-kernel: 8 rows x 8 cols, 16 FMA per K step.
// Registers: 16 accumulators + 2 B + 2 A = 20 of 32.
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=GEMM tiling=[MR,NR] fusion=post_op
NNR_FORCEINLINE void ukernel_nchw(
    int kc,
    const float* __restrict pa,
    const float* __restrict pb,
    int pb_stride,
    float* const __restrict pc[8],
    int v,
    bool zero_init,
    bool do_fuse,
    const float* bp,
    float fmin, float fmax)
{
    float32x4_t c0_0, c0_1;
    float32x4_t c1_0, c1_1;
    float32x4_t c2_0, c2_1;
    float32x4_t c3_0, c3_1;
    float32x4_t c4_0, c4_1;
    float32x4_t c5_0, c5_1;
    float32x4_t c6_0, c6_1;
    float32x4_t c7_0, c7_1;
    if (zero_init) {
        c0_0 = c0_1 = vdupq_n_f32(0.0f);
        c1_0 = c1_1 = vdupq_n_f32(0.0f);
        c2_0 = c2_1 = vdupq_n_f32(0.0f);
        c3_0 = c3_1 = vdupq_n_f32(0.0f);
        c4_0 = c4_1 = vdupq_n_f32(0.0f);
        c5_0 = c5_1 = vdupq_n_f32(0.0f);
        c6_0 = c6_1 = vdupq_n_f32(0.0f);
        c7_0 = c7_1 = vdupq_n_f32(0.0f);
    } else {
        c0_0 = vld1q_f32(pc[0] + v); c0_1 = vld1q_f32(pc[0] + v + 4);
        c1_0 = vld1q_f32(pc[1] + v); c1_1 = vld1q_f32(pc[1] + v + 4);
        c2_0 = vld1q_f32(pc[2] + v); c2_1 = vld1q_f32(pc[2] + v + 4);
        c3_0 = vld1q_f32(pc[3] + v); c3_1 = vld1q_f32(pc[3] + v + 4);
        c4_0 = vld1q_f32(pc[4] + v); c4_1 = vld1q_f32(pc[4] + v + 4);
        c5_0 = vld1q_f32(pc[5] + v); c5_1 = vld1q_f32(pc[5] + v + 4);
        c6_0 = vld1q_f32(pc[6] + v); c6_1 = vld1q_f32(pc[6] + v + 4);
        c7_0 = vld1q_f32(pc[7] + v); c7_1 = vld1q_f32(pc[7] + v + 4);
    }
    for (int k = 0; k < kc; k++) {
        float32x4_t b0 = vld1q_f32(pb + (size_t)k * pb_stride);
        float32x4_t b1 = vld1q_f32(pb + (size_t)k * pb_stride + 4);
        float32x4_t a0 = vld1q_f32(pa + k * 8);
        float32x4_t a1 = vld1q_f32(pa + k * 8 + 4);
        c0_0 = vfmaq_laneq_f32(c0_0, b0, a0, 0);
        c0_1 = vfmaq_laneq_f32(c0_1, b1, a0, 0);
        c1_0 = vfmaq_laneq_f32(c1_0, b0, a0, 1);
        c1_1 = vfmaq_laneq_f32(c1_1, b1, a0, 1);
        c2_0 = vfmaq_laneq_f32(c2_0, b0, a0, 2);
        c2_1 = vfmaq_laneq_f32(c2_1, b1, a0, 2);
        c3_0 = vfmaq_laneq_f32(c3_0, b0, a0, 3);
        c3_1 = vfmaq_laneq_f32(c3_1, b1, a0, 3);
        c4_0 = vfmaq_laneq_f32(c4_0, b0, a1, 0);
        c4_1 = vfmaq_laneq_f32(c4_1, b1, a1, 0);
        c5_0 = vfmaq_laneq_f32(c5_0, b0, a1, 1);
        c5_1 = vfmaq_laneq_f32(c5_1, b1, a1, 1);
        c6_0 = vfmaq_laneq_f32(c6_0, b0, a1, 2);
        c6_1 = vfmaq_laneq_f32(c6_1, b1, a1, 2);
        c7_0 = vfmaq_laneq_f32(c7_0, b0, a1, 3);
        c7_1 = vfmaq_laneq_f32(c7_1, b1, a1, 3);
    }
    if (do_fuse) {
        float32x4_t vmin = vdupq_n_f32(fmin);
        float32x4_t vmax = vdupq_n_f32(fmax);
        float32x4_t vb;
        vb = vdupq_n_f32(bp[0]);
        c0_0 = vaddq_f32(c0_0, vb); c0_1 = vaddq_f32(c0_1, vb);
        c0_0 = vmaxq_f32(c0_0, vmin); c0_0 = vminq_f32(c0_0, vmax); c0_1 = vmaxq_f32(c0_1, vmin); c0_1 = vminq_f32(c0_1, vmax);
        vb = vdupq_n_f32(bp[1]);
        c1_0 = vaddq_f32(c1_0, vb); c1_1 = vaddq_f32(c1_1, vb);
        c1_0 = vmaxq_f32(c1_0, vmin); c1_0 = vminq_f32(c1_0, vmax); c1_1 = vmaxq_f32(c1_1, vmin); c1_1 = vminq_f32(c1_1, vmax);
        vb = vdupq_n_f32(bp[2]);
        c2_0 = vaddq_f32(c2_0, vb); c2_1 = vaddq_f32(c2_1, vb);
        c2_0 = vmaxq_f32(c2_0, vmin); c2_0 = vminq_f32(c2_0, vmax); c2_1 = vmaxq_f32(c2_1, vmin); c2_1 = vminq_f32(c2_1, vmax);
        vb = vdupq_n_f32(bp[3]);
        c3_0 = vaddq_f32(c3_0, vb); c3_1 = vaddq_f32(c3_1, vb);
        c3_0 = vmaxq_f32(c3_0, vmin); c3_0 = vminq_f32(c3_0, vmax); c3_1 = vmaxq_f32(c3_1, vmin); c3_1 = vminq_f32(c3_1, vmax);
        vb = vdupq_n_f32(bp[4]);
        c4_0 = vaddq_f32(c4_0, vb); c4_1 = vaddq_f32(c4_1, vb);
        c4_0 = vmaxq_f32(c4_0, vmin); c4_0 = vminq_f32(c4_0, vmax); c4_1 = vmaxq_f32(c4_1, vmin); c4_1 = vminq_f32(c4_1, vmax);
        vb = vdupq_n_f32(bp[5]);
        c5_0 = vaddq_f32(c5_0, vb); c5_1 = vaddq_f32(c5_1, vb);
        c5_0 = vmaxq_f32(c5_0, vmin); c5_0 = vminq_f32(c5_0, vmax); c5_1 = vmaxq_f32(c5_1, vmin); c5_1 = vminq_f32(c5_1, vmax);
        vb = vdupq_n_f32(bp[6]);
        c6_0 = vaddq_f32(c6_0, vb); c6_1 = vaddq_f32(c6_1, vb);
        c6_0 = vmaxq_f32(c6_0, vmin); c6_0 = vminq_f32(c6_0, vmax); c6_1 = vmaxq_f32(c6_1, vmin); c6_1 = vminq_f32(c6_1, vmax);
        vb = vdupq_n_f32(bp[7]);
        c7_0 = vaddq_f32(c7_0, vb); c7_1 = vaddq_f32(c7_1, vb);
        c7_0 = vmaxq_f32(c7_0, vmin); c7_0 = vminq_f32(c7_0, vmax); c7_1 = vmaxq_f32(c7_1, vmin); c7_1 = vminq_f32(c7_1, vmax);
    }
    vst1q_f32(pc[0] + v, c0_0); vst1q_f32(pc[0] + v + 4, c0_1);
    vst1q_f32(pc[1] + v, c1_0); vst1q_f32(pc[1] + v + 4, c1_1);
    vst1q_f32(pc[2] + v, c2_0); vst1q_f32(pc[2] + v + 4, c2_1);
    vst1q_f32(pc[3] + v, c3_0); vst1q_f32(pc[3] + v + 4, c3_1);
    vst1q_f32(pc[4] + v, c4_0); vst1q_f32(pc[4] + v + 4, c4_1);
    vst1q_f32(pc[5] + v, c5_0); vst1q_f32(pc[5] + v + 4, c5_1);
    vst1q_f32(pc[6] + v, c6_0); vst1q_f32(pc[6] + v + 4, c6_1);
    vst1q_f32(pc[7] + v, c7_0); vst1q_f32(pc[7] + v + 4, c7_1);
}

// NHWC micro-kernel: 1 row x 8 cols
// @nnr-meta isa=NEON dtype=fp32 layout=NHWC special=GEMM tiling=NR fusion=post_op
NNR_FORCEINLINE void ukernel_nhwc(
    int kc,
    const float* __restrict pa,
    const float* __restrict pb,
    int pb_stride,
    float* __restrict pc,
    int v,
    bool zero_init,
    bool do_fuse,
    const float* bias_col,
    float fmin, float fmax)
{
    float32x4_t a0, a1;
    if (zero_init) {
        a0 = a1 = vdupq_n_f32(0.0f);
    } else {
        a0 = vld1q_f32(pc + v);
        a1 = vld1q_f32(pc + v + 4);
    }
    for (int k = 0; k < kc; ++k) {
        float32x4_t av = vdupq_n_f32(pa[k]);
        a0 = vfmaq_f32(a0, av, vld1q_f32(pb + (size_t)k * pb_stride));
        a1 = vfmaq_f32(a1, av, vld1q_f32(pb + (size_t)k * pb_stride + 4));
    }
    if (do_fuse) {
        float32x4_t vmin = vdupq_n_f32(fmin);
        float32x4_t vmax = vdupq_n_f32(fmax);
        a0 = vaddq_f32(a0, vld1q_f32(bias_col + v));
        a1 = vaddq_f32(a1, vld1q_f32(bias_col + v + 4));
        a0 = vmaxq_f32(a0, vmin); a0 = vminq_f32(a0, vmax);
        a1 = vmaxq_f32(a1, vmin); a1 = vminq_f32(a1, vmax);
    }
    vst1q_f32(pc + v, a0);
    vst1q_f32(pc + v + 4, a1);
}

// NHWC multi-row micro-kernel: 8 rows x 8 cols.
// Reuses B loads across all 8 rows. FMA:load = 16:10 vs 2:3 for single-row.
// @nnr-meta isa=NEON dtype=fp32 layout=NHWC special=GEMM tiling=[MR,NR] fusion=post_op
NNR_FORCEINLINE void ukernel_nhwc_mr(
    int kc,
    const float* __restrict pa,   // A base, first row
    int a_stride,                  // distance between A rows (= K)
    const float* __restrict pb,
    int pb_stride,
    float* __restrict pc,          // C base, first row
    int c_stride,                  // distance between C rows (= M)
    int v,
    bool zero_init,
    bool do_fuse,
    const float* bias_col,
    float fmin, float fmax)
{
    float32x4_t c0_0, c0_1, c1_0, c1_1, c2_0, c2_1, c3_0, c3_1;
    float32x4_t c4_0, c4_1, c5_0, c5_1, c6_0, c6_1, c7_0, c7_1;

    if (zero_init) {
        c0_0 = c0_1 = c1_0 = c1_1 = c2_0 = c2_1 = c3_0 = c3_1 = vdupq_n_f32(0.0f);
        c4_0 = c4_1 = c5_0 = c5_1 = c6_0 = c6_1 = c7_0 = c7_1 = vdupq_n_f32(0.0f);
    } else {
        c0_0 = vld1q_f32(pc + v); c0_1 = vld1q_f32(pc + v + 4);
        c1_0 = vld1q_f32(pc + c_stride + v); c1_1 = vld1q_f32(pc + c_stride + v + 4);
        c2_0 = vld1q_f32(pc + 2*c_stride + v); c2_1 = vld1q_f32(pc + 2*c_stride + v + 4);
        c3_0 = vld1q_f32(pc + 3*c_stride + v); c3_1 = vld1q_f32(pc + 3*c_stride + v + 4);
        c4_0 = vld1q_f32(pc + 4*c_stride + v); c4_1 = vld1q_f32(pc + 4*c_stride + v + 4);
        c5_0 = vld1q_f32(pc + 5*c_stride + v); c5_1 = vld1q_f32(pc + 5*c_stride + v + 4);
        c6_0 = vld1q_f32(pc + 6*c_stride + v); c6_1 = vld1q_f32(pc + 6*c_stride + v + 4);
        c7_0 = vld1q_f32(pc + 7*c_stride + v); c7_1 = vld1q_f32(pc + 7*c_stride + v + 4);
    }

    const float* pa1 = pa + a_stride, *pa2 = pa + 2*a_stride, *pa3 = pa + 3*a_stride;
    const float* pa4 = pa + 4*a_stride, *pa5 = pa + 5*a_stride;
    const float* pa6 = pa + 6*a_stride, *pa7 = pa + 7*a_stride;

    for (int k = 0; k < kc; k++) {
        float32x4_t b0 = vld1q_f32(pb + (size_t)k * pb_stride);
        float32x4_t b1 = vld1q_f32(pb + (size_t)k * pb_stride + 4);
        float32x4_t av;
        av = vdupq_n_f32(pa[k]);  c0_0 = vfmaq_f32(c0_0, av, b0); c0_1 = vfmaq_f32(c0_1, av, b1);
        av = vdupq_n_f32(pa1[k]); c1_0 = vfmaq_f32(c1_0, av, b0); c1_1 = vfmaq_f32(c1_1, av, b1);
        av = vdupq_n_f32(pa2[k]); c2_0 = vfmaq_f32(c2_0, av, b0); c2_1 = vfmaq_f32(c2_1, av, b1);
        av = vdupq_n_f32(pa3[k]); c3_0 = vfmaq_f32(c3_0, av, b0); c3_1 = vfmaq_f32(c3_1, av, b1);
        av = vdupq_n_f32(pa4[k]); c4_0 = vfmaq_f32(c4_0, av, b0); c4_1 = vfmaq_f32(c4_1, av, b1);
        av = vdupq_n_f32(pa5[k]); c5_0 = vfmaq_f32(c5_0, av, b0); c5_1 = vfmaq_f32(c5_1, av, b1);
        av = vdupq_n_f32(pa6[k]); c6_0 = vfmaq_f32(c6_0, av, b0); c6_1 = vfmaq_f32(c6_1, av, b1);
        av = vdupq_n_f32(pa7[k]); c7_0 = vfmaq_f32(c7_0, av, b0); c7_1 = vfmaq_f32(c7_1, av, b1);
    }

    if (do_fuse) {
        float32x4_t vmin = vdupq_n_f32(fmin), vmax = vdupq_n_f32(fmax);
        float32x4_t bv0 = vld1q_f32(bias_col + v), bv1 = vld1q_f32(bias_col + v + 4);
        c0_0 = vminq_f32(vmaxq_f32(vaddq_f32(c0_0, bv0), vmin), vmax);
        c0_1 = vminq_f32(vmaxq_f32(vaddq_f32(c0_1, bv1), vmin), vmax);
        c1_0 = vminq_f32(vmaxq_f32(vaddq_f32(c1_0, bv0), vmin), vmax);
        c1_1 = vminq_f32(vmaxq_f32(vaddq_f32(c1_1, bv1), vmin), vmax);
        c2_0 = vminq_f32(vmaxq_f32(vaddq_f32(c2_0, bv0), vmin), vmax);
        c2_1 = vminq_f32(vmaxq_f32(vaddq_f32(c2_1, bv1), vmin), vmax);
        c3_0 = vminq_f32(vmaxq_f32(vaddq_f32(c3_0, bv0), vmin), vmax);
        c3_1 = vminq_f32(vmaxq_f32(vaddq_f32(c3_1, bv1), vmin), vmax);
        c4_0 = vminq_f32(vmaxq_f32(vaddq_f32(c4_0, bv0), vmin), vmax);
        c4_1 = vminq_f32(vmaxq_f32(vaddq_f32(c4_1, bv1), vmin), vmax);
        c5_0 = vminq_f32(vmaxq_f32(vaddq_f32(c5_0, bv0), vmin), vmax);
        c5_1 = vminq_f32(vmaxq_f32(vaddq_f32(c5_1, bv1), vmin), vmax);
        c6_0 = vminq_f32(vmaxq_f32(vaddq_f32(c6_0, bv0), vmin), vmax);
        c6_1 = vminq_f32(vmaxq_f32(vaddq_f32(c6_1, bv1), vmin), vmax);
        c7_0 = vminq_f32(vmaxq_f32(vaddq_f32(c7_0, bv0), vmin), vmax);
        c7_1 = vminq_f32(vmaxq_f32(vaddq_f32(c7_1, bv1), vmin), vmax);
    }

    vst1q_f32(pc + v, c0_0); vst1q_f32(pc + v + 4, c0_1);
    vst1q_f32(pc + c_stride + v, c1_0); vst1q_f32(pc + c_stride + v + 4, c1_1);
    vst1q_f32(pc + 2*c_stride + v, c2_0); vst1q_f32(pc + 2*c_stride + v + 4, c2_1);
    vst1q_f32(pc + 3*c_stride + v, c3_0); vst1q_f32(pc + 3*c_stride + v + 4, c3_1);
    vst1q_f32(pc + 4*c_stride + v, c4_0); vst1q_f32(pc + 4*c_stride + v + 4, c4_1);
    vst1q_f32(pc + 5*c_stride + v, c5_0); vst1q_f32(pc + 5*c_stride + v + 4, c5_1);
    vst1q_f32(pc + 6*c_stride + v, c6_0); vst1q_f32(pc + 6*c_stride + v + 4, c6_1);
    vst1q_f32(pc + 7*c_stride + v, c7_0); vst1q_f32(pc + 7*c_stride + v + 4, c7_1);
}

} // namespace neon
} // namespace nnr

#endif // __aarch64__ || _M_ARM64
