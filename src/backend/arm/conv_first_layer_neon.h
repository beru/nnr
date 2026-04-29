#pragma once
// Direct convolution for first-layer Conv (small input channels, e.g., RGB).
// ARM NEON version: vectorizes over output channels (4 per q-register).
//
// Loop order: outer=output pixels (tiled, WT=8), inner=kernel (IC×KH×KW).
// Accumulators stay in registers across all kernel positions.
// 8 q-register accumulators + 1 weight vector = 9 registers (well within 32 NEON regs).
//
// Weights: repacked as [IC][KH][KW][OC/4][4].

#ifdef NNR_ARCH_ARM64

#include <arm_neon.h>
#include <algorithm>
#include <cstring>
#include <vector>

namespace nnr {

// @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=FirstLayer tiling=spatial fusion=post_op
inline bool conv_first_layer_neon(
    float* __restrict y,
    const float* __restrict x,
    const float* __restrict w,    // repacked [IC, KH, KW, OC_blocks, 4]
    const float* __restrict bias,
    int IC, int iH, int iW,
    int OC, int oH, int oW,
    int KH, int KW,
    int sH, int sW,
    int padH, int padW,
    operator_t::post_fn_t post_fn,
    const operator_t* fused_op)
{
    const int OC_blocks = (OC + 3) / 4;
    const int kSpatial = KH * KW;
    const int spatial = oH * oW;
    constexpr int WT = 8;  // output width tile

    // Work buffer: one row in interleaved layout [oW][OC_blocks*4]
    const size_t row_elems = (size_t)oW * OC_blocks * 4;

    // Per-thread scratch for the interleaved row buffer. Hoisted out of
    // the row loop so we don't malloc 150 KB × oH per inference (Conv_219
    // on ssd-12: 600 rows × 3 reps = 1800 alloc/free pairs).
    NNR_POOL_ENSURE_SCRATCH(row_elems * sizeof(float));

    nnr::for_dynamic(0, oH, oH >= 4, [&](int tid, int oh) {
        float* buf_ptr = (float*)NNR_POOL_SCRATCH(tid);
        const int out_stride = OC_blocks * 4;

        for (int ob = 0; ob < OC_blocks; ob++) {
            float* out_ob = buf_ptr + ob * 4;

            // Initialize with bias
            float32x4_t vbias;
            if (bias) {
                int oc0 = ob * 4;
                if (oc0 + 4 <= OC) {
                    vbias = vld1q_f32(bias + oc0);
                } else {
                    float tmp[4] = {};
                    for (int i = 0; i < OC - oc0; i++) tmp[i] = bias[oc0 + i];
                    vbias = vld1q_f32(tmp);
                }
            } else {
                vbias = vdupq_n_f32(0.f);
            }
            for (int ow = 0; ow < oW; ow++)
                vst1q_f32(out_ob + (size_t)ow * out_stride, vbias);

            // Main loop: tile over oW, inner loop over IC×KH×KW
            int ow = 0;
            for (; ow + WT <= oW; ow += WT) {
                float32x4_t a0 = vld1q_f32(out_ob + (size_t)(ow+0) * out_stride);
                float32x4_t a1 = vld1q_f32(out_ob + (size_t)(ow+1) * out_stride);
                float32x4_t a2 = vld1q_f32(out_ob + (size_t)(ow+2) * out_stride);
                float32x4_t a3 = vld1q_f32(out_ob + (size_t)(ow+3) * out_stride);
                float32x4_t a4 = vld1q_f32(out_ob + (size_t)(ow+4) * out_stride);
                float32x4_t a5 = vld1q_f32(out_ob + (size_t)(ow+5) * out_stride);
                float32x4_t a6 = vld1q_f32(out_ob + (size_t)(ow+6) * out_stride);
                float32x4_t a7 = vld1q_f32(out_ob + (size_t)(ow+7) * out_stride);

                for (int ic = 0; ic < IC; ic++) {
                    const float* x_ic = x + (size_t)ic * iH * iW;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * sH + kh - padH;
                        if (ih < 0 || ih >= iH) continue;
                        const float* x_row = x_ic + ih * iW;
                        for (int kw = 0; kw < KW; kw++) {
                            float32x4_t wv = vld1q_f32(
                                w + (((size_t)ic * kSpatial + kh * KW + kw) * OC_blocks + ob) * 4);
                            int iw_base = kw - padW;

                            #define FMA_T(T) { \
                                int iw = (ow + T) * sW + iw_base; \
                                if (iw >= 0 && iw < iW) \
                                    a##T = vfmaq_f32(a##T, vdupq_n_f32(x_row[iw]), wv); \
                            }
                            FMA_T(0) FMA_T(1) FMA_T(2) FMA_T(3)
                            FMA_T(4) FMA_T(5) FMA_T(6) FMA_T(7)
                            #undef FMA_T
                        }
                    }
                }

                vst1q_f32(out_ob + (size_t)(ow+0) * out_stride, a0);
                vst1q_f32(out_ob + (size_t)(ow+1) * out_stride, a1);
                vst1q_f32(out_ob + (size_t)(ow+2) * out_stride, a2);
                vst1q_f32(out_ob + (size_t)(ow+3) * out_stride, a3);
                vst1q_f32(out_ob + (size_t)(ow+4) * out_stride, a4);
                vst1q_f32(out_ob + (size_t)(ow+5) * out_stride, a5);
                vst1q_f32(out_ob + (size_t)(ow+6) * out_stride, a6);
                vst1q_f32(out_ob + (size_t)(ow+7) * out_stride, a7);
            }

            // Remainder pixels (< WT)
            for (; ow < oW; ow++) {
                float32x4_t a = vld1q_f32(out_ob + (size_t)ow * out_stride);
                for (int ic = 0; ic < IC; ic++) {
                    const float* x_ic = x + (size_t)ic * iH * iW;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * sH + kh - padH;
                        if (ih < 0 || ih >= iH) continue;
                        const float* x_row = x_ic + ih * iW;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * sW + kw - padW;
                            if (iw < 0 || iw >= iW) continue;
                            float32x4_t wv = vld1q_f32(
                                w + (((size_t)ic * kSpatial + kh * KW + kw) * OC_blocks + ob) * 4);
                            a = vfmaq_f32(a, vdupq_n_f32(x_row[iw]), wv);
                        }
                    }
                }
                vst1q_f32(out_ob + (size_t)ow * out_stride, a);
            }
        }

        // Transpose: interleaved [ow][OC_blocks*4] → NCHW [OC][spatial]
        for (int oc = 0; oc < OC; oc++) {
            float* y_row = y + (size_t)oc * spatial + oh * oW;
            int ob = oc / 4, lane = oc % 4;
            for (int ow2 = 0; ow2 < oW; ow2++)
                y_row[ow2] = buf_ptr[(size_t)ow2 * OC_blocks * 4 + ob * 4 + lane];
        }

        if (post_fn) {
            for (int oc = 0; oc < OC; oc++) {
                float* y_row = y + (size_t)oc * spatial + oh * oW;
                int offset = (int)((size_t)oc * spatial + oh * oW);
                post_fn(y_row, 1, oW, oW, fused_op, nullptr, offset);
            }
        }
    });

    return true;
}

// @nnr-meta isa=scalar dtype=fp32 layout=NCHW special=FirstLayer
inline void pack_weights_first_layer_neon(
    float* __restrict dst,
    const float* __restrict src,
    int OC, int IC, int KH, int KW)
{
    int OC_blocks = (OC + 3) / 4;
    int kSpatial = KH * KW;
    memset(dst, 0, (size_t)IC * kSpatial * OC_blocks * 4 * sizeof(float));
    for (int ic = 0; ic < IC; ic++)
        for (int k = 0; k < kSpatial; k++)
            for (int oc = 0; oc < OC; oc++)
                dst[((size_t)ic * kSpatial + k) * OC_blocks * 4 + oc] =
                    src[((size_t)oc * IC + ic) * kSpatial + k];
}

// @nnr-meta isa=scalar dtype=fp32 layout=NCHW special=FirstLayer
inline size_t pack_weights_first_layer_neon_size(int OC, int IC, int KH, int KW)
{
    return (size_t)IC * KH * KW * ((OC + 3) / 4) * 4;
}

} // namespace nnr

#endif // NNR_ARCH_ARM64
