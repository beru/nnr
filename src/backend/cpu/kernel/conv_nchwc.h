#pragma once
// NCHWc blocked-layout convolution — weight packing and dispatch.
//
// NCHWc layout: [N, C/c, H, W, c]  where c = SIMD width (16 for AVX-512, 8 for AVX2).
// Channels within a block are contiguous — natural SIMD loads.
//
// Weight layout for pointwise (1x1) conv:
//   [OC/c, IC, c]  — for each output channel block, IC input channels, c output lanes.
//
// Weight layout for general (KxK) conv:
//   [OC/c, IC, KH, KW, c]  — for each output channel block, IC*KH*KW*c weights.
//
// Platform-specific kernels are in backend/x64/ (AVX-512).

#include "nnr.h"
#include "cpu_features.h"

#ifdef NNR_ARCH_X64
#include "backend/x64/conv_nchwc_avx512.h"
#include "backend/x64/winograd_nchwc_x64.h"
#elif defined(NNR_ARCH_ARM64)
#include "backend/arm/conv_nchwc_neon.h"
#endif

namespace nnr {

// ---------------------------------------------------------------------------
// Pack weights for NCHWc 1x1 convolution.
// src: [OC, IC] in NCHW order (row-major: src[oc * IC + ic])
// dst: [OC/block, IC, block] — each output block has IC x block weights.
// When OC is not divisible by block, extra output lanes are zero-filled.
// ---------------------------------------------------------------------------
inline void pack_weight_nchwc_1x1(float* __restrict dst, const float* __restrict src,
    int OC, int IC, int block)
{
    const int OCb = (OC + block - 1) / block;
    for (int ob = 0; ob < OCb; ob++) {
        const int oc_start = ob * block;
        const int oc_valid = std::min(block, OC - oc_start);
        for (int ic = 0; ic < IC; ic++) {
            float* d = dst + ((size_t)ob * IC + ic) * block;
            int oc = 0;
            for (; oc < oc_valid; oc++) {
                d[oc] = src[(oc_start + oc) * IC + ic];
            }
            for (; oc < block; oc++) {
                d[oc] = 0.0f;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pack weights for NCHWc general (KxK) convolution.
// src: [OC, IC, KH, KW] in NCHW order
// dst: [OC/block, ICb, KH, KW, block_in, block_out]
//   = [OC/block, IC/block, spatial, 16, 16]
// The kernel accesses: w_ob[(icb * spatial + s) * 256 + il * 16 + oc_lane]
// When OC or IC is not divisible by block, extra lanes are zero-filled.
// ---------------------------------------------------------------------------
inline void pack_weight_nchwc(float* __restrict dst, const float* __restrict src,
    int OC, int IC, int KH, int KW, int block)
{
    const int OCb = (OC + block - 1) / block;
    const int ICb = (IC + block - 1) / block;
    const int spatial = KH * KW;
    const size_t ob_stride = (size_t)ICb * spatial * block * block;
    memset(dst, 0, (size_t)OCb * ob_stride * sizeof(float));

    for (int ob = 0; ob < OCb; ob++) {
        const int oc_start = ob * block;
        const int oc_valid = std::min(block, OC - oc_start);
        for (int icb = 0; icb < ICb; icb++) {
            const int ic_start = icb * block;
            const int ic_valid = std::min(block, IC - ic_start);
            for (int s = 0; s < spatial; s++) {
                // dst offset: [ob, icb, s, il, oc_lane]
                float* d = dst + ob * ob_stride
                    + ((size_t)icb * spatial + s) * block * block;
                for (int il = 0; il < ic_valid; il++) {
                    int ic = ic_start + il;
                    for (int oc = 0; oc < oc_valid; oc++) {
                        d[il * block + oc] =
                            src[((size_t)(oc_start + oc) * IC + ic) * spatial + s];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pack weights for NCHWc general (KxK) convolution — IC-blocked layout.
// src: [OC, IC, KH, KW] in NCHW order
// dst: [OC/block, IC/block, KH, KW, block_ic, block_oc]
//
// Both IC and OC are blocked by `block` (16 for AVX-512).
// The inner 16×16 weight tile is laid out as [ic_lane, oc_lane] so that
// the kernel can stream 16 sequential weight vectors (one per input channel
// within a block) during the IC-reduction loop.
//
// When IC or OC is not divisible by block, extra lanes are zero-filled.
// ---------------------------------------------------------------------------
inline void pack_weight_nchwc_blocked(float* __restrict dst, const float* __restrict src,
    int OC, int IC, int KH, int KW, int block)
{
    const int OCb = (OC + block - 1) / block;
    const int ICb = (IC + block - 1) / block;
    const int spatial = KH * KW;
    const int blk2 = block * block;  // 256 for block=16

    // Zero the entire destination (handles IC/OC padding implicitly)
    std::memset(dst, 0, (size_t)OCb * ICb * spatial * blk2 * sizeof(float));

    for (int ob = 0; ob < OCb; ob++) {
        const int oc_start = ob * block;
        const int oc_valid = std::min(block, OC - oc_start);
        for (int icb = 0; icb < ICb; icb++) {
            const int ic_start = icb * block;
            const int ic_valid = std::min(block, IC - ic_start);
            for (int s = 0; s < spatial; s++) {
                float* d = dst + ((size_t)(ob * ICb + icb) * spatial + s) * blk2;
                // d[il * block + oc] = src[(oc_start+oc) * IC * spatial + (ic_start+il) * spatial + s]
                for (int il = 0; il < ic_valid; il++) {
                    for (int oc = 0; oc < oc_valid; oc++) {
                        d[il * block + oc] = src[((size_t)(oc_start + oc) * IC + ic_start + il) * spatial + s];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pack bias for NCHWc layout.
// src: [OC] or nullptr
// dst: [OC_padded] — zero-padded to multiple of block
// ---------------------------------------------------------------------------
inline void pack_bias_nchwc(float* __restrict dst, const float* src,
    int OC, int block)
{
    const int OC_padded = nchwc_padded_channels(OC, block);
    if (src) {
        int i = 0;
        for (; i < OC; i++) dst[i] = src[i];
        for (; i < OC_padded; i++) dst[i] = 0.0f;
    } else {
        for (int i = 0; i < OC_padded; i++) dst[i] = 0.0f;
    }
}

} // namespace nnr
