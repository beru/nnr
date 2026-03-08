#pragma once
#include "nnr.h"
#include "thread_pool.h"
#include "cpu_features.h"
#include <cstring>

#ifdef NNR_ARCH_X64
#include "backend/x64/layout_x64.h"
#elifdef NNR_ARCH_ARM64
#include "backend/arm/layout_neon.h"
#endif

namespace nnr {

// ---------------------------------------------------------------------------
// NCHW -> NHWC reorder for 4D float tensors.
// src layout: [N, C, H, W]  (c * HW + h * W + w)
// dst layout: [N, H, W, C]  (h * W * C + w * C + c)
//
// This is a transpose of a [C x HW] matrix to [HW x C] per batch.
// Uses SIMD 8x8 or 16x16 block transpose for full cache-line utilization.
// For N=1 (single batch), parallelizes over spatial blocks instead of batch.
// ---------------------------------------------------------------------------
inline void nchw_to_nhwc(float* __restrict dst, const float* __restrict src,
    int N, int C, int H, int W)
{
    const int HW = H * W;
    const size_t batch = (size_t)C * HW;

    auto transpose_batch = [&](const float* src_n, float* dst_n, int hw_begin, int hw_end) {
#ifdef NNR_ARCH_X64
        nchw_nhwc_batch_x64(src_n, dst_n, C, HW, hw_begin, hw_end);
#elifdef NNR_ARCH_ARM64
        nchw_nhwc_batch_neon(src_n, dst_n, C, HW, hw_begin, hw_end);
#else
        for (int c = 0; c < C; c++) {
            const float* sc = src_n + (size_t)c * HW + hw_begin;
            for (int hw = 0; hw < hw_end - hw_begin; hw++)
                dst_n[(hw_begin + hw) * C + c] = sc[hw];
        }
#endif
    };

    if (N > 1) {
        nnr::for_static(0, N, true, [&](int n) {
            transpose_batch(src + n * batch, dst + n * batch, 0, HW);
        });
    } else {
        constexpr int TARGET_BLOCK_BYTES = 128 * 1024;
        int block_hw = C > 0 ? std::max(16, TARGET_BLOCK_BYTES / (C * 4)) : HW;
#ifdef NNR_ARCH_X64
        int align = layout_block_align_x64();
        block_hw = (block_hw + align - 1) & ~(align - 1);
#elifdef NNR_ARCH_ARM64
        block_hw = (block_hw + layout_block_align_neon() - 1) & ~(layout_block_align_neon() - 1);
#endif
        int nblocks = (HW + block_hw - 1) / block_hw;
        nnr::for_static(0, nblocks, nblocks > 1, [&](int b) {
            int hw0 = b * block_hw;
            int hw1 = std::min(hw0 + block_hw, HW);
            transpose_batch(src, dst, hw0, hw1);
        });
    }
}

// ---------------------------------------------------------------------------
// NHWC -> NCHW reorder for 4D float tensors.
// src layout: [N, H, W, C]  (h * W * C + w * C + c)
// dst layout: [N, C, H, W]  (c * HW + h * W + w)
//
// This is a transpose of a [HW x C] matrix to [C x HW] per batch.
// ---------------------------------------------------------------------------
inline void nhwc_to_nchw(float* __restrict dst, const float* __restrict src,
    int N, int C, int H, int W)
{
    const int HW = H * W;
    const size_t batch = (size_t)C * HW;

    auto transpose_batch = [&](const float* src_n, float* dst_n, int hw_begin, int hw_end) {
#ifdef NNR_ARCH_X64
        nhwc_nchw_batch_x64(src_n, dst_n, C, HW, hw_begin, hw_end);
#elifdef NNR_ARCH_ARM64
        nhwc_nchw_batch_neon(src_n, dst_n, C, HW, hw_begin, hw_end);
#else
        for (int c = 0; c < C; c++) {
            float* dc = dst_n + (size_t)c * HW + hw_begin;
            for (int hw = 0; hw < hw_end - hw_begin; hw++)
                dc[hw] = src_n[(hw_begin + hw) * C + c];
        }
#endif
    };

    if (N > 1) {
        nnr::for_static(0, N, true, [&](int n) {
            transpose_batch(src + n * batch, dst + n * batch, 0, HW);
        });
    } else {
        constexpr int TARGET_BLOCK_BYTES = 128 * 1024;
        int block_hw = C > 0 ? std::max(16, TARGET_BLOCK_BYTES / (C * 4)) : HW;
#ifdef NNR_ARCH_X64
        int align = layout_block_align_x64();
        block_hw = (block_hw + align - 1) & ~(align - 1);
#elifdef NNR_ARCH_ARM64
        block_hw = (block_hw + layout_block_align_neon() - 1) & ~(layout_block_align_neon() - 1);
#endif
        int nblocks = (HW + block_hw - 1) / block_hw;
        nnr::for_static(0, nblocks, nblocks > 1, [&](int b) {
            int hw0 = b * block_hw;
            int hw1 = std::min(hw0 + block_hw, HW);
            transpose_batch(src, dst, hw0, hw1);
        });
    }
}

// ---------------------------------------------------------------------------
// NCHW -> NCHWc reorder for 4D float tensors.
// src layout: [N, C, H, W]  — C contiguous planes of H×W
// dst layout: [N, Cb, H, W, c]  — Cb = ceil(C/c) channel blocks of c channels
// c = block size (16 for AVX-512, 8 for AVX2)
// dst must be pre-allocated with N * Cb * H * W * c floats.
// When C is not divisible by c, the last block's extra lanes are zero-filled.
// ---------------------------------------------------------------------------
inline void nchw_to_nchwc(float* __restrict dst, const float* __restrict src,
    int N, int C, int H, int W, int block)
{
    const int HW = H * W;
    const int Cb = (C + block - 1) / block;
    const size_t dst_batch = (size_t)Cb * HW * block;

    auto convert_batch = [&](const float* src_n, float* dst_n) {
        for (int cb = 0; cb < Cb; cb++) {
            const int c_start = cb * block;
            const int c_valid = std::min(block, C - c_start);
            float* dst_block = dst_n + (size_t)cb * HW * block;

            if (c_valid == block) {
#ifdef NNR_ARCH_X64
                if (has_avx512() && block == 16) {
                    nchw_to_nchwc_block16_avx512(dst_block,
                        src_n + (size_t)c_start * HW, HW);
                } else
#endif
                {
                    for (int hw = 0; hw < HW; hw++) {
                        for (int ci = 0; ci < block; ci++) {
                            dst_block[hw * block + ci] = src_n[(size_t)(c_start + ci) * HW + hw];
                        }
                    }
                }
            } else {
                for (int hw = 0; hw < HW; hw++) {
                    int ci = 0;
                    for (; ci < c_valid; ci++) {
                        dst_block[hw * block + ci] = src_n[(size_t)(c_start + ci) * HW + hw];
                    }
                    for (; ci < block; ci++) {
                        dst_block[hw * block + ci] = 0.0f;
                    }
                }
            }
        }
    };

    nnr::for_static(0, N, N > 1, [&](int n) {
        convert_batch(src + (size_t)n * C * HW, dst + (size_t)n * dst_batch);
    });
}

// ---------------------------------------------------------------------------
// NCHW -> NCHWc reorder with spatial zero-padding.
// dst layout: [N, Cb, H+padT+padB, W+padL+padR, block]
// The pad region is zero-filled; data is placed at offset (padT, padL).
// This lets the Conv kernel run with pad=0 on the padded input,
// eliminating all bounds-checking branches in the inner loop.
// ---------------------------------------------------------------------------
inline void nchw_to_nchwc_padded(float* __restrict dst, const float* __restrict src,
    int N, int C, int H, int W,
    int padT, int padL, int padB, int padR, int block)
{
    const int HW = H * W;
    const int pH = H + padT + padB;
    const int pW = W + padL + padR;
    const int pHW = pH * pW;
    const int Cb = (C + block - 1) / block;
    const size_t dst_batch = (size_t)Cb * pHW * block;

    nnr::for_static(0, N, N > 1, [&](int n) {
        const float* src_n = src + (size_t)n * C * HW;
        float* dst_n = dst + (size_t)n * dst_batch;

        memset(dst_n, 0, dst_batch * sizeof(float));

        for (int cb = 0; cb < Cb; cb++) {
            const int c_start = cb * block;
            const int c_valid = std::min(block, C - c_start);
            float* dst_block = dst_n + (size_t)cb * pHW * block;

            for (int h = 0; h < H; h++) {
                float* dst_row = dst_block + (size_t)(h + padT) * pW * block + padL * block;
                for (int w = 0; w < W; w++) {
                    for (int ci = 0; ci < c_valid; ci++) {
                        dst_row[w * block + ci] = src_n[(size_t)(c_start + ci) * HW + h * W + w];
                    }
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// NCHWc -> NCHWc spatial zero-padding (copy with pad border).
// src layout: [N, Cb, H, W, block]
// dst layout: [N, Cb, H+padT+padB, W+padL+padR, block]
// ---------------------------------------------------------------------------
inline void nchwc_pad(float* __restrict dst, const float* __restrict src,
    int N, int C, int H, int W,
    int padT, int padL, int padB, int padR, int block)
{
    const int pH = H + padT + padB;
    const int pW = W + padL + padR;
    const int Cb = (C + block - 1) / block;
    const size_t src_batch = (size_t)Cb * H * W * block;
    const size_t dst_batch = (size_t)Cb * pH * pW * block;

    nnr::for_static(0, N, N > 1, [&](int n) {
        const float* src_n = src + (size_t)n * src_batch;
        float* dst_n = dst + (size_t)n * dst_batch;

        memset(dst_n, 0, dst_batch * sizeof(float));

        for (int cb = 0; cb < Cb; cb++) {
            const float* src_block = src_n + (size_t)cb * H * W * block;
            float* dst_block = dst_n + (size_t)cb * pH * pW * block;

            for (int h = 0; h < H; h++) {
                memcpy(dst_block + (size_t)(h + padT) * pW * block + padL * block,
                       src_block + (size_t)h * W * block,
                       (size_t)W * block * sizeof(float));
            }
        }
    });
}

// ---------------------------------------------------------------------------
// NCHWc -> NCHW reorder for 4D float tensors.
// src layout: [N, Cb, H, W, c]  — Cb = ceil(C/c) channel blocks
// dst layout: [N, C, H, W]  — standard NCHW
// Only the first C channels are extracted (ignoring padding).
// ---------------------------------------------------------------------------
inline void nchwc_to_nchw(float* __restrict dst, const float* __restrict src,
    int N, int C, int H, int W, int block)
{
    const int HW = H * W;
    const int Cb = (C + block - 1) / block;
    const size_t src_batch = (size_t)Cb * HW * block;

    auto convert_batch = [&](const float* src_n, float* dst_n) {
        for (int cb = 0; cb < Cb; cb++) {
            const int c_start = cb * block;
            const int c_valid = std::min(block, C - c_start);
            const float* src_block = src_n + (size_t)cb * HW * block;

            if (c_valid == block) {
#ifdef NNR_ARCH_X64
                if (has_avx512() && block == 16) {
                    nchwc_to_nchw_block16_avx512(dst_n + (size_t)c_start * HW,
                        src_block, HW);
                } else
#endif
                {
                    for (int ci = 0; ci < block; ci++) {
                        float* dst_c = dst_n + (size_t)(c_start + ci) * HW;
                        for (int hw = 0; hw < HW; hw++) {
                            dst_c[hw] = src_block[hw * block + ci];
                        }
                    }
                }
            } else {
                for (int ci = 0; ci < c_valid; ci++) {
                    float* dst_c = dst_n + (size_t)(c_start + ci) * HW;
                    for (int hw = 0; hw < HW; hw++) {
                        dst_c[hw] = src_block[hw * block + ci];
                    }
                }
            }
        }
    };

    nnr::for_static(0, N, N > 1, [&](int n) {
        convert_batch(src + (size_t)n * src_batch, dst + (size_t)n * C * HW);
    });
}

// In-place reorder using temp buffer.
// temp must hold at least N*C*H*W floats.
inline void reorder_inplace(float* data, int N, int C, int H, int W,
    memory_layout_t from, memory_layout_t to, float* temp)
{
    if (from == to) return;
    size_t count = (size_t)N * C * H * W;
    memcpy(temp, data, count * sizeof(float));
    if (from == memory_layout_t::NCHW && to == memory_layout_t::NHWC)
        nchw_to_nhwc(data, temp, N, C, H, W);
    else if (from == memory_layout_t::NHWC && to == memory_layout_t::NCHW)
        nhwc_to_nchw(data, temp, N, C, H, W);
}

// Transpose weight matrix [M x C] -> [C x M] for NHWC 1x1 Conv.
inline void transpose_weights(float* __restrict dst, const float* __restrict src,
    int M, int C)
{
    for (int m = 0; m < M; m++)
        for (int c = 0; c < C; c++)
            dst[c * M + m] = src[m * C + c];
}

} // namespace nnr
