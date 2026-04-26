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
// NCHW -> NHWC reorder for 4D tensors. Templated on element type so any
// 1/2/4/8-byte trivially-copyable T works. fp32 (T=float) uses the AVX-512
// / NEON SIMD fast path; other byte sizes fall back to scalar.
// src layout: [N, C, H, W]  (c * HW + h * W + w)
// dst layout: [N, H, W, C]  (h * W * C + w * C + c)
//
// Strip-bounded mode: if row_start/row_end are provided, only output rows
// [row_start, row_end) are written. Caller must allocate the full output
// tensor; the helper writes a band only. row_end == -1 → full tensor.
//
// In ring-buffer mode (scroll_chains.cpp), H here is the *stride* — i.e.
// dims[2] which is replaced by ring_H — and the virtual-pointer trick in
// the caller makes (data + h * row_stride) land at the right ring slot for
// h in the strip's logical range. row_start/row_end come from the caller
// in logical-tensor space; they are NOT clamped against H.
// ---------------------------------------------------------------------------
template<typename T>
inline void nchw_to_nhwc(T* __restrict dst, const T* __restrict src,
    int N, int C, int H, int W, int row_start = 0, int row_end = -1)
{
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                  "T must be 1, 2, 4, or 8 bytes");
    if (row_end < 0) row_end = H;
    const int HW = H * W;
    const int hw_begin = row_start * W;
    const int hw_end   = row_end   * W;
    if (hw_end <= hw_begin) return;
    const size_t batch = (size_t)C * HW;

    auto transpose_batch = [&](const T* src_n, T* dst_n, int hwb, int hwe) {
        if constexpr (sizeof(T) == 4) {
#ifdef NNR_ARCH_X64
            nchw_nhwc_batch_x64(reinterpret_cast<const float*>(src_n),
                                reinterpret_cast<float*>(dst_n), C, HW, hwb, hwe);
            return;
#elifdef NNR_ARCH_ARM64
            nchw_nhwc_batch_neon(reinterpret_cast<const float*>(src_n),
                                 reinterpret_cast<float*>(dst_n), C, HW, hwb, hwe);
            return;
#endif
        }
        // Scalar fallback (non-fp32, or arch without a SIMD path).
        for (int c = 0; c < C; c++) {
            const T* sc = src_n + (size_t)c * HW + hwb;
            for (int hw = 0; hw < hwe - hwb; hw++)
                dst_n[(hwb + hw) * C + c] = sc[hw];
        }
    };

    if (N > 1) {
        nnr::for_static(0, N, true, [&](int n) {
            transpose_batch(src + n * batch, dst + n * batch, hw_begin, hw_end);
        });
    } else {
        // N=1: parallelize over HW blocks within [hw_begin, hw_end).
        constexpr int TARGET_BLOCK_BYTES = 128 * 1024;
        int block_hw = C > 0 ? std::max(16, TARGET_BLOCK_BYTES / (C * (int)sizeof(T))) : (hw_end - hw_begin);
#ifdef NNR_ARCH_X64
        int align = layout_block_align_x64();
        block_hw = (block_hw + align - 1) & ~(align - 1);
#elifdef NNR_ARCH_ARM64
        block_hw = (block_hw + layout_block_align_neon() - 1) & ~(layout_block_align_neon() - 1);
#endif
        int range = hw_end - hw_begin;
        int nblocks = (range + block_hw - 1) / block_hw;
        nnr::for_static(0, nblocks, nblocks > 1, [&](int b) {
            int hw0 = hw_begin + b * block_hw;
            int hw1 = std::min(hw0 + block_hw, hw_end);
            transpose_batch(src, dst, hw0, hw1);
        });
    }
}

// ---------------------------------------------------------------------------
// NHWC -> NCHW reorder for 4D tensors. Templated + strip-bounded; see
// nchw_to_nhwc above for the row-range convention and ring-buffer notes.
// ---------------------------------------------------------------------------
template<typename T>
inline void nhwc_to_nchw(T* __restrict dst, const T* __restrict src,
    int N, int C, int H, int W, int row_start = 0, int row_end = -1)
{
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                  "T must be 1, 2, 4, or 8 bytes");
    if (row_end < 0) row_end = H;
    const int HW = H * W;
    const int hw_begin = row_start * W;
    const int hw_end   = row_end   * W;
    if (hw_end <= hw_begin) return;
    const size_t batch = (size_t)C * HW;

    auto transpose_batch = [&](const T* src_n, T* dst_n, int hwb, int hwe) {
        if constexpr (sizeof(T) == 4) {
#ifdef NNR_ARCH_X64
            nhwc_nchw_batch_x64(reinterpret_cast<const float*>(src_n),
                                reinterpret_cast<float*>(dst_n), C, HW, hwb, hwe);
            return;
#elifdef NNR_ARCH_ARM64
            nhwc_nchw_batch_neon(reinterpret_cast<const float*>(src_n),
                                 reinterpret_cast<float*>(dst_n), C, HW, hwb, hwe);
            return;
#endif
        }
        for (int c = 0; c < C; c++) {
            T* dc = dst_n + (size_t)c * HW + hwb;
            for (int hw = 0; hw < hwe - hwb; hw++)
                dc[hw] = src_n[(hwb + hw) * C + c];
        }
    };

    if (N > 1) {
        nnr::for_static(0, N, true, [&](int n) {
            transpose_batch(src + n * batch, dst + n * batch, hw_begin, hw_end);
        });
    } else {
        constexpr int TARGET_BLOCK_BYTES = 128 * 1024;
        int block_hw = C > 0 ? std::max(16, TARGET_BLOCK_BYTES / (C * (int)sizeof(T))) : (hw_end - hw_begin);
#ifdef NNR_ARCH_X64
        int align = layout_block_align_x64();
        block_hw = (block_hw + align - 1) & ~(align - 1);
#elifdef NNR_ARCH_ARM64
        block_hw = (block_hw + layout_block_align_neon() - 1) & ~(layout_block_align_neon() - 1);
#endif
        int range = hw_end - hw_begin;
        int nblocks = (range + block_hw - 1) / block_hw;
        nnr::for_static(0, nblocks, nblocks > 1, [&](int b) {
            int hw0 = hw_begin + b * block_hw;
            int hw1 = std::min(hw0 + block_hw, hw_end);
            transpose_batch(src, dst, hw0, hw1);
        });
    }
}

// ---------------------------------------------------------------------------
// NCHW -> NCHWc reorder for 4D tensors. Templated + strip-bounded (see
// nchw_to_nhwc above for the row-range / ring-buffer convention).
// src layout: [N, C, H, W]  — C contiguous planes of H×W
// dst layout: [N, Cb, H, W, c]  — Cb = ceil(C/c) channel blocks of c channels
// c = block size (16 for AVX-512, 8 for AVX2)
// dst must be pre-allocated with N * Cb * H * W * c elements.
// When C is not divisible by c, the last block's extra lanes are zero-filled.
// ---------------------------------------------------------------------------
template<typename T>
inline void nchw_to_nchwc(T* __restrict dst, const T* __restrict src,
    int N, int C, int H, int W, int block, int row_start = 0, int row_end = -1)
{
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                  "T must be 1, 2, 4, or 8 bytes");
    if (row_end < 0) row_end = H;
    if (row_end <= row_start) return;
    const int HW = H * W;
    const int Cb = (C + block - 1) / block;
    const size_t dst_batch = (size_t)Cb * HW * block;
    const bool full_range = (row_start == 0 && row_end == H);
    const int hw_begin = row_start * W;
    const int hw_end   = row_end   * W;

    auto convert_batch = [&](const T* src_n, T* dst_n) {
        for (int cb = 0; cb < Cb; cb++) {
            const int c_start = cb * block;
            const int c_valid = std::min(block, C - c_start);
            T* dst_block = dst_n + (size_t)cb * HW * block;

            if (c_valid == block) {
                if constexpr (sizeof(T) == 4) {
#ifdef NNR_ARCH_X64
                    if (full_range && has_avx512() && block == 16) {
                        nchw_to_nchwc_block16_avx512(reinterpret_cast<float*>(dst_block),
                            reinterpret_cast<const float*>(src_n) + (size_t)c_start * HW, HW);
                        continue;
                    }
#endif
                }
                for (int hw = hw_begin; hw < hw_end; hw++) {
                    for (int ci = 0; ci < block; ci++) {
                        dst_block[hw * block + ci] = src_n[(size_t)(c_start + ci) * HW + hw];
                    }
                }
            } else {
                for (int hw = hw_begin; hw < hw_end; hw++) {
                    int ci = 0;
                    for (; ci < c_valid; ci++) {
                        dst_block[hw * block + ci] = src_n[(size_t)(c_start + ci) * HW + hw];
                    }
                    for (; ci < block; ci++) {
                        dst_block[hw * block + ci] = T{};   // IC-tail zero pad
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
// NCHWc -> NCHW reorder for 4D tensors. Templated + strip-bounded.
// src layout: [N, Cb, H, W, c]  — Cb = ceil(C/c) channel blocks
// dst layout: [N, C, H, W]  — standard NCHW
// Only the first C channels are extracted (ignoring padding).
// ---------------------------------------------------------------------------
template<typename T>
inline void nchwc_to_nchw(T* __restrict dst, const T* __restrict src,
    int N, int C, int H, int W, int block, int row_start = 0, int row_end = -1)
{
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                  "T must be 1, 2, 4, or 8 bytes");
    if (row_end < 0) row_end = H;
    if (row_end <= row_start) return;
    const int HW = H * W;
    const int Cb = (C + block - 1) / block;
    const size_t src_batch = (size_t)Cb * HW * block;
    const bool full_range = (row_start == 0 && row_end == H);
    const int hw_begin = row_start * W;
    const int hw_end   = row_end   * W;

    auto convert_batch = [&](const T* src_n, T* dst_n) {
        for (int cb = 0; cb < Cb; cb++) {
            const int c_start = cb * block;
            const int c_valid = std::min(block, C - c_start);
            const T* src_block = src_n + (size_t)cb * HW * block;

            if (c_valid == block) {
                if constexpr (sizeof(T) == 4) {
#ifdef NNR_ARCH_X64
                    if (full_range && has_avx512() && block == 16) {
                        nchwc_to_nchw_block16_avx512(
                            reinterpret_cast<float*>(dst_n) + (size_t)c_start * HW,
                            reinterpret_cast<const float*>(src_block), HW);
                        continue;
                    }
#endif
                }
                for (int ci = 0; ci < block; ci++) {
                    T* dst_c = dst_n + (size_t)(c_start + ci) * HW;
                    for (int hw = hw_begin; hw < hw_end; hw++) {
                        dst_c[hw] = src_block[hw * block + ci];
                    }
                }
            } else {
                for (int ci = 0; ci < c_valid; ci++) {
                    T* dst_c = dst_n + (size_t)(c_start + ci) * HW;
                    for (int hw = hw_begin; hw < hw_end; hw++) {
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

// Element-size-generic in-place NHWC↔NCHW reorder for non-float tensor types
// (uint8 quantized outputs, float16, etc.). Scalar; intended for the once-per-run
// graph-output reorder, not for hot-path tile transposes.
// temp must hold at least N*C*H*W*elem_size bytes.
inline void reorder_inplace_bytes(void* data, int N, int C, int H, int W,
    memory_layout_t from, memory_layout_t to, void* temp, size_t elem_size)
{
    if (from == to) return;
    const size_t count = (size_t)N * C * H * W;
    memcpy(temp, data, count * elem_size);
    auto* dst = (uint8_t*)data;
    const auto* src = (const uint8_t*)temp;
    const int HW = H * W;
    if (from == memory_layout_t::NHWC && to == memory_layout_t::NCHW) {
        for (int n = 0; n < N; n++) {
            const uint8_t* src_n = src + (size_t)n * HW * C * elem_size;
            uint8_t*       dst_n = dst + (size_t)n * C * HW * elem_size;
            for (int c = 0; c < C; c++)
                for (int hw = 0; hw < HW; hw++)
                    memcpy(dst_n + ((size_t)c * HW + hw) * elem_size,
                           src_n + ((size_t)hw * C + c) * elem_size,
                           elem_size);
        }
    } else if (from == memory_layout_t::NCHW && to == memory_layout_t::NHWC) {
        for (int n = 0; n < N; n++) {
            const uint8_t* src_n = src + (size_t)n * C * HW * elem_size;
            uint8_t*       dst_n = dst + (size_t)n * HW * C * elem_size;
            for (int c = 0; c < C; c++)
                for (int hw = 0; hw < HW; hw++)
                    memcpy(dst_n + ((size_t)hw * C + c) * elem_size,
                           src_n + ((size_t)c * HW + hw) * elem_size,
                           elem_size);
        }
    }
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
