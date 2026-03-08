#pragma once
// AVX2/AVX-512 SIMD transpose kernels for layout conversion (NCHW <-> NHWC).

#include <immintrin.h>
#include "thread_pool.h"
#include "cpu_features.h"

namespace nnr {

// ---------------------------------------------------------------------------
// SIMD 8x8 transpose (AVX2): transposes an 8x8 block of floats.
// src[row] points to 8 consecutive floats per row, rows strided by src_stride.
// dst[col] writes 8 consecutive floats per col, cols strided by dst_stride.
// ---------------------------------------------------------------------------
// @nnr-meta isa=AVX2 dtype=fp32
inline void transpose_8x8_avx2(const float* src, int src_stride,
    float* dst, int dst_stride)
{
#if 0 // LOCUST
;for i in range(8):
    __m256 r@i@ = _mm256_loadu_ps(src + @i@ * src_stride);
;    pass
#else // LOCUST
    __m256 r0 = _mm256_loadu_ps(src + 0 * src_stride);
    __m256 r1 = _mm256_loadu_ps(src + 1 * src_stride);
    __m256 r2 = _mm256_loadu_ps(src + 2 * src_stride);
    __m256 r3 = _mm256_loadu_ps(src + 3 * src_stride);
    __m256 r4 = _mm256_loadu_ps(src + 4 * src_stride);
    __m256 r5 = _mm256_loadu_ps(src + 5 * src_stride);
    __m256 r6 = _mm256_loadu_ps(src + 6 * src_stride);
    __m256 r7 = _mm256_loadu_ps(src + 7 * src_stride);
#endif // LOCUST

    // Phase 1: interleave 32-bit within 128-bit lanes
#if 0 // LOCUST
;for i in range(0, 8, 2):
    __m256 t@i@ = _mm256_unpacklo_ps(r@i@, r@i+1@); // a0b0 a1b1 | a4b4 a5b5
    __m256 t@i+1@ = _mm256_unpackhi_ps(r@i@, r@i+1@); // a2b2 a3b3 | a6b6 a7b7
;    pass
#else // LOCUST
    __m256 t0 = _mm256_unpacklo_ps(r0, r1); // a0b0 a1b1 | a4b4 a5b5
    __m256 t1 = _mm256_unpackhi_ps(r0, r1); // a2b2 a3b3 | a6b6 a7b7
    __m256 t2 = _mm256_unpacklo_ps(r2, r3); // a0b0 a1b1 | a4b4 a5b5
    __m256 t3 = _mm256_unpackhi_ps(r2, r3); // a2b2 a3b3 | a6b6 a7b7
    __m256 t4 = _mm256_unpacklo_ps(r4, r5); // a0b0 a1b1 | a4b4 a5b5
    __m256 t5 = _mm256_unpackhi_ps(r4, r5); // a2b2 a3b3 | a6b6 a7b7
    __m256 t6 = _mm256_unpacklo_ps(r6, r7); // a0b0 a1b1 | a4b4 a5b5
    __m256 t7 = _mm256_unpackhi_ps(r6, r7); // a2b2 a3b3 | a6b6 a7b7
#endif // LOCUST

    // Phase 2: interleave 64-bit within 128-bit lanes
#if 0 // LOCUST
;for i in range(0, 8, 4):
    r@i@ = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(t@i@), _mm256_castps_pd(t@i+2@)));
    r@i+1@ = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(t@i@), _mm256_castps_pd(t@i+2@)));
    r@i+2@ = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(t@i+1@), _mm256_castps_pd(t@i+3@)));
    r@i+3@ = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(t@i+1@), _mm256_castps_pd(t@i+3@)));
;    pass
#else // LOCUST
    r0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(t0), _mm256_castps_pd(t2)));
    r1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(t0), _mm256_castps_pd(t2)));
    r2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(t1), _mm256_castps_pd(t3)));
    r3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(t1), _mm256_castps_pd(t3)));
    r4 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(t4), _mm256_castps_pd(t6)));
    r5 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(t4), _mm256_castps_pd(t6)));
    r6 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(t5), _mm256_castps_pd(t7)));
    r7 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(t5), _mm256_castps_pd(t7)));
#endif // LOCUST

    // Phase 3: swap 128-bit halves
#if 0 // LOCUST
;# row order after phase 2: r0,r1,r2,r3 = cols 0,2,1,3 (lo half), r4..r7 = cols 0,2,1,3 (hi half)
;order = [0,1,2,3,0,1,2,3]
;hi_src = [4,5,6,7,4,5,6,7]
;perm = ["0x20","0x20","0x20","0x20","0x31","0x31","0x31","0x31"]
;for i in range(8):
    _mm256_storeu_ps(dst + @i@ * dst_stride, _mm256_permute2f128_ps(r@order[i]@, r@hi_src[i]@, @perm[i]@));
;    pass
#else // LOCUST
    _mm256_storeu_ps(dst + 0 * dst_stride, _mm256_permute2f128_ps(r0, r4, 0x20));
    _mm256_storeu_ps(dst + 1 * dst_stride, _mm256_permute2f128_ps(r1, r5, 0x20));
    _mm256_storeu_ps(dst + 2 * dst_stride, _mm256_permute2f128_ps(r2, r6, 0x20));
    _mm256_storeu_ps(dst + 3 * dst_stride, _mm256_permute2f128_ps(r3, r7, 0x20));
    _mm256_storeu_ps(dst + 4 * dst_stride, _mm256_permute2f128_ps(r0, r4, 0x31));
    _mm256_storeu_ps(dst + 5 * dst_stride, _mm256_permute2f128_ps(r1, r5, 0x31));
    _mm256_storeu_ps(dst + 6 * dst_stride, _mm256_permute2f128_ps(r2, r6, 0x31));
    _mm256_storeu_ps(dst + 7 * dst_stride, _mm256_permute2f128_ps(r3, r7, 0x31));
#endif // LOCUST
}

// ---------------------------------------------------------------------------
// AVX-512 16x16 transpose using two-phase 8x8 approach.
// Loads 16 rows of 16 floats, transposes via vunpcklps/vunpckhps + vperm2f128
// in 256-bit sub-lanes, then 512-bit shuffle. Uses vshufi64x2 for final permute.
// ---------------------------------------------------------------------------
// @nnr-meta isa=AVX512 dtype=fp32
inline void transpose_16x16_avx512(const float* src, int src_stride,
    float* dst, int dst_stride)
{
    // Load 16 rows
    __m512 r[16];
    for (int i = 0; i < 16; i++)
        r[i] = _mm512_loadu_ps(src + i * src_stride);

    // Phase 1: interleave 32-bit pairs
    __m512 t[16];
    for (int i = 0; i < 16; i += 2) {
        t[i]     = _mm512_unpacklo_ps(r[i], r[i + 1]);
        t[i + 1] = _mm512_unpackhi_ps(r[i], r[i + 1]);
    }

    // Phase 2: interleave 64-bit pairs
    for (int i = 0; i < 16; i += 4) {
        r[i]     = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t[i]),     _mm512_castps_pd(t[i + 2])));
        r[i + 1] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t[i]),     _mm512_castps_pd(t[i + 2])));
        r[i + 2] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t[i + 1]), _mm512_castps_pd(t[i + 3])));
        r[i + 3] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t[i + 1]), _mm512_castps_pd(t[i + 3])));
    }

    // Phase 3: shuffle 128-bit lanes within 512-bit registers
    for (int i = 0; i < 16; i += 8) {
        t[i]     = _mm512_shuffle_f32x4(r[i],     r[i + 4], 0x88); // lanes 0,2 from both
        t[i + 1] = _mm512_shuffle_f32x4(r[i + 1], r[i + 5], 0x88);
        t[i + 2] = _mm512_shuffle_f32x4(r[i + 2], r[i + 6], 0x88);
        t[i + 3] = _mm512_shuffle_f32x4(r[i + 3], r[i + 7], 0x88);
        t[i + 4] = _mm512_shuffle_f32x4(r[i],     r[i + 4], 0xDD); // lanes 1,3 from both
        t[i + 5] = _mm512_shuffle_f32x4(r[i + 1], r[i + 5], 0xDD);
        t[i + 6] = _mm512_shuffle_f32x4(r[i + 2], r[i + 6], 0xDD);
        t[i + 7] = _mm512_shuffle_f32x4(r[i + 3], r[i + 7], 0xDD);
    }

    // Phase 4: final 256-bit shuffle across the two 8-column halves
#if 0 // LOCUST
;for i in range(8):
    _mm512_storeu_ps(dst + @i@  * dst_stride, _mm512_shuffle_f32x4(t[@i@],  t[@i+8@],  0x88));
;    pass
;for i in range(8):
    _mm512_storeu_ps(dst + @i+8@  * dst_stride, _mm512_shuffle_f32x4(t[@i@],  t[@i+8@],  0xDD));
;    pass
#else // LOCUST
    _mm512_storeu_ps(dst + 0  * dst_stride, _mm512_shuffle_f32x4(t[0],  t[8],  0x88));
    _mm512_storeu_ps(dst + 1  * dst_stride, _mm512_shuffle_f32x4(t[1],  t[9],  0x88));
    _mm512_storeu_ps(dst + 2  * dst_stride, _mm512_shuffle_f32x4(t[2],  t[10],  0x88));
    _mm512_storeu_ps(dst + 3  * dst_stride, _mm512_shuffle_f32x4(t[3],  t[11],  0x88));
    _mm512_storeu_ps(dst + 4  * dst_stride, _mm512_shuffle_f32x4(t[4],  t[12],  0x88));
    _mm512_storeu_ps(dst + 5  * dst_stride, _mm512_shuffle_f32x4(t[5],  t[13],  0x88));
    _mm512_storeu_ps(dst + 6  * dst_stride, _mm512_shuffle_f32x4(t[6],  t[14],  0x88));
    _mm512_storeu_ps(dst + 7  * dst_stride, _mm512_shuffle_f32x4(t[7],  t[15],  0x88));
    _mm512_storeu_ps(dst + 8  * dst_stride, _mm512_shuffle_f32x4(t[0],  t[8],  0xDD));
    _mm512_storeu_ps(dst + 9  * dst_stride, _mm512_shuffle_f32x4(t[1],  t[9],  0xDD));
    _mm512_storeu_ps(dst + 10  * dst_stride, _mm512_shuffle_f32x4(t[2],  t[10],  0xDD));
    _mm512_storeu_ps(dst + 11  * dst_stride, _mm512_shuffle_f32x4(t[3],  t[11],  0xDD));
    _mm512_storeu_ps(dst + 12  * dst_stride, _mm512_shuffle_f32x4(t[4],  t[12],  0xDD));
    _mm512_storeu_ps(dst + 13  * dst_stride, _mm512_shuffle_f32x4(t[5],  t[13],  0xDD));
    _mm512_storeu_ps(dst + 14  * dst_stride, _mm512_shuffle_f32x4(t[6],  t[14],  0xDD));
    _mm512_storeu_ps(dst + 15  * dst_stride, _mm512_shuffle_f32x4(t[7],  t[15],  0xDD));
#endif // LOCUST
}

// ---------------------------------------------------------------------------
// AVX-512 NCHW -> NCHWc block conversion using gather.
// Gathers 16 channels from 16 separate NCHW planes at each spatial position.
// src_planes: pointer to first channel plane at c_start * HW
// dst_block: [HW, 16] interleaved output
// HW: spatial size (H * W)
// ---------------------------------------------------------------------------
// @nnr-meta isa=AVX512 dtype=fp32 layout=[NCHW,BLOCKED_16] special=NCHWc
inline void nchw_to_nchwc_block16_avx512(float* __restrict dst_block,
    const float* __restrict src_planes, int HW)
{
    __m512i idx = _mm512_mullo_epi32(
        _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0),
        _mm512_set1_epi32(HW));
    for (int hw = 0; hw < HW; hw++) {
        __m512 v = _mm512_i32gather_ps(idx, src_planes + hw, 4);
        _mm512_storeu_ps(dst_block + hw * 16, v);
    }
}

// ---------------------------------------------------------------------------
// AVX-512 NCHWc -> NCHW block conversion using 16x16 transpose.
// Inverse of nchw_to_nchwc_block16_avx512.
// src_block:  [HW, 16] interleaved input  (NCHWc layout for one channel block)
// dst_planes: pointer to first channel plane at c_start * HW  (NCHW output)
// HW: spatial size (H * W)
// ---------------------------------------------------------------------------
// @nnr-meta isa=AVX512 dtype=fp32 layout=[BLOCKED_16,NCHW] special=NCHWc
inline void nchwc_to_nchw_block16_avx512(float* __restrict dst_planes,
    const float* __restrict src_block, int HW)
{
    // Main body: transpose 16 spatial positions at a time.
    // src is [HW, 16] (stride 16), dst is [16, HW] (stride HW).
    int hw = 0;
    for (; hw + 16 <= HW; hw += 16) {
        transpose_16x16_avx512(src_block + hw * 16, 16, dst_planes + hw, HW);
    }
    // Scalar remainder
    for (; hw < HW; hw++) {
        for (int c = 0; c < 16; c++)
            dst_planes[c * HW + hw] = src_block[hw * 16 + c];
    }
}

// Tiled 2D matrix transpose using AVX-512 16x16 blocks.
// Transposes src[M x N] -> dst[N x M]. Threaded over tiles.
// @nnr-meta isa=AVX512 dtype=fp32
inline void transpose_2d_avx512(const float* src, float* dst, int M, int N)
{
    constexpr int TILE = 16;
    int total_work = ((M + TILE - 1) / TILE) * ((N + TILE - 1) / TILE);
    int tile_cols = (N + TILE - 1) / TILE;
    nnr::for_static(0, total_work, total_work > 4, [&](int work) {
        int ti = work / tile_cols;
        int tj = work % tile_cols;
        int i0 = ti * TILE, j0 = tj * TILE;
        int irem = M - i0, jrem = N - j0;
        if (irem >= TILE && jrem >= TILE) {
            transpose_16x16_avx512(src + i0 * N + j0, N, dst + j0 * M + i0, M);
        } else {
            int imax = std::min(i0 + TILE, M);
            int jmax = std::min(j0 + TILE, N);
            for (int i = i0; i < imax; ++i)
                for (int j = j0; j < jmax; ++j)
                    dst[j * M + i] = src[i * N + j];
        }
    });
}

// Transpose batch [C x HW] -> [HW x C] using AVX-512/AVX2 blocks.
// src_n: [C][HW], dst_n: [HW][C], processes hw_begin..hw_end range.
// @nnr-meta isa=[AVX512,AVX2] dtype=fp32 layout=[NCHW,NHWC]
inline void nchw_nhwc_batch_x64(const float* src_n, float* dst_n,
    int C, int HW, int hw_begin, int hw_end)
{
    const int hw_range = hw_end - hw_begin;
    if (has_avx512() && C >= 16 && hw_range >= 16) {
        int hw, c;
        for (hw = 0; hw + 16 <= hw_range; hw += 16) {
            for (c = 0; c + 16 <= C; c += 16) {
                transpose_16x16_avx512(
                    src_n + (size_t)c * HW + hw_begin + hw, HW,
                    dst_n + (size_t)(hw_begin + hw) * C + c, C);
            }
            for (; c < C; c++) {
                const float* sc = src_n + (size_t)c * HW + hw_begin + hw;
                for (int h2 = 0; h2 < 16; h2++)
                    dst_n[(hw_begin + hw + h2) * C + c] = sc[h2];
            }
        }
        for (; hw < hw_range; hw++) {
            for (c = 0; c < C; c++)
                dst_n[(hw_begin + hw) * C + c] = src_n[(size_t)c * HW + hw_begin + hw];
        }
    } else if (detect_isa() >= isa_t::avx2 && C >= 8 && hw_range >= 8) {
        int hw, c;
        for (hw = 0; hw + 8 <= hw_range; hw += 8) {
            for (c = 0; c + 8 <= C; c += 8) {
                transpose_8x8_avx2(
                    src_n + (size_t)c * HW + hw_begin + hw, HW,
                    dst_n + (size_t)(hw_begin + hw) * C + c, C);
            }
            for (; c < C; c++) {
                const float* sc = src_n + (size_t)c * HW + hw_begin + hw;
                for (int h2 = 0; h2 < 8; h2++)
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

// Transpose batch [HW x C] -> [C x HW] using AVX-512/AVX2 blocks.
// @nnr-meta isa=[AVX512,AVX2] dtype=fp32 layout=[NHWC,NCHW]
inline void nhwc_nchw_batch_x64(const float* src_n, float* dst_n,
    int C, int HW, int hw_begin, int hw_end)
{
    const int hw_range = hw_end - hw_begin;
    if (has_avx512() && C >= 16 && hw_range >= 16) {
        int hw, c;
        for (hw = 0; hw + 16 <= hw_range; hw += 16) {
            for (c = 0; c + 16 <= C; c += 16) {
                transpose_16x16_avx512(
                    src_n + (size_t)(hw_begin + hw) * C + c, C,
                    dst_n + (size_t)c * HW + hw_begin + hw, HW);
            }
            for (; c < C; c++) {
                float* dc = dst_n + (size_t)c * HW + hw_begin + hw;
                for (int h2 = 0; h2 < 16; h2++)
                    dc[h2] = src_n[(hw_begin + hw + h2) * C + c];
            }
        }
        for (; hw < hw_range; hw++) {
            for (c = 0; c < C; c++)
                dst_n[(size_t)c * HW + hw_begin + hw] = src_n[(hw_begin + hw) * C + c];
        }
    } else if (detect_isa() >= isa_t::avx2 && C >= 8 && hw_range >= 8) {
        int hw, c;
        for (hw = 0; hw + 8 <= hw_range; hw += 8) {
            for (c = 0; c + 8 <= C; c += 8) {
                transpose_8x8_avx2(
                    src_n + (size_t)(hw_begin + hw) * C + c, C,
                    dst_n + (size_t)c * HW + hw_begin + hw, HW);
            }
            for (; c < C; c++) {
                float* dc = dst_n + (size_t)c * HW + hw_begin + hw;
                for (int h2 = 0; h2 < 8; h2++)
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
inline int layout_block_align_x64() {
    return has_avx512() ? 16 : 8;
}

} // namespace nnr
