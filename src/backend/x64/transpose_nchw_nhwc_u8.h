#pragma once
// Fast uint8 NCHW → NHWC transpose for QLinearConv input packing.
//
// Input  : src[C][HW]   (row-major, contiguous C × HW bytes)
// Output : dst[HW][C]   (row-major, contiguous HW × C bytes)
//
// The scalar reference at QLinearConv.cpp:742 was the single biggest
// densenet-12-int8 bottleneck: it does one-byte strided stores into the
// NHWC buffer, thrashing L2 on every IC sweep. This helper replaces it
// with a 16×16 SSE byte transpose tiled over (HW_block, C_block), which
// is bandwidth-bound.

#include <cstdint>
#include <cstring>
#include <algorithm>

#ifdef NNR_ARCH_X64
#include <immintrin.h>
#endif

namespace nnr { namespace int8 {

#ifdef NNR_ARCH_X64
// Transpose a 16×16 byte tile: load 16 rows of 16 bytes each from src (with
// src_stride bytes between rows) and store 16 rows of 16 bytes each to dst
// (with dst_stride bytes between rows). Uses 4 stages of unpack: after stage
// k, lanes at pair-distance 2^k have been interleaved. Result: output row i
// holds byte i from every input row, i.e. a row/column swap.
// @nnr-meta isa=SSE dtype=uint8 layout=NCHW
static inline void transpose_16x16_u8(
    uint8_t* __restrict dst, size_t dst_stride,
    const uint8_t* __restrict src, size_t src_stride)
{
    __m128i r[16];
    for (int i = 0; i < 16; i++)
        r[i] = _mm_loadu_si128((const __m128i*)(src + (size_t)i * src_stride));

    // Stage 1: interleave byte pairs from adjacent rows.
    __m128i s[16];
    for (int i = 0; i < 8; i++) {
        s[i * 2]     = _mm_unpacklo_epi8(r[i * 2], r[i * 2 + 1]);
        s[i * 2 + 1] = _mm_unpackhi_epi8(r[i * 2], r[i * 2 + 1]);
    }
    // Stage 2: interleave 16-bit pairs from adjacent (stage-1) rows.
    for (int i = 0; i < 4; i++) {
        r[i * 4 + 0] = _mm_unpacklo_epi16(s[i * 4 + 0], s[i * 4 + 2]);
        r[i * 4 + 1] = _mm_unpackhi_epi16(s[i * 4 + 0], s[i * 4 + 2]);
        r[i * 4 + 2] = _mm_unpacklo_epi16(s[i * 4 + 1], s[i * 4 + 3]);
        r[i * 4 + 3] = _mm_unpackhi_epi16(s[i * 4 + 1], s[i * 4 + 3]);
    }
    // Stage 3: interleave 32-bit pairs.
    for (int i = 0; i < 2; i++) {
        s[i * 8 + 0] = _mm_unpacklo_epi32(r[i * 8 + 0], r[i * 8 + 4]);
        s[i * 8 + 1] = _mm_unpackhi_epi32(r[i * 8 + 0], r[i * 8 + 4]);
        s[i * 8 + 2] = _mm_unpacklo_epi32(r[i * 8 + 1], r[i * 8 + 5]);
        s[i * 8 + 3] = _mm_unpackhi_epi32(r[i * 8 + 1], r[i * 8 + 5]);
        s[i * 8 + 4] = _mm_unpacklo_epi32(r[i * 8 + 2], r[i * 8 + 6]);
        s[i * 8 + 5] = _mm_unpackhi_epi32(r[i * 8 + 2], r[i * 8 + 6]);
        s[i * 8 + 6] = _mm_unpacklo_epi32(r[i * 8 + 3], r[i * 8 + 7]);
        s[i * 8 + 7] = _mm_unpackhi_epi32(r[i * 8 + 3], r[i * 8 + 7]);
    }
    // Stage 4: interleave 64-bit halves.
    for (int i = 0; i < 8; i++) {
        r[i]     = _mm_unpacklo_epi64(s[i], s[i + 8]);
        r[i + 8] = _mm_unpackhi_epi64(s[i], s[i + 8]);
    }

    // Stores: the stage-4 output isn't in straight row order. Trace:
    //   r[i] = unpacklo_epi64(s[i], s[i+8])
    //   r[i+8] = unpackhi_epi64(s[i], s[i+8])
    // where s[i] for i∈[0..7] holds rows 0..7 of transposed columns (2i, 2i+1)
    // and s[i+8] holds rows 8..15 of the same pair. So:
    //   r[0]=row0, r[1]=row2, r[2]=row4, r[3]=row6, r[4]=row8, r[5]=row10,
    //   r[6]=row12, r[7]=row14, r[8]=row1, r[9]=row3, ..., r[15]=row15.
    static constexpr int perm[16] = {
        0, 2, 4, 6, 8, 10, 12, 14,
        1, 3, 5, 7, 9, 11, 13, 15,
    };
    for (int i = 0; i < 16; i++)
        _mm_storeu_si128((__m128i*)(dst + (size_t)perm[i] * dst_stride), r[i]);
}
#endif

// NCHW → NHWC for uint8 [C, HW] → [HW, C]. Handles arbitrary C, HW; the
// 16×16 fast path covers the bulk, a scalar tail handles the edges.
// @nnr-meta isa=SSE dtype=uint8 layout=NCHW
inline void transpose_nchw_to_nhwc_u8(
    uint8_t* __restrict dst,
    const uint8_t* __restrict src,
    int C, int HW)
{
#ifdef NNR_ARCH_X64
    int c0 = 0;
    for (; c0 + 16 <= C; c0 += 16) {
        const uint8_t* src_c = src + (size_t)c0 * HW;
        uint8_t* dst_c = dst + c0;
        int p0 = 0;
        for (; p0 + 16 <= HW; p0 += 16) {
            transpose_16x16_u8(
                dst_c + (size_t)p0 * C, (size_t)C,
                src_c + p0, (size_t)HW);
        }
        // Scalar tail over the HW dimension.
        for (int pi = p0; pi < HW; pi++) {
            uint8_t* d = dst_c + (size_t)pi * C;
            for (int ci = 0; ci < 16; ci++)
                d[ci] = src_c[(size_t)ci * HW + pi];
        }
    }
    // Scalar tail over the C dimension.
    for (int ci = c0; ci < C; ci++) {
        const uint8_t* s = src + (size_t)ci * HW;
        uint8_t* d = dst + ci;
        for (int p = 0; p < HW; p++)
            d[(size_t)p * C] = s[p];
    }
#else
    for (int ci = 0; ci < C; ci++) {
        const uint8_t* s = src + (size_t)ci * HW;
        for (int p = 0; p < HW; p++)
            dst[(size_t)p * C + ci] = s[p];
    }
#endif
}

} }  // namespace nnr::int8
