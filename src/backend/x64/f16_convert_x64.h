#pragma once
// F16C SIMD conversion: float16 <-> float32.
// 8 elements per iteration using _mm256_cvtph_ps / _mm256_cvtps_ph.
// F16C is available on all x64 CPUs since Ivy Bridge (2012).

#include <immintrin.h>
#include "nnrconf.h"

namespace nnr::x64 {

// @nnr-meta isa=AVX2 dtype=[fp16,fp32]
inline void f16_to_f32(float* __restrict dst, const uint16_t* __restrict src, int n)
{
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i*)(src + i));
        __m256 f = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(dst + i, f);
    }
    for (; i < n; i++)
        dst[i] = float16_to_float32(src[i]);
}

// @nnr-meta isa=AVX2 dtype=[fp32,fp16]
inline void f32_to_f16(uint16_t* __restrict dst, const float* __restrict src, int n)
{
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 f = _mm256_loadu_ps(src + i);
        __m128i h = _mm256_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i*)(dst + i), h);
    }
    for (; i < n; i++)
        dst[i] = float32_to_float16(src[i]);
}

// BF16 ↔ FP32 bulk conversion using AVX2 bit-shift.
// BF16 is the upper 16 bits of FP32, so conversion is just shift left/right by 16.

// @nnr-meta isa=AVX2 dtype=[bf16,fp32]
inline void bf16_to_f32(float* __restrict dst, const uint16_t* __restrict src, int n)
{
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i*)(src + i));
        __m256i w = _mm256_cvtepu16_epi32(h);          // zero-extend 16→32
        __m256i shifted = _mm256_slli_epi32(w, 16);     // shift left 16 = BF16→FP32
        _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(shifted));
    }
    for (; i < n; i++)
        dst[i] = bfloat16_to_float32(src[i]);
}

// @nnr-meta isa=AVX2 dtype=[fp32,bf16]
inline void f32_to_bf16(uint16_t* __restrict dst, const float* __restrict src, int n)
{
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i u = _mm256_loadu_si256((const __m256i*)(src + i));
        // Round-to-nearest-even: add rounding bias (bit 16 + 0x7FFF)
        __m256i lsb = _mm256_srli_epi32(u, 16);
        lsb = _mm256_and_si256(lsb, _mm256_set1_epi32(1));
        __m256i bias = _mm256_add_epi32(lsb, _mm256_set1_epi32(0x7FFF));
        __m256i rounded = _mm256_add_epi32(u, bias);
        __m256i shifted = _mm256_srli_epi32(rounded, 16);
        // Pack 32-bit → 16-bit: use packs with unsigned saturation
        __m128i lo = _mm256_castsi256_si128(shifted);
        __m128i hi = _mm256_extracti128_si256(shifted, 1);
        __m128i packed = _mm_packus_epi32(lo, hi);
        _mm_storeu_si128((__m128i*)(dst + i), packed);
    }
    for (; i < n; i++)
        dst[i] = float32_to_bfloat16(src[i]);
}

} // namespace nnr::x64
