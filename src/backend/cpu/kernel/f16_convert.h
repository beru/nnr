#pragma once
// ISA-dispatch wrapper for bulk float16/bfloat16 <-> float32 conversion.
// Routes to F16C/AVX2 (x64) or NEON (ARM) SIMD, with scalar fallback.

#include "nnr.h"
#include "float16.h"
#include "bfloat16.h"

#ifdef NNR_ARCH_X64
#include "backend/x64/f16_convert_x64.h"
#elifdef NNR_ARCH_ARM64
#include "backend/arm/f16_convert_neon.h"
#endif

namespace nnr {

inline void convert_f16_to_f32(float* __restrict dst, const float16_t* __restrict src, size_t n)
{
#ifdef NNR_ARCH_X64
    x64::f16_to_f32(dst, reinterpret_cast<const uint16_t*>(src), (int)n);
#elifdef NNR_ARCH_ARM64
    neon::f16_to_f32(dst, reinterpret_cast<const uint16_t*>(src), (int)n);
#else
    for (size_t i = 0; i < n; i++)
        dst[i] = (float)src[i];
#endif
}

inline void convert_f32_to_f16(float16_t* __restrict dst, const float* __restrict src, size_t n)
{
#ifdef NNR_ARCH_X64
    x64::f32_to_f16(reinterpret_cast<uint16_t*>(dst), src, (int)n);
#elifdef NNR_ARCH_ARM64
    neon::f32_to_f16(reinterpret_cast<uint16_t*>(dst), src, (int)n);
#else
    for (size_t i = 0; i < n; i++)
        dst[i] = float16_t(src[i]);
#endif
}

// ── BF16 ↔ FP32 ──────────────────────────────────────────────────────────────

inline void convert_bf16_to_f32(float* __restrict dst, const bfloat16_t* __restrict src, size_t n)
{
#ifdef NNR_ARCH_X64
    x64::bf16_to_f32(dst, reinterpret_cast<const uint16_t*>(src), (int)n);
#else
    for (size_t i = 0; i < n; i++)
        dst[i] = (float)src[i];
#endif
}

inline void convert_f32_to_bf16(bfloat16_t* __restrict dst, const float* __restrict src, size_t n)
{
#ifdef NNR_ARCH_X64
    x64::f32_to_bf16(reinterpret_cast<uint16_t*>(dst), src, (int)n);
#else
    for (size_t i = 0; i < n; i++)
        dst[i] = bfloat16_t(src[i]);
#endif
}

} // namespace nnr
