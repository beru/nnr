#pragma once
// ARM NEON conversion: float16 <-> float32.
// 4 elements per iteration using vcvt_f32_f16 / vcvt_f16_f32.

#if defined(__aarch64__) || defined(_M_ARM64)

#ifdef _MSC_VER
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif
#include "nnrconf.h"

namespace nnr::neon {

// @nnr-meta isa=NEON dtype=[fp16,fp32]
inline void f16_to_f32(float* __restrict dst, const uint16_t* __restrict src, int n)
{
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float16x4_t h = vreinterpret_f16_u16(vld1_u16(src + i));
        float32x4_t f = vcvt_f32_f16(h);
        vst1q_f32(dst + i, f);
    }
    for (; i < n; i++)
        dst[i] = float16_to_float32(src[i]);
}

// @nnr-meta isa=NEON dtype=[fp32,fp16]
inline void f32_to_f16(uint16_t* __restrict dst, const float* __restrict src, int n)
{
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t f = vld1q_f32(src + i);
        float16x4_t h = vcvt_f16_f32(f);
        vst1_u16(dst + i, vreinterpret_u16_f16(h));
    }
    for (; i < n; i++)
        dst[i] = float32_to_float16(src[i]);
}

} // namespace nnr::neon

#endif // __aarch64__ || _M_ARM64
