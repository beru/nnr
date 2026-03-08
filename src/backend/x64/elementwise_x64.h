#pragma once
// x64 ISA dispatch for elementwise operations.
// Called from kernel/elementwise.h.

#ifdef NNR_ARCH_X64
#include "backend/x64/vec_ops_avx512.h"
#include "backend/x64/vec_ops_avx2.h"
#include "cpu_features.h"
#include <cfloat>

namespace nnr {

// @nnr-meta isa=[AVX512,AVX2] dtype=fp32
inline void relu_inplace_x64(float* data, int len) {
    if (has_avx512()) {
        avx512::clip(data, data, len, 0.0f, FLT_MAX);
    } else if (detect_isa() == isa_t::avx2) {
        avx2::clip(data, data, len, 0.0f, FLT_MAX);
    } else {
        for (int i = 0; i < len; ++i)
            if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

// @nnr-meta isa=[AVX512,AVX2] dtype=fp32
inline void clip_inplace_x64(float* data, int len, float lo, float hi) {
    if (has_avx512()) {
        avx512::clip(data, data, len, lo, hi);
    } else if (detect_isa() == isa_t::avx2) {
        avx2::clip(data, data, len, lo, hi);
    } else {
        for (int i = 0; i < len; ++i)
            data[i] = std::max(lo, std::min(hi, data[i]));
    }
}

// @nnr-meta isa=[AVX512,AVX2] dtype=fp32
inline void leaky_relu_inplace_x64(float* data, int len, float alpha) {
    if (has_avx512()) {
        avx512::leaky_relu(data, data, len, alpha);
    } else if (detect_isa() == isa_t::avx2) {
        avx2::leaky_relu(data, data, len, alpha);
    } else {
        for (int i = 0; i < len; ++i)
            if (data[i] < 0.0f) data[i] *= alpha;
    }
}

} // namespace nnr

#endif // NNR_ARCH_X64
