#pragma once
// x64 AVX-512/AVX2 dispatch for post-op row processing.
// Called from kernel/post_ops.h.

#ifdef NNR_ARCH_X64
#include "backend/x64/vec_ops_avx512.h"
#include "backend/x64/vec_ops_avx2.h"
#include "cpu_features.h"
#include <cfloat>

namespace nnr {

// @nnr-meta isa=[AVX2,AVX512] dtype=fp32 fusion=post_op
inline void relu_post_row_x64(float* row, int cols, float bv, isa_t isa) {
    if (isa == isa_t::avx512) {
        if (bv != 0.0f)
            avx512::bias_relu(row, cols, bv);
        else
            avx512::clip(row, row, cols, 0.0f, FLT_MAX);
    } else if (isa == isa_t::avx2) {
        if (bv != 0.0f)
            avx2::bias_relu(row, cols, bv);
        else
            avx2::clip(row, row, cols, 0.0f, FLT_MAX);
    } else {
        for (int i = 0; i < cols; ++i)
            row[i] = std::max(0.0f, row[i] + bv);
    }
}

// @nnr-meta isa=[AVX2,AVX512] dtype=fp32 fusion=post_op
inline void clip_post_row_x64(float* row, int cols, float bv, float lo, float hi, isa_t isa) {
    if (isa == isa_t::avx512) {
        if (bv != 0.0f)
            avx512::bias_clip(row, cols, bv, lo, hi);
        else
            avx512::clip(row, row, cols, lo, hi);
    } else if (isa == isa_t::avx2) {
        if (bv != 0.0f)
            avx2::bias_clip(row, cols, bv, lo, hi);
        else
            avx2::clip(row, row, cols, lo, hi);
    } else {
        for (int i = 0; i < cols; ++i)
            row[i] = std::clamp(row[i] + bv, lo, hi);
    }
}

} // namespace nnr

#endif // NNR_ARCH_X64
