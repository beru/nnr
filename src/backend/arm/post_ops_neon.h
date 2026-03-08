#pragma once
// ARM NEON dispatch for post-op row processing.
// Called from kernel/post_ops.h.

#ifdef NNR_ARCH_ARM64
#include "backend/arm/vec_ops_neon.h"
#include "cpu_features.h"
#include <cfloat>

namespace nnr {

// @nnr-meta isa=NEON dtype=fp32
inline void relu_post_row_neon(float* row, int cols, float bv, isa_t isa) {
    if (isa == isa_t::neon) {
        if (bv != 0.0f)
            neon::bias_relu(row, cols, bv);
        else
            neon::clip(row, row, cols, 0.0f, FLT_MAX);
    } else {
        for (int i = 0; i < cols; ++i)
            row[i] = std::max(0.0f, row[i] + bv);
    }
}

// @nnr-meta isa=NEON dtype=fp32
inline void clip_post_row_neon(float* row, int cols, float bv, float lo, float hi, isa_t isa) {
    if (isa == isa_t::neon) {
        if (bv != 0.0f)
            neon::bias_clip(row, cols, bv, lo, hi);
        else
            neon::clip(row, row, cols, lo, hi);
    } else {
        for (int i = 0; i < cols; ++i)
            row[i] = std::clamp(row[i] + bv, lo, hi);
    }
}

} // namespace nnr

#endif // NNR_ARCH_ARM64
