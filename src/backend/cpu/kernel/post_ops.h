#pragma once
// Standalone post-op functions for codegen fusion.
// Match operator_t::post_fn_t signature so they can be used with gemm_post_t.

#include "nnr.h"
#include "cpu_features.h"

#ifdef NNR_ARCH_X64
#include "backend/x64/post_ops_x64.h"
#elifdef NNR_ARCH_ARM64
#include "backend/arm/post_ops_neon.h"
#endif

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace nnr {

// Tag for register-fusible post-ops. The micro-kernel branches on this
// once per tile to apply bias+activation on accumulators before the store,
// eliminating the post-op load-store round trip.
enum class post_op_kind : uint8_t {
    none,       // opaque post_fn or no post-op -- fall through to apply_rows
    bias_only,  // c += bias[row]
    relu,       // c = max(c + bias[row], 0)
    clip,       // c = clamp(c + bias[row], lo, hi)
};

// Relu post-op: max(0, x + bias)
inline void relu_post_fn(float* data, int rows, int cols, int stride,
                          const operator_t*, const float* bias, int) {
    const auto isa = detect_isa();
    for (int r = 0; r < rows; r++) {
        float* row = data + (size_t)r * stride;
        float bv = bias ? bias[r] : 0.0f;
#ifdef NNR_ARCH_X64
        relu_post_row_x64(row, cols, bv, isa);
#elifdef NNR_ARCH_ARM64
        relu_post_row_neon(row, cols, bv, isa);
#else
        for (int i = 0; i < cols; ++i)
            row[i] = std::max(0.0f, row[i] + bv);
#endif
    }
}

// Clip params — reinterpret_cast'd from operator_t* in clip_post_fn
struct clip_post_params_t {
    float min_val;
    float max_val;
};

// Clip post-op: clamp(x + bias, min, max)
// fused_op must point to a clip_post_params_t
inline void clip_post_fn(float* data, int rows, int cols, int stride,
                          const operator_t* op, const float* bias, int) {
    auto* p = reinterpret_cast<const clip_post_params_t*>(op);
    float lo = p->min_val;
    float hi = p->max_val;
    const auto isa = detect_isa();
    for (int r = 0; r < rows; r++) {
        float* row = data + (size_t)r * stride;
        float bv = bias ? bias[r] : 0.0f;
#ifdef NNR_ARCH_X64
        clip_post_row_x64(row, cols, bv, lo, hi, isa);
#elifdef NNR_ARCH_ARM64
        clip_post_row_neon(row, cols, bv, lo, hi, isa);
#else
        for (int i = 0; i < cols; ++i)
            row[i] = std::clamp(row[i] + bv, lo, hi);
#endif
    }
}

// Sigmoid post-op: 1 / (1 + exp(-(x + bias)))
inline void sigmoid_post_fn(float* data, int rows, int cols, int stride,
                              const operator_t*, const float* bias, int) {
    for (int r = 0; r < rows; r++) {
        float* row = data + (size_t)r * stride;
        float bv = bias ? bias[r] : 0.0f;
        for (int i = 0; i < cols; ++i)
            row[i] = 1.0f / (1.0f + expf(-(row[i] + bv)));
    }
}

// Tanh post-op: tanh(x + bias)
inline void tanh_post_fn(float* data, int rows, int cols, int stride,
                           const operator_t*, const float* bias, int) {
    for (int r = 0; r < rows; r++) {
        float* row = data + (size_t)r * stride;
        float bv = bias ? bias[r] : 0.0f;
        for (int i = 0; i < cols; ++i)
            row[i] = tanhf(row[i] + bv);
    }
}

} // namespace nnr
