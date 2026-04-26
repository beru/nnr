#pragma once
// Shared cost model utilities for layout assignment.
// Used by operator estimate_costs() overrides and graph_optimizer.
// op_cost_t is defined in nnr.h (core type used in virtual interface).

#include <algorithm>
#include <cmath>

namespace nnr {

// --- Cache efficiency utilities ---


constexpr int COST_CACHE_LINE = 64;
constexpr float COST_NCHW_STRIDE_FLOOR = 0.50f;

// NHWC reads C channels per pixel: utilization = min(1, C*4/cacheline).
inline float nhwc_patch_util(int C) {
    return std::min(1.0f, C * 4.0f / COST_CACHE_LINE);
}

// NCHW reads stride-W across rows: utilization >= floor, improves with larger HW.
inline float nchw_stride_util(int H, int W) {
    return std::max(COST_NCHW_STRIDE_FLOOR, 1.0f / (float)(H * W));
}

// BLOCKED (NCHWc) reads channel-blocks: each c-block load is exactly one
// SIMD register / cache line. Utilization = C / round_up(C, block) — i.e.
// 1.0 when C is block-aligned (the chain-eligibility gate enforces this on
// input C, so eligible chains see util == 1). For ineligible chains the
// IC-tail waste lowers util proportionally.
inline float block_simd_util(int C, int block) {
    if (block <= 0) return 1.0f;
    int padded = ((C + block - 1) / block) * block;
    if (padded <= 0) return 1.0f;
    return (float)C / (float)padded;
}

// --- Empirical penalties (AVX-512) ---


constexpr float GEMM_NHWC_PENALTY = 1.15f;
constexpr float GEMM_NHWC_1x1_PENALTY = 1.00f;
constexpr float WINO_NHWC_PENALTY = 1.10f;

// --- Reorder cost utilities (tensor-level, used by graph_optimizer) ---

// NCHW<->NHWC reorder: memcpy + block-transpose ~ 2.5x data size.
// Used by assign_layouts.cpp (NHWC pass). The NCHWc pass no longer uses
// a cost model — see assign_blocked_layouts.cpp for the ORT-style gate.
inline float reorder_cost(const tensor_t* t) {
    if (!t || t->ndim != 4) return 0;
    float bytes = (float)t->dims[0] * t->dims[1] * t->dims[2] * t->dims[3] * 4;
    return 2.5f * bytes;
}

// --- Scalar reduction (used by layout_cost() wrapper) ---

// Reduce op_cost_t to a single "effective bytes" scalar for layout comparison.
inline float reduce_to_scalar(const op_cost_t& c) {
    float seq = std::max(0.1f, c.read_sequential);
    return c.read_bytes / seq + c.write_bytes;
}

} // namespace nnr
