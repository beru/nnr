// DECLS-PLACEHOLDER wrapper (not a format string)
#pragma once
// Internal forward declarations for graph_optimizer passes.
//
// The optimizer was split in 2026-04-10 from a single 3,303-line
// graph_optimizer.cpp into per-pass translation units. This header is the
// glue: each pass lives in its own .cpp under src/graph_optimizer/ and is
// called from graph_optimizer.cpp::optimize() / ::preprocess().
//
// DO NOT include this header outside src/graph_optimizer/. It is not part
// of the public NNR API.

#include "nnr.h"
#include "float16.h"
#include "graph_optimizer.h"
#include "layout_cost.h"
#include "backend/cpu/solve_operator.h"
#include "cpu_features.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace nnr {

// --- Pass forward declarations (one per free function moved out) -----------
void decompose_ops(context_t* ctx);
void fuse_conv_bn(context_t* ctx);
void fuse_pad(context_t* ctx);
void fuse_post_ops(context_t* ctx);
void fuse_post_ops_silu(context_t* ctx);
void fold_bn_qdq(context_t* ctx);
void fuse_qdq_compute(context_t* ctx);
void fuse_qdq(context_t* ctx);
void fuse_silu(context_t* ctx);
void fuse_layer_norm(context_t* ctx);
void fuse_gelu(context_t* ctx);
void fuse_sdpa(context_t* ctx);
void fuse_webgpu_elementwise(context_t* ctx);
void fuse_webgpu_matmul_chain(context_t* ctx);
void fold_constants(context_t* ctx);
void detect_scroll_chains(graph_optimizer_t* opt, context_t* ctx);
void assign_layouts(context_t* ctx);
void assign_blocked_layouts(context_t* ctx);
void insert_reorders(context_t* ctx);
void cancel_reorders(context_t* ctx);
void optimize_transposes(context_t* ctx);

} // namespace nnr
