// Shared graph-walking helpers used by the QDQ fusion / fold passes
// (fuse_qdq.cpp, fuse_qdq_compute.cpp, fold_bn_qdq.cpp) and the layout
// assignment passes (assign_blocked_layouts.cpp, assign_layouts.cpp).
//
// Two primitives:
//   1. tensor → producing-node-index map (with optional skip/folded filter)
//   2. count of active consumers of a given tensor
//
// Header-only (inline). DO NOT include outside src/graph_optimizer/. Not
// part of the public NNR API.
//
// Note: the scale/zero-point extraction logic that appears in fuse_qdq.cpp
// and fold_bn_qdq.cpp is intentionally NOT extracted here — the call sites
// have subtly different acceptance criteria (fuse_qdq accepts FLOAT16
// scales; fold_bn_qdq requires FLOAT32 because the rest of the BN math is
// FLOAT32-only). Unifying them would silently broaden fold_bn_qdq's input
// surface.

#pragma once

#include "nnr.h"

#include <unordered_map>
#include <vector>

namespace nnr::qdq_helpers {

// Build a tensor → producing-node-index map.
// If `include_folded` is true, includes outputs of skip/folded nodes too —
// used by fuse_qdq_compute to track weight DQs that fold_run pre-folded
// before the optimizer runs, and by the layout passes which inspect every
// graph edge regardless of liveness.
inline std::unordered_map<tensor_t*, int>
build_producer_map(const std::vector<operator_t*>& nodes,
                   bool include_folded = false)
{
    std::unordered_map<tensor_t*, int> producer;
    const int n = static_cast<int>(nodes.size());
    for (int i = 0; i < n; i++) {
        if (!include_folded && (nodes[i]->skip || nodes[i]->folded)) continue;
        for (auto* t : nodes[i]->outputs)
            if (t) producer[t] = i;
    }
    return producer;
}

// Count active consumers of `tensor`, excluding node `skip_idx` and any
// skip/folded nodes.
inline int count_consumers(const std::vector<operator_t*>& nodes,
                           tensor_t* tensor, int skip_idx)
{
    int count = 0;
    const int n = static_cast<int>(nodes.size());
    for (int j = 0; j < n; j++) {
        if (j == skip_idx || nodes[j]->skip || nodes[j]->folded) continue;
        for (auto* t : nodes[j]->inputs)
            if (t == tensor) count++;
    }
    return count;
}

} // namespace nnr::qdq_helpers
