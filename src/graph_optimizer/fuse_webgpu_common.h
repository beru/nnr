// Shared utilities for fuse_webgpu_elementwise.cpp and
// fuse_webgpu_matmul_chain.cpp. Header-only (anonymous namespace inline);
// included only from those two TUs.
//
// DO NOT include outside src/graph_optimizer/. Not part of the public API.

#pragma once

#ifdef NNR_ENABLE_WEBGPU

#include "nnr.h"

#include <cstdlib>
#include <string>

namespace nnr {
namespace {

// Substitutes every `$s` in pattern with side_var. `$` can't start an
// identifier in WGSL, so a raw find-and-replace is unambiguous.
inline std::string substitute_side(const char* pattern, const std::string& side_var)
{
    std::string out;
    for (const char* p = pattern; *p; ) {
        if (p[0] == '$' && p[1] == 's') {
            out += side_var;
            p += 2;
        } else {
            out += *p++;
        }
    }
    return out;
}

inline bool same_shape(const tensor_t* a, const tensor_t* b)
{
    if (!a || !b) return false;
    if (a->ndim != b->ndim) return false;
    if (a->ndata != b->ndata) return false;
    for (int i = 0; i < a->ndim; ++i)
        if (a->dims[i] != b->dims[i]) return false;
    return true;
}

// True iff `side` is broadcastable *into* `pipe` without changing the
// output shape. Required: side.ndim ≤ pipe.ndim, and each right-aligned
// side axis is either 1 or equal to the pipe's corresponding axis. Same
// shape is a special case (returns true, but same_shape() captures it
// separately for the fast Path U fall-through).
inline bool side_broadcasts_into(const tensor_t* side, const tensor_t* pipe)
{
    if (!side || !pipe) return false;
    if (side->ndim > pipe->ndim) return false;
    int off = pipe->ndim - side->ndim;
    for (int i = 0; i < side->ndim; ++i) {
        int ds = side->dims[i];
        int dp = pipe->dims[off + i];
        if (ds != 1 && ds != dp) return false;
    }
    return true;
}

// NNR_LOG_WEBGPU_FUSION=1 enables per-chain / per-producer diagnostic logs
// that reveal exactly which ops reach the fused exec path and which don't.
// Unset by default; both passes treat the env var as a pure no-op when it
// isn't set, so production runs pay only a single getenv() per pass call.
inline bool fusion_log_enabled()
{
    const char* v = std::getenv("NNR_LOG_WEBGPU_FUSION");
    return v && *v && *v != '0';
}

} // namespace
} // namespace nnr

#endif // NNR_ENABLE_WEBGPU
