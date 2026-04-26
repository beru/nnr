#include "graph_optimizer/graph_optimizer_internal.h"

#include <unordered_map>

namespace nnr {

// T3 M1 step 4: explicit-Reorder cleanup pass.
//
// No-op when NNR_EXPLICIT_REORDERS is OFF (M1 default). When ON, runs after
// insert_reorders and iterates a two-rule fixed point until the graph stops
// changing:
//
//   1. Round-trip cancellation. A Reorder(B→A) feeding directly off a
//      Reorder(A→B) is a no-op pair. Consumers of the inner Reorder's output
//      are rewired to the original A producer's tensor, both Reorders are
//      marked `skip`. Single-step only — multi-hop round-trips collapse via
//      iteration (cancel the inner pair, then the outer pair becomes adjacent).
//
//   2. Sibling merge. If a producer tensor X feeds N>1 active Reorders that
//      share the same `to_layout`, every consumer of every duplicate gets
//      redirected to the first Reorder's output, and the duplicates are
//      marked `skip`. This collapses the N reorders insert_reorders synthesizes
//      when X feeds multiple downstream Convs all needing the same layout.
//
// Skipped Reorders stay in the node list. The runtime SKIP handler in nnr.cpp
// aliases their `outputs[0]->data` to `inputs[0]->data`, but with consumers
// redirected past them nobody reads that aliased output, so the alias is
// harmless. The memory planner already excludes skip-node outputs from the
// lifetime allocator (memory_planner.cpp:60-62).

#ifdef NNR_EXPLICIT_REORDERS

namespace {

bool is_active_reorder(operator_t* n)
{
    return n && !n->skip && !n->folded
        && n->op_type == "Reorder"
        && !n->inputs.empty()  && n->inputs[0]
        && !n->outputs.empty() && n->outputs[0];
}

memory_layout_t reorder_to(operator_t* n)
{
    return n->outputs[0]->declared_layout;
}

memory_layout_t reorder_from(operator_t* n)
{
    return n->inputs[0]->declared_layout;
}

// Replace every input pointer equal to `old_t` with `new_t` across all
// active nodes. The input span aliases pool-backed storage that may be
// shared, so we always allocate a fresh array on rewrite (mirroring
// insert_reorders.cpp's discipline).
void redirect_consumers(context_t* ctx, tensor_t* old_t, tensor_t* new_t)
{
    if (old_t == new_t || !old_t) return;
    for (auto* n : ctx->graph->nodes) {
        if (!n || n->skip || n->folded) continue;
        bool any = false;
        for (auto* in : n->inputs) {
            if (in == old_t) { any = true; break; }
        }
        if (!any) continue;
        const size_t nin = n->inputs.size();
        tensor_t** rep = ctx->attr_pool.alloc_arr<tensor_t*>(nin);
        for (size_t i = 0; i < nin; ++i)
            rep[i] = (n->inputs[i] == old_t) ? new_t : n->inputs[i];
        n->inputs = {rep, nin};
    }
}

// True when any active node other than `skip_op` reads `t` as an input.
bool tensor_has_consumer(context_t* ctx, operator_t* skip_op, tensor_t* t)
{
    if (!t) return false;
    for (auto* n : ctx->graph->nodes) {
        if (!n || n == skip_op || n->skip || n->folded) continue;
        for (auto* in : n->inputs) {
            if (in == t) return true;
        }
    }
    return false;
}

bool cancel_round_trips(context_t* ctx)
{
    auto& nodes = ctx->graph->nodes;
    bool changed = false;

    std::unordered_map<tensor_t*, operator_t*> producer;
    producer.reserve(nodes.size() * 2 + 4);
    for (auto* n : nodes) {
        if (!n || n->skip || n->folded) continue;
        for (auto* o : n->outputs)
            if (o) producer.emplace(o, n);
    }

    for (auto* n : nodes) {
        if (!is_active_reorder(n)) continue;
        auto it = producer.find(n->inputs[0]);
        if (it == producer.end()) continue;
        operator_t* prev = it->second;
        if (!is_active_reorder(prev)) continue;
        if (reorder_from(prev) != reorder_to(n))   continue;
        if (reorder_to(prev)   != reorder_from(n)) continue;

        // A→B (prev) followed by B→A (n) — round-trip. Redirect consumers
        // of n's output to prev's input (the original A producer's tensor),
        // then mark n skip; mark prev skip too if its intermediate output
        // is no longer consumed by anyone else.
        tensor_t* original = prev->inputs[0];
        redirect_consumers(ctx, n->outputs[0], original);
        n->skip = true;
        changed = true;

        if (!tensor_has_consumer(ctx, n, prev->outputs[0]))
            prev->skip = true;
    }

    return changed;
}

bool merge_siblings(context_t* ctx)
{
    auto& nodes = ctx->graph->nodes;
    bool changed = false;

    // Group active Reorders by source-tensor pointer. All siblings of a
    // group share `from_layout` automatically (they read the same producer);
    // we pair them by `to_layout`.
    // DenseNet transition tensors fan out to many consumers; allow generous
    // capacity to avoid the small_vector push_back assert overflowing.
    std::unordered_map<tensor_t*, small_vector<operator_t*, 64>> by_src;
    by_src.reserve(nodes.size());
    for (auto* n : nodes) {
        if (!is_active_reorder(n)) continue;
        by_src[n->inputs[0]].push_back(n);
    }

    for (auto& [src, group] : by_src) {
        if (group.size() < 2) continue;
        for (size_t i = 0; i < group.size(); ++i) {
            operator_t* keeper = group[i];
            if (keeper->skip) continue;
            for (size_t j = i + 1; j < group.size(); ++j) {
                operator_t* dup = group[j];
                if (dup->skip) continue;
                if (reorder_to(keeper) != reorder_to(dup)) continue;
                redirect_consumers(ctx, dup->outputs[0], keeper->outputs[0]);
                dup->skip = true;
                changed = true;
            }
        }
    }

    return changed;
}

} // namespace

#endif // NNR_EXPLICIT_REORDERS

void cancel_reorders(context_t* ctx)
{
    if (!ctx || !ctx->graph) return;
#ifndef NNR_EXPLICIT_REORDERS
    return;
#else
    constexpr int MAX_ROUNDS = 16;
    for (int r = 0; r < MAX_ROUNDS; ++r) {
        bool changed = false;
        if (cancel_round_trips(ctx)) changed = true;
        if (merge_siblings(ctx))     changed = true;
        if (!changed) break;
    }
#endif
}

} // namespace nnr
