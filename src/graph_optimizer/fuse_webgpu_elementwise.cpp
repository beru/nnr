// WebGPU elementwise chain fusion.
//
// Collapses N ≥ 2 consecutive same-shape f32 elementwise ops (unary +
// pipe-first binary) on the WebGPU backend into a single
// fused_elementwise_chain_t dispatch. Per-stage WGSL expressions are
// composed into one shader (see src/backend/webgpu/elementwise.{h,cpp}),
// eliminating all intermediate buffer round-trips within the chain.
//
// Runs at FUSION level, gated on ctx->preferred_backend == WEBGPU. FUSION
// runs from context_t::prepare()'s optimize() call, so it fires in the
// common prepare→run path (PREPROCESS would only run in the run-only
// first_run path). The fused op's reshape() requires shapes, so running
// post fold_run (which is what FUSION level means) is the right choice.
//
// A chain link is valid if:
//   - op->resolved_backend == WEBGPU
//   - op->is_fused_silu is false
//   - op is a unary pipe op (fusable_unary_op_expr returns non-null) OR
//     a pipe-first binary op where inputs[0] is the previous stage's
//     output and inputs[1] has the same shape as the pipe
//   - the intermediate output has exactly one consumer, and is not a
//     graph output

#include "graph_optimizer/graph_optimizer_internal.h"

#ifdef NNR_ENABLE_WEBGPU
#include "graph_optimizer/fuse_webgpu_common.h"
#include "backend/webgpu/elementwise.h"
#include "pool.h"
#include "registry.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#endif

namespace nnr {

#ifndef NNR_ENABLE_WEBGPU
void fuse_webgpu_elementwise(context_t*) {}
#else

void fuse_webgpu_elementwise(context_t* ctx)
{
    if (!ctx || !ctx->graph) return;
    if (ctx->preferred_backend != static_cast<uint8_t>(backend_t::WEBGPU)) return;
    if (const char* v = std::getenv("NNR_DISABLE_WEBGPU_FUSION"); v && *v) return;

    auto& nodes = ctx->graph->nodes;
    const int n = static_cast<int>(nodes.size());
    if (n < 2) return;

    // Consumer map over live nodes. Skip/folded nodes don't count as consumers
    // since their exec is suppressed.
    std::unordered_map<tensor_t*, std::vector<int>> consumers;
    for (int i = 0; i < n; ++i) {
        auto* op = nodes[i];
        if (op->skip || op->folded) continue;
        for (auto* t : op->inputs)
            if (t) consumers[t].push_back(i);
    }

    // Graph outputs — we can't fuse past a tensor the caller wants to read.
    std::unordered_set<tensor_t*> is_graph_output;
    for (auto& name : ctx->graph_outputs) {
        tensor_t* t = ctx->search_tensor(name);
        if (t) is_graph_output.insert(t);
    }

    // Basic fusibility prerequisites (backend, live, is_fused_silu, span sizes).
    auto link_is_candidate = [&](operator_t* op) -> bool {
        if (!op || op->skip || op->folded)                        return false;
        if (op->resolved_backend != (uint8_t)backend_t::WEBGPU)   return false;
        if (op->is_fused_silu)                                    return false;
        if (op->outputs.size() != 1 || !op->outputs[0])           return false;
        return true;
    };

    // Classify op as a unary pipe link. Returns op_expr or nullptr.
    auto as_unary_link = [&](operator_t* op) -> const char* {
        if (!link_is_candidate(op))                       return nullptr;
        if (op->inputs.size() != 1 || !op->inputs[0])     return nullptr;
        return webgpu::fusable_unary_op_expr(op->op_type);
    };

    // Classify op as a pipe-first binary link. Returns the WGSL pattern
    // if op is a fusable binary whose inputs[0] is the previous pipe
    // output and whose inputs[1] is broadcastable INTO the pipe's shape.
    // Returns nullptr otherwise.
    auto as_binary_pipe_first_link = [&](operator_t* op,
                                         tensor_t* prev_pipe_output)
                                     -> const char* {
        if (!link_is_candidate(op))                             return nullptr;
        if (op->inputs.size() != 2)                             return nullptr;
        if (op->inputs[0] != prev_pipe_output)                  return nullptr;
        const char* pat = webgpu::fusable_binary_op_pattern(op->op_type);
        if (!pat)                                               return nullptr;
        if (!side_broadcasts_into(op->inputs[1], op->inputs[0])) return nullptr;
        return pat;
    };

    // A chain entry collected during walking. We defer `$s` substitution
    // to the end of the walk because the substitution form depends on
    // whether any side in the final chain needs broadcast (Path U vs M).
    struct stage_t {
        const char*        pattern;   // nullptr for unary; points into unary_exprs.inc otherwise binary_exprs.inc
        const char*        unary_expr;  // set for unary stages (stored pre-substituted)
        tensor_t*          side;      // nullptr for unary
        bool               side_is_broadcast;  // true iff side.shape != pipe.shape
    };

    std::vector<char> absorbed(n, 0);

    const bool log = fusion_log_enabled();

    int fused_count = 0;
    for (int i = 0; i < n; ++i) {
        if (absorbed[i]) continue;

        // Head must be a unary link. A binary head would have its own
        // pipe from outside the chain; we'd have to represent "v =
        // binary(A, B)" as the seed instead of "v = X[i]". Skipping
        // binary heads loses negligible coverage since any chain starting
        // with a binary is still caught one iteration later at i+1.
        const char* head_expr = as_unary_link(nodes[i]);
        if (!head_expr) continue;

        std::vector<int>     chain = { i };
        std::vector<stage_t> stages;
        stages.push_back({ nullptr, head_expr, nullptr, false });
        int cur = i;

        while (true) {
            tensor_t* out = nodes[cur]->outputs[0];
            auto it = consumers.find(out);
            if (it == consumers.end() || it->second.size() != 1) break;
            if (is_graph_output.count(out))                       break;
            int next = it->second[0];
            if (next <= cur)                                      break;

            if (const char* u = as_unary_link(nodes[next])) {
                if (nodes[next]->inputs[0] != out) break;
                chain.push_back(next);
                stages.push_back({ nullptr, u, nullptr, false });
                cur = next;
                continue;
            }
            if (const char* b = as_binary_pipe_first_link(nodes[next], out)) {
                tensor_t* side = nodes[next]->inputs[1];
                bool bc = !same_shape(side, nodes[next]->inputs[0]);
                chain.push_back(next);
                stages.push_back({ b, nullptr, side, bc });
                cur = next;
                continue;
            }
            break;
        }

        if (chain.size() < 2) {
            if (log) {
                // Analyze why this eligible unary head couldn't extend. The
                // walk broke on the very first step, so nodes[i]'s output is
                // the right thing to inspect.
                tensor_t* out = nodes[i]->outputs[0];
                auto it = consumers.find(out);
                std::string reason;
                if (it == consumers.end() || it->second.empty()) {
                    reason = "no live consumer";
                } else if (it->second.size() > 1) {
                    reason = "multi-fanout (" + std::to_string(it->second.size()) + " consumers)";
                } else if (is_graph_output.count(out)) {
                    reason = "output is a graph output";
                } else {
                    operator_t* next_op = nodes[it->second[0]];
                    std::string cnext(next_op->op_type);
                    reason = "consumer " + cnext + " is not a fusable unary/pipe-binary";
                }
                std::fprintf(stderr,
                    "[NNR-FUSION] elementwise singleton: %.*s (node %d '%.*s') — %s\n",
                    (int)nodes[i]->op_type.size(), nodes[i]->op_type.data(),
                    i,
                    (int)nodes[i]->node_name.size(), nodes[i]->node_name.data(),
                    reason.c_str());
            }
            continue;
        }

        // Decide Path U vs Path M. Path M is required if any side in the
        // chain needs broadcast; otherwise the simpler/faster Path U
        // suffices.
        bool needs_meta = false;
        for (auto& s : stages)
            if (s.side && s.side_is_broadcast) { needs_meta = true; break; }

        // Build stage_wgsl with the placeholder substitution matched to
        // the chosen path.
        std::vector<std::string> stage_wgsl;
        std::vector<tensor_t*>   sides;
        stage_wgsl.reserve(stages.size());
        for (auto& s : stages) {
            if (!s.side) {
                stage_wgsl.emplace_back(s.unary_expr);
                continue;
            }
            std::string side_var = "S";
            side_var += std::to_string(sides.size());
            side_var += needs_meta
                ? ("[side_" + std::to_string(sides.size()) + "_flat]")
                : "[i]";
            stage_wgsl.emplace_back(substitute_side(s.pattern, side_var));
            sides.push_back(s.side);
        }

        auto* fused = pool_new<webgpu::fused_elementwise_chain_t>(ctx->attr_pool);
        fused->ctx              = ctx;
        fused->opset            = nodes[i]->opset;
        fused->op_type          = "FusedElementwiseChain";
        fused->node_name        = nodes[i]->node_name;
        fused->resolved_backend = static_cast<uint8_t>(backend_t::WEBGPU);
        fused->stage_wgsl       = std::move(stage_wgsl);
        fused->n_sides          = static_cast<int>(sides.size());
        fused->needs_meta       = needs_meta;

        const int n_inputs = 1 + (int)sides.size();
        tensor_t** ins = ctx->attr_pool.alloc_arr<tensor_t*>((size_t)n_inputs);
        ins[0] = nodes[i]->inputs[0];
        for (size_t k = 0; k < sides.size(); ++k) ins[1 + k] = sides[k];
        fused->inputs = std::span<tensor_t*>(ins, (size_t)n_inputs);

        tensor_t** outs = ctx->attr_pool.alloc_arr<tensor_t*>(1);
        outs[0] = nodes[chain.back()]->outputs[0];
        fused->outputs = std::span<tensor_t*>(outs, 1);

        if (!fused->init()) {
            // Shader compilation failed — leave the originals alone.
            fused->~fused_elementwise_chain_t();
            continue;
        }
        // FUSION-level pass runs after fold_run reshaped the originals, so
        // the fused op's input/output tensors have shapes. Reshape now to
        // allocate GPU buffers; run_graph won't re-reshape when shapes_dirty
        // is already false (the prepare→run path).
        if (!fused->reshape()) {
            fused->~fused_elementwise_chain_t();
            continue;
        }

        // Replace the chain head, fold the rest. Match fuse_silu /
        // fuse_layer_norm — folded (not skip) because skip aliases
        // output=input and would clobber the fused op's result.
        nodes[i] = fused;
        for (size_t k = 1; k < chain.size(); ++k) {
            nodes[chain[k]]->folded = true;
            absorbed[chain[k]] = 1;
        }
        ++fused_count;

        if (log) {
            std::string desc;
            for (size_t k = 0; k < chain.size(); ++k) {
                if (k) desc += " -> ";
                auto& t = nodes[chain[k]]->op_type;
                desc.append(t.data(), t.size());
            }
            std::fprintf(stderr,
                "[NNR-FUSION] elementwise chain: %s (nodes %d..%d, %zu stages)\n",
                desc.c_str(), chain.front(), chain.back(), chain.size());
        }
    }

    if (fused_count > 0)
        std::fprintf(stderr, "[NNR] Fused %d WebGPU elementwise chain(s)\n", fused_count);
}

#endif // NNR_ENABLE_WEBGPU

} // namespace nnr
