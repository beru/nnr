// WebGPU MatMul / Gemm / Conv / LayerNorm / Softmax + FusedElementwiseChain
// post-op fusion.
//
// Second-level fusion that runs after fuse_webgpu_elementwise. For each
// pattern:
//
//   MatMul             → FusedElementwiseChain   →  MatMulFusedChain
//   Gemm               → FusedElementwiseChain   →  GemmFusedChain
//   Conv               → FusedElementwiseChain   →  ConvFusedChain
//   LayerNormalization → FusedElementwiseChain   →  LayerNormFusedChain
//   Softmax            → FusedElementwiseChain   →  SoftmaxFusedChain
//
// where the chain's pipe input is the producer's output and that output has
// exactly one consumer (the chain) and isn't a graph output, absorb the
// chain's entire `stage_wgsl` list into a new fused operator that splices
// the chain stages into the producer kernel's output epilogue. The chain's
// side inputs become additional producer bindings.
//
// This eliminates one full-tensor write+read compared to running the
// producer and the chain as two dispatches — the chain's accumulator lives
// in a per-invocation `var v` inside the producer kernel rather than being
// materialized to an intermediate tensor.
//
// Correctness relies on the producer's output being written once per (row,
// col) and exclusively consumed by the chain — identical to the chain's
// Path U invariant for its pipe input.

#include "graph_optimizer/graph_optimizer_internal.h"

#ifdef NNR_ENABLE_WEBGPU
#include "graph_optimizer/fuse_webgpu_common.h"
#include "backend/webgpu/elementwise.h"
#include "pool.h"
#include "registry.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#endif

namespace nnr {

#ifndef NNR_ENABLE_WEBGPU
void fuse_webgpu_matmul_chain(context_t*) {}
#else

namespace {

// Walker used by fuse_webgpu_matmul_chain to synthesize an absorbable chain
// on the fly when the producer's direct consumer is a plain fusable op
// (Relu, Add(bias), Gelu, ...) that the standalone FEC pass left untouched.
// Differences from fuse_webgpu_elementwise's internal walker:
//   - Head may be a pipe-first binary (e.g. Add with broadcast bias), not
//     just unary. The standalone FEC kernel's seed is `v = X[i]` which
//     precludes binary heads, but an absorbed chain seeds `v` from the
//     producer's accumulator (matmul mma / conv dot / ln-mean-norm) — so
//     a binary head is representable as `v = producer.v op side`.
//   - Walks forward through any sequence of fusable unary / pipe-first
//     binary ops, accumulating broadcast-aware stage_wgsl + side tensors.
struct walked_chain_t {
    std::vector<std::string> stage_wgsl;
    std::vector<tensor_t*>   sides;
    bool                     needs_meta = false;
    std::vector<int>         chain_idx;   // nodes to fold, in traversal order
    bool                     ok = false;
};

walked_chain_t walk_absorbable_chain(
    const std::vector<operator_t*>&                         nodes,
    const std::unordered_map<tensor_t*, std::vector<int>>&  consumers,
    const std::unordered_set<tensor_t*>&                    is_graph_output,
    int                                                     start_idx,
    tensor_t*                                               initial_pipe)
{
    walked_chain_t out;

    struct raw_stage_t {
        const char* pattern     = nullptr;   // binary pattern (pipe-first), nullptr for unary
        const char* unary_expr  = nullptr;
        tensor_t*   side        = nullptr;
        bool        side_is_bc  = false;
    };
    std::vector<raw_stage_t> stages;

    operator_t* cur      = (start_idx >= 0 && start_idx < (int)nodes.size()) ? nodes[start_idx] : nullptr;
    tensor_t*   prev_out = initial_pipe;
    int         cur_idx  = start_idx;

    while (cur) {
        // Basic eligibility (mirrors fuse_webgpu_elementwise's link_is_candidate).
        if (cur->skip || cur->folded)                                 break;
        if (cur->resolved_backend != (uint8_t)backend_t::WEBGPU)      break;
        if (cur->is_fused_silu)                                       break;
        if (cur->outputs.size() != 1 || !cur->outputs[0])             break;

        const char* u_expr    = nullptr;
        const char* b_pat     = nullptr;
        tensor_t*   side      = nullptr;
        bool        side_bc   = false;

        if (cur->inputs.size() == 1 && cur->inputs[0] == prev_out) {
            u_expr = webgpu::fusable_unary_op_expr(cur->op_type);
        }
        if (!u_expr && cur->inputs.size() == 2 && cur->inputs[0] == prev_out) {
            if (const char* pat = webgpu::fusable_binary_op_pattern(cur->op_type)) {
                tensor_t* s = cur->inputs[1];
                if (s && side_broadcasts_into(s, cur->inputs[0])) {
                    b_pat   = pat;
                    side    = s;
                    side_bc = !same_shape(s, cur->inputs[0]);
                }
            }
        }

        if (!u_expr && !b_pat) break;   // not a chainable op

        stages.push_back({ b_pat, u_expr, side, side_bc });
        out.chain_idx.push_back(cur_idx);

        // Can we advance to cur's single live consumer?
        tensor_t* cur_out = cur->outputs[0];
        auto it = consumers.find(cur_out);
        if (it == consumers.end() || it->second.size() != 1)      break;
        if (is_graph_output.count(cur_out))                       break;
        int next_idx = it->second[0];
        if (next_idx <= cur_idx)                                  break;

        cur_idx  = next_idx;
        cur      = nodes[cur_idx];
        prev_out = cur_out;
    }

    if (stages.empty()) return out;

    // Decide Path U vs Path M. If any side in the chain needs broadcast, the
    // whole chain runs in meta form (consistent with fuse_webgpu_elementwise).
    for (auto& s : stages) {
        if (s.side && s.side_is_bc) { out.needs_meta = true; break; }
    }

    out.stage_wgsl.reserve(stages.size());
    for (auto& s : stages) {
        if (s.unary_expr) {
            out.stage_wgsl.emplace_back(s.unary_expr);
            continue;
        }
        std::string side_var = "S";
        side_var += std::to_string(out.sides.size());
        side_var += out.needs_meta
            ? ("[side_" + std::to_string(out.sides.size()) + "_flat]")
            : "[i]";
        out.stage_wgsl.emplace_back(substitute_side(s.pattern, side_var));
        out.sides.push_back(s.side);
    }

    out.ok = true;
    return out;
}

} // namespace

void fuse_webgpu_matmul_chain(context_t* ctx)
{
    if (!ctx || !ctx->graph) return;
    if (ctx->preferred_backend != static_cast<uint8_t>(backend_t::WEBGPU)) return;
    if (const char* v = std::getenv("NNR_DISABLE_WEBGPU_FUSION"); v && *v) return;

    auto& nodes = ctx->graph->nodes;
    const int n = static_cast<int>(nodes.size());
    if (n < 2) return;

    const bool log = fusion_log_enabled();

    // Consumer map, same rules as the elementwise pass: only live nodes count.
    std::unordered_map<tensor_t*, std::vector<int>> consumers;
    for (int i = 0; i < n; ++i) {
        auto* op = nodes[i];
        if (op->skip || op->folded) continue;
        for (auto* t : op->inputs)
            if (t) consumers[t].push_back(i);
    }

    std::unordered_set<tensor_t*> is_graph_output;
    for (auto& name : ctx->graph_outputs) {
        tensor_t* t = ctx->search_tensor(name);
        if (t) is_graph_output.insert(t);
    }

    int fused_count = 0;
    for (int i = 0; i < n; ++i) {
        auto* prod = nodes[i];
        if (!prod || prod->skip || prod->folded)                     continue;
        if (prod->resolved_backend != (uint8_t)backend_t::WEBGPU)    continue;
        const bool is_matmul  = (prod->op_type == "MatMul");
        const bool is_gemm    = (prod->op_type == "Gemm");
        const bool is_conv    = (prod->op_type == "Conv");
        const bool is_ln      = (prod->op_type == "LayerNormalization");
        const bool is_softmax = (prod->op_type == "Softmax");
        if (!is_matmul && !is_gemm && !is_conv && !is_ln && !is_softmax) continue;

        // Past this point `prod` is an eligible producer kind. Any further
        // rejection is "chain absorption didn't happen" — logged so the user
        // can see why this specific Conv/MatMul/… stayed standalone.
        auto reject = [&](const std::string& why) {
            if (!log) return;
            std::fprintf(stderr,
                "[NNR-FUSION] producer %.*s (node %d '%.*s') skipped — %s\n",
                (int)prod->op_type.size(), prod->op_type.data(), i,
                (int)prod->node_name.size(), prod->node_name.data(),
                why.c_str());
        };

        if (prod->outputs.size() != 1)                               { reject("producer outputs != 1"); continue; }

        // Shape constraints on producer inputs (matches the corresponding
        // fused op's reshape guards — avoid building a fused op we'd have to
        // immediately tear down).
        if (is_matmul) {
            if (prod->inputs.size() != 2)                            { reject("MatMul inputs != 2"); continue; }
            // matmul_fused_chain_t::reshape() now handles 2D @ 2D and
            // left-batched [..., M, K] @ [K, N] (mode b). Mode (c)/(d) —
            // same-rank or broadcast-batched — is not yet supported; pre-filter
            // those here so we don't construct a doomed fused op.
            if (!prod->inputs[0] || !prod->inputs[1]) {
                reject("MatMul inputs null");
                continue;
            }
            if (prod->inputs[1]->ndim != 2) {
                reject("MatMul RHS isn't 2D (batched-RHS matmul not yet supported by MatMulFusedChain)");
                continue;
            }
        } else if (is_gemm) {
            // Gemm has 2 or 3 inputs; anything else is malformed.
            if (prod->inputs.size() < 2 || prod->inputs.size() > 3)  { reject("Gemm inputs not in {2,3}"); continue; }
        } else if (is_conv) {
            // Conv: X + W + optional bias. 2 or 3 inputs.
            if (prod->inputs.size() < 2 || prod->inputs.size() > 3)  { reject("Conv inputs not in {2,3}"); continue; }
        } else if (is_ln) {
            // LayerNormalization: X + Scale + optional Bias. 2 or 3 inputs.
            if (prod->inputs.size() < 2 || prod->inputs.size() > 3)  { reject("LayerNorm inputs not in {2,3}"); continue; }
        } else if (is_softmax) {
            if (prod->inputs.size() != 1)                            { reject("Softmax inputs != 1"); continue; }
        }

        tensor_t* prod_out = prod->outputs[0];
        if (!prod_out)                                               { reject("null output"); continue; }
        if (is_graph_output.count(prod_out))                         { reject("output is a graph output"); continue; }

        auto it = consumers.find(prod_out);
        if (it == consumers.end() || it->second.empty())             { reject("no live consumer"); continue; }
        if (it->second.size() != 1) {
            reject("multi-fanout output (" + std::to_string(it->second.size()) + " consumers)");
            continue;
        }
        int next_idx = it->second[0];
        if (next_idx <= i)                                           { reject("consumer index not forward"); continue; }

        auto* chain = nodes[next_idx];
        if (!chain || chain->skip || chain->folded)                  { reject("consumer is skipped/folded"); continue; }

        // Absorption source. Two shapes are accepted:
        //   A. An actual FusedElementwiseChain produced by fuse_webgpu_elementwise
        //      when ≥2 consecutive elementwise ops chained together.
        //   B. A standalone fusable unary OR pipe-first binary (with
        //      broadcast-compatible side), extended forward through further
        //      fusable unary/binary ops by walk_absorbable_chain(). Handles
        //      cnn_bench's Conv→Relu (size-1 unary chain) and transformer
        //      MatMul→Add(bias)→Gelu (size-2: binary head + unary) the same
        //      way. Synthesizing the chain in place avoids materializing a
        //      standalone FEC (which would reject size<2 anyway).
        std::vector<std::string> absorbed_stage_wgsl;
        std::vector<tensor_t*>   absorbed_sides;
        int                      absorbed_n_sides    = 0;
        bool                     absorbed_needs_meta = false;
        tensor_t*                absorbed_output     = nullptr;
        std::vector<operator_t*> absorbed_ops_to_fold;
        bool                     absorbed_was_unary_chain = false;

        if (chain->op_type == "FusedElementwiseChain") {
            if (chain->resolved_backend != (uint8_t)backend_t::WEBGPU) { reject("chain resolved_backend != WEBGPU"); continue; }
            if (chain->inputs.empty() || chain->outputs.size() != 1)  { reject("chain has bad input/output arity"); continue; }
            if (chain->inputs[0] != prod_out)                         { reject("chain's pipe input isn't the producer's output"); continue; }
            auto* fecp = static_cast<webgpu::fused_elementwise_chain_t*>(chain);
            absorbed_stage_wgsl = fecp->stage_wgsl;  // copy; moved into producer below
            absorbed_n_sides    = fecp->n_sides;
            absorbed_needs_meta = fecp->needs_meta;
            absorbed_sides.reserve((size_t)absorbed_n_sides);
            for (int k = 0; k < absorbed_n_sides; ++k)
                absorbed_sides.push_back(chain->inputs[1 + k]);
            absorbed_output = chain->outputs[0];
            absorbed_ops_to_fold.push_back(chain);
        } else {
            // Walker path — synthesize an absorbable chain from a standalone
            // fusable head. Returns ok=false if `chain` isn't a valid head.
            auto w = walk_absorbable_chain(nodes, consumers, is_graph_output, next_idx, prod_out);
            if (!w.ok) {
                std::string why = "consumer is ";
                why.append(chain->op_type.data(), chain->op_type.size());
                why += ", not FusedElementwiseChain or fusable unary/binary";
                reject(why);
                continue;
            }
            absorbed_stage_wgsl  = std::move(w.stage_wgsl);
            absorbed_sides       = std::move(w.sides);
            absorbed_n_sides     = (int)absorbed_sides.size();
            absorbed_needs_meta  = w.needs_meta;
            // The fused op writes where the LAST op in the walked chain would
            // have written — that's the tensor any further downstream reader
            // is already pointed at.
            absorbed_output = nodes[w.chain_idx.back()]->outputs[0];
            for (int idx : w.chain_idx) absorbed_ops_to_fold.push_back(nodes[idx]);
            absorbed_was_unary_chain = true;
        }

        const size_t n_stages = absorbed_stage_wgsl.size();

        operator_t* fused = nullptr;

        if (is_ln) {
            auto* lf = pool_new<webgpu::layer_norm_fused_chain_t>(ctx->attr_pool);
            lf->ctx              = ctx;
            lf->opset            = prod->opset;
            lf->op_type          = "LayerNormFusedChain";
            lf->node_name        = prod->node_name;
            lf->resolved_backend = static_cast<uint8_t>(backend_t::WEBGPU);
            lf->stage_wgsl       = absorbed_stage_wgsl;
            lf->n_sides          = absorbed_n_sides;
            lf->needs_meta       = absorbed_needs_meta;
            lf->has_bias         = (prod->inputs.size() == 3 && prod->inputs[2]);
            lf->attrs            = prod->attrs;  // axis, epsilon

            const int n_inputs = 2 + (lf->has_bias ? 1 : 0) + absorbed_n_sides;
            tensor_t** ins = ctx->attr_pool.alloc_arr<tensor_t*>((size_t)n_inputs);
            ins[0] = prod->inputs[0];
            ins[1] = prod->inputs[1];
            int side_off = 2;
            if (lf->has_bias) { ins[2] = prod->inputs[2]; side_off = 3; }
            for (int kk = 0; kk < absorbed_n_sides; ++kk)
                ins[side_off + kk] = absorbed_sides[kk];
            lf->inputs = std::span<tensor_t*>(ins, (size_t)n_inputs);

            tensor_t** outs = ctx->attr_pool.alloc_arr<tensor_t*>(1);
            outs[0] = absorbed_output;
            lf->outputs = std::span<tensor_t*>(outs, 1);

            if (!lf->init())    { reject("LayerNormFusedChain init() failed");    lf->~layer_norm_fused_chain_t(); continue; }
            if (!lf->reshape()) { reject("LayerNormFusedChain reshape() failed"); lf->~layer_norm_fused_chain_t(); continue; }
            fused = lf;
        } else if (is_softmax) {
            auto* sf = pool_new<webgpu::softmax_fused_chain_t>(ctx->attr_pool);
            sf->ctx              = ctx;
            sf->opset            = prod->opset;
            sf->op_type          = "SoftmaxFusedChain";
            sf->node_name        = prod->node_name;
            sf->resolved_backend = static_cast<uint8_t>(backend_t::WEBGPU);
            sf->stage_wgsl       = absorbed_stage_wgsl;
            sf->n_sides          = absorbed_n_sides;
            sf->needs_meta       = absorbed_needs_meta;
            sf->attrs            = prod->attrs;  // axis

            const int n_inputs = 1 + absorbed_n_sides;
            tensor_t** ins = ctx->attr_pool.alloc_arr<tensor_t*>((size_t)n_inputs);
            ins[0] = prod->inputs[0];
            for (int kk = 0; kk < absorbed_n_sides; ++kk)
                ins[1 + kk] = absorbed_sides[kk];
            sf->inputs = std::span<tensor_t*>(ins, (size_t)n_inputs);

            tensor_t** outs = ctx->attr_pool.alloc_arr<tensor_t*>(1);
            outs[0] = absorbed_output;
            sf->outputs = std::span<tensor_t*>(outs, 1);

            if (!sf->init())    { reject("SoftmaxFusedChain init() failed");    sf->~softmax_fused_chain_t(); continue; }
            if (!sf->reshape()) { reject("SoftmaxFusedChain reshape() failed"); sf->~softmax_fused_chain_t(); continue; }
            fused = sf;
        } else if (is_conv) {
            // Build the Conv variant. Inputs = [X, W, (B?), side_0, ...].
            auto* cf = pool_new<webgpu::conv_fused_chain_t>(ctx->attr_pool);
            cf->ctx              = ctx;
            cf->opset            = prod->opset;
            cf->op_type          = "ConvFusedChain";
            cf->node_name        = prod->node_name;
            cf->resolved_backend = static_cast<uint8_t>(backend_t::WEBGPU);
            cf->stage_wgsl       = absorbed_stage_wgsl;
            cf->n_sides          = absorbed_n_sides;
            cf->needs_meta       = absorbed_needs_meta;
            cf->has_bias         = (prod->inputs.size() == 3 && prod->inputs[2]);
            cf->attrs            = prod->attrs;   // strides/pads/dilations/group/auto_pad

            const int n_inputs = 2 + (cf->has_bias ? 1 : 0) + absorbed_n_sides;
            tensor_t** ins = ctx->attr_pool.alloc_arr<tensor_t*>((size_t)n_inputs);
            ins[0] = prod->inputs[0];
            ins[1] = prod->inputs[1];
            int side_off = 2;
            if (cf->has_bias) { ins[2] = prod->inputs[2]; side_off = 3; }
            for (int kk = 0; kk < absorbed_n_sides; ++kk)
                ins[side_off + kk] = absorbed_sides[kk];
            cf->inputs = std::span<tensor_t*>(ins, (size_t)n_inputs);

            tensor_t** outs = ctx->attr_pool.alloc_arr<tensor_t*>(1);
            outs[0] = absorbed_output;
            cf->outputs = std::span<tensor_t*>(outs, 1);

            if (!cf->init())    { reject("ConvFusedChain init() failed");    cf->~conv_fused_chain_t(); continue; }
            if (!cf->reshape()) { reject("ConvFusedChain reshape() failed"); cf->~conv_fused_chain_t(); continue; }
            fused = cf;
        } else if (is_matmul) {
            // Build the MatMul variant. Inputs = [A, B, side_0, ...].
            auto* mf = pool_new<webgpu::matmul_fused_chain_t>(ctx->attr_pool);
            mf->ctx              = ctx;
            mf->opset            = prod->opset;
            mf->op_type          = "MatMulFusedChain";
            mf->node_name        = prod->node_name;
            mf->resolved_backend = static_cast<uint8_t>(backend_t::WEBGPU);
            mf->stage_wgsl       = absorbed_stage_wgsl;
            mf->n_sides          = absorbed_n_sides;
            mf->needs_meta       = absorbed_needs_meta;

            const int n_inputs = 2 + absorbed_n_sides;
            tensor_t** ins = ctx->attr_pool.alloc_arr<tensor_t*>((size_t)n_inputs);
            ins[0] = prod->inputs[0];
            ins[1] = prod->inputs[1];
            for (int kk = 0; kk < absorbed_n_sides; ++kk)
                ins[2 + kk] = absorbed_sides[kk];
            mf->inputs = std::span<tensor_t*>(ins, (size_t)n_inputs);

            tensor_t** outs = ctx->attr_pool.alloc_arr<tensor_t*>(1);
            outs[0] = absorbed_output;
            mf->outputs = std::span<tensor_t*>(outs, 1);

            if (!mf->init())    { reject("MatMulFusedChain init() failed");    mf->~matmul_fused_chain_t(); continue; }
            if (!mf->reshape()) { reject("MatMulFusedChain reshape() failed"); mf->~matmul_fused_chain_t(); continue; }
            fused = mf;
        } else {
            // Build the Gemm variant. Inputs = [A, B, (C?), side_0, ...].
            auto* gf = pool_new<webgpu::gemm_fused_chain_t>(ctx->attr_pool);
            gf->ctx              = ctx;
            gf->opset            = prod->opset;
            gf->op_type          = "GemmFusedChain";
            gf->node_name        = prod->node_name;
            gf->resolved_backend = static_cast<uint8_t>(backend_t::WEBGPU);
            gf->stage_wgsl       = absorbed_stage_wgsl;
            gf->n_sides          = absorbed_n_sides;
            gf->needs_meta       = absorbed_needs_meta;
            gf->has_bias         = (prod->inputs.size() == 3 && prod->inputs[2]);
            gf->attrs            = prod->attrs;   // alpha/beta/transA/transB flow through

            const int n_inputs = 2 + (gf->has_bias ? 1 : 0) + absorbed_n_sides;
            tensor_t** ins = ctx->attr_pool.alloc_arr<tensor_t*>((size_t)n_inputs);
            ins[0] = prod->inputs[0];
            ins[1] = prod->inputs[1];
            int side_off = 2;
            if (gf->has_bias) { ins[2] = prod->inputs[2]; side_off = 3; }
            for (int kk = 0; kk < absorbed_n_sides; ++kk)
                ins[side_off + kk] = absorbed_sides[kk];
            gf->inputs = std::span<tensor_t*>(ins, (size_t)n_inputs);

            tensor_t** outs = ctx->attr_pool.alloc_arr<tensor_t*>(1);
            outs[0] = absorbed_output;
            gf->outputs = std::span<tensor_t*>(outs, 1);

            if (!gf->init())    { reject("GemmFusedChain init() failed");    gf->~gemm_fused_chain_t(); continue; }
            if (!gf->reshape()) { reject("GemmFusedChain reshape() failed"); gf->~gemm_fused_chain_t(); continue; }
            fused = gf;
        }

        // Replace the producer in-place, fold every absorbed op.
        nodes[i] = fused;
        for (auto* folded_op : absorbed_ops_to_fold) folded_op->folded = true;
        ++fused_count;

        if (log) {
            if (absorbed_was_unary_chain) {
                // Describe the walked chain by op_type (Add -> Gelu, Relu, ...)
                std::string desc;
                for (size_t k = 0; k < absorbed_ops_to_fold.size(); ++k) {
                    if (k) desc += " -> ";
                    auto& t = absorbed_ops_to_fold[k]->op_type;
                    desc.append(t.data(), t.size());
                }
                std::fprintf(stderr,
                    "[NNR-FUSION] absorbed: %.*s (node %d) + [%s] -> %.*s\n",
                    (int)prod->op_type.size(), prod->op_type.data(), i,
                    desc.c_str(),
                    (int)fused->op_type.size(), fused->op_type.data());
            } else {
                std::fprintf(stderr,
                    "[NNR-FUSION] absorbed: %.*s (node %d) + FEC[%zu stage(s)] -> %.*s\n",
                    (int)prod->op_type.size(), prod->op_type.data(), i,
                    n_stages,
                    (int)fused->op_type.size(), fused->op_type.data());
            }
        }
    }

    if (fused_count > 0)
        std::fprintf(stderr, "[NNR] Fused %d WebGPU producer+chain op(s)\n", fused_count);

    // End-of-fusion tally: which op_types still run as standalone dispatches?
    // This pass runs after fuse_webgpu_elementwise, so what's live here is
    // what actually dispatches at run time (ignoring downstream CPU-only
    // layout rewrites, which are gated off on the WebGPU path anyway).
    if (log) {
        std::map<std::string_view, int> tally;
        int live = 0;
        for (auto* op : nodes) {
            if (!op || op->skip || op->folded) continue;
            tally[op->op_type]++;
            ++live;
        }
        std::string summary;
        for (auto& kv : tally) {
            if (!summary.empty()) summary += " ";
            summary.append(kv.first.data(), kv.first.size());
            summary += "=";
            summary += std::to_string(kv.second);
        }
        std::fprintf(stderr,
            "[NNR-FUSION] live ops after fusion: %d total — %s\n",
            live, summary.c_str());
    }
}

#endif // NNR_ENABLE_WEBGPU

} // namespace nnr
