#include "graph_optimizer/graph_optimizer_internal.h"

namespace nnr {


void fuse_post_ops(context_t* ctx)
{
    if (!ctx->graph) return;
    auto& nodes = ctx->graph->nodes;
    const int n = static_cast<int>(nodes.size());

    auto count_users = [&](tensor_t* tensor, int skip_node) -> int {
        int users = 0;
        for (int j = 0; j < n; ++j) {
            if (j == skip_node) continue;
            if (nodes[j]->skip || nodes[j]->folded) continue;
            for (auto* t : nodes[j]->inputs) {
                if (t == tensor) users++;
            }
        }
        return users;
    };

    for (int i = 0; i + 1 < n; ++i) {
        operator_t* producer = nodes[i];
        if (producer->skip) continue;
        if (producer->post_fn) continue; // already fused
        std::string_view op_type = producer->op_type;
        if (op_type != "Conv" && op_type != "Gemm" && op_type != "MatMul"
            && op_type != "QLinearConv" && op_type != "Add"
            && op_type != "BatchNormalization") continue;
        if (producer->outputs.empty()) continue;
        // Trace output tensor through skipped/folded nodes (e.g., folded BN,
        // Constant nodes producing Clip min/max) and find the next active consumer.
        tensor_t* effective_out = producer->outputs[0];
        int j = i + 1;
        for (; j < n && (nodes[j]->skip || nodes[j]->folded); ++j) {
            auto* sk = nodes[j];
            if (!sk->inputs.empty() && !sk->outputs.empty()
                && sk->inputs[0] == effective_out)
                effective_out = sk->outputs[0];
        }
        if (j >= n) continue;
        operator_t* consumer = nodes[j];
        if (!consumer->fusable_apply) continue;
        if (consumer->inputs.empty()) continue;
        // SiLU-fused Sigmoid is admitted by `fuse_post_ops_silu` instead.
        // That pass runs at LAYOUT level (after declared_layout is committed)
        // so it can gate on producer/consumer layout agreement — fusing
        // across a layout boundary makes the runtime SKIP-alias bridge
        // BLOCKED data through an NCHW-declared tensor and yolov10s/n hang.
        if (consumer->is_fused_silu) continue;
        // Check producer output feeds into consumer (any input slot for binary ops).
        // After QDQ fusion, the consumer may read directly from producer's raw output
        // (not through the traced effective_out), so check both.
        tensor_t* raw_out = producer->outputs[0];
        bool input_match = false;
        tensor_t* matched_tensor = nullptr;
        for (auto* t : consumer->inputs) {
            if (t == effective_out || t == raw_out) {
                input_match = true;
                matched_tensor = t;
                break;
            }
        }
        if (!input_match) continue;
        if (count_users(matched_tensor, -1) != 1) continue;

        // Skip binary ops (Add, Sub, etc.) — leave them for scroll chains.
        // Conv → Add → Relu is handled strip-by-strip, keeping data L2-hot.
        // Only fuse unary ops (Relu, Clip, etc.) into the producer.
        if (consumer->inputs.size() == 2) continue;

        // QDQ residual interference guard: when the producer is an Add and
        // the consumer Relu's output flows into a QuantizeLinear, leave the
        // chain alone — fuse_qdq_compute folds DQ+DQ→Add→Relu→Q into a
        // single QLinearAdd, but only over multiple fixed-point rounds. If
        // we skip the Relu here, fuse_qdq_compute's count_consumers check
        // sees Add's output with 0 consumers in the next round and aborts,
        // leaving most of the residual chain unfused (ssd-12-qdq idiom).
        if (op_type == "Add" && !consumer->outputs.empty() && consumer->outputs[0]) {
            tensor_t* relu_out = consumer->outputs[0];
            bool feeds_quantize = false;
            for (int k = 0; k < n; ++k) {
                if (nodes[k]->skip || nodes[k]->folded) continue;
                if (nodes[k]->op_type != "QuantizeLinear") continue;
                if (!nodes[k]->inputs.empty() && nodes[k]->inputs[0] == relu_out) {
                    feeds_quantize = true;
                    break;
                }
            }
            if (feeds_quantize) continue;
        }

        producer->post_fn = consumer->fusable_apply;
        producer->fused_op = consumer;
        consumer->skip = true;
    }
}

// SiLU-only post-op fusion. Split out of `fuse_post_ops` because it must run
// AFTER LAYOUT — see kb/conv_silu_fusion_blocker.md.
//
// `fuse_silu` (PREPROCESS) rewrote `Sigmoid->outputs[0] = Mul->outputs[0]`,
// then `assign_blocked_layouts` (LAYOUT) committed declared_layout on every
// tensor. If the producer's output and the consumer's output disagree on
// declared_layout, admitting Conv→SiLU would make the runtime SKIP handler
// bridge BLOCKED data through an NCHW-declared tensor (or vice versa);
// downstream dispatch reads the wrong addressing and yolov10s/n hang.
void fuse_post_ops_silu(context_t* ctx)
{
    if (!ctx->graph) return;
    auto& nodes = ctx->graph->nodes;
    const int n = static_cast<int>(nodes.size());

    auto count_users = [&](tensor_t* tensor) -> int {
        int users = 0;
        for (int j = 0; j < n; ++j) {
            if (nodes[j]->skip || nodes[j]->folded) continue;
            for (auto* t : nodes[j]->inputs)
                if (t == tensor) users++;
        }
        return users;
    };

    for (int i = 0; i + 1 < n; ++i) {
        operator_t* producer = nodes[i];
        if (producer->skip || producer->folded) continue;
        if (producer->post_fn) continue;
        std::string_view op_type = producer->op_type;
        if (op_type != "Conv" && op_type != "Gemm" && op_type != "MatMul"
            && op_type != "QLinearConv" && op_type != "Add"
            && op_type != "BatchNormalization") continue;
        if (producer->outputs.empty()) continue;

        tensor_t* effective_out = producer->outputs[0];
        int j = i + 1;
        for (; j < n && (nodes[j]->skip || nodes[j]->folded); ++j) {
            auto* sk = nodes[j];
            if (!sk->inputs.empty() && !sk->outputs.empty()
                && sk->inputs[0] == effective_out)
                effective_out = sk->outputs[0];
        }
        if (j >= n) continue;
        operator_t* consumer = nodes[j];
        if (!consumer->is_fused_silu) continue;
        if (!consumer->fusable_apply) continue;
        if (consumer->inputs.empty() || consumer->outputs.empty()) continue;

        tensor_t* raw_out = producer->outputs[0];
        tensor_t* matched_tensor = nullptr;
        for (auto* t : consumer->inputs) {
            if (t == effective_out || t == raw_out) {
                matched_tensor = t;
                break;
            }
        }
        if (!matched_tensor) continue;
        if (count_users(matched_tensor) != 1) continue;

        // Layout reconciliation gate. Producer writes its result to `raw_out`
        // and the SKIP handler aliases the consumer's (post-fuse_silu) output
        // to that same buffer. If declared_layout disagrees, downstream
        // queries see contradictory layouts.
        if (raw_out->declared_layout != consumer->outputs[0]->declared_layout)
            continue;

        producer->post_fn = consumer->fusable_apply;
        producer->fused_op = consumer;
        consumer->skip = true;
    }
}

} // namespace nnr
