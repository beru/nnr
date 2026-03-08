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
            && op_type != "QLinearConv") continue;
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
        // Don't fuse SiLU-fused Sigmoid — it must run as SiLU, not plain sigmoid
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

        producer->post_fn = consumer->fusable_apply;
        producer->fused_op = consumer;
        consumer->skip = true;
    }
}

} // namespace nnr
