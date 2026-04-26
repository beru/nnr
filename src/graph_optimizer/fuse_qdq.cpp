#include "graph_optimizer/graph_optimizer_internal.h"
#include "aligned_alloc.h"

namespace nnr {

//
// Also handles:
//  - Standalone DequantizeLinear on initializer weights (fold into tensor)
//  - Consecutive Q→DQ identity chains (skip both)

void fuse_qdq(context_t* ctx)
{
    if (!ctx->graph) return;
    auto& nodes = ctx->graph->nodes;
    const int n = static_cast<int>(nodes.size());
    int fused_count = 0;

    // Build producer map: tensor_t* → node index
    std::unordered_map<tensor_t*, int> producer;
    for (int i = 0; i < n; i++) {
        if (nodes[i]->skip || nodes[i]->folded) continue;
        for (auto* t : nodes[i]->outputs)
            producer[t] = i;
    }

    // Count consumers for each tensor
    auto count_consumers = [&](tensor_t* tensor, int skip_idx) -> int {
        int count = 0;
        for (int j = 0; j < n; j++) {
            if (j == skip_idx || nodes[j]->skip || nodes[j]->folded) continue;
            for (auto* t : nodes[j]->inputs)
                if (t == tensor) count++;
        }
        return count;
    };

    // Helper: extract scalar scale and zero-point from DequantizeLinear/QuantizeLinear inputs
    auto get_scale_zp = [](operator_t* dq_or_q, float& scale, int32_t& zp) -> bool {
        if (dq_or_q->inputs.size() < 2) return false;
        tensor_t* scale_t = dq_or_q->inputs[1];
        if (!scale_t || scale_t->ndata != 1) return false;  // per-tensor only for now
        if (scale_t->type == NNR_DATA_TYPE_FLOAT32)
            scale = *(float*)scale_t->data;
        else if (scale_t->type == NNR_DATA_TYPE_FLOAT16)
            scale = (float)*(float16_t*)scale_t->data;
        else
            return false;

        zp = 0;
        if (dq_or_q->inputs.size() >= 3 && dq_or_q->inputs[2] && dq_or_q->inputs[2]->ndata > 0) {
            tensor_t* zp_t = dq_or_q->inputs[2];
            switch (zp_t->type) {
            case NNR_DATA_TYPE_UINT8:  zp = *(uint8_t*)zp_t->data; break;
            case NNR_DATA_TYPE_INT8:   zp = *(int8_t*)zp_t->data; break;
            case NNR_DATA_TYPE_UINT16: zp = *(uint16_t*)zp_t->data; break;
            case NNR_DATA_TYPE_INT16:  zp = *(int16_t*)zp_t->data; break;
            case NNR_DATA_TYPE_INT32:  zp = *(int32_t*)zp_t->data; break;
            default: break;
            }
        }
        return true;
    };

    for (int i = 0; i < n; i++) {
        operator_t* nd = nodes[i];
        if (nd->skip || nd->folded) continue;

        // Pattern 1: DequantizeLinear → Op → QuantizeLinear
        // Look for DequantizeLinear nodes
        if (nd->op_type != "DequantizeLinear") continue;
        if (nd->outputs.empty()) continue;

        tensor_t* dq_output = nd->outputs[0];  // float tensor after dequantize
        tensor_t* dq_input = nd->inputs[0];     // quantized input tensor

        // DQ output must have exactly 1 consumer
        if (count_consumers(dq_output, i) != 1) continue;

        // Find the consumer op
        int consumer_idx = -1;
        for (int j = 0; j < n; j++) {
            if (j == i || nodes[j]->skip || nodes[j]->folded) continue;
            for (auto* t : nodes[j]->inputs)
                if (t == dq_output) { consumer_idx = j; break; }
            if (consumer_idx >= 0) break;
        }
        if (consumer_idx < 0) continue;

        operator_t* consumer = nodes[consumer_idx];

        // Skip if consumer is another DequantizeLinear or QuantizeLinear (handled separately)
        if (consumer->op_type == "DequantizeLinear" || consumer->op_type == "QuantizeLinear")
            continue;

        // Only fuse when the consumer op supports quantized (uint8/int8) input.
        // Ops not in this list keep DQ/Q in place → correct float fallback.
        static const std::unordered_set<std::string_view> quant_capable_ops = {
            "MaxPool", "AveragePool", "Concat", "Reshape", "Transpose",
            "Flatten", "Squeeze", "Unsqueeze", "Relu", "Clip", "Pad",
            "Slice", "Gather", "Split", "Resize",
        };
        if (quant_capable_ops.find(consumer->op_type) == quant_capable_ops.end())
            continue;

        // Check if consumer's output goes to a QuantizeLinear
        if (consumer->outputs.empty()) continue;
        tensor_t* consumer_output = consumer->outputs[0];

        int q_idx = -1;
        for (int j = 0; j < n; j++) {
            if (j == consumer_idx || nodes[j]->skip || nodes[j]->folded) continue;
            if (nodes[j]->op_type == "QuantizeLinear") {
                for (auto* t : nodes[j]->inputs)
                    if (t == consumer_output) { q_idx = j; break; }
            }
            if (q_idx >= 0) break;
        }
        if (q_idx < 0) continue;

        operator_t* q_node = nodes[q_idx];

        // Consumer output must have exactly 1 consumer (the Q node)
        if (count_consumers(consumer_output, consumer_idx) != 1) continue;

        // Extract scale/zp from both DQ and Q
        float dq_scale, q_scale;
        int32_t dq_zp, q_zp;
        if (!get_scale_zp(nd, dq_scale, dq_zp)) continue;
        if (!get_scale_zp(q_node, q_scale, q_zp)) continue;

        // Fuse: propagate quant metadata onto tensors
        dq_input->set_quant(dq_scale, dq_zp);

        tensor_t* q_output = q_node->outputs[0];
        q_output->set_quant(q_scale, q_zp);

        // Rewire: consumer reads from DQ's input (quantized) directly
        for (auto& t : consumer->inputs)
            if (t == dq_output) t = dq_input;

        // Rewire: consumer writes to Q's output (quantized) directly
        for (auto& t : consumer->outputs)
            if (t == consumer_output) t = q_output;

        // Output type and shape must match for quantized storage. Use
        // reshape() rather than reinit() so the call short-circuits when
        // q_output already has the right shape/type — bare reinit() always
        // frees data + calls webgpu::forget(), evicting any GPU buffer that
        // an earlier fuse_qdq_compute round registered for this tensor.
        if (!q_output->reshape(consumer_output->dim_span(), dq_input->type))
            continue;

        // Mark DQ and Q as folded (not skip).  After rewiring the consumer,
        // these nodes' I/O tensors are orphaned.  Using 'skip' would cause
        // assign_layouts' follow_skip_chain to traverse them, breaking NHWC
        // chain formation.  'folded' is also excluded from consumer_count,
        // preventing false multi-consumer chain breaks.
        nd->folded = true;
        q_node->folded = true;

        // Re-reshape the consumer so workspace_size() reflects quantized path,
        // and ensure workspace is large enough for the dequantize float buffer.
        // GPU backends (WebGPU pool/elementwise) commonly reject non-float32
        // inputs; after rewiring the inputs to int8/uint8 their reshape returns
        // false and any per-tensor backend state (GPU buffers, pipelines) is
        // never set up — runtime exec then dereferences null. Demote to a
        // CPU consumer in that case.
        if (!consumer->reshape() && consumer->resolved_backend != static_cast<uint8_t>(backend_t::CPU)) {
            operator_t* cpu_c = solve_operator(consumer->op_type, consumer->opset,
                                               ctx->attr_pool, backend_t::CPU);
            if (cpu_c) {
                cpu_c->ctx       = ctx;
                cpu_c->opset     = consumer->opset;
                cpu_c->op_type   = consumer->op_type;
                cpu_c->node_name = consumer->node_name;
                cpu_c->domain    = consumer->domain;
                cpu_c->inputs    = consumer->inputs;
                cpu_c->outputs   = consumer->outputs;
                cpu_c->attrs     = consumer->attrs;
                if (cpu_c->init() && cpu_c->reshape()) {
                    nodes[consumer_idx] = cpu_c;
                    consumer = cpu_c;
                }
            }
        }
        size_t ws = consumer->workspace_size();
        if (ws > ctx->workspace_size) {
            nnr_aligned_free(ctx->workspace);
            ctx->workspace = nnr_aligned_alloc(ws, 64);
            ctx->workspace_size = ws;
        }

        fused_count++;
    }

    if (fused_count > 0)
        fprintf(stderr, "[NNR] Fused %d QDQ patterns\n", fused_count);
}

} // namespace nnr
