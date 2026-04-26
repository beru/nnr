#include "graph_optimizer/graph_optimizer_internal.h"

namespace nnr {

// Convert uint8 weight/zp tensors to int8 in-place (subtract 128).
// Preserves (val - zp) for integer arithmetic.
static void convert_weight_uint8_to_int8(tensor_t* w, tensor_t* w_zp)
{
    if (w->type != NNR_DATA_TYPE_UINT8) return;
    auto* src = (uint8_t*)w->data;
    auto* dst = (int8_t*)src;  // same size, in-place
    for (size_t i = 0; i < w->ndata; i++)
        dst[i] = (int8_t)((int)src[i] - 128);
    w->type = NNR_DATA_TYPE_INT8;

    if (w_zp && w_zp->ndata > 0 && w_zp->type == NNR_DATA_TYPE_UINT8) {
        auto* zs = (uint8_t*)w_zp->data;
        auto* zd = (int8_t*)zs;
        for (size_t i = 0; i < w_zp->ndata; i++)
            zd[i] = (int8_t)((int)zs[i] - 128);
        w_zp->type = NNR_DATA_TYPE_INT8;
    }
}

void fuse_qdq_compute(context_t* ctx)
{
    if (!ctx->graph) return;
    auto& nodes = ctx->graph->nodes;
    const int n = static_cast<int>(nodes.size());
    const auto backend = static_cast<backend_t>(ctx->preferred_backend);
    int default_opset = 13;

    // Return either `fused` (if its init+reshape succeeded) or a freshly-built
    // CPU equivalent. Used after the fused op is constructed and its inputs/
    // outputs wired up — when the chosen backend's QLinear op rejects a
    // shape/dtype combo, init/reshape returns false and any backend-specific
    // state (GPU buffers, pipelines) is never set up. Demote to CPU so runtime
    // exec doesn't dereference null. Returns nullptr only if even CPU fails.
    auto demote_if_needed = [&](operator_t* fused) -> operator_t* {
        if (fused->init() && fused->reshape()) return fused;
        if (fused->resolved_backend == static_cast<uint8_t>(backend_t::CPU))
            return nullptr;
        operator_t* cpu_f = solve_operator(fused->op_type, fused->opset,
                                           ctx->attr_pool, backend_t::CPU);
        if (!cpu_f) return nullptr;
        cpu_f->ctx       = ctx;
        cpu_f->opset     = fused->opset;
        cpu_f->op_type   = fused->op_type;
        cpu_f->node_name = fused->node_name;
        cpu_f->domain    = fused->domain;
        cpu_f->inputs    = fused->inputs;
        cpu_f->outputs   = fused->outputs;
        cpu_f->attrs     = fused->attrs;
        if (cpu_f->init() && cpu_f->reshape()) return cpu_f;
        return nullptr;
    };
    for (auto& [domain, version] : ctx->meta_opsets) {
        if (domain == "ai.onnx" || domain.empty()) {
            default_opset = std::max(default_opset, (int)version); break;
        }
    }

    // Build two producer maps:
    // - producer: active nodes only (for activation DQ and Q detection)
    // - all_producer: includes folded nodes (for weight/bias DQ that fold_run pre-folded)
    std::unordered_map<tensor_t*, int> producer, all_producer;
    for (int i = 0; i < n; i++) {
        for (auto* t : nodes[i]->outputs)
            all_producer[t] = i;
        if (nodes[i]->skip || nodes[i]->folded) continue;
        for (auto* t : nodes[i]->outputs)
            producer[t] = i;
    }

    // Count consumers of a tensor (excluding skip/folded nodes and a specific index)
    auto count_consumers = [&](tensor_t* tensor, int skip_idx) -> int {
        int count = 0;
        for (int j = 0; j < n; j++) {
            if (j == skip_idx || nodes[j]->skip || nodes[j]->folded) continue;
            for (auto* t : nodes[j]->inputs)
                if (t == tensor) count++;
        }
        return count;
    };

    // Find the single consumer Q node for a tensor, or -1
    auto find_q_consumer = [&](tensor_t* tensor, int src_idx) -> int {
        for (int j = 0; j < n; j++) {
            if (j == src_idx || nodes[j]->skip || nodes[j]->folded) continue;
            if (nodes[j]->op_type != "QuantizeLinear") continue;
            if (!nodes[j]->inputs.empty() && nodes[j]->inputs[0] == tensor)
                return j;
        }
        return -1;
    };

    // Helper: get active DQ node feeding a tensor, or nullptr
    auto get_dq_producer = [&](tensor_t* tensor) -> operator_t* {
        auto it = producer.find(tensor);
        if (it == producer.end()) return nullptr;
        auto* nd = nodes[it->second];
        if (nd->op_type != "DequantizeLinear") return nullptr;
        return nd;
    };

    // Helper: get DQ node feeding a tensor, including folded DQ nodes
    // (weight/bias DQ nodes are pre-folded by fold_run before optimizer runs)
    auto get_dq_producer_any = [&](tensor_t* tensor) -> operator_t* {
        auto it = all_producer.find(tensor);
        if (it == all_producer.end()) return nullptr;
        auto* nd = nodes[it->second];
        if (nd->op_type != "DequantizeLinear") return nullptr;
        return nd;
    };

    // Helper: get zero-point tensor from DQ/Q node (index 2), returns nullptr if absent
    auto get_zp = [](operator_t* dq_or_q) -> tensor_t* {
        if (dq_or_q->inputs.size() >= 3 && dq_or_q->inputs[2])
            return dq_or_q->inputs[2];
        return nullptr;
    };

    int fused_count = 0;

    for (int i = 0; i < n; i++) {
        operator_t* op = nodes[i];
        if (op->skip || op->folded) continue;

        bool is_conv = (op->op_type == "Conv");
        bool is_matmul = (op->op_type == "MatMul");
        if (!is_conv && !is_matmul) continue;

        // --- Check all inputs come from DQ nodes ---

        // Activation (input 0) — must be from an active (non-folded) DQ
        if (op->inputs.empty() || !op->inputs[0]) continue;
        operator_t* dq_x = get_dq_producer(op->inputs[0]);
        if (!dq_x || dq_x->inputs.size() < 2) continue;

        // Weight (input 1) — may be from a folded DQ (fold_run pre-folds constant DQs)
        if (op->inputs.size() < 2 || !op->inputs[1]) continue;
        operator_t* dq_w = get_dq_producer_any(op->inputs[1]);
        if (!dq_w || dq_w->inputs.size() < 2) continue;

        // Bias (Conv only, optional — input 2, also may be folded)
        operator_t* dq_bias = nullptr;
        if (is_conv && op->inputs.size() >= 3 && op->inputs[2]) {
            dq_bias = get_dq_producer_any(op->inputs[2]);
            // bias DQ is optional — Conv can work without bias
        }

        // --- Check output goes to a single Q node ---
        if (op->outputs.empty() || !op->outputs[0]) continue;
        tensor_t* op_output = op->outputs[0];
        if (count_consumers(op_output, i) != 1) continue;
        int q_idx = find_q_consumer(op_output, i);
        if (q_idx < 0) continue;
        operator_t* q_y = nodes[q_idx];
        if (q_y->inputs.size() < 2) continue;

        // --- Activation DQ may have multiple consumers (e.g. skip connections) ---
        // We still fuse: the fused node reads dq_x's quantized input directly.
        // The DQ stays active if other consumers still need its float output.

        // --- Create fused QLinearConv / QLinearMatMul ---
        const char* new_op_type = is_conv ? "QLinearConv" : "QLinearMatMul";
        operator_t* fused = solve_operator(new_op_type, default_opset,
                                           ctx->attr_pool, backend);
        if (!fused) continue;

        fused->ctx       = ctx;
        fused->opset     = default_opset;
        fused->op_type   = new_op_type;
        fused->node_name = op->node_name;

        if (is_conv) {
            // QLinearConv: x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, [bias]
            int n_inputs = dq_bias ? 9 : 8;
            tensor_t** ins = ctx->attr_pool.alloc_arr<tensor_t*>(n_inputs);
            ins[0] = dq_x->inputs[0];                          // x (quantized)
            ins[1] = dq_x->inputs[1];                          // x_scale
            ins[2] = get_zp(dq_x);                             // x_zp (may be nullptr)
            ins[3] = dq_w->inputs[0];                          // w (quantized)
            ins[4] = dq_w->inputs[1];                          // w_scale
            ins[5] = get_zp(dq_w);                             // w_zp (may be nullptr)
            ins[6] = q_y->inputs[1];                           // y_scale
            ins[7] = get_zp(q_y);                              // y_zp (may be nullptr)
            if (dq_bias)
                ins[8] = dq_bias->inputs[0];                   // bias (int32)
            fused->inputs = {ins, static_cast<size_t>(n_inputs)};

            // Copy Conv attributes (group, kernel_shape, pads, strides, dilations, auto_pad)
            fused->attrs = op->attrs;
        } else {
            // QLinearMatMul: a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp
            tensor_t** ins = ctx->attr_pool.alloc_arr<tensor_t*>(8);
            ins[0] = dq_x->inputs[0];                          // a (quantized)
            ins[1] = dq_x->inputs[1];                          // a_scale
            ins[2] = get_zp(dq_x);                             // a_zp
            ins[3] = dq_w->inputs[0];                          // b (quantized)
            ins[4] = dq_w->inputs[1];                          // b_scale
            ins[5] = get_zp(dq_w);                             // b_zp
            ins[6] = q_y->inputs[1];                           // y_scale
            ins[7] = get_zp(q_y);                              // y_zp
            fused->inputs = {ins, 8};
        }

        // Output: quantized output from Q node
        tensor_t** outs = ctx->attr_pool.alloc_arr<tensor_t*>(1);
        outs[0] = q_y->outputs[0];
        fused->outputs = {outs, 1};

        // Convert uint8 weights to int8 (QLinearConv/MatMul cast weights as int8_t*)
        tensor_t* w_tensor = is_conv ? fused->inputs[3] : fused->inputs[3];
        tensor_t* w_zp_tensor = is_conv ? fused->inputs[5] : fused->inputs[5];
        convert_weight_uint8_to_int8(w_tensor, w_zp_tensor);

        operator_t* placed = demote_if_needed(fused);
        if (!placed) continue;

        // Replace the Conv/MatMul node, mark source DQ/Q as folded
        nodes[i] = placed;
        // Only fold activation DQ if no other active consumers remain
        // (fused node bypasses dq_x, so check after replacement)
        if (count_consumers(dq_x->outputs[0], all_producer[op->inputs[0]]) == 0)
            dq_x->folded = true;
        dq_w->folded = true;
        if (dq_bias) dq_bias->folded = true;
        q_y->folded = true;

        fused_count++;
    }

    // --- Phase 2: Fuse DQ+DQ → Add → Q into QLinearAdd ---
    // Residual skip connections in QDQ models: both Add inputs come from DQ,
    // output goes to Q. Fusing eliminates DQ+Q overhead (~93ms in SSD-12-QDQ).
    int add_fused_count = 0;

    // Rebuild producer map after Conv/MatMul fusion changed nodes[]
    producer.clear();
    all_producer.clear();
    for (int i = 0; i < n; i++) {
        for (auto* t : nodes[i]->outputs)
            all_producer[t] = i;
        if (nodes[i]->skip || nodes[i]->folded) continue;
        for (auto* t : nodes[i]->outputs)
            producer[t] = i;
    }

    // Helper: when Add's input is the fp32 output of a quant-capable op
    // (e.g. Relu in the ssd-12-qdq residual idiom), the SAME fp32 value
    // is almost always quantized downstream via a Q node for the main path
    // (e.g. Relu → Q → int8 → DQ → next Conv). We can treat the parallel
    // DQ of that Q as the effective source, which makes the residual path
    // fusable. Returns the DQ node if the pattern holds, nullptr otherwise.
    auto find_implicit_dq = [&](tensor_t* fp32_tensor) -> operator_t* {
        // Scan consumers of fp32_tensor for a QuantizeLinear. Accept folded
        // Q nodes too: their inputs[0]/outputs[0] references are still valid,
        // and they may have been folded in a prior fusion iteration (after
        // the first Add in a block chain fuses, the Q on the shared Relu
        // output gets folded — subsequent Adds still need to find it).
        int q_idx = -1;
        for (int j = 0; j < n; j++) {
            if (nodes[j]->skip) continue;
            if (nodes[j]->op_type != "QuantizeLinear") continue;
            if (!nodes[j]->inputs.empty() && nodes[j]->inputs[0] == fp32_tensor) {
                q_idx = j;
                break;
            }
        }
        if (q_idx < 0) return nullptr;
        // Scan consumers of Q's output for a DequantizeLinear. Phase 1 has
        // already folded the DQ feeding a main-path Conv (rewiring the Conv
        // to read Q's int8 output directly). That folded DQ still carries
        // the scale/zp tensors in its inputs[1..2] — use it as-is.
        tensor_t* q_out = nodes[q_idx]->outputs[0];
        for (int j = 0; j < n; j++) {
            if (nodes[j]->skip) continue;
            if (nodes[j]->op_type != "DequantizeLinear") continue;
            if (!nodes[j]->inputs.empty() && nodes[j]->inputs[0] == q_out)
                return nodes[j];
        }
        return nullptr;
    };

    for (int i = 0; i < n; i++) {
        operator_t* op = nodes[i];
        if (op->skip || op->folded) continue;
        if (op->op_type != "Add") continue;
        if (op->inputs.size() < 2 || !op->inputs[0] || !op->inputs[1]) continue;

        // Both inputs must come from active DQ nodes. For the ssd-12-qdq
        // fp32-direct residual idiom (input from Relu with parallel Q/DQ),
        // find the implicit DQ from the parallel main path.
        operator_t* dq_a = get_dq_producer(op->inputs[0]);
        if (!dq_a) dq_a = find_implicit_dq(op->inputs[0]);
        operator_t* dq_b = get_dq_producer(op->inputs[1]);
        if (!dq_b) dq_b = find_implicit_dq(op->inputs[1]);
        if (!dq_a || dq_a->inputs.size() < 2) continue;
        if (!dq_b || dq_b->inputs.size() < 2) continue;

        // Output must go to a single Q node (possibly via Relu)
        if (op->outputs.empty() || !op->outputs[0]) continue;
        tensor_t* add_output = op->outputs[0];
        if (count_consumers(add_output, i) != 1) continue;

        // Check for optional Relu between Add and Q
        int relu_idx = -1;
        tensor_t* pre_q_output = add_output;
        {
            // Find the single consumer of Add's output
            int consumer = -1;
            for (int j = 0; j < n; j++) {
                if (j == i || nodes[j]->skip || nodes[j]->folded) continue;
                for (auto* t : nodes[j]->inputs)
                    if (t == add_output) { consumer = j; break; }
                if (consumer >= 0) break;
            }
            if (consumer >= 0 && nodes[consumer]->op_type == "Relu") {
                relu_idx = consumer;
                pre_q_output = nodes[consumer]->outputs[0];
                if (count_consumers(pre_q_output, consumer) != 1) continue;
            }
        }

        int q_idx = find_q_consumer(pre_q_output, relu_idx >= 0 ? relu_idx : i);
        if (q_idx < 0) continue;
        operator_t* q_y = nodes[q_idx];
        if (q_y->inputs.size() < 2) continue;

        // Create QLinearAdd: (a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp)
        operator_t* fused = solve_operator("QLinearAdd", default_opset,
                                           ctx->attr_pool, backend);
        if (!fused) continue;

        fused->ctx       = ctx;
        fused->opset     = default_opset;
        fused->op_type   = "QLinearAdd";
        fused->node_name = op->node_name;

        tensor_t** ins = ctx->attr_pool.alloc_arr<tensor_t*>(8);
        ins[0] = dq_a->inputs[0];    // a (quantized)
        ins[1] = dq_a->inputs[1];    // a_scale
        ins[2] = get_zp(dq_a);       // a_zp
        ins[3] = dq_b->inputs[0];    // b (quantized)
        ins[4] = dq_b->inputs[1];    // b_scale
        ins[5] = get_zp(dq_b);       // b_zp
        ins[6] = q_y->inputs[1];     // y_scale
        ins[7] = get_zp(q_y);        // y_zp
        fused->inputs = {ins, 8};

        tensor_t** outs = ctx->attr_pool.alloc_arr<tensor_t*>(1);
        outs[0] = q_y->outputs[0];
        fused->outputs = {outs, 1};

        operator_t* placed = demote_if_needed(fused);
        if (!placed) continue;

        // Replace Add node, fold source DQ/Q/Relu
        nodes[i] = placed;
        if (count_consumers(dq_a->outputs[0], all_producer[op->inputs[0]]) == 0)
            dq_a->folded = true;
        if (count_consumers(dq_b->outputs[0], all_producer[op->inputs[1]]) == 0)
            dq_b->folded = true;
        if (relu_idx >= 0)
            nodes[relu_idx]->folded = true;
        q_y->folded = true;

        add_fused_count++;
    }

    if (fused_count > 0 || add_fused_count > 0)
        fprintf(stderr, "[NNR] Fused %d QDQ compute + %d QDQ add patterns\n",
                fused_count, add_fused_count);
}

} // namespace nnr
