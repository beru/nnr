#include "graph_optimizer/graph_optimizer_internal.h"

namespace nnr {

// ---------------------------------------------------------------------------
// Scaled Dot-Product Attention fusion
// ---------------------------------------------------------------------------
// Detects: MatMul(Q, Transpose(K)) → Softmax → MatMul(attn, V)
// Replaces with fused SDPA operator that avoids materializing [B,S,S] attention matrix.
// Also folds the K Transpose into the fused kernel.

extern operator_t* create_sdpa_operator(pool_t& pool);

void fuse_sdpa(context_t* ctx)
{
    if (!ctx->graph) return;
    auto& nodes = ctx->graph->nodes;
    const int n = static_cast<int>(nodes.size());

    // Build tensor → consumer map
    std::unordered_map<tensor_t*, std::vector<int>> consumers;
    for (int i = 0; i < n; i++) {
        if (nodes[i]->skip || nodes[i]->folded) continue;
        for (auto* t : nodes[i]->inputs)
            if (t) consumers[t].push_back(i);
    }

    auto single_consumer = [&](tensor_t* t) -> int {
        auto it = consumers.find(t);
        if (it == consumers.end() || it->second.size() != 1) return -1;
        return it->second[0];
    };

    int fused_count = 0;

    for (int i = 0; i < n; i++) {
        auto* mm1 = nodes[i];
        if (mm1->skip || mm1->folded) continue;
        if (mm1->op_type != "MatMul") continue;
        if (mm1->inputs.size() < 2 || mm1->outputs.empty()) continue;

        // MatMul_1 output must feed into Softmax
        tensor_t* mm1_out = mm1->outputs[0];
        int softmax_idx = single_consumer(mm1_out);
        if (softmax_idx < 0) continue;
        auto* softmax = nodes[softmax_idx];
        if (softmax->op_type != "Softmax" || softmax->skip) continue;
        tensor_t* softmax_out = softmax->outputs[0];

        // Softmax output must feed into MatMul_2
        int mm2_idx = single_consumer(softmax_out);
        if (mm2_idx < 0) continue;
        auto* mm2 = nodes[mm2_idx];
        if (mm2->op_type != "MatMul" || mm2->skip) continue;
        if (mm2->inputs.size() < 2) continue;
        // Softmax output must be first input of mm2
        if (mm2->inputs[0] != softmax_out) continue;

        // Check shapes: mm1 should be [B, S, D] × [B, D, S] → [B, S, S]
        tensor_t* q_tensor = mm1->inputs[0];    // Q: [B, S, D]
        tensor_t* kt_tensor = mm1->inputs[1];   // K^T: [B, D, S]
        tensor_t* v_tensor = mm2->inputs[1];     // V: [B, S, D]

        if (!q_tensor || !kt_tensor || !v_tensor) continue;
        if (q_tensor->ndim < 3 || kt_tensor->ndim < 3 || v_tensor->ndim < 3) continue;

        int qndim = q_tensor->ndim;
        int S = q_tensor->dims[qndim - 2];      // sequence length
        int D = q_tensor->dims[qndim - 1];       // head dimension
        int KS = kt_tensor->dims[kt_tensor->ndim - 1]; // K's sequence length

        // Verify dimensions match for attention
        if (S != KS) continue;  // Q seq_len must equal K seq_len
        if (kt_tensor->dims[kt_tensor->ndim - 2] != D) continue; // K^T has D rows

        // V must be [B, S, D]
        if (v_tensor->dims[v_tensor->ndim - 2] != S) continue;
        int VD = v_tensor->dims[v_tensor->ndim - 1];

        // Find K before transpose: K^T comes from Transpose(K, perm=[0,2,1])
        // Trace kt_tensor back to find the Transpose node
        tensor_t* k_tensor = nullptr;
        int transpose_idx = -1;
        for (int j = 0; j < n; j++) {
            auto* nd = nodes[j];
            if (nd->skip || nd->folded) continue;
            if (nd->op_type != "Transpose") continue;
            if (nd->outputs.empty() || nd->outputs[0] != kt_tensor) continue;
            // Verify perm is [0, 2, 1] (or equivalent for swapping last two dims)
            int64_t* perm_data = nullptr;
            int perm_len = nd->attribute(attr_key_t::perm, perm_data);
            if (perm_len < 2) continue;
            // Check last two dims are swapped
            bool last_two_swapped = (perm_data[perm_len - 2] == perm_len - 1 &&
                                     perm_data[perm_len - 1] == perm_len - 2);
            if (!last_two_swapped) continue;
            k_tensor = nd->inputs[0];
            transpose_idx = j;
            break;
        }
        if (!k_tensor || transpose_idx < 0) continue;

        // K must be [B, S, D] (same shape as V)
        if (k_tensor->ndim != v_tensor->ndim) continue;
        if (k_tensor->dims[k_tensor->ndim - 2] != S) continue;
        if (k_tensor->dims[k_tensor->ndim - 1] != D) continue;

        tensor_t* final_out = mm2->outputs[0]; // [B, S, VD]

        // --- Pattern matched: Q×K^T → Softmax → ×V ---
        operator_t* fused = create_sdpa_operator(ctx->attr_pool);
        if (!fused) continue;

        fused->ctx = ctx;
        fused->opset = 1;
        fused->op_type = "SDPA";
        fused->node_name = mm1->node_name;

        // Inputs: Q, K (not transposed), V
        tensor_t** ins = ctx->attr_pool.alloc_arr<tensor_t*>(3);
        ins[0] = q_tensor;  // Q [B, S, D] (already scaled)
        ins[1] = k_tensor;  // K [B, S, D] (NOT transposed)
        ins[2] = v_tensor;  // V [B, S, VD]
        fused->inputs = {ins, 3};

        // Output: reuse mm2's output
        tensor_t** outs = ctx->attr_pool.alloc_arr<tensor_t*>(1);
        outs[0] = final_out;
        fused->outputs = {outs, 1};

        fused->init();
        fused->reshape();

        // Replace first MatMul with fused, mark others as folded
        nodes[i] = fused;
        softmax->folded = true;
        mm2->folded = true;
        // Also fold the K Transpose since we handle it internally
        auto it = consumers.find(kt_tensor);
        if (it != consumers.end() && it->second.size() == 1)
            nodes[transpose_idx]->folded = true;

        fused_count++;
    }

    if (fused_count > 0)
        fprintf(stderr, "[NNR] Fused %d SDPA instances\n", fused_count);
}


} // namespace nnr
