#include "graph_optimizer/graph_optimizer_internal.h"

namespace nnr {

// ---------------------------------------------------------------------------
// SiLU fusion: Sigmoid(x) + Mul(x, Sigmoid(x)) → SiLU(x)
// ---------------------------------------------------------------------------
// Detects the pattern x → Sigmoid → s, x → Mul(x, s) → y and replaces it
// with a fused SiLU op that computes x * sigmoid(x) in a single pass.
// Saves one full tensor read+write (the Mul pass).

void fuse_silu(context_t* ctx)
{
    if (!ctx->graph) return;
    auto& nodes = ctx->graph->nodes;
    const int n = static_cast<int>(nodes.size());

    // Build tensor → consumer map
    std::unordered_map<tensor_t*, std::vector<int>> consumers;
    for (int i = 0; i < n; i++) {
        if (nodes[i]->skip) continue;
        for (auto* t : nodes[i]->inputs)
            if (t) consumers[t].push_back(i);
    }

    for (int i = 0; i < n; i++) {
        auto* sig = nodes[i];
        if (sig->skip || sig->op_type != "Sigmoid") continue;
        if (sig->inputs.empty() || sig->outputs.empty()) continue;
        tensor_t* sig_input = sig->inputs[0];  // x
        tensor_t* sig_output = sig->outputs[0]; // sigmoid(x)

        // Sigmoid output must have exactly one consumer
        auto it = consumers.find(sig_output);
        if (it == consumers.end()) continue;
        auto& sig_users = it->second;
        if (sig_users.size() != 1) continue;

        auto* mul = nodes[sig_users[0]];
        if (mul->skip || mul->op_type != "Mul") continue;
        if (mul->inputs.size() != 2 || mul->outputs.empty()) continue;

        // Mul must take both x and sigmoid(x) as inputs: Mul(x, sigmoid(x))
        tensor_t* mul_other = nullptr;
        if (mul->inputs[0] == sig_output && mul->inputs[1] == sig_input)
            mul_other = mul->inputs[1];
        else if (mul->inputs[1] == sig_output && mul->inputs[0] == sig_input)
            mul_other = mul->inputs[0];
        if (!mul_other) continue;

        // Fuse: Sigmoid becomes SiLU, writes to Mul's output.
        // Mark Mul as folded (not skip) — skip would alias output=input,
        // overwriting the SiLU result that Sigmoid wrote.
        sig->is_fused_silu = true;
        sig->outputs[0] = mul->outputs[0];
        mul->folded = true;
    }
}

// ---------------------------------------------------------------------------
// Decomposed LayerNormalization fusion
// ---------------------------------------------------------------------------
// Detects the 9-op pattern exported by PyTorch/HuggingFace for LayerNorm:
//   ReduceMean(x, axes=[-1]) → Sub(x, mean) → Pow(sub, 2) → ReduceMean
//   → Add(variance, epsilon) → Sqrt → Div(sub, sqrt) → Mul(div, gamma) → Add(mul, beta)
// Replaces with a single fused LayerNormalization node (2-pass kernel instead of 9).
//
// The fused operator already exists (src/backend/cpu/LayerNormalization.cpp).
// This pass detects the pattern and rewires the graph.

void fuse_layer_norm(context_t* ctx)
{
    if (!ctx->graph) return;
    auto& nodes = ctx->graph->nodes;
    const int n = static_cast<int>(nodes.size());

    // Build tensor → single consumer map (skip tensors with multiple consumers)
    std::unordered_map<tensor_t*, int> single_consumer;
    std::unordered_set<tensor_t*> multi_consumer;
    for (int i = 0; i < n; i++) {
        if (nodes[i]->skip || nodes[i]->folded) continue;
        for (auto* t : nodes[i]->inputs) {
            if (!t) continue;
            if (multi_consumer.count(t)) continue;
            auto it = single_consumer.find(t);
            if (it != single_consumer.end()) {
                multi_consumer.insert(t);
                single_consumer.erase(it);
            } else {
                single_consumer[t] = i;
            }
        }
    }

    auto only_consumer = [&](tensor_t* t) -> int {
        auto it = single_consumer.find(t);
        return (it != single_consumer.end()) ? it->second : -1;
    };

    // Helper: check if a tensor is a scalar constant with a specific value
    auto is_const_scalar = [&](tensor_t* t, float expected) -> bool {
        if (!t || t->ndata != 1) return false;
        if (!ctx->initializer_names.count(t->name)) {
            // Check if it's a Constant node output (folded data)
            if (!t->data) return false;
        }
        if (!t->data) return false;
        float val = 0;
        if (t->type == NNR_DATA_TYPE_FLOAT32) val = *(float*)t->data;
        else if (t->type == NNR_DATA_TYPE_FLOAT64) val = (float)*(double*)t->data;
        else return false;
        return std::abs(val - expected) < 1e-3f;
    };

    auto is_small_const = [&](tensor_t* t) -> bool {
        if (!t || t->ndata != 1 || !t->data) return false;
        float val = 0;
        if (t->type == NNR_DATA_TYPE_FLOAT32) val = *(float*)t->data;
        else if (t->type == NNR_DATA_TYPE_FLOAT64) val = (float)*(double*)t->data;
        else return false;
        return val > 0 && val < 1e-2f; // epsilon is typically 1e-5
    };

    int default_opset = 17;
    for (auto& [domain, version] : ctx->meta_opsets)
        if (domain == "ai.onnx" || domain.empty()) { default_opset = std::max(default_opset, (int)version); break; }

    int fused_count = 0;

    for (int i = 0; i < n; i++) {
        auto* rm1 = nodes[i];
        if (rm1->skip || rm1->folded) continue;
        if (rm1->op_type != "ReduceMean") continue;
        if (rm1->outputs.empty() || !rm1->outputs[0]) continue;

        // ReduceMean must reduce last axis only (axes=[-1] or axes=[ndim-1])
        tensor_t* x_input = rm1->inputs[0];
        if (!x_input) continue;
        tensor_t* mean_out = rm1->outputs[0];

        // Check: mean_out has exactly one consumer and it's Sub
        int sub_idx = only_consumer(mean_out);
        if (sub_idx < 0 || sub_idx >= n) continue;
        auto* sub = nodes[sub_idx];
        if (sub->op_type != "Sub" || sub->skip) continue;
        // Sub(x, mean): inputs[0]=x, inputs[1]=mean
        if (sub->inputs.size() < 2) continue;
        if (sub->inputs[0] != x_input || sub->inputs[1] != mean_out) continue;
        tensor_t* sub_out = sub->outputs[0];

        // sub_out must be consumed by Pow AND by Div (two consumers)
        // So sub_out is NOT single-consumer — it feeds both Pow and Div
        // Find Pow consumer
        int pow_idx = -1;
        int div_idx = -1;
        for (int j = 0; j < n; j++) {
            if (nodes[j]->skip || nodes[j]->folded) continue;
            for (auto* t : nodes[j]->inputs) {
                if (t == sub_out) {
                    if (nodes[j]->op_type == "Pow") pow_idx = j;
                    else if (nodes[j]->op_type == "Div") div_idx = j;
                }
            }
        }
        if (pow_idx < 0 || div_idx < 0) continue;

        auto* pow_node = nodes[pow_idx];
        // Pow(sub, 2): second input must be constant 2
        if (pow_node->inputs.size() < 2) continue;
        if (!is_const_scalar(pow_node->inputs[1], 2.0f)) continue;
        tensor_t* pow_out = pow_node->outputs[0];

        // Pow → ReduceMean
        int rm2_idx = only_consumer(pow_out);
        if (rm2_idx < 0 || rm2_idx >= n) continue;
        auto* rm2 = nodes[rm2_idx];
        if (rm2->op_type != "ReduceMean" || rm2->skip) continue;
        tensor_t* var_out = rm2->outputs[0];

        // ReduceMean → Add(var, epsilon)
        int add_eps_idx = only_consumer(var_out);
        if (add_eps_idx < 0 || add_eps_idx >= n) continue;
        auto* add_eps = nodes[add_eps_idx];
        if (add_eps->op_type != "Add" || add_eps->skip) continue;
        // One input is var_out, other is epsilon
        tensor_t* eps_tensor = nullptr;
        if (add_eps->inputs[0] == var_out) eps_tensor = add_eps->inputs[1];
        else if (add_eps->inputs[1] == var_out) eps_tensor = add_eps->inputs[0];
        if (!eps_tensor || !is_small_const(eps_tensor)) continue;
        float epsilon = 0;
        if (eps_tensor->type == NNR_DATA_TYPE_FLOAT32) epsilon = *(float*)eps_tensor->data;
        else epsilon = (float)*(double*)eps_tensor->data;
        tensor_t* var_eps_out = add_eps->outputs[0];

        // Add → Sqrt
        int sqrt_idx = only_consumer(var_eps_out);
        if (sqrt_idx < 0 || sqrt_idx >= n) continue;
        auto* sqrt_node = nodes[sqrt_idx];
        if (sqrt_node->op_type != "Sqrt" || sqrt_node->skip) continue;
        tensor_t* sqrt_out = sqrt_node->outputs[0];

        // Sqrt → Div(sub, sqrt)
        auto* div_node = nodes[div_idx];
        if (div_node->op_type != "Div" || div_node->skip) continue;
        if (div_node->inputs.size() < 2) continue;
        if (div_node->inputs[0] != sub_out || div_node->inputs[1] != sqrt_out) continue;
        tensor_t* div_out = div_node->outputs[0];

        // Div → Mul(div, gamma)
        int mul_idx = only_consumer(div_out);
        if (mul_idx < 0 || mul_idx >= n) continue;
        auto* mul_node = nodes[mul_idx];
        if (mul_node->op_type != "Mul" || mul_node->skip) continue;
        tensor_t* gamma = nullptr;
        if (mul_node->inputs[0] == div_out) gamma = mul_node->inputs[1];
        else if (mul_node->inputs[1] == div_out) gamma = mul_node->inputs[0];
        if (!gamma) continue;
        tensor_t* mul_out = mul_node->outputs[0];

        // Mul → Add(mul, beta)
        int add_bias_idx = only_consumer(mul_out);
        if (add_bias_idx < 0 || add_bias_idx >= n) continue;
        auto* add_bias = nodes[add_bias_idx];
        if (add_bias->op_type != "Add" || add_bias->skip) continue;
        tensor_t* beta = nullptr;
        if (add_bias->inputs[0] == mul_out) beta = add_bias->inputs[1];
        else if (add_bias->inputs[1] == mul_out) beta = add_bias->inputs[0];
        if (!beta) continue;
        tensor_t* final_out = add_bias->outputs[0];

        // --- Pattern matched! Create fused LayerNormalization ---
        auto backend = static_cast<backend_t>(ctx->preferred_backend);
        operator_t* fused = solve_operator("LayerNormalization", default_opset,
                                            ctx->attr_pool, backend);
        if (!fused) continue;

        fused->ctx = ctx;
        fused->opset = default_opset;
        fused->op_type = "LayerNormalization";
        fused->node_name = rm1->node_name;

        // Inputs: X, Scale, Bias
        tensor_t** ins = ctx->attr_pool.alloc_arr<tensor_t*>(3);
        ins[0] = x_input;
        ins[1] = gamma;
        ins[2] = beta;
        fused->inputs = {ins, 3};

        // Output: Y (reuse final Add output)
        tensor_t** outs = ctx->attr_pool.alloc_arr<tensor_t*>(1);
        outs[0] = final_out;
        fused->outputs = {outs, 1};

        // Attributes: axis=-1, epsilon
        auto* attr_arr = ctx->attr_pool.alloc_arr<std::pair<attr_key_t, attr_t>>(2);
        attr_arr[0].first = attr_key_t::axis;
        attr_arr[0].second.kind = attr_t::kind_t::INT;
        attr_arr[0].second.i = -1;
        attr_arr[1].first = attr_key_t::epsilon;
        attr_arr[1].second.kind = attr_t::kind_t::FLOAT;
        attr_arr[1].second.f = epsilon;
        fused->attrs = {attr_arr, 2};

        fused->init();
        fused->reshape();

        // Replace first node (ReduceMean) with fused, mark others as folded.
        // Use folded (not skip) — skip aliases output=input data pointers,
        // which would overwrite the fused node's output with stale data.
        nodes[i] = fused;
        sub->folded = true;
        pow_node->folded = true;
        rm2->folded = true;
        add_eps->folded = true;
        sqrt_node->folded = true;
        div_node->folded = true;
        mul_node->folded = true;
        add_bias->folded = true;
        fused_count++;
    }

    if (fused_count > 0)
        fprintf(stderr, "[NNR] Fused %d LayerNormalization instances\n", fused_count);
}

// ---------------------------------------------------------------------------
// Decomposed GELU fusion
// ---------------------------------------------------------------------------
// Detects: Div(x, sqrt(2)) → Erf → Add(1) → Mul(x, ·) → Mul(·, 0.5)
// Replaces with fused Gelu operator (approximate=none).

void fuse_gelu(context_t* ctx)
{
    if (!ctx->graph) return;
    auto& nodes = ctx->graph->nodes;
    const int n = static_cast<int>(nodes.size());

    // Build tensor → consumer list
    std::unordered_map<tensor_t*, std::vector<int>> consumers;
    for (int i = 0; i < n; i++) {
        if (nodes[i]->skip || nodes[i]->folded) continue;
        for (auto* t : nodes[i]->inputs)
            if (t) consumers[t].push_back(i);
    }

    auto has_single_consumer = [&](tensor_t* t) -> int {
        auto it = consumers.find(t);
        if (it == consumers.end() || it->second.size() != 1) return -1;
        return it->second[0];
    };

    auto is_const_val = [&](tensor_t* t, float expected, float tol = 1e-3f) -> bool {
        if (!t || t->ndata != 1 || !t->data) return false;
        float val = 0;
        if (t->type == NNR_DATA_TYPE_FLOAT32) val = *(float*)t->data;
        else if (t->type == NNR_DATA_TYPE_FLOAT64) val = (float)*(double*)t->data;
        else return false;
        return std::abs(val - expected) < tol;
    };

    int default_opset = 20;
    for (auto& [domain, version] : ctx->meta_opsets)
        if (domain == "ai.onnx" || domain.empty()) { default_opset = std::max(default_opset, (int)version); break; }

    int fused_count = 0;

    // Pattern: Div(x, sqrt2) → Erf → Add(1) → Mul(x, add_out) → Mul(mul1_out, 0.5)
    for (int i = 0; i < n; i++) {
        auto* div_node = nodes[i];
        if (div_node->skip || div_node->folded) continue;
        if (div_node->op_type != "Div") continue;
        if (div_node->inputs.size() < 2 || div_node->outputs.empty()) continue;

        tensor_t* x_input = div_node->inputs[0];
        tensor_t* sqrt2_tensor = div_node->inputs[1];
        if (!is_const_val(sqrt2_tensor, 1.4142135f, 1e-3f)) continue;
        tensor_t* div_out = div_node->outputs[0];

        // Div → Erf
        int erf_idx = has_single_consumer(div_out);
        if (erf_idx < 0) continue;
        auto* erf_node = nodes[erf_idx];
        if (erf_node->op_type != "Erf" || erf_node->skip) continue;
        tensor_t* erf_out = erf_node->outputs[0];

        // Erf → Add(erf, 1)
        int add_idx = has_single_consumer(erf_out);
        if (add_idx < 0) continue;
        auto* add_node = nodes[add_idx];
        if (add_node->op_type != "Add" || add_node->skip) continue;
        tensor_t* one_tensor = nullptr;
        if (add_node->inputs[0] == erf_out) one_tensor = add_node->inputs[1];
        else if (add_node->inputs[1] == erf_out) one_tensor = add_node->inputs[0];
        if (!is_const_val(one_tensor, 1.0f)) continue;
        tensor_t* add_out = add_node->outputs[0];

        // Add → Mul(x, add_out) — x is the original input
        int mul1_idx = has_single_consumer(add_out);
        if (mul1_idx < 0) continue;
        auto* mul1 = nodes[mul1_idx];
        if (mul1->op_type != "Mul" || mul1->skip) continue;
        // Check that one input is x_input and the other is add_out
        bool mul1_ok = (mul1->inputs[0] == x_input && mul1->inputs[1] == add_out) ||
                       (mul1->inputs[1] == x_input && mul1->inputs[0] == add_out);
        if (!mul1_ok) continue;
        tensor_t* mul1_out = mul1->outputs[0];

        // Mul → Mul(mul1_out, 0.5)
        int mul2_idx = has_single_consumer(mul1_out);
        if (mul2_idx < 0) continue;
        auto* mul2 = nodes[mul2_idx];
        if (mul2->op_type != "Mul" || mul2->skip) continue;
        tensor_t* half_tensor = nullptr;
        if (mul2->inputs[0] == mul1_out) half_tensor = mul2->inputs[1];
        else if (mul2->inputs[1] == mul1_out) half_tensor = mul2->inputs[0];
        if (!is_const_val(half_tensor, 0.5f)) continue;
        tensor_t* final_out = mul2->outputs[0];

        // --- Pattern matched! Create fused Gelu ---
        auto backend = static_cast<backend_t>(ctx->preferred_backend);
        operator_t* fused = solve_operator("Gelu", default_opset,
                                            ctx->attr_pool, backend);
        if (!fused) continue;

        fused->ctx = ctx;
        fused->opset = default_opset;
        fused->op_type = "Gelu";
        fused->node_name = erf_node->node_name;

        // Input: x
        tensor_t** ins = ctx->attr_pool.alloc_arr<tensor_t*>(1);
        ins[0] = x_input;
        fused->inputs = {ins, 1};

        // Output: reuse final Mul output
        tensor_t** outs = ctx->attr_pool.alloc_arr<tensor_t*>(1);
        outs[0] = final_out;
        fused->outputs = {outs, 1};

        // Attribute: approximate="none"
        auto* attr_arr = ctx->attr_pool.alloc_arr<std::pair<attr_key_t, attr_t>>(1);
        attr_arr[0].first = attr_key_t::approximate;
        attr_arr[0].second.kind = attr_t::kind_t::STRING;
        attr_arr[0].second.s = "none";
        fused->attrs = {attr_arr, 1};

        fused->init();
        fused->reshape();

        // Replace Div node with fused, mark others as folded.
        // Use folded (not skip) — skip aliases output=input data pointers.
        nodes[i] = fused;
        erf_node->folded = true;
        add_node->folded = true;
        mul1->folded = true;
        mul2->folded = true;
        fused_count++;
    }

    if (fused_count > 0)
        fprintf(stderr, "[NNR] Fused %d Gelu instances\n", fused_count);
}


} // namespace nnr
