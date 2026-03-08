#include "graph_optimizer/graph_optimizer_internal.h"

namespace nnr {

// ---------------------------------------------------------------------------
// Operator decomposition — rewrite composite ops into supported primitives
// ---------------------------------------------------------------------------
// Runs before first inference. Scans for ops that have no backend
// implementation and replaces them with equivalent sequences of primitive ops.
// This is format-agnostic: loaders emit canonical op names (e.g. "Rsqrt"),
// and this pass decomposes them for the target backend.
//
// Decomposition recipes:
//   Rsqrt(x) → y           :  t = Sqrt(x);  y = Reciprocal(t)
//   NotEqual(a, b) → y     :  t = Equal(a, b);  y = Not(t)
//   SquaredDifference(a,b)→y:  t = Sub(a, b);  y = Mul(t, t)

struct decomp_recipe {
    const char* op;          // composite op name to match
    const char* first_op;    // first primitive op
    const char* second_op;   // second primitive op
    int second_n_inputs;     // 1 = unary(t), 2 = binary(t,t)
};

static constexpr decomp_recipe decomp_table[] = {
    {"Rsqrt",              "Sqrt",  "Reciprocal", 1},
    {"NotEqual",           "Equal", "Not",        1},
    {"SquaredDifference",  "Sub",   "Mul",        2},
};

void decompose_ops(context_t* ctx)
{
    if (!ctx || !ctx->graph) return;
    auto& nodes = ctx->graph->nodes;
    const auto backend = static_cast<backend_t>(ctx->preferred_backend);

    // Find default opset from model metadata
    int default_opset = 13;
    for (auto& [domain, version] : ctx->meta_opsets) {
        if (domain == "ai.onnx" || domain.empty()) {
            default_opset = (int)version; break;
        }
    }

    // Build new node list, expanding decompositions in place
    std::vector<operator_t*> new_nodes;
    new_nodes.reserve(nodes.size());

    for (auto* n : nodes) {
        // Check if this op needs decomposition
        const decomp_recipe* recipe = nullptr;
        for (auto& r : decomp_table) {
            if (n->op_type == r.op) { recipe = &r; break; }
        }

        if (!recipe) {
            new_nodes.push_back(n);
            continue;
        }

        // Skip if the backend actually supports this op natively
        if (solve_operator(n->op_type, default_opset, ctx->attr_pool, backend)) {
            new_nodes.push_back(n);
            continue;
        }

        if (n->outputs.empty() || !n->outputs[0]) {
            new_nodes.push_back(n);
            continue;
        }

        tensor_t* orig_output = n->outputs[0];

        // Create intermediate tensor
        std::string tmp_name = std::string(orig_output->name) + "_decomp";
        size_t len = tmp_name.size();
        char* name_buf = (char*)ctx->attr_pool.alloc(len + 1, 1);
        memcpy(name_buf, tmp_name.data(), len + 1);
        std::string_view intermed_name{name_buf, len};

        tensor_t* intermed = new (std::nothrow) tensor_t(intermed_name, orig_output->type, {});
        if (!intermed) { new_nodes.push_back(n); continue; }
        ctx->map.emplace_back(intermed_name, intermed);

        // --- First op: same inputs as original, output → intermediate ---
        operator_t* op1 = solve_operator(recipe->first_op, default_opset,
                                         ctx->attr_pool, backend);
        struct op_dummy : public operator_t { bool exec() override { return false; } };
        if (!op1) op1 = pool_new<op_dummy>(ctx->attr_pool);

        op1->ctx      = ctx;
        op1->opset    = default_opset;
        op1->op_type  = recipe->first_op;
        op1->domain   = "ai.onnx";
        op1->inputs   = n->inputs; // same inputs as original

        tensor_t** out1 = ctx->attr_pool.alloc_arr<tensor_t*>(1);
        out1[0] = intermed;
        op1->outputs = {out1, 1};
        op1->init();

        // --- Second op: input(s) = intermediate, output → original output ---
        operator_t* op2 = solve_operator(recipe->second_op, default_opset,
                                         ctx->attr_pool, backend);
        if (!op2) op2 = pool_new<op_dummy>(ctx->attr_pool);

        op2->ctx      = ctx;
        op2->opset    = default_opset;
        op2->op_type  = recipe->second_op;
        op2->domain   = "ai.onnx";

        int n_in2 = recipe->second_n_inputs;
        tensor_t** in2 = ctx->attr_pool.alloc_arr<tensor_t*>(n_in2);
        for (int k = 0; k < n_in2; ++k) in2[k] = intermed;
        op2->inputs = {in2, (size_t)n_in2};

        tensor_t** out2 = ctx->attr_pool.alloc_arr<tensor_t*>(1);
        out2[0] = orig_output;
        op2->outputs = {out2, 1};
        op2->init();

        new_nodes.push_back(op1);
        new_nodes.push_back(op2);
    }

    if (new_nodes.size() != nodes.size())
        nodes = std::move(new_nodes);
}

} // namespace nnr
