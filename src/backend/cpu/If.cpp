#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct If_operator : public operator_t {
    graph_t* else_branch = nullptr;
    graph_t* then_branch = nullptr;
    std::span<const std::string_view> then_output_names;
    std::span<const std::string_view> else_output_names;

    bool init() override {
        if (!(inputs.size() == 1 && outputs.size() >= 1))
            return false;
        then_branch = attribute_subgraph("then_branch");
        else_branch = attribute_subgraph("else_branch");
        if (!then_branch || !else_branch)
            return false;
        // Retrieve output name lists stored by the ONNX loader
        attr_t* then_attr = find_attr("then_branch");
        attr_t* else_attr = find_attr("else_branch");
        if (then_attr) then_output_names = then_attr->subgraph_outputs;
        if (else_attr) else_output_names = else_attr->subgraph_outputs;
        return true;
    }

    bool reshape() override {
        if (!then_branch || !else_branch) return false;
        if (!inputs[0] || !inputs[0]->data)
            return false;
        const uint8_t* px = (const uint8_t*)inputs[0]->data;
        graph_t* g = px[0] ? then_branch : else_branch;
        auto& names = px[0] ? then_output_names : else_output_names;
        for (auto* n : g->nodes)
            n->reshape();
        for (size_t i = 0; i < std::min(names.size(), (size_t)outputs.size()); ++i) {
            tensor_t* src = ctx->search_tensor(names[i]);
            if (src && outputs[i])
                outputs[i]->reshape_identity(src);
        }
        return true;
    }

    bool exec() override {
        if (!then_branch || !else_branch) return false;
        if (!inputs[0] || !inputs[0]->data)
            return false;
        const uint8_t* px = (const uint8_t*)inputs[0]->data;
        graph_t* g = px[0] ? then_branch : else_branch;
        auto& names = px[0] ? then_output_names : else_output_names;
        for (auto* n : g->nodes) {
            if (!n->reshape()) continue;
            // Guard against ops dereferencing nulls when an upstream branch
            // node failed reshape — same pattern as Loop's body iteration.
            bool safe = true;
            for (auto* t : n->inputs)
                if (t && (!t->data || t->type == NNR_DATA_TYPE_UNDEFINED)) { safe = false; break; }
            for (auto* t : n->outputs)
                if (t && !t->data && t->type != NNR_DATA_TYPE_UNDEFINED) { safe = false; break; }
            if (safe) n->exec();
        }
        for (size_t i = 0; i < std::min(names.size(), (size_t)outputs.size()); ++i) {
            tensor_t* src = ctx->search_tensor(names[i]);
            if (src && outputs[i] && src->data) {
                if (!outputs[i]->apply(*src)) return false;
            }
        }
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_If(int opset, pool_t& pool) { return pool_new<If_operator>(pool); }

} // namespace nnr
