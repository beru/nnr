#include "nnr.h"
#include "util.h"
#include "arena.h"

namespace nnr {

namespace {

struct SequenceMap_operator : public operator_t {
    graph_t* body = nullptr;
    std::span<const std::string_view> body_input_names;
    std::span<const std::string_view> body_output_names;

    bool init() override {
        if (inputs.empty() || outputs.empty())
            return false;
        body = attribute_subgraph("body");
        if (!body) return false;
        attr_t* body_attr = find_attr("body");
        if (!body_attr) return false;
        body_input_names  = body_attr->subgraph_inputs;
        body_output_names = body_attr->subgraph_outputs;
        return true;
    }

    bool reshape() override {
        for (auto* out : outputs)
            if (out) out->reshape({}, NNR_DATA_TYPE_SEQUENCE);
        return true;
    }

    bool exec() override {
        const sequence_t* in_seq = tensor_get_sequence(inputs[0]);
        if (!in_seq) return false;
        int n = (int)in_seq->tensors.size();
        int n_out = (int)outputs.size();

        // Initialize output sequences (arena-backed fixed-size array)
        arena_scope_t scope(ctx->arena);
        sequence_t** out_seqs = scope.alloc_arr<sequence_t*>(n_out);
        for (int k = 0; k < n_out; ++k) out_seqs[k] = nullptr;
        for (int k = 0; k < n_out; ++k) {
            if (!outputs[k]) continue;
            sequence_t* seq = tensor_get_sequence(outputs[k]);
            if (!seq) continue;
            for (auto* t : seq->tensors) delete t;
            seq->tensors.clear();
            out_seqs[k] = seq;
        }

        for (int iter = 0; iter < n; ++iter) {
            // Set body input[0] = current element from input_sequence[0]
            if (!body_input_names.empty()) {
                tensor_t* t = ctx->search_tensor(body_input_names[0]);
                if (t && in_seq->tensors[iter])
                    t->apply(*in_seq->tensors[iter]);
            }

            // Set additional body inputs from inputs[1..]
            for (size_t j = 1; j < inputs.size() && j < body_input_names.size(); ++j) {
                const tensor_t* add = inputs[j];
                if (!add) continue;
                tensor_t* t = ctx->search_tensor(body_input_names[j]);
                if (!t) continue;
                if (add->type == NNR_DATA_TYPE_SEQUENCE) {
                    const sequence_t* add_seq = tensor_get_sequence(add);
                    if (add_seq && iter < (int)add_seq->tensors.size() && add_seq->tensors[iter])
                        t->apply(*add_seq->tensors[iter]);
                } else {
                    t->apply(*add);
                }
            }

            // Execute body
            for (auto* node : body->nodes) {
                node->reshape();
                node->exec();
            }

            // Collect body outputs and append to output sequences
            for (int k = 0; k < n_out && k < (int)body_output_names.size(); ++k) {
                if (!out_seqs[k]) continue;
                tensor_t* ot = ctx->search_tensor(body_output_names[k]);
                if (!ot) continue;
                tensor_t* elem = new (std::nothrow) tensor_t("", ot->type, ot->dim_span());
                if (elem) copy_data(elem, ot);
                out_seqs[k]->tensors.push_back(elem);
                if (out_seqs[k]->elem_type == NNR_DATA_TYPE_UNDEFINED)
                    out_seqs[k]->elem_type = ot->type;
            }
        }
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_SequenceMap(int opset, pool_t& pool)
{
    if (opset >= 17)
        return pool_new<SequenceMap_operator>(pool);
    return nullptr;
}

} // namespace nnr
