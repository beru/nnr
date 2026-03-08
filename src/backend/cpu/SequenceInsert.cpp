#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct SequenceInsert_operator : public operator_t {
    bool reshape() override
    {
        return outputs[0]->reshape({}, NNR_DATA_TYPE_SEQUENCE);
    }

    bool exec() override
    {
        // inputs: sequence, tensor, [position]
        const sequence_t* in_seq = tensor_get_sequence(inputs[0]);
        if (!in_seq) return false;
        const tensor_t* tensor = inputs[1];
        if (!tensor) return false;

        sequence_t* out_seq = tensor_get_sequence(outputs[0]);
        if (!out_seq) return false;

        // Clone input sequence into output
        for (auto* t : out_seq->tensors) delete t;
        out_seq->tensors.clear();
        out_seq->elem_type = in_seq->elem_type;
        for (const auto* t : in_seq->tensors) {
            tensor_t* copy = new (std::nothrow) tensor_t("", t->type, t->dim_span());
            if (copy && copy->data && t->data) {
                copy_data(copy, t);
            }
            out_seq->tensors.push_back(copy);
        }

        // Determine insertion position
        int64_t pos = (int64_t)out_seq->tensors.size(); // default: append
        if (inputs.size() >= 3 && inputs[2] && inputs[2]->data) {
            pos = *(const int64_t*)inputs[2]->data;
            if (pos < 0) pos += (int64_t)out_seq->tensors.size() + 1;
        }
        pos = std::max<int64_t>(0, std::min<int64_t>(pos, (int64_t)out_seq->tensors.size()));

        // Copy the new tensor and insert
        tensor_t* elem = new (std::nothrow) tensor_t("", tensor->type, tensor->dim_span());
        if (elem && elem->data && tensor->data) {
            copy_data(elem, tensor);
        }
        out_seq->tensors.insert(out_seq->tensors.begin() + pos, elem);
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_SequenceInsert(int opset, pool_t& pool)
{
    if (opset >= 11) {
        return pool_new<SequenceInsert_operator>(pool);
    }
    return nullptr;
}

} // namespace nnr
