#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct SequenceErase_operator : public operator_t {
    bool reshape() override
    {
        return outputs[0]->reshape({}, NNR_DATA_TYPE_SEQUENCE);
    }

    bool exec() override
    {
        const sequence_t* in_seq = tensor_get_sequence(inputs[0]);
        if (!in_seq) return false;

        sequence_t* out_seq = tensor_get_sequence(outputs[0]);
        if (!out_seq) return false;

        // Clone input sequence
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

        // Determine erase position (default: last)
        int64_t pos = (int64_t)out_seq->tensors.size() - 1;
        if (inputs.size() >= 2 && inputs[1] && inputs[1]->data) {
            pos = *(const int64_t*)inputs[1]->data;
            if (pos < 0) pos += (int64_t)out_seq->tensors.size();
        }
        if (pos < 0 || pos >= (int64_t)out_seq->tensors.size()) return false;

        delete out_seq->tensors[pos];
        out_seq->tensors.erase(out_seq->tensors.begin() + pos);
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_SequenceErase(int opset, pool_t& pool)
{
    if (opset >= 11) {
        return pool_new<SequenceErase_operator>(pool);
    }
    return nullptr;
}

} // namespace nnr
