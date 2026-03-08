#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct SequenceConstruct_operator : public operator_t {
    bool reshape() override
    {
        return outputs[0]->reshape({}, NNR_DATA_TYPE_SEQUENCE);
    }

    bool exec() override
    {
        sequence_t* seq = tensor_get_sequence(outputs[0]);
        if (!seq) return false;
        for (auto* t : seq->tensors) delete t;
        seq->tensors.clear();

        for (const auto* inp : inputs) {
            if (!inp) continue;
            tensor_t* copy = new (std::nothrow) tensor_t("", inp->type, inp->dim_span());
            if (copy && copy->data && inp->data) {
                copy_data(copy, inp);
            }
            seq->tensors.push_back(copy);
            if (seq->elem_type == NNR_DATA_TYPE_UNDEFINED) {
                seq->elem_type = inp->type;
            }
        }
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_SequenceConstruct(int opset, pool_t& pool)
{
    if (opset >= 11) {
        return pool_new<SequenceConstruct_operator>(pool);
    }
    return nullptr;
}

} // namespace nnr
