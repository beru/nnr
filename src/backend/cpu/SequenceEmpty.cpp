#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct SequenceEmpty_operator : public operator_t {
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
        seq->elem_type = (data_type_t)(int)attribute(attr_key_t::dtype, (int64_t)NNR_DATA_TYPE_FLOAT32);
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_SequenceEmpty(int opset, pool_t& pool)
{
    if (opset >= 11) {
        return pool_new<SequenceEmpty_operator>(pool);
    }
    return nullptr;
}

} // namespace nnr
