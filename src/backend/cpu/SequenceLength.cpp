#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct SequenceLength_operator : public operator_t {
    bool reshape() override
    {
        return outputs[0]->reshape({}, NNR_DATA_TYPE_INT64);
    }

    bool exec() override
    {
        const sequence_t* seq = tensor_get_sequence(inputs[0]);
        if (!seq) return false;
        if (!outputs[0]->data) return false;
        *(int64_t*)outputs[0]->data = (int64_t)seq->tensors.size();
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_SequenceLength(int opset, pool_t& pool)
{
    if (opset >= 11) {
        return pool_new<SequenceLength_operator>(pool);
    }
    return nullptr;
}

} // namespace nnr
