#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct SequenceAt_operator : public operator_t {
    bool reshape() override { return true; }

    bool exec() override
    {
        const sequence_t* seq = tensor_get_sequence(inputs[0]);
        if (!seq) return false;
        if (!inputs[1] || !inputs[1]->data) return false;

        int64_t idx = *(const int64_t*)inputs[1]->data;
        if (idx < 0) idx += (int64_t)seq->tensors.size();
        if (idx < 0 || idx >= (int64_t)seq->tensors.size()) return false;

        const tensor_t* src = seq->tensors[idx];
        if (!src) return false;
        return outputs[0]->apply(*src);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_SequenceAt(int opset, pool_t& pool)
{
    if (opset >= 11) {
        return pool_new<SequenceAt_operator>(pool);
    }
    return nullptr;
}

} // namespace nnr
