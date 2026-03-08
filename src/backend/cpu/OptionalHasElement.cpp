#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

// OptionalHasElement([optional]) -> bool scalar
// Returns true if the optional has a value, false if empty (UNDEFINED type).
struct OptionalHasElement_operator : public operator_t {
    bool reshape() override
    {
        return outputs[0]->reshape({}, NNR_DATA_TYPE_BOOL);
    }

    bool exec() override
    {
        if (!outputs[0]->data) return false;
        bool has_value = false;
        if (!inputs.empty() && inputs[0]) {
            const tensor_t* x = inputs[0];
            if (x->type == NNR_DATA_TYPE_SEQUENCE) {
                has_value = (x->data != nullptr);
            } else if (x->type != NNR_DATA_TYPE_UNDEFINED && x->data != nullptr) {
                has_value = true;
            }
        }
        *(bool*)outputs[0]->data = has_value;
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_OptionalHasElement(int opset, pool_t& pool)
{
    if (opset >= 15) {
        return pool_new<OptionalHasElement_operator>(pool);
    }
    return nullptr;
}

} // namespace nnr
