#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

// OptionalGetElement(optional) -> value
// Input can be: plain tensor, sequence, or optional<tensor/sequence>
// Output is the unwrapped value.
struct OptionalGetElement_operator : public operator_t {
    bool reshape() override
    {
        tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        if (!x || x->type == NNR_DATA_TYPE_UNDEFINED) return true;
        if (x->type == NNR_DATA_TYPE_SEQUENCE) {
            return y->reshape({}, NNR_DATA_TYPE_SEQUENCE);
        }
        return y->reshape_identity(x);
    }

    bool exec() override
    {
        tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        // Empty optional (no element present): produce an empty output.
        if (!x || x->type == NNR_DATA_TYPE_UNDEFINED) return true;
        if (x->type == NNR_DATA_TYPE_SEQUENCE) {
            const sequence_t* src = tensor_get_sequence(x);
            sequence_t* dst = tensor_get_sequence(y);
            if (!src || !dst) return false;
            for (auto* t : dst->tensors) delete t;
            dst->tensors.clear();
            dst->elem_type = src->elem_type;
            for (const auto* t : src->tensors) {
                tensor_t* copy = new (std::nothrow) tensor_t("", t->type, t->dim_span());
                if (copy && copy->data && t->data) copy_data(copy, t);
                dst->tensors.push_back(copy);
            }
            return true;
        }
        return y->apply(*x);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_OptionalGetElement(int opset, pool_t& pool)
{
    if (opset >= 15) {
        return pool_new<OptionalGetElement_operator>(pool);
    }
    return nullptr;
}

} // namespace nnr
