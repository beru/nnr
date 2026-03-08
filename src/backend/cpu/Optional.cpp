#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

// Optional(value?) -> optional
// If no input (empty optional): output is UNDEFINED
// If input tensor: output = input (identity; tensor IS the optional value)
// If input sequence: output = input sequence
struct Optional_operator : public operator_t {
    bool reshape() override
    {
        if (inputs.empty() || !inputs[0] || inputs[0]->type == NNR_DATA_TYPE_UNDEFINED) {
            // Empty optional — output stays UNDEFINED
            return true;
        }
        // Pass-through shape
        tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        if (x->type == NNR_DATA_TYPE_SEQUENCE) {
            return y->reshape({}, NNR_DATA_TYPE_SEQUENCE);
        }
        return y->reshape_identity(x);
    }

    bool exec() override
    {
        if (inputs.empty() || !inputs[0] || inputs[0]->type == NNR_DATA_TYPE_UNDEFINED) {
            // Empty optional: leave output UNDEFINED
            return true;
        }
        tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        if (x->type == NNR_DATA_TYPE_SEQUENCE) {
            // Copy sequence
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
        y->apply(*x);
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Optional(int opset, pool_t& pool)
{
    if (opset >= 15) {
        return pool_new<Optional_operator>(pool);
    }
    return nullptr;
}

} // namespace nnr
