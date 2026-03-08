#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Dropout_operator : public operator_t {
    bool init() override {
        return (inputs.size() >= 1) && (outputs.size() >= 1);
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        y->reshape_identity(x);
        // mask output (all true during inference)
        if (outputs.size() > 1) {
            outputs[1]->reshape(x->dim_span(), NNR_DATA_TYPE_BOOL);
        }
        return true;
    }

    template <typename T>
    bool exec() {
        foreach_tensor<T>([](auto x){return x;});
        // fill mask with all true
        if (outputs.size() > 1) {
            tensor_t* mask = outputs[1];
            memset(mask->data, 1, mask->ndata);
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Dropout_operator,
            opset_t<13, bfloat16_t, float16_t, float, double>,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace {

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Dropout(int opset, pool_t& pool) { return pool_new<Dropout_operator>(pool); }

} // namespace nnr
