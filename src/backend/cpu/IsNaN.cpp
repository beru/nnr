#include <cmath>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct IsNaN_operator : public operator_t {

    bool init() override {
        return is_inout_size(1, 1);
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        return y->reshape_identity(x, NNR_DATA_TYPE_BOOL);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        bool_t* py = (bool_t*)y->data;

        for (size_t i = 0, l = y->ndata; i < l; ++i) {
            py[i] = std::isnan((float)px[i]);
        }
        return true;
    }

    bool exec() override {
        return typed_exec<IsNaN_operator,
            opset_t<13, bfloat16_t, float16_t, float, double>,
            opset_t<9, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }

};

} // namespace {

// @nnr-meta-op mt=no
operator_t* resolver_default_op_IsNaN(int opset, pool_t& pool) { return pool_new<IsNaN_operator>(pool); }

} // namespace nnr
