#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct ThresholdedRelu_operator : public operator_t {
    float alpha;

    bool init() override {
        if (!is_inout_size(1, 1)) {
            return false;
        }
        alpha = attribute(attr_key_t::alpha, 1.0f);
        return true;
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;
        for (size_t i = 0, l = y->ndata; i < l; ++i) {
            T xv = px[i];
            py[i] = (xv > alpha) ? xv : (T)0;
        }
        return true;
    }

    bool exec() override {
        return typed_exec<ThresholdedRelu_operator,
            opset_t<10, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace {

// @nnr-meta-op mt=no
operator_t* resolver_default_op_ThresholdedRelu(int opset, pool_t& pool)
{
    return pool_new<ThresholdedRelu_operator>(pool);
}

} // namespace nnr
