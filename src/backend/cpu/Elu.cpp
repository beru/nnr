#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Elu_operator : public operator_t {
    float alpha;

    bool init() override {
        if (!is_inout_size(1, 1)) {
            return false;
        }
        layout_mask = LAYOUT_ALL;
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
            T v = px[i];
            if (v < 0) {
                v = (exp(v) - 1) * alpha;
            }
            py[i] = v;
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Elu_operator,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }

};

} // namespace {

// @nnr-meta-op mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
operator_t* resolver_default_op_Elu(int opset, pool_t& pool) { return pool_new<Elu_operator>(pool); }

} // namespace nnr
