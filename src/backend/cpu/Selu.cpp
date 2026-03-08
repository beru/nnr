#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Selu_operator : public operator_t {
    float alpha;
    float gamma;

    bool init() override {
        if (!is_inout_size(1, 1)) {
            return false;
        }
        layout_mask = LAYOUT_ALL;
        alpha = attribute(attr_key_t::alpha, 1.67326319217681884765625f);
        gamma = attribute(attr_key_t::gamma, 1.05070102214813232421875f);
        return true;
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;

        for (size_t i = 0, l = y->ndata; i < l; ++i) {
            if (px[i] > 0) {
                py[i] = gamma * px[i];
            }else {
                py[i] = gamma * (alpha * exp(px[i]) - alpha);
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Selu_operator,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }

};

} // namespace {

// @nnr-meta-op mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
operator_t* resolver_default_op_Selu(int opset, pool_t& pool) { return pool_new<Selu_operator>(pool); }

} // namespace nnr
