#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct HardSigmoid_operator : public operator_t {
    float alpha;
    float beta;

    bool init() override {
        if (!(inputs.size() > 0 && outputs.size() > 0)) {
            return false;
        }
        layout_mask = LAYOUT_ALL;
        alpha = attribute(attr_key_t::alpha, 0.2f);
        beta = attribute(attr_key_t::beta, 0.5f);
        return true;
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;

        for (size_t i = 0, l = y->ndata; i < l; ++i) {
            py[i] = max((T)0, min((T)1, (T)(alpha * px[i] + beta)));
        }
        return true;
    }

    bool exec() override {
        return typed_exec<HardSigmoid_operator,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace {

// @nnr-meta-op mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
operator_t* resolver_default_op_HardSigmoid(int opset, pool_t& pool) { return pool_new<HardSigmoid_operator>(pool); }

} // namespace nnr
