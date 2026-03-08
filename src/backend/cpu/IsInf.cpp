#include <cmath>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct IsInf_operator : public operator_t {
    int detect_negative;
    int detect_positive;

    bool init() override {
        if (!is_inout_size(1, 1)) {
            return false;
        }
        detect_negative = attribute(attr_key_t::detect_negative, 1);
        detect_positive = attribute(attr_key_t::detect_positive, 1);
        return true;
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
            double v = (double)(float)px[i];
            bool inf = std::isinf(v);
            if (inf) {
                py[i] = (detect_negative && v < 0) || (detect_positive && v > 0);
            }else {
                py[i] = false;
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<IsInf_operator,
            opset_t<20, bfloat16_t, float16_t, float, double>,
            opset_t<10, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }

};

} // namespace {

// @nnr-meta-op mt=no
operator_t* resolver_default_op_IsInf(int opset, pool_t& pool) { return pool_new<IsInf_operator>(pool); }

} // namespace nnr
