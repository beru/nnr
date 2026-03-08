#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Celu_operator : public operator_t {
    float alpha;

    bool init() override {
        if (!is_inout_size(1, 1)) {
            return false;
        }
        alpha = attribute(attr_key_t::alpha, 1.0f);
        return true;
    }

    bool exec() override {
        data_type_t type = inputs[0]->type;
        if (opset >= 12) {
            switch (type) {
            case NNR_DATA_TYPE_FLOAT32:
            {
                const tensor_t* x = inputs[0];
                tensor_t* y = outputs[0];
                const float* px = (const float*)x->data;
                float* py = (float*)y->data;
                for (size_t i = 0, l = y->ndata; i < l; ++i) {
                    float xv = (float)px[i];
                    py[i] = max(0.0f, xv) + min(0.0f, alpha * (expf(xv / alpha) - 1));
                }
                return true;
            }
            default:
                return false;
            }
        }else {
            return false;
        }
    }
};

} // namespace {

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Celu(int opset, pool_t& pool) { return pool_new<Celu_operator>(pool); }

} // namespace nnr
