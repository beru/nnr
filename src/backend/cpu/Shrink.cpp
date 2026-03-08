#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Shrink_operator : public operator_t {
    float bias;
    float lambd;

    bool init() override {
        if (!is_inout_size(1, 1)) {
            return false;
        }
        bias = attribute(attr_key_t::bias, 0.0f);
        lambd = attribute(attr_key_t::lambd, 0.5f);
        return true;
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;

        for (size_t i = 0, l = y->ndata; i < l; ++i) {
            if (px[i] < -lambd) {
                py[i] = px[i] + (T)bias;
            }else if (px[i] > lambd) {
                py[i] = px[i] - (T)bias;
            }else {
                py[i] = 0;
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Shrink_operator,
            opset_t<9, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }

};

} // namespace {

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Shrink(int opset, pool_t& pool) { return pool_new<Shrink_operator>(pool); }

} // namespace nnr
