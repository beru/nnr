#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct CumProd_operator : public operator_t {
    int exclusive;
    int reverse;

    bool init() override {
        if (!is_inout_size(2, 1)) {
            return false;
        }
        exclusive = attribute(attr_key_t::exclusive, (int32_t)0);
        reverse = attribute(attr_key_t::reverse, (int32_t)0);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        return y->reshape_identity(x);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;
        const int ndim = x->ndim;

        int caxis;
        if (inputs[1]->type == NNR_DATA_TYPE_INT32) {
            caxis = *(const int32_t*)inputs[1]->data;
        } else {
            caxis = static_cast<int>(*(const int64_t*)inputs[1]->data);
        }
        if (caxis < 0) caxis += ndim;

        int outer = 1;
        for (int i = 0; i < caxis; ++i) outer *= x->dims[i];
        int axis_dim = x->dims[caxis];
        int inner = 1;
        for (int i = caxis + 1; i < ndim; ++i) inner *= x->dims[i];

        for (int o = 0; o < outer; ++o) {
            for (int k = 0; k < inner; ++k) {
                int base = o * axis_dim * inner + k;
                if (!reverse) {
                    T prod = T(1);
                    for (int a = 0; a < axis_dim; ++a) {
                        int idx = base + a * inner;
                        if (exclusive) {
                            py[idx] = prod;
                            prod *= px[idx];
                        } else {
                            prod *= px[idx];
                            py[idx] = prod;
                        }
                    }
                } else {
                    T prod = T(1);
                    for (int a = axis_dim - 1; a >= 0; --a) {
                        int idx = base + a * inner;
                        if (exclusive) {
                            py[idx] = prod;
                            prod *= px[idx];
                        } else {
                            prod *= px[idx];
                            py[idx] = prod;
                        }
                    }
                }
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<CumProd_operator,
            opset_t<11, int32_t, int64_t,
                uint32_t, uint64_t,
                float16_t, float, double, bfloat16_t>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_CumProd(int opset, pool_t& pool)
{
    return pool_new<CumProd_operator>(pool);
}

} // namespace nnr
