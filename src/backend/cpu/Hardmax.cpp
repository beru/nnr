#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Hardmax_operator : public operator_t {
    int axis;

    bool init() override {
        if (!is_inout_size(1, 1)) {
            return false;
        }
        if (opset >= 13) {
            axis = attribute(attr_key_t::axis, (int32_t)-1);
        } else {
            axis = attribute(attr_key_t::axis, (int32_t)1);
        }
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

        int caxis = axis;
        if (caxis < 0) caxis += ndim;

        // Zero output
        for (size_t i = 0; i < y->ndata; ++i) {
            py[i] = T{};
        }

        if (opset >= 13) {
            // Hardmax along a single axis
            int outer = 1;
            for (int i = 0; i < caxis; ++i) outer *= x->dims[i];
            int axis_dim = x->dims[caxis];
            int inner = 1;
            for (int i = caxis + 1; i < ndim; ++i) inner *= x->dims[i];

            for (int o = 0; o < outer; ++o) {
                for (int k = 0; k < inner; ++k) {
                    int base = o * axis_dim * inner + k;
                    int max_idx = 0;
                    T max_val = px[base];
                    for (int a = 1; a < axis_dim; ++a) {
                        T v = px[base + a * inner];
                        if ((float)v > (float)max_val) {
                            max_val = v;
                            max_idx = a;
                        }
                    }
                    py[base + max_idx * inner] = T(1);
                }
            }
        } else {
            // opset < 13: flatten into 2D [outer, inner] around axis
            int outer = 1;
            for (int i = 0; i < caxis; ++i) outer *= x->dims[i];
            int inner = 1;
            for (int i = caxis; i < ndim; ++i) inner *= x->dims[i];

            for (int o = 0; o < outer; ++o) {
                int base = o * inner;
                int max_idx = 0;
                T max_val = px[base];
                for (int k = 1; k < inner; ++k) {
                    T v = px[base + k];
                    if ((float)v > (float)max_val) {
                        max_val = v;
                        max_idx = k;
                    }
                }
                py[base + max_idx] = T(1);
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Hardmax_operator,
            opset_t<13, float16_t, float, double, bfloat16_t>,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Hardmax(int opset, pool_t& pool)
{
    return pool_new<Hardmax_operator>(pool);
}

} // namespace nnr
