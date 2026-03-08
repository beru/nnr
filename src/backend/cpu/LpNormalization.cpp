#include <cmath>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct LpNormalization_operator : public operator_t {
    int axis;
    int p;

    bool init() override {
        if (!is_inout_size(1, 1)) return false;
        axis = attribute(attr_key_t::axis, (int32_t)-1);
        p = attribute(attr_key_t::p, (int32_t)2);
        return true;
    }

    bool reshape() override {
        return outputs[0]->reshape_identity(inputs[0]);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;

        int a = axis;
        if (a < 0) a += x->ndim;

        // Compute outer and inner sizes
        int outer = 1, norm_size = x->dims[a], inner = 1;
        for (int i = 0; i < a; ++i) outer *= x->dims[i];
        for (int i = a + 1; i < x->ndim; ++i) inner *= x->dims[i];

        for (int o = 0; o < outer; ++o) {
            for (int i = 0; i < inner; ++i) {
                // Compute norm along axis
                double norm = 0;
                for (int n = 0; n < norm_size; ++n) {
                    int idx = (o * norm_size + n) * inner + i;
                    double v = (double)px[idx];
                    if (p == 1) {
                        norm += std::abs(v);
                    } else {
                        norm += v * v;
                    }
                }
                if (p == 2) norm = std::sqrt(norm);
                if (norm == 0) norm = 1.0; // avoid division by zero

                for (int n = 0; n < norm_size; ++n) {
                    int idx = (o * norm_size + n) * inner + i;
                    py[idx] = (T)((double)px[idx] / norm);
                }
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<LpNormalization_operator,
            float16_t, float, double
        >(this, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_LpNormalization(int opset, pool_t& pool) { return pool_new<LpNormalization_operator>(pool); }

} // namespace nnr
