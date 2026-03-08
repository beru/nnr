#include <cmath>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct RMSNormalization_operator : public operator_t {
    int64_t axis;
    float epsilon;
    int stash_type;

    bool init() override {
        if (inputs.size() < 2 || outputs.empty()) return false;
        axis = attribute(attr_key_t::axis, (int64_t)-1);
        epsilon = attribute(attr_key_t::epsilon, 1e-5f);
        stash_type = attribute(attr_key_t::stash_type, 1);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        int a = (int)axis;
        if (a < 0) a += x->ndim;

        if (!y->reshape_identity(x)) return false;

        // InvStdDev output (optional)
        if (outputs.size() > 1 && outputs[1]) {
            small_vector<int> dims(x->ndim);
            for (int i = 0; i < x->ndim; ++i)
                dims[i] = (i < a) ? x->dims[i] : 1;
            outputs[1]->reshape(dims, x->type);
        }
        return true;
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        const tensor_t* scale = inputs[1];
        tensor_t* y = outputs[0];
        tensor_t* invstd_out = (outputs.size() > 1) ? outputs[1] : nullptr;

        const T* px = (const T*)x->data;
        const T* ps = (const T*)scale->data;
        T* py = (T*)y->data;

        int a = (int)axis;
        if (a < 0) a += x->ndim;

        int outer = 1, inner = 1;
        for (int i = 0; i < a; ++i) outer *= x->dims[i];
        for (int i = a; i < x->ndim; ++i) inner *= x->dims[i];

        T* pinvstd = invstd_out ? (T*)invstd_out->data : nullptr;

        for (int o = 0; o < outer; ++o) {
            const T* row = px + o * inner;
            T* out = py + o * inner;

            // Compute RMS: sqrt(mean(x^2) + epsilon)
            double sum_sq = 0;
            for (int i = 0; i < inner; ++i) {
                double v = (double)row[i];
                sum_sq += v * v;
            }
            double rms = std::sqrt(sum_sq / inner + epsilon);
            double inv_rms = 1.0 / rms;

            // Normalize and scale (no bias in RMS norm)
            for (int i = 0; i < inner; ++i) {
                out[i] = (T)((double)row[i] * inv_rms * (double)ps[i]);
            }

            if (pinvstd) pinvstd[o] = (T)inv_rms;
        }
        return true;
    }

    bool exec() override {
        data_type_t type = inputs[0]->type;
        return typed_exec<RMSNormalization_operator,
            float16_t, float, double, bfloat16_t
        >(this, type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_RMSNormalization(int opset, pool_t& pool) { return pool_new<RMSNormalization_operator>(pool); }

} // namespace nnr
