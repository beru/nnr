#include "nnr.h"
#include "util.h"
#include <cmath>

namespace nnr {

namespace {

struct Adagrad_operator : public operator_t {
    float decay_factor;
    float epsilon;
    float norm_coefficient;

    bool init() override {
        // Inputs: R, T, X_1..X_N, G_1..G_N, H_1..H_N
        // Outputs: X_new_1..X_new_N, H_new_1..H_new_N
        int remaining = (int)inputs.size() - 2;
        if (remaining < 3 || remaining % 3 != 0)
            return false;
        int n = remaining / 3;
        if (outputs.size() != (size_t)(2 * n))
            return false;
        decay_factor = attribute(attr_key_t::decay_factor, 0.0f);
        epsilon = attribute(attr_key_t::epsilon, 0.0f);
        norm_coefficient = attribute(attr_key_t::norm_coefficient, 0.0f);
        return true;
    }

    bool reshape() override {
        int n = ((int)inputs.size() - 2) / 3;
        for (int k = 0; k < n; ++k) {
            if (!outputs[k]->reshape_identity(inputs[2 + k]))
                return false;
            if (!outputs[n + k]->reshape_identity(inputs[2 + 2 * n + k]))
                return false;
        }
        return true;
    }

    template <typename T>
    bool exec() {
        const T* R = (const T*)inputs[0]->data;
        const int64_t* T_step = (const int64_t*)inputs[1]->data;

        float r = (float)R[0];
        int64_t t = T_step[0];
        float adjusted_r = r / (1.0f + t * decay_factor);

        int n = ((int)inputs.size() - 2) / 3;
        for (int k = 0; k < n; ++k) {
            const T* X = (const T*)inputs[2 + k]->data;
            const T* G = (const T*)inputs[2 + n + k]->data;
            const T* H = (const T*)inputs[2 + 2 * n + k]->data;
            T* X_new = (T*)outputs[k]->data;
            T* H_new = (T*)outputs[n + k]->data;

            size_t sz = inputs[2 + k]->ndata;
            for (size_t i = 0; i < sz; ++i) {
                float x = (float)X[i];
                float g = (float)G[i];
                float h = (float)H[i];
                float g_reg = g + norm_coefficient * x;
                float h_new = h + g_reg * g_reg;
                float x_new = x - adjusted_r * g_reg / (std::sqrt(h_new) + epsilon);
                H_new[i] = (T)h_new;
                X_new[i] = (T)x_new;
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Adagrad_operator, float, double>(this, inputs[2]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Adagrad(int opset, pool_t& pool)
{
    return pool_new<Adagrad_operator>(pool);
}

} // namespace nnr
