#include "nnr.h"
#include "util.h"
#include <cmath>

namespace nnr {

namespace {

struct Momentum_operator : public operator_t {
    float alpha;
    float beta;
    float norm_coefficient;
    bool nesterov;

    bool init() override {
        // Inputs: R, T, X_1..X_N, G_1..G_N, V_1..V_N
        // Outputs: X_new_1..X_new_N, V_new_1..V_new_N
        int remaining = (int)inputs.size() - 2;
        if (remaining < 3 || remaining % 3 != 0)
            return false;
        int n = remaining / 3;
        if (outputs.size() != (size_t)(2 * n))
            return false;
        alpha = attribute(attr_key_t::alpha, 0.0f);
        beta = attribute(attr_key_t::beta, 0.0f);
        norm_coefficient = attribute(attr_key_t::norm_coefficient, 0.0f);
        auto mode = attribute(attr_key_t::mode, "standard");
        nesterov = (mode == std::string_view("nesterov"));
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
        (void)T_step;

        float r = (float)R[0];

        int n = ((int)inputs.size() - 2) / 3;
        for (int k = 0; k < n; ++k) {
            const T* X = (const T*)inputs[2 + k]->data;
            const T* G = (const T*)inputs[2 + n + k]->data;
            const T* V = (const T*)inputs[2 + 2 * n + k]->data;
            T* X_new = (T*)outputs[k]->data;
            T* V_new = (T*)outputs[n + k]->data;

            size_t sz = inputs[2 + k]->ndata;
            for (size_t i = 0; i < sz; ++i) {
                float x = (float)X[i];
                float g = (float)G[i];
                float v = (float)V[i];

                float g_reg = g + norm_coefficient * x;
                float v_new = alpha * v + g_reg;
                float x_new;
                if (nesterov) {
                    x_new = x - r * (g_reg + alpha * v_new);
                } else {
                    x_new = x - r * v_new;
                }

                X_new[i] = (T)x_new;
                V_new[i] = (T)v_new;
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Momentum_operator, float, double>(this, inputs[2]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Momentum(int opset, pool_t& pool)
{
    return pool_new<Momentum_operator>(pool);
}

} // namespace nnr
