#include "nnr.h"
#include "util.h"
#include <cmath>

namespace nnr {

namespace {

struct Adam_operator : public operator_t {
    float alpha;
    float beta;
    float epsilon;
    float norm_coefficient;
    float norm_coefficient_post;

    bool init() override {
        // Inputs: R, T, X_1..X_N, G_1..G_N, V_1..V_N, H_1..H_N
        // Outputs: X_new_1..X_new_N, V_new_1..V_new_N, H_new_1..H_new_N
        int remaining = (int)inputs.size() - 2;
        if (remaining < 4 || remaining % 4 != 0)
            return false;
        int n = remaining / 4;
        if (outputs.size() != (size_t)(3 * n))
            return false;
        alpha = attribute(attr_key_t::alpha, 0.9f);
        beta = attribute(attr_key_t::beta, 0.999f);
        epsilon = attribute(attr_key_t::epsilon, 1e-2f);
        norm_coefficient = attribute(attr_key_t::norm_coefficient, 0.0f);
        norm_coefficient_post = attribute(attr_key_t::norm_coefficient_post, 0.0f);
        return true;
    }

    bool reshape() override {
        int n = ((int)inputs.size() - 2) / 4;
        for (int k = 0; k < n; ++k) {
            if (!outputs[k]->reshape_identity(inputs[2 + k]))
                return false;
            if (!outputs[n + k]->reshape_identity(inputs[2 + 2 * n + k]))
                return false;
            if (!outputs[2 * n + k]->reshape_identity(inputs[2 + 3 * n + k]))
                return false;
        }
        return true;
    }

    template <typename T>
    bool exec() {
        const T* R = (const T*)inputs[0]->data;
        const int64_t* T_step = (const int64_t*)inputs[1]->data;
        (void)T_step;

        double r = (double)R[0];

        int n = ((int)inputs.size() - 2) / 4;
        for (int k = 0; k < n; ++k) {
            const T* X = (const T*)inputs[2 + k]->data;
            const T* G = (const T*)inputs[2 + n + k]->data;
            const T* V = (const T*)inputs[2 + 2 * n + k]->data;
            const T* H = (const T*)inputs[2 + 3 * n + k]->data;
            T* X_new = (T*)outputs[k]->data;
            T* V_new = (T*)outputs[n + k]->data;
            T* H_new = (T*)outputs[2 * n + k]->data;

            size_t sz = inputs[2 + k]->ndata;
            for (size_t i = 0; i < sz; ++i) {
                double x = (double)X[i];
                double g = (double)G[i];
                double v = (double)V[i];
                double h = (double)H[i];

                double g_reg = g + (double)norm_coefficient * x;
                double v_new = (double)alpha * v + (1.0 - (double)alpha) * g_reg;
                double h_new = (double)beta * h + (1.0 - (double)beta) * g_reg * g_reg;
                double x_new = x - r * v_new / (std::sqrt(h_new) + (double)epsilon);
                if (norm_coefficient_post != 0.0f)
                    x_new *= (1.0 - (double)norm_coefficient_post);

                X_new[i] = (T)x_new;
                V_new[i] = (T)v_new;
                H_new[i] = (T)h_new;
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Adam_operator, float, double>(this, inputs[2]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Adam(int opset, pool_t& pool)
{
    return pool_new<Adam_operator>(pool);
}

} // namespace nnr
