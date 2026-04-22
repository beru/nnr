#include <cmath>
#include "nnr.h"
#include "util.h"
#include "cpu_features.h"
#include "thread_pool.h"
#ifdef NNR_ARCH_X64
#include "backend/x64/simd_math_avx512.h"
#elifdef NNR_ARCH_ARM64
#include "backend/arm/simd_math_neon.h"
#endif

namespace nnr {

namespace {

struct Gelu_operator : public operator_t {
    int approximate; // 0=none, 1=tanh

    bool init() override {
        if (!is_inout_size(1, 1)) return false;
        auto approx = attribute(attr_key_t::approximate, "none");
        approximate = (approx == "tanh") ? 1 : 0;
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

        if (approximate) {
            // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            constexpr double sqrt2pi = 0.7978845608028654; // sqrt(2/pi)
            for (size_t i = 0; i < x->ndata; ++i) {
                double v = (double)px[i];
                double t = std::tanh(sqrt2pi * (v + 0.044715 * v * v * v));
                py[i] = (T)(0.5 * v * (1.0 + t));
            }
        } else {
            // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
            constexpr double inv_sqrt2 = 0.7071067811865476; // 1/sqrt(2)
            for (size_t i = 0; i < x->ndata; ++i) {
                double v = (double)px[i];
                py[i] = (T)(0.5 * v * (1.0 + std::erf(v * inv_sqrt2)));
            }
        }
        return true;
    }

    bool exec() override {
#ifdef NNR_ARCH_X64
        if (inputs[0]->type == NNR_DATA_TYPE_FLOAT32 && !approximate) {
            gelu_avx512((const float*)inputs[0]->data,
                        (float*)outputs[0]->data, outputs[0]->ndata);
            return true;
        }
#elifdef NNR_ARCH_ARM64
        if (!approximate) {
            if (inputs[0]->type == NNR_DATA_TYPE_FLOAT32) {
                gelu_neon((const float*)inputs[0]->data,
                          (float*)outputs[0]->data, outputs[0]->ndata);
                return true;
            }
            if (inputs[0]->type == NNR_DATA_TYPE_FLOAT16 && has_neon_fp16()) {
                gelu_neon_fp16((const uint16_t*)inputs[0]->data,
                               (uint16_t*)outputs[0]->data, outputs[0]->ndata);
                return true;
            }
        }
#endif
        return typed_exec<Gelu_operator,
            float16_t, float, double, bfloat16_t
        >(this, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Gelu(int opset, pool_t& pool) { return pool_new<Gelu_operator>(pool); }

} // namespace nnr
