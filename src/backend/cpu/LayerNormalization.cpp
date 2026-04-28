#include <cmath>
#include "nnr.h"
#include "util.h"
#include "cpu_features.h"
#include "thread_pool.h"
#ifdef NNR_ARCH_X64
#include "backend/x64/ops_x64.h"
#endif

namespace nnr {

namespace {

struct LayerNormalization_operator : public operator_t {
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

        // Mean and InvStdDev outputs (optional)
        if (outputs.size() > 1 && outputs[1]) {
            small_vector<int> mean_dims(x->ndim);
            for (int i = 0; i < x->ndim; ++i)
                mean_dims[i] = (i < a) ? x->dims[i] : 1;
            outputs[1]->reshape(mean_dims, x->type);
        }
        if (outputs.size() > 2 && outputs[2]) {
            small_vector<int> mean_dims(x->ndim);
            for (int i = 0; i < x->ndim; ++i)
                mean_dims[i] = (i < a) ? x->dims[i] : 1;
            outputs[2]->reshape(mean_dims, x->type);
        }
        return true;
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        const tensor_t* scale = inputs[1];
        const tensor_t* bias = (inputs.size() > 2 && inputs[2]) ? inputs[2] : nullptr;
        tensor_t* y = outputs[0];
        tensor_t* mean_out = (outputs.size() > 1) ? outputs[1] : nullptr;
        tensor_t* invstd_out = (outputs.size() > 2) ? outputs[2] : nullptr;

        const T* px = (const T*)x->data;
        const T* ps = (const T*)scale->data;
        const T* pb = bias ? (const T*)bias->data : nullptr;
        T* py = (T*)y->data;

        int a = (int)axis;
        if (a < 0) a += x->ndim;

        // Compute outer_size (product of dims before axis) and inner_size (product of dims from axis)
        int outer = 1, inner = 1;
        for (int i = 0; i < a; ++i) outer *= x->dims[i];
        for (int i = a; i < x->ndim; ++i) inner *= x->dims[i];

        T* pmean = mean_out ? (T*)mean_out->data : nullptr;
        T* pinvstd = invstd_out ? (T*)invstd_out->data : nullptr;

        for (int o = 0; o < outer; ++o) {
            const T* row = px + o * inner;
            T* out = py + o * inner;

            // Compute mean
            double sum = 0;
            for (int i = 0; i < inner; ++i) sum += (double)row[i];
            double mean = sum / inner;

            // Compute variance
            double var = 0;
            for (int i = 0; i < inner; ++i) {
                double d = (double)row[i] - mean;
                var += d * d;
            }
            var /= inner;
            double invstd = 1.0 / std::sqrt(var + epsilon);

            // Normalize, scale, bias
            for (int i = 0; i < inner; ++i) {
                double norm = ((double)row[i] - mean) * invstd;
                out[i] = (T)(norm * (double)ps[i] + (pb ? (double)pb[i] : 0.0));
            }

            if (pmean) pmean[o] = (T)mean;
            if (pinvstd) pinvstd[o] = (T)invstd;
        }
        return true;
    }

    bool exec() override {
#ifdef NNR_ARCH_X64
        if (inputs[0]->type == NNR_DATA_TYPE_FLOAT32) {
            const tensor_t* x = inputs[0];
            const float* px = (const float*)x->data;
            const float* ps = (const float*)inputs[1]->data;
            const float* pb = (inputs.size() > 2 && inputs[2]) ? (const float*)inputs[2]->data : nullptr;
            float* py = (float*)outputs[0]->data;

            int a = (int)axis;
            if (a < 0) a += x->ndim;
            int outer = 1, inner = 1;
            for (int i = 0; i < a; ++i) outer *= x->dims[i];
            for (int i = a; i < x->ndim; ++i) inner *= x->dims[i];

            float eps = epsilon;
            nnr::for_static(0, outer, outer > 4, [&](int o) {
                if (has_avx512())
                    layer_norm_row_avx512(px + (size_t)o * inner,
                        py + (size_t)o * inner, ps, pb, inner, eps);
                else
                    layer_norm_row_avx2  (px + (size_t)o * inner,
                        py + (size_t)o * inner, ps, pb, inner, eps);
            });
            return true;
        }
#endif
        return typed_exec<LayerNormalization_operator,
            opset_t<17, float16_t, float, double, bfloat16_t>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=static
operator_t* resolver_default_op_LayerNormalization(int opset, pool_t& pool) { return pool_new<LayerNormalization_operator>(pool); }

} // namespace nnr
