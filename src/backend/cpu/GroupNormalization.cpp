#include <cmath>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct GroupNormalization_operator : public operator_t {
    int64_t num_groups;
    float epsilon;

    bool init() override {
        if (inputs.size() < 3 || outputs.empty()) return false;
        num_groups = attribute(attr_key_t::num_groups, (int64_t)1);
        epsilon = attribute(attr_key_t::epsilon, 1e-5f);
        return true;
    }

    bool reshape() override {
        return outputs[0]->reshape_identity(inputs[0]);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        const tensor_t* scale = inputs[1];
        const tensor_t* bias = inputs[2];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        const T* ps = (const T*)scale->data;
        const T* pb = (const T*)bias->data;
        T* py = (T*)y->data;

        int N = x->dims[0];
        int C = x->dims[1];
        int G = (int)num_groups;
        int cpg = C / G; // channels per group

        // spatial size = product of dims[2:]
        int spatial = 1;
        for (int i = 2; i < x->ndim; ++i) spatial *= x->dims[i];

        int group_size = cpg * spatial;

        for (int n = 0; n < N; ++n) {
            for (int g = 0; g < G; ++g) {
                const T* group_start = px + (n * C + g * cpg) * spatial;

                // Compute mean
                double sum = 0;
                for (int i = 0; i < group_size; ++i) sum += (double)group_start[i];
                double mean = sum / group_size;

                // Compute variance
                double var = 0;
                for (int i = 0; i < group_size; ++i) {
                    double d = (double)group_start[i] - mean;
                    var += d * d;
                }
                var /= group_size;
                double invstd = 1.0 / std::sqrt(var + epsilon);

                // Normalize, scale, bias per channel
                for (int c = 0; c < cpg; ++c) {
                    int ci = g * cpg + c;
                    double s = (double)ps[ci];
                    double b = (double)pb[ci];
                    const T* src = px + (n * C + ci) * spatial;
                    T* dst = py + (n * C + ci) * spatial;
                    for (int j = 0; j < spatial; ++j) {
                        dst[j] = (T)(((double)src[j] - mean) * invstd * s + b);
                    }
                }
            }
        }
        return true;
    }

    bool exec() override {
        data_type_t type = inputs[0]->type;
        return typed_exec<GroupNormalization_operator,
            float16_t, float, double, bfloat16_t
        >(this, type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_GroupNormalization(int opset, pool_t& pool) { return pool_new<GroupNormalization_operator>(pool); }

} // namespace nnr
