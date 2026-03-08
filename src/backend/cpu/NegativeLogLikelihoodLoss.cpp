#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct NegativeLogLikelihoodLoss_operator : public operator_t {
    int64_t ignore_index = INT64_MIN;
    bool has_ignore_index = false;
    std::string_view reduction;

    bool init() override {
        if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() < 1)
            return false;
        auto* attr = find_attr("ignore_index");
        if (attr) {
            ignore_index = attr->i;
            has_ignore_index = true;
        }
        reduction = attribute(attr_key_t::reduction, "mean");
        return true;
    }

    bool reshape() override {
        const tensor_t* input = inputs[0];
        const tensor_t* target = inputs[1];
        tensor_t* loss = outputs[0];

        if (reduction == "none") {
            // output shape = target shape = N x d1 x d2 ... dk
            loss->reshape(target->dim_span(), input->type);
        } else {
            // scalar output
            small_vector<int> dims;
            loss->reshape(dims, input->type);
        }
        return true;
    }

    template <typename T>
    bool exec() {
        const tensor_t* input = inputs[0];
        const tensor_t* target = inputs[1];
        const tensor_t* weight = (inputs.size() > 2) ? inputs[2] : nullptr;
        tensor_t* loss = outputs[0];

        const T* px = (const T*)input->data;
        const int64_t* pt = (const int64_t*)target->data;
        const T* pw = weight ? (const T*)weight->data : nullptr;
        T* py = (T*)loss->data;

        int N = input->dims[0];
        int C = input->dims[1];

        // spatial dims: d1*d2*...*dk
        int D = 1;
        for (int i = 2; i < input->ndim; ++i)
            D *= input->dims[i];

        if (reduction == "none") {
            // output shape = N x d1 x ... x dk
            for (int n = 0; n < N; ++n) {
                for (int d = 0; d < D; ++d) {
                    int out_idx = n * D + d;
                    int64_t c = pt[out_idx];
                    if (has_ignore_index && c == ignore_index) {
                        py[out_idx] = T(0);
                    } else {
                        T w = pw ? pw[c] : T(1);
                        // input index: n*C*D + c*D + d
                        py[out_idx] = -px[n * C * D + c * D + d] * w;
                    }
                }
            }
        } else {
            // sum or mean
            double total_loss = 0;
            double total_weight = 0;
            for (int n = 0; n < N; ++n) {
                for (int d = 0; d < D; ++d) {
                    int t_idx = n * D + d;
                    int64_t c = pt[t_idx];
                    if (has_ignore_index && c == ignore_index)
                        continue;
                    double w = pw ? (double)pw[c] : 1.0;
                    total_loss += -(double)px[n * C * D + c * D + d] * w;
                    total_weight += w;
                }
            }
            if (reduction == "mean" && total_weight > 0) {
                py[0] = T(total_loss / total_weight);
            } else {
                py[0] = T(total_loss);
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<NegativeLogLikelihoodLoss_operator,
            opset_t<13, float16_t, float, double, bfloat16_t>,
            opset_t<12, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_NegativeLogLikelihoodLoss(int opset, pool_t& pool)
{
    return pool_new<NegativeLogLikelihoodLoss_operator>(pool);
}

} // namespace nnr
