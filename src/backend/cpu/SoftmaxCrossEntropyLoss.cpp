#include <cmath>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct SoftmaxCrossEntropyLoss_operator : public operator_t {
    int64_t ignore_index = INT64_MIN;
    bool has_ignore_index = false;
    std::string_view reduction;

    bool init() override {
        if (inputs.size() < 2 || outputs.empty())
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
        const tensor_t* scores = inputs[0];
        const tensor_t* labels = inputs[1];
        tensor_t* loss = outputs[0];

        if (reduction == "none") {
            loss->reshape(labels->dim_span(), scores->type);
        } else {
            small_vector<int> dims;
            loss->reshape(dims, scores->type);
        }

        if (outputs.size() > 1 && outputs[1]) {
            outputs[1]->reshape(scores->dim_span(), scores->type);
        }
        return true;
    }

    template <typename T>
    bool exec() {
        const tensor_t* scores = inputs[0];
        const tensor_t* labels = inputs[1];
        const tensor_t* weight = (inputs.size() > 2 && inputs[2] && inputs[2]->ndata > 0) ? inputs[2] : nullptr;
        tensor_t* loss = outputs[0];
        tensor_t* log_prob_out = (outputs.size() > 1 && outputs[1]) ? outputs[1] : nullptr;

        const T* px = (const T*)scores->data;
        const int64_t* pt = (const int64_t*)labels->data;
        const T* pw = weight ? (const T*)weight->data : nullptr;
        T* py = (T*)loss->data;
        T* plp = log_prob_out ? (T*)log_prob_out->data : nullptr;

        int N, C, D;
        if (scores->ndim == 1) {
            N = 1; C = scores->dims[0]; D = 1;
        } else {
            N = scores->dims[0];
            C = scores->dims[1];
            D = 1;
            for (int i = 2; i < scores->ndim; ++i)
                D *= scores->dims[i];
        }

        arena_scope_t scope(ctx->arena);
        double* log_probs = scope.alloc_arr<double>(C);

        if (reduction == "none") {
            for (int n = 0; n < N; ++n) {
                for (int d = 0; d < D; ++d) {
                    double max_val = -1e300;
                    for (int c = 0; c < C; ++c) {
                        double v = (double)px[n * C * D + c * D + d];
                        if (v > max_val) max_val = v;
                    }
                    double sum_exp = 0;
                    for (int c = 0; c < C; ++c)
                        sum_exp += std::exp((double)px[n * C * D + c * D + d] - max_val);
                    double log_sum = std::log(sum_exp) + max_val;
                    for (int c = 0; c < C; ++c) {
                        log_probs[c] = (double)px[n * C * D + c * D + d] - log_sum;
                        if (plp) plp[n * C * D + c * D + d] = (T)log_probs[c];
                    }

                    int out_idx = n * D + d;
                    int64_t target = pt[out_idx];
                    if (has_ignore_index && target == ignore_index) {
                        py[out_idx] = T(0);
                    } else {
                        double w = pw ? (double)pw[target] : 1.0;
                        py[out_idx] = (T)(-log_probs[target] * w);
                    }
                }
            }
        } else {
            double total_loss = 0;
            double total_weight = 0;
            for (int n = 0; n < N; ++n) {
                for (int d = 0; d < D; ++d) {
                    double max_val = -1e300;
                    for (int c = 0; c < C; ++c) {
                        double v = (double)px[n * C * D + c * D + d];
                        if (v > max_val) max_val = v;
                    }
                    double sum_exp = 0;
                    for (int c = 0; c < C; ++c)
                        sum_exp += std::exp((double)px[n * C * D + c * D + d] - max_val);
                    double log_sum = std::log(sum_exp) + max_val;
                    for (int c = 0; c < C; ++c) {
                        log_probs[c] = (double)px[n * C * D + c * D + d] - log_sum;
                        if (plp) plp[n * C * D + c * D + d] = (T)log_probs[c];
                    }

                    int t_idx = n * D + d;
                    int64_t target = pt[t_idx];
                    if (has_ignore_index && target == ignore_index)
                        continue;
                    double w = pw ? (double)pw[target] : 1.0;
                    total_loss += -log_probs[target] * w;
                    total_weight += w;
                }
            }
            if (reduction == "mean" && total_weight > 0) {
                py[0] = (T)(total_loss / total_weight);
            } else {
                py[0] = (T)total_loss;
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<SoftmaxCrossEntropyLoss_operator,
            float16_t, float, double, bfloat16_t
        >(this, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no workspace=yes
operator_t* resolver_default_op_SoftmaxCrossEntropyLoss(int opset, pool_t& pool)
{
    return pool_new<SoftmaxCrossEntropyLoss_operator>(pool);
}

} // namespace nnr
