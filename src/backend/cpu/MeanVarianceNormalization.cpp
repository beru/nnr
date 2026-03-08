#include "nnr.h"
#include "util.h"
#include <cmath>

namespace nnr {

namespace {

struct MeanVarianceNormalization_operator : public operator_t {
    small_vector<int> axes;
    int naxes = 0;

    bool init() override {
        if (!is_inout_size(1, 1))
            return false;
        int64_t* ints;
        naxes = attribute(attr_key_t::axes, ints);
        if (naxes > 0) {
            axes.resize(naxes);
            for (int i = 0; i < naxes; ++i)
                axes[i] = (int)ints[i];
        } else {
            // Default axes = {0, 2, 3}
            axes.resize(3);
            axes[0] = 0; axes[1] = 2; axes[2] = 3;
            naxes = 3;
        }
        return true;
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;

        int total = (int)x->ndata;
        int ndim = x->ndim;
        const double eps = 1e-9;

        bool is_reduce[MAX_NDIM] = {};
        for (int i = 0; i < naxes; ++i) {
            int ax = axes[i];
            if (ax < 0) ax += ndim;
            is_reduce[ax] = true;
        }

        int reduce_size = 1;
        for (int i = 0; i < ndim; ++i) {
            if (is_reduce[i])
                reduce_size *= x->dims[i];
        }

        small_vector<int> idx(ndim);

        for (int flat = 0; flat < total; ++flat) {
            x->offset_to_indices(flat, idx);

            bool is_first = true;
            for (int d = 0; d < ndim; ++d) {
                if (is_reduce[d] && idx[d] != 0) {
                    is_first = false;
                    break;
                }
            }
            if (!is_first) continue;

            small_vector<int> iter(ndim);
            for (int d = 0; d < ndim; ++d)
                iter[d] = is_reduce[d] ? 0 : idx[d];

            double sum = 0;
            for (int r = 0; r < reduce_size; ++r) {
                sum += (double)px[x->indices_to_offset(iter)];
                for (int d = ndim - 1; d >= 0; --d) {
                    if (!is_reduce[d]) continue;
                    iter[d]++;
                    if (iter[d] < x->dims[d]) break;
                    iter[d] = 0;
                }
            }
            double mean = sum / reduce_size;

            double var_sum = 0;
            for (int d = 0; d < ndim; ++d)
                iter[d] = is_reduce[d] ? 0 : idx[d];
            for (int r = 0; r < reduce_size; ++r) {
                double diff = (double)px[x->indices_to_offset(iter)] - mean;
                var_sum += diff * diff;
                for (int d = ndim - 1; d >= 0; --d) {
                    if (!is_reduce[d]) continue;
                    iter[d]++;
                    if (iter[d] < x->dims[d]) break;
                    iter[d] = 0;
                }
            }
            double variance = var_sum / reduce_size;
            double inv_std = 1.0 / std::sqrt(variance + eps);

            for (int d = 0; d < ndim; ++d)
                iter[d] = is_reduce[d] ? 0 : idx[d];
            for (int r = 0; r < reduce_size; ++r) {
                int off = x->indices_to_offset(iter);
                py[off] = (T)(((double)px[off] - mean) * inv_std);
                for (int d = ndim - 1; d >= 0; --d) {
                    if (!is_reduce[d]) continue;
                    iter[d]++;
                    if (iter[d] < x->dims[d]) break;
                    iter[d] = 0;
                }
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<MeanVarianceNormalization_operator,
            opset_t<13, float16_t, float, double, bfloat16_t>,
            opset_t<9, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_MeanVarianceNormalization(int opset, pool_t& pool)
{
    return pool_new<MeanVarianceNormalization_operator>(pool);
}

} // namespace nnr
