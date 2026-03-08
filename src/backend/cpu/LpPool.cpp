#include <cmath>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct LpPool_operator : public operator_t {
    enum auto_pad_t {
        NOTSET,
        SAME_UPPER,
        SAME_LOWER,
        VALID,
    } auto_pad;
    int ceil_mode = 0;
    int p = 2;
    small_vector<int> kernels;
    small_vector<int> dilations;
    small_vector<int, MAX_NDIM * 2> pads;
    small_vector<int> strides;
    int cpads[32] = {0};

    bool init() override {
        if (!is_inout_size(1, 1)) return false;
        int64_t* ints;

        auto_pad = string2enum(attribute(attr_key_t::auto_pad, "NOTSET"), NOTSET);
        ceil_mode = attribute(attr_key_t::ceil_mode, 0);
        p = (int)attribute(attr_key_t::p, (int64_t)2);

        int kernel_shape = attribute(attr_key_t::kernel_shape, ints);
        if (kernel_shape < 0) return false;
        kernels.resize(kernel_shape);
        for (int i = 0; i < kernels.size(); ++i) kernels[i] = ints[i];

        dilations.resize(kernels.size());
        int dl = attribute(attr_key_t::dilations, ints);
        for (int i = 0; i < dl && i < dilations.size(); ++i) dilations[i] = ints[i];
        for (int i = dl; i < dilations.size(); ++i) dilations[i] = 1;

        pads.resize(kernels.size() * 2);
        int pl = attribute(attr_key_t::pads, ints);
        for (int i = 0; i < pl && i < pads.size(); ++i) pads[i] = ints[i];
        for (int i = pl; i < pads.size(); ++i) pads[i] = 0;

        strides.resize(kernels.size());
        int sl = attribute(attr_key_t::strides, ints);
        for (int i = 0; i < sl && i < strides.size(); ++i) strides[i] = ints[i];
        for (int i = sl; i < strides.size(); ++i) strides[i] = 1;

        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        int ndim_spatial = kernels.size();
        small_vector<int> dims(x->ndim);

        switch (auto_pad) {
        case NOTSET:
            memcpy(cpads, pads.data(), sizeof(int) * pads.size());
            break;
        case SAME_UPPER:
            for (int i = 0; i < ndim_spatial; ++i) {
                int ek = (kernels[i] - 1) * dilations[i] + 1;
                int pad = (int)(ceilf(x->dims[i + 2] / (float)strides[i]) - 1) * strides[i] + ek - x->dims[i + 2];
                if (pad < 0) pad = 0;
                cpads[i] = pad / 2;
                cpads[i + ndim_spatial] = pad - cpads[i];
            }
            break;
        case SAME_LOWER:
            for (int i = 0; i < ndim_spatial; ++i) {
                int ek = (kernels[i] - 1) * dilations[i] + 1;
                int pad = (int)(ceilf(x->dims[i + 2] / (float)strides[i]) - 1) * strides[i] + ek - x->dims[i + 2];
                if (pad < 0) pad = 0;
                cpads[i + ndim_spatial] = pad / 2;
                cpads[i] = pad - cpads[i + ndim_spatial];
            }
            break;
        case VALID:
            memset(cpads, 0, sizeof(int) * pads.size());
            break;
        }

        dims[0] = x->dims[0];
        dims[1] = x->dims[1];
        for (int i = 0; i < ndim_spatial; ++i) {
            int ek = (kernels[i] - 1) * dilations[i] + 1;
            if (auto_pad == SAME_UPPER || auto_pad == SAME_LOWER) {
                dims[i + 2] = (int)ceilf(x->dims[i + 2] / (float)strides[i]);
            } else if (ceil_mode) {
                dims[i + 2] = (int)ceilf((x->dims[i + 2] + cpads[i] + cpads[i + ndim_spatial] - ek) / (float)strides[i] + 1);
            } else {
                dims[i + 2] = (int)floorf((x->dims[i + 2] + cpads[i] + cpads[i + ndim_spatial] - ek) / (float)strides[i] + 1);
            }
        }
        return y->reshape(dims, x->type);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;
        int ndim_spatial = kernels.size();

        small_vector<int> k_dim(ndim_spatial);
        small_vector<int> i_dim(x->ndim);
        small_vector<int> o_dim(x->ndim);
        small_vector<int> b_dim(x->ndim);

        do {
            for (int i = 2; i < x->ndim; ++i)
                b_dim[i] = o_dim[i] * strides[i - 2] - cpads[i - 2];

            double sum = 0;
            std::fill(k_dim.begin(), k_dim.end(), 0);
            do {
                i_dim[0] = o_dim[0];
                i_dim[1] = o_dim[1];
                bool ispad = false;
                for (int i = 2; i < x->ndim; ++i) {
                    i_dim[i] = b_dim[i] + k_dim[i - 2] * dilations[i - 2];
                    if (i_dim[i] < 0 || i_dim[i] >= x->dims[i]) {
                        ispad = true;
                        break;
                    }
                }
                if (!ispad) {
                    double v = std::abs((double)px[dim_offset(i_dim, x->dim_span())]);
                    if (p == 1) {
                        sum += v;
                    } else if (p == 2) {
                        sum += v * v;
                    } else {
                        sum += std::pow(v, (double)p);
                    }
                }
            } while (dim_next(k_dim, kernels));

            if (p == 1) {
                py[dim_offset(o_dim, y->dim_span())] = (T)sum;
            } else if (p == 2) {
                py[dim_offset(o_dim, y->dim_span())] = (T)std::sqrt(sum);
            } else {
                py[dim_offset(o_dim, y->dim_span())] = (T)std::pow(sum, 1.0 / p);
            }
        } while (dim_next(o_dim, y->dim_span()));
        return true;
    }

    bool exec() override {
        return typed_exec<LpPool_operator,
            float16_t, float, double
        >(this, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_LpPool(int opset, pool_t& pool) { return pool_new<LpPool_operator>(pool); }

} // namespace nnr
