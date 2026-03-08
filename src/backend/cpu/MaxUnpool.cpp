#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct MaxUnpool_operator : public operator_t {
    small_vector<int> kernels;
    small_vector<int, MAX_NDIM * 2> pads;
    small_vector<int> strides;

    bool init() override {
        if (inputs.size() < 2 || outputs.size() != 1)
            return false;
        int64_t* ints;
        int l;

        int nk = attribute(attr_key_t::kernel_shape, ints);
        if (nk > 0) {
            kernels.resize(nk);
            for (int i = 0; i < nk; ++i)
                kernels[i] = (int)ints[i];
        }
        int spatial = kernels.size();

        pads.resize(spatial * 2);
        l = attribute(attr_key_t::pads, ints);
        for (int i = 0; i < l; ++i) pads[i] = (int)ints[i];
        for (int i = l; i < spatial * 2; ++i) pads[i] = 0;

        strides.resize(spatial);
        l = attribute(attr_key_t::strides, ints);
        for (int i = 0; i < l; ++i) strides[i] = (int)ints[i];
        for (int i = l; i < spatial; ++i) strides[i] = 1;

        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];

        // If output_shape is given as 3rd input
        if (inputs.size() > 2 && inputs[2] && inputs[2]->ndata > 0) {
            const int64_t* sz = (const int64_t*)inputs[2]->data;
            small_vector<int> dims((int)inputs[2]->ndata);
            for (int i = 0; i < (int)inputs[2]->ndata; ++i)
                dims[i] = (int)sz[i];
            return y->reshape(dims, x->type);
        }

        // Compute from kernel/stride/pads
        int ndim = x->ndim;
        int spatial = ndim - 2;
        small_vector<int> dims(ndim);
        dims[0] = x->dims[0];
        dims[1] = x->dims[1];
        for (int i = 0; i < spatial; ++i) {
            dims[i + 2] = strides[i] * (x->dims[i + 2] - 1) + kernels[i] - pads[i] - pads[i + spatial];
        }
        return y->reshape(dims, x->type);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        const tensor_t* indices = inputs[1];
        tensor_t* y = outputs[0];

        const T* px = (const T*)x->data;
        const int64_t* pi = (const int64_t*)indices->data;
        T* py = (T*)y->data;

        // Zero output
        memset(py, 0, y->ndata * sizeof(T));

        int N = x->dims[0];
        int C = x->dims[1];
        int ndim = x->ndim;
        int spatial = ndim - 2;

        // Compute "default" output spatial dims (from kernel/stride/pads)
        // The indices from MaxPool are flat indices into this default space
        small_vector<int> default_dims(spatial);
        for (int i = 0; i < spatial; ++i)
            default_dims[i] = strides[i] * (x->dims[i + 2] - 1) + kernels[i] - pads[i] - pads[i + spatial];

        // Check if we need to remap indices (output_shape differs from default)
        bool need_remap = false;
        for (int i = 0; i < spatial; ++i) {
            if (y->dims[i + 2] != default_dims[i]) {
                need_remap = true;
                break;
            }
        }

        int out_spatial = 1;
        for (int i = 2; i < y->ndim; ++i)
            out_spatial *= y->dims[i];

        int in_spatial = 1;
        for (int i = 2; i < x->ndim; ++i)
            in_spatial *= x->dims[i];

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                int base_in = (n * C + c) * in_spatial;
                int base_out = (n * C + c) * out_spatial;
                for (int i = 0; i < in_spatial; ++i) {
                    int idx = (int)pi[base_in + i];
                    if (need_remap) {
                        // Convert flat index in default spatial dims to multi-dim coords,
                        // then to flat index in actual output spatial dims
                        small_vector<int> coords(spatial);
                        int tmp = idx;
                        for (int d = spatial - 1; d >= 0; --d) {
                            coords[d] = tmp % default_dims[d];
                            tmp /= default_dims[d];
                        }
                        idx = 0;
                        int stride = 1;
                        for (int d = spatial - 1; d >= 0; --d) {
                            idx += coords[d] * stride;
                            stride *= y->dims[d + 2];
                        }
                    }
                    if (idx >= 0 && idx < out_spatial) {
                        py[base_out + idx] = px[base_in + i];
                    }
                }
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<MaxUnpool_operator,
            opset_t<9, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_MaxUnpool(int opset, pool_t& pool)
{
    return pool_new<MaxUnpool_operator>(pool);
}

} // namespace nnr
