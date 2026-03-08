#include "nnr.h"
#include "util.h"
#include <cstring>
#include <algorithm>

namespace nnr {

namespace {

struct CenterCropPad_operator : public operator_t {
    small_vector<int> axes;

    bool init() override {
        if (inputs.size() != 2 || outputs.empty()) return false;
        int64_t* axes_data = nullptr;
        int n = attribute(attr_key_t::axes, axes_data);
        for (int i = 0; i < n; ++i)
            axes.push_back((int)axes_data[i]);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        const tensor_t* shape_tensor = inputs[1];
        tensor_t* y = outputs[0];

        // shape_tensor contains the target shape for the specified axes
        const int64_t* target = (const int64_t*)shape_tensor->data;
        int ntarget = (int)shape_tensor->ndata;

        small_vector<int> dims(x->ndim);
        for (int i = 0; i < x->ndim; ++i)
            dims[i] = x->dims[i];

        if (axes.empty()) {
            // Default: all axes
            for (int i = 0; i < ntarget && i < x->ndim; ++i)
                dims[i] = (int)target[i];
        } else {
            for (int i = 0; i < (int)axes.size() && i < ntarget; ++i) {
                int a = axes[i];
                if (a < 0) a += x->ndim;
                if (a >= 0 && a < x->ndim)
                    dims[a] = (int)target[i];
            }
        }
        return y->reshape(dims, x->type);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        const tensor_t* shape_tensor = inputs[1];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;

        // Zero-fill output
        memset(py, 0, y->ndata * sizeof(T));

        // For each axis, compute the crop offset in input and pad offset in output
        small_vector<int> src_offset(x->ndim);
        small_vector<int> dst_offset(x->ndim);
        small_vector<int> copy_size(x->ndim);

        for (int i = 0; i < x->ndim; ++i) {
            int in_dim = x->dims[i];
            int out_dim = y->dims[i];
            if (in_dim > out_dim) {
                // Crop: center-crop input
                src_offset[i] = (in_dim - out_dim) / 2;
                dst_offset[i] = 0;
                copy_size[i] = out_dim;
            } else {
                // Pad: center-pad output
                src_offset[i] = 0;
                dst_offset[i] = (out_dim - in_dim) / 2;
                copy_size[i] = in_dim;
            }
        }

        // Copy region using nested iteration
        int ndim = x->ndim;
        small_vector<int> idx(ndim);
        for (int i = 0; i < ndim; ++i) idx[i] = 0;

        size_t total = 1;
        for (int i = 0; i < ndim; ++i) total *= copy_size[i];

        for (size_t n = 0; n < total; ++n) {
            // Compute source and dest linear index
            size_t si = 0, di = 0;
            for (int d = 0; d < ndim; ++d) {
                si = si * x->dims[d] + (idx[d] + src_offset[d]);
                di = di * y->dims[d] + (idx[d] + dst_offset[d]);
            }
            py[di] = px[si];

            // Increment idx
            for (int d = ndim - 1; d >= 0; --d) {
                if (++idx[d] < copy_size[d]) break;
                idx[d] = 0;
            }
        }
        return true;
    }

    bool exec() override {
        if (inputs[0]->type == NNR_DATA_TYPE_STRING) {
            return exec<std::string>();
        }
        return typed_exec<CenterCropPad_operator,
            int8_t, int16_t, int32_t, int64_t,
            uint8_t, uint16_t, uint32_t, uint64_t,
            float16_t, bfloat16_t, float, double
        >(this, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_CenterCropPad(int opset, pool_t& pool) { return pool_new<CenterCropPad_operator>(pool); }

} // namespace nnr
