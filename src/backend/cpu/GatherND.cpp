#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct GatherND_operator : public operator_t {
    int batch_dims = 0;

    bool init() override {
        if (!is_inout_size(2, 1)) {
            return false;
        }
        batch_dims = attribute(attr_key_t::batch_dims, (int32_t)0);
        return true;
    }

    bool reshape() override {
        const tensor_t* data = inputs[0];
        const tensor_t* indices = inputs[1];
        tensor_t* y = outputs[0];

        const int data_ndim = data->ndim;
        const int indices_ndim = indices->ndim;
        if (indices_ndim < 1) {
            return false;
        }
        if (batch_dims < 0 || batch_dims >= data_ndim || batch_dims >= indices_ndim) {
            return false;
        }

        // last dimension of indices = length of index tuples
        const int last = indices->dims[indices_ndim - 1];
        if (last < 1 || last > (data_ndim - batch_dims)) {
            return false;
        }

        // Output shape = data.shape[:batch_dims] + indices.shape[batch_dims:-1] + data.shape[batch_dims+last:]
        int out_ndim = batch_dims + (indices_ndim - 1 - batch_dims) + (data_ndim - batch_dims - last);
        if (out_ndim <= 0) {
            // Scalar output
            out_ndim = 0;
        }
        small_vector<int> dims(out_ndim);
        int d = 0;
        // batch dims from data
        for (int i = 0; i < batch_dims; ++i) {
            dims[d++] = data->dims[i];
        }
        // indices.shape[batch_dims:-1]
        for (int i = batch_dims; i < indices_ndim - 1; ++i) {
            dims[d++] = indices->dims[i];
        }
        // data.shape[batch_dims+last:]
        for (int i = batch_dims + last; i < data_ndim; ++i) {
            dims[d++] = data->dims[i];
        }
        return y->reshape(dims, data->type);
    }

    template <typename T>
    bool exec() {
        const tensor_t* data = inputs[0];
        const tensor_t* indices = inputs[1];
        tensor_t* y = outputs[0];
        const T* pdata = (const T*)data->data;
        const int64_t* pidx = (const int64_t*)indices->data;
        T* py = (T*)y->data;

        const int data_ndim = data->ndim;
        const int indices_ndim = indices->ndim;
        const int last = indices->dims[indices_ndim - 1];

        // Compute the number of slices per batch
        // indices "outer" shape = indices.shape[batch_dims:-1]
        int num_slices = 1;
        for (int i = batch_dims; i < indices_ndim - 1; ++i) {
            num_slices *= indices->dims[i];
        }

        // Compute slice size = product of data.shape[batch_dims+last:]
        int slice_size = 1;
        for (int i = batch_dims + last; i < data_ndim; ++i) {
            slice_size *= data->dims[i];
        }

        // Compute batch count = product of data.shape[:batch_dims]
        int batch_count = 1;
        for (int i = 0; i < batch_dims; ++i) {
            batch_count *= data->dims[i];
        }

        // Compute data batch stride = product of data.shape[batch_dims:]
        int data_batch_stride = 1;
        for (int i = batch_dims; i < data_ndim; ++i) {
            data_batch_stride *= data->dims[i];
        }

        // Compute indices batch stride = product of indices.shape[batch_dims:]
        int indices_batch_stride = 1;
        for (int i = batch_dims; i < indices_ndim; ++i) {
            indices_batch_stride *= indices->dims[i];
        }

        // Compute strides for data dimensions [batch_dims .. batch_dims+last)
        small_vector<int> inner_strides(last);
        for (int i = 0; i < last; ++i) {
            int s = 1;
            for (int j = batch_dims + i + 1; j < data_ndim; ++j) {
                s *= data->dims[j];
            }
            inner_strides[i] = s;
        }

        for (int b = 0; b < batch_count; ++b) {
            const T* batch_data = pdata + b * data_batch_stride;
            const int64_t* batch_indices = pidx + b * indices_batch_stride;
            T* batch_out = py + b * (num_slices * slice_size);

            for (int s = 0; s < num_slices; ++s) {
                const int64_t* idx_tuple = batch_indices + s * last;

                // Compute flat offset into data for this index tuple
                int data_offset = 0;
                for (int k = 0; k < last; ++k) {
                    int64_t idx = idx_tuple[k];
                    int dim_size = data->dims[batch_dims + k];
                    if (idx < 0) {
                        idx += dim_size;
                    }
                    data_offset += static_cast<int>(idx) * inner_strides[k];
                }

                // Copy slice
                const T* src = batch_data + data_offset;
                T* dst = batch_out + s * slice_size;
                for (int i = 0; i < slice_size; ++i) {
                    dst[i] = src[i];
                }
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<GatherND_operator,
            opset_t<13, bool_t,
                uint8_t, uint16_t, uint32_t, uint64_t,
                int8_t, int16_t, int32_t, int64_t,
                float16_t, float, double, bfloat16_t,
                std::complex<float>, std::complex<double>,
                std::string>,
            opset_t<11, bool_t,
                uint8_t, uint16_t, uint32_t, uint64_t,
                int8_t, int16_t, int32_t, int64_t,
                float16_t, float, double,
                std::complex<float>, std::complex<double>,
                std::string>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_GatherND(int opset, pool_t& pool)
{
    return pool_new<GatherND_operator>(pool);
}

} // namespace nnr
