#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct GatherElements_operator : public operator_t {
    int axis = 0;
    int caxis = 0;

    bool init() override {
        if (!is_inout_size(2, 1)) {
            return false;
        }
        axis = attribute(attr_key_t::axis, (int32_t)0);
        return true;
    }

    bool reshape() override {
        const tensor_t* data = inputs[0];
        const tensor_t* indices = inputs[1];
        tensor_t* y = outputs[0];

        const int ndim = data->ndim;
        caxis = axis;
        if (caxis < 0) {
            caxis += ndim;
        }
        if (caxis < 0 || caxis >= ndim) {
            return false;
        }
        if (indices->ndim != ndim) {
            return false;
        }
        // Output shape = indices shape
        small_vector<int> dims(indices->ndim);
        for (int i = 0; i < indices->ndim; ++i) {
            dims[i] = indices->dims[i];
        }
        return y->reshape(dims, data->type);
    }

    template <typename T>
    bool exec() {
        const tensor_t* data = inputs[0];
        const tensor_t* indices = inputs[1];
        tensor_t* y = outputs[0];
        const T* pdata = (const T*)data->data;
        T* py = (T*)y->data;
        const int ndim = data->ndim;
        const int axis_dim = data->dims[caxis];

        for (size_t oi = 0, l = y->ndata; oi < l; ++oi) {
            small_vector<int> idx(ndim);
            y->offset_to_indices(oi, idx);

            // Get index value from indices tensor
            int64_t index_val;
            if (indices->type == NNR_DATA_TYPE_INT32) {
                index_val = ((const int32_t*)indices->data)[oi];
            } else {
                index_val = ((const int64_t*)indices->data)[oi];
            }

            // Handle negative indices
            if (index_val < 0) {
                index_val += axis_dim;
            }

            // Replace axis dimension with the gathered index
            idx[caxis] = static_cast<int>(index_val);
            int src_offset = data->indices_to_offset(idx);
            py[oi] = pdata[src_offset];
        }
        return true;
    }

    bool exec() override {
        return typed_exec<GatherElements_operator,
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
operator_t* resolver_default_op_GatherElements(int opset, pool_t& pool)
{
    return pool_new<GatherElements_operator>(pool);
}

} // namespace nnr
