#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct OneHot_operator : public operator_t {
    int axis;

    bool init() override {
        if (!(inputs.size() == 3 && outputs.size() == 1)) {
            return false;
        }
        axis = attribute(attr_key_t::axis, (int32_t)-1);
        return true;
    }

    bool reshape() override {
        const tensor_t* indices = inputs[0];
        const tensor_t* depth_t = inputs[1];
        tensor_t* y = outputs[0];

        int64_t depth;
        if (depth_t->type == NNR_DATA_TYPE_INT32) {
            depth = *(const int32_t*)depth_t->data;
        } else if (depth_t->type == NNR_DATA_TYPE_INT64) {
            depth = *(const int64_t*)depth_t->data;
        } else if (depth_t->type == NNR_DATA_TYPE_FLOAT32) {
            depth = static_cast<int64_t>(*(const float*)depth_t->data);
        } else {
            depth = static_cast<int64_t>(*(const double*)depth_t->data);
        }

        int out_ndim = indices->ndim + 1;
        int caxis = axis;
        if (caxis < 0) caxis += out_ndim;

        small_vector<int> dims(out_ndim);
        int d = 0;
        for (int i = 0; i < out_ndim; ++i) {
            if (i == caxis) {
                dims[i] = static_cast<int>(depth);
            } else {
                dims[i] = indices->dims[d++];
            }
        }

        // Output type = type of values tensor
        return y->reshape(dims, inputs[2]->type);
    }

    template <typename T>
    bool exec() {
        const tensor_t* indices = inputs[0];
        const tensor_t* depth_t = inputs[1];
        const tensor_t* values_t = inputs[2];
        tensor_t* y = outputs[0];

        T off_value = ((const T*)values_t->data)[0];
        T on_value = ((const T*)values_t->data)[1];

        int64_t depth;
        if (depth_t->type == NNR_DATA_TYPE_INT32) {
            depth = *(const int32_t*)depth_t->data;
        } else if (depth_t->type == NNR_DATA_TYPE_INT64) {
            depth = *(const int64_t*)depth_t->data;
        } else if (depth_t->type == NNR_DATA_TYPE_FLOAT32) {
            depth = static_cast<int64_t>(*(const float*)depth_t->data);
        } else {
            depth = static_cast<int64_t>(*(const double*)depth_t->data);
        }

        T* py = (T*)y->data;

        int out_ndim = y->ndim;
        int caxis = axis;
        if (caxis < 0) caxis += out_ndim;

        // Fill with off_value
        for (size_t i = 0; i < y->ndata; ++i) {
            py[i] = off_value;
        }

        // For each index, set the on_value at the right position
        int outer = 1;
        for (int i = 0; i < caxis; ++i) outer *= y->dims[i];
        int inner = 1;
        for (int i = caxis + 1; i < out_ndim; ++i) inner *= y->dims[i];

        for (size_t i = 0; i < indices->ndata; ++i) {
            int64_t idx;
            if (indices->type == NNR_DATA_TYPE_INT32) {
                idx = ((const int32_t*)indices->data)[i];
            } else if (indices->type == NNR_DATA_TYPE_INT64) {
                idx = ((const int64_t*)indices->data)[i];
            } else if (indices->type == NNR_DATA_TYPE_FLOAT32) {
                idx = static_cast<int64_t>(((const float*)indices->data)[i]);
            } else {
                idx = static_cast<int64_t>(((const double*)indices->data)[i]);
            }
            if (idx < 0) idx += depth;
            if (idx < 0 || idx >= depth) continue;

            // Compute position in output: (outer_pos, idx, inner_pos)
            int outer_pos = static_cast<int>(i) / inner;
            int inner_pos = static_cast<int>(i) % inner;
            int out_off = outer_pos * static_cast<int>(depth) * inner + static_cast<int>(idx) * inner + inner_pos;
            py[out_off] = on_value;
        }
        return true;
    }

    bool exec() override {
        return typed_exec<OneHot_operator,
            opset_t<9, bool_t, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float16_t, float, double, std::complex<float>, std::complex<double>>
        >(this, opset, inputs[2]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_OneHot(int opset, pool_t& pool)
{
    return pool_new<OneHot_operator>(pool);
}

} // namespace nnr
