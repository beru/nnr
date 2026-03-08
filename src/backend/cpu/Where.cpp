#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Where_operator : public operator_t {

    bool init() override {
        return is_inout_size(3, 1);
    }

    bool reshape() override {
        const tensor_t* condition = inputs[0];
        const tensor_t* x = inputs[1];
        const tensor_t* y_in = inputs[2];
        tensor_t* output = outputs[0];
        if (condition->type != NNR_DATA_TYPE_BOOL) {
            return false;
        }
        if (x->type != y_in->type && x->type != NNR_DATA_TYPE_UNDEFINED && y_in->type != NNR_DATA_TYPE_UNDEFINED) {
            return false;
        }
        data_type_t out_type = (x->type != NNR_DATA_TYPE_UNDEFINED) ? x->type : y_in->type;

        // Compute broadcast shape of condition, x, y
        // First broadcast x and y
        const int ndim_xy = max(x->ndim, y_in->ndim);
        small_vector<int> dims_xy(ndim_xy);
        for (int i = x->ndim - 1, j = y_in->ndim - 1, k = ndim_xy - 1; k >= 0; k--) {
            int a = (i >= 0) ? x->dims[i--] : 1;
            int b = (j >= 0) ? y_in->dims[j--] : 1;
            if (a != b && a != 1 && b != 1) return false;
            dims_xy[k] = max(a, b);
        }
        // Then broadcast with condition
        const int ndim = max(ndim_xy, condition->ndim);
        small_vector<int> dims(ndim);
        for (int i = ndim_xy - 1, j = condition->ndim - 1, k = ndim - 1; k >= 0; k--) {
            int a = (i >= 0) ? dims_xy[i--] : 1;
            int b = (j >= 0) ? condition->dims[j--] : 1;
            if (a != b && a != 1 && b != 1) return false;
            dims[k] = max(a, b);
        }
        return output->reshape(dims, out_type);
    }

    template <typename T>
    bool exec() {
        const tensor_t* condition = inputs[0];
        const tensor_t* x = inputs[1];
        const tensor_t* y = inputs[2];
        tensor_t* output = outputs[0];
        T* pout = (T*)output->data;
        T* pin;

        for (size_t i = 0, l = output->ndata; i < l; ++i) {
            bool_t* c = (bool_t*)condition->broadcast_map_address(output, i);
            if (*c) {
                pin = (T*)x->broadcast_map_address(output, i);
            }else {
                pin = (T*)y->broadcast_map_address(output, i);
            }
            pout[i] = *pin;
        }
        return true;
    }

    bool exec() override {
        data_type_t type = inputs[1]->type;
        if (inputs.size() != 3) {
            return false;
        }
        if (opset >= 16) {
            return typed_exec<Where_operator,
                bool_t,
                uint8_t, uint16_t, uint32_t, uint64_t,
                int8_t, int16_t, int32_t, int64_t,
                float16_t, float, double, bfloat16_t,
                std::complex<float>, std::complex<double>,
                std::string
            >(this, type);
        }else if (opset >= 9) {
            return typed_exec<Where_operator,
                bool_t,
                uint8_t, uint16_t, uint32_t, uint64_t,
                int8_t, int16_t, int32_t, int64_t,
                float16_t, float, double,
                std::complex<float>, std::complex<double>,
                std::string
            >(this, type);
        }
        return false;
    }
};

} // namespace {

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Where(int opset, pool_t& pool)
{
    return pool_new<Where_operator>(pool);
}

} // namespace nnr
