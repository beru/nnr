#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct EyeLike_operator : public operator_t {
    int k;

    bool init() override {
        if (!is_inout_size(1, 1)) {
            return false;
        }
        k = attribute(attr_key_t::k, (int32_t)0);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        if (x->ndim != 2) {
            return false;
        }
        int64_t dtype_val = attribute(attr_key_t::dtype, (int64_t)0);
        data_type_t otype = (dtype_val > 0) ? static_cast<data_type_t>(dtype_val) : x->type;
        return y->reshape(x->dim_span(), otype);
    }

    template <typename T>
    bool exec() {
        tensor_t* y = outputs[0];
        T* py = (T*)y->data;
        int rows = y->dims[0];
        int cols = y->dims[1];

        // Zero output
        for (size_t i = 0; i < y->ndata; ++i) {
            py[i] = T{};
        }

        // Set diagonal
        for (int i = 0; i < rows; ++i) {
            int j = i + k;
            if (j >= 0 && j < cols) {
                py[i * cols + j] = T(1);
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<EyeLike_operator,
            opset_t<9, bool_t,
                uint8_t, uint16_t, uint32_t, uint64_t,
                int8_t, int16_t, int32_t, int64_t,
                float16_t, float, double>
        >(this, opset, outputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_EyeLike(int opset, pool_t& pool)
{
    return pool_new<EyeLike_operator>(pool);
}

} // namespace nnr
