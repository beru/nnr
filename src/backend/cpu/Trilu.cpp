#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Trilu_operator : public operator_t {
    int upper;

    bool init() override {
        if (outputs.size() != 1) {
            return false;
        }
        upper = attribute(attr_key_t::upper, (int32_t)1);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        if (x->ndim < 2) {
            return false;
        }
        return y->reshape_identity(x);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;
        const int ndim = x->ndim;

        int64_t k = 0;
        if (inputs.size() > 1 && inputs[1] && inputs[1]->ndata > 0) {
            k = *(const int64_t*)inputs[1]->data;
        }

        const int rows = x->dims[ndim - 2];
        const int cols = x->dims[ndim - 1];

        // Batch size = product of all dims except last two
        int batch = 1;
        for (int i = 0; i < ndim - 2; ++i) {
            batch *= x->dims[i];
        }

        int mat_size = rows * cols;
        for (int b = 0; b < batch; ++b) {
            const T* src = px + b * mat_size;
            T* dst = py + b * mat_size;
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    bool keep;
                    if (upper) {
                        keep = (j >= i + k);
                    } else {
                        keep = (j <= i + k);
                    }
                    dst[i * cols + j] = keep ? src[i * cols + j] : T{};
                }
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Trilu_operator,
            opset_t<14, bool_t, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float16_t, float, double, bfloat16_t, std::complex<float>, std::complex<double>>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Trilu(int opset, pool_t& pool)
{
    return pool_new<Trilu_operator>(pool);
}

} // namespace nnr
