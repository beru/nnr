#include "nnr.h"
#include "util.h"
#include <cmath>
#include <algorithm>

namespace nnr {

namespace {

// QLinearGlobalAveragePool: quantized global average pooling.
// Inputs: (X, X_scale, X_zp, Y_scale, Y_zp)
// Pools over all spatial dimensions → output shape [N, C, 1, 1] (or [N, C] for 3D).
struct QLinearGlobalAveragePool_operator : public operator_t {
    bool init() override {
        return inputs.size() == 5 && outputs.size() == 1;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        if (x->ndim < 3) return false;
        small_vector<int> dims(x->ndim);
        dims[0] = x->dims[0];
        dims[1] = x->dims[1];
        for (int d = 2; d < x->ndim; d++) dims[d] = 1;
        return outputs[0]->reshape(dims, x->type);
    }

    template <typename T>
    bool exec_typed() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        float x_scale = *(float*)inputs[1]->data;
        float y_scale = *(float*)inputs[3]->data;

        int32_t x_zp = 0, y_zp = 0;
        if (inputs[2]->ndata > 0) x_zp = (int32_t)((T*)inputs[2]->data)[0];
        if (inputs[4]->ndata > 0) y_zp = (int32_t)((T*)inputs[4]->data)[0];

        int clamp_min, clamp_max;
        if constexpr (std::is_same_v<T, uint8_t>) { clamp_min = 0; clamp_max = 255; }
        else { clamp_min = -128; clamp_max = 127; }

        int N = x->dims[0], C = x->dims[1];
        size_t spatial = 1;
        for (int d = 2; d < x->ndim; d++) spatial *= x->dims[d];

        float rs = x_scale / (y_scale * (float)spatial);
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;

        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                const T* src = px + ((size_t)n * C + c) * spatial;
                int32_t sum = 0;
                for (size_t s = 0; s < spatial; s++)
                    sum += (int32_t)src[s];
                sum -= (int32_t)(x_zp * spatial);
                int32_t q = (int32_t)std::nearbyint((float)sum * rs) + y_zp;
                py[(size_t)n * C + c] = (T)std::clamp(q, clamp_min, clamp_max);
            }
        }
        return true;
    }

    bool exec() override {
        data_type_t type = inputs[0]->type;
        if (type == NNR_DATA_TYPE_UINT8) return exec_typed<uint8_t>();
        if (type == NNR_DATA_TYPE_INT8) return exec_typed<int8_t>();
        return false;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_QLinearGlobalAveragePool(int opset, pool_t& pool) {
    return pool_new<QLinearGlobalAveragePool_operator>(pool);
}

} // namespace nnr
