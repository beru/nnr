#include "nnr.h"
#include "util.h"
#include <cmath>
#include <algorithm>

namespace nnr {

namespace {

struct DynamicQuantizeLinear_operator : public operator_t {
    bool init() override {
        if (inputs.size() != 1 || outputs.size() != 3)
            return false;
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        outputs[0]->reshape_identity(x, NNR_DATA_TYPE_UINT8);
        // y_scale: scalar float
        small_vector<int> scalar_dims;
        outputs[1]->reshape(scalar_dims, NNR_DATA_TYPE_FLOAT32);
        // y_zero_point: scalar uint8
        outputs[2]->reshape(scalar_dims, NNR_DATA_TYPE_UINT8);
        return true;
    }

    bool exec() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        tensor_t* y_scale = outputs[1];
        tensor_t* y_zero = outputs[2];

        const float* px = (const float*)x->data;
        uint8_t* py = (uint8_t*)y->data;

        // Find min/max
        float x_min = 0, x_max = 0;
        for (size_t i = 0; i < x->ndata; ++i) {
            x_min = std::min(x_min, px[i]);
            x_max = std::max(x_max, px[i]);
        }

        // Ensure range includes 0
        x_min = std::min(x_min, 0.0f);
        x_max = std::max(x_max, 0.0f);

        float scale = (x_max - x_min) / 255.0f;
        if (scale == 0) scale = 1.0f;

        uint8_t zero_point = (uint8_t)std::clamp((int)std::nearbyint(-x_min / scale), 0, 255);

        ((float*)y_scale->data)[0] = scale;
        ((uint8_t*)y_zero->data)[0] = zero_point;

        for (size_t i = 0; i < x->ndata; ++i) {
            py[i] = (uint8_t)std::clamp((int)std::nearbyint(px[i] / scale) + zero_point, 0, 255);
        }

        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_DynamicQuantizeLinear(int opset, pool_t& pool)
{
    return pool_new<DynamicQuantizeLinear_operator>(pool);
}

} // namespace nnr
