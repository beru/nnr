#include "nnr.h"
#include "util.h"
#include <cmath>
#include <algorithm>

namespace nnr {

namespace {

// QLinearAveragePool: quantized 2D average pooling.
// Inputs: (X, X_scale, X_zp, Y_scale, Y_zp)
// Attributes: kernel_shape, strides, pads, auto_pad, ceil_mode, count_include_pad
struct QLinearAveragePool_operator : public operator_t {
    small_vector<int> kernels;
    small_vector<int> strides_;
    small_vector<int, MAX_NDIM * 2> pads;
    int cpads[4] = {};
    int ceil_mode = 0;
    int count_include_pad = 0;

    bool init() override {
        if (inputs.size() != 5 || outputs.size() != 1) return false;
        int64_t* ints = nullptr;
        int i, l;

        l = attribute(attr_key_t::kernel_shape, ints);
        kernels.resize(l);
        for (i = 0; i < l; i++) kernels[i] = ints[i];
        if (kernels.size() != 2) return false;  // 2D only

        strides_.resize(2);
        l = attribute(attr_key_t::strides, ints);
        for (i = 0; i < l; i++) strides_[i] = ints[i];
        for (; i < 2; i++) strides_[i] = 1;

        pads.resize(4);
        l = attribute(attr_key_t::pads, ints);
        for (i = 0; i < l; i++) pads[i] = ints[i];
        for (; i < 4; i++) pads[i] = 0;

        ceil_mode = attribute(attr_key_t::ceil_mode, 0);
        // Note: count_include_pad not in attr_key_t; default 0
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        if (x->ndim != 4) return false;
        int N = x->dims[0], C = x->dims[1], H = x->dims[2], W = x->dims[3];
        int kH = kernels[0], kW = kernels[1];
        int sH = strides_[0], sW = strides_[1];
        memcpy(cpads, pads.data(), sizeof(int) * 4);

        int oH, oW;
        if (ceil_mode) {
            oH = (int)ceilf((H + cpads[0] + cpads[2] - kH) / (float)sH + 1);
            oW = (int)ceilf((W + cpads[1] + cpads[3] - kW) / (float)sW + 1);
        } else {
            oH = (H + cpads[0] + cpads[2] - kH) / sH + 1;
            oW = (W + cpads[1] + cpads[3] - kW) / sW + 1;
        }
        small_vector<int> dims = {N, C, oH, oW};
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

        int N = x->dims[0], C = x->dims[1], iH = x->dims[2], iW = x->dims[3];
        int oH = y->dims[2], oW = y->dims[3];
        int kH = kernels[0], kW = kernels[1];
        int sH = strides_[0], sW = strides_[1];
        int pH = cpads[0], pW = cpads[1];

        const T* px = (const T*)x->data;
        T* py = (T*)y->data;

        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                const T* xc = px + ((size_t)n * C + c) * iH * iW;
                T* yc = py + ((size_t)n * C + c) * oH * oW;
                for (int oh = 0; oh < oH; oh++) {
                    for (int ow = 0; ow < oW; ow++) {
                        int32_t sum = 0;
                        int count = 0;
                        for (int kh = 0; kh < kH; kh++) {
                            int ih = oh * sH - pH + kh;
                            if (ih < 0 || ih >= iH) {
                                if (count_include_pad) count += kW;
                                continue;
                            }
                            for (int kw = 0; kw < kW; kw++) {
                                int iw = ow * sW - pW + kw;
                                if (iw < 0 || iw >= iW) {
                                    if (count_include_pad) count++;
                                    continue;
                                }
                                sum += (int32_t)xc[ih * iW + iw] - x_zp;
                                count++;
                            }
                        }
                        if (count == 0) count = 1;
                        float avg = (float)sum * x_scale / (float)count;
                        int32_t q = (int32_t)std::nearbyint(avg / y_scale) + y_zp;
                        yc[oh * oW + ow] = (T)std::clamp(q, clamp_min, clamp_max);
                    }
                }
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
operator_t* resolver_default_op_QLinearAveragePool(int opset, pool_t& pool) {
    return pool_new<QLinearAveragePool_operator>(pool);
}

} // namespace nnr
