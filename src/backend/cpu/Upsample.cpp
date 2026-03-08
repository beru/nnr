#include "nnr.h"
#include "util.h"
#include <cmath>

namespace nnr {

namespace {

struct Upsample_operator : public operator_t {
    int mode; // 0=nearest, 1=linear

    bool init() override {
        if (inputs.empty() || outputs.empty()) return false;
        std::string_view mode_str = attribute(attr_key_t::mode, std::string_view("nearest"));
        if (mode_str == "linear" || mode_str == "bilinear") mode = 1;
        else mode = 0;
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];

        // Opset 7: scales from attribute; Opset 9: scales from input[1]
        small_vector<int> dims(x->ndim);
        const float* scales = nullptr;
        int nscales = 0;

        if (inputs.size() >= 2 && inputs[1] && inputs[1]->ndata > 0) {
            scales = (const float*)inputs[1]->data;
            nscales = (int)inputs[1]->ndata;
        }

        for (int i = 0; i < x->ndim; ++i) {
            if (scales && i < nscales) {
                dims[i] = (int)std::floor(x->dims[i] * scales[i]);
            } else {
                dims[i] = x->dims[i];
            }
        }
        return y->reshape(dims, x->type);
    }

    void get_scales(small_vector<float>& scales) const {
        int ndim = inputs[0]->ndim;
        scales.resize(ndim);
        if (inputs.size() >= 2 && inputs[1] && inputs[1]->ndata > 0) {
            const float* sc = (const float*)inputs[1]->data;
            for (int i = 0; i < ndim && i < (int)inputs[1]->ndata; ++i)
                scales[i] = sc[i];
        } else {
            for (int i = 0; i < ndim; ++i)
                scales[i] = (float)outputs[0]->dims[i] / inputs[0]->dims[i];
        }
    }

    // Fast 4D nearest: precompute index maps, direct nested loop
    template <typename T>
    bool exec_4d_nearest() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;

        small_vector<float> scales;
        get_scales(scales);

        int iN = x->dims[0], iC = x->dims[1], iH = x->dims[2], iW = x->dims[3];
        int oN = y->dims[0], oC = y->dims[1], oH = y->dims[2], oW = y->dims[3];

        // Precompute W index map (reused for every row)
        std::vector<int> wmap(oW);
        float inv_sw = 1.0f / scales[3];
        for (int w = 0; w < oW; w++)
            wmap[w] = std::min((int)std::floor(w * inv_sw), iW - 1);

        // Precompute H index map
        std::vector<int> hmap(oH);
        float inv_sh = 1.0f / scales[2];
        for (int h = 0; h < oH; h++)
            hmap[h] = std::min((int)std::floor(h * inv_sh), iH - 1);

        for (int n = 0; n < oN; n++) {
            int in_ = std::min((int)std::floor(n / scales[0]), iN - 1);
            for (int c = 0; c < oC; c++) {
                int ic = std::min((int)std::floor(c / scales[1]), iC - 1);
                const T* src_ch = px + ((size_t)in_ * iC + ic) * iH * iW;
                T* dst_ch = py + ((size_t)n * oC + c) * oH * oW;
                for (int h = 0; h < oH; h++) {
                    const T* src_row = src_ch + hmap[h] * iW;
                    T* dst_row = dst_ch + h * oW;
                    for (int w = 0; w < oW; w++)
                        dst_row[w] = src_row[wmap[w]];
                }
            }
        }
        return true;
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];

        // Fast 4D nearest path
        if (x->ndim == 4 && mode == 0)
            return exec_4d_nearest<T>();

        const T* px = (const T*)x->data;
        T* py = (T*)y->data;
        int ndim = x->ndim;

        small_vector<float> scales;
        get_scales(scales);

        // Generic fallback
        small_vector<int> out_idx(ndim);
        for (size_t i = 0; i < y->ndata; ++i) {
            size_t rem = i;
            for (int d = ndim - 1; d >= 0; --d) {
                out_idx[d] = (int)(rem % y->dims[d]);
                rem /= y->dims[d];
            }
            size_t src = 0;
            for (int d = 0; d < ndim; ++d) {
                int in_coord = (int)std::min((float)(x->dims[d] - 1),
                    std::floor(out_idx[d] / scales[d]));
                src = src * x->dims[d] + in_coord;
            }
            py[i] = px[src];
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Upsample_operator,
            int8_t, int16_t, int32_t, int64_t,
            uint8_t, uint16_t, uint32_t, uint64_t,
            float16_t, bfloat16_t, float, double
        >(this, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Upsample(int opset, pool_t& pool) { return pool_new<Upsample_operator>(pool); }

} // namespace nnr
