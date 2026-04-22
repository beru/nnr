#include <cmath>

#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct DeformConv_operator : public operator_t {
    int group = 1;
    int offset_group = 1;
    small_vector<int> kernels;
    small_vector<int> dilations;
    small_vector<int, MAX_NDIM * 2> pads;
    small_vector<int> strides;

    bool init() override {
        if (!(inputs.size() >= 3 && inputs.size() <= 5 && outputs.size() == 1)) {
            return false;
        }
        int64_t* ints = nullptr;
        int i, l;

        group = attribute(attr_key_t::group, 1);
        offset_group = attribute(attr_key_t::offset_group, 1);

        int nk = attribute(attr_key_t::kernel_shape, ints);
        if (nk > 0) {
            kernels.resize(nk);
            for (i = 0; i < nk; ++i) kernels[i] = (int)ints[i];
        }
        l = attribute(attr_key_t::dilations, ints);
        if (l > 0) {
            dilations.resize(l);
            for (i = 0; i < l; ++i) dilations[i] = (int)ints[i];
        }
        l = attribute(attr_key_t::pads, ints);
        if (l > 0) {
            pads.resize(l);
            for (i = 0; i < l; ++i) pads[i] = (int)ints[i];
        }
        l = attribute(attr_key_t::strides, ints);
        if (l > 0) {
            strides.resize(l);
            for (i = 0; i < l; ++i) strides[i] = (int)ints[i];
        }
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        const tensor_t* w = inputs[1];
        tensor_t* y = outputs[0];
        if (x->ndim != 4 || w->ndim != 4) return false;  // 2D spatial only
        const int spatial = 2;

        if (kernels.empty()) {
            kernels.resize(spatial);
            for (int i = 0; i < spatial; ++i) kernels[i] = w->dims[i + 2];
        }
        if (dilations.empty()) {
            dilations.resize(spatial);
            for (int i = 0; i < spatial; ++i) dilations[i] = 1;
        }
        if (pads.empty()) {
            pads.resize(spatial * 2);
            for (int i = 0; i < spatial * 2; ++i) pads[i] = 0;
        }
        if (strides.empty()) {
            strides.resize(spatial);
            for (int i = 0; i < spatial; ++i) strides[i] = 1;
        }

        const int N = x->dims[0];
        const int iH = x->dims[2];
        const int iW = x->dims[3];
        const int M = w->dims[0];
        const int kH = kernels[0];
        const int kW = kernels[1];
        const int dH = dilations[0];
        const int dW = dilations[1];
        const int sH = strides[0];
        const int sW = strides[1];
        const int pT = pads[0];
        const int pL = pads[1];
        const int pB = pads[2];
        const int pR = pads[3];

        const int oH = (iH + pT + pB - dH * (kH - 1) - 1) / sH + 1;
        const int oW = (iW + pL + pR - dW * (kW - 1) - 1) / sW + 1;

        int dims[4] = {N, M, oH, oW};
        return y->reshape(std::span<const int>(dims, 4), x->type);
    }

    bool exec() override {
        const tensor_t* X = inputs[0];
        const tensor_t* W = inputs[1];
        const tensor_t* OFF = inputs[2];
        const tensor_t* B = (inputs.size() >= 4) ? inputs[3] : nullptr;
        const tensor_t* MASK = (inputs.size() >= 5) ? inputs[4] : nullptr;
        tensor_t* Y = outputs[0];

        if (X->type != NNR_DATA_TYPE_FLOAT32) return false;

        const int N = X->dims[0];
        const int C = X->dims[1];
        const int iH = X->dims[2];
        const int iW = X->dims[3];
        const int M = W->dims[0];
        const int kC = W->dims[1];  // C / group
        const int kH = kernels[0];
        const int kW = kernels[1];
        const int oH = Y->dims[2];
        const int oW = Y->dims[3];
        const int dH = dilations[0];
        const int dW = dilations[1];
        const int sH = strides[0];
        const int sW = strides[1];
        const int pT = pads[0];
        const int pL = pads[1];

        if (group != 1) return false;
        if (kC * group != C) return false;

        const float* px = (const float*)X->data;
        const float* pw = (const float*)W->data;
        const float* poff = (const float*)OFF->data;
        const float* pb = B ? (const float*)B->data : nullptr;
        const float* pmask = MASK ? (const float*)MASK->data : nullptr;
        float* py = (float*)Y->data;

        const int K = kH * kW;
        const int C_per_off_group = C / offset_group;
        if (C_per_off_group * offset_group != C) return false;

        // Strides
        const int x_stride_n = C * iH * iW;
        const int x_stride_c = iH * iW;
        const int w_stride_m = kC * kH * kW;
        const int off_stride_n = offset_group * K * 2 * oH * oW;
        const int off_stride_g = K * 2 * oH * oW;
        const int off_spatial = oH * oW;
        const int mask_stride_n = offset_group * K * oH * oW;
        const int mask_stride_g = K * oH * oW;
        const int y_stride_n = M * oH * oW;
        const int y_stride_m = oH * oW;

        auto sample = [&](const float* xc_plane, float py_pos, float px_pos) -> float {
            if (py_pos <= -1.0f || px_pos <= -1.0f ||
                py_pos >= (float)iH || px_pos >= (float)iW) {
                return 0.0f;
            }
            int iy0 = (int)std::floor(py_pos);
            int ix0 = (int)std::floor(px_pos);
            int iy1 = iy0 + 1;
            int ix1 = ix0 + 1;
            float wy1 = py_pos - (float)iy0;
            float wy0 = 1.0f - wy1;
            float wx1 = px_pos - (float)ix0;
            float wx0 = 1.0f - wx1;
            auto at = [&](int y, int x) -> float {
                if (y < 0 || y >= iH || x < 0 || x >= iW) return 0.0f;
                return xc_plane[y * iW + x];
            };
            return wy0 * wx0 * at(iy0, ix0)
                 + wy0 * wx1 * at(iy0, ix1)
                 + wy1 * wx0 * at(iy1, ix0)
                 + wy1 * wx1 * at(iy1, ix1);
        };

        for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
                const float bias = pb ? pb[m] : 0.0f;
                for (int oy = 0; oy < oH; ++oy) {
                    for (int ox = 0; ox < oW; ++ox) {
                        float acc = bias;
                        for (int c = 0; c < C; ++c) {
                            const int g = c / C_per_off_group;
                            const float* xc_plane = px + n * x_stride_n + c * x_stride_c;
                            for (int ky = 0; ky < kH; ++ky) {
                                for (int kx = 0; kx < kW; ++kx) {
                                    const int k = ky * kW + kx;
                                    const int off_base = n * off_stride_n + g * off_stride_g
                                                       + (2 * k) * off_spatial
                                                       + oy * oW + ox;
                                    const float off_y = poff[off_base];
                                    const float off_x = poff[off_base + off_spatial];
                                    const float iy_base = (float)(oy * sH - pT + ky * dH);
                                    const float ix_base = (float)(ox * sW - pL + kx * dW);
                                    const float py_pos = iy_base + off_y;
                                    const float px_pos = ix_base + off_x;
                                    float v = sample(xc_plane, py_pos, px_pos);
                                    if (pmask) {
                                        const int mask_idx = n * mask_stride_n
                                                           + g * mask_stride_g
                                                           + k * off_spatial
                                                           + oy * oW + ox;
                                        v *= pmask[mask_idx];
                                    }
                                    const float wgt = pw[m * w_stride_m
                                                       + c * (kH * kW)
                                                       + ky * kW + kx];
                                    acc += wgt * v;
                                }
                            }
                        }
                        py[n * y_stride_n + m * y_stride_m + oy * oW + ox] = acc;
                    }
                }
            }
        }
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_DeformConv(int opset, pool_t& pool)
{
    return pool_new<DeformConv_operator>(pool);
}

} // namespace nnr
