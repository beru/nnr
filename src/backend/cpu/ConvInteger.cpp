#include "nnr.h"
#include "refnd.h"
#include "util.h"
#include "kernel/conv.h"

namespace nnr {

namespace {

enum auto_pad_t {
    NOTSET,
    SAME_UPPER,
    SAME_LOWER,
    VALID,
};

struct ConvInteger_operator : public operator_t {
    auto_pad_t auto_pad = NOTSET;
    int group = 0;
    small_vector<int> kernels;
    small_vector<int> dilations;
    small_vector<int, MAX_NDIM * 2> pads;
    small_vector<int> strides;

    int cpads[32] = {0};

    bool init() override {
        if (!(inputs.size() >= 2 && outputs.size() == 1)) {
            return false;
        }
        int64_t* ints = nullptr;
        int i, l;

        auto_pad = string2enum(attribute(attr_key_t::auto_pad, "NOTSET"), NOTSET);
        group = attribute(attr_key_t::group, 1);

        int nk = attribute(attr_key_t::kernel_shape, ints);
        if (nk > 0) {
            kernels.resize(nk);
            for (i = 0; i < nk; ++i) {
                kernels[i] = (int)ints[i];
            }
        }
        // dilations, pads, strides are deferred to reshape if kernel_shape not provided
        l = attribute(attr_key_t::dilations, ints);
        if (l > 0) {
            dilations.resize(l);
            for (i = 0; i < l; ++i)
                dilations[i] = (int)ints[i];
        }
        l = attribute(attr_key_t::pads, ints);
        if (l > 0) {
            pads.resize(l);
            for (i = 0; i < l; ++i)
                pads[i] = (int)ints[i];
        }
        l = attribute(attr_key_t::strides, ints);
        if (l > 0) {
            strides.resize(l);
            for (i = 0; i < l; ++i)
                strides[i] = (int)ints[i];
        }
        return true;
    }

    bool reshape() override {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        const tensor_t* w = inputs[1];
        const int ndim = x->ndim;
        const int spatial = ndim - 2;

        // Infer kernel_shape from weight if not provided
        if (kernels.empty()) {
            kernels.resize(spatial);
            for (int i = 0; i < spatial; ++i)
                kernels[i] = w->dims[i + 2];
        }
        // Default dilations
        if (dilations.empty()) {
            dilations.resize(spatial);
            for (int i = 0; i < spatial; ++i)
                dilations[i] = 1;
        }
        // Default pads
        if (pads.empty()) {
            pads.resize(spatial * 2);
            for (int i = 0; i < spatial * 2; ++i)
                pads[i] = 0;
        }
        // Default strides
        if (strides.empty()) {
            strides.resize(spatial);
            for (int i = 0; i < spatial; ++i)
                strides[i] = 1;
        }

        small_vector<int> dims(ndim);

        switch (auto_pad) {
        case NOTSET:
            memcpy(cpads, pads.data(), sizeof(int) * pads.size());
            break;
        case SAME_UPPER:
            for (int i = 0; i < spatial; ++i) {
                int pad = (int)(ceilf(x->dims[i + 2] / (float)strides[i]) - 1) * strides[i] + ((kernels[i] - 1) * dilations[i] + 1) - x->dims[i + 2];
                cpads[i] = pad / 2;
                cpads[i + spatial] = pad - cpads[i];
            }
            break;
        case SAME_LOWER:
            for (int i = 0; i < spatial; ++i) {
                int pad = (int)(ceilf(x->dims[i + 2] / (float)strides[i]) - 1) * strides[i] + ((kernels[i] - 1) * dilations[i] + 1) - x->dims[i + 2];
                cpads[i + spatial] = pad / 2;
                cpads[i] = pad - cpads[i + spatial];
            }
            break;
        case VALID:
            memset(cpads, 0, sizeof(cpads));
            break;
        default:
            break;
        }

        dims[0] = x->dims[0];
        dims[1] = w->dims[0];
        for (int i = 0; i < spatial; ++i) {
            switch (auto_pad) {
            case NOTSET:
                dims[i + 2] = (int)floorf((x->dims[i + 2] + cpads[i] + cpads[i + spatial] - ((kernels[i] - 1) * dilations[i] + 1)) / (float)strides[i] + 1);
                break;
            case SAME_UPPER:
            case SAME_LOWER:
                dims[i + 2] = (int)ceilf(x->dims[i + 2] / (float)strides[i]);
                break;
            case VALID:
                dims[i + 2] = (int)ceilf((x->dims[i + 2] - ((kernels[i] - 1) * dilations[i] + 1) + 1) / (float)strides[i]);
                break;
            default:
                break;
            }
        }
        return y->reshape(dims, NNR_DATA_TYPE_INT32);
    }

    // SIMD-accelerated 4D path: convert int8→float, im2col + float GEMM, round to int32.
    bool exec_simd_4d() {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        const tensor_t* w = inputs[1];
        if (x->ndim != 4) return false;

        int32_t x_zp = 0;
        if (inputs.size() > 2 && inputs[2]) {
            if (inputs[2]->type == NNR_DATA_TYPE_UINT8) x_zp = ((uint8_t*)inputs[2]->data)[0];
            else if (inputs[2]->type == NNR_DATA_TYPE_INT8) x_zp = ((int8_t*)inputs[2]->data)[0];
        }

        // w_zero_point: scalar or per-channel
        int w_zp_count = 0;
        int32_t w_zp_scalar = 0;
        int32_t* w_zp_arr = nullptr;
        if (inputs.size() > 3 && inputs[3]) {
            w_zp_count = (int)inputs[3]->ndata;
            if (w_zp_count > 1) {
                w_zp_arr = (int32_t*)_aligned_malloc(w_zp_count * sizeof(int32_t), 64);
                if (inputs[3]->type == NNR_DATA_TYPE_UINT8)
                    for (int i = 0; i < w_zp_count; i++) w_zp_arr[i] = ((uint8_t*)inputs[3]->data)[i];
                else
                    for (int i = 0; i < w_zp_count; i++) w_zp_arr[i] = ((int8_t*)inputs[3]->data)[i];
            } else {
                if (inputs[3]->type == NNR_DATA_TYPE_UINT8) w_zp_scalar = ((uint8_t*)inputs[3]->data)[0];
                else if (inputs[3]->type == NNR_DATA_TYPE_INT8) w_zp_scalar = ((int8_t*)inputs[3]->data)[0];
            }
        }

        int iC = x->dims[1], iH = x->dims[2], iW = x->dims[3];
        int M = w->dims[0], C = w->dims[1], kH = w->dims[2], kW = w->dims[3];
        int oN = y->dims[0], oH = y->dims[2], oW = y->dims[3];
        int MM = M / group, CC = iC / group;
        int CHW = C * kH * kW;
        int spatial = oH * oW;
        int sH = strides[0], sW = strides[1];
        int dH = dilations[0], dW = dilations[1];
        int pH = cpads[0], pW = cpads[1];

        // Convert weights to float (subtract per-channel zero-point)
        float* w_f32 = (float*)_aligned_malloc(w->ndata * sizeof(float), 64);
        for (int oc = 0; oc < M; oc++) {
            int32_t wzp = w_zp_arr ? w_zp_arr[oc] : w_zp_scalar;
            size_t base = (size_t)oc * C * kH * kW;
            for (int i = 0; i < C * kH * kW; i++) {
                int32_t val;
                if (w->type == NNR_DATA_TYPE_UINT8) val = ((uint8_t*)w->data)[base + i];
                else val = ((int8_t*)w->data)[base + i];
                w_f32[base + i] = (float)(val - wzp);
            }
        }

        // Convert input to float (subtract zero-point)
        float* x_f32 = (float*)_aligned_malloc(x->ndata * sizeof(float), 64);
        if (x->type == NNR_DATA_TYPE_UINT8) {
            const uint8_t* px = (const uint8_t*)x->data;
            for (size_t i = 0; i < x->ndata; i++) x_f32[i] = (float)((int32_t)px[i] - x_zp);
        } else {
            const int8_t* px = (const int8_t*)x->data;
            for (size_t i = 0; i < x->ndata; i++) x_f32[i] = (float)((int32_t)px[i] - x_zp);
        }

        float* col = (float*)_aligned_malloc((size_t)CHW * spatial * sizeof(float), 64);
        int32_t* py = (int32_t*)y->data;

        for (int n = 0; n < oN; n++) {
            for (int g = 0; g < group; g++) {
                const float* xn = x_f32 + ((size_t)n * iC + g * CC) * iH * iW;
                float* yn_f = (float*)(py + ((size_t)n * M + g * MM) * spatial);
                // Temporarily use output as float buffer — sizes match (int32 = float = 4 bytes)
                im2col(col, xn, C, iH, iW, kH, kW, oH, oW, sH, sW, pH, pW, dH, dW);
                dgemm_generic(MM, spatial, CHW,
                    w_f32 + (size_t)g * MM * CHW, col, yn_f);
            }
            // Convert float output to int32 (round to nearest)
            int32_t* out = py + (size_t)n * M * spatial;
            float* outf = (float*)out;
            for (int i = 0; i < M * spatial; i++)
                out[i] = (int32_t)std::nearbyint(outf[i]);
        }

        _aligned_free(w_f32);
        _aligned_free(x_f32);
        _aligned_free(col);
        _aligned_free(w_zp_arr);
        return true;
    }

    bool exec() override {
        // SIMD-accelerated path for 4D
        if (inputs[0]->ndim == 4)
            return exec_simd_4d();

        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        const tensor_t* w = inputs[1];
        const int ndim = x->ndim;

        // Get zero points (default to 0)
        int32_t x_zp = 0;
        if (inputs.size() > 2 && inputs[2]) {
            const tensor_t* xzp = inputs[2];
            if (xzp->type == NNR_DATA_TYPE_UINT8) {
                x_zp = ((uint8_t*)xzp->data)[0];
            } else if (xzp->type == NNR_DATA_TYPE_INT8) {
                x_zp = ((int8_t*)xzp->data)[0];
            }
        }

        // w_zero_point can be scalar or per-output-channel
        int32_t w_zp_scalar = 0;
        int32_t* w_zp_per_channel = nullptr;
        bool w_zp_is_per_channel = false;
        if (inputs.size() > 3 && inputs[3]) {
            const tensor_t* wzp = inputs[3];
            int wzp_numel = (int)wzp->ndata; // uint8/int8 are 1 byte each
            if (wzp_numel > 1) {
                w_zp_is_per_channel = true;
                w_zp_per_channel = new (std::nothrow) int32_t[wzp_numel];
                if (wzp->type == NNR_DATA_TYPE_UINT8) {
                    uint8_t* p = (uint8_t*)wzp->data;
                    for (int i = 0; i < wzp_numel; ++i)
                        w_zp_per_channel[i] = p[i];
                } else {
                    int8_t* p = (int8_t*)wzp->data;
                    for (int i = 0; i < wzp_numel; ++i)
                        w_zp_per_channel[i] = p[i];
                }
            } else {
                if (wzp->type == NNR_DATA_TYPE_UINT8) {
                    w_zp_scalar = ((uint8_t*)wzp->data)[0];
                } else if (wzp->type == NNR_DATA_TYPE_INT8) {
                    w_zp_scalar = ((int8_t*)wzp->data)[0];
                }
            }
        }

        if (ndim == 4) {
            int iC = x->dims[1];
            int iH = x->dims[2];
            int iW = x->dims[3];

            int M = w->dims[0];
            int C = w->dims[1];
            int H = w->dims[2];
            int W = w->dims[3];

            int oN = y->dims[0];
            int oC = w->dims[0];
            int oH = y->dims[2];
            int oW = y->dims[3];

            int MM = M / group;
            int CC = iC / group;

            int32_t* py = (int32_t*)y->data;
            ref4d<int32_t> oy(oW, oH, oC, py);

            auto get_x = [&](int n, int c, int h, int w) -> int32_t {
                if (x->type == NNR_DATA_TYPE_UINT8) {
                    return (int32_t)((uint8_t*)x->data)[((n * iC + c) * iH + h) * iW + w] - x_zp;
                } else {
                    return (int32_t)((int8_t*)x->data)[((n * iC + c) * iH + h) * iW + w] - x_zp;
                }
            };

            auto get_w = [&](int m, int c, int h, int w) -> int32_t {
                int32_t wval;
                if (this->inputs[1]->type == NNR_DATA_TYPE_UINT8) {
                    wval = (int32_t)((uint8_t*)this->inputs[1]->data)[((m * C + c) * H + h) * W + w];
                } else {
                    wval = (int32_t)((int8_t*)this->inputs[1]->data)[((m * C + c) * H + h) * W + w];
                }
                if (w_zp_is_per_channel) {
                    wval -= w_zp_per_channel[m];
                } else {
                    wval -= w_zp_scalar;
                }
                return wval;
            };

            for (int n = 0; n < oN; ++n) {
                for (int g = 0; g < group; ++g) {
                    for (int m = 0; m < MM; ++m) {
                        int oc = g * MM + m;
                        for (int oh = 0; oh < oH; ++oh) {
                            for (int ow = 0; ow < oW; ++ow) {
                                int base_h = oh * strides[0] - cpads[0];
                                int base_w = ow * strides[1] - cpads[1];
                                int32_t sum = 0;
                                for (int c = 0; c < C; ++c) {
                                    int ic = g * CC + c;
                                    for (int kh = 0; kh < H; ++kh) {
                                        int ih = base_h + kh * dilations[0];
                                        if (ih < 0 || ih >= iH)
                                            continue;
                                        for (int kw = 0; kw < W; ++kw) {
                                            int iw = base_w + kw * dilations[1];
                                            if (iw < 0 || iw >= iW)
                                                continue;
                                            sum += get_x(n, ic, ih, iw) * get_w(oc, c, kh, kw);
                                        }
                                    }
                                }
                                oy[n][oc][oh][ow] = sum;
                            }
                        }
                    }
                }
            }
        }

        if (w_zp_per_channel) {
            delete[] w_zp_per_channel;
        }
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_ConvInteger(int opset, pool_t& pool) {
    return pool_new<ConvInteger_operator>(pool);
}

} // namespace nnr
