#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

enum auto_pad_t {
    NOTSET,
    SAME_UPPER,
    SAME_LOWER,
    VALID,
};

struct ConvTranspose_operator : public operator_t {
    auto_pad_t auto_pad = NOTSET;
    int group = 1;
    small_vector<int> kernels;
    small_vector<int> dilations;
    small_vector<int, MAX_NDIM * 2> pads;
    small_vector<int> strides;
    small_vector<int, MAX_NDIM * 2> output_padding;
    small_vector<int> output_shape;

    bool init() override {
        if (!(inputs.size() >= 2 && outputs.size() == 1))
            return false;

        int64_t* ints = nullptr;
        int l;

        auto_pad = string2enum(attribute(attr_key_t::auto_pad, "NOTSET"), NOTSET);
        group = attribute(attr_key_t::group, 1);

        const tensor_t* w = inputs[1];
        int spatial = w->ndim - 2;

        int nk = attribute(attr_key_t::kernel_shape, ints);
        if (nk > 0) {
            kernels.resize(nk);
            for (int i = 0; i < nk; ++i)
                kernels[i] = (int)ints[i];
        } else {
            kernels.resize(spatial);
            for (int i = 0; i < spatial; ++i)
                kernels[i] = w->dims[i + 2];
        }

        dilations.resize(spatial);
        l = attribute(attr_key_t::dilations, ints);
        for (int i = 0; i < l; ++i) dilations[i] = (int)ints[i];
        for (int i = l; i < spatial; ++i) dilations[i] = 1;

        strides.resize(spatial);
        l = attribute(attr_key_t::strides, ints);
        for (int i = 0; i < l; ++i) strides[i] = (int)ints[i];
        for (int i = l; i < spatial; ++i) strides[i] = 1;

        pads.resize(spatial * 2);
        l = attribute(attr_key_t::pads, ints);
        for (int i = 0; i < l; ++i) pads[i] = (int)ints[i];
        for (int i = l; i < spatial * 2; ++i) pads[i] = 0;

        output_padding.resize(spatial);
        l = attribute(attr_key_t::output_padding, ints);
        for (int i = 0; i < l; ++i) output_padding[i] = (int)ints[i];
        for (int i = l; i < spatial; ++i) output_padding[i] = 0;

        l = attribute(attr_key_t::output_shape, ints);
        if (l > 0) {
            output_shape.resize(l);
            for (int i = 0; i < l; ++i) output_shape[i] = (int)ints[i];
        }

        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        const tensor_t* w = inputs[1];
        int ndim = x->ndim;
        int spatial = ndim - 2;

        small_vector<int> dims(ndim);
        dims[0] = x->dims[0]; // N
        dims[1] = w->dims[1] * group; // C_out

        if (output_shape.size() > 0) {
            for (int i = 0; i < spatial; ++i)
                dims[i + 2] = output_shape[i];
        } else {
            for (int i = 0; i < spatial; ++i) {
                int k_eff = (kernels[i] - 1) * dilations[i] + 1;
                switch (auto_pad) {
                case SAME_UPPER:
                case SAME_LOWER:
                    dims[i + 2] = x->dims[i + 2] * strides[i];
                    break;
                case VALID:
                    dims[i + 2] = x->dims[i + 2] * strides[i] + k_eff - strides[i];
                    break;
                default: // NOTSET
                    dims[i + 2] = strides[i] * (x->dims[i + 2] - 1) + output_padding[i] + k_eff - pads[i] - pads[i + spatial];
                    break;
                }
            }
        }

        // Compute auto_pad pads if needed
        if (auto_pad == SAME_UPPER || auto_pad == SAME_LOWER) {
            for (int i = 0; i < spatial; ++i) {
                int k_eff = (kernels[i] - 1) * dilations[i] + 1;
                int total_pad = strides[i] * (x->dims[i + 2] - 1) + output_padding[i] + k_eff - dims[i + 2];
                if (total_pad < 0) total_pad = 0;
                if (auto_pad == SAME_UPPER) {
                    pads[i] = total_pad / 2;
                    pads[i + spatial] = total_pad - pads[i];
                } else {
                    pads[i + spatial] = total_pad / 2;
                    pads[i] = total_pad - pads[i + spatial];
                }
            }
        }

        return outputs[0]->reshape(dims, x->type);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        const tensor_t* w = inputs[1];
        const tensor_t* b = (inputs.size() > 2) ? inputs[2] : nullptr;
        tensor_t* y = outputs[0];

        const T* px = (const T*)x->data;
        const T* pw = (const T*)w->data;
        const T* pb = b ? (const T*)b->data : nullptr;
        T* py = (T*)y->data;

        int ndim = x->ndim;
        int spatial = ndim - 2;
        int N = x->dims[0];
        int C_in = x->dims[1];
        int C_out = w->dims[1] * group;
        int C_in_per_group = C_in / group;
        int C_out_per_group = C_out / group;

        // Zero output
        memset(py, 0, y->ndata * sizeof(T));

        if (spatial == 1) {
            // 1D ConvTranspose
            int iW = x->dims[2];
            int oW = y->dims[2];
            int kW = kernels[0];

            for (int n = 0; n < N; ++n) {
                for (int g = 0; g < group; ++g) {
                    for (int ic = 0; ic < C_in_per_group; ++ic) {
                        int c_in = g * C_in_per_group + ic;
                        for (int oc = 0; oc < C_out_per_group; ++oc) {
                            int c_out = g * C_out_per_group + oc;
                            for (int ix = 0; ix < iW; ++ix) {
                                T val = px[(n * C_in + c_in) * iW + ix];
                                for (int kx = 0; kx < kW; ++kx) {
                                    int ox = ix * strides[0] + kx * dilations[0] - pads[0];
                                    if (ox >= 0 && ox < oW) {
                                        // w shape: [C_in, C_out_per_group, kW]
                                        T wt = pw[((c_in * C_out_per_group + oc) * kW) + kx];
                                        py[(n * C_out + c_out) * oW + ox] += val * wt;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Add bias
            if (pb) {
                for (int n = 0; n < N; ++n)
                    for (int c = 0; c < C_out; ++c)
                        for (int i = 0; i < oW; ++i)
                            py[(n * C_out + c) * oW + i] += pb[c];
            }
        } else if (spatial == 2) {
            // 2D ConvTranspose
            int iH = x->dims[2], iW = x->dims[3];
            int oH = y->dims[2], oW = y->dims[3];
            int kH = kernels[0], kW = kernels[1];

            for (int n = 0; n < N; ++n) {
                for (int g = 0; g < group; ++g) {
                    for (int ic = 0; ic < C_in_per_group; ++ic) {
                        int c_in = g * C_in_per_group + ic;
                        for (int oc = 0; oc < C_out_per_group; ++oc) {
                            int c_out = g * C_out_per_group + oc;
                            for (int ih = 0; ih < iH; ++ih) {
                                for (int iw = 0; iw < iW; ++iw) {
                                    T val = px[((n * C_in + c_in) * iH + ih) * iW + iw];
                                    for (int kh = 0; kh < kH; ++kh) {
                                        int oh = ih * strides[0] + kh * dilations[0] - pads[0];
                                        if (oh < 0 || oh >= oH) continue;
                                        for (int kw = 0; kw < kW; ++kw) {
                                            int ow = iw * strides[1] + kw * dilations[1] - pads[1];
                                            if (ow < 0 || ow >= oW) continue;
                                            // w shape: [C_in, C_out_per_group, kH, kW]
                                            T wt = pw[((c_in * C_out_per_group + oc) * kH + kh) * kW + kw];
                                            py[((n * C_out + c_out) * oH + oh) * oW + ow] += val * wt;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (pb) {
                for (int n = 0; n < N; ++n)
                    for (int c = 0; c < C_out; ++c)
                        for (int h = 0; h < oH; ++h)
                            for (int w = 0; w < oW; ++w)
                                py[((n * C_out + c) * oH + h) * oW + w] += pb[c];
            }
        } else if (spatial == 3) {
            // 3D ConvTranspose
            int iD = x->dims[2], iH = x->dims[3], iW = x->dims[4];
            int oD = y->dims[2], oH = y->dims[3], oW = y->dims[4];
            int kD = kernels[0], kH = kernels[1], kW = kernels[2];

            for (int n = 0; n < N; ++n) {
                for (int g = 0; g < group; ++g) {
                    for (int ic = 0; ic < C_in_per_group; ++ic) {
                        int c_in = g * C_in_per_group + ic;
                        for (int oc = 0; oc < C_out_per_group; ++oc) {
                            int c_out = g * C_out_per_group + oc;
                            for (int id = 0; id < iD; ++id) {
                                for (int ih = 0; ih < iH; ++ih) {
                                    for (int iw = 0; iw < iW; ++iw) {
                                        T val = px[(((n * C_in + c_in) * iD + id) * iH + ih) * iW + iw];
                                        for (int kd = 0; kd < kD; ++kd) {
                                            int od = id * strides[0] + kd * dilations[0] - pads[0];
                                            if (od < 0 || od >= oD) continue;
                                            for (int kh = 0; kh < kH; ++kh) {
                                                int oh = ih * strides[1] + kh * dilations[1] - pads[1];
                                                if (oh < 0 || oh >= oH) continue;
                                                for (int kw = 0; kw < kW; ++kw) {
                                                    int ow = iw * strides[2] + kw * dilations[2] - pads[2];
                                                    if (ow < 0 || ow >= oW) continue;
                                                    T wt = pw[(((c_in * C_out_per_group + oc) * kD + kd) * kH + kh) * kW + kw];
                                                    py[(((n * C_out + c_out) * oD + od) * oH + oh) * oW + ow] += val * wt;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (pb) {
                for (int n = 0; n < N; ++n)
                    for (int c = 0; c < C_out; ++c)
                        for (int d = 0; d < oD; ++d)
                            for (int h = 0; h < oH; ++h)
                                for (int w = 0; w < oW; ++w)
                                    py[(((n * C_out + c) * oD + d) * oH + h) * oW + w] += pb[c];
            }
        } else {
            return false;
        }

        return true;
    }

    bool exec() override {
        return typed_exec<ConvTranspose_operator,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_ConvTranspose(int opset, pool_t& pool)
{
    return pool_new<ConvTranspose_operator>(pool);
}

} // namespace nnr
