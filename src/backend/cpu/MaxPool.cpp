#include "nnr.h"
#include "layout_cost.h"
#include "util.h"
#include "kernel/pool.h"
#include "layout_cost.h"
#ifdef NNR_ARCH_X64
#include "backend/x64/pool_x64.h"
#elifdef NNR_ARCH_ARM64
#include "backend/arm/pool_neon.h"
#endif
#include <cfloat>

namespace nnr {

namespace {

enum auto_pad_t {
    NOTSET,
    SAME_UPPER,
    SAME_LOWER,
    VALID,
};

struct MaxPool_operator : public operator_t {

    auto_pad_t auto_pad;
    int ceil_mode = 0;
    int storage_order = 0;
    small_vector<int> kernels;
    small_vector<int> dilations;
    small_vector<int, MAX_NDIM * 2> pads;
    small_vector<int> strides;

    int cpads[32] = {0};

    bool init() override {
        if (!(inputs.size() == 1 && outputs.size() >= 1)) {
            return false;
        }
        int64_t* ints;
        int i, l;

        auto_pad = string2enum(attribute(attr_key_t::auto_pad, "NOTSET"), NOTSET);
        ceil_mode = attribute(attr_key_t::ceil_mode, 0);
        storage_order = attribute(attr_key_t::storage_order, 0);
        {
            int nkernel = attribute(attr_key_t::kernel_shape, ints);
            if (nkernel > 0) {
                kernels.resize(nkernel);
                for (i = 0; i < kernels.size(); ++i) {
                    kernels[i] = ints[i];
                }
            }
        }
        dilations.resize(kernels.size());
        if (!dilations.empty()) {
            l = attribute(attr_key_t::dilations, ints);
            for (i = 0; i < l; ++i) {
                dilations[i] = ints[i];
            }
            for (; i < dilations.size(); ++i) {
                dilations[i] = 1;
            }
        }
        pads.resize(kernels.size() * 2);
        if (!pads.empty()) {
            l = attribute(attr_key_t::pads, ints);
            for (i = 0; i < l; ++i) {
                pads[i] = ints[i];
            }
            for (; i < pads.size(); ++i) {
                pads[i] = 0;
            }
        }
        strides.resize(kernels.size());
        if (!strides.empty()) {
            l = attribute(attr_key_t::strides, ints);
            for (i = 0; i < l; ++i) {
                strides[i] = ints[i];
            }
            for (; i < strides.size(); ++i) {
                strides[i] = 1;
            }
        }
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        small_vector<int> dims(x->ndim);

        switch (auto_pad) {
        case NOTSET:
            memcpy(cpads, pads.data(), sizeof(int) * pads.size());
            break;
        case SAME_UPPER:
            for (int i = 0; i < pads.size() / 2; ++i) {
                int pad = (ceilf(x->dims[i + 2] / (float)strides[i]) - 1) * strides[i] + ((kernels[i] - 1) * dilations[i] + 1) - x->dims[i + 2];
                cpads[i] = pad / 2;
                cpads[i + kernels.size()] = pad - cpads[i];
            }
            break;
        case SAME_LOWER:
            for (int i = 0; i < pads.size() / 2; ++i) {
                int pad = (ceilf(x->dims[i + 2] / (float)strides[i]) - 1) * strides[i] + ((kernels[i] - 1) * dilations[i] + 1) - x->dims[i + 2];
                cpads[i + kernels.size()] = pad / 2;
                cpads[i] = pad - cpads[i + kernels.size()];
            }
            break;
        case VALID:
            memset(cpads, 0, sizeof(int) * pads.size());
            break;
        default:
            break;
        }
        dims[0] = x->dims[0];
        dims[1] = x->dims[1];
        for (int i = 0; i < x->ndim - 2; ++i) {
            switch (auto_pad) {
            case NOTSET:
                if (ceil_mode) {
                    dims[i + 2] = ceilf((x->dims[i + 2] + cpads[i] + cpads[i + kernels.size()] - ((kernels[i] - 1) * dilations[i] + 1)) / (float)strides[i] + 1);
                    if ((dims[i + 2] - 1) * strides[i] - cpads[i] >= x->dims[i + 2])
                        dims[i + 2] -= 1;
                }else {
                    dims[i + 2] = floorf((x->dims[i + 2] + cpads[i] + cpads[i + kernels.size()] - ((kernels[i] - 1) * dilations[i] + 1)) / (float)strides[i] + 1);
                }
                break;
            case SAME_UPPER:
            case SAME_LOWER:
                dims[i + 2] = ceilf(x->dims[i + 2] / (float)strides[i]);
                break;
            case VALID:
                dims[i + 2] = ceilf((x->dims[i + 2] - ((kernels[i] - 1) * dilations[i] + 1) + 1) / (float)strides[i]);
                break;
            default:
                break;
            }
        }
        if (!y->reshape(dims, x->type))
            return false;
        // MaxPool is quantization-transparent: propagate quant metadata
        if (x->is_quantized())
            y->set_quant(x->quant_scale, x->quant_zero_point);
        if (outputs.size() > 1 && outputs[1]) {
            if (!outputs[1]->reshape(dims, NNR_DATA_TYPE_INT64))
                return false;
        }
        // NHWC/NCHWc support for 4D without indices
        if (x->ndim == 4
            && (x->type == NNR_DATA_TYPE_FLOAT32 || x->type == NNR_DATA_TYPE_UINT8)
            && dilations_are_one()
            && !(outputs.size() > 1 && outputs[1])) {
            layout_mask = LAYOUT_NCHW | LAYOUT_NHWC;
#ifdef NNR_ARCH_X64
            // NCHWc: channels must be multiple of 16
            if (x->dims[1] % 16 == 0 && x->dims[1] >= 16)
                layout_mask |= LAYOUT_BLOCKED_16;
#endif
        } else
            layout_mask = LAYOUT_NCHW;
        return true;
    }

    bool dilations_are_one() const {
        for (int d : dilations) if (d != 1) return false;
        return true;
    }

    float layout_cost(memory_layout_t layout, bool /*input_nhwc*/) const override {
        auto* x = inputs[0];
        auto* y = outputs.empty() ? nullptr : outputs[0];
        if (!x || !y || x->ndim != 4 || y->ndim != 4) return 0;
        int C = x->dims[1], H = x->dims[2], W = x->dims[3];
        float bytes = (float)C * H * W * 4 + (float)y->dims[1] * y->dims[2] * y->dims[3] * 4;
        if (layout == memory_layout_t::NHWC)
            return bytes / nhwc_patch_util(C);
        return bytes;
    }

    small_vector<op_cost_t, 8> estimate_costs(bool /*input_nhwc*/) const override {
        small_vector<op_cost_t, 8> out;
        auto* x = inputs[0];
        auto* y = outputs.empty() ? nullptr : outputs[0];
        if (!x || !y || x->ndim != 4 || y->ndim != 4) return out;
        int C = x->dims[1], H = x->dims[2], W = x->dims[3];
        int oH = y->dims[2], oW = y->dims[3];
        float in_bytes = (float)C * H * W * 4;
        float out_bytes = (float)C * oH * oW * 4;
        int kH = kernels.empty() ? 1 : kernels[0];
        int kW = kernels.size() < 2 ? 1 : kernels[1];
        // NCHW
        op_cost_t nchw{};
        nchw.layout = memory_layout_t::NCHW;
        nchw.read_bytes = in_bytes;
        nchw.write_bytes = out_bytes;
        nchw.read_sequential = 1.0f;
        nchw.working_set_bytes = (float)kH * W * 4;
        nchw.max_threads = C * oH;
        nchw.scrollable = true;
        out.push_back(nchw);
        // NHWC
        op_cost_t nhwc{};
        nhwc.layout = memory_layout_t::NHWC;
        nhwc.read_bytes = in_bytes;
        nhwc.write_bytes = out_bytes;
        nhwc.read_sequential = nhwc_patch_util(C);
        nhwc.working_set_bytes = (float)kH * kW * C * 4;
        nhwc.max_threads = C * oH;
        nhwc.scrollable = true;
        out.push_back(nhwc);
        return out;
    }

    template <typename T>
    bool exec_2d() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        bool has_indices = (outputs.size() > 1 && outputs[1] && outputs[1]->ndata > 0);
        if constexpr (std::is_same_v<T, float>) {
            // NCHWc (BLOCKED_16) path
            if (x->format == memory_layout_t::BLOCKED_16
                && (layout_mask & LAYOUT_BLOCKED_16) && !has_indices) {
#ifdef NNR_ARCH_X64
                if (has_avx512()) {
                    // Fast path: 3x3 stride=2 pad=0 (adv_inception_v3's
                    // 4 reduction MaxPool nodes all use this shape).
                    if (kernels[0] == 3 && kernels[1] == 3
                        && strides[0] == 2 && strides[1] == 2
                        && cpads[0] == 0 && cpads[1] == 0
                        && cpads[2] == 0 && cpads[3] == 0) {
                        maxpool_2d_nchwc_3x3s2p0_x64(
                            (const float*)x->data, (float*)y->data,
                            x->dims[0], x->dims[1], x->dims[2], x->dims[3],
                            y->dims[2], y->dims[3]);
                    } else {
                    maxpool_2d_nchwc_x64((const float*)x->data, (float*)y->data,
                        x->dims[0], x->dims[1], x->dims[2], x->dims[3],
                        y->dims[2], y->dims[3],
                        kernels[0], kernels[1], strides[0], strides[1], cpads[0], cpads[1]);
                    }
                } else
#endif
                {
                    maxpool_2d_nchwc((const float*)x->data, (float*)y->data,
                        x->dims[0], x->dims[1], x->dims[2], x->dims[3],
                        y->dims[2], y->dims[3],
                        kernels[0], kernels[1], strides[0], strides[1], cpads[0], cpads[1], 16);
                }
                y->format = memory_layout_t::BLOCKED_16;
                return true;
            }
            // NHWC path
            if (x->format == memory_layout_t::NHWC && !has_indices) {
#ifdef NNR_ARCH_X64
                maxpool_2d_nhwc_x64((const float*)x->data, (float*)y->data,
                    x->dims[0], x->dims[1], x->dims[2], x->dims[3],
                    y->dims[2], y->dims[3],
                    kernels[0], kernels[1], strides[0], strides[1], cpads[0], cpads[1]);
#elifdef NNR_ARCH_ARM64
                neon::maxpool_2d_nhwc_neon((const float*)x->data, (float*)y->data,
                    x->dims[0], x->dims[1], x->dims[2], x->dims[3],
                    y->dims[2], y->dims[3],
                    kernels[0], kernels[1], strides[0], strides[1], cpads[0], cpads[1]);
#else
                maxpool_2d_nhwc((const float*)x->data, (float*)y->data,
                    x->dims[0], x->dims[1], x->dims[2], x->dims[3],
                    y->dims[2], y->dims[3],
                    kernels[0], kernels[1], strides[0], strides[1], cpads[0], cpads[1]);
#endif
                y->format = memory_layout_t::NHWC;
                return true;
            }
        }
        // NHWC path for uint8 (used in quantized chains)
        if (x->format == memory_layout_t::NHWC && !has_indices) {
#ifdef NNR_ARCH_X64
            if constexpr (std::is_same_v<T, uint8_t>) {
                maxpool_2d_uint8_nhwc_x64((const uint8_t*)x->data, (uint8_t*)y->data,
                    x->dims[0], x->dims[1], x->dims[2], x->dims[3],
                    y->dims[2], y->dims[3],
                    kernels[0], kernels[1], strides[0], strides[1], cpads[0], cpads[1]);
                y->format = memory_layout_t::NHWC;
                return true;
            }
#endif
            maxpool_2d_nhwc((const T*)x->data, (T*)y->data,
                x->dims[0], x->dims[1], x->dims[2], x->dims[3],
                y->dims[2], y->dims[3],
                kernels[0], kernels[1], strides[0], strides[1], cpads[0], cpads[1]);
            y->format = memory_layout_t::NHWC;
            return true;
        }
        int64_t* pi = has_indices ? (int64_t*)outputs[1]->data : nullptr;
        maxpool_2d((const T*)x->data, (T*)y->data, pi,
            x->dims[0], x->dims[1], x->dims[2], x->dims[3],
            y->dims[2], y->dims[3],
            kernels[0], kernels[1], strides[0], strides[1], cpads[0], cpads[1]);
        return true;
    }

    template <typename T>
    bool exec() {
        if (inputs[0]->ndim == 4 && dilations_are_one() && storage_order == 0)
            return exec_2d<T>();
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;
        bool has_indices = (outputs.size() > 1 && outputs[1] && outputs[1]->ndata > 0);
        int64_t* pi = has_indices ? (int64_t*)outputs[1]->data : nullptr;
        small_vector<int> k_dim(x->ndim - 2);
        small_vector<int> i_dim(x->ndim);
        small_vector<int> o_dim(x->ndim);
        small_vector<int> b_dim(x->ndim);
        int i;

        do {
            for (i = 2; i < x->ndim; ++i) {
                b_dim[i] = o_dim[i] * strides[i - 2] - cpads[i - 2];
            }
            T maxv = std::numeric_limits<T>::lowest();
            int64_t max_idx = 0;
            std::fill(k_dim.begin(), k_dim.end(), 0);
            do {
                i_dim[0] = o_dim[0];
                i_dim[1] = o_dim[1];
                for (i = 2; i < x->ndim; ++i) {
                    i_dim[i] = b_dim[i] + k_dim[i - 2] * dilations[i - 2];
                }
                for (i = 0; i < x->ndim; ++i) {
                    if ((i_dim[i] < 0) || (i_dim[i] >= x->dims[i])) {
                        break;
                    }
                }
                if (i >= x->ndim) {
                    T v = px[dim_offset(i_dim, x->dim_span())];
                    if (v > maxv) {
                        maxv = v;
                        if (storage_order == 0) {
                            max_idx = dim_offset(i_dim, x->dim_span());
                        } else {
                            // Column-major: N,C dims use row-major, spatial dims use column-major
                            int64_t idx = 0;
                            int64_t stride = 1;
                            // Spatial dims in forward order (column-major = innermost dim first)
                            for (int d = 2; d < x->ndim; ++d) {
                                idx += i_dim[d] * stride;
                                stride *= x->dims[d];
                            }
                            // N, C dims
                            idx += i_dim[1] * stride;
                            stride *= x->dims[1];
                            idx += i_dim[0] * stride;
                            max_idx = idx;
                        }
                    }
                }
            } while (dim_next(k_dim, kernels));
            int64_t out_off = dim_offset(o_dim, y->dim_span());
            py[out_off] = maxv;
            if (pi) pi[out_off] = max_idx;
        } while (dim_next(o_dim, y->dim_span()));
        return true;
    }

    scroll_info_t scroll_info() const override {
        if (kernels.size() != 2) return {};
        // MaxPool with indices output is not scrollable (indices are global)
        if (outputs.size() > 1 && outputs[1] && outputs[1]->ndata > 0) return {};
        int kH = kernels[0];
        int dH = (dilations.size() > 0) ? dilations[0] : 1;
        int sH = (strides.size() > 0) ? strides[0] : 1;
        int pad_top = cpads[0];
        int eff_kH = (kH - 1) * dH + 1;
        return {
            .scrollable = true,
            .halo_top = pad_top,
            .halo_bottom = eff_kH - 1 - pad_top,
            .stride_h = sH,
        };
    }

    bool exec_strip(int out_row_start, int out_rows,
        int in_row_start, int in_rows) override
    {
        if (inputs[0]->ndim != 4 || !dilations_are_one()) return false;
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const float* px = (const float*)x->data;
        float* py = (float*)y->data;
        int N = x->dims[0], C = x->dims[1];
        int iH = x->dims[2], iW = x->dims[3];
        int oH = y->dims[2], oW = y->dims[3];
        int kH = kernels[0], kW = kernels[1];
        int sH = strides[0], sW = strides[1];
        int pH = cpads[0], pW = cpads[1];

        // Ring buffer: dims[2] may be ring_H, use orig_H for bounds
        int iH_pad = ring_in.orig_H > 0 ? ring_in.orig_H : iH;
        int oH_clamp = ring_out.orig_H > 0 ? ring_out.orig_H : oH;

        int NC = N * C;
        int out_end = std::min(out_row_start + out_rows, oH_clamp);
#ifdef NNR_ARCH_ARM64
        neon::maxpool_2d_strip_neon(px, py, NC, iH, iW, oH, oW,
            kH, kW, sH, sW, pH, pW, iH_pad, oH_clamp, out_row_start, out_end);
#else
        nnr::for_static(0, NC, nnr::compute_threads(NC), [&](int nc) {
            const float* inp = px + (size_t)nc * iH * iW;
            float* out = py + (size_t)nc * oH * oW;
            for (int oh = out_row_start; oh < out_end; ++oh) {
                int ih0 = oh * sH - pH;
                int kh0 = std::max(0, -ih0), kh1 = std::min(kH, iH_pad - ih0);
                for (int ow = 0; ow < oW; ++ow) {
                    int iw0 = ow * sW - pW;
                    int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
                    float maxv = -FLT_MAX;
                    for (int kh = kh0; kh < kh1; ++kh)
                        for (int kw = kw0; kw < kw1; ++kw) {
                            float v = inp[(ih0 + kh) * iW + (iw0 + kw)];
                            if (v > maxv) maxv = v;
                        }
                    out[oh * oW + ow] = maxv;
                }
            }
        });
#endif
        return true;
    }

    bool exec() override {
        return typed_exec<MaxPool_operator,
            opset_t<12, int8_t, uint8_t, float16_t, float, double>,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }

};

} // namespace {

// @nnr-meta-op mt=static layout=[NCHW,NHWC,BLOCKED_16] scroll=yes
operator_t* resolver_default_op_MaxPool(int opset, pool_t& pool) { return pool_new<MaxPool_operator>(pool); }

} // namespace nnr
