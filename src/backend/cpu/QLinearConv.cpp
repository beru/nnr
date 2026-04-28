#include "nnr.h"
#include "aligned_alloc.h"
#include "util.h"
#include "conv_shape.h"
#include "kernel/conv.h"
#include "kernel/f16_convert.h"
#include "cpu_features.h"
#include <cmath>
#include <algorithm>
#ifdef NNR_ARCH_X64
#include <immintrin.h>
#include "backend/x64/gemm_int8_avx512.h"
#include "backend/x64/conv_int8_nhwc_direct.h"
#include "backend/x64/conv_int8_direct_avx512.h"
#include "backend/x64/conv_first_layer_int8_avx512.h"
#include "backend/x64/depthwise_int8_avx512.h"
#include "backend/x64/transpose_nchw_nhwc_u8.h"
#elifdef NNR_ARCH_ARM64
#include "backend/arm/gemm_int8_neon.h"
#include "backend/arm/conv_int8_direct_neon.h"
#include "backend/arm/conv_int8_nhwc_direct_neon.h"
#include "backend/arm/depthwise_int8_neon.h"
#endif

namespace nnr {

namespace {


enum auto_pad_t {
    NOTSET,
    SAME_UPPER,
    SAME_LOWER,
    VALID,
};

struct QLinearConv_operator : public operator_t {
    auto_pad_t auto_pad = NOTSET;
    int group = 1;
    small_vector<int> kernels;
    small_vector<int> dilations;
    small_vector<int, MAX_NDIM * 2> pads;

    // VNNI-packed weights: W shifted from int8→uint8 (+128).
    std::vector<uint8_t> w_vnni_buf;     // W shifted uint8 [M, CHW]
    std::vector<int32_t> w_row_sums_buf; // row sums of shifted W, per output channel

    // Pre-packed FP32 weights for dgemm_packed_a fast path
    std::vector<float> w_packed_f32;     // pack_a format (dequantized float32)

    // Direct int8 conv (VNNI, no im2col)
    std::vector<int8_t> w_direct_buf;    // VNNI-packed weights for direct conv
    std::vector<int32_t> w_direct_sums;  // per-OC weight sums for zero-point compensation
#ifdef NNR_ARCH_ARM64
    // Separate SMMLA-packed weights for the NHWC i8mm fast path. SMMLA uses a
    // different tile layout (2-OC × 8-IC) than the SDOT pack above, so both
    // coexist when has_neon_i8mm() — SDOT still services the NCHW exec path.
    std::vector<int8_t> w_direct_smmla_buf;
#endif

    // First-layer int8 direct conv (IC<=4, e.g. RGB stem): [KH, KW, OC_blocks, 16, 4]
    std::vector<int8_t> w_first_layer_int8;      // VNNI-packed first-layer weights
    std::vector<int32_t> w_first_layer_int8_sums; // per-OC weight sums

    // Gather-GEMM: pre-packed weights in NR=48-blocked VNNI format (int8, no shift)
    std::vector<int8_t> w_gather_packed;    // NR-blocked VNNI weights
    std::vector<int32_t> w_gather_col_sums; // [OC] signed weight column sums

    // Fused im2col k_off tables (pre-computed at reshape)
    std::vector<size_t> k_off_base;     // [K4] oh-independent offsets
    std::vector<size_t> k_off_oh_all;   // [oH × K4] oh-expanded offsets

    // Pre-allocated work buffers (avoid per-inference nnr_aligned_alloc)
    std::vector<uint8_t> x_pad_buf;     // padded input (pre-filled with x_zp)
    std::vector<int32_t> y_i32_buf;     // int32 GEMM output tile
    uint8_t x_pad_zp = 0;              // x_zp value used to pre-fill x_pad_buf

    // Packed NR=48 GEMM (ORT-style: A=input, B=weights): sub-stride format
    std::vector<int8_t> w_packed_nr48;      // sub-stride NR=48 format, (kh,kw,ic) K-order
    std::vector<int32_t> w_packed_nr48_col_sums; // [OC] signed weight column sums
    std::vector<uint8_t> x_pad_nhwc_buf;    // NHWC padded input for packed NR=48 path
    std::vector<uint8_t> y_nhwc_buf;         // temp NHWC output for packed NR=48 when output is NCHW

    // NHWC-direct NR=16 path (memcpy-free): pre-padded NHWC input.
    // Sized in reshape only when the dispatch predicate matches.

    // Depthwise int8 conv (NHWC, indirection-based)
    std::vector<int8_t> w_dw_repacked;       // [kHW, C_pad] repacked depthwise weights
    std::vector<const uint8_t*> dw_ind_buf;  // [oH*oW*kHW] indirection pointers
    std::vector<uint8_t> dw_zero_buf;        // [C_pad] zero buffer filled with x_zp

    // Pre-computed requantize params (avoid per-inference heap alloc)
    std::vector<float> rq_output_scales;    // [M] = (x_scale * w_scale[oc]) / y_scale
    std::vector<float> rq_combined_scales;  // [M] = x_scale * w_scale[oc]
    std::vector<float> rq_bias_f;           // [M] = bias[oc] * combined_scale[oc] (empty if no bias)
#ifdef NNR_ARCH_X64
    int8::conv_rq_params_t rq_cached;       // pre-filled constant fields (x64 int8 JIT kernels only)
#endif

    small_vector<int> strides;
    int cpads[32] = {0};

    // M6 placeholder: NHWC channel-axis Concat alias with strided output
    // requires LDC support in the int8 NR=48 packed/direct/gather kernels
    // (see .claude/kb/nhwc_concat_strided.md). Until those land, return false
    // so memory_planner won't alias int8 producers. When the int8 LDC kernels
    // ship, flip this override.
    bool supports_strided_output(memory_layout_t /*format*/) const override {
        return false;
    }

    bool init() override {
        if (!(inputs.size() >= 8 && inputs.size() <= 9 && outputs.size() == 1))
            return false;

        int64_t* ints = nullptr;
        int i, l;

        auto_pad = string2enum(attribute(attr_key_t::auto_pad, "NOTSET"), NOTSET);
        group = attribute(attr_key_t::group, 1);

        int nk = attribute(attr_key_t::kernel_shape, ints);
        if (nk > 0) {
            kernels.resize(nk);
            for (i = 0; i < kernels.size(); ++i)
                kernels[i] = ints[i];
        } else {
            // Infer from weight shape: w is input[3], spatial dims start at index 2
            const tensor_t* w = inputs[3];
            int sdims = w->ndim - 2;
            kernels.resize(sdims);
            for (i = 0; i < sdims; ++i)
                kernels[i] = w->dims[i + 2];
        }

        dilations.resize(kernels.size());
        l = attribute(attr_key_t::dilations, ints);
        for (i = 0; i < l; ++i)
            dilations[i] = ints[i];
        for (; i < dilations.size(); ++i)
            dilations[i] = 1;

        pads.resize(kernels.size() * 2);
        l = attribute(attr_key_t::pads, ints);
        for (i = 0; i < l; ++i)
            pads[i] = ints[i];
        for (; i < pads.size(); ++i)
            pads[i] = 0;

        strides.resize(kernels.size());
        l = attribute(attr_key_t::strides, ints);
        for (i = 0; i < l; ++i)
            strides[i] = ints[i];
        for (; i < strides.size(); ++i)
            strides[i] = 1;

        // Check for per-channel weight zero-point (limits VNNI eligibility).
        // If all per-channel zero-points are identical, treat as per-tensor.
        per_channel_zp_ = false;
        if (inputs.size() > 5 && inputs[5] && inputs[5]->ndata > 1) {
            per_channel_zp_ = true;
            const auto* zp = inputs[5];
            bool all_same = true;
            if (zp->type == NNR_DATA_TYPE_INT8) {
                int8_t v0 = ((const int8_t*)zp->data)[0];
                for (size_t i = 1; i < zp->ndata && all_same; i++)
                    all_same = (((const int8_t*)zp->data)[i] == v0);
            } else {
                uint8_t v0 = ((const uint8_t*)zp->data)[0];
                for (size_t i = 1; i < zp->ndata && all_same; i++)
                    all_same = (((const uint8_t*)zp->data)[i] == v0);
            }
            if (all_same) per_channel_zp_ = false;
        }

        return true;
    }

#include "QLinearConv_reshape.h"

    // Track per-channel zero-point flag for VNNI eligibility
    bool per_channel_zp_ = false;

    float layout_cost(memory_layout_t layout, bool /*input_nhwc*/) const override {
        if (inputs.size() < 4 || !inputs[0] || !inputs[3]) return 0;
        auto* x = inputs[0];
        auto* w = inputs[3];
        if (x->ndim != 4 || w->ndim != 4) return 0;
        int C = x->dims[1], H = x->dims[2], W = x->dims[3];
        int M = w->dims[0], K = w->dims[1] * w->dims[2] * w->dims[3];
        float data_bytes = (float)(C * H * W + M * K + M * H * W);
        bool is_depthwise = (group == M && w->dims[1] == 1);
        if (layout == memory_layout_t::NHWC) {
            if (is_depthwise)
                return data_bytes * 0.1f;  // depthwise: NHWC eliminates im2col, 16ch vectorization
            return data_bytes * 0.5f;      // regular: gather eliminates shuffles
        }
        if (layout == memory_layout_t::BLOCKED_16 || layout == memory_layout_t::BLOCKED_8) {
            // TODO(T1+): real int8 BLOCKED cost. NR=16/48 packed kernel
            // family has different SIMD utilization than fp32 NCHWc and the
            // NHWC-direct NR=16 path often beats it. 0.85 is a placeholder
            // that captures the typical NCHWc-int8-vs-NCHW-int8 advantage at
            // channel-aligned shapes; not honest for IC-tail or small-K.
            return data_bytes * 0.85f;
        }
        return data_bytes;                 // NCHW baseline
    }

    template <typename T>
    bool exec_typed() {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        const tensor_t* x_scale_t = inputs[1];
        const tensor_t* x_zp_t = inputs[2];
        const tensor_t* w = inputs[3];
        const tensor_t* w_scale_t = inputs[4];
        const tensor_t* w_zp_t = inputs[5];
        const tensor_t* y_scale_t = inputs[6];
        const tensor_t* y_zp_t = inputs[7];
        const tensor_t* bias_t = (inputs.size() > 8) ? inputs[8] : nullptr;

        float x_scale = *(float*)x_scale_t->data;
        T x_zp = (x_zp_t->ndata > 0) ? ((T*)x_zp_t->data)[0] : 0;
        float y_scale = *(float*)y_scale_t->data;
        T y_zp = (y_zp_t->ndata > 0) ? ((T*)y_zp_t->data)[0] : 0;

        // w_scale and w_zp can be per-channel (one per output channel M) or scalar
        const float* w_scale_data = (float*)w_scale_t->data;
        const T* w_zp_data = (T*)w_zp_t->data;
        int w_scale_count = (int)w_scale_t->ndata;
        int w_zp_count = (int)w_zp_t->ndata;
        bool per_channel_scale = (w_scale_count > 1);
        bool per_channel_zp = per_channel_zp_;  // use member with "all-same" optimization

        const T* px = (const T*)x->data;
        const T* pw = (const T*)w->data;
        T* py = (T*)y->data;
        const int32_t* pbias = bias_t ? (const int32_t*)bias_t->data : nullptr;

        const int ndim = x->ndim;

        if (ndim == 4) {
            int iN = x->dims[0];
            int iC = x->dims[1];
            int iH = x->dims[2];
            int iW = x->dims[3];

            int M = w->dims[0];   // output channels
            int C = w->dims[1];   // input channels per group
            int kH = w->dims[2];
            int kW = w->dims[3];

            int oN = y->dims[0];
            int oC = y->dims[1];
            int oH = y->dims[2];
            int oW = y->dims[3];

            int MM = M / group;
            int CC = iC / group;

            // Clamp range depends on type
            int clamp_min, clamp_max;
            if constexpr (std::is_same_v<T, uint8_t>) {
                clamp_min = 0;
                clamp_max = 255;
            } else {
                clamp_min = -128;
                clamp_max = 127;
            }

            for (int n = 0; n < oN; ++n) {
                for (int g = 0; g < group; ++g) {
                    for (int m = 0; m < MM; ++m) {
                        int oc = g * MM + m;
                        float ws = per_channel_scale ? w_scale_data[oc] : w_scale_data[0];
                        int32_t wzp = per_channel_zp ? (int32_t)w_zp_data[oc] : (w_zp_count > 0 ? (int32_t)w_zp_data[0] : 0);
                        float combined_scale = x_scale * ws;
                        float bias_val = pbias ? (float)pbias[oc] * combined_scale : 0.0f;

                        for (int oh = 0; oh < oH; ++oh) {
                            for (int ow = 0; ow < oW; ++ow) {
                                int32_t sum = 0;
                                for (int c = 0; c < C; ++c) {
                                    int ic = g * CC + c;
                                    for (int kh = 0; kh < kH; ++kh) {
                                        int ih = oh * strides[0] - cpads[0] + kh * dilations[0];
                                        if (ih < 0 || ih >= iH)
                                            continue;
                                        for (int kw = 0; kw < kW; ++kw) {
                                            int iw = ow * strides[1] - cpads[1] + kw * dilations[1];
                                            if (iw < 0 || iw >= iW)
                                                continue;
                                            int32_t xv = (int32_t)px[((n * iC + ic) * iH + ih) * iW + iw] - (int32_t)x_zp;
                                            int32_t wv = (int32_t)pw[((oc * C + c) * kH + kh) * kW + kw] - wzp;
                                            sum += xv * wv;
                                        }
                                    }
                                }
                                float result = (float)sum * combined_scale + bias_val;
                                int32_t quantized = (int32_t)std::nearbyint(result / y_scale) + (int32_t)y_zp;
                                quantized = std::clamp(quantized, clamp_min, clamp_max);
                                py[((n * oC + oc) * oH + oh) * oW + ow] = (T)quantized;
                            }
                        }
                    }
                }
            }
        } else if (ndim == 3) {
            // 1D conv
            int iN = x->dims[0];
            int iC = x->dims[1];
            int iW = x->dims[2];

            int M = w->dims[0];
            int C = w->dims[1];
            int kW = w->dims[2];

            int oN = y->dims[0];
            int oC = y->dims[1];
            int oW = y->dims[2];

            int MM = M / group;
            int CC = iC / group;

            int clamp_min, clamp_max;
            if constexpr (std::is_same_v<T, uint8_t>) {
                clamp_min = 0;
                clamp_max = 255;
            } else {
                clamp_min = -128;
                clamp_max = 127;
            }

            for (int n = 0; n < oN; ++n) {
                for (int g = 0; g < group; ++g) {
                    for (int m = 0; m < MM; ++m) {
                        int oc = g * MM + m;
                        float ws = per_channel_scale ? w_scale_data[oc] : w_scale_data[0];
                        int32_t wzp = per_channel_zp ? (int32_t)w_zp_data[oc] : (w_zp_count > 0 ? (int32_t)w_zp_data[0] : 0);
                        float combined_scale = x_scale * ws;
                        float bias_val = pbias ? (float)pbias[oc] * combined_scale : 0.0f;

                        for (int ow = 0; ow < oW; ++ow) {
                            int32_t sum = 0;
                            for (int c = 0; c < C; ++c) {
                                int ic = g * CC + c;
                                for (int kw = 0; kw < kW; ++kw) {
                                    int iw = ow * strides[0] - cpads[0] + kw * dilations[0];
                                    if (iw < 0 || iw >= iW)
                                        continue;
                                    int32_t xv = (int32_t)px[(n * iC + ic) * iW + iw] - (int32_t)x_zp;
                                    int32_t wv = (int32_t)pw[(oc * C + c) * kW + kw] - wzp;
                                    sum += xv * wv;
                                }
                            }
                            float result = (float)sum * combined_scale + bias_val;
                            int32_t quantized = (int32_t)std::nearbyint(result / y_scale) + (int32_t)y_zp;
                            quantized = std::clamp(quantized, clamp_min, clamp_max);
                            py[(n * oC + oc) * oW + ow] = (T)quantized;
                        }
                    }
                }
            }
        }
        return true;
    }

    // SIMD-accelerated 4D path: convert int8→float, im2col + float GEMM, requantize.
    // Much faster than scalar nested loops — reuses optimized float GEMM kernels.
    // QLinearConv::exec_simd_4d() helpers --------------------------------
    // Bundle of per-call locals so each branch helper can be written with
    // the original identifier names. Populated by init_qlinear4d_vars()
    // at the top of exec_simd_4d(), passed by const reference to helpers.
    template <typename T>
    struct qlinear4d_vars_t {
        const tensor_t* x;
        const tensor_t* w;
        tensor_t* y;
        float x_scale;
        T x_zp;
        float y_scale;
        T y_zp;
        const float* w_scale_data;
        const T* w_zp_data;
        int w_scale_count;
        int w_zp_count;
        bool per_channel_scale;
        bool per_channel_zp;
        const int32_t* pbias;
        int iN;
        int iC;
        int iH;
        int iW;
        int M;
        int C;
        int kH;
        int kW;
        int oN;
        int oH;
        int oW;
        int MM;
        int CC;
        int CHW;
        int spatial;
        int sH;
        int sW;
        int dH;
        int dW;
        int pH;
        int pW;
        int clamp_min;
        int clamp_max;
        int qmin;
        float inv_y_scale;
    };

    template <typename T>
    bool init_qlinear4d_vars(qlinear4d_vars_t<T>& v) {
        v.x = inputs[0];
        v.w = inputs[3];
        v.y = outputs[0];
        if (v.x->ndim != 4) return false;

        v.x_scale = *(float*)inputs[1]->data;
        v.x_zp = (inputs[2]->ndata > 0) ? ((T*)inputs[2]->data)[0] : (T)0;
        v.y_scale = *(float*)inputs[6]->data;
        v.y_zp = (inputs[7]->ndata > 0) ? ((T*)inputs[7]->data)[0] : (T)0;
        v.w_scale_data = (float*)inputs[4]->data;
        v.w_zp_data = (T*)inputs[5]->data;
        v.w_scale_count = (int)inputs[4]->ndata;
        v.w_zp_count = (int)inputs[5]->ndata;
        v.per_channel_scale = (v.w_scale_count > 1);
        v.per_channel_zp = per_channel_zp_;
        v.pbias = (inputs.size() > 8 && inputs[8]) ? (const int32_t*)inputs[8]->data : nullptr;

        v.iN = v.x->dims[0]; v.iC = v.x->dims[1]; v.iH = v.x->dims[2]; v.iW = v.x->dims[3];
        v.M = v.w->dims[0]; v.C = v.w->dims[1]; v.kH = v.w->dims[2]; v.kW = v.w->dims[3];
        v.oN = v.y->dims[0]; v.oH = v.y->dims[2]; v.oW = v.y->dims[3];
        v.MM = v.M / group; v.CC = v.iC / group;
        v.CHW = v.C * v.kH * v.kW;
        v.spatial = v.oH * v.oW;
        v.sH = strides[0]; v.sW = strides[1];
        v.dH = dilations[0]; v.dW = dilations[1];
        v.pH = cpads[0]; v.pW = cpads[1];

        v.clamp_min = std::is_same_v<T, uint8_t> ? 0 : -128;
        v.clamp_max = std::is_same_v<T, uint8_t> ? 255 : 127;
        v.qmin = post_fn ? std::max(v.clamp_min, (int)v.y_zp) : v.clamp_min;
        v.inv_y_scale = 1.0f / v.y_scale;
        return true;
    }

#ifdef NNR_ARCH_X64
    template <typename T>
    bool exec_dw_int8_nhwc(const qlinear4d_vars_t<T>& v) {
        auto& x = v.x;
        auto& w = v.w;
        auto& y = v.y;
        auto& x_scale = v.x_scale;
        auto& x_zp = v.x_zp;
        auto& y_scale = v.y_scale;
        auto& y_zp = v.y_zp;
        auto& w_scale_data = v.w_scale_data;
        auto& w_zp_data = v.w_zp_data;
        auto& w_scale_count = v.w_scale_count;
        auto& w_zp_count = v.w_zp_count;
        auto& per_channel_scale = v.per_channel_scale;
        auto& per_channel_zp = v.per_channel_zp;
        auto& pbias = v.pbias;
        auto& iN = v.iN;
        auto& iC = v.iC;
        auto& iH = v.iH;
        auto& iW = v.iW;
        auto& M = v.M;
        auto& C = v.C;
        auto& kH = v.kH;
        auto& kW = v.kW;
        auto& oN = v.oN;
        auto& oH = v.oH;
        auto& oW = v.oW;
        auto& MM = v.MM;
        auto& CC = v.CC;
        auto& CHW = v.CHW;
        auto& spatial = v.spatial;
        auto& sH = v.sH;
        auto& sW = v.sW;
        auto& dH = v.dH;
        auto& dW = v.dW;
        auto& pH = v.pH;
        auto& pW = v.pW;
        auto& clamp_min = v.clamp_min;
        auto& clamp_max = v.clamp_max;
        auto& qmin = v.qmin;
        auto& inv_y_scale = v.inv_y_scale;
        (void)x;(void)w;(void)y;(void)iN;(void)x_scale;(void)y_scale;

        if (!w_dw_repacked.empty() && std::is_same_v<T, uint8_t>
            && y->format == memory_layout_t::NHWC) {
            int32_t w_zp_scalar = (w_zp_count > 0) ? (int32_t)w_zp_data[0] : 0;
            int kHW = kH * kW;

            // Fill zero buffer with x_zp (for padding pixels)
            memset(dw_zero_buf.data(), (uint8_t)x_zp, dw_zero_buf.size());

            for (int n = 0; n < oN; n++) {
                const uint8_t* x_nhwc;
                if (x->format == memory_layout_t::NHWC) {
                    x_nhwc = (const uint8_t*)x->data + (size_t)n * iH * iW * iC;
                } else {
                    // NCHW→NHWC transpose into reusable buffer
                    if (x_pad_nhwc_buf.empty())
                        x_pad_nhwc_buf.resize((size_t)iH * iW * iC + 64);
                    uint8_t* buf = (uint8_t*)(((uintptr_t)x_pad_nhwc_buf.data() + 63) & ~63);
                    for (int ic = 0; ic < iC; ic++) {
                        const uint8_t* src = (const uint8_t*)x->data
                            + ((size_t)n * iC + ic) * iH * iW;
                        for (int h = 0; h < iH; h++)
                            for (int w = 0; w < iW; w++)
                                buf[((size_t)h * iW + w) * iC + ic] = src[h * iW + w];
                    }
                    x_nhwc = buf;
                }

                T* py = (T*)y->data + (size_t)n * spatial * M;

                // Direct-addressing kernel: no indirection buffer needed.
                // Requires dilation==1; falls back to indirection-based for dilated.
                if (dH == 1 && dW == 1) {
                    int8::depthwise_int8_nhwc_avx512_direct(
                        (uint8_t*)py, x_nhwc,
                        w_dw_repacked.data(),
                        rq_combined_scales.data(),
                        rq_bias_f.empty() ? nullptr : rq_bias_f.data(),
                        dw_zero_buf.data(),
                        iC, oH, oW, iH, iW,
                        kH, kW, sH, sW, pH, pW,
                        (int)x_zp, w_zp_scalar,
                        inv_y_scale, (float)y_zp, (float)clamp_min, (float)clamp_max);
                } else {
                    // Indirection-based fallback for dilated depthwise conv.
                    int8::build_depthwise_indirection(
                        dw_ind_buf.data(), x_nhwc, dw_zero_buf.data(),
                        oH, oW, iH, iW, iC,
                        kH, kW, sH, sW, pH, pW, dH, dW);
                    int8::depthwise_int8_nhwc_avx512_mt(
                        (uint8_t*)py, dw_ind_buf.data(),
                        w_dw_repacked.data(),
                        rq_combined_scales.data(),
                        rq_bias_f.empty() ? nullptr : rq_bias_f.data(),
                        iC, oH, oW, kHW,
                        (int)x_zp, w_zp_scalar,
                        inv_y_scale, (float)y_zp, (float)clamp_min, (float)clamp_max);
                }
            }
            y->format = memory_layout_t::NHWC;
            return true;
        }

        return false; // no branch matched
    }

#endif

#ifdef NNR_ARCH_X64
    template <typename T>
    bool exec_direct_int8(const qlinear4d_vars_t<T>& v) {
        auto& x = v.x;
        auto& w = v.w;
        auto& y = v.y;
        auto& x_scale = v.x_scale;
        auto& x_zp = v.x_zp;
        auto& y_scale = v.y_scale;
        auto& y_zp = v.y_zp;
        auto& w_scale_data = v.w_scale_data;
        auto& w_zp_data = v.w_zp_data;
        auto& w_scale_count = v.w_scale_count;
        auto& w_zp_count = v.w_zp_count;
        auto& per_channel_scale = v.per_channel_scale;
        auto& per_channel_zp = v.per_channel_zp;
        auto& pbias = v.pbias;
        auto& iN = v.iN;
        auto& iC = v.iC;
        auto& iH = v.iH;
        auto& iW = v.iW;
        auto& M = v.M;
        auto& C = v.C;
        auto& kH = v.kH;
        auto& kW = v.kW;
        auto& oN = v.oN;
        auto& oH = v.oH;
        auto& oW = v.oW;
        auto& MM = v.MM;
        auto& CC = v.CC;
        auto& CHW = v.CHW;
        auto& spatial = v.spatial;
        auto& sH = v.sH;
        auto& sW = v.sW;
        auto& dH = v.dH;
        auto& dW = v.dW;
        auto& pH = v.pH;
        auto& pW = v.pW;
        auto& clamp_min = v.clamp_min;
        auto& clamp_max = v.clamp_max;
        auto& qmin = v.qmin;
        auto& inv_y_scale = v.inv_y_scale;
        (void)x;(void)w;(void)y;(void)iN;(void)x_scale;(void)y_scale;

        if (!w_direct_buf.empty() && std::is_same_v<T, uint8_t>
            && !per_channel_zp
            && (w_zp_count == 0 || (int32_t)w_zp_data[0] == 0)
            && spatial <= 256
            && y->format != memory_layout_t::NHWC) {
            int IC4 = (iC + 3) / 4;
            int IC_padded = IC4 * 4;
            int padded_H = iH + pH + cpads[2];  // top + bottom padding
            int padded_W = iW + pW + cpads[3];  // left + right padding
            size_t pad_plane = (size_t)padded_H * padded_W;

            uint8_t* x_pad = (uint8_t*)nnr_aligned_alloc(IC_padded * pad_plane, 64);
            int32_t* y_i32 = (int32_t*)nnr_aligned_alloc((size_t)M * spatial * sizeof(int32_t), 64);

            for (int n = 0; n < oN; n++) {
                // Fill padded input: x_zp everywhere, then copy actual channels
                memset(x_pad, (uint8_t)x_zp, IC_padded * pad_plane);
                for (int ic = 0; ic < iC; ic++) {
                    const uint8_t* src = (const uint8_t*)x->data
                        + ((size_t)n * iC + ic) * iH * iW;
                    uint8_t* dst = x_pad + (size_t)ic * pad_plane
                        + pH * padded_W + pW;
                    for (int h = 0; h < iH; h++)
                        memcpy(dst + h * padded_W, src + h * iW, iW);
                }

                // Run direct conv kernel → raw int32 dot products
                conv_int8_direct_avx512(y_i32, x_pad, w_direct_buf.data(),
                    iC, padded_H, padded_W, M, oH, oW, kH, kW, sH, sW);

                // Zero-point compensation + requantize
                T* out_n = (T*)y->data + (size_t)n * M * spatial;
                __m512 vis = _mm512_set1_ps(inv_y_scale);
                __m512 vzp = _mm512_set1_ps((float)y_zp);
                __m512 vqmin = _mm512_set1_ps((float)qmin);
                __m512 vqmax = _mm512_set1_ps((float)clamp_max);

                for (int oc = 0; oc < M; oc++) {
                    float ws = per_channel_scale ? w_scale_data[oc] : w_scale_data[0];
                    float combined_scale = x_scale * ws;
                    float bias_val = pbias ? (float)pbias[oc] * combined_scale : 0.0f;
                    int32_t zp_comp = (int32_t)x_zp * w_direct_sums[oc];

                    __m512 vcs = _mm512_set1_ps(combined_scale);
                    __m512 vbias = _mm512_set1_ps(bias_val);
                    __m512i vzpc = _mm512_set1_epi32(zp_comp);

                    int32_t* irow = y_i32 + (size_t)oc * spatial;
                    T* out = out_n + (size_t)oc * spatial;
                    int s = 0;
                    for (; s + 16 <= spatial; s += 16) {
                        __m512i raw = _mm512_loadu_si512(irow + s);
                        __m512i comp = _mm512_sub_epi32(raw, vzpc);
                        __m512 fv = _mm512_cvtepi32_ps(comp);
                        fv = _mm512_fmadd_ps(fv, vcs, vbias);
                        fv = _mm512_add_ps(_mm512_roundscale_ps(
                            _mm512_mul_ps(fv, vis), _MM_FROUND_TO_NEAREST_INT), vzp);
                        fv = _mm512_max_ps(_mm512_min_ps(fv, vqmax), vqmin);
                        _mm_storeu_si128((__m128i*)(out + s),
                            _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(fv)));
                    }
                    for (; s < spatial; s++) {
                        int32_t raw_val = irow[s] - zp_comp;
                        float val = (float)raw_val * combined_scale + bias_val;
                        int32_t q = (int32_t)std::nearbyint(val * inv_y_scale) + (int32_t)y_zp;
                        out[s] = (T)std::clamp(q, qmin, clamp_max);
                    }
                }
            }

            nnr_aligned_free(x_pad);
            nnr_aligned_free(y_i32);
            return true;
        }

        return false; // no branch matched
    }

    // First-layer int8 direct conv (VPDPBUSD, no im2col).
    // Eligibility: uint8 input/output, int8 weights, symmetric w_zp, IC<=4,
    // group=1, dilation=1, NCHW output. Skips the packed GEMM path which
    // wastes 75% of VPDPBUSD lanes when IC=3 (dword K dim).
    template <typename T>
    bool exec_first_layer_int8(const qlinear4d_vars_t<T>& v) {
        if constexpr (!std::is_same_v<T, uint8_t>) {
            (void)v;
            return false;
        } else {
            if (w_first_layer_int8.empty()) return false;
            if (v.per_channel_zp) return false;
            if (v.iC > 4) return false;
            if (dilations[0] != 1 || dilations[1] != 1) return false;
            // Kernel assumes symmetric padding on each axis.
            if (cpads[0] != cpads[2] || cpads[1] != cpads[3]) return false;
            // Input must be NCHW (simple first-layer case).
            if (v.x->format == memory_layout_t::NHWC) return false;

            const bool out_nhwc = (v.y->format == memory_layout_t::NHWC);

            const int32_t qmin = v.qmin;
            const int32_t qmax = v.clamp_max;
            for (int n = 0; n < v.oN; n++) {
                // NHWC output y laid out as [n][oH][oW][OC], NCHW as [n][OC][oH*oW].
                // Byte count per batch is the same (M * spatial).
                uint8_t* yn = (uint8_t*)v.y->data + (size_t)n * v.M * v.spatial;
                const uint8_t* xn = (const uint8_t*)v.x->data
                    + (size_t)n * v.iC * v.iH * v.iW;
                bool ok = conv_first_layer_int8_avx512(
                    yn, xn, w_first_layer_int8.data(),
                    w_first_layer_int8_sums.data(),
                    v.pbias, v.w_scale_data, v.w_scale_count,
                    v.x_scale, v.inv_y_scale,
                    (int32_t)v.x_zp, (int32_t)v.y_zp, qmin, qmax,
                    v.iC, v.iH, v.iW, v.M, v.oH, v.oW,
                    v.kH, v.kW, v.sH, v.sW, v.pH, v.pW,
                    out_nhwc);
                if (!ok) return false;
            }
            if (out_nhwc) v.y->format = memory_layout_t::NHWC;
            return true;
        }
    }

#endif

#ifdef NNR_ARCH_ARM64
    // NEON depthwise int8 exec — indirection-based (no direct-addressing variant yet on ARM).
    // Parallel to exec_dw_int8_nhwc (x64) but uses the `_neon` suffixed pack/build/compute
    // functions and C_pad = 4-aligned (vs 16-aligned on AVX-512).
    template <typename T>
    bool exec_dw_int8_nhwc_neon(const qlinear4d_vars_t<T>& v) {
        if constexpr (!std::is_same_v<T, uint8_t>) { (void)v; return false; }
        else {
            if (w_dw_repacked.empty()) return false;
            if (v.y->format != memory_layout_t::NHWC) return false;
            if (v.per_channel_zp) return false;
            if (!has_neon_dotprod()) return false;

            int32_t w_zp_scalar = (v.w_zp_count > 0) ? (int32_t)v.w_zp_data[0] : 0;
            int kHW = v.kH * v.kW;

            std::memset(dw_zero_buf.data(), (uint8_t)v.x_zp, dw_zero_buf.size());

            float qmin_f = (float)v.qmin;
            float qmax_f = (float)v.clamp_max;

            for (int n = 0; n < v.oN; n++) {
                const uint8_t* x_nhwc;
                if (v.x->format == memory_layout_t::NHWC) {
                    x_nhwc = (const uint8_t*)v.x->data + (size_t)n * v.iH * v.iW * v.iC;
                } else {
                    if (x_pad_nhwc_buf.empty())
                        x_pad_nhwc_buf.resize((size_t)v.iH * v.iW * v.iC + 64);
                    uint8_t* buf = (uint8_t*)(((uintptr_t)x_pad_nhwc_buf.data() + 63) & ~63);
                    for (int ic = 0; ic < v.iC; ic++) {
                        const uint8_t* src = (const uint8_t*)v.x->data
                            + ((size_t)n * v.iC + ic) * v.iH * v.iW;
                        for (int h = 0; h < v.iH; h++)
                            for (int ww = 0; ww < v.iW; ww++)
                                buf[((size_t)h * v.iW + ww) * v.iC + ic] = src[h * v.iW + ww];
                    }
                    x_nhwc = buf;
                }

                int8::neon::build_depthwise_indirection_neon(
                    dw_ind_buf.data(), x_nhwc, dw_zero_buf.data(),
                    v.oH, v.oW, v.iH, v.iW, v.iC,
                    v.kH, v.kW, v.sH, v.sW, v.pH, v.pW, v.dH, v.dW);

                T* py = (T*)v.y->data + (size_t)n * v.spatial * v.M;
                int8::neon::depthwise_int8_nhwc_neon_mt(
                    (uint8_t*)py, dw_ind_buf.data(),
                    w_dw_repacked.data(),
                    rq_combined_scales.data(),
                    rq_bias_f.empty() ? nullptr : rq_bias_f.data(),
                    v.iC, v.oH, v.oW, kHW,
                    (int)v.x_zp, w_zp_scalar,
                    v.inv_y_scale, (float)v.y_zp, qmin_f, qmax_f);
            }
            v.y->format = memory_layout_t::NHWC;
            return true;
        }
    }

    // NEON SDOT direct int8 conv. Parallel to exec_direct_int8 (x64) — same w_direct_buf/
    // w_direct_sums buffers, but packed via the ARM-specific layout in reshape.
    // On i8mm-capable chips, prefers the SMMLA path (conv_int8_direct_nchw_smmla), which
    // pre-transposes the NCHW activation to NHWC in a scratch buffer and runs the SMMLA
    // body with NCHW-layout stores — typically 3-4× faster than the SDOT path at C=64 56×56.
    template <typename T>
    bool exec_direct_int8_neon(const qlinear4d_vars_t<T>& v) {
        if constexpr (!std::is_same_v<T, uint8_t>) { (void)v; return false; }
        else {
            if (w_direct_buf.empty()) return false;
            if (v.per_channel_zp) return false;
            if (v.w_zp_count != 0 && (int32_t)v.w_zp_data[0] != 0) return false;
            if (v.spatial > 256) return false;
            if (v.y->format == memory_layout_t::NHWC) return false;
            if (!has_neon_dotprod()) return false;

  #if defined(__ARM_FEATURE_MATMUL_INT8) || (defined(_MSC_VER) && defined(_M_ARM64))
            const bool use_smmla = !w_direct_smmla_buf.empty() && has_neon_i8mm();
  #else
            const bool use_smmla = false;
  #endif

            int IC4 = (v.iC + 3) / 4;
            int IC_padded = IC4 * 4;  // x_pad stays NCHW-padded to IC4; SMMLA path internally
                                      // extends to IC8 inside its scratch buffer.
            int padded_H = v.iH + v.pH + cpads[2];
            int padded_W = v.iW + v.pW + cpads[3];
            size_t pad_plane = (size_t)padded_H * padded_W;

            std::vector<uint8_t> x_pad_vec((size_t)IC_padded * pad_plane);
            std::vector<int32_t> y_i32_vec((size_t)v.M * v.spatial);
            std::vector<uint8_t> x_nhwc_scratch;
  #if defined(__ARM_FEATURE_MATMUL_INT8) || (defined(_MSC_VER) && defined(_M_ARM64))
            if (use_smmla) {
                x_nhwc_scratch.resize(
                    conv_int8_direct_nchw_smmla_x_scratch_size(v.iC, padded_H, padded_W));
            }
  #endif
            uint8_t* x_pad = x_pad_vec.data();
            int32_t* y_i32 = y_i32_vec.data();

            for (int n = 0; n < v.oN; n++) {
                std::memset(x_pad, (uint8_t)v.x_zp, (size_t)IC_padded * pad_plane);
                for (int ic = 0; ic < v.iC; ic++) {
                    const uint8_t* src = (const uint8_t*)v.x->data
                        + ((size_t)n * v.iC + ic) * v.iH * v.iW;
                    uint8_t* dst = x_pad + (size_t)ic * pad_plane
                        + v.pH * padded_W + v.pW;
                    for (int h = 0; h < v.iH; h++)
                        std::memcpy(dst + h * padded_W, src + h * v.iW, v.iW);
                }

  #if defined(__ARM_FEATURE_MATMUL_INT8) || (defined(_MSC_VER) && defined(_M_ARM64))
                if (use_smmla) {
                    conv_int8_direct_nchw_smmla(y_i32, x_pad,
                        w_direct_smmla_buf.data(), w_direct_sums.data(),
                        v.iC, padded_H, padded_W, v.M, v.oH, v.oW,
                        v.kH, v.kW, v.sH, v.sW,
                        (uint8_t)v.x_zp, x_nhwc_scratch.data());
                } else
  #endif
                conv_int8_direct_neon(y_i32, x_pad, w_direct_buf.data(),
                    w_direct_sums.data(),
                    v.iC, padded_H, padded_W, v.M, v.oH, v.oW,
                    v.kH, v.kW, v.sH, v.sW);

                // Zero-point compensation + requantize (scalar — NEON requantize is a follow-up).
                T* out_n = (T*)v.y->data + (size_t)n * v.M * v.spatial;
                for (int oc = 0; oc < v.M; oc++) {
                    float ws = v.per_channel_scale ? v.w_scale_data[oc] : v.w_scale_data[0];
                    float combined_scale = v.x_scale * ws;
                    float bias_val = v.pbias ? (float)v.pbias[oc] * combined_scale : 0.0f;
                    int32_t zp_comp = (int32_t)v.x_zp * w_direct_sums[oc];
                    int32_t* irow = y_i32 + (size_t)oc * v.spatial;
                    T* out = out_n + (size_t)oc * v.spatial;
                    for (int s = 0; s < v.spatial; s++) {
                        int32_t raw_val = irow[s] - zp_comp;
                        float val = (float)raw_val * combined_scale + bias_val;
                        int32_t q = (int32_t)std::nearbyint(val * v.inv_y_scale) + (int32_t)v.y_zp;
                        out[s] = (T)std::clamp(q, v.qmin, v.clamp_max);
                    }
                }
            }
            return true;
        }
    }

    // NEON direct int8 conv (NHWC output). Mirrors exec_direct_int8_neon but
    // reads/writes NHWC tensors. Picks the SMMLA (i8mm) pack when
    // w_direct_smmla_buf is non-empty, falling back to the SDOT pack.
    template <typename T>
    bool exec_direct_int8_nhwc_neon(const qlinear4d_vars_t<T>& v) {
        if constexpr (!std::is_same_v<T, uint8_t>) { (void)v; return false; }
        else {
            if (w_direct_buf.empty()) return false;
            if (v.per_channel_zp) return false;
            if (v.w_zp_count != 0 && (int32_t)v.w_zp_data[0] != 0) return false;
            if (v.y->format != memory_layout_t::NHWC) return false;
            if (group != 1) return false;
            if (v.spatial > 256) return false;
            if (!has_neon_dotprod()) return false;

  #if defined(__ARM_FEATURE_MATMUL_INT8) || (defined(_MSC_VER) && defined(_M_ARM64))
            const bool use_smmla = !w_direct_smmla_buf.empty() && has_neon_i8mm();
  #else
            const bool use_smmla = false;
  #endif
            // SMMLA needs IC padded to 8, SDOT to 4. Pick the pad that matches.
            const int IC_block = use_smmla ? 8 : 4;
            const int IC_padded = ((v.iC + IC_block - 1) / IC_block) * IC_block;
            const int padded_H = v.iH + v.pH + cpads[2];
            const int padded_W = v.iW + v.pW + cpads[3];

            std::vector<uint8_t> x_pad_vec((size_t)padded_H * padded_W * IC_padded);
            std::vector<int32_t> y_i32_vec((size_t)v.oH * v.oW * v.M);
            uint8_t* x_pad = x_pad_vec.data();
            int32_t* y_i32 = y_i32_vec.data();

            for (int n = 0; n < v.oN; n++) {
                std::memset(x_pad, (uint8_t)v.x_zp,
                    (size_t)padded_H * padded_W * IC_padded);

                if (v.x->format == memory_layout_t::NHWC) {
                    const uint8_t* src_n = (const uint8_t*)v.x->data
                        + (size_t)n * v.iH * v.iW * v.iC;
                    for (int h = 0; h < v.iH; h++) {
                        uint8_t* drow = x_pad
                            + ((size_t)(h + v.pH) * padded_W + v.pW) * IC_padded;
                        const uint8_t* srow = src_n + (size_t)h * v.iW * v.iC;
                        for (int ww = 0; ww < v.iW; ww++)
                            std::memcpy(drow + (size_t)ww * IC_padded,
                                        srow + (size_t)ww * v.iC, v.iC);
                    }
                } else {
                    // NCHW → NHWC transpose into the padded buffer.
                    const uint8_t* src_n = (const uint8_t*)v.x->data
                        + (size_t)n * v.iC * v.iH * v.iW;
                    for (int ic = 0; ic < v.iC; ic++) {
                        const uint8_t* ch = src_n + (size_t)ic * v.iH * v.iW;
                        for (int h = 0; h < v.iH; h++) {
                            const uint8_t* srow = ch + (size_t)h * v.iW;
                            uint8_t* drow = x_pad
                                + ((size_t)(h + v.pH) * padded_W + v.pW) * IC_padded
                                + ic;
                            for (int ww = 0; ww < v.iW; ww++)
                                drow[(size_t)ww * IC_padded] = srow[ww];
                        }
                    }
                }

                bool ok;
  #if defined(__ARM_FEATURE_MATMUL_INT8) || (defined(_MSC_VER) && defined(_M_ARM64))
                if (use_smmla) {
                    ok = conv_int8_direct_nhwc_smmla(
                        y_i32, x_pad, w_direct_smmla_buf.data(), w_direct_sums.data(),
                        v.iC, padded_H, padded_W,
                        v.M, v.oH, v.oW,
                        v.kH, v.kW, v.sH, v.sW);
                } else
  #endif
                {
                    ok = conv_int8_direct_nhwc_neon(
                        y_i32, x_pad, w_direct_buf.data(), w_direct_sums.data(),
                        v.iC, padded_H, padded_W,
                        v.M, v.oH, v.oW,
                        v.kH, v.kW, v.sH, v.sW);
                }
                if (!ok) return false;

                T* out_n = (T*)v.y->data + (size_t)n * v.oH * v.oW * v.M;
                for (int oh = 0; oh < v.oH; oh++) {
                    for (int ow = 0; ow < v.oW; ow++) {
                        const int32_t* yp = y_i32
                            + ((size_t)oh * v.oW + ow) * v.M;
                        T* out = out_n + ((size_t)oh * v.oW + ow) * v.M;
                        for (int oc = 0; oc < v.M; oc++) {
                            float ws = v.per_channel_scale ? v.w_scale_data[oc] : v.w_scale_data[0];
                            float combined_scale = v.x_scale * ws;
                            float bias_val = v.pbias ? (float)v.pbias[oc] * combined_scale : 0.0f;
                            int32_t zp_comp = (int32_t)v.x_zp * w_direct_sums[oc];
                            int32_t raw_val = yp[oc] - zp_comp;
                            float val = (float)raw_val * combined_scale + bias_val;
                            int32_t q = (int32_t)std::nearbyint(val * v.inv_y_scale) + (int32_t)v.y_zp;
                            out[oc] = (T)std::clamp(q, v.qmin, v.clamp_max);
                        }
                    }
                }
            }
            v.y->format = memory_layout_t::NHWC;
            return true;
        }
    }
#endif

#ifdef NNR_ARCH_X64
    template <typename T>
    bool exec_packed_nr48(const qlinear4d_vars_t<T>& v) {
        auto& x = v.x;
        auto& w = v.w;
        auto& y = v.y;
        auto& x_scale = v.x_scale;
        auto& x_zp = v.x_zp;
        auto& y_scale = v.y_scale;
        auto& y_zp = v.y_zp;
        auto& w_scale_data = v.w_scale_data;
        auto& w_zp_data = v.w_zp_data;
        auto& w_scale_count = v.w_scale_count;
        auto& w_zp_count = v.w_zp_count;
        auto& per_channel_scale = v.per_channel_scale;
        auto& per_channel_zp = v.per_channel_zp;
        auto& pbias = v.pbias;
        auto& iN = v.iN;
        auto& iC = v.iC;
        auto& iH = v.iH;
        auto& iW = v.iW;
        auto& M = v.M;
        auto& C = v.C;
        auto& kH = v.kH;
        auto& kW = v.kW;
        auto& oN = v.oN;
        auto& oH = v.oH;
        auto& oW = v.oW;
        auto& MM = v.MM;
        auto& CC = v.CC;
        auto& CHW = v.CHW;
        auto& spatial = v.spatial;
        auto& sH = v.sH;
        auto& sW = v.sW;
        auto& dH = v.dH;
        auto& dW = v.dW;
        auto& pH = v.pH;
        auto& pW = v.pW;
        auto& clamp_min = v.clamp_min;
        auto& clamp_max = v.clamp_max;
        auto& qmin = v.qmin;
        auto& inv_y_scale = v.inv_y_scale;
        (void)x;(void)w;(void)y;(void)iN;(void)x_scale;(void)y_scale;

        if (!w_packed_nr48.empty() && has_avx512() && cpu_features().avx512vnni
            && std::is_same_v<T, uint8_t> && !per_channel_zp
            && group == 1) {
            int32_t w_zp_scalar = (w_zp_count > 0) ? (int32_t)w_zp_data[0] : 0;
            bool output_nhwc = (y->format == memory_layout_t::NHWC);
            // Padded stride for temp buffer: JIT writes NR=48-wide tiles,
            // so stride must be at least NR to avoid overlapping rows.
            int oc_padded = (MM + 47) & ~47;

            // NR=16 NHWC-direct (memcpy-free) predicate. Conservative v1
            // envelope: SSD-12 ResNet-34 Block 1/2/3 shape (3x3 s=1 pad=1
            // dilation=1, IC/OC >= 64, OC%16==0, w_zp==0, NHWC out).
            // Reuses w_packed_nr48 unchanged; reads raw x_nhwc directly
            // with OOB pixels substituted via an in-kernel x_zp_row buffer.
            const bool use_nhwc_direct_nr16 =
                   kH == 3 && kW == 3
                && sH == 1 && sW == 1
                && dH == 1 && dW == 1
                && pH == 1 && pW == 1
                && iC >= 64 && MM >= 64
                && (MM % 16 == 0)
                && w_zp_scalar == 0
                && output_nhwc;

            for (int n = 0; n < oN; n++) {
                const uint8_t* x_nhwc;

                if (x->format == memory_layout_t::NHWC) {
                    // NHWC: pass raw pointer directly (no pad copy)
                    x_nhwc = (const uint8_t*)x->data + (size_t)n * iH * iW * iC;
                } else {
                    // NCHW: transpose [C,H*W] → [H*W,C]. The scalar nested loop
                    // was the biggest densenet-int8 bottleneck (~10.5 MB of
                    // strided scalar stores per inference); the AVX-512 16×16
                    // helper replaces it with tiled 128-bit loads/stores.
                    uint8_t* buf = (uint8_t*)(((uintptr_t)x_pad_nhwc_buf.data() + 63) & ~63);
                    const uint8_t* src_n = (const uint8_t*)x->data
                        + (size_t)n * iC * iH * iW;
                    int8::transpose_nchw_to_nhwc_u8(buf, src_n, iC, iH * iW);
                    x_nhwc = buf;
                }

                // GEMM writes NHWC [spatial × OC]. JIT requantize uses 16-byte
                // stores, so y_out_stride must be 16-aligned to avoid overflow.
                // When OC is not 16-aligned or output is NCHW, use padded temp buffer.
                int8::conv_rq_params_t rq_p = rq_cached;
                uint8_t* y_out;
                bool nhwc_direct = output_nhwc && (MM % 16 == 0);
                if (nhwc_direct) {
                    y_out = (uint8_t*)y->data + (size_t)n * spatial * MM;
                } else {
                    y_out = y_nhwc_buf.data();
                    rq_p.y_out_stride = oc_padded;
                }
                rq_p.Y_out = y_out;

                if (use_nhwc_direct_nr16) {
                    // Memcpy-free: kernel reads raw x_nhwc directly,
                    // substitutes a local x_zp_row for OOB kernel pixels.
                    int8::conv_int8_nhwc_direct_nr16(
                        MM, oH, oW, iC, kH, kW,
                        sH, sW, dH, dW,
                        x_nhwc, iH, iW, pH, pW,
                        (int)x_zp,
                        w_packed_nr48.data(),
                        w_packed_nr48_col_sums.data(),
                        &rq_p);
                } else {
                    int8::conv_int8_packed_nr48(
                        MM, oH, oW, iC, kH, kW,
                        pH, pW, sH, sW, dH, dW,
                        x_nhwc, iH, iW, (int)x_zp,
                        w_packed_nr48.data(),
                        w_packed_nr48_col_sums.data(),
                        w_zp_scalar,
                        &rq_p);
                }

                if (!nhwc_direct) {
                    T* py = (T*)y->data + (size_t)n * (output_nhwc ? spatial * MM : MM * spatial);
                    if (output_nhwc) {
                        // Copy padded NHWC [spatial × oc_padded] → compact NHWC [spatial × OC]
                        for (int s = 0; s < spatial; s++)
                            memcpy(py + (size_t)s * MM, y_out + (size_t)s * oc_padded, MM);
                    } else {
                        // Transpose NHWC [spatial × oc_padded] → NCHW [OC × spatial]
                        for (int oc = 0; oc < MM; oc++) {
                            T* dst = py + (size_t)oc * spatial;
                            for (int s = 0; s < spatial; s++)
                                dst[s] = (T)y_out[(size_t)s * oc_padded + oc];
                        }
                    }
                }
            }
            if (output_nhwc)
                y->format = memory_layout_t::NHWC;
            return true;
        }

        return false; // no branch matched
    }

#endif

#ifdef NNR_ARCH_X64
    template <typename T>
    bool exec_gather_gemm(const qlinear4d_vars_t<T>& v) {
        auto& x = v.x;
        auto& w = v.w;
        auto& y = v.y;
        auto& x_scale = v.x_scale;
        auto& x_zp = v.x_zp;
        auto& y_scale = v.y_scale;
        auto& y_zp = v.y_zp;
        auto& w_scale_data = v.w_scale_data;
        auto& w_zp_data = v.w_zp_data;
        auto& w_scale_count = v.w_scale_count;
        auto& w_zp_count = v.w_zp_count;
        auto& per_channel_scale = v.per_channel_scale;
        auto& per_channel_zp = v.per_channel_zp;
        auto& pbias = v.pbias;
        auto& iN = v.iN;
        auto& iC = v.iC;
        auto& iH = v.iH;
        auto& iW = v.iW;
        auto& M = v.M;
        auto& C = v.C;
        auto& kH = v.kH;
        auto& kW = v.kW;
        auto& oN = v.oN;
        auto& oH = v.oH;
        auto& oW = v.oW;
        auto& MM = v.MM;
        auto& CC = v.CC;
        auto& CHW = v.CHW;
        auto& spatial = v.spatial;
        auto& sH = v.sH;
        auto& sW = v.sW;
        auto& dH = v.dH;
        auto& dW = v.dW;
        auto& pH = v.pH;
        auto& pW = v.pW;
        auto& clamp_min = v.clamp_min;
        auto& clamp_max = v.clamp_max;
        auto& qmin = v.qmin;
        auto& inv_y_scale = v.inv_y_scale;
        (void)x;(void)w;(void)y;(void)iN;(void)x_scale;(void)y_scale;

        if (!w_gather_packed.empty() && has_avx512() && cpu_features().avx512vnni
            && std::is_same_v<T, uint8_t> && !per_channel_zp
            && sH == 1 && sW == 1
            && y->format == memory_layout_t::NHWC) {
            // fprintf(stderr, "[NNR] QLinearConv NHWC gather path: %dx%d OC=%d\n", oH, oW, MM);
            int32_t w_zp_scalar = (w_zp_count > 0) ? (int32_t)w_zp_data[0] : 0;

            int padded_H = iH + pH + cpads[2];
            int padded_W = iW + pW + cpads[3];
            size_t pad_plane = (size_t)padded_H * padded_W;

            // Tile y_i32 in [spatial × OC] layout
            constexpr size_t Y_TILE_BYTES = 2 * 1024 * 1024;
            int y_tile_h = std::max(1, (int)(Y_TILE_BYTES / ((size_t)MM * oW * sizeof(int32_t))));
            y_tile_h = std::min(y_tile_h, oH);

            uint8_t* x_pad = (uint8_t*)(((uintptr_t)x_pad_buf.data() + 63) & ~63);
            // Reuse y_i32_buf but in [spatial × OC] layout (same total size)
            int32_t* y_i32 = y_i32_buf.data();

            for (int n = 0; n < oN; n++) {
                // x_pad is always NCHW: [C, padH, padW]. Fill from input.
                memset(x_pad, (uint8_t)x_zp, (size_t)iC * pad_plane);
                if (x->format == memory_layout_t::NHWC) {
                    // Input is NHWC [H, W, C]: transpose to NCHW x_pad
                    const uint8_t* xn = (const uint8_t*)x->data + (size_t)n * iH * iW * iC;
                    for (int h = 0; h < iH; h++)
                        for (int w = 0; w < iW; w++)
                            for (int c = 0; c < iC; c++)
                                x_pad[c * pad_plane + (h + pH) * padded_W + (w + pW)]
                                    = xn[(h * iW + w) * iC + c];
                } else {
                    for (int ic = 0; ic < iC; ic++) {
                        const uint8_t* src = (const uint8_t*)x->data
                            + ((size_t)n * iC + ic) * iH * iW;
                        uint8_t* dst = x_pad + (size_t)ic * pad_plane
                            + pH * padded_W + pW;
                        for (int h = 0; h < iH; h++)
                            memcpy(dst + h * padded_W, src + h * iW, iW);
                    }
                }

                // NHWC output: Y_out[(n*spatial + sp)*OC + oc]
                T* py = (T*)y->data + (size_t)n * spatial * MM;

                int8::conv_rq_params_t rq_p = rq_cached;
                rq_p.Y_out           = (uint8_t*)py;

                for (int oh0 = 0; oh0 < oH; oh0 += y_tile_h) {
                    int th = std::min(y_tile_h, oH - oh0);
                    int8::conv_int8_gather_gemm(
                        MM, oW, oh0, th,
                        CHW,
                        w_gather_packed.data(),
                        w_gather_col_sums.data(),
                        x_pad,
                        (int)x_zp,
                        k_off_oh_all.data(),
                        &rq_p);
                }
            }
            y->format = memory_layout_t::NHWC;
            return true;
        }

        return false; // no branch matched
    }

#endif

#ifdef NNR_ARCH_X64
    template <typename T>
    bool exec_vnni_im2col(const qlinear4d_vars_t<T>& v) {
        auto& x = v.x;
        auto& w = v.w;
        auto& y = v.y;
        auto& x_scale = v.x_scale;
        auto& x_zp = v.x_zp;
        auto& y_scale = v.y_scale;
        auto& y_zp = v.y_zp;
        auto& w_scale_data = v.w_scale_data;
        auto& w_zp_data = v.w_zp_data;
        auto& w_scale_count = v.w_scale_count;
        auto& w_zp_count = v.w_zp_count;
        auto& per_channel_scale = v.per_channel_scale;
        auto& per_channel_zp = v.per_channel_zp;
        auto& pbias = v.pbias;
        auto& iN = v.iN;
        auto& iC = v.iC;
        auto& iH = v.iH;
        auto& iW = v.iW;
        auto& M = v.M;
        auto& C = v.C;
        auto& kH = v.kH;
        auto& kW = v.kW;
        auto& oN = v.oN;
        auto& oH = v.oH;
        auto& oW = v.oW;
        auto& MM = v.MM;
        auto& CC = v.CC;
        auto& CHW = v.CHW;
        auto& spatial = v.spatial;
        auto& sH = v.sH;
        auto& sW = v.sW;
        auto& dH = v.dH;
        auto& dW = v.dW;
        auto& pH = v.pH;
        auto& pW = v.pW;
        auto& clamp_min = v.clamp_min;
        auto& clamp_max = v.clamp_max;
        auto& qmin = v.qmin;
        auto& inv_y_scale = v.inv_y_scale;
        (void)x;(void)w;(void)y;(void)iN;(void)x_scale;(void)y_scale;

        if (!w_vnni_buf.empty() && has_avx512() && cpu_features().avx512vnni
            && std::is_same_v<T, uint8_t> && !per_channel_zp) {
            int32_t w_zp_scalar = (w_zp_count > 0) ? (int32_t)w_zp_data[0] : 0;
            int a_zp_eff = 128 + w_zp_scalar;
            int b_zp_eff = (int)x_zp - 128;

            // Fused im2col+GEMM: read input directly, no im2col buffer or B-packing.
            // Requires stride=1 (loads are contiguous within output rows).
            // JIT software-pipelined K-loop makes fused faster than packed on all layers.
            if (sH == 1 && sW == 1) {
                int padded_H = iH + pH + cpads[2];
                int padded_W = iW + pW + cpads[3];
                size_t pad_plane = (size_t)padded_H * padded_W;

                // Tile y_i32 to keep it in L2 for requantization
                constexpr size_t Y_TILE_BYTES = 2 * 1024 * 1024;
                int y_tile_h = std::max(1, (int)(Y_TILE_BYTES / ((size_t)MM * oW * sizeof(int32_t))));
                y_tile_h = std::min(y_tile_h, oH);
                int y_tile_sp = y_tile_h * oW;

                // Use pre-allocated buffers (aligned start for AVX-512)
                uint8_t* x_pad = (uint8_t*)(((uintptr_t)x_pad_buf.data() + 63) & ~63);
                int32_t* y_i32 = y_i32_buf.data();

                for (int n = 0; n < oN; n++) {
                    // Pre-pad input: fill with x_zp, then copy actual pixels.
                    // The memset also warms the cache for the fused GEMM's scattered reads.
                    memset(x_pad, (uint8_t)x_zp, (size_t)iC * pad_plane);
                    for (int ic = 0; ic < iC; ic++) {
                        const uint8_t* src = (const uint8_t*)x->data
                            + ((size_t)n * iC + ic) * iH * iW;
                        uint8_t* dst = x_pad + (size_t)ic * pad_plane
                            + pH * padded_W + pW;
                        for (int h = 0; h < iH; h++)
                            memcpy(dst + h * padded_W, src + h * iW, iW);
                    }

                    for (int g = 0; g < group; g++) {
                        const uint8_t* w_shifted = w_vnni_buf.data() + (size_t)g * MM * CHW;
                        const int32_t* w_rs = w_row_sums_buf.data() + g * MM;
                        T* py = (T*)y->data + ((size_t)n * M + g * MM) * spatial;

                        // Use pre-computed rq params with per-group offset
                        int8::conv_rq_params_t rq_p = rq_cached;
                        rq_p.output_scales   = rq_output_scales.data() + g * MM;
                        rq_p.bias_int32      = pbias ? pbias + g * MM : nullptr;
                        rq_p.combined_scales = rq_combined_scales.data() + g * MM;
                        rq_p.bias_vals       = pbias ? rq_bias_f.data() + g * MM : nullptr;
                        rq_p.Y_out           = (uint8_t*)py;
                        rq_p.y_out_stride    = spatial;

                        for (int oh0 = 0; oh0 < oH; oh0 += y_tile_h) {
                            int th = std::min(y_tile_h, oH - oh0);

                            int8::conv_int8_fused_gemm(
                                MM, oW, oh0, th,
                                CC, kH, kW, dH, dW,
                                w_shifted, a_zp_eff, w_rs,
                                x_pad + (size_t)g * CC * pad_plane,
                                padded_H, padded_W,
                                b_zp_eff, y_i32,
                                k_off_base.empty() ? nullptr : k_off_base.data(),
                                k_off_oh_all.empty() ? nullptr : k_off_oh_all.data(),
                                &rq_p);
                        }
                    }
                }

                return true;
            }

            // Fallback: tiled im2col + NR=48 packed GEMM.
            {
                constexpr size_t TILE_BYTES = 256 * 1024;
                int tile_h = oH;
                if ((size_t)CHW * spatial > TILE_BYTES) {
                    tile_h = std::max(1, (int)(TILE_BYTES / ((size_t)CHW * oW)));
                    tile_h = std::min(tile_h, oH);
                }
                int tile_spatial_max = tile_h * oW;
                // Pad ldc to 16-column boundary: NR=48 JIT writes full zmm (16 int32s) in tail
                int tile_sp_padded = (tile_spatial_max + 15) & ~15;

                int8_t* col_s = (int8_t*)nnr_aligned_alloc((size_t)CHW * tile_spatial_max, 64);
                size_t col_pack_sz = int8::pack_b_int8_nr48_size(CHW, tile_spatial_max);
                int8_t* col_packed = (int8_t*)nnr_aligned_alloc(col_pack_sz, 64);
                int32_t* col_col_sums = (int32_t*)nnr_aligned_alloc((size_t)tile_sp_padded * sizeof(int32_t), 64);
                int32_t* y_i32 = (int32_t*)nnr_aligned_alloc((size_t)MM * tile_sp_padded * sizeof(int32_t), 64);

                const __m512i v128 = _mm512_set1_epi8((char)128);

                for (int n = 0; n < oN; n++) {
                    for (int g = 0; g < group; g++) {
                        const T* xn = (const T*)x->data + ((size_t)n * iC + g * CC) * iH * iW;
                        const uint8_t* w_shifted = w_vnni_buf.data() + (size_t)g * MM * CHW;
                        const int32_t* w_rs = w_row_sums_buf.data() + g * MM;
                        T* py = (T*)y->data + ((size_t)n * M + g * MM) * spatial;

                        __m512 vis = _mm512_set1_ps(inv_y_scale);
                        __m512 vzp = _mm512_set1_ps((float)y_zp);
                        __m512 vqmin = _mm512_set1_ps((float)qmin);
                        __m512 vqmax = _mm512_set1_ps((float)clamp_max);

                        for (int oh0 = 0; oh0 < oH; oh0 += tile_h) {
                            int th = std::min(tile_h, oH - oh0);
                            int tile_sp = th * oW;

                            int8_t pad_val = (int8_t)((int)x_zp - 128);
                            memset(col_s, *(uint8_t*)&pad_val, (size_t)CHW * tile_sp);
                            for (int c = 0; c < C; c++) {
                                const T* xc = xn + (size_t)c * iH * iW;
                                for (int kh = 0; kh < kH; kh++) {
                                    for (int kw = 0; kw < kW; kw++) {
                                        int k_idx = (c * kH + kh) * kW + kw;
                                        int8_t* dst_row = col_s + (size_t)k_idx * tile_sp;
                                        for (int toh = 0; toh < th; toh++) {
                                            int oh = oh0 + toh;
                                            int ih = oh * sH - pH + kh * dH;
                                            if (ih < 0 || ih >= iH) continue;
                                            const T* srow = xc + ih * iW;
                                            int8_t* drow = dst_row + toh * oW;
                                            int iw_base = -pW + kw * dW;
                                            if (sW == 1) {
                                                int w0 = std::max(0, -iw_base);
                                                int w1 = std::min(oW, iW - iw_base);
                                                const uint8_t* src = (const uint8_t*)srow + iw_base + w0;
                                                int8_t* dst = drow + w0;
                                                int len = w1 - w0;
                                                int j = 0;
                                                for (; j + 64 <= len; j += 64) {
                                                    __m512i v = _mm512_loadu_si512(src + j);
                                                    _mm512_storeu_si512(dst + j, _mm512_sub_epi8(v, v128));
                                                }
                                                for (; j < len; j++)
                                                    dst[j] = (int8_t)((int)src[j] - 128);
                                            } else {
                                                for (int ow = 0; ow < oW; ow++) {
                                                    int iw = ow * sW + iw_base;
                                                    if (iw >= 0 && iw < iW)
                                                        drow[ow] = (int8_t)((int)srow[iw] - 128);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            int tile_sp16 = (tile_sp + 15) & ~15;
                            int8::pack_b_int8_nr48_and_col_sums(col_packed, col_col_sums, col_s, CHW, tile_sp);
                            int8::gemm_int8_nr48(MM, tile_sp, CHW,
                                w_shifted, a_zp_eff, col_packed, b_zp_eff,
                                col_col_sums, w_rs, y_i32, tile_sp16);

                            for (int oc = 0; oc < MM; oc++) {
                                float ws = per_channel_scale ? w_scale_data[g * MM + oc] : w_scale_data[0];
                                float combined_scale = x_scale * ws;
                                float bias_val = pbias ? (float)pbias[g * MM + oc] * combined_scale : 0.0f;
                                __m512 vcs = _mm512_set1_ps(combined_scale);
                                __m512 vbias = _mm512_set1_ps(bias_val);
                                int32_t* irow = y_i32 + (size_t)oc * tile_sp16;
                                T* out = py + (size_t)oc * spatial + oh0 * oW;
                                int s = 0;
                                for (; s + 16 <= tile_sp; s += 16) {
                                    __m512i raw = _mm512_loadu_si512(irow + s);
                                    __m512 fv = _mm512_cvtepi32_ps(raw);
                                    fv = _mm512_fmadd_ps(fv, vcs, vbias);
                                    fv = _mm512_add_ps(_mm512_roundscale_ps(
                                        _mm512_mul_ps(fv, vis), _MM_FROUND_TO_NEAREST_INT), vzp);
                                    fv = _mm512_max_ps(_mm512_min_ps(fv, vqmax), vqmin);
                                    _mm_storeu_si128((__m128i*)(out + s),
                                        _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(fv)));
                                }
                                for (; s < tile_sp; s++) {
                                    float val = (float)irow[s] * combined_scale + bias_val;
                                    int32_t q = (int32_t)std::nearbyint(val * inv_y_scale) + (int32_t)y_zp;
                                    out[s] = (T)std::clamp(q, qmin, clamp_max);
                                }
                            }
                        }
                    }
                }

                nnr_aligned_free(col_s);
                nnr_aligned_free(col_packed);
                nnr_aligned_free(col_col_sums);
                nnr_aligned_free(y_i32);
                return true;
            }
        }

        return false; // no branch matched
    }

#endif

    template <typename T>
    bool exec_float_fallback(const qlinear4d_vars_t<T>& v) {
        auto& x = v.x;
        auto& w = v.w;
        auto& y = v.y;
        auto& x_scale = v.x_scale;
        auto& x_zp = v.x_zp;
        auto& y_scale = v.y_scale;
        auto& y_zp = v.y_zp;
        auto& w_scale_data = v.w_scale_data;
        auto& w_zp_data = v.w_zp_data;
        auto& w_scale_count = v.w_scale_count;
        auto& w_zp_count = v.w_zp_count;
        auto& per_channel_scale = v.per_channel_scale;
        auto& per_channel_zp = v.per_channel_zp;
        auto& pbias = v.pbias;
        auto& iN = v.iN;
        auto& iC = v.iC;
        auto& iH = v.iH;
        auto& iW = v.iW;
        auto& M = v.M;
        auto& C = v.C;
        auto& kH = v.kH;
        auto& kW = v.kW;
        auto& oN = v.oN;
        auto& oH = v.oH;
        auto& oW = v.oW;
        auto& MM = v.MM;
        auto& CC = v.CC;
        auto& CHW = v.CHW;
        auto& spatial = v.spatial;
        auto& sH = v.sH;
        auto& sW = v.sW;
        auto& dH = v.dH;
        auto& dW = v.dW;
        auto& pH = v.pH;
        auto& pW = v.pW;
        auto& clamp_min = v.clamp_min;
        auto& clamp_max = v.clamp_max;
        auto& qmin = v.qmin;
        auto& inv_y_scale = v.inv_y_scale;
        (void)x;(void)w;(void)y;(void)iN;(void)x_scale;(void)y_scale;

        // Float path: convert int8→float, im2col + dgemm_packed_a, requantize.
        // Uses pre-packed FP32 weights from reshape for maximum GEMM performance.
        float* x_f32 = (float*)nnr_aligned_alloc(x->ndata * sizeof(float), 64);
        {
            const T* px = (const T*)x->data;
            int32_t xzp = (int32_t)x_zp;
            size_t total = x->ndata;
            size_t i = 0;
#ifdef NNR_ARCH_X64
            __m512 vzp = _mm512_set1_ps((float)xzp);
            if (std::is_same_v<T, uint8_t>) {
                for (; i + 16 <= total; i += 16) {
                    __m128i bytes = _mm_loadu_si128((const __m128i*)(px + i));
                    __m512 fv = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(bytes));
                    _mm512_storeu_ps(x_f32 + i, _mm512_sub_ps(fv, vzp));
                }
            }
#endif
            for (; i < total; i++)
                x_f32[i] = (float)((int32_t)px[i] - xzp);
        }
        float* col = (float*)nnr_aligned_alloc((size_t)CHW * spatial * sizeof(float), 64);
        float* y_f32 = (float*)nnr_aligned_alloc((size_t)M * spatial * sizeof(float), 64);

        size_t per_group_pack = pack_a_size(MM, CHW);

        for (int n = 0; n < oN; n++) {
            for (int g = 0; g < group; g++) {
                const float* xn = x_f32 + ((size_t)n * iC + g * CC) * iH * iW;
                float* yn = y_f32 + (size_t)g * MM * spatial;
                im2col(col, xn, C, iH, iW, kH, kW, oH, oW, sH, sW, pH, pW, dH, dW);
                dgemm_packed_a(MM, spatial, CHW,
                    w_packed_f32.data() + g * per_group_pack, col, yn);
            }

            T* py = (T*)y->data + (size_t)n * M * spatial;
#ifdef NNR_ARCH_X64
            if (has_avx512()) {
                __m512 vis = _mm512_set1_ps(inv_y_scale);
                __m512 vzp = _mm512_set1_ps((float)y_zp);
                __m512 vqmin = _mm512_set1_ps((float)qmin);
                __m512 vqmax = _mm512_set1_ps((float)clamp_max);
                for (int oc = 0; oc < M; oc++) {
                    float ws = per_channel_scale ? w_scale_data[oc] : w_scale_data[0];
                    __m512 vcs = _mm512_set1_ps(x_scale * ws);
                    float cs = x_scale * ws;
                    __m512 vbias = _mm512_set1_ps(pbias ? (float)pbias[oc] * cs : 0.0f);
                    float* yrow = y_f32 + (size_t)oc * spatial;
                    T* out = py + (size_t)oc * spatial;
                    int s = 0;
                    for (; s + 16 <= spatial; s += 16) {
                        __m512 v = _mm512_loadu_ps(yrow + s);
                        v = _mm512_fmadd_ps(v, vcs, vbias);
                        v = _mm512_add_ps(_mm512_roundscale_ps(
                            _mm512_mul_ps(v, vis), _MM_FROUND_TO_NEAREST_INT), vzp);
                        v = _mm512_max_ps(_mm512_min_ps(v, vqmax), vqmin);
                        __m512i iv = _mm512_cvtps_epi32(v);
                        _mm_storeu_si128((__m128i*)(out + s), _mm512_cvtepi32_epi8(iv));
                    }
                    for (; s < spatial; s++) {
                        float val = yrow[s] * cs + (pbias ? (float)pbias[oc] * cs : 0.0f);
                        int32_t q = (int32_t)std::nearbyint(val * inv_y_scale) + (int32_t)y_zp;
                        out[s] = (T)std::clamp(q, qmin, clamp_max);
                    }
                }
            } else
#endif
            {
                for (int oc = 0; oc < M; oc++) {
                    float ws = per_channel_scale ? w_scale_data[oc] : w_scale_data[0];
                    float combined_scale = x_scale * ws;
                    float bias_val = pbias ? (float)pbias[oc] * combined_scale : 0.0f;
                    float* yrow = y_f32 + (size_t)oc * spatial;
                    T* out = py + (size_t)oc * spatial;
                    for (int s = 0; s < spatial; s++) {
                        float val = yrow[s] * combined_scale + bias_val;
                        int32_t q = (int32_t)std::nearbyint(val * inv_y_scale) + (int32_t)y_zp;
                        out[s] = (T)std::clamp(q, qmin, clamp_max);
                    }
                }
            }
        }
        nnr_aligned_free(x_f32);
        nnr_aligned_free(col);
        nnr_aligned_free(y_f32);
        return true;

        return true;
    }

    template <typename T>
    bool exec_simd_4d() {
        qlinear4d_vars_t<T> v;
        if (!init_qlinear4d_vars<T>(v)) return false;

#ifdef NNR_ARCH_X64
        // All x64 int8 conv specializations require AVX-512+VNNI (VPDPBUSD).
        // Without it, dispatch falls through to the float dequant fallback.
        if (has_avx512()) {
            if (exec_dw_int8_nhwc<T>(v))  return true;
            if (exec_first_layer_int8<T>(v)) return true;
            if (exec_direct_int8<T>(v))   return true;
            if (exec_packed_nr48<T>(v))   return true;
            if (exec_gather_gemm<T>(v))   return true;
            if (exec_vnni_im2col<T>(v))   return true;
        }
#elifdef NNR_ARCH_ARM64
        if (exec_dw_int8_nhwc_neon<T>(v))    return true;
        if (exec_direct_int8_nhwc_neon<T>(v)) return true;
        if (exec_direct_int8_neon<T>(v)) return true;
#endif
        return exec_float_fallback<T>(v);
    }


    bool exec() override {
        data_type_t type = inputs[0]->type;
        // Use SIMD-accelerated im2col+GEMM path for 4D convolutions
        if (inputs[0]->ndim == 4) {
            if (type == NNR_DATA_TYPE_UINT8) return exec_simd_4d<uint8_t>();
            if (type == NNR_DATA_TYPE_INT8) return exec_simd_4d<int8_t>();
        }
        // Scalar fallback for 3D and other cases
        if (type == NNR_DATA_TYPE_UINT8) {
            return exec_typed<uint8_t>();
        } else if (type == NNR_DATA_TYPE_INT8) {
            return exec_typed<int8_t>();
        }
        return false;
    }
};

} // namespace

// @nnr-meta-op mt=static layout=[NCHW,NHWC] prepack=yes
operator_t* resolver_default_op_QLinearConv(int opset, pool_t& pool) {
    if (opset >= 10)
        return pool_new<QLinearConv_operator>(pool);
    return nullptr;
}

} // namespace nnr
