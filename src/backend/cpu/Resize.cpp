#include "nnr.h"
#include "util.h"
#include "thread_pool.h"
#include <cmath>
#include <algorithm>
#include <string>

namespace nnr {

namespace {

struct Resize_operator : public operator_t {

    enum coord_mode_t {
        half_pixel,
        asymmetric,
        pytorch_half_pixel,
        tf_half_pixel_for_nn,
        align_corners,
        tf_crop_and_resize,
        half_pixel_symmetric,
    } coord_mode;

    float cubic_coeff_a;
    int exclude_outside;
    float extrapolation_value;
    int antialias;

    enum interp_mode_t { nearest, linear, cubic } mode;

    enum nearest_mode_t {
        round_prefer_floor, round_prefer_ceil, floor_mode, ceil_mode,
    } nearest_mode;

    enum keep_aspect_ratio_policy_t {
        stretch, not_larger, not_smaller,
    } keep_aspect_ratio_policy;

    bool init() override {
        if (inputs.size() < 1 || outputs.size() != 1) return false;

        auto cm = std::string(attribute(attr_key_t::coordinate_transformation_mode, "half_pixel"));
        if (cm == "half_pixel") coord_mode = half_pixel;
        else if (cm == "asymmetric") coord_mode = asymmetric;
        else if (cm == "pytorch_half_pixel") coord_mode = pytorch_half_pixel;
        else if (cm == "tf_half_pixel_for_nn") coord_mode = tf_half_pixel_for_nn;
        else if (cm == "align_corners") coord_mode = align_corners;
        else if (cm == "tf_crop_and_resize") coord_mode = tf_crop_and_resize;
        else if (cm == "half_pixel_symmetric") coord_mode = half_pixel_symmetric;
        else coord_mode = half_pixel;

        cubic_coeff_a = attribute(attr_key_t::cubic_coeff_a, -0.75f);
        exclude_outside = attribute(attr_key_t::exclude_outside, 0);
        extrapolation_value = attribute(attr_key_t::extrapolation_value, 0.0f);
        antialias = attribute(attr_key_t::antialias, 0);

        auto m = std::string(attribute(attr_key_t::mode, "nearest"));
        if (m == "nearest") mode = nearest;
        else if (m == "linear") mode = linear;
        else if (m == "cubic") mode = cubic;
        else mode = nearest;

        auto nm = std::string(attribute(attr_key_t::nearest_mode, "round_prefer_floor"));
        if (nm == "round_prefer_floor") nearest_mode = round_prefer_floor;
        else if (nm == "round_prefer_ceil") nearest_mode = round_prefer_ceil;
        else if (nm == "floor") nearest_mode = floor_mode;
        else if (nm == "ceil") nearest_mode = ceil_mode;
        else nearest_mode = round_prefer_floor;

        auto kp = std::string(attribute(attr_key_t::keep_aspect_ratio_policy, "stretch"));
        if (kp == "not_larger") keep_aspect_ratio_policy = not_larger;
        else if (kp == "not_smaller") keep_aspect_ratio_policy = not_smaller;
        else keep_aspect_ratio_policy = stretch;

        return true;
    }

    // Get axes from attribute
    void get_axes(int ndim, small_vector<int>& axes) {
        int64_t* axes_data;
        int naxes = attribute(attr_key_t::axes, axes_data);
        if (naxes > 0) {
            axes.resize(naxes);
            for (int i = 0; i < naxes; ++i) {
                int a = (int)axes_data[i];
                if (a < 0) a += ndim;
                axes[i] = a;
            }
        } else {
            axes.resize(ndim);
            for (int i = 0; i < ndim; ++i) axes[i] = i;
        }
    }

    bool reshape() override {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        int ndim = x->ndim;

        // Find scales and sizes inputs
        // Opset 11+: inputs are [X, roi, scales, sizes]
        // Opset 19+: inputs are [X, roi?, scales?, sizes?] with optional empty
        const tensor_t* roi_t = nullptr;
        const tensor_t* scales_t = nullptr;
        const tensor_t* sizes_t = nullptr;

        for (size_t i = 1; i < inputs.size(); ++i) {
            if (!inputs[i] || inputs[i]->ndata == 0) continue;
            auto name = std::string_view(inputs[i]->name);
            if (i == 1 && name.find("roi") != std::string_view::npos) { roi_t = inputs[i]; continue; }
            if (i == 1 && name.find("scale") == std::string_view::npos && name.find("size") == std::string_view::npos) { roi_t = inputs[i]; continue; }
        }
        // Standard positional: [X, roi, scales, sizes]
        if (inputs.size() >= 3 && inputs[2] && inputs[2]->ndata > 0) scales_t = inputs[2];
        if (inputs.size() >= 4 && inputs[3] && inputs[3]->ndata > 0) sizes_t = inputs[3];
        if (!scales_t && inputs.size() >= 2 && inputs[1] && inputs[1]->ndata > 0) {
            // Check if input[1] might be scales (opset 19 format with no ROI)
            if (inputs[1]->type == NNR_DATA_TYPE_FLOAT32) {
                // Could be ROI or scales - use name hint
                auto name = std::string_view(inputs[1]->name);
                if (name.find("scale") != std::string_view::npos) {
                    scales_t = inputs[1];
                    roi_t = nullptr;
                }
            }
        }

        small_vector<int> axes;
        get_axes(ndim, axes);

        small_vector<int> dims(ndim);
        for (int i = 0; i < ndim; ++i) dims[i] = x->dims[i];

        if (sizes_t && sizes_t->ndata > 0) {
            const int64_t* sz = (const int64_t*)sizes_t->data;
            int nvals = (int)sizes_t->ndata;

            if (keep_aspect_ratio_policy != stretch && nvals > 0) {
                // Compute uniform scale from the specified axes
                float scale = (keep_aspect_ratio_policy == not_larger) ? 1e30f : 0.0f;
                for (int i = 0; i < nvals && i < (int)axes.size(); ++i) {
                    int a = axes[i];
                    float s = (float)sz[i] / (float)x->dims[a];
                    if (keep_aspect_ratio_policy == not_larger) scale = std::min(scale, s);
                    else scale = std::max(scale, s);
                }
                for (int i = 0; i < nvals && i < (int)axes.size(); ++i) {
                    int a = axes[i];
                    dims[a] = std::max(1, (int)std::round(x->dims[a] * scale));
                }
            } else {
                for (int i = 0; i < nvals && i < (int)axes.size(); ++i) {
                    int a = axes[i];
                    dims[a] = (int)sz[i];
                }
            }
        } else if (scales_t && scales_t->ndata > 0) {
            const float* sc = (const float*)scales_t->data;
            int nvals = (int)scales_t->ndata;
            for (int i = 0; i < nvals && i < (int)axes.size(); ++i) {
                int a = axes[i];
                dims[a] = (int)std::floor(x->dims[a] * sc[i]);
            }
        }
        y->reinit(x->type, dims);
        return true;
    }

    float transform_coord(float out_coord, int in_size, int out_size, float scale,
                           float roi_start = 0, float roi_end = 1) {
        switch (coord_mode) {
        case half_pixel:
            return (out_coord + 0.5f) / scale - 0.5f;
        case asymmetric:
            return out_coord / scale;
        case pytorch_half_pixel:
            return out_size > 1 ? (out_coord + 0.5f) / scale - 0.5f : 0.0f;
        case align_corners: {
            float len_resized = in_size * scale;
            return (len_resized > 1) ? out_coord * (in_size - 1.0f) / (len_resized - 1.0f) : 0.0f;
        }
        case tf_half_pixel_for_nn:
            return (out_coord + 0.5f) / scale;
        case tf_crop_and_resize:
            return out_size > 1
                ? roi_start * (in_size - 1) + out_coord * (roi_end - roi_start) * (in_size - 1) / (out_size - 1)
                : 0.5f * (roi_start + roi_end) * (in_size - 1);
        case half_pixel_symmetric: {
            float adj = (float)out_size / (scale * in_size);
            float center = in_size / 2.0f;
            float offset = center * (1.0f - adj);
            return offset + (out_coord + 0.5f) / scale - 0.5f;
        }
        default:
            return out_coord / scale;
        }
    }

    int nearest_idx(float coord, int in_size) {
        int idx;
        switch (nearest_mode) {
        case round_prefer_floor:
            idx = (int)std::round(coord);
            if (coord == std::floor(coord) + 0.5f)
                idx = (int)std::floor(coord);
            break;
        case round_prefer_ceil:
            idx = (int)std::round(coord);
            break;
        case floor_mode:
            idx = (int)std::floor(coord);
            break;
        case ceil_mode:
            idx = (int)std::ceil(coord);
            break;
        default:
            idx = (int)std::round(coord);
            break;
        }
        return std::clamp(idx, 0, in_size - 1);
    }

    float cubic_weight(float x, float a) {
        x = std::abs(x);
        if (x <= 1.0f)
            return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
        else if (x < 2.0f)
            return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
        return 0.0f;
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;
        int ndim = x->ndim;

        // Get actual scales from input or compute from dims
        float scales[MAX_NDIM];
        const tensor_t* scales_t = (inputs.size() >= 3 && inputs[2] && inputs[2]->ndata > 0) ? inputs[2] : nullptr;

        small_vector<int> axes;
        get_axes(ndim, axes);

        // Default: compute from dims
        for (int i = 0; i < ndim; ++i)
            scales[i] = (x->dims[i] > 0) ? (float)y->dims[i] / (float)x->dims[i] : 1.0f;

        // Override with actual scale values if scales input is provided (and sizes is not)
        const tensor_t* sizes_t = (inputs.size() >= 4 && inputs[3] && inputs[3]->ndata > 0) ? inputs[3] : nullptr;
        if (scales_t && !sizes_t) {
            const float* sc = (const float*)scales_t->data;
            int nvals = (int)scales_t->ndata;
            for (int i = 0; i < nvals && i < (int)axes.size(); ++i) {
                scales[axes[i]] = sc[i];
            }
        }

        // Get ROI data
        float roi_data[MAX_NDIM * 2] = {};
        for (int i = 0; i < ndim; ++i) { roi_data[i] = 0; roi_data[i + ndim] = 1; }
        if (coord_mode == tf_crop_and_resize) {
            const tensor_t* roi_in = (inputs.size() >= 2 && inputs[1] && inputs[1]->ndata > 0) ? inputs[1] : nullptr;
            if (roi_in) {
                const float* pr = (const float*)roi_in->data;
                int n = (int)roi_in->ndata;
                if ((int)axes.size() * 2 == n) {
                    // ROI is [start_axes..., end_axes...]
                    for (int i = 0; i < (int)axes.size(); ++i) {
                        roi_data[axes[i]] = pr[i];
                        roi_data[axes[i] + ndim] = pr[i + (int)axes.size()];
                    }
                } else if (n == ndim * 2) {
                    for (int i = 0; i < n; ++i) roi_data[i] = pr[i];
                }
            }
        }

        int total = (int)y->ndata;
        small_vector<int> out_idx(ndim);

        if (mode == nearest && !antialias) {
            // Fast path: 4D NCHW nearest resize with precomputed index tables.
            // Precompute source index for each output position per axis,
            // then use threaded copy with direct indexing (no per-pixel transforms).
            if (ndim == 4 && scales[0] == 1.0f && scales[1] == 1.0f
                && x->dims[2] > 0 && x->dims[3] > 0) {
                int N = x->dims[0], C = x->dims[1];
                int iH = x->dims[2], iW = x->dims[3];
                int oH = y->dims[2], oW = y->dims[3];

                // Precompute h and w index tables
                std::vector<int> h_idx(oH), w_idx(oW);
                for (int oh = 0; oh < oH; oh++) {
                    float coord = transform_coord((float)oh, iH, oH, scales[2],
                        roi_data[2], roi_data[2 + ndim]);
                    h_idx[oh] = nearest_idx(coord, iH);
                }
                for (int ow = 0; ow < oW; ow++) {
                    float coord = transform_coord((float)ow, iW, oW, scales[3],
                        roi_data[3], roi_data[3 + ndim]);
                    w_idx[ow] = nearest_idx(coord, iW);
                }

                int NC = N * C;
                nnr::for_static(0, NC * oH, NC * oH > 4, [&](int work) {
                    int nc = work / oH;
                    int oh = work % oH;
                    const T* src = px + (size_t)nc * iH * iW + (size_t)h_idx[oh] * iW;
                    T* dst = py + (size_t)nc * oH * oW + (size_t)oh * oW;
                    for (int ow = 0; ow < oW; ow++)
                        dst[ow] = src[w_idx[ow]];
                });
                return true;
            }

            // Generic fallback: per-element coordinate transform
            for (int flat = 0; flat < total; ++flat) {
                y->offset_to_indices(flat, out_idx);
                int offset = 0;
                for (int d = 0; d < ndim; ++d) {
                    float coord = transform_coord((float)out_idx[d], x->dims[d], y->dims[d], scales[d],
                        roi_data[d], roi_data[d + ndim]);
                    int idx = nearest_idx(coord, x->dims[d]);
                    offset += idx * x->strides[d];
                }
                py[flat] = px[offset];
            }
        } else if (mode == linear && !antialias) {
            int num_corners = 1 << ndim;
            for (int flat = 0; flat < total; ++flat) {
                y->offset_to_indices(flat, out_idx);

                if (coord_mode == tf_crop_and_resize) {
                    // Check extrapolation
                    bool extrapolate = false;
                    for (int d = 0; d < ndim; ++d) {
                        float coord = transform_coord((float)out_idx[d], x->dims[d], y->dims[d], scales[d],
                            roi_data[d], roi_data[d + ndim]);
                        if (coord < 0 || coord > x->dims[d] - 1) {
                            extrapolate = true;
                            break;
                        }
                    }
                    if (extrapolate) {
                        py[flat] = (T)extrapolation_value;
                        continue;
                    }
                }

                double result = 0;
                for (int corner = 0; corner < num_corners; ++corner) {
                    double weight = 1.0;
                    int offset = 0;
                    for (int d = 0; d < ndim; ++d) {
                        float coord = transform_coord((float)out_idx[d], x->dims[d], y->dims[d], scales[d],
                            roi_data[d], roi_data[d + ndim]);
                        int low = (int)std::floor(coord);
                        float frac = coord - low;
                        int idx = (corner & (1 << d)) ? low + 1 : low;
                        weight *= (corner & (1 << d)) ? frac : (1.0f - frac);
                        idx = std::clamp(idx, 0, x->dims[d] - 1);
                        offset += idx * x->strides[d];
                    }
                    result += weight * (double)px[offset];
                }
                py[flat] = (T)result;
            }
        } else if (mode == cubic && !antialias) {
            for (int flat = 0; flat < total; ++flat) {
                y->offset_to_indices(flat, out_idx);
                float a = cubic_coeff_a;

                if (ndim == 4) {
                    int n = out_idx[0], c = out_idx[1];
                    float hy = transform_coord((float)out_idx[2], x->dims[2], y->dims[2], scales[2],
                        roi_data[2], roi_data[2 + ndim]);
                    float hx = transform_coord((float)out_idx[3], x->dims[3], y->dims[3], scales[3],
                        roi_data[3], roi_data[3 + ndim]);
                    int fy = (int)std::floor(hy);
                    int fx = (int)std::floor(hx);
                    float dy = hy - fy;
                    float dx = hx - fx;
                    double sum = 0, wsum = 0;
                    for (int j = -1; j <= 2; ++j) {
                        for (int i = -1; i <= 2; ++i) {
                            float wy = cubic_weight(dy - j, a);
                            float wx = cubic_weight(dx - i, a);
                            float w = wy * wx;
                            int sy = fy + j, sx = fx + i;
                            if (exclude_outside && (sy < 0 || sy >= x->dims[2] || sx < 0 || sx >= x->dims[3]))
                                continue;
                            sy = std::clamp(sy, 0, x->dims[2] - 1);
                            sx = std::clamp(sx, 0, x->dims[3] - 1);
                            sum += w * (double)px[((n * x->dims[1] + c) * x->dims[2] + sy) * x->dims[3] + sx];
                            wsum += w;
                        }
                    }
                    if (exclude_outside && wsum != 0) sum /= wsum;
                    py[flat] = (T)sum;
                } else {
                    // Fallback: nearest
                    int offset = 0;
                    for (int d = 0; d < ndim; ++d) {
                        float coord = transform_coord((float)out_idx[d], x->dims[d], y->dims[d], scales[d],
                            roi_data[d], roi_data[d + ndim]);
                        int idx = std::clamp((int)std::round(coord), 0, x->dims[d] - 1);
                        offset += idx * x->strides[d];
                    }
                    py[flat] = px[offset];
                }
            }
        } else if (antialias && (mode == linear || mode == cubic)) {
            // Antialias: use wider kernel when downsampling
            exec_antialias(px, py, x, y, ndim, scales, roi_data);
        } else {
            // antialias + nearest: just do nearest
            for (int flat = 0; flat < total; ++flat) {
                y->offset_to_indices(flat, out_idx);
                int offset = 0;
                for (int d = 0; d < ndim; ++d) {
                    float coord = transform_coord((float)out_idx[d], x->dims[d], y->dims[d], scales[d],
                        roi_data[d], roi_data[d + ndim]);
                    int idx = nearest_idx(coord, x->dims[d]);
                    offset += idx * x->strides[d];
                }
                py[flat] = px[offset];
            }
        }
        return true;
    }

    template <typename T>
    void exec_antialias(const T* px, T* py, const tensor_t* x, const tensor_t* y,
                        int ndim, const float* scales, const float* roi_data) {
        // Only handle NCHW (ndim=4) with antialias on spatial dims (2,3)
        if (ndim != 4) {
            // Fallback: no antialias
            int total = (int)y->ndata;
            small_vector<int> out_idx(ndim);
            for (int flat = 0; flat < total; ++flat) {
                y->offset_to_indices(flat, out_idx);
                int offset = 0;
                for (int d = 0; d < ndim; ++d) {
                    float coord = transform_coord((float)out_idx[d], x->dims[d], y->dims[d], scales[d],
                        roi_data[d], roi_data[d + ndim]);
                    int idx = std::clamp((int)std::round(coord), 0, x->dims[d] - 1);
                    offset += idx * x->strides[d];
                }
                py[flat] = px[offset];
            }
            return;
        }

        float a = cubic_coeff_a;
        int N = x->dims[0], C = x->dims[1];
        int IH = x->dims[2], IW = x->dims[3];
        int OH = y->dims[2], OW = y->dims[3];

        // For antialias, we widen the kernel by 1/scale when scale < 1
        float scale_h = scales[2], scale_w = scales[3];
        float support_h = (mode == cubic) ? 2.0f : 1.0f;
        float support_w = (mode == cubic) ? 2.0f : 1.0f;
        if (scale_h < 1.0f) support_h /= scale_h;
        if (scale_w < 1.0f) support_w /= scale_w;

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                const T* src = px + (n * C + c) * IH * IW;
                T* dst = py + (n * C + c) * OH * OW;
                for (int oy = 0; oy < OH; ++oy) {
                    float fy = transform_coord((float)oy, IH, OH, scale_h,
                        roi_data[2], roi_data[2 + ndim]);
                    for (int ox = 0; ox < OW; ++ox) {
                        float fx = transform_coord((float)ox, IW, OW, scale_w,
                            roi_data[3], roi_data[3 + ndim]);

                        double sum = 0, wsum = 0;
                        int iy_min = (int)std::floor(fy - support_h);
                        int iy_max = (int)std::ceil(fy + support_h);
                        int ix_min = (int)std::floor(fx - support_w);
                        int ix_max = (int)std::ceil(fx + support_w);

                        for (int iy = iy_min; iy <= iy_max; ++iy) {
                            float dy = fy - iy;
                            if (scale_h < 1.0f) dy *= scale_h;
                            float wy;
                            if (mode == cubic) wy = cubic_weight(dy, a);
                            else wy = std::max(0.0f, 1.0f - std::abs(dy)); // linear

                            if (wy == 0) continue;

                            for (int ix = ix_min; ix <= ix_max; ++ix) {
                                float dx = fx - ix;
                                if (scale_w < 1.0f) dx *= scale_w;
                                float wx;
                                if (mode == cubic) wx = cubic_weight(dx, a);
                                else wx = std::max(0.0f, 1.0f - std::abs(dx));

                                if (wx == 0) continue;
                                float w = wy * wx;

                                int sy = iy, sx = ix;
                                if (exclude_outside && (sy < 0 || sy >= IH || sx < 0 || sx >= IW))
                                    continue;
                                sy = std::clamp(sy, 0, IH - 1);
                                sx = std::clamp(sx, 0, IW - 1);
                                sum += w * (double)src[sy * IW + sx];
                                wsum += w;
                            }
                        }
                        dst[oy * OW + ox] = (wsum != 0) ? (T)(sum / wsum) : T(0);
                    }
                }
            }
        }
    }

    bool exec() override {
        return typed_exec<Resize_operator,
            opset_t<13, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float16_t, float, double, bfloat16_t>,
            opset_t<10, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=static
operator_t* resolver_default_op_Resize(int opset, pool_t& pool) { return pool_new<Resize_operator>(pool); }

} // namespace nnr
