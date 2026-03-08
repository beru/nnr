#include <cmath>
#include <algorithm>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct RoiAlign_operator : public operator_t {
    int roi_mode; // 0=avg, 1=max
    int output_height;
    int output_width;
    int sampling_ratio;
    float spatial_scale;
    bool half_pixel;

    bool init() override {
        if (inputs.size() < 3 || outputs.empty()) return false;
        std::string_view mode_str = attribute(attr_key_t::mode, std::string_view("avg"));
        roi_mode = (mode_str == "max") ? 1 : 0;
        output_height = attribute(attr_key_t::output_height, (int32_t)1);
        output_width = attribute(attr_key_t::output_width, (int32_t)1);
        sampling_ratio = attribute(attr_key_t::sampling_ratio, (int32_t)0);
        spatial_scale = attribute(attr_key_t::spatial_scale, 1.0f);
        std::string_view ctm = attribute(attr_key_t::coordinate_transformation_mode, std::string_view("output_half_pixel"));
        half_pixel = (ctm == "half_pixel");
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        const tensor_t* rois = inputs[1];
        tensor_t* y = outputs[0];
        int num_rois = rois->dims[0];
        int C = x->dims[1];
        small_vector<int> dims(4);
        dims[0] = num_rois;
        dims[1] = C;
        dims[2] = output_height;
        dims[3] = output_width;
        return y->reshape(dims, x->type);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        const tensor_t* rois_tensor = inputs[1];
        const tensor_t* batch_indices_tensor = inputs[2];
        tensor_t* y = outputs[0];

        const T* px = (const T*)x->data;
        const T* prois = (const T*)rois_tensor->data;
        const int64_t* batch_indices = (const int64_t*)batch_indices_tensor->data;
        T* py = (T*)y->data;

        int C = x->dims[1], H = x->dims[2], W = x->dims[3];
        int num_rois = rois_tensor->dims[0];

        for (int n = 0; n < num_rois; ++n) {
            int batch = (int)batch_indices[n];
            double x1 = (double)prois[n * 4 + 0] * spatial_scale;
            double y1 = (double)prois[n * 4 + 1] * spatial_scale;
            double x2 = (double)prois[n * 4 + 2] * spatial_scale;
            double y2 = (double)prois[n * 4 + 3] * spatial_scale;

            if (half_pixel) {
                x1 -= 0.5; y1 -= 0.5; x2 -= 0.5; y2 -= 0.5;
            }

            double roi_h = y2 - y1;
            double roi_w = x2 - x1;
            if (!half_pixel) {
                roi_h = std::max(roi_h, 1.0);
                roi_w = std::max(roi_w, 1.0);
            }

            double bin_h = roi_h / output_height;
            double bin_w = roi_w / output_width;

            int sample_h = sampling_ratio > 0 ? sampling_ratio : std::max(1, (int)std::ceil(bin_h));
            int sample_w = sampling_ratio > 0 ? sampling_ratio : std::max(1, (int)std::ceil(bin_w));

            for (int c = 0; c < C; ++c) {
                for (int oh = 0; oh < output_height; ++oh) {
                    for (int ow = 0; ow < output_width; ++ow) {
                        double accum = roi_mode == 1 ? -1e38 : 0.0;
                        int count = 0;

                        for (int sh = 0; sh < sample_h; ++sh) {
                            for (int sw = 0; sw < sample_w; ++sw) {
                                double fy = y1 + bin_h * (oh + (sh + 0.5) / sample_h);
                                double fx = x1 + bin_w * (ow + (sw + 0.5) / sample_w);

                                if (fy < -1.0 || fy > H || fx < -1.0 || fx > W) {
                                    if (roi_mode == 0) count++;
                                    continue;
                                }

                                fy = std::max(fy, 0.0);
                                fx = std::max(fx, 0.0);
                                int iy = std::min((int)fy, H - 1);
                                int ix = std::min((int)fx, W - 1);
                                double ly = fy - iy;
                                double lx = fx - ix;
                                if (iy >= H - 1) { iy = H - 1; ly = 0; }
                                if (ix >= W - 1) { ix = W - 1; lx = 0; }

                                auto get = [&](int yy, int xx) -> double {
                                    if (yy < 0 || yy >= H || xx < 0 || xx >= W) return 0.0;
                                    return (double)px[((batch * C + c) * H + yy) * W + xx];
                                };

                                if (roi_mode == 1) {
                                    double w1 = (1-ly)*(1-lx), w2 = (1-ly)*lx;
                                    double w3 = ly*(1-lx), w4 = ly*lx;
                                    double v = std::max({w1*get(iy,ix), w2*get(iy,ix+1),
                                                         w3*get(iy+1,ix), w4*get(iy+1,ix+1)});
                                    accum = (count == 0) ? v : std::max(accum, v);
                                } else {
                                    double v = get(iy, ix) * (1-ly) * (1-lx)
                                             + get(iy, ix+1) * (1-ly) * lx
                                             + get(iy+1, ix) * ly * (1-lx)
                                             + get(iy+1, ix+1) * ly * lx;
                                    accum += v;
                                }
                                count++;
                            }
                        }

                        if (roi_mode == 0) {
                            accum /= (sample_h * sample_w);
                        }
                        if (roi_mode == 1 && count == 0) accum = 0.0;

                        int out_idx = ((n * C + c) * output_height + oh) * output_width + ow;
                        py[out_idx] = (T)accum;
                    }
                }
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<RoiAlign_operator,
            float16_t, float, double
        >(this, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_RoiAlign(int opset, pool_t& pool) { return pool_new<RoiAlign_operator>(pool); }

} // namespace nnr
