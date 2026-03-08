#include <cmath>
#include <cstring>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Col2Im_operator : public operator_t {
    small_vector<int> dilations;
    small_vector<int> pads;
    small_vector<int> strides;

    bool init() override {
        if (inputs.size() < 3 || outputs.empty()) return false;
        int64_t* data;
        int n;
        n = attribute(attr_key_t::dilations, data);
        for (int i = 0; i < n; ++i) dilations.push_back((int)data[i]);
        n = attribute(attr_key_t::pads, data);
        for (int i = 0; i < n; ++i) pads.push_back((int)data[i]);
        n = attribute(attr_key_t::strides, data);
        for (int i = 0; i < n; ++i) strides.push_back((int)data[i]);
        return true;
    }

    bool reshape() override {
        const tensor_t* input = inputs[0]; // [N, C*prod(kernel), L]
        const tensor_t* image_shape_tensor = inputs[1]; // [ndim_spatial]
        tensor_t* y = outputs[0];

        const int64_t* image_shape = (const int64_t*)image_shape_tensor->data;
        int ndim_spatial = (int)image_shape_tensor->ndata;

        // Read block_shape from inputs[2]
        const int64_t* block_shape = (const int64_t*)inputs[2]->data;
        int block_prod = 1;
        for (int i = 0; i < ndim_spatial; ++i) block_prod *= (int)block_shape[i];

        int N = input->dims[0];
        int C = input->dims[1] / block_prod;

        small_vector<int> dims(2 + ndim_spatial);
        dims[0] = N;
        dims[1] = C;
        for (int i = 0; i < ndim_spatial; ++i)
            dims[2 + i] = (int)image_shape[i];

        return y->reshape(dims, input->type);
    }

    template <typename T>
    bool exec() {
        const tensor_t* input = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)input->data;
        T* py = (T*)y->data;

        const int64_t* image_shape = (const int64_t*)inputs[1]->data;
        const int64_t* block_shape = (const int64_t*)inputs[2]->data;
        int ndim_spatial = (int)inputs[1]->ndata;

        // Fill defaults
        while ((int)dilations.size() < ndim_spatial) dilations.push_back(1);
        while ((int)strides.size() < ndim_spatial) strides.push_back(1);
        while ((int)pads.size() < ndim_spatial * 2) pads.push_back(0);

        int N = y->dims[0], C = y->dims[1];
        memset(py, 0, y->ndata * sizeof(T));

        if (ndim_spatial == 2) {
            int H = y->dims[2], W = y->dims[3];
            int kH = (int)block_shape[0], kW = (int)block_shape[1];
            int dH = dilations[0], dW = dilations[1];
            int sH = strides[0], sW = strides[1];
            int pH0 = pads[0], pW0 = pads[1], pH1 = pads[2], pW1 = pads[3];

            int oH = (H + pH0 + pH1 - dH * (kH - 1) - 1) / sH + 1;
            int oW = (W + pW0 + pW1 - dW * (kW - 1) - 1) / sW + 1;

            for (int n = 0; n < N; ++n) {
                for (int c = 0; c < C; ++c) {
                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            int col_c = (c * kH + kh) * kW + kw;
                            for (int oh = 0; oh < oH; ++oh) {
                                for (int ow = 0; ow < oW; ++ow) {
                                    int ih = oh * sH + kh * dH - pH0;
                                    int iw = ow * sW + kw * dW - pW0;
                                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                        int col_idx = (n * C * kH * kW + col_c) * (oH * oW) + oh * oW + ow;
                                        int img_idx = ((n * C + c) * H + ih) * W + iw;
                                        py[img_idx] = (T)((double)py[img_idx] + (double)px[col_idx]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Generic N-D col2im
            // For simplicity, handle 1D and 3D
            small_vector<int> out_spatial(ndim_spatial);
            small_vector<int> kernel(ndim_spatial);
            for (int i = 0; i < ndim_spatial; ++i) {
                out_spatial[i] = (int)image_shape[i];
                kernel[i] = (int)block_shape[i];
            }

            // Compute output spatial size (number of patches)
            small_vector<int> patch_dims(ndim_spatial);
            int L = 1;
            for (int i = 0; i < ndim_spatial; ++i) {
                patch_dims[i] = (out_spatial[i] + pads[i] + pads[i + ndim_spatial] - dilations[i] * (kernel[i] - 1) - 1) / strides[i] + 1;
                L *= patch_dims[i];
            }

            int kernel_prod = 1;
            for (int i = 0; i < ndim_spatial; ++i) kernel_prod *= kernel[i];

            for (int n = 0; n < N; ++n) {
                for (int c = 0; c < C; ++c) {
                    // Iterate over kernel positions
                    small_vector<int> k_idx(ndim_spatial);
                    for (int k = 0; k < kernel_prod; ++k) {
                        int col_c = c * kernel_prod + k;
                        // Decompose k into kernel indices
                        int rem = k;
                        for (int d = ndim_spatial - 1; d >= 0; --d) {
                            k_idx[d] = rem % kernel[d];
                            rem /= kernel[d];
                        }

                        // Iterate over patches
                        small_vector<int> p_idx(ndim_spatial);
                        for (int l = 0; l < L; ++l) {
                            // Decompose l into patch indices
                            rem = l;
                            for (int d = ndim_spatial - 1; d >= 0; --d) {
                                p_idx[d] = rem % patch_dims[d];
                                rem /= patch_dims[d];
                            }

                            // Compute input position
                            bool valid = true;
                            small_vector<int> i_idx(ndim_spatial);
                            for (int d = 0; d < ndim_spatial; ++d) {
                                i_idx[d] = p_idx[d] * strides[d] + k_idx[d] * dilations[d] - pads[d];
                                if (i_idx[d] < 0 || i_idx[d] >= out_spatial[d]) {
                                    valid = false;
                                    break;
                                }
                            }

                            if (valid) {
                                int col_idx = (n * C * kernel_prod + col_c) * L + l;
                                // Compute image linear index
                                int img_idx = n * C;
                                img_idx = (img_idx + c);
                                for (int d = 0; d < ndim_spatial; ++d)
                                    img_idx = img_idx * out_spatial[d] + i_idx[d];
                                py[img_idx] = (T)((double)py[img_idx] + (double)px[col_idx]);
                            }
                        }
                    }
                }
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Col2Im_operator,
            float16_t, float, double
        >(this, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Col2Im(int opset, pool_t& pool) { return pool_new<Col2Im_operator>(pool); }

} // namespace nnr
