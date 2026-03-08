#include <cmath>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct AffineGrid_operator : public operator_t {
    int align_corners;

    bool init() override {
        if (inputs.size() < 2 || outputs.empty()) return false;
        align_corners = attribute(attr_key_t::align_corners, (int32_t)0);
        return true;
    }

    bool reshape() override {
        const tensor_t* theta = inputs[0]; // [N, 2, 3] or [N, 3, 4]
        const tensor_t* size = inputs[1]; // [4] or [5]
        tensor_t* y = outputs[0];

        const int64_t* sz = (const int64_t*)size->data;
        int ndim_spatial = (int)size->ndata - 2;

        if (ndim_spatial == 2) {
            // Output: [N, H, W, 2]
            small_vector<int> dims(4);
            dims[0] = (int)sz[0]; // N
            dims[1] = (int)sz[2]; // H
            dims[2] = (int)sz[3]; // W
            dims[3] = 2;
            return y->reshape(dims, theta->type);
        } else if (ndim_spatial == 3) {
            // Output: [N, D, H, W, 3]
            small_vector<int> dims(5);
            dims[0] = (int)sz[0]; // N
            dims[1] = (int)sz[2]; // D
            dims[2] = (int)sz[3]; // H
            dims[3] = (int)sz[4]; // W
            dims[4] = 3;
            return y->reshape(dims, theta->type);
        }
        return false;
    }

    template <typename T>
    bool exec() {
        const tensor_t* theta_tensor = inputs[0];
        const tensor_t* size_tensor = inputs[1];
        tensor_t* y = outputs[0];

        const T* theta = (const T*)theta_tensor->data;
        const int64_t* sz = (const int64_t*)size_tensor->data;
        T* py = (T*)y->data;

        int N = (int)sz[0];
        int ndim_spatial = (int)size_tensor->ndata - 2;

        if (ndim_spatial == 2) {
            int H = (int)sz[2], W = (int)sz[3];

            for (int n = 0; n < N; ++n) {
                const T* th = theta + n * 6; // [2, 3] row-major
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        double ny_coord, nx_coord;
                        if (align_corners) {
                            ny_coord = H > 1 ? 2.0 * h / (H - 1) - 1.0 : 0.0;
                            nx_coord = W > 1 ? 2.0 * w / (W - 1) - 1.0 : 0.0;
                        } else {
                            ny_coord = H > 0 ? 2.0 * (h + 0.5) / H - 1.0 : 0.0;
                            nx_coord = W > 0 ? 2.0 * (w + 0.5) / W - 1.0 : 0.0;
                        }
                        // theta * [nx, ny, 1]^T
                        double gx = (double)th[0] * nx_coord + (double)th[1] * ny_coord + (double)th[2];
                        double gy = (double)th[3] * nx_coord + (double)th[4] * ny_coord + (double)th[5];

                        int out_idx = ((n * H + h) * W + w) * 2;
                        py[out_idx + 0] = (T)gx;
                        py[out_idx + 1] = (T)gy;
                    }
                }
            }
        } else if (ndim_spatial == 3) {
            int D = (int)sz[2], H = (int)sz[3], W = (int)sz[4];

            for (int n = 0; n < N; ++n) {
                const T* th = theta + n * 12; // [3, 4] row-major
                for (int d = 0; d < D; ++d) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            double nz, ny_c, nx_c;
                            if (align_corners) {
                                nz = D > 1 ? 2.0 * d / (D - 1) - 1.0 : 0.0;
                                ny_c = H > 1 ? 2.0 * h / (H - 1) - 1.0 : 0.0;
                                nx_c = W > 1 ? 2.0 * w / (W - 1) - 1.0 : 0.0;
                            } else {
                                nz = D > 0 ? 2.0 * (d + 0.5) / D - 1.0 : 0.0;
                                ny_c = H > 0 ? 2.0 * (h + 0.5) / H - 1.0 : 0.0;
                                nx_c = W > 0 ? 2.0 * (w + 0.5) / W - 1.0 : 0.0;
                            }
                            double gx = (double)th[0] * nx_c + (double)th[1] * ny_c + (double)th[2] * nz + (double)th[3];
                            double gy = (double)th[4] * nx_c + (double)th[5] * ny_c + (double)th[6] * nz + (double)th[7];
                            double gz = (double)th[8] * nx_c + (double)th[9] * ny_c + (double)th[10] * nz + (double)th[11];

                            int out_idx = (((n * D + d) * H + h) * W + w) * 3;
                            py[out_idx + 0] = (T)gx;
                            py[out_idx + 1] = (T)gy;
                            py[out_idx + 2] = (T)gz;
                        }
                    }
                }
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<AffineGrid_operator,
            float16_t, float, double
        >(this, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_AffineGrid(int opset, pool_t& pool) { return pool_new<AffineGrid_operator>(pool); }

} // namespace nnr
