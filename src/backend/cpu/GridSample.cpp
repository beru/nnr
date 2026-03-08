#include <cmath>
#include <algorithm>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct GridSample_operator : public operator_t {
    int align_corners;
    int mode; // 0=bilinear, 1=nearest, 2=bicubic
    int padding_mode; // 0=zeros, 1=border, 2=reflection

    bool init() override {
        if (inputs.size() != 2 || outputs.empty()) return false;
        align_corners = attribute(attr_key_t::align_corners, (int32_t)0);
        std::string_view mode_str = attribute(attr_key_t::mode, std::string_view("bilinear"));
        std::string_view pad_str = attribute(attr_key_t::padding_mode, std::string_view("zeros"));
        if (mode_str == "nearest") mode = 1;
        else if (mode_str == "bicubic" || mode_str == "cubic") mode = 2;
        else mode = 0; // bilinear
        if (pad_str == "border") padding_mode = 1;
        else if (pad_str == "reflection") padding_mode = 2;
        else padding_mode = 0; // zeros
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];  // [N, C, D1, D2, ...] or [N, C, H, W]
        const tensor_t* grid = inputs[1]; // [N, D1_out, D2_out, ..., ndim_spatial] or [N, H_out, W_out, 2]
        tensor_t* y = outputs[0];

        int ndim = x->ndim;
        small_vector<int> dims(ndim);
        dims[0] = x->dims[0]; // N
        dims[1] = x->dims[1]; // C
        // Spatial dims come from grid
        for (int i = 2; i < ndim; ++i)
            dims[i] = grid->dims[i - 1];
        return y->reshape(dims, x->type);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        const tensor_t* grid = inputs[1];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        const T* pgrid = (const T*)grid->data;
        T* py = (T*)y->data;

        int ndim_spatial = x->ndim - 2;

        if (ndim_spatial == 2) {
            return exec_2d(px, pgrid, py, x, grid, y);
        } else if (ndim_spatial == 3) {
            return exec_3d(px, pgrid, py, x, grid, y);
        }
        return false;
    }

    double unnormalize(double coord, int size) {
        if (align_corners) {
            return (coord + 1.0) / 2.0 * (size - 1);
        } else {
            return ((coord + 1.0) * size - 1.0) / 2.0;
        }
    }

    double reflect(double x, double lo, double hi) {
        double range = hi - lo;
        if (range == 0) return lo;
        // Reflect x into [lo, hi]
        double dx = x - lo;
        // Handle negative
        if (dx < 0) dx = -dx;
        // Modulo 2*range
        double period = 2.0 * range;
        dx = std::fmod(dx, period);
        if (dx > range) dx = period - dx;
        return lo + dx;
    }

    double compute_coord(double coord, int size) {
        double x = unnormalize(coord, size);
        if (padding_mode == 1) { // border
            x = std::max(0.0, std::min(x, (double)(size - 1)));
        } else if (padding_mode == 2) { // reflection
            if (align_corners) {
                x = reflect(x, 0.0, (double)(size - 1));
            } else {
                x = reflect(x, -0.5, (double)size - 0.5);
            }
            x = std::max(0.0, std::min(x, (double)(size - 1)));
        }
        return x;
    }

    template <typename T>
    T sample_2d(const T* px, int N_idx, int C_idx, double fy, double fx, int H, int W, int C) {
        int iy = (int)std::floor(fy);
        int ix = (int)std::floor(fx);

        auto get_pixel = [&](int y, int x) -> double {
            if (y < 0 || y >= H || x < 0 || x >= W) return 0.0;
            return (double)px[((N_idx * C + C_idx) * H + y) * W + x];
        };

        if (mode == 1) { // nearest
            int ny = (int)std::rint(fy);
            int nx = (int)std::rint(fx);
            return (T)get_pixel(ny, nx);
        } else if (mode == 0) { // bilinear
            double ly = fy - iy;
            double lx = fx - ix;
            double v = get_pixel(iy, ix) * (1 - ly) * (1 - lx)
                     + get_pixel(iy, ix + 1) * (1 - ly) * lx
                     + get_pixel(iy + 1, ix) * ly * (1 - lx)
                     + get_pixel(iy + 1, ix + 1) * ly * lx;
            return (T)v;
        } else { // bicubic
            auto cubic = [](double x) -> double {
                // Keys cubic with a = -0.75
                double a = -0.75;
                x = std::abs(x);
                if (x <= 1.0) return ((a + 2) * x - (a + 3)) * x * x + 1;
                if (x < 2.0) return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
                return 0.0;
            };
            double sum = 0;
            for (int j = -1; j <= 2; ++j) {
                for (int i = -1; i <= 2; ++i) {
                    double w = cubic(fy - (iy + j)) * cubic(fx - (ix + i));
                    sum += get_pixel(iy + j, ix + i) * w;
                }
            }
            return (T)sum;
        }
    }

    template <typename T>
    bool exec_2d(const T* px, const T* pgrid, T* py,
                 const tensor_t* x, const tensor_t* grid, tensor_t* y) {
        int N = x->dims[0], C = x->dims[1], H = x->dims[2], W = x->dims[3];
        int oH = grid->dims[1], oW = grid->dims[2];

        for (int n = 0; n < N; ++n) {
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    int grid_idx = ((n * oH + oh) * oW + ow) * 2;
                    double gx = (double)pgrid[grid_idx];
                    double gy = (double)pgrid[grid_idx + 1];

                    double fx = compute_coord(gx, W);
                    double fy = compute_coord(gy, H);

                    for (int c = 0; c < C; ++c) {
                        int out_idx = ((n * C + c) * oH + oh) * oW + ow;
                        py[out_idx] = sample_2d(px, n, c, fy, fx, H, W, C);
                    }
                }
            }
        }
        return true;
    }

    template <typename T>
    bool exec_3d(const T* px, const T* pgrid, T* py,
                 const tensor_t* x, const tensor_t* grid, tensor_t* y) {
        int N = x->dims[0], C = x->dims[1], D = x->dims[2], H = x->dims[3], W = x->dims[4];
        int oD = grid->dims[1], oH = grid->dims[2], oW = grid->dims[3];

        auto get_voxel = [&](int n, int c, int d, int h, int w) -> double {
            if (d < 0 || d >= D || h < 0 || h >= H || w < 0 || w >= W) return 0.0;
            return (double)px[(((n * C + c) * D + d) * H + h) * W + w];
        };

        for (int n = 0; n < N; ++n) {
            for (int od = 0; od < oD; ++od) {
                for (int oh = 0; oh < oH; ++oh) {
                    for (int ow = 0; ow < oW; ++ow) {
                        int grid_idx = (((n * oD + od) * oH + oh) * oW + ow) * 3;
                        double gx = (double)pgrid[grid_idx];
                        double gy = (double)pgrid[grid_idx + 1];
                        double gz = (double)pgrid[grid_idx + 2];

                        double fx = compute_coord(gx, W);
                        double fy = compute_coord(gy, H);
                        double fz = compute_coord(gz, D);

                        for (int c = 0; c < C; ++c) {
                            double val;
                            if (mode == 1) { // nearest
                                int nz = (int)std::rint(fz), ny = (int)std::rint(fy), nx = (int)std::rint(fx);
                                val = get_voxel(n, c, nz, ny, nx);
                            } else { // bilinear (trilinear for 3D)
                                int iz = (int)std::floor(fz), iy = (int)std::floor(fy), ix = (int)std::floor(fx);
                                double lz = fz - iz, ly = fy - iy, lx = fx - ix;
                                val = 0;
                                for (int dz = 0; dz <= 1; ++dz) {
                                    for (int dy = 0; dy <= 1; ++dy) {
                                        for (int dx = 0; dx <= 1; ++dx) {
                                            double w = (dz ? lz : 1-lz) * (dy ? ly : 1-ly) * (dx ? lx : 1-lx);
                                            val += get_voxel(n, c, iz+dz, iy+dy, ix+dx) * w;
                                        }
                                    }
                                }
                            }
                            int out_idx = (((n * C + c) * oD + od) * oH + oh) * oW + ow;
                            py[out_idx] = (T)val;
                        }
                    }
                }
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<GridSample_operator,
            float16_t, float, double
        >(this, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_GridSample(int opset, pool_t& pool) { return pool_new<GridSample_operator>(pool); }

} // namespace nnr
