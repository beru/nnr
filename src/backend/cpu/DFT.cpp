#include <cmath>
#include <cstring>
#include <vector>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

static constexpr double PI = 3.14159265358979323846;

struct DFT_operator : public operator_t {
    int is_inverse;
    int is_onesided;

    bool init() override {
        if (inputs.empty() || outputs.empty()) return false;
        is_inverse = attribute(attr_key_t::inverse, (int32_t)0);
        is_onesided = attribute(attr_key_t::onesided, (int32_t)0);
        return true;
    }

    int get_axis() {
        const tensor_t* x = inputs[0];
        int axis = attribute(attr_key_t::axis, (int32_t)1);
        // Opset 20+: axis comes from input[2]
        if (inputs.size() >= 3 && inputs[2] && inputs[2]->ndata > 0) {
            if (inputs[2]->type == NNR_DATA_TYPE_INT64)
                axis = (int)*(const int64_t*)inputs[2]->data;
            else if (inputs[2]->type == NNR_DATA_TYPE_INT32)
                axis = *(const int32_t*)inputs[2]->data;
        }
        if (axis < 0) axis += x->ndim;
        return axis;
    }

    int get_dft_length(int axis) {
        const tensor_t* x = inputs[0];
        int dft_length = x->dims[axis];
        if (inputs.size() >= 2 && inputs[1] && inputs[1]->ndata > 0) {
            if (inputs[1]->type == NNR_DATA_TYPE_INT64)
                dft_length = (int)*(const int64_t*)inputs[1]->data;
            else if (inputs[1]->type == NNR_DATA_TYPE_INT32)
                dft_length = *(const int32_t*)inputs[1]->data;
        }
        return dft_length;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];

        int axis = get_axis();
        int dft_length = get_dft_length(axis);

        // Determine if input is complex (last dim == 2) or real
        bool input_complex = x->dims[x->ndim - 1] == 2;

        small_vector<int> dims(x->ndim);
        for (int d = 0; d < x->ndim; ++d)
            dims[d] = x->dims[d];
        dims[axis] = dft_length;
        if (is_onesided && !is_inverse)
            dims[axis] = dft_length / 2 + 1;

        if (input_complex) {
            // Complex input -> complex output, same shape
            return y->reshape(dims, x->type);
        }

        // Real input -> complex output: replace last dim with 2
        dims[x->ndim - 1] = 2;
        return y->reshape(dims, x->type);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;

        int axis = get_axis();
        int N = x->dims[axis]; // input length along axis
        int dft_length = get_dft_length(axis);

        bool input_complex = x->dims[x->ndim - 1] == 2;

        int out_length = is_onesided && !is_inverse ? dft_length / 2 + 1 : dft_length;

        // Compute outer and inner sizes (exclude last dim = real/complex indicator)
        size_t outer = 1;
        for (int d = 0; d < axis; ++d) outer *= x->dims[d];

        size_t inner = 1;
        for (int d = axis + 1; d < x->ndim - 1; ++d) inner *= x->dims[d];

        size_t out_inner = 1;
        for (int d = axis + 1; d < y->ndim - 1; ++d) out_inner *= y->dims[d];

        int in_stride = input_complex ? 2 : 1;

        memset(py, 0, y->ndata * sizeof(T));

        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                for (int k = 0; k < out_length; ++k) {
                    double re = 0, im = 0;
                    for (int n = 0; n < N; ++n) {
                        double angle = -2.0 * PI * k * n / dft_length;
                        if (is_inverse) angle = -angle;
                        double cos_a = std::cos(angle);
                        double sin_a = std::sin(angle);

                        double xr, xi;
                        if (input_complex) {
                            size_t idx = (o * N + n) * inner * 2 + i * 2;
                            if (idx + 1 >= x->ndata) continue;
                            xr = (double)px[idx];
                            xi = (double)px[idx + 1];
                        } else {
                            size_t idx = (o * N + n) * inner + i;
                            if (idx >= x->ndata) continue;
                            xr = (double)px[idx];
                            xi = 0;
                        }

                        re += xr * cos_a - xi * sin_a;
                        im += xr * sin_a + xi * cos_a;
                    }

                    if (is_inverse && dft_length > 0) {
                        re /= dft_length;
                        im /= dft_length;
                    }

                    size_t out_idx = (o * out_length + k) * out_inner * 2 + i * 2;
                    if (out_idx + 1 < y->ndata) {
                        py[out_idx] = (T)re;
                        py[out_idx + 1] = (T)im;
                    }
                }
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<DFT_operator,
            float16_t, float, double
        >(this, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_DFT(int opset, pool_t& pool) { return pool_new<DFT_operator>(pool); }

} // namespace nnr
