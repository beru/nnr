#include <cmath>
#include <vector>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

static constexpr double PI = 3.14159265358979323846;

struct STFT_operator : public operator_t {
    int onesided;

    bool init() override {
        if (inputs.size() < 2 || outputs.empty()) return false;
        onesided = attribute(attr_key_t::onesided, (int32_t)1);
        return true;
    }

    bool reshape() override {
        const tensor_t* signal = inputs[0]; // [batch, signal_length, 1] or [batch, signal_length, 2]
        int batch = signal->dims[0];
        int signal_length = signal->dims[1];
        int is_complex = (signal->ndim >= 3 && signal->dims[2] == 2) ? 1 : 0;

        // frame_step from input[1]
        int frame_step = 1;
        if (inputs[1]->type == NNR_DATA_TYPE_INT64)
            frame_step = (int)*(const int64_t*)inputs[1]->data;
        else if (inputs[1]->type == NNR_DATA_TYPE_INT32)
            frame_step = *(const int32_t*)inputs[1]->data;

        // frame_length from input[3] or window size from input[2]
        int frame_length = 0;
        if (inputs.size() >= 4 && inputs[3] && inputs[3]->ndata > 0) {
            if (inputs[3]->type == NNR_DATA_TYPE_INT64)
                frame_length = (int)*(const int64_t*)inputs[3]->data;
            else if (inputs[3]->type == NNR_DATA_TYPE_INT32)
                frame_length = *(const int32_t*)inputs[3]->data;
        } else if (inputs.size() >= 3 && inputs[2] && inputs[2]->ndata > 0) {
            frame_length = inputs[2]->dims[0];
        }
        if (frame_length <= 0) return false;

        int num_frames = (signal_length - frame_length) / frame_step + 1;
        int dft_out = onesided && !is_complex ? frame_length / 2 + 1 : frame_length;

        small_vector<int> dims(4);
        dims[0] = batch;
        dims[1] = num_frames;
        dims[2] = dft_out;
        dims[3] = 2;
        return outputs[0]->reshape(dims, signal->type);
    }

    template <typename T>
    bool exec() {
        const tensor_t* signal = inputs[0];
        tensor_t* y = outputs[0];
        const T* ps = (const T*)signal->data;
        T* py = (T*)y->data;

        int batch = signal->dims[0];
        int signal_length = signal->dims[1];
        int is_complex = (signal->ndim >= 3 && signal->dims[2] == 2) ? 1 : 0;
        int signal_stride = is_complex ? 2 : 1;

        int frame_step = 1;
        if (inputs[1]->type == NNR_DATA_TYPE_INT64)
            frame_step = (int)*(const int64_t*)inputs[1]->data;
        else if (inputs[1]->type == NNR_DATA_TYPE_INT32)
            frame_step = *(const int32_t*)inputs[1]->data;

        int frame_length = 0;
        const T* window = nullptr;
        if (inputs.size() >= 4 && inputs[3] && inputs[3]->ndata > 0) {
            if (inputs[3]->type == NNR_DATA_TYPE_INT64)
                frame_length = (int)*(const int64_t*)inputs[3]->data;
            else if (inputs[3]->type == NNR_DATA_TYPE_INT32)
                frame_length = *(const int32_t*)inputs[3]->data;
        }
        if (inputs.size() >= 3 && inputs[2] && inputs[2]->ndata > 0) {
            window = (const T*)inputs[2]->data;
            if (frame_length <= 0)
                frame_length = inputs[2]->dims[0];
        }
        if (frame_length <= 0) return false;

        int num_frames = (signal_length - frame_length) / frame_step + 1;
        int dft_out = onesided && !is_complex ? frame_length / 2 + 1 : frame_length;

        memset(py, 0, y->ndata * sizeof(T));

        for (int b = 0; b < batch; ++b) {
            for (int f = 0; f < num_frames; ++f) {
                int frame_start = f * frame_step;
                for (int k = 0; k < dft_out; ++k) {
                    double re = 0, im = 0;
                    for (int n = 0; n < frame_length; ++n) {
                        int sig_idx = frame_start + n;
                        double xr, xi;
                        if (is_complex) {
                            size_t idx = (b * signal_length + sig_idx) * 2;
                            xr = (double)ps[idx];
                            xi = (double)ps[idx + 1];
                        } else {
                            size_t idx = b * signal_length * signal_stride + sig_idx;
                            xr = (double)ps[idx];
                            xi = 0;
                        }
                        if (window) {
                            double w = (double)window[n];
                            xr *= w;
                            xi *= w;
                        }
                        double angle = -2.0 * PI * k * n / frame_length;
                        re += xr * std::cos(angle) - xi * std::sin(angle);
                        im += xr * std::sin(angle) + xi * std::cos(angle);
                    }
                    size_t out_idx = ((b * num_frames + f) * dft_out + k) * 2;
                    py[out_idx] = (T)re;
                    py[out_idx + 1] = (T)im;
                }
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<STFT_operator,
            float16_t, float, double
        >(this, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_STFT(int opset, pool_t& pool) { return pool_new<STFT_operator>(pool); }

} // namespace nnr
