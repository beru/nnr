#include <cmath>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

static constexpr double PI = 3.14159265358979323846;

struct MelWeightMatrix_operator : public operator_t {
    data_type_t output_datatype;

    bool init() override {
        if (inputs.size() < 5 || outputs.empty()) return false;
        output_datatype = (data_type_t)attribute(attr_key_t::output_datatype, (int32_t)NNR_DATA_TYPE_FLOAT32);
        return true;
    }

    bool reshape() override {
        // Inputs: num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz
        int num_mel_bins = 0, dft_length = 0;
        auto read_int = [](const tensor_t* t) -> int {
            if (t->type == NNR_DATA_TYPE_INT32) return *(const int32_t*)t->data;
            if (t->type == NNR_DATA_TYPE_INT64) return (int)*(const int64_t*)t->data;
            return (int)*(const float*)t->data;
        };
        num_mel_bins = read_int(inputs[0]);
        dft_length = read_int(inputs[1]);
        int num_spectrogram_bins = dft_length / 2 + 1;

        small_vector<int> dims(2);
        dims[0] = num_spectrogram_bins;
        dims[1] = num_mel_bins;
        return outputs[0]->reshape(dims, output_datatype);
    }

    template <typename T>
    bool exec() {
        tensor_t* y = outputs[0];
        T* py = (T*)y->data;

        auto read_int = [](const tensor_t* t) -> int {
            if (t->type == NNR_DATA_TYPE_INT32) return *(const int32_t*)t->data;
            if (t->type == NNR_DATA_TYPE_INT64) return (int)*(const int64_t*)t->data;
            return (int)*(const float*)t->data;
        };
        auto read_float = [](const tensor_t* t) -> double {
            if (t->type == NNR_DATA_TYPE_FLOAT32) return *(const float*)t->data;
            if (t->type == NNR_DATA_TYPE_FLOAT64) return *(const double*)t->data;
            return (double)*(const int32_t*)t->data;
        };

        int num_mel_bins = read_int(inputs[0]);
        int dft_length = read_int(inputs[1]);
        double sample_rate = read_float(inputs[2]);
        double lower_edge = read_float(inputs[3]);
        double upper_edge = read_float(inputs[4]);

        int num_spectrogram_bins = dft_length / 2 + 1;
        arena_scope_t scope(ctx->arena);

        // Following ONNX reference implementation exactly
        double low_mel = 2595.0 * std::log10(1.0 + lower_edge / 700.0);
        double high_mel = 2595.0 * std::log10(1.0 + upper_edge / 700.0);
        double mel_step = (high_mel - low_mel) / (num_mel_bins + 2);

        // Compute frequency bin indices (integer)
        int* freq_bins = scope.alloc_arr<int>(num_mel_bins + 2);
        for (int i = 0; i < num_mel_bins + 2; ++i) {
            double mel = i * mel_step + low_mel;
            double hz = 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
            freq_bins[i] = (int)((dft_length + 1) * hz / sample_rate);
        }

        memset(py, 0, y->ndata * sizeof(T));

        for (int i = 0; i < num_mel_bins; ++i) {
            int left = freq_bins[i];
            int center = freq_bins[i + 1];
            int right = freq_bins[i + 2];

            int low_to_center = center - left;
            if (low_to_center == 0) {
                if (center >= 0 && center < num_spectrogram_bins)
                    py[center * num_mel_bins + i] = (T)1;
            } else {
                for (int j = left; j <= center && j < num_spectrogram_bins; ++j) {
                    if (j >= 0)
                        py[j * num_mel_bins + i] = (T)((double)(j - left) / (double)low_to_center);
                }
            }

            int center_to_high = right - center;
            if (center_to_high > 0) {
                for (int j = center; j < right && j < num_spectrogram_bins; ++j) {
                    if (j >= 0)
                        py[j * num_mel_bins + i] = (T)((double)(right - j) / (double)center_to_high);
                }
            }
        }

        return true;
    }

    bool exec() override {
        return typed_exec<MelWeightMatrix_operator,
            float16_t, float, double
        >(this, outputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no workspace=yes
operator_t* resolver_default_op_MelWeightMatrix(int opset, pool_t& pool) { return pool_new<MelWeightMatrix_operator>(pool); }

} // namespace nnr
