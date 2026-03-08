#include "nnr.h"
#include "util.h"
#include <cstring>

namespace nnr {

namespace {

struct Einsum_operator : public operator_t {
    static constexpr int MAX_INPUTS = 4;
    static constexpr int MAX_LABELS = 52;

    // Pre-parsed at init time from equation attribute:
    int input_labels[MAX_INPUTS][MAX_NDIM]; // input_labels[inp][dim] = label index
    int input_ndim[MAX_INPUTS];
    int n_inputs_parsed = 0;
    int output_labels[MAX_NDIM];
    int output_ndim = 0;
    int num_labels = 0;

    bool init() override {
        if (inputs.empty() || outputs.size() != 1)
            return false;
        std::string_view eq_attr = attribute(attr_key_t::equation, "");
        if (eq_attr.empty())
            return false;

        // Copy equation into a writable buffer, stripping spaces
        char eq_buf[256] = {};
        int eq_len = 0;
        for (char c : eq_attr) {
            if (c != ' ') {
                if (eq_len >= 250) return false;
                eq_buf[eq_len++] = c;
            }
        }

        // Split into inputs_str and output_str at "->"
        char inputs_str[200] = {};
        char output_str[64]  = {};
        bool has_output = false;
        const char* arrow = strstr(eq_buf, "->");
        if (arrow) {
            int ilen = (int)(arrow - eq_buf);
            memcpy(inputs_str, eq_buf, ilen);
            inputs_str[ilen] = 0;
            strncpy(output_str, arrow + 2, sizeof(output_str) - 1);
            has_output = true;
        } else {
            strncpy(inputs_str, eq_buf, sizeof(inputs_str) - 1);
        }

        // Check for ellipsis
        bool has_ellipsis = (strstr(inputs_str, "...") != nullptr);

        // Split inputs_str by ',' into input_parts
        char input_parts[MAX_INPUTS][MAX_NDIM * 2 + 4] = {};
        int n_parts = 0;
        {
            const char* p = inputs_str;
            if (*p == 0) {
                // Scalar: one empty subscript
                input_parts[n_parts++][0] = 0;
            } else {
                while (*p) {
                    const char* comma = strchr(p, ',');
                    if (!comma) comma = p + strlen(p);
                    int len = (int)(comma - p);
                    if (n_parts >= MAX_INPUTS || len >= (int)sizeof(input_parts[0]))
                        return false;
                    memcpy(input_parts[n_parts], p, len);
                    input_parts[n_parts][len] = 0;
                    n_parts++;
                    p = *comma ? comma + 1 : comma;
                }
            }
        }

        if (n_parts != (int)inputs.size())
            return false;

        // Handle ellipsis expansion
        if (has_ellipsis) {
            // Find the maximum ellipsis ndim across all inputs
            int ellipsis_ndim = 0;
            for (int i = 0; i < n_parts; ++i) {
                const char* ep = strstr(input_parts[i], "...");
                if (ep) {
                    int explicit_dims = (int)strlen(input_parts[i]) - 3;
                    int this_ndim = inputs[i]->ndim - explicit_dims;
                    if (this_ndim > ellipsis_ndim) ellipsis_ndim = this_ndim;
                }
            }

            // Generate uppercase labels for ellipsis dims: A, B, C, ...
            char ellipsis_labels[MAX_NDIM + 1] = {};
            for (int i = 0; i < ellipsis_ndim; ++i)
                ellipsis_labels[i] = (char)('A' + i);

            // Replace "..." in each input part
            for (int i = 0; i < n_parts; ++i) {
                char* ep = strstr(input_parts[i], "...");
                if (!ep) continue;
                int explicit_dims = (int)strlen(input_parts[i]) - 3;
                int this_ndim = inputs[i]->ndim - explicit_dims;
                // Use same ellipsis_labels for all; singleton broadcast dims also get the same labels
                int before = (int)(ep - input_parts[i]);
                int after  = (int)strlen(ep + 3);
                char tmp[MAX_NDIM * 2 + 4];
                memcpy(tmp, input_parts[i], before);
                for (int j = 0; j < ellipsis_ndim - this_ndim; ++j)
                    tmp[before + j] = ellipsis_labels[j]; // broadcast dims
                for (int j = 0; j < this_ndim; ++j)
                    tmp[before + (ellipsis_ndim - this_ndim) + j] = ellipsis_labels[(ellipsis_ndim - this_ndim) + j];
                memcpy(tmp + before + ellipsis_ndim, ep + 3, after + 1);
                strcpy(input_parts[i], tmp);
            }

            // Replace "..." in output_str
            char* ep = strstr(output_str, "...");
            if (ep) {
                int before = (int)(ep - output_str);
                int after  = (int)strlen(ep + 3);
                char tmp[MAX_NDIM * 2 + 4];
                memcpy(tmp, output_str, before);
                memcpy(tmp + before, ellipsis_labels, ellipsis_ndim);
                memcpy(tmp + before + ellipsis_ndim, ep + 3, after + 1);
                strcpy(output_str, tmp);
            }
        }

        // Map chars to label indices
        int char_to_label[128];
        memset(char_to_label, -1, sizeof(char_to_label));
        num_labels = 0;
        auto get_label = [&](char c) -> int {
            if (char_to_label[(unsigned char)c] == -1)
                char_to_label[(unsigned char)c] = num_labels++;
            return char_to_label[(unsigned char)c];
        };

        // Parse input labels into fixed arrays
        n_inputs_parsed = n_parts;
        for (int i = 0; i < n_parts; ++i) {
            const char* s = input_parts[i];
            input_ndim[i] = (int)strlen(s);
            if (input_ndim[i] > MAX_NDIM) return false;
            for (int j = 0; j < input_ndim[i]; ++j)
                input_labels[i][j] = get_label(s[j]);
        }

        // Parse output labels
        if (has_output) {
            output_ndim = (int)strlen(output_str);
            if (output_ndim > MAX_NDIM) return false;
            for (int j = 0; j < output_ndim; ++j)
                output_labels[j] = get_label(output_str[j]);
        } else {
            // Default: sorted unique labels that appear exactly once across all inputs
            int count[MAX_LABELS] = {};
            for (int i = 0; i < n_inputs_parsed; ++i)
                for (int d = 0; d < input_ndim[i]; ++d)
                    count[input_labels[i][d]]++;
            output_ndim = 0;
            for (int i = 0; i < num_labels; ++i)
                if (count[i] == 1)
                    output_labels[output_ndim++] = i;
        }

        return true;
    }

    bool reshape() override {
        // Fill label sizes from current input shapes
        int label_sizes[MAX_LABELS] = {};
        for (int i = 0; i < n_inputs_parsed; ++i)
            for (int d = 0; d < input_ndim[i]; ++d)
                label_sizes[input_labels[i][d]] = inputs[i]->dims[d];

        small_vector<int> dims(output_ndim);
        for (int i = 0; i < output_ndim; ++i)
            dims[i] = label_sizes[output_labels[i]];
        return outputs[0]->reshape(dims, inputs[0]->type);
    }

    template <typename T>
    bool exec() {
        // Fill label sizes from current input shapes
        int label_sizes[MAX_LABELS] = {};
        for (int i = 0; i < n_inputs_parsed; ++i)
            for (int d = 0; d < input_ndim[i]; ++d)
                label_sizes[input_labels[i][d]] = inputs[i]->dims[d];

        tensor_t* y = outputs[0];
        T* py = (T*)y->data;
        memset(py, 0, y->ndata * sizeof(T));

        // Iterate over all label combinations
        int64_t total = 1;
        for (int i = 0; i < num_labels; ++i)
            total *= label_sizes[i];

        int all_labels[MAX_LABELS] = {};
        for (int64_t iter = 0; iter < total; ++iter) {
            // Decode iteration index to per-label indices
            int64_t tmp = iter;
            for (int i = num_labels - 1; i >= 0; --i) {
                all_labels[i] = (int)(tmp % label_sizes[i]);
                tmp /= label_sizes[i];
            }

            // Multiply all input values at these label positions
            double prod = 1.0;
            for (int inp = 0; inp < n_inputs_parsed; ++inp) {
                const T* px = (const T*)inputs[inp]->data;
                int offset = 0;
                for (int d = 0; d < input_ndim[inp]; ++d)
                    offset += all_labels[input_labels[inp][d]] * inputs[inp]->strides[d];
                prod *= (double)px[offset];
            }

            // Accumulate into output
            int out_offset = 0;
            for (int d = 0; d < output_ndim; ++d)
                out_offset += all_labels[output_labels[d]] * y->strides[d];
            py[out_offset] += (T)prod;
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Einsum_operator,
            opset_t<12, uint8_t, uint16_t, uint32_t, uint64_t,
                int8_t, int16_t, int32_t, int64_t,
                float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Einsum(int opset, pool_t& pool)
{
    return pool_new<Einsum_operator>(pool);
}

} // namespace nnr
