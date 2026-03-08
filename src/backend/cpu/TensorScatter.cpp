#include <cmath>
#include <cstring>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct TensorScatter_operator : public operator_t {
    int scatter_mode; // 0=linear, 1=circular

    bool init() override {
        if (inputs.size() < 3 || outputs.empty()) return false;
        std::string_view mode_str = attribute(attr_key_t::mode, std::string_view("linear"));
        scatter_mode = (mode_str == "circular") ? 1 : 0;
        return true;
    }

    bool reshape() override {
        return outputs[0]->reshape_identity(inputs[0]);
    }

    template <typename T>
    bool exec() {
        const tensor_t* past = inputs[0]; // [batch..., seq_len, ...]
        const tensor_t* update = inputs[1]; // [batch..., update_len, ...]
        const tensor_t* indices = inputs[2]; // [batch_size] int64
        tensor_t* y = outputs[0];
        const T* pp = (const T*)past->data;
        const T* pu = (const T*)update->data;
        const int64_t* pi = (const int64_t*)indices->data;
        T* py = (T*)y->data;

        // Copy past to output
        memcpy(py, pp, past->ndata * sizeof(T));

        // TensorScatter: scatter update into output
        // past shape: [B1, B2, ..., seq_len, D1, D2, ...]
        // update shape: [B1, B2, ..., update_len, D1, D2, ...]
        // indices shape: [B1, B2, ...]
        // The scatter axis is the first axis after the batch dims covered by indices

        int ndim = past->ndim;
        // Find scatter axis: the axis where past and update shapes differ
        int scatter_axis = 0;
        for (int d = 0; d < ndim; ++d) {
            if (past->dims[d] != update->dims[d]) {
                scatter_axis = d;
                break;
            }
        }

        // Compute batch_size, seq_len, update_len, inner_size
        size_t batch_size = 1;
        for (int d = 0; d < scatter_axis; ++d) batch_size *= past->dims[d];

        int seq_len = past->dims[scatter_axis];
        int update_len = update->dims[scatter_axis];

        size_t inner_size = 1;
        for (int d = scatter_axis + 1; d < ndim; ++d) inner_size *= past->dims[d];

        size_t past_batch_stride = 1;
        for (int d = scatter_axis; d < ndim; ++d) past_batch_stride *= past->dims[d];

        size_t update_batch_stride = 1;
        for (int d = scatter_axis; d < update->ndim; ++d) update_batch_stride *= update->dims[d];

        for (size_t b = 0; b < batch_size; ++b) {
            int64_t write_pos = pi[b];

            for (int u = 0; u < update_len; ++u) {
                int64_t target;
                if (scatter_mode == 1) { // circular
                    target = (write_pos + u) % seq_len;
                    if (target < 0) target += seq_len;
                } else { // linear
                    target = write_pos + u;
                    if (target < 0 || target >= seq_len) continue;
                }

                size_t src_off = b * update_batch_stride + u * inner_size;
                size_t dst_off = b * past_batch_stride + target * inner_size;
                memcpy(py + dst_off, pu + src_off, inner_size * sizeof(T));
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<TensorScatter_operator,
            int8_t, int16_t, int32_t, int64_t,
            uint8_t, uint16_t, uint32_t, uint64_t,
            float16_t, bfloat16_t, float, double
        >(this, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_TensorScatter(int opset, pool_t& pool) { return pool_new<TensorScatter_operator>(pool); }

} // namespace nnr
