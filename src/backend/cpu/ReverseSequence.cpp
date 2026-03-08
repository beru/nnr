#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct ReverseSequence_operator : public operator_t {
    int batch_axis;
    int time_axis;

    bool init() override {
        if (!is_inout_size(2, 1)) return false;
        batch_axis = attribute(attr_key_t::batch_axis, (int32_t)1);
        time_axis = attribute(attr_key_t::time_axis, (int32_t)0);
        return true;
    }

    bool reshape() override {
        return outputs[0]->reshape_identity(inputs[0]);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        const tensor_t* seq_lens = inputs[1];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;
        const int64_t* psl = (const int64_t*)seq_lens->data;

        // Copy input to output first
        memcpy(py, px, x->ndata * sizeof(T));

        int ndim = x->ndim;
        int ba = batch_axis;
        int ta = time_axis;

        // Compute strides
        int strides[MAX_NDIM];
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * x->dims[i + 1];

        // For each batch element, reverse the first seq_lens[b] elements along time_axis
        int batch_size = x->dims[ba];
        int time_size = x->dims[ta];

        // Iterate over all elements
        small_vector<int> idx(ndim);
        for (size_t i = 0; i < x->ndata; ++i) {
            // Compute multi-dimensional index
            int rem = (int)i;
            for (int d = 0; d < ndim; ++d) {
                idx[d] = rem / strides[d];
                rem %= strides[d];
            }

            int b = idx[ba];
            int t = idx[ta];
            int sl = (int)psl[b];

            if (t < sl) {
                // Reverse: map t -> sl - 1 - t
                int new_t = sl - 1 - t;
                int src_offset = (int)i;
                int dst_offset = src_offset + (new_t - t) * strides[ta];
                py[dst_offset] = px[src_offset];
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<ReverseSequence_operator,
            int8_t, int16_t, int32_t, int64_t,
            uint8_t, uint16_t, uint32_t, uint64_t,
            float16_t, bfloat16_t, float, double
        >(this, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_ReverseSequence(int opset, pool_t& pool) { return pool_new<ReverseSequence_operator>(pool); }

} // namespace nnr
