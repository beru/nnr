#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Gather_operator : public operator_t {
    int axis = 0;
    int caxis = 0;

    bool init() override {
        if (!is_inout_size(2, 1)) {
            return false;
        }
        axis = attribute(attr_key_t::axis, (int32_t)0);
        return true;
    }

    bool reshape() override {
        const tensor_t* data = inputs[0];
        const tensor_t* indices = inputs[1];
        tensor_t* y = outputs[0];

        caxis = axis;
        if (caxis < 0) {
            caxis += data->ndim;
        }
        if (caxis < 0 || caxis >= data->ndim) {
            return false;
        }

        int ondim = data->ndim - 1 + indices->ndim;
        small_vector<int> dims(ondim);

        int d = 0;
        for (int i = 0; i < caxis; ++i) {
            dims[d++] = data->dims[i];
        }
        for (int i = 0; i < indices->ndim; ++i) {
            dims[d++] = indices->dims[i];
        }
        for (int i = caxis + 1; i < data->ndim; ++i) {
            dims[d++] = data->dims[i];
        }

        return y->reshape(std::span<const int>(dims.data_, ondim), data->type);
    }

    template <typename T>
    bool exec() {
        tensor_t* y = outputs[0];
        const tensor_t* data = inputs[0];
        const tensor_t* indices = inputs[1];

        // Re-derive y from data's live dims — an upstream
        // NonMaxSuppression can shrink data->dims during exec(), and the
        // fast path skips reshape(). Without this y->ndata stays at the
        // prepare-time upper bound while data has shrunk, so the gather
        // walks past the live region (ssd-12-int8 NMS→Gather crash).
        // reshape() also re-types y if needed.
        if (!reshape()) return false;
        // Empty output: nothing to gather (e.g. data shrunk to 0 along a
        // pre-axis dim by NMS). Don't dereference data->data.
        if (y->ndata == 0) return true;

        if (!data || !indices || !y) return false;

        const int data_ndim = data->ndim;
        const int indices_ndim = indices->ndim;
        if (caxis >= data_ndim) return false;
        const int axis_dim = data->dims[caxis];
        // axis_dim==0 means upstream produced no rows to gather (e.g. NMS
        // returned no detections). y's other-axis dims may still be
        // non-zero, but there is no valid index into a 0-element axis;
        // leave the (uninitialized / pool-zeroed) output as-is and
        // propagate empty downstream.
        if (axis_dim <= 0) return true;
        if (!data->data || !indices->data || !y->data) return false;
        const T* pdata = (const T*)data->data;
        T* py = (T*)y->data;

        // Compute the number of elements in the prefix (before axis), indices, and suffix (after axis).
        // Widened to int64_t because outer * axis_dim * inner overflows int32 on large embedding
        // tables (vocab_size * hidden_dim easily exceeds 2^31 on real LLMs).
        int64_t outer = 1;
        for (int i = 0; i < caxis; ++i) {
            outer *= data->dims[i];
        }
        int64_t inner = 1;
        for (int i = caxis + 1; i < data_ndim; ++i) {
            inner *= data->dims[i];
        }
        int64_t indices_size = (int64_t)indices->ndata;

        // data layout: [outer, axis_dim, inner]
        // output layout: [outer, indices_size, inner]
        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t idx = 0; idx < indices_size; ++idx) {
                int64_t index;
                if (indices->type == NNR_DATA_TYPE_INT64) {
                    index = ((const int64_t*)indices->data)[idx];
                } else {
                    index = ((const int32_t*)indices->data)[idx];
                }
                if (index < 0) {
                    index += axis_dim;
                }
                if (index < 0 || index >= axis_dim) {
                    index = 0; // clamp OOB — produces garbage but won't crash
                }
                const T* src = pdata + (o * axis_dim + index) * inner;
                T* dst = py + (o * indices_size + idx) * inner;
                for (int64_t s = 0; s < inner; ++s) {
                    dst[s] = src[s];
                }
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Gather_operator,
            opset_t<13, bool_t,
                uint8_t, uint16_t, uint32_t, uint64_t,
                int8_t, int16_t, int32_t, int64_t,
                float16_t, float, double, bfloat16_t,
                std::complex<float>, std::complex<double>,
                std::string>,
            opset_t<1, bool_t,
                uint8_t, uint16_t, uint32_t, uint64_t,
                int8_t, int16_t, int32_t, int64_t,
                float16_t, float, double,
                std::complex<float>, std::complex<double>,
                std::string>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Gather(int opset, pool_t& pool)
{
    return pool_new<Gather_operator>(pool);
}

} // namespace nnr
