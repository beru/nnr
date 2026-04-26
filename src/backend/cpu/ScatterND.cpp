#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct ScatterND_operator : public operator_t {
    std::string_view reduction;

    bool init() override {
        if (!(inputs.size() == 3 && outputs.size() == 1)) {
            return false;
        }
        reduction = attribute(attr_key_t::reduction, "none");
        return true;
    }

    bool reshape() override {
        const tensor_t* data = inputs[0];
        tensor_t* y = outputs[0];
        return y->reshape_identity(data);
    }

    template <typename T>
    bool exec() {
        const tensor_t* data = inputs[0];
        const tensor_t* indices_t = inputs[1];
        const tensor_t* updates = inputs[2];
        tensor_t* y = outputs[0];
        const T* pdata = (const T*)data->data;
        const T* pupdates = (const T*)updates->data;
        T* py = (T*)y->data;

        // Copy data to output
        for (size_t i = 0; i < data->ndata; ++i) {
            py[i] = pdata[i];
        }

        const int data_ndim = data->ndim;
        const int idx_ndim = indices_t->ndim;

        // ONNX requires idx_ndim >= 1; without the guard, reading
        // indices.dims[-1] below would read an adjacent stack slot.
        if (idx_ndim < 1) return false;

        // indices shape: [i_0, i_1, ..., i_{q-2}, last_dim]
        // last_dim = indices.dims[idx_ndim - 1], must be <= data_ndim.
        const int last_dim = indices_t->dims[idx_ndim - 1];
        if (last_dim < 0 || last_dim > data_ndim) return false;

        // Number of index tuples = product of indices.dims[0..idx_ndim-2]
        int64_t num_tuples = 1;
        for (int i = 0; i < idx_ndim - 1; ++i) {
            num_tuples *= indices_t->dims[i];
        }

        // Slice size = product of data.dims[last_dim .. data_ndim-1]
        int64_t slice_size = 1;
        for (int d = last_dim; d < data_ndim; ++d) {
            slice_size *= data->dims[d];
        }

        const data_type_t idx_type = indices_t->type;
        const void* idx_raw = indices_t->data;

        // Fetch one element from the index tensor, promoting the storage
        // type to int64_t. ONNX spec historically allowed int32 and int64 for
        // ScatterND indices (different opsets); supporting both here prevents
        // a misread when the exporter emitted int32.
        auto read_idx = [&](int64_t flat_pos) -> int64_t {
            if (idx_type == NNR_DATA_TYPE_INT32) {
                return ((const int32_t*)idx_raw)[flat_pos];
            }
            // Default to int64 (previously assumed unconditionally).
            return ((const int64_t*)idx_raw)[flat_pos];
        };

        for (int64_t t = 0; t < num_tuples; ++t) {
            // Compute the base offset into data from the index tuple
            int64_t base_offset = 0;
            int64_t stride = 1;
            bool oob = false;
            for (int d = last_dim - 1; d >= 0; --d) {
                int64_t idx_val = read_idx(t * last_dim + d);
                if (idx_val < 0) {
                    idx_val += data->dims[d];
                }
                // Bounds check after negative-index normalisation: an
                // out-of-range attacker-supplied value would otherwise flow
                // into py[di] as an arbitrary write.
                if (idx_val < 0 || idx_val >= data->dims[d]) { oob = true; break; }
                base_offset += idx_val * stride;
                stride *= data->dims[d];
            }
            if (oob) continue;
            // Full offset including remaining dims: base_offset * slice_size
            int64_t data_start = base_offset * slice_size;
            int64_t update_start = t * slice_size;

            for (int64_t s = 0; s < slice_size; ++s) {
                int64_t di = data_start + s;
                int64_t ui = update_start + s;
                if constexpr (std::is_arithmetic_v<T>) {
                    if (reduction == "add") {
                        py[di] += pupdates[ui];
                    }else if (reduction == "mul") {
                        py[di] *= pupdates[ui];
                    }else if (reduction == "min") {
                        if (pupdates[ui] < py[di]) py[di] = pupdates[ui];
                    }else if (reduction == "max") {
                        if (pupdates[ui] > py[di]) py[di] = pupdates[ui];
                    }else {
                        py[di] = pupdates[ui];
                    }
                } else {
                    py[di] = pupdates[ui];
                }
            }
        }

        return true;
    }

    bool exec() override {
        return typed_exec<ScatterND_operator,
            opset_t<13, bool_t, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float16_t, float, double, bfloat16_t, std::complex<float>, std::complex<double>, std::string>,
            opset_t<11, bool_t, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float16_t, float, double, std::complex<float>, std::complex<double>, std::string>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_ScatterND(int opset, pool_t& pool)
{
    return pool_new<ScatterND_operator>(pool);
}

} // namespace nnr
