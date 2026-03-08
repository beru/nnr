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
        const int64_t* pidx = (const int64_t*)indices_t->data;
        const T* pupdates = (const T*)updates->data;
        T* py = (T*)y->data;

        // Copy data to output
        for (size_t i = 0; i < data->ndata; ++i) {
            py[i] = pdata[i];
        }

        const int data_ndim = data->ndim;
        const int idx_ndim = indices_t->ndim;

        // indices shape: [i_0, i_1, ..., i_{q-2}, last_dim]
        // last_dim = indices.dims[idx_ndim - 1], must be <= data_ndim
        const int last_dim = indices_t->dims[idx_ndim - 1];

        // Number of index tuples = product of indices.dims[0..idx_ndim-2]
        int num_tuples = 1;
        for (int i = 0; i < idx_ndim - 1; ++i) {
            num_tuples *= indices_t->dims[i];
        }

        // Slice size = product of data.dims[last_dim .. data_ndim-1]
        int slice_size = 1;
        for (int d = last_dim; d < data_ndim; ++d) {
            slice_size *= data->dims[d];
        }

        for (int t = 0; t < num_tuples; ++t) {
            // Compute the base offset into data from the index tuple
            const int64_t* tuple = pidx + t * last_dim;
            int base_offset = 0;
            int stride = 1;
            for (int d = last_dim - 1; d >= 0; --d) {
                int64_t idx_val = tuple[d];
                if (idx_val < 0) {
                    idx_val += data->dims[d];
                }
                base_offset += (int)idx_val * stride;
                stride *= data->dims[d];
            }
            // But we need full offset including remaining dims
            // base_offset * slice_size gives us the start in the flattened data
            int data_start = base_offset * slice_size;
            int update_start = t * slice_size;

            for (int s = 0; s < slice_size; ++s) {
                int di = data_start + s;
                int ui = update_start + s;
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
