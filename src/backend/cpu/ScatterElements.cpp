#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct ScatterElements_operator : public operator_t {
    int axis;
    int caxis;
    std::string_view reduction;

    bool init() override {
        if (!(inputs.size() == 3 && outputs.size() == 1)) {
            return false;
        }
        axis = attribute(attr_key_t::axis, (int32_t)0);
        reduction = attribute(attr_key_t::reduction, "none");
        return true;
    }

    bool reshape() override {
        const tensor_t* data = inputs[0];
        tensor_t* y = outputs[0];

        caxis = axis;
        if (caxis < 0) {
            caxis += data->ndim;
        }
        if (caxis < 0 || caxis >= data->ndim) {
            return false;
        }

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

        const int ndim = data->ndim;
        const size_t nidx = indices_t->ndata;
        bool idx_is_int32 = (indices_t->type == NNR_DATA_TYPE_INT32);

        small_vector<int> idx_indices(ndim);
        small_vector<int> out_indices(ndim);

        for (size_t i = 0; i < nidx; ++i) {
            indices_t->offset_to_indices(i, idx_indices);

            // Get the index value
            int64_t idx_val;
            if (idx_is_int32) {
                idx_val = ((const int32_t*)indices_t->data)[i];
            }else {
                idx_val = ((const int64_t*)indices_t->data)[i];
            }

            // Handle negative indices
            if (idx_val < 0) {
                idx_val += data->dims[caxis];
            }

            // Build output indices: same as idx_indices but replace axis dim with idx_val
            for (int d = 0; d < ndim; ++d) {
                out_indices[d] = idx_indices[d];
            }
            out_indices[caxis] = (int)idx_val;

            int out_off = y->indices_to_offset(out_indices);

            if constexpr (std::is_arithmetic_v<T>) {
                if (reduction == "add") {
                    py[out_off] += pupdates[i];
                }else if (reduction == "mul") {
                    py[out_off] *= pupdates[i];
                }else if (reduction == "min") {
                    if (pupdates[i] < py[out_off]) py[out_off] = pupdates[i];
                }else if (reduction == "max") {
                    if (pupdates[i] > py[out_off]) py[out_off] = pupdates[i];
                }else {
                    py[out_off] = pupdates[i];
                }
            } else {
                py[out_off] = pupdates[i];
            }
        }

        return true;
    }

    bool exec() override {
        return typed_exec<ScatterElements_operator,
            opset_t<13, bool_t, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float16_t, float, double, bfloat16_t, std::complex<float>, std::complex<double>, std::string>,
            opset_t<1, bool_t, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float16_t, float, double, std::complex<float>, std::complex<double>, std::string>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_ScatterElements(int opset, pool_t& pool)
{
    return pool_new<ScatterElements_operator>(pool);
}

} // namespace nnr
