#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct NonZero_operator : public operator_t {
    bool init() override {
        return is_inout_size(1, 1);
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const int ndim = x->ndim;

        // Count non-zero elements
        int count = 0;
        if (x->type == NNR_DATA_TYPE_FLOAT32) {
            const float* p = (const float*)x->data;
            for (size_t i = 0; i < x->ndata; ++i) if (p[i] != 0.0f) ++count;
        } else if (x->type == NNR_DATA_TYPE_INT64) {
            const int64_t* p = (const int64_t*)x->data;
            for (size_t i = 0; i < x->ndata; ++i) if (p[i] != 0) ++count;
        } else if (x->type == NNR_DATA_TYPE_INT32) {
            const int32_t* p = (const int32_t*)x->data;
            for (size_t i = 0; i < x->ndata; ++i) if (p[i] != 0) ++count;
        } else if (x->type == NNR_DATA_TYPE_BOOL) {
            const bool_t* p = (const bool_t*)x->data;
            for (size_t i = 0; i < x->ndata; ++i) if ((bool)p[i]) ++count;
        } else if (x->type == NNR_DATA_TYPE_FLOAT64) {
            const double* p = (const double*)x->data;
            for (size_t i = 0; i < x->ndata; ++i) if (p[i] != 0.0) ++count;
        } else if (x->type == NNR_DATA_TYPE_UINT8) {
            const uint8_t* p = (const uint8_t*)x->data;
            for (size_t i = 0; i < x->ndata; ++i) if (p[i] != 0) ++count;
        } else if (x->type == NNR_DATA_TYPE_INT8) {
            const int8_t* p = (const int8_t*)x->data;
            for (size_t i = 0; i < x->ndata; ++i) if (p[i] != 0) ++count;
        } else {
            // Generic: check bytes
            size_t sz = data_type_sizeof(x);
            const char* p = (const char*)x->data;
            for (size_t i = 0; i < x->ndata; ++i) {
                bool nz = false;
                for (size_t b = 0; b < sz; ++b) {
                    if (p[i * sz + b] != 0) { nz = true; break; }
                }
                if (nz) ++count;
            }
        }

        int dims[] = { ndim, count };
        return y->reshape(dims, NNR_DATA_TYPE_INT64);
    }

    bool exec() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const int ndim = x->ndim;
        int64_t* py = (int64_t*)y->data;

        int count = (ndim > 0) ? y->dims[1] : 0;

        // For each non-zero element, compute and store its multi-dimensional index
        small_vector<int> idx(ndim);
        int c = 0;

        auto is_nonzero = [&](size_t i) -> bool {
            switch (x->type) {
            case NNR_DATA_TYPE_FLOAT32: return ((const float*)x->data)[i] != 0.0f;
            case NNR_DATA_TYPE_FLOAT64: return ((const double*)x->data)[i] != 0.0;
            case NNR_DATA_TYPE_INT32: return ((const int32_t*)x->data)[i] != 0;
            case NNR_DATA_TYPE_INT64: return ((const int64_t*)x->data)[i] != 0;
            case NNR_DATA_TYPE_BOOL: return (bool)((const bool_t*)x->data)[i];
            case NNR_DATA_TYPE_UINT8: return ((const uint8_t*)x->data)[i] != 0;
            case NNR_DATA_TYPE_INT8: return ((const int8_t*)x->data)[i] != 0;
            default: {
                size_t sz = data_type_sizeof(x);
                const char* p = (const char*)x->data + i * sz;
                for (size_t b = 0; b < sz; ++b) if (p[b] != 0) return true;
                return false;
            }
            }
        };

        for (size_t i = 0; i < x->ndata; ++i) {
            if (is_nonzero(i)) {
                x->offset_to_indices(static_cast<int>(i), idx);
                // Output format: column-major (ndim x count)
                for (int d = 0; d < ndim; ++d) {
                    py[d * count + c] = idx[d];
                }
                ++c;
            }
        }
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_NonZero(int opset, pool_t& pool)
{
    return pool_new<NonZero_operator>(pool);
}

} // namespace nnr
