#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Reshape_operator : public operator_t {

    bool init() override {
        return is_inout_size(2, 1);
    }

    int view_input_index() const override { return 0; }

    bool reshape() override {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        const tensor_t* s = inputs[1];

        if ((x->ndim == 0) || (x->type == NNR_DATA_TYPE_UNDEFINED)) {
            return false;
        }
        if ((s->ndim == 0) || (s->type != NNR_DATA_TYPE_INT64)) {
            return false;
        }
        int allowzero = attribute(attr_key_t::allowzero, (int32_t)0);
        int64_t* ps = (int64_t*)s->data;
        int total_dim = 1;
        int total_shape = 1;
        const size_t ndim = s->ndata;
        small_vector<int> dims(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            if (ps[i] == 0) {
                if (allowzero) {
                    dims[i] = 0;
                } else if (i < x->ndim) {
                    dims[i] = x->dims[i];
                }
            }else if (ps[i] > 0) {
                dims[i] = ps[i];
            }else {
                total_dim = std::reduce(x->dims, x->dims + x->ndim, 1, std::multiplies<>{});
                for (int j = 0; j < ndim; ++j) {
                    if (ps[j] > 0) {
                        total_shape *= ps[j];
                    }else if (ps[j] == 0) {
                        total_shape *= x->dims[j];
                    }
                }
                dims[i] = total_dim / total_shape;
            }
        }
        if (!y->reshape(dims, x->type)) return false;
        if (x->is_quantized()) y->set_quant(x->quant_scale, x->quant_zero_point);
        return true;
    }

    bool exec_impl() {
        tensor_t* y = outputs[0];
        // Self-reshape on view path: shape input may have been recomputed by
        // an upstream Shape→Cast→Concat chain when input dims change at exec
        // time. See kb/dynamic_shape_pool_view.md.
        if (!y->owns_data) {
            if (!reshape()) return false;
            y->data = inputs[0]->data;  // zero-copy view
        } else if (y->data != inputs[0]->data)
            copy_data(y, inputs[0]);
        return true;
    }

    bool exec() override {
        data_type_t type = inputs[0]->type;
        if (opset >= 14) {
            switch (type) {
            case NNR_DATA_TYPE_BOOL:
            case NNR_DATA_TYPE_INT8:
            case NNR_DATA_TYPE_INT16:
            case NNR_DATA_TYPE_INT32:
            case NNR_DATA_TYPE_INT64:
            case NNR_DATA_TYPE_UINT8:
            case NNR_DATA_TYPE_UINT16:
            case NNR_DATA_TYPE_UINT32:
            case NNR_DATA_TYPE_UINT64:
            case NNR_DATA_TYPE_BFLOAT16:
            case NNR_DATA_TYPE_FLOAT16:
            case NNR_DATA_TYPE_FLOAT32:
            case NNR_DATA_TYPE_FLOAT64:
            case NNR_DATA_TYPE_COMPLEX64:
            case NNR_DATA_TYPE_COMPLEX128:
            case NNR_DATA_TYPE_STRING:
                return exec_impl();
            default:
                break;
            }
        }else if (opset >= 13) {
            switch (type) {
            case NNR_DATA_TYPE_BOOL:
            case NNR_DATA_TYPE_INT8:
            case NNR_DATA_TYPE_INT16:
            case NNR_DATA_TYPE_INT32:
            case NNR_DATA_TYPE_INT64:
            case NNR_DATA_TYPE_UINT8:
            case NNR_DATA_TYPE_UINT16:
            case NNR_DATA_TYPE_UINT32:
            case NNR_DATA_TYPE_UINT64:
            case NNR_DATA_TYPE_BFLOAT16:
            case NNR_DATA_TYPE_FLOAT16:
            case NNR_DATA_TYPE_FLOAT32:
            case NNR_DATA_TYPE_FLOAT64:
            case NNR_DATA_TYPE_COMPLEX64:
            case NNR_DATA_TYPE_COMPLEX128:
            case NNR_DATA_TYPE_STRING:
                return exec_impl();
            default:
                break;
            }
        }else if (opset >= 5) {
            switch (type) {
            case NNR_DATA_TYPE_BOOL:
            case NNR_DATA_TYPE_INT8:
            case NNR_DATA_TYPE_INT16:
            case NNR_DATA_TYPE_INT32:
            case NNR_DATA_TYPE_INT64:
            case NNR_DATA_TYPE_UINT8:
            case NNR_DATA_TYPE_UINT16:
            case NNR_DATA_TYPE_UINT32:
            case NNR_DATA_TYPE_UINT64:
            case NNR_DATA_TYPE_FLOAT16:
            case NNR_DATA_TYPE_FLOAT32:
            case NNR_DATA_TYPE_FLOAT64:
            case NNR_DATA_TYPE_COMPLEX64:
            case NNR_DATA_TYPE_COMPLEX128:
            case NNR_DATA_TYPE_STRING:
                return exec_impl();
            default:
                break;
            }
        }else if (opset >= 1) {
        }
        return false;
    }

};

} // namespace {

// @nnr-meta-op mt=no inplace=yes
operator_t* resolver_default_op_Reshape(int opset, pool_t& pool) { return pool_new<Reshape_operator>(pool); }

} // namespace nnr
