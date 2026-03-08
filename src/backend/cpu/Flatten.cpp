#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Flatten_operator : public operator_t {
    int axis;

    bool init() override {
        if (!is_inout_size(1, 1)) {
            return false;
        }
        axis = attribute(attr_key_t::axis, 1);
        return true;
    }

    int view_input_index() const override { return 0; }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        small_vector<int> dims;

        if (axis < 0) {
            axis += x->ndim;
        }
        if (axis < 0 || axis >= x->ndim) {
            return false;
        }
        int j = 1;
        for (int i = 0; i < x->ndim; ++i) {
            if (i != axis) {
                j *= x->dims[i];
            }else {
                dims.push_back(j);
                j = x->dims[i];
            }
        }
        dims.push_back(j);
        if (!y->reshape(dims, x->type)) return false;
        if (x->is_quantized()) y->set_quant(x->quant_scale, x->quant_zero_point);
        return true;
    }

    bool exec_impl() {
        tensor_t* y = outputs[0];
        if (!y->owns_data)
            y->data = inputs[0]->data;  // zero-copy view
        else if (y->data != inputs[0]->data)
            copy_data(y, inputs[0]);
        return true;
    }

    bool exec() override {
        data_type_t type = inputs[0]->type;
        if (opset >= 13) {
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
        }else if (opset >= 11) {
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
        }else if (opset >= 9) {
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
            switch (type) {
            case NNR_DATA_TYPE_FLOAT16:
            case NNR_DATA_TYPE_FLOAT32:
            case NNR_DATA_TYPE_FLOAT64:
                return exec_impl();
            default:
                break;
            }
        }
        return false;
    }
};

} // namespace {

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Flatten(int opset, pool_t& pool) { return pool_new<Flatten_operator>(pool); }

} // namespace nnr
