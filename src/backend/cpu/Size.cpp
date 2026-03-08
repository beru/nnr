#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Size_operator : public operator_t {

    bool init() override {
        return is_inout_size(1, 1);
    }
    bool reshape() override {
        tensor_t* y = outputs[0];
        return y->reshape({}, NNR_DATA_TYPE_INT64);
    }
    bool exec_impl() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        int64_t* py = (int64_t*)y->data;
        py[0] = x->ndata;
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
        }else if (opset >= 1) {
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
        }
        return false;
    }
};

} // namespace {

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Size(int opset, pool_t& pool) { return pool_new<Size_operator>(pool); }

} // namespace nnr
