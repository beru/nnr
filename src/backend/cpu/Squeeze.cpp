#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Squeeze_1_11_operator : public operator_t {

    bool init() override {
        return is_inout_size(1, 1);
    }

    int view_input_index() const override { return 0; }

    bool reshape() override {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        small_vector<int> dims;

        int64_t* axes = nullptr;
        int naxes = attribute(attr_key_t::axes, axes);
        for (int i = 0; i < x->ndim; ++i) {
            if (x->dims[i] > 1) {
                dims.push_back(x->dims[i]);
            }else {
                bool flag = false;
                for (int j = 0; j < naxes; ++j) {
                    int axis = axes[j];
                    if (axis < 0) {
                        axis += x->ndim;
                    }
                    if (i == axis) {
                        flag = true;
                        break;
                    }
                }
                if (!flag) {
                    dims.push_back(x->dims[i]);
                }
            }
        }
        if (!y->reshape(dims, x->type)) return false;
        if (x->is_quantized()) y->set_quant(x->quant_scale, x->quant_zero_point);
        return true;
    }

    bool exec_impl() {
        tensor_t* y = outputs[0];
        // Self-reshape on view path: dynamic-shape producer (NMS/TopK/...)
        // upstream may shrink x at exec time. See kb/dynamic_shape_pool_view.md.
        if (!y->owns_data) {
            if (!reshape()) return false;
            y->data = inputs[0]->data;  // zero-copy view
        } else if (y->data != inputs[0]->data)
            copy_data(y, inputs[0]);
        return true;
    }

    bool exec() override {
        data_type_t type = inputs[0]->type;
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
            return false;
        }
    }

};

struct Squeeze_13_operator : public operator_t {

    bool init() override {
        return (inputs.size() >= 1) && (outputs.size() == 1);
    }

    int view_input_index() const override { return 0; }

    bool reshape() override {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        small_vector<int> dims;

        if (inputs.size() > 1) {
            const tensor_t* a = inputs[1];
            const int64_t* pa = (const int64_t*)a->data;
            for (int i = 0; i < x->ndim; ++i) {
                if (x->dims[i] > 1) {
                    dims.push_back(x->dims[i]);
                }else {
                    bool flag = false;
                    for (size_t j = 0; j < a->ndata; ++j) {
                        int axis = pa[j];
                        if (axis < 0) {
                            axis += x->ndim;
                        }
                        if (i == axis) {
                            flag = true;
                            break;
                        }
                    }
                    if (!flag) {
                        dims.push_back(x->dims[i]);
                    }
                }
            }
        }else {
            for (int i = 0; i < x->ndim; ++i) {
                if (x->dims[i] > 1) {
                    dims.push_back(x->dims[i]);
                }
            }
        }
        if (!y->reshape(dims, x->type)) return false;
        if (x->is_quantized()) y->set_quant(x->quant_scale, x->quant_zero_point);
        return true;
    }

    bool exec_impl() {
        tensor_t* y = outputs[0];
        // Self-reshape on view path. See kb/dynamic_shape_pool_view.md.
        if (!y->owns_data) {
            if (!reshape()) return false;
            y->data = inputs[0]->data;  // zero-copy view
        } else if (y->data != inputs[0]->data)
            copy_data(y, inputs[0]);
        return true;
    }

    bool exec() override {
        data_type_t type = inputs[0]->type;
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
            return false;
        }
    }

};

} // namespace {

// @nnr-meta-op mt=no inplace=yes
operator_t* resolver_default_op_Squeeze(int opset, pool_t& pool)
{
    if (opset >= 13) {
        return pool_new<Squeeze_13_operator>(pool);
    }else {
        return pool_new<Squeeze_1_11_operator>(pool);
    }
}

} // namespace nnr
