#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Unsqueeze_1_11_operator : public operator_t {

    bool init() override {
        return is_inout_size(1, 1);
    }

    int view_input_index() const override { return 0; }

    bool reshape() override {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        int64_t* axes = nullptr;
        int naxes = attribute(attr_key_t::axes, axes);
        const int ndim = x->ndim + naxes;
        small_vector<int> dims(ndim);

        for (int i = 0; i < naxes; ++i) {
            int axis = axes[i];
            if (axis >= 0 && axis < ndim) {
                dims[axis] = 1;
            }
        }
        for (int i = 0, j = 0; i < ndim; ++i) {
            if (dims[i] != 1) {
                dims[i] = x->dims[j++];
            }
        }
        if (!y->reshape(dims, x->type)) return false;
        if (x->is_quantized()) y->set_quant(x->quant_scale, x->quant_zero_point);
        return true;
    }

    bool exec_impl() {
        tensor_t* y = outputs[0];
        if (!y->owns_data) {
            // Re-derive y's shape from x's live dims — an upstream
            // NonMaxSuppression can shrink x->dims during exec(), and the
            // fast path skips reshape(). Without this, y->ndata stays at
            // the prepare-time upper bound while y->data aliases the
            // shrunken input, so downstream consumers read past the live
            // region (yolov3-tiny NMS→Unsqueeze→Cast crash).
            if (!reshape()) return false;
            y->data = inputs[0]->data;  // zero-copy view
        }
        else if (y->data != inputs[0]->data) {
            copy_data(y, inputs[0]);
        }
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

struct Unsqueeze_13_operator : public operator_t {

    bool init() override {
        return is_inout_size(2, 1);
    }

    int view_input_index() const override { return 0; }

    bool reshape() override {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        const tensor_t* a = inputs[1];
        if (!x || !a || !a->data || a->ndata == 0) return false;
        const int64_t* pa = (const int64_t*)a->data;
        const int ndim = x->ndim + (int)a->ndata;
        if (ndim > MAX_NDIM) return false;
        small_vector<int> dims(ndim);

        for (size_t i = 0; i < a->ndata; ++i) {
            int axis = (int)pa[i];
            if (axis < 0) {
                axis += ndim;
            }
            if (axis >= 0 && axis < ndim) {
                dims[axis] = 1;
            }
        }
        for (int i = 0, j = 0; i < ndim; ++i) {
            if (dims[i] != 1) {
                if (j < x->ndim) {
                    dims[i] = x->dims[j++];
                }
            }
        }
        if (!y) return false;
        if (!y->reshape(dims, x->type)) return false;
        if (x->is_quantized()) y->set_quant(x->quant_scale, x->quant_zero_point);
        return true;
    }

    bool exec_impl() {
        tensor_t* y = outputs[0];
        if (!y->owns_data) {
            // See Unsqueeze_1_11_operator::exec_impl — same dynamic-shape rule.
            if (!reshape()) return false;
            y->data = inputs[0]->data;
        }
        else if (y->data != inputs[0]->data) {
            copy_data(y, inputs[0]);
        }
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
operator_t* resolver_default_op_Unsqueeze(int opset, pool_t& pool)
{
    if (opset >= 13) {
        return pool_new<Unsqueeze_13_operator>(pool);
    }else {
        return pool_new<Unsqueeze_1_11_operator>(pool);
    }
}

} // namespace nnr
