#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Not_operator : public operator_t {

    bool init() override {
        return is_inout_size(1, 1);
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        return y->reshape_identity(x, NNR_DATA_TYPE_BOOL);
    }

    bool exec_impl() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const bool_t* px = (const bool_t*)x->data;
        bool_t* py = (bool_t*)y->data;

        for (size_t i = 0, l = y->ndata; i < l; ++i) {
            py[i] = !px[i];
        }
        return true;
    }

    bool exec() override {
        data_type_t type = inputs[0]->type;
        if (opset >= 1) {
            switch (type) {
            case NNR_DATA_TYPE_BOOL:
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
operator_t* resolver_default_op_Not(int opset, pool_t& pool) { return pool_new<Not_operator>(pool); }

} // namespace nnr
