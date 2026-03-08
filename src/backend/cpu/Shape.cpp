#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Shape_operator : public operator_t {
    int start_attr;
    int end_attr;
    bool has_end;

    bool init() override {
        if (!is_inout_size(1, 1))
            return false;
        // Shape reads only dims, never element data — layout-agnostic.
        // Setting LAYOUT_ALL lets BLOCKED_16 tensors propagate through
        // Shape without a forced boundary reorder at its input.
        layout_mask = LAYOUT_ALL;
        start_attr = (int)attribute(attr_key_t::start, (int64_t)0);
        int64_t e = attribute(attr_key_t::end, (int64_t)INT64_MAX);
        has_end = (e != INT64_MAX);
        end_attr = (int)e;
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        int ndim = x->ndim;

        int s = start_attr;
        if (s < 0) s += ndim;
        if (s < 0) s = 0;
        if (s > ndim) s = ndim;

        int e;
        if (!has_end) {
            e = ndim;
        } else {
            e = end_attr;
            if (e < 0) e += ndim;
            if (e < 0) e = 0;
            if (e > ndim) e = ndim;
        }

        int len = (e > s) ? (e - s) : 0;
        int tmp[] = { len };
        return y->reshape(tmp, NNR_DATA_TYPE_INT64);
    }

    bool exec() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        int64_t* py = (int64_t*)y->data;
        int ndim = x->ndim;

        int s = start_attr;
        if (s < 0) s += ndim;
        if (s < 0) s = 0;
        if (s > ndim) s = ndim;

        int e;
        if (!has_end) {
            e = ndim;
        } else {
            e = end_attr;
            if (e < 0) e += ndim;
            if (e < 0) e = 0;
            if (e > ndim) e = ndim;
        }

        for (int i = s; i < e; ++i) {
            py[i - s] = x->dims[i];
        }
        return true;
    }
};

} // namespace {

// @nnr-meta-op mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
operator_t* resolver_default_op_Shape(int opset, pool_t& pool) { return pool_new<Shape_operator>(pool); }

} // namespace nnr
