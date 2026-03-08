#pragma once
#include "nnr.h"
#include "util.h"

namespace nnr {

template <bool IsMax>
struct ArgReduce_operator : public operator_t {
    int axis;
    int keepdims;
    int select_last_index;

    int dim;
    int stride;

    bool init() override {
        if (!is_inout_size(1, 1)) {
            return false;
        }
        axis = attribute(attr_key_t::axis, 0);
        keepdims = attribute(attr_key_t::keepdims, 1);
        select_last_index = attribute(attr_key_t::select_last_index, 0);
        return true;
    }

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
        dim = x->dims[axis];
        stride = x->strides[axis];
        if (keepdims) {
            dims.assign(x->dims, x->dims + x->ndim);
            dims[axis] = 1;
        }else {
            for (int i = 0; i < x->ndim; ++i) {
                if (i != axis) {
                    dims.push_back(x->dims[i]);
                }
            }
        }
        return y->reshape(dims, NNR_DATA_TYPE_INT64);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        int64_t* py = (int64_t*)y->data;
        size_t len = x->ndata;
        size_t idx = 0;
        int cnt = 0;

        while (idx < len) {
            if (cnt < stride) {
                T bestv = px[idx];
                int64_t besti = 0;
                const T* p = px + idx + stride;
                for (int i = 1; i < dim; i++, p += stride) {
                    bool update;
                    if constexpr (IsMax)
                        update = select_last_index ? (*p >= bestv) : (*p > bestv);
                    else
                        update = select_last_index ? (*p <= bestv) : (*p < bestv);
                    if (update) {
                        bestv = *p;
                        besti = i;
                    }
                }
                *py++ = besti;
                idx++;
                cnt++;
            }else {
                idx += (dim - 1) * stride;
                cnt = 0;
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<ArgReduce_operator<IsMax>,
            opset_t<13, int8_t, int16_t, int32_t, int64_t,
                uint8_t, uint16_t, uint32_t, uint64_t,
                bfloat16_t, float16_t, float, double>,
            opset_t<1, int8_t, int16_t, int32_t, int64_t,
                uint8_t, uint16_t, uint32_t, uint64_t,
                float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace nnr
