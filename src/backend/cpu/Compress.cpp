#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Compress_operator : public operator_t {
    int has_axis;
    int caxis;

    bool init() override {
        if (!(inputs.size() == 2 && outputs.size() == 1)) {
            return false;
        }
        int64_t axis_val = attribute(attr_key_t::axis, (int64_t)INT64_MIN);
        if (axis_val == INT64_MIN) {
            has_axis = 0;
        } else {
            has_axis = 1;
            caxis = static_cast<int>(axis_val);
        }
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        const tensor_t* cond = inputs[1];
        tensor_t* y = outputs[0];

        const bool_t* pc = (const bool_t*)cond->data;
        int ncond = static_cast<int>(cond->ndata);

        if (!has_axis) {
            // Flatten input, select elements where condition is true
            int total = static_cast<int>(x->ndata);
            int n = std::min(ncond, total);
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if ((bool)pc[i]) ++count;
            }
            int dims[] = { count };
            return y->reshape(dims, x->type);
        } else {
            int ax = caxis;
            if (ax < 0) ax += x->ndim;
            // Count true values in condition (up to axis dim)
            int axis_dim = x->dims[ax];
            int n = std::min(ncond, axis_dim);
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if ((bool)pc[i]) ++count;
            }
            small_vector<int> dims(x->ndim);
            for (int i = 0; i < x->ndim; ++i) {
                dims[i] = x->dims[i];
            }
            dims[ax] = count;
            return y->reshape(dims, x->type);
        }
    }

    bool exec() override {
        const tensor_t* x = inputs[0];
        const tensor_t* cond = inputs[1];
        tensor_t* y = outputs[0];
        const bool_t* pc = (const bool_t*)cond->data;
        int ncond = static_cast<int>(cond->ndata);
        size_t sz = data_type_sizeof(x);

        if (!has_axis) {
            // Flatten, select
            const char* px = (const char*)x->data;
            char* py = (char*)y->data;
            int total = static_cast<int>(x->ndata);
            int n = std::min(ncond, total);
            int c = 0;
            if (x->type == NNR_DATA_TYPE_STRING) {
                const std::string* pxs = (const std::string*)x->data;
                std::string* pys = (std::string*)y->data;
                for (int i = 0; i < n; ++i) {
                    if ((bool)pc[i]) {
                        pys[c++] = pxs[i];
                    }
                }
            } else {
                for (int i = 0; i < n; ++i) {
                    if ((bool)pc[i]) {
                        memcpy(py + c * sz, px + i * sz, sz);
                        ++c;
                    }
                }
            }
        } else {
            int ax = caxis;
            if (ax < 0) ax += x->ndim;
            int axis_dim = x->dims[ax];
            int n = std::min(ncond, axis_dim);

            // Compute outer and inner
            int outer = 1;
            for (int i = 0; i < ax; ++i) outer *= x->dims[i];
            int inner = 1;
            for (int i = ax + 1; i < x->ndim; ++i) inner *= x->dims[i];

            int out_axis_dim = y->dims[ax];

            if (x->type == NNR_DATA_TYPE_STRING) {
                const std::string* pxs = (const std::string*)x->data;
                std::string* pys = (std::string*)y->data;
                for (int o = 0; o < outer; ++o) {
                    int c = 0;
                    for (int a = 0; a < n; ++a) {
                        if ((bool)pc[a]) {
                            for (int k = 0; k < inner; ++k) {
                                pys[o * out_axis_dim * inner + c * inner + k] =
                                    pxs[o * axis_dim * inner + a * inner + k];
                            }
                            ++c;
                        }
                    }
                }
            } else {
                const char* px = (const char*)x->data;
                char* py = (char*)y->data;
                for (int o = 0; o < outer; ++o) {
                    int c = 0;
                    for (int a = 0; a < n; ++a) {
                        if ((bool)pc[a]) {
                            memcpy(
                                py + (o * out_axis_dim * inner + c * inner) * sz,
                                px + (o * axis_dim * inner + a * inner) * sz,
                                inner * sz);
                            ++c;
                        }
                    }
                }
            }
        }
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Compress(int opset, pool_t& pool)
{
    return pool_new<Compress_operator>(pool);
}

} // namespace nnr
