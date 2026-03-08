#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Det_operator : public operator_t {
    bool init() override {
        return is_inout_size(1, 1);
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        if (x->ndim < 2)
            return false;
        int n = x->dims[x->ndim - 1];
        int m = x->dims[x->ndim - 2];
        if (n != m)
            return false;
        // output shape = batch dims (all except last 2)
        int ondim = x->ndim - 2;
        if (ondim == 0) {
            small_vector<int> dims;
            return y->reshape(dims, x->type);
        }
        small_vector<int> dims(ondim);
        for (int i = 0; i < ondim; ++i)
            dims[i] = x->dims[i];
        return y->reshape(dims, x->type);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;

        int n = x->dims[x->ndim - 1];
        int batch = 1;
        for (int i = 0; i < x->ndim - 2; ++i)
            batch *= x->dims[i];

        int n2 = n * n;
        arena_scope_t scope(ctx->arena);
        double* tmp = scope.alloc_arr<double>(n2);

        for (int b = 0; b < batch; ++b) {
            const T* mat = px + b * n2;
            for (int i = 0; i < n2; ++i)
                tmp[i] = (double)mat[i];

            // LU decomposition with partial pivoting
            double det = 1.0;
            int sign = 1;
            for (int col = 0; col < n; ++col) {
                // find pivot
                int pivot = col;
                double maxval = std::abs(tmp[col * n + col]);
                for (int row = col + 1; row < n; ++row) {
                    double v = std::abs(tmp[row * n + col]);
                    if (v > maxval) {
                        maxval = v;
                        pivot = row;
                    }
                }
                if (pivot != col) {
                    sign = -sign;
                    for (int j = 0; j < n; ++j)
                        std::swap(tmp[col * n + j], tmp[pivot * n + j]);
                }
                double diag = tmp[col * n + col];
                if (diag == 0.0) {
                    det = 0.0;
                    break;
                }
                det *= diag;
                for (int row = col + 1; row < n; ++row) {
                    double factor = tmp[row * n + col] / diag;
                    for (int j = col + 1; j < n; ++j)
                        tmp[row * n + j] -= factor * tmp[col * n + j];
                }
            }
            py[b] = T(det * sign);
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Det_operator,
            opset_t<11, float16_t, float, double, bfloat16_t>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Det(int opset, pool_t& pool)
{
    return pool_new<Det_operator>(pool);
}

} // namespace nnr
