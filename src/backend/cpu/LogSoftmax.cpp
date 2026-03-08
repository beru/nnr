#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct LogSoftmax_13_operator : public operator_t {
    int axis;
    int caxis;
    int current;
    int outer;
    int inner;

    bool init() override {
        if (!is_inout_size(1, 1)) {
            return false;
        }
        axis = attribute(attr_key_t::axis, -1);
        return true;
    }

    bool reshape() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];

        caxis = axis;
        if (caxis < 0) {
            caxis += x->ndim;
        }
        if (caxis < 0 || caxis >= x->ndim) {
            return false;
        }
        outer = 1;
        inner = 1;
        for (int i = 0; i < x->ndim; ++i) {
            if (i == caxis) {
                current = x->dims[i];
            }else if (i < caxis) {
                outer *= x->dims[i];
            }else {
                inner *= x->dims[i];
            }
        }
        return y->reshape_identity(x);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;

        if (inner == 1) {
            for (int i = 0; i < outer; ++i) {
                const T* row = px + i * current;
                T* out = py + i * current;
                T maxv = *std::max_element(row, row + current);
                T sum = 0;
                for (int j = 0; j < current; ++j) {
                    out[j] = exp(row[j] - maxv);
                    sum += out[j];
                }
                if (sum != 0) {
                    T log_sum = log(sum);
                    for (int j = 0; j < current; ++j)
                        out[j] = row[j] - maxv - log_sum;
                }
            }
            return true;
        }

        for (int i = 0; i < outer; ++i) {
            int oo = i * current * inner;
            for (int k = 0; k < inner; ++k) {
                int io = oo + k;
                T maxv = px[io];
                for (int j = 1; j < current; ++j) {
                    int o = io + j * inner;
                    maxv = max(maxv, px[o]);
                }
                T sum = 0;
                for (int j = 0; j < current; ++j) {
                    int o = io + j * inner;
                    py[o] = exp(px[o] - maxv);
                    sum += py[o];
                }
                if (sum != 0) {
                    for (int j = 0; j < current; ++j) {
                        int io = oo + j * inner + k;
                        py[io] = log(py[io] / sum);
                    }
                }
            }
        }
        return true;
    }

    bool exec() override {
        data_type_t type = inputs[0]->type;
        return typed_exec<LogSoftmax_13_operator,
            bfloat16_t, float16_t, float, double
        >(this, type);
    }
};

struct LogSoftmax_1_11_operator : public operator_t {
    int axis;
    int N;
    int D;

    bool init() override {
        if (!(inputs.size() == 1 && outputs.size() == 1)) {
            return false;
        }
        axis = attribute(attr_key_t::axis, 1);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];

        if (axis < 0) {
            axis += x->ndim;
        }
        if (axis < 0 || axis >= x->ndim) {
            return false;
        }
        N = 1, D = 1;
        for (int i = 0; i < x->ndim; ++i) {
            if (i < axis) {
                N *= x->dims[i];
            }else {
                D *= x->dims[i];
            }
        }
        return y->reshape_identity(x);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;

        for (int i = 0, o = 0; i < N; i++, o += D) {
            T maxv = std::numeric_limits<T>::lowest();
            for (int j = 0; j < D; ++j) {
                if (px[o + j] > maxv) {
                    maxv = px[o + j];
                }
            }
            T sum = 0;
            for (int j = 0; j < D; ++j) {
                py[o + j] = exp(px[o + j] - maxv);
                sum += py[o + j];
            }
            if (sum != 0) {
                for (int j = 0; j < D; ++j) {
                    py[o + j] = log(py[o + j] / sum);
                }
            }
        }
        return true;
    }

    bool exec() override {
        data_type_t type = inputs[0]->type;
        return typed_exec<LogSoftmax_1_11_operator,
            float16_t, float, double
        >(this, type);
    }
};

} // namespace {

// @nnr-meta-op mt=no
operator_t* resolver_default_op_LogSoftmax(int opset, pool_t& pool)
{
    if (opset >= 13) {
        return pool_new<LogSoftmax_13_operator>(pool);
    }else {
        return pool_new<LogSoftmax_1_11_operator>(pool);
    }
}

} // namespace nnr
