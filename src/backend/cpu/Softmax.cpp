#include "nnr.h"
#include "util.h"
#include "thread_pool.h"
#ifdef NNR_ARCH_X64
#include "backend/x64/softmax_avx512.h"
#elifdef NNR_ARCH_ARM64
#include "backend/arm/softmax_neon.h"
#endif

namespace nnr {

namespace {

struct Softmax_13_operator : public operator_t {
    int axis;

    int caxis;
    int current;
    int outer;
    int inner;

    bool init() override {
        if (!is_inout_size(1, 1)) {return false;
        }
        axis = attribute(attr_key_t::axis, -1);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];

        caxis = axis;
        if (caxis < 0) {
            caxis += x->ndim;
        }
        if (caxis < 0 || caxis >= x->ndim) {
            return false;
        }
        outer = 1, inner = 1;
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

        // An ONNX tensor with a zero-length reduction axis is legal (e.g.
        // dynamic-batch shapes). Avoid calling std::max_element on an empty
        // range — that is UB — and avoid dividing by a trivially-zero sum.
        // Output is already allocated and zero-initialized; nothing to do.
        if (current == 0 || outer == 0 || inner == 0) return true;

        if (inner == 1) {
#ifdef NNR_ARCH_X64
            if constexpr (std::is_same_v<T, float>) {
                // Cost-based threading: each row is load + exp + normalize
                // over `current` floats. Exp is ~15 cycles/elt so total
                // ~50 cycles/elt per row. for_cost stays serial below
                // ~500K cycles total (MIN_PARALLEL_COST), which keeps
                // small softmaxes off the thread pool where dispatch
                // overhead otherwise dominates the tiny per-row work.
                int64_t cost_per_row = (int64_t)current * 50;
                nnr::for_cost(0, outer, cost_per_row, [&](int i) {
                    softmax_row_avx512(px + (size_t)i * current,
                                       py + (size_t)i * current, current);
                });
                return true;
            }
#elifdef NNR_ARCH_ARM64
            if constexpr (std::is_same_v<T, float>) {
                nnr::for_static(0, outer, outer > 4, [&](int i) {
                    softmax_row_neon(px + (size_t)i * current,
                                     py + (size_t)i * current, current);
                });
                return true;
            }
#endif
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
                    T inv = T(1) / sum;
                    for (int j = 0; j < current; ++j)
                        out[j] *= inv;
                }
            }
            return true;
        }

        for (int i = 0; i < outer; ++i) {
            int oo = i * current * inner;
            for (int k = 0; k < inner; ++k) {
                int io = oo + k;
                T maxv = px[io];
                for (int j = 0; j < current; ++j) {
                    int o = io + j * inner;
                    if (px[o] > maxv) {
                        maxv = px[o];
                    }
                }
                T sum = 0;
                for (int j = 0; j < current; ++j) {
                    int o = io + j * inner;
                    T v = exp(px[o] - maxv);
                    py[o] = v;
                    sum += v;
                }
                if (sum != 0) {
                    for (int j = 0; j < current; ++j) {
                        io = oo + j * inner + k;
                        py[io] /= sum;
                    }
                }
            }
        }
        return true;
    }

    bool exec() override {
        data_type_t type = inputs[0]->type;
        return typed_exec<Softmax_13_operator,
            bfloat16_t, float16_t, float, double
        >(this, type);
    }
};

struct Softmax_1_11_operator : public operator_t {
    int axis;
    int N;
    int D;

    bool init() override {
        if (!is_inout_size(1, 1)) {
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

#ifdef NNR_ARCH_X64
        if constexpr (std::is_same_v<T, float>) {
            // Opset 1-11 semantics: axis flattens dims [axis..end] into D.
            // Each of N rows is a contiguous span of D floats, so the same
            // AVX-512 row kernel + cost-based dispatch used by Softmax_13
            // (inner==1 case) applies directly.
            int64_t cost_per_row = (int64_t)D * 50;
            nnr::for_cost(0, N, cost_per_row, [&](int i) {
                softmax_row_avx512(px + (size_t)i * D,
                                   py + (size_t)i * D, D);
            });
            return true;
        }
#elifdef NNR_ARCH_ARM64
        if constexpr (std::is_same_v<T, float>) {
            nnr::for_static(0, N, N > 4, [&](int i) {
                softmax_row_neon(px + (size_t)i * D,
                                 py + (size_t)i * D, D);
            });
            return true;
        }
#endif
        for (int i = 0, o = 0; i < N; i++, o += D) {
            T maxv = std::numeric_limits<T>::lowest();
            for (int j = 0; j < D; ++j) {
                if (px[o + j] > maxv) {
                    maxv = px[o + j];
                }
            }
            T sum = 0;
            for (int j = 0; j < D; ++j) {
                T v = exp(px[o + j] - maxv);
                py[o + j] = v;
                sum += v;
            }
            if (sum != 0) {
                for (int j = 0; j < D; ++j) {
                    py[o + j] /= sum;
                }
            }
        }
        return true;
    }

    bool exec() override {
        data_type_t type = inputs[0]->type;
        return typed_exec<Softmax_1_11_operator,
            float16_t, float, double
        >(this, type);
    }

};

} // namespace {

// @nnr-meta-op mt=cost
operator_t* resolver_default_op_Softmax(int opset, pool_t& pool)
{
    if (opset >= 13) {
        return pool_new<Softmax_13_operator>(pool);
    }else {
        return pool_new<Softmax_1_11_operator>(pool);
    }
}

} // namespace nnr
