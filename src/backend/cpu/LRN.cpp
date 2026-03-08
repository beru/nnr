#include "nnr.h"
#include "util.h"
#include "kernel/lrn.h"
#include <cassert>

namespace nnr {

namespace {

struct LRN_operator : public operator_t {
    float alpha;
    float beta;
    float bias;
    int size;

    bool init() override {
        if (!is_inout_size(1, 1)) {
            return false;
        }
        alpha = attribute(attr_key_t::alpha, 0.0001f);
        beta = attribute(attr_key_t::beta, 0.75f);
        bias = attribute(attr_key_t::bias, 1.0f);
        size = attribute(attr_key_t::size, 1);
        return true;
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;
        const T over = (T)(alpha / size);
        const int N = x->dims[0];
        const int C = x->dims[1];
        const int L = x->strides[1]; // spatial elements per channel
        const int half = size / 2;

        // Sliding window along C: precompute sq[c*L+i] = px[...]^2,
        // then maintain running sum as we slide the window.
        std::vector<T> sq_buf((size_t)C * L);
        T* sq = sq_buf.data();
        for (int u = 0; u < N; ++u) {
            const T* xn = px + (size_t)u * C * L;
            T* yn = py + (size_t)u * C * L;

            // Precompute squares
            for (int cl = 0; cl < C * L; ++cl)
                sq[cl] = xn[cl] * xn[cl];

            // For each spatial element, slide window along C
            for (int i = 0; i < L; ++i) {
                // Initialize sum for channel 0
                T sum = 0;
                int c0_end = std::min(half, C - 1);
                for (int j = 0; j <= c0_end; ++j)
                    sum += sq[j * L + i];
                yn[i] = xn[i] * (T)powf((float)(bias + over * sum), -beta);

                // Slide for channels 1..C-1
                for (int v = 1; v < C; ++v) {
                    int add_c = v + half;
                    if (add_c < C)
                        sum += sq[add_c * L + i];
                    int rem_c = v - half - 1;
                    if (rem_c >= 0)
                        sum -= sq[rem_c * L + i];
                    int o = v * L + i;
                    yn[o] = xn[o] * (T)powf((float)(bias + over * sum), -beta);
                }
            }
        }
        return true;
    }

    bool exec() override {
        if (inputs[0]->type == NNR_DATA_TYPE_FLOAT32) {
            const tensor_t* x = inputs[0];
            tensor_t* y = outputs[0];
            assert(x->format == memory_layout_t::NCHW);
            assert(y->format == memory_layout_t::NCHW);
            int N = x->dims[0];
            int C = x->dims[1];
            int spatial = x->strides[1]; // H*W (or product of all spatial dims)
            lrn((const float*)x->data, (float*)y->data,
                N, C, 1, spatial, size, alpha / size, beta, bias);
            return true;
        }
        return typed_exec<LRN_operator,
            opset_t<13, bfloat16_t, float16_t, float, double>,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace {

// @nnr-meta-op mt=no
operator_t* resolver_default_op_LRN(int opset, pool_t& pool) { return pool_new<LRN_operator>(pool); }

} // namespace nnr
