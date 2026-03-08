#include "nnr.h"
#include "util.h"
#include "kernel/pool.h"

namespace nnr {

namespace {

struct global_pool_base : operator_t {
    bool init() override { return is_inout_size(1, 1); }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        small_vector<int> dims(x->ndim);
        for (int i = 0; i < x->ndim; ++i)
            dims[i] = (i < 2) ? x->dims[i] : 1;
        return outputs[0]->reshape(dims, x->type);
    }
};

struct GlobalMaxPool_op : global_pool_base {
    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)outputs[0]->data;
        int N = outputs[0]->dims[0], C = outputs[0]->dims[1];
        int m = x->strides[1];
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < C; ++j) {
                int o = i * C + j;
                T maxv = px[o * m];
                for (int k = 1; k < m; ++k)
                    maxv = max(maxv, px[o * m + k]);
                py[o] = maxv;
            }
        return true;
    }
    bool exec() override {
        return typed_exec<GlobalMaxPool_op, opset_t<1, float16_t, float, double>>(this, opset, inputs[0]->type);
    }
};

struct GlobalAveragePool_op : global_pool_base {
    bool reshape() override {
        if (!global_pool_base::reshape()) return false;
        const tensor_t* x = inputs[0];
        // GlobalAveragePool can accept NHWC input for 4D float tensors.
        // Output is [N,C,1,1] — layout-neutral (same element order in both).
        if (x->ndim == 4 && x->type == NNR_DATA_TYPE_FLOAT32) {
            layout_mask = LAYOUT_NCHW | LAYOUT_NHWC;
#ifdef NNR_ARCH_X64
            // NCHWc: channels must be multiple of 16
            if (x->dims[1] % 16 == 0 && x->dims[1] >= 16)
                layout_mask |= LAYOUT_BLOCKED_16;
#endif
        }
        return true;
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        int N = y->dims[0], C = y->dims[1];
        int spatial = (int)(x->ndata / (N * C));
        if constexpr (std::is_same_v<T, float>) {
            // NCHWc (BLOCKED_16) path
            // Input: [N, Cb, H, W, 16], spatial = H*W
            // Output: [N, Cb, 1, 1, 16] — but stored in [N, C, 1, 1] tensor
            // For BLOCKED_16, ndata includes padding; compute spatial from dims.
            if (x->format == memory_layout_t::BLOCKED_16
                && (layout_mask & LAYOUT_BLOCKED_16)) {
                int H = x->dims[2], W = x->dims[3];
#ifdef NNR_ARCH_X64
                if (has_avx512()) {
                    global_avgpool_nchwc_x64((const float*)x->data, (float*)y->data,
                        N, C, H * W);
                } else
#endif
                {
                    global_avgpool_nchwc((const float*)x->data, (float*)y->data,
                        N, C, H * W, 16);
                }
                // Output [N,C,1,1] with spatial=1 is layout-neutral.
                // Mark BLOCKED_16 to keep the chain alive if consumed by another
                // BLOCKED_16 op, or let the reorder pass convert to NCHW if needed.
                y->format = memory_layout_t::BLOCKED_16;
                return true;
            }
            // NHWC path
            if (x->format == memory_layout_t::NHWC) {
                global_avgpool_nhwc((const float*)x->data, (float*)y->data, N, C, spatial);
                // Output [N,C,1,1] is layout-neutral — mark NCHW to avoid
                // unnecessary graph output reorder.
                y->format = memory_layout_t::NCHW;
                return true;
            }
        }
        global_avgpool((const T*)x->data, (T*)y->data, N, C, spatial);
        return true;
    }
    bool exec() override {
        return typed_exec<GlobalAveragePool_op, opset_t<1, float16_t, float, double>>(this, opset, inputs[0]->type);
    }
};

struct GlobalLpPool_op : global_pool_base {
    float p;
    bool init() override {
        if (!is_inout_size(1, 1)) return false;
        if (opset >= 2) p = (float)attribute(attr_key_t::p, 2);
        else p = attribute(attr_key_t::p, 2.0f);
        return true;
    }
    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)outputs[0]->data;
        int N = outputs[0]->dims[0], C = outputs[0]->dims[1];
        int m = x->strides[1];
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < C; ++j) {
                int o = i * C + j;
                py[o] = 0;
                for (int k = 0; k < m; ++k)
                    py[o] += pow(abs(px[o * m + k]), (T)p);
                py[o] = pow(py[o], T(1.0 / p));
            }
        return true;
    }
    bool exec() override {
        return typed_exec<GlobalLpPool_op, opset_t<1, float16_t, float, double>>(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op op=GlobalMaxPool mt=no
operator_t* resolver_default_op_GlobalMaxPool(int opset, pool_t& pool) { return pool_new<GlobalMaxPool_op>(pool); }
// @nnr-meta-op op=GlobalAveragePool mt=no layout=[NCHW,NHWC,BLOCKED_16]
operator_t* resolver_default_op_GlobalAveragePool(int opset, pool_t& pool) { return pool_new<GlobalAveragePool_op>(pool); }
// @nnr-meta-op op=GlobalLpPool mt=no
operator_t* resolver_default_op_GlobalLpPool(int opset, pool_t& pool) { return pool_new<GlobalLpPool_op>(pool); }

} // namespace nnr
