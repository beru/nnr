#include "pool_base.h"
#include "pool.h"

#include "backend/webgpu/buffer.h"

namespace nnr {

#define DEFINE_POOL(name, init, combine, finalize)                             \
    namespace {                                                                \
    struct name##_op_webgpu : webgpu::pool_elementwise_t {                     \
        const char* init_expr()     const override { return init; }            \
        const char* combine_expr()  const override { return combine; }         \
        const char* finalize_expr() const override { return finalize; }        \
    };                                                                         \
    }                                                                          \
    operator_t* resolver_default_op_##name##_webgpu(int, pool_t& pool) {       \
        return pool_new<name##_op_webgpu>(pool);                               \
    }

DEFINE_POOL(MaxPool,     "-3.4e38", "max(acc, v)", "acc")
DEFINE_POOL(AveragePool, "0.0",     "acc + v",     "select(0.0, acc / f32(n_div), n_div > 0u)")

#undef DEFINE_POOL

// Global pools: reduce each channel's spatial extent to a single value. The
// shared pool kernel handles this natively — we just substitute kernel=(H,W),
// stride=1, pad=0 without touching attributes, so a graph can use these ops
// verbatim regardless of spatial dims.
namespace {

template <const char* (*InitFn)(), const char* (*CombFn)(), const char* (*FinFn)()>
struct GlobalPool_op_webgpu : webgpu::pool_elementwise_t {
    const char* init_expr()     const override { return InitFn(); }
    const char* combine_expr()  const override { return CombFn(); }
    const char* finalize_expr() const override { return FinFn(); }

    bool reshape() override {
        const tensor_t* X = inputs[0];
        tensor_t*       Y = outputs[0];
        if (X->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (X->ndim != 4) return false;
        int N = X->dims[0], C = X->dims[1], H = X->dims[2], W = X->dims[3];

        int out_dims[4] = { N, C, 1, 1 };
        if (!Y->reshape(std::span<const int>(out_dims, 4), X->type)) return false;

        total_u        = (uint32_t)(N * C);
        meta_vals[0]   = total_u;
        meta_vals[1]   = (uint32_t)N;
        meta_vals[2]   = (uint32_t)C;
        meta_vals[3]   = 1u;                 // H_out
        meta_vals[4]   = 1u;                 // W_out
        meta_vals[5]   = (uint32_t)H;        // kH
        meta_vals[6]   = (uint32_t)W;        // kW
        meta_vals[7]   = 1u;                 // stride_h
        meta_vals[8]   = 1u;                 // stride_w
        meta_vals[9]   = 0u;                 // pad_top
        meta_vals[10]  = 0u;                 // pad_left
        meta_vals[11]  = 1u;                 // dilation_h
        meta_vals[12]  = 1u;                 // dilation_w
        meta_vals[13]  = (uint32_t)H;
        meta_vals[14]  = (uint32_t)W;
        meta_vals[15]  = 0u;                 // count_include_pad (irrelevant; no pad)

        webgpu::ensure_buffer(X, X->ndata * sizeof(float));
        webgpu::ensure_buffer(Y, Y->ndata * sizeof(float));
        this->finalize_meta_for_dispatch();
        return true;
    }
};

const char* gmax_init() { return "-3.4e38"; }
const char* gmax_comb() { return "max(acc, v)"; }
const char* gmax_fin () { return "acc"; }
const char* gavg_init() { return "0.0"; }
const char* gavg_comb() { return "acc + v"; }
const char* gavg_fin () { return "acc / f32(n_div)"; }

} // namespace

operator_t* resolver_default_op_GlobalMaxPool_webgpu(int, pool_t& pool) {
    return pool_new<GlobalPool_op_webgpu<gmax_init, gmax_comb, gmax_fin>>(pool);
}
operator_t* resolver_default_op_GlobalAveragePool_webgpu(int, pool_t& pool) {
    return pool_new<GlobalPool_op_webgpu<gavg_init, gavg_comb, gavg_fin>>(pool);
}

} // namespace nnr
