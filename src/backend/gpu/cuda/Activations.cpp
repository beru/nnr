#if defined(NNR_USE_CUDA)

// Activations with attributes (alpha/beta/min/max) + simple unary math ops
// (Neg, Sqrt, Exp, Log, Erf, Ceil, Floor, Reciprocal, Softplus, Softsign,
// HardSwish, PRelu). All f32 only; broadcast-free (same-shape).

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "elementwise.h"
#include "attr_key.h"

namespace nnr {

operator_t* resolver_default_op_LeakyRelu  (int opset, pool_t& pool);
operator_t* resolver_default_op_Elu        (int opset, pool_t& pool);
operator_t* resolver_default_op_Celu       (int opset, pool_t& pool);
operator_t* resolver_default_op_Selu       (int opset, pool_t& pool);
operator_t* resolver_default_op_HardSigmoid(int opset, pool_t& pool);
operator_t* resolver_default_op_HardSwish  (int opset, pool_t& pool);
operator_t* resolver_default_op_Clip       (int opset, pool_t& pool);
operator_t* resolver_default_op_Softplus   (int opset, pool_t& pool);
operator_t* resolver_default_op_Softsign   (int opset, pool_t& pool);
operator_t* resolver_default_op_Neg        (int opset, pool_t& pool);
operator_t* resolver_default_op_Sqrt       (int opset, pool_t& pool);
operator_t* resolver_default_op_Exp        (int opset, pool_t& pool);
operator_t* resolver_default_op_Log        (int opset, pool_t& pool);
operator_t* resolver_default_op_Erf        (int opset, pool_t& pool);
operator_t* resolver_default_op_Ceil       (int opset, pool_t& pool);
operator_t* resolver_default_op_Floor      (int opset, pool_t& pool);
operator_t* resolver_default_op_Reciprocal (int opset, pool_t& pool);
operator_t* resolver_default_op_PRelu      (int opset, pool_t& pool);
operator_t* resolver_default_op_Sin        (int opset, pool_t& pool);
operator_t* resolver_default_op_Cos        (int opset, pool_t& pool);
operator_t* resolver_default_op_Tan        (int opset, pool_t& pool);
operator_t* resolver_default_op_Asin       (int opset, pool_t& pool);
operator_t* resolver_default_op_Acos       (int opset, pool_t& pool);
operator_t* resolver_default_op_Atan       (int opset, pool_t& pool);
operator_t* resolver_default_op_Sinh       (int opset, pool_t& pool);
operator_t* resolver_default_op_Cosh       (int opset, pool_t& pool);
operator_t* resolver_default_op_Asinh      (int opset, pool_t& pool);
operator_t* resolver_default_op_Acosh      (int opset, pool_t& pool);
operator_t* resolver_default_op_Atanh      (int opset, pool_t& pool);
operator_t* resolver_default_op_Round      (int opset, pool_t& pool);
operator_t* resolver_default_op_Sign       (int opset, pool_t& pool);
operator_t* resolver_default_op_Swish      (int opset, pool_t& pool);
operator_t* resolver_default_op_Mish       (int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static CUfunction get_elem(gpu::cuda_backend_t* be, const char* name) {
    return be->nvrtc.get("nnr_elementwise_f32",
                         gpu::elementwise_f32_source(),
                         name,
                         gpu::nvrtc_arch_option(be->device));
}

static bool launch_1d(gpu::cuda_backend_t* be, CUfunction f, void** args, unsigned long long n) {
    constexpr unsigned BLK = 256;
    unsigned grid = (unsigned)((n + BLK - 1) / BLK);
    return gpu::nvrtc_launch(be->device, f, grid, 1, 1, BLK, 1, 1, args);
}

// -------- simple no-attr unary ops --------

struct SimpleUnary_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    const char* kernel_name = nullptr;

    virtual operator_t* make_fallback() = 0;

    bool init() override {
        if (!(inputs.size() == 1 && outputs.size() == 1)) return false;
        fallback = make_fallback();
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        const tensor_t* x = inputs[0];
        prim_valid = (x->type == NNR_DATA_TYPE_FLOAT32 && outputs[0]->ndata == x->ndata);
        device_tag = prim_valid ? static_cast<uint8_t>(backend_t::CUDA) : 0;
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        CUfunction f = get_elem(be, kernel_name);
        if (!f) { return fallback->exec(); }

        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        unsigned long long n = outputs[0]->ndata;
        void* args[] = { &d_x, &d_y, &n };
        if (!launch_1d(be, f, args, n)) { return fallback->exec(); }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

// Declare a SimpleUnary_cuda-derived struct (inside anon namespace).
#define NNR_UNARY_STRUCT(Name, Kernel) \
struct Name##_cuda : SimpleUnary_cuda { \
    Name##_cuda() { kernel_name = Kernel; } \
    operator_t* make_fallback() override { return resolver_default_op_##Name(opset, ctx->attr_pool); } \
};

NNR_UNARY_STRUCT(Neg,        "neg_f32")
NNR_UNARY_STRUCT(Sqrt,       "sqrt_f32")
NNR_UNARY_STRUCT(Exp,        "exp_f32")
NNR_UNARY_STRUCT(Log,        "log_f32")
NNR_UNARY_STRUCT(Erf,        "erf_f32")
NNR_UNARY_STRUCT(Ceil,       "ceil_f32")
NNR_UNARY_STRUCT(Floor,      "floor_f32")
NNR_UNARY_STRUCT(Reciprocal, "reciprocal_f32")
NNR_UNARY_STRUCT(Softplus,   "softplus_f32")
NNR_UNARY_STRUCT(Softsign,   "softsign_f32")
NNR_UNARY_STRUCT(HardSwish,  "hard_swish_f32")
NNR_UNARY_STRUCT(Sin,        "sin_f32")
NNR_UNARY_STRUCT(Cos,        "cos_f32")
NNR_UNARY_STRUCT(Tan,        "tan_f32")
NNR_UNARY_STRUCT(Asin,       "asin_f32")
NNR_UNARY_STRUCT(Acos,       "acos_f32")
NNR_UNARY_STRUCT(Atan,       "atan_f32")
NNR_UNARY_STRUCT(Sinh,       "sinh_f32")
NNR_UNARY_STRUCT(Cosh,       "cosh_f32")
NNR_UNARY_STRUCT(Asinh,      "asinh_f32")
NNR_UNARY_STRUCT(Acosh,      "acosh_f32")
NNR_UNARY_STRUCT(Atanh,      "atanh_f32")
NNR_UNARY_STRUCT(Round,      "round_f32")
NNR_UNARY_STRUCT(Sign,       "sign_f32")
NNR_UNARY_STRUCT(Swish,      "swish_f32")
NNR_UNARY_STRUCT(Mish,       "mish_f32")

#undef NNR_UNARY_STRUCT

// -------- unary with one float attribute --------

struct Alpha1_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    const char* kernel_name = nullptr;
    float alpha = 0.f;
    float default_alpha = 0.f;

    virtual operator_t* make_fallback() = 0;

    bool init() override {
        if (!(inputs.size() == 1 && outputs.size() == 1)) return false;
        fallback = make_fallback();
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        alpha = attribute(attr_key_t::alpha, default_alpha);
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        const tensor_t* x = inputs[0];
        prim_valid = (x->type == NNR_DATA_TYPE_FLOAT32 && outputs[0]->ndata == x->ndata);
        device_tag = prim_valid ? static_cast<uint8_t>(backend_t::CUDA) : 0;
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        CUfunction f = get_elem(be, kernel_name);
        if (!f) { return fallback->exec(); }

        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        unsigned long long n = outputs[0]->ndata;
        float a = alpha;
        void* args[] = { &d_x, &d_y, &n, &a };
        if (!launch_1d(be, f, args, n)) { return fallback->exec(); }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

struct LeakyRelu_cuda : Alpha1_cuda {
    LeakyRelu_cuda() { kernel_name = "leaky_relu_f32"; default_alpha = 0.01f; }
    operator_t* make_fallback() override { return resolver_default_op_LeakyRelu(opset, ctx->attr_pool); }
};
struct Elu_cuda : Alpha1_cuda {
    Elu_cuda() { kernel_name = "elu_f32"; default_alpha = 1.0f; }
    operator_t* make_fallback() override { return resolver_default_op_Elu(opset, ctx->attr_pool); }
};
struct Celu_cuda : Alpha1_cuda {
    Celu_cuda() { kernel_name = "celu_f32"; default_alpha = 1.0f; }
    operator_t* make_fallback() override { return resolver_default_op_Celu(opset, ctx->attr_pool); }
};

// -------- unary with two float attributes (alpha, beta) --------

struct AlphaBeta_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    const char* kernel_name = nullptr;
    float alpha = 0.f, beta = 0.f;
    float default_alpha = 0.f, default_beta = 0.f;

    virtual operator_t* make_fallback() = 0;

    bool init() override {
        if (!(inputs.size() == 1 && outputs.size() == 1)) return false;
        fallback = make_fallback();
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        alpha = attribute(attr_key_t::alpha, default_alpha);
        beta  = attribute(attr_key_t::beta,  default_beta);
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        const tensor_t* x = inputs[0];
        prim_valid = (x->type == NNR_DATA_TYPE_FLOAT32 && outputs[0]->ndata == x->ndata);
        device_tag = prim_valid ? static_cast<uint8_t>(backend_t::CUDA) : 0;
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        CUfunction f = get_elem(be, kernel_name);
        if (!f) { return fallback->exec(); }

        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        unsigned long long n = outputs[0]->ndata;
        float a = alpha, b = beta;
        void* args[] = { &d_x, &d_y, &n, &a, &b };
        if (!launch_1d(be, f, args, n)) { return fallback->exec(); }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

struct Selu_cuda : AlphaBeta_cuda {
    Selu_cuda() { kernel_name = "selu_f32"; default_alpha = 1.67326319217681884765625f; default_beta = 1.05070102214813232421875f; }
    operator_t* make_fallback() override { return resolver_default_op_Selu(opset, ctx->attr_pool); }
    bool init() override {
        // Selu's second attr is "gamma", not "beta"
        if (!AlphaBeta_cuda::init()) return false;
        beta = attribute(attr_key_t::gamma, default_beta);
        return true;
    }
};

struct HardSigmoid_cuda : AlphaBeta_cuda {
    HardSigmoid_cuda() { kernel_name = "hard_sigmoid_f32"; default_alpha = 0.2f; default_beta = 0.5f; }
    operator_t* make_fallback() override { return resolver_default_op_HardSigmoid(opset, ctx->attr_pool); }
};

// -------- Clip: min/max as input tensors (opset 11+) or attributes (opset 6-10) --------

struct Clip_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    float lo = -3.4e38f, hi = 3.4e38f;

    bool init() override {
        if (inputs.size() < 1 || outputs.size() != 1) return false;
        fallback = resolver_default_op_Clip(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        // Inherit fusable_apply so fuse_post_ops folds this into an upstream Conv.
        fusable_apply = fallback->fusable_apply;
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        const tensor_t* x = inputs[0];
        prim_valid = (x->type == NNR_DATA_TYPE_FLOAT32 && outputs[0]->ndata == x->ndata);

        // Resolve min/max — either attribute (old) or host-side scalar input (new).
        lo = -3.4e38f; hi = 3.4e38f;
        if (inputs.size() >= 2 && inputs[1] && inputs[1]->data
            && inputs[1]->type == NNR_DATA_TYPE_FLOAT32 && inputs[1]->ndata == 1) {
            lo = *(const float*)inputs[1]->data;
        }
        if (inputs.size() >= 3 && inputs[2] && inputs[2]->data
            && inputs[2]->type == NNR_DATA_TYPE_FLOAT32 && inputs[2]->ndata == 1) {
            hi = *(const float*)inputs[2]->data;
        }

        device_tag = prim_valid ? static_cast<uint8_t>(backend_t::CUDA) : 0;
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        CUfunction f = get_elem(be, "clip_f32");
        if (!f) { return fallback->exec(); }

        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        unsigned long long n = outputs[0]->ndata;
        float _lo = lo, _hi = hi;
        void* args[] = { &d_x, &d_y, &n, &_lo, &_hi };
        if (!launch_1d(be, f, args, n)) { return fallback->exec(); }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

// -------- PRelu: per-channel or scalar slope --------

struct PRelu_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int bc_C = 0, bc_inner = 0;
    bool slope_scalar = false;

    bool init() override {
        if (!(inputs.size() == 2 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_PRelu(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        const tensor_t* X = inputs[0];
        const tensor_t* S = inputs[1];
        prim_valid = false; device_tag = 0;
        if (X->type != NNR_DATA_TYPE_FLOAT32 || S->type != NNR_DATA_TYPE_FLOAT32) return true;
        if (outputs[0]->ndata != X->ndata) return true;
        if (S->ndata == 1) {
            slope_scalar = true; prim_valid = true;
        } else if (X->ndim >= 2 && (int64_t)S->ndata == X->dims[1]) {
            slope_scalar = false;
            bc_C = X->dims[1];
            size_t inner = 1;
            for (int d = 2; d < X->ndim; ++d) inner *= X->dims[d];
            bc_inner = (int)inner;
            prim_valid = true;
        }
        device_tag = prim_valid ? static_cast<uint8_t>(backend_t::CUDA) : 0;
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        const char* kn = slope_scalar ? "prelu_f32_scalar" : "prelu_f32_per_channel";
        CUfunction f = get_elem(be, kn);
        if (!f) { return fallback->exec(); }

        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_s = (float*)be->cache->ensure_device(inputs[1]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_s || !d_y) { return fallback->exec(); }

        unsigned long long n = outputs[0]->ndata;
        bool ok;
        if (slope_scalar) {
            void* args[] = { &d_x, &d_s, &d_y, &n };
            ok = launch_1d(be, f, args, n);
        } else {
            int C = bc_C, inner = bc_inner;
            void* args[] = { &d_x, &d_s, &d_y, &n, &C, &inner };
            ok = launch_1d(be, f, args, n);
        }
        if (!ok) { return fallback->exec(); }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_LeakyRelu  (int opset, pool_t& pool) { return pool_new<LeakyRelu_cuda>  (pool); }
operator_t* resolver_cuda_op_Elu        (int opset, pool_t& pool) { return pool_new<Elu_cuda>        (pool); }
operator_t* resolver_cuda_op_Celu       (int opset, pool_t& pool) { return pool_new<Celu_cuda>       (pool); }
operator_t* resolver_cuda_op_Selu       (int opset, pool_t& pool) { return pool_new<Selu_cuda>       (pool); }
operator_t* resolver_cuda_op_HardSigmoid(int opset, pool_t& pool) { return pool_new<HardSigmoid_cuda>(pool); }
operator_t* resolver_cuda_op_Clip       (int opset, pool_t& pool) { return pool_new<Clip_cuda>       (pool); }
operator_t* resolver_cuda_op_PRelu      (int opset, pool_t& pool) { return pool_new<PRelu_cuda>      (pool); }

#define NNR_UNARY_RESOLVER(Name) \
operator_t* resolver_cuda_op_##Name(int opset, pool_t& pool) { return pool_new<Name##_cuda>(pool); }

NNR_UNARY_RESOLVER(Neg)
NNR_UNARY_RESOLVER(Sqrt)
NNR_UNARY_RESOLVER(Exp)
NNR_UNARY_RESOLVER(Log)
NNR_UNARY_RESOLVER(Erf)
NNR_UNARY_RESOLVER(Ceil)
NNR_UNARY_RESOLVER(Floor)
NNR_UNARY_RESOLVER(Reciprocal)
NNR_UNARY_RESOLVER(Softplus)
NNR_UNARY_RESOLVER(Softsign)
NNR_UNARY_RESOLVER(HardSwish)
NNR_UNARY_RESOLVER(Sin)
NNR_UNARY_RESOLVER(Cos)
NNR_UNARY_RESOLVER(Tan)
NNR_UNARY_RESOLVER(Asin)
NNR_UNARY_RESOLVER(Acos)
NNR_UNARY_RESOLVER(Atan)
NNR_UNARY_RESOLVER(Sinh)
NNR_UNARY_RESOLVER(Cosh)
NNR_UNARY_RESOLVER(Asinh)
NNR_UNARY_RESOLVER(Acosh)
NNR_UNARY_RESOLVER(Atanh)
NNR_UNARY_RESOLVER(Round)
NNR_UNARY_RESOLVER(Sign)
NNR_UNARY_RESOLVER(Swish)
NNR_UNARY_RESOLVER(Mish)

#undef NNR_UNARY_RESOLVER

} // namespace nnr

#endif // NNR_USE_CUDA
