#if defined(NNR_USE_CUDA)

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "elementwise.h"

namespace nnr {

operator_t* resolver_default_op_Add(int opset, pool_t& pool);
operator_t* resolver_default_op_Mul(int opset, pool_t& pool);
operator_t* resolver_default_op_Sub(int opset, pool_t& pool);
operator_t* resolver_default_op_Div(int opset, pool_t& pool);
operator_t* resolver_default_op_Relu(int opset, pool_t& pool);
operator_t* resolver_default_op_Sigmoid(int opset, pool_t& pool);
operator_t* resolver_default_op_Tanh(int opset, pool_t& pool);
operator_t* resolver_default_op_Abs(int opset, pool_t& pool);
operator_t* resolver_default_op_Gelu(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

// Launch an elementwise kernel (binary or unary) over `n` float elements.
// args_ptrs: array of arg pointers matching the kernel signature.
static bool launch_pointwise(gpu::cuda_backend_t* be, CUfunction f,
                             void** args, unsigned long long n)
{
    constexpr unsigned BLK = 256;
    unsigned long long blocks_ll = (n + BLK - 1) / BLK;
    unsigned grid = (blocks_ll > 0x7fffffffu) ? 0x7fffffffu : (unsigned)blocks_ll;
    return gpu::nvrtc_launch(be->device, f, grid, 1, 1, BLK, 1, 1, args);
}

static CUfunction get_elem_kernel(gpu::cuda_backend_t* be, const char* name) {
    return be->nvrtc.get("nnr_elementwise_f32",
                         gpu::elementwise_f32_source(),
                         name,
                         gpu::nvrtc_arch_option(be->device));
}

// ----- Binary op base (Add, Mul, Sub, Div) -----
// Supports three broadcast patterns:
//   SAME         — A.ndata == B.ndata, elementwise.
//   SCALAR_B     — B.ndata == 1, scalar on the B side.
//   BIAS_NCHW    — A is (..., C, inner...), B.ndata == A.dims[channel_axis].
// Other broadcast shapes fall back to the CPU op.

enum class broadcast_t { NONE, SAME, SCALAR_B, BIAS_NCHW };

struct Binary_cuda : public operator_t {
    operator_t* fallback = nullptr;
    // Per-op kernel prefix ("add_f32", "mul_f32", ...); we append suffixes
    // "_scalar_b" or "_bias_nchw" at launch time.
    const char* kernel_base = nullptr;  // e.g. "add_f32" (also used for SAME)
    const char* kernel_scalar = nullptr;
    const char* kernel_bias   = nullptr;
    // Optional int64 kernel set; if non-null, int64 inputs go through these.
    const char* kernel_base_i64   = nullptr;
    const char* kernel_scalar_i64 = nullptr;

    broadcast_t pattern = broadcast_t::NONE;
    int bc_C = 0, bc_inner = 0;
    bool swapped = false;  // a/b swapped when the tensor is input[1] and bias/scalar is input[0]
    bool is_i64 = false;   // selected dtype path for this run

    virtual operator_t* make_fallback() = 0;

    bool init() override {
        if (!(inputs.size() == 2 && outputs.size() == 1)) return false;
        fallback = make_fallback();
        fallback->ctx = ctx;
        fallback->opset = opset;
        fallback->op_type = op_type;
        fallback->inputs = inputs;
        fallback->outputs = outputs;
        fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        pattern = broadcast_t::NONE;
        swapped = false;
        is_i64 = false;
        const tensor_t* A = inputs[0];
        const tensor_t* B = inputs[1];
        bool f32 = (A->type == NNR_DATA_TYPE_FLOAT32 && B->type == NNR_DATA_TYPE_FLOAT32);
        bool i64 = (A->type == NNR_DATA_TYPE_INT64 && B->type == NNR_DATA_TYPE_INT64
                    && kernel_base_i64 != nullptr);
        if (i64) {
            is_i64 = true;
            auto check = [&](const tensor_t* a, const tensor_t* b) -> broadcast_t {
                if (outputs[0]->ndata != a->ndata) return broadcast_t::NONE;
                if (a->ndata == b->ndata) return broadcast_t::SAME;
                if (b->ndata == 1) return (kernel_scalar_i64 ? broadcast_t::SCALAR_B : broadcast_t::NONE);
                return broadcast_t::NONE;
            };
            pattern = check(A, B);
            if (pattern == broadcast_t::NONE && is_commutative()) {
                auto p2 = check(B, A);
                if (p2 != broadcast_t::NONE) { pattern = p2; swapped = true; std::swap(A, B); }
            }
        } else if (f32) {
            // Check (A,B) as-is; if that doesn't match, also try (B,A) swapped.
            // Swapping is safe for Add/Mul (commutative); for Sub/Div the kernel
            // uses fixed order so we only allow swap when the op is Add or Mul.
            // Swap-allowed when kernel_bias is non-null AND op is commutative —
            // encoded by also providing a "scalar-a" kernel isn't necessary since
            // we handle swap by reversing input indices at launch time.
            bool try_swap = is_commutative();

            auto check = [&](const tensor_t* a, const tensor_t* b) -> broadcast_t {
                if (outputs[0]->ndata != a->ndata) return broadcast_t::NONE;
                if (a->ndata == b->ndata) return broadcast_t::SAME;
                if (b->ndata == 1) return (kernel_scalar ? broadcast_t::SCALAR_B : broadcast_t::NONE);
                if (a->ndim >= 2 && kernel_bias != nullptr) {
                    int C = a->dims[1];
                    if ((int64_t)b->ndata == C) return broadcast_t::BIAS_NCHW;
                }
                return broadcast_t::NONE;
            };

            pattern = check(A, B);
            if (pattern == broadcast_t::NONE && try_swap) {
                auto p2 = check(B, A);
                if (p2 != broadcast_t::NONE) {
                    pattern = p2;
                    swapped = true;
                    std::swap(A, B);
                }
            }
            if (pattern == broadcast_t::BIAS_NCHW) {
                bc_C = A->dims[1];
                size_t inner = 1;
                for (int d = 2; d < A->ndim; ++d) inner *= A->dims[d];
                bc_inner = (int)inner;
            }
        }
        device_tag = (pattern != broadcast_t::NONE)
                   ? static_cast<uint8_t>(backend_t::CUDA) : 0;
        return true;
    }

    virtual bool is_commutative() const { return false; }

    bool exec() override {
        if (pattern == broadcast_t::NONE) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }

        const char* kn = nullptr;
        switch (pattern) {
            case broadcast_t::SAME:      kn = is_i64 ? kernel_base_i64   : kernel_base;   break;
            case broadcast_t::SCALAR_B:  kn = is_i64 ? kernel_scalar_i64 : kernel_scalar; break;
            case broadcast_t::BIAS_NCHW: kn = kernel_bias;   break;
            default: break;
        }
        CUfunction f = get_elem_kernel(be, kn);
        if (!f) { return fallback->exec(); }

        void* d_a = be->cache->ensure_device(inputs[swapped ? 1 : 0]);
        void* d_b = be->cache->ensure_device(inputs[swapped ? 0 : 1]);
        void* d_y = be->cache->alloc_output(outputs[0]);
        if (!d_a || !d_b || !d_y) { return fallback->exec(); }

        unsigned long long n = outputs[0]->ndata;
        bool ok = false;
        if (pattern == broadcast_t::BIAS_NCHW) {
            int C = bc_C, inner = bc_inner;
            void* args[] = { &d_a, &d_b, &d_y, &n, &C, &inner };
            ok = launch_pointwise(be, f, args, n);
        } else {
            void* args[] = { &d_a, &d_b, &d_y, &n };
            ok = launch_pointwise(be, f, args, n);
        }
        if (!ok) { return fallback->exec(); }

        be->cache->mark_written(outputs[0]);
        return true;
    }
};

struct Add_cuda : Binary_cuda {
    Add_cuda() {
        kernel_base = "add_f32"; kernel_scalar = "add_f32_scalar_b"; kernel_bias = "add_f32_bias_nchw";
        kernel_base_i64 = "add_i64"; kernel_scalar_i64 = "add_i64_scalar_b";
    }
    operator_t* make_fallback() override { return resolver_default_op_Add(opset, ctx->attr_pool); }
    bool is_commutative() const override { return true; }
};

struct Mul_cuda : Binary_cuda {
    Mul_cuda() {
        kernel_base = "mul_f32"; kernel_scalar = "mul_f32_scalar_b"; kernel_bias = "mul_f32_bias_nchw";
        kernel_base_i64 = "mul_i64"; kernel_scalar_i64 = "mul_i64_scalar_b";
    }
    operator_t* make_fallback() override { return resolver_default_op_Mul(opset, ctx->attr_pool); }
    bool is_commutative() const override { return true; }
};

struct Sub_cuda : Binary_cuda {
    Sub_cuda() {
        kernel_base = "sub_f32"; kernel_scalar = "sub_f32_scalar_b";
        kernel_base_i64 = "sub_i64"; kernel_scalar_i64 = "sub_i64_scalar_b";
    }
    operator_t* make_fallback() override { return resolver_default_op_Sub(opset, ctx->attr_pool); }
};

struct Div_cuda : Binary_cuda {
    Div_cuda() { kernel_base = "div_f32"; kernel_scalar = "div_f32_scalar_b"; }
    operator_t* make_fallback() override { return resolver_default_op_Div(opset, ctx->attr_pool); }
};

// ----- Unary op base (Relu, Sigmoid) -----

struct Unary_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    const char* kernel_name = nullptr;

    virtual operator_t* make_fallback() = 0;

    bool init() override {
        if (!(inputs.size() == 1 && outputs.size() == 1)) return false;
        fallback = make_fallback();
        fallback->ctx = ctx;
        fallback->opset = opset;
        fallback->op_type = op_type;
        fallback->inputs = inputs;
        fallback->outputs = outputs;
        fallback->attrs = attrs;
        fallback->init();
        // Inherit fusable_apply so fuse_post_ops can fold this op into an
        // upstream Conv/Gemm. CUDA Conv re-decodes by op_type in reshape() —
        // the function pointer itself is only invoked on the rare CPU-fallback
        // path (prim_valid=false), where it's paired with the wrapper's
        // `fallback` as operand (not the CUDA wrapper).
        fusable_apply = fallback->fusable_apply;
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        const tensor_t* X = inputs[0];
        prim_valid = (X->type == NNR_DATA_TYPE_FLOAT32
                   && outputs[0]->ndata == X->ndata);
        device_tag = prim_valid ? static_cast<uint8_t>(backend_t::CUDA) : 0;
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }

        CUfunction f = get_elem_kernel(be, kernel_name);
        if (!f) { return fallback->exec(); }

        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        unsigned long long n = outputs[0]->ndata;
        void* args[] = { &d_x, &d_y, &n };
        if (!launch_pointwise(be, f, args, n)) { return fallback->exec(); }

        be->cache->mark_written(outputs[0]);
        return true;
    }
};

struct Relu_cuda : Unary_cuda {
    Relu_cuda() { kernel_name = "relu_f32"; }
    operator_t* make_fallback() override { return resolver_default_op_Relu(opset, ctx->attr_pool); }
};

struct Sigmoid_cuda : Unary_cuda {
    Sigmoid_cuda() { kernel_name = "sigmoid_f32"; }
    operator_t* make_fallback() override { return resolver_default_op_Sigmoid(opset, ctx->attr_pool); }
};

struct Tanh_cuda : Unary_cuda {
    Tanh_cuda() { kernel_name = "tanh_f32"; }
    operator_t* make_fallback() override { return resolver_default_op_Tanh(opset, ctx->attr_pool); }
};

struct Abs_cuda : Unary_cuda {
    Abs_cuda() { kernel_name = "abs_f32"; }
    operator_t* make_fallback() override { return resolver_default_op_Abs(opset, ctx->attr_pool); }
};

struct Gelu_cuda : Unary_cuda {
    Gelu_cuda() { kernel_name = "gelu_f32"; }
    operator_t* make_fallback() override { return resolver_default_op_Gelu(opset, ctx->attr_pool); }
};

} // namespace

operator_t* resolver_cuda_op_Add    (int opset, pool_t& pool) { return pool_new<Add_cuda>(pool); }
operator_t* resolver_cuda_op_Mul    (int opset, pool_t& pool) { return pool_new<Mul_cuda>(pool); }
operator_t* resolver_cuda_op_Sub    (int opset, pool_t& pool) { return pool_new<Sub_cuda>(pool); }
operator_t* resolver_cuda_op_Div    (int opset, pool_t& pool) { return pool_new<Div_cuda>(pool); }
operator_t* resolver_cuda_op_Relu   (int opset, pool_t& pool) { return pool_new<Relu_cuda>(pool); }
operator_t* resolver_cuda_op_Sigmoid(int opset, pool_t& pool) { return pool_new<Sigmoid_cuda>(pool); }
operator_t* resolver_cuda_op_Tanh   (int opset, pool_t& pool) { return pool_new<Tanh_cuda>(pool); }
operator_t* resolver_cuda_op_Abs    (int opset, pool_t& pool) { return pool_new<Abs_cuda>(pool); }
operator_t* resolver_cuda_op_Gelu   (int opset, pool_t& pool) { return pool_new<Gelu_cuda>(pool); }

} // namespace nnr

#endif // NNR_USE_CUDA
