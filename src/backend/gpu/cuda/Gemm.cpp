#if defined(NNR_USE_CUDA)

// Gemm on CUDA: Y = alpha * op(A) × op(B) + beta * C, via WMMA TF32 kernel
// + scalar epilogue for alpha/beta/bias-broadcast. No cuBLAS.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"

namespace nnr {

operator_t* resolver_default_op_Gemm(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

struct Gemm_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;

    float alpha_ = 1.f, beta_ = 1.f;
    int transA_ = 0, transB_ = 0;

    bool init() override {
        if (!(inputs.size() >= 2 && outputs.size() == 1))
            return false;
        fallback = resolver_default_op_Gemm(opset, ctx->attr_pool);
        fallback->ctx = ctx;
        fallback->opset = opset;
        fallback->op_type = op_type;
        fallback->inputs = inputs;
        fallback->outputs = outputs;
        fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    size_t workspace_size() const override { return fallback ? fallback->workspace_size() : 0; }

    bool reshape() override {
        fallback->post_fn = post_fn;
        fallback->fused_op = fused_op;
        if (!fallback->reshape()) return false;

        const tensor_t* A = inputs[0];
        const tensor_t* B = inputs[1];

        alpha_ = attribute("alpha", 1.0f);
        beta_  = attribute("beta",  1.0f);
        transA_ = static_cast<int>(attribute("transA", (int64_t)0));
        transB_ = static_cast<int>(attribute("transB", (int64_t)0));

        prim_valid = false;
        device_tag = 0;
        if (A->ndim != 2 || B->ndim != 2) return true;
        if (A->type != NNR_DATA_TYPE_FLOAT32) return true;

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();

        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) return fallback->exec();

        const tensor_t* A = inputs[0];
        const tensor_t* B = inputs[1];
        tensor_t* Y = outputs[0];

        const int M = transA_ ? A->dims[1] : A->dims[0];
        const int K = transA_ ? A->dims[0] : A->dims[1];
        const int N = transB_ ? B->dims[0] : B->dims[1];

        float* d_A = (float*)be->cache->ensure_device(A);
        float* d_B = (float*)be->cache->ensure_device(B);
        float* d_Y = (float*)be->cache->alloc_output(Y);
        if (!d_A || !d_B || !d_Y) return fallback->exec();

        // Y = op(A) × op(B) (pre-alpha, pre-beta)
        if (!gpu::gemm_device_f32(be, d_A, d_B, d_Y, M, N, K, transA_, transB_))
            return fallback->exec();

        // Epilogue: Y = alpha * Y + beta * C (if C and beta != 0)
        int bias_kind = 0;
        const float* d_C = nullptr;
        if (inputs.size() > 2 && inputs[2] && beta_ != 0.f) {
            const tensor_t* C = inputs[2];
            d_C = (float*)be->cache->ensure_device(C);
            if (!d_C) return fallback->exec();
            if      (C->ndata == (size_t)M * N) bias_kind = 1;     // elementwise
            else if (C->ndata == (size_t)N)     bias_kind = 2;     // row broadcast (N,)
            else if (C->ndata == (size_t)M)     bias_kind = 3;     // col broadcast (M,)
            else return fallback->exec();                          // bail on odd shapes
        }

        if (!gpu::gemm_epilogue_f32(be, d_Y, d_C, M, N, alpha_,
                                     bias_kind ? beta_ : 0.f, bias_kind))
            return fallback->exec();

        be->cache->mark_written(Y);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_Gemm(int opset, pool_t& pool) {
    return pool_new<Gemm_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
