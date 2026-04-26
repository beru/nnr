#if defined(NNR_USE_CUDA)

// MatMul on CUDA: 2D → TF32 TensorCore GEMM via NVRTC. 3D/4D handled by
// looping over batch (kernel launch per matmul); batches share one graph
// so replay collapses the overhead.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"

namespace nnr {

operator_t* resolver_default_op_MatMul(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

struct MatMul_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;

    int batch = 1;
    int M = 0, K = 0, N = 0;
    int64_t strideA = 0, strideB = 0, strideC = 0;

    bool init() override {
        if (!(inputs.size() >= 2 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_MatMul(opset, ctx->attr_pool);
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

        prim_valid = false; device_tag = 0;
        const tensor_t* A = inputs[0];
        const tensor_t* B = inputs[1];
        const tensor_t* C = outputs[0];
        if (A->type != NNR_DATA_TYPE_FLOAT32) return true;
        if (A->ndim < 2 || B->ndim < 2) return true;

        M = A->dims[A->ndim - 2];
        K = A->dims[A->ndim - 1];
        if (B->dims[B->ndim - 2] != K) return true;
        N = B->dims[B->ndim - 1];

        int bA = 1, bB = 1;
        for (int d = 0; d < A->ndim - 2; ++d) bA *= A->dims[d];
        for (int d = 0; d < B->ndim - 2; ++d) bB *= B->dims[d];
        if (bA != bB) return true;
        batch = bA;

        if (C->ndim < 2) return true;
        if (C->dims[C->ndim - 2] != M || C->dims[C->ndim - 1] != N) return true;
        int bC = 1;
        for (int d = 0; d < C->ndim - 2; ++d) bC *= C->dims[d];
        if (bC != batch) return true;

        strideA = (int64_t)M * K;
        strideB = (int64_t)K * N;
        strideC = (int64_t)M * N;

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();

        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) return fallback->exec();

        float* d_A = (float*)be->cache->ensure_device(inputs[0]);
        float* d_B = (float*)be->cache->ensure_device(inputs[1]);
        float* d_C = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_A || !d_B || !d_C) return fallback->exec();

        for (int b = 0; b < batch; ++b) {
            const float* Ab = d_A + b * strideA;
            const float* Bb = d_B + b * strideB;
            float*       Cb = d_C + b * strideC;
            if (!gpu::gemm_device_f32(be, Ab, Bb, Cb, M, N, K, 0, 0))
                return fallback->exec();
        }

        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_MatMul(int opset, pool_t& pool) {
    return pool_new<MatMul_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
