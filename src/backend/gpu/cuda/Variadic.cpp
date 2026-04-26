#if defined(NNR_USE_CUDA)

// Multi-input elementwise ops: Sum, Max, Min, Mean.
// ONNX variadic ops — accepts 1..N inputs, same shape; output same shape.
// Strategy: allocate output, copy input[0] in, then fold each subsequent input
// via a running op-and-store kernel. For Mean, divide output by N at the end.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"

namespace nnr {

operator_t* resolver_default_op_Sum (int opset, pool_t& pool);
operator_t* resolver_default_op_Max (int opset, pool_t& pool);
operator_t* resolver_default_op_Min (int opset, pool_t& pool);
operator_t* resolver_default_op_Mean(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* variadic_source() {
    return R"CUDA(
extern "C" {

// y[i] = a[i] (copy)
__global__ void copy_f32(const float* __restrict__ a, float* __restrict__ y, unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i];
}

// y[i] += b[i]
__global__ void add_inplace_f32(const float* __restrict__ b, float* __restrict__ y, unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] += b[i];
}

// y[i] = max(y[i], b[i])
__global__ void max_inplace_f32(const float* __restrict__ b, float* __restrict__ y, unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) { float a = y[i], v = b[i]; y[i] = (v > a) ? v : a; }
}

// y[i] = min(y[i], b[i])
__global__ void min_inplace_f32(const float* __restrict__ b, float* __restrict__ y, unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) { float a = y[i], v = b[i]; y[i] = (v < a) ? v : a; }
}

__global__ void scale_f32(float* __restrict__ y, unsigned long long n, float s)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] *= s;
}

} // extern "C"
)CUDA";
}

static CUfunction get_var_kernel(gpu::cuda_backend_t* be, const char* name) {
    return be->nvrtc.get("nnr_variadic_f32", variadic_source(), name,
                         gpu::nvrtc_arch_option(be->device));
}

static bool launch_1d(gpu::cuda_backend_t* be, CUfunction f, void** args, unsigned long long n) {
    constexpr unsigned BLK = 256;
    unsigned grid = (unsigned)((n + BLK - 1) / BLK);
    return gpu::nvrtc_launch(be->device, f, grid, 1, 1, BLK, 1, 1, args);
}

enum class variadic_op { sum, max, min, mean };

struct Variadic_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    variadic_op op = variadic_op::sum;

    virtual operator_t* make_fallback() = 0;

    bool init() override {
        if (inputs.size() < 1 || outputs.size() != 1) return false;
        fallback = make_fallback();
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        size_t n = outputs[0]->ndata;
        if (outputs[0]->type != NNR_DATA_TYPE_FLOAT32) return true;
        for (auto* in : inputs) {
            if (!in || in->type != NNR_DATA_TYPE_FLOAT32 || in->ndata != n) return true;
        }
        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }

        CUfunction f_copy = get_var_kernel(be, "copy_f32");
        const char* fold_name = nullptr;
        switch (op) {
            case variadic_op::sum:  fold_name = "add_inplace_f32"; break;
            case variadic_op::mean: fold_name = "add_inplace_f32"; break;
            case variadic_op::max:  fold_name = "max_inplace_f32"; break;
            case variadic_op::min:  fold_name = "min_inplace_f32"; break;
        }
        CUfunction f_fold = get_var_kernel(be, fold_name);
        if (!f_copy || !f_fold) { return fallback->exec(); }

        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_y) { return fallback->exec(); }

        unsigned long long n = outputs[0]->ndata;
        // Copy input[0] into y
        float* d_0 = (float*)be->cache->ensure_device(inputs[0]);
        if (!d_0) { return fallback->exec(); }
        void* cargs[] = { &d_0, &d_y, &n };
        if (!launch_1d(be, f_copy, cargs, n)) { return fallback->exec(); }

        // Fold the rest
        for (size_t i = 1; i < inputs.size(); ++i) {
            float* d_i = (float*)be->cache->ensure_device(inputs[i]);
            if (!d_i) { return fallback->exec(); }
            void* fargs[] = { &d_i, &d_y, &n };
            if (!launch_1d(be, f_fold, fargs, n)) { return fallback->exec(); }
        }

        if (op == variadic_op::mean) {
            CUfunction f_scale = get_var_kernel(be, "scale_f32");
            if (!f_scale) { return fallback->exec(); }
            float s = 1.f / (float)inputs.size();
            void* sargs[] = { &d_y, &n, &s };
            if (!launch_1d(be, f_scale, sargs, n)) { return fallback->exec(); }
        }

        be->cache->mark_written(outputs[0]);
        return true;
    }
};

struct Sum_cuda  : Variadic_cuda { Sum_cuda()  { op = variadic_op::sum;  } operator_t* make_fallback() override { return resolver_default_op_Sum (opset, ctx->attr_pool); } };
struct Max_cuda  : Variadic_cuda { Max_cuda()  { op = variadic_op::max;  } operator_t* make_fallback() override { return resolver_default_op_Max (opset, ctx->attr_pool); } };
struct Min_cuda  : Variadic_cuda { Min_cuda()  { op = variadic_op::min;  } operator_t* make_fallback() override { return resolver_default_op_Min (opset, ctx->attr_pool); } };
struct Mean_cuda : Variadic_cuda { Mean_cuda() { op = variadic_op::mean; } operator_t* make_fallback() override { return resolver_default_op_Mean(opset, ctx->attr_pool); } };

} // namespace

operator_t* resolver_cuda_op_Sum (int opset, pool_t& pool) { return pool_new<Sum_cuda> (pool); }
operator_t* resolver_cuda_op_Max (int opset, pool_t& pool) { return pool_new<Max_cuda> (pool); }
operator_t* resolver_cuda_op_Min (int opset, pool_t& pool) { return pool_new<Min_cuda> (pool); }
operator_t* resolver_cuda_op_Mean(int opset, pool_t& pool) { return pool_new<Mean_cuda>(pool); }

} // namespace nnr

#endif // NNR_USE_CUDA
