#if defined(NNR_USE_CUDA)

// Reductions: ReduceSum, ReduceMean, ReduceMax, ReduceMin via NVRTC.
// Also ArgMax / ArgMin and LogSoftmax.
// Constraint: reduced axes must form a contiguous trailing suffix of the
// input's dimensions (the common case: reduce over last K dims). Non-contiguous
// or leading reductions fall back to CPU.
// F32 input; ArgMax/Min output INT64.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "attr_key.h"
#include <vector>

namespace nnr {

operator_t* resolver_default_op_ReduceSum (int opset, pool_t& pool);
operator_t* resolver_default_op_ReduceMean(int opset, pool_t& pool);
operator_t* resolver_default_op_ReduceMax (int opset, pool_t& pool);
operator_t* resolver_default_op_ReduceMin (int opset, pool_t& pool);
operator_t* resolver_default_op_ArgMax    (int opset, pool_t& pool);
operator_t* resolver_default_op_ArgMin    (int opset, pool_t& pool);
operator_t* resolver_default_op_LogSoftmax(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* reduce_source() {
    return R"CUDA(
#ifndef INFINITY
#define INFINITY __int_as_float(0x7F800000)
#endif
extern "C" {

// Block per outer row; threads reduce over R elements. Op selected by kernel name.

__device__ __forceinline__ float _block_reduce_sum(float v, float* smem, int tid, int nth) {
    smem[tid] = v; __syncthreads();
    for (int off = nth >> 1; off > 0; off >>= 1) {
        if (tid < off) smem[tid] += smem[tid + off];
        __syncthreads();
    }
    return smem[0];
}

__device__ __forceinline__ float _block_reduce_max(float v, float* smem, int tid, int nth) {
    smem[tid] = v; __syncthreads();
    for (int off = nth >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            float a = smem[tid], b = smem[tid + off];
            smem[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }
    return smem[0];
}

__device__ __forceinline__ float _block_reduce_min(float v, float* smem, int tid, int nth) {
    smem[tid] = v; __syncthreads();
    for (int off = nth >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            float a = smem[tid], b = smem[tid + off];
            smem[tid] = (a < b) ? a : b;
        }
        __syncthreads();
    }
    return smem[0];
}

__global__ void reduce_sum_trailing_f32(const float* __restrict__ x,
                                        float* __restrict__ y,
                                        int outer, int R)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= outer) return;
    const float* xr = x + (size_t)row * R;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    float acc = 0.f;
    for (int j = tid; j < R; j += nth) acc += xr[j];
    float s = _block_reduce_sum(acc, smem, tid, nth);
    if (tid == 0) y[row] = s;
}

__global__ void reduce_mean_trailing_f32(const float* __restrict__ x,
                                         float* __restrict__ y,
                                         int outer, int R)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= outer) return;
    const float* xr = x + (size_t)row * R;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    float acc = 0.f;
    for (int j = tid; j < R; j += nth) acc += xr[j];
    float s = _block_reduce_sum(acc, smem, tid, nth);
    if (tid == 0) y[row] = s / (float)R;
}

__global__ void reduce_max_trailing_f32(const float* __restrict__ x,
                                        float* __restrict__ y,
                                        int outer, int R)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= outer) return;
    const float* xr = x + (size_t)row * R;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    float acc = -INFINITY;
    for (int j = tid; j < R; j += nth) { float v = xr[j]; if (v > acc) acc = v; }
    float s = _block_reduce_max(acc, smem, tid, nth);
    if (tid == 0) y[row] = s;
}

__global__ void reduce_min_trailing_f32(const float* __restrict__ x,
                                        float* __restrict__ y,
                                        int outer, int R)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= outer) return;
    const float* xr = x + (size_t)row * R;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    float acc = INFINITY;
    for (int j = tid; j < R; j += nth) { float v = xr[j]; if (v < acc) acc = v; }
    float s = _block_reduce_min(acc, smem, tid, nth);
    if (tid == 0) y[row] = s;
}

// int64 versions (used by SSD post-NMS index reductions).

#define I64_MIN_R ((long long)0x8000000000000000LL)
#define I64_MAX_R ((long long)0x7fffffffffffffffLL)

__device__ __forceinline__ long long _block_reduce_sum_i64(long long v, long long* smem, int tid, int nth) {
    smem[tid] = v; __syncthreads();
    for (int off = nth >> 1; off > 0; off >>= 1) {
        if (tid < off) smem[tid] += smem[tid + off];
        __syncthreads();
    }
    return smem[0];
}
__device__ __forceinline__ long long _block_reduce_max_i64(long long v, long long* smem, int tid, int nth) {
    smem[tid] = v; __syncthreads();
    for (int off = nth >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            long long a = smem[tid], b = smem[tid + off];
            smem[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }
    return smem[0];
}
__device__ __forceinline__ long long _block_reduce_min_i64(long long v, long long* smem, int tid, int nth) {
    smem[tid] = v; __syncthreads();
    for (int off = nth >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            long long a = smem[tid], b = smem[tid + off];
            smem[tid] = (a < b) ? a : b;
        }
        __syncthreads();
    }
    return smem[0];
}

__global__ void reduce_sum_trailing_i64(const long long* __restrict__ x,
                                        long long* __restrict__ y,
                                        int outer, int R)
{
    extern __shared__ long long lmem[];
    const int row = blockIdx.x;
    if (row >= outer) return;
    const long long* xr = x + (size_t)row * R;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    long long acc = 0;
    for (int j = tid; j < R; j += nth) acc += xr[j];
    long long s = _block_reduce_sum_i64(acc, lmem, tid, nth);
    if (tid == 0) y[row] = s;
}

__global__ void reduce_max_trailing_i64(const long long* __restrict__ x,
                                        long long* __restrict__ y,
                                        int outer, int R)
{
    extern __shared__ long long lmem[];
    const int row = blockIdx.x;
    if (row >= outer) return;
    const long long* xr = x + (size_t)row * R;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    long long acc = I64_MIN_R;
    for (int j = tid; j < R; j += nth) { long long v = xr[j]; if (v > acc) acc = v; }
    long long s = _block_reduce_max_i64(acc, lmem, tid, nth);
    if (tid == 0) y[row] = s;
}

__global__ void reduce_min_trailing_i64(const long long* __restrict__ x,
                                        long long* __restrict__ y,
                                        int outer, int R)
{
    extern __shared__ long long lmem[];
    const int row = blockIdx.x;
    if (row >= outer) return;
    const long long* xr = x + (size_t)row * R;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    long long acc = I64_MAX_R;
    for (int j = tid; j < R; j += nth) { long long v = xr[j]; if (v < acc) acc = v; }
    long long s = _block_reduce_min_i64(acc, lmem, tid, nth);
    if (tid == 0) y[row] = s;
}

// int32 variants (used by ssd-12 post-NMS index chain).

#define I32_MIN_R ((int)0x80000000)
#define I32_MAX_R ((int)0x7fffffff)

__device__ __forceinline__ int _block_reduce_sum_i32(int v, int* smem, int tid, int nth) {
    smem[tid] = v; __syncthreads();
    for (int off = nth >> 1; off > 0; off >>= 1) {
        if (tid < off) smem[tid] += smem[tid + off];
        __syncthreads();
    }
    return smem[0];
}
__device__ __forceinline__ int _block_reduce_max_i32(int v, int* smem, int tid, int nth) {
    smem[tid] = v; __syncthreads();
    for (int off = nth >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            int a = smem[tid], b = smem[tid + off];
            smem[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }
    return smem[0];
}
__device__ __forceinline__ int _block_reduce_min_i32(int v, int* smem, int tid, int nth) {
    smem[tid] = v; __syncthreads();
    for (int off = nth >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            int a = smem[tid], b = smem[tid + off];
            smem[tid] = (a < b) ? a : b;
        }
        __syncthreads();
    }
    return smem[0];
}

__global__ void reduce_sum_trailing_i32(const int* __restrict__ x,
                                        int* __restrict__ y,
                                        int outer, int R)
{
    extern __shared__ int imem[];
    const int row = blockIdx.x;
    if (row >= outer) return;
    const int* xr = x + (size_t)row * R;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    int acc = 0;
    for (int j = tid; j < R; j += nth) acc += xr[j];
    int s = _block_reduce_sum_i32(acc, imem, tid, nth);
    if (tid == 0) y[row] = s;
}
__global__ void reduce_max_trailing_i32(const int* __restrict__ x,
                                        int* __restrict__ y,
                                        int outer, int R)
{
    extern __shared__ int imem[];
    const int row = blockIdx.x;
    if (row >= outer) return;
    const int* xr = x + (size_t)row * R;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    int acc = I32_MIN_R;
    for (int j = tid; j < R; j += nth) { int v = xr[j]; if (v > acc) acc = v; }
    int s = _block_reduce_max_i32(acc, imem, tid, nth);
    if (tid == 0) y[row] = s;
}
__global__ void reduce_min_trailing_i32(const int* __restrict__ x,
                                        int* __restrict__ y,
                                        int outer, int R)
{
    extern __shared__ int imem[];
    const int row = blockIdx.x;
    if (row >= outer) return;
    const int* xr = x + (size_t)row * R;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    int acc = I32_MAX_R;
    for (int j = tid; j < R; j += nth) { int v = xr[j]; if (v < acc) acc = v; }
    int s = _block_reduce_min_i32(acc, imem, tid, nth);
    if (tid == 0) y[row] = s;
}

// ArgMax/ArgMin: one block per outer; thread finds local best (value, index),
// then block-reduce picks the winning index. smem: 2 * nth * (float + int).
// For simplicity we pack (val, idx) into sequential smem regions: floats first, ints next.
__global__ void argmax_trailing_i64(const float* __restrict__ x,
                                    long long* __restrict__ y,
                                    int outer, int R)
{
    extern __shared__ float smem[];
    float* vmem = smem;
    int*   imem = (int*)(smem + blockDim.x);
    const int row = blockIdx.x;
    if (row >= outer) return;
    const float* xr = x + (size_t)row * R;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    float best = -INFINITY;
    int   best_i = 0;
    for (int j = tid; j < R; j += nth) {
        float v = xr[j];
        if (v > best) { best = v; best_i = j; }
    }
    vmem[tid] = best; imem[tid] = best_i;
    __syncthreads();
    for (int off = nth >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            float a = vmem[tid], b = vmem[tid + off];
            if (b > a) { vmem[tid] = b; imem[tid] = imem[tid + off]; }
        }
        __syncthreads();
    }
    if (tid == 0) y[row] = (long long)imem[0];
}

__global__ void argmin_trailing_i64(const float* __restrict__ x,
                                    long long* __restrict__ y,
                                    int outer, int R)
{
    extern __shared__ float smem[];
    float* vmem = smem;
    int*   imem = (int*)(smem + blockDim.x);
    const int row = blockIdx.x;
    if (row >= outer) return;
    const float* xr = x + (size_t)row * R;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    float best = INFINITY;
    int   best_i = 0;
    for (int j = tid; j < R; j += nth) {
        float v = xr[j];
        if (v < best) { best = v; best_i = j; }
    }
    vmem[tid] = best; imem[tid] = best_i;
    __syncthreads();
    for (int off = nth >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            float a = vmem[tid], b = vmem[tid + off];
            if (b < a) { vmem[tid] = b; imem[tid] = imem[tid + off]; }
        }
        __syncthreads();
    }
    if (tid == 0) y[row] = (long long)imem[0];
}

// LogSoftmax along last axis: y = x - max - log(sum(exp(x - max))).
__global__ void logsoftmax_lastaxis_f32(const float* __restrict__ x,
                                        float* __restrict__ y,
                                        int D, int outer)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= outer) return;
    const float* xr = x + (size_t)row * D;
    float* yr       = y + (size_t)row * D;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    // pass 1: max
    float lmax = -INFINITY;
    for (int j = tid; j < D; j += nth) { float v = xr[j]; if (v > lmax) lmax = v; }
    float mx = _block_reduce_max(lmax, smem, tid, nth);

    // pass 2: sum of exp(x - max)
    float lsum = 0.f;
    for (int j = tid; j < D; j += nth) lsum += __expf(xr[j] - mx);
    float sm = _block_reduce_sum(lsum, smem, tid, nth);
    float log_sm = __logf(sm);

    // pass 3: y = x - max - log(sum)
    for (int j = tid; j < D; j += nth) yr[j] = xr[j] - mx - log_sm;
}

} // extern "C"
)CUDA";
}

static CUfunction get_reduce_kernel(gpu::cuda_backend_t* be, const char* name) {
    return be->nvrtc.get("nnr_reduce_f32",
                         reduce_source(),
                         name,
                         gpu::nvrtc_arch_option(be->device));
}

// Resolve "axes" attribute/input to a set of axis indices (normalized to [0,ndim)).
// Returns true on success.
static bool parse_axes(operator_t* self, std::vector<int>& out_axes, int ndim, int opset,
                       int axes_since_opset)
{
    out_axes.clear();
    int64_t* ints = nullptr;
    int n = 0;

    if (opset >= axes_since_opset
        && self->inputs.size() >= 2 && self->inputs[1] && self->inputs[1]->data
        && self->inputs[1]->type == NNR_DATA_TYPE_INT64) {
        const int64_t* p = (const int64_t*)self->inputs[1]->data;
        n = (int)self->inputs[1]->ndata;
        for (int i = 0; i < n; ++i) {
            int a = (int)p[i]; if (a < 0) a += ndim;
            if (a < 0 || a >= ndim) return false;
            out_axes.push_back(a);
        }
    } else {
        n = self->attribute(attr_key_t::axes, ints);
        for (int i = 0; i < n; ++i) {
            int a = (int)ints[i]; if (a < 0) a += ndim;
            if (a < 0 || a >= ndim) return false;
            out_axes.push_back(a);
        }
    }
    return true;
}

// ---- shared Reduce base ----

struct Reduce_cuda_base : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    const char* kernel_name = nullptr;        // fp32 kernel
    const char* kernel_name_i64 = nullptr;    // optional int64 kernel
    const char* kernel_name_i32 = nullptr;    // optional int32 kernel
    int axes_since_opset = 18;   // ReduceSum uses 13
    int outer = 0, R = 0;

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
        const tensor_t* x = inputs[0];
        if ((x->type != NNR_DATA_TYPE_FLOAT32 && x->type != NNR_DATA_TYPE_INT64
             && x->type != NNR_DATA_TYPE_INT32) || x->ndim < 1) return true;
        if (outputs[0]->type != x->type) return true;
        if (x->type == NNR_DATA_TYPE_INT64 && !kernel_name_i64) return true;
        if (x->type == NNR_DATA_TYPE_INT32 && !kernel_name_i32) return true;

        std::vector<int> axes;
        if (!parse_axes(this, axes, x->ndim, opset, axes_since_opset)) return true;
        // If no axes given, ONNX convention: reduce all (unless noop_with_empty_axes).
        int noop = (int)attribute(attr_key_t::noop_with_empty_axes, (int64_t)0);
        if (axes.empty()) {
            if (noop) return true;   // identity — let CPU handle (cheap)
            for (int d = 0; d < x->ndim; ++d) axes.push_back(d);
        }
        // Require axes to be a contiguous trailing suffix.
        std::sort(axes.begin(), axes.end());
        for (size_t i = 1; i < axes.size(); ++i) if (axes[i] != axes[i-1] + 1) return true;
        if (axes.back() != x->ndim - 1) return true;

        int kept = (int)axes.front();   // first reduced axis
        outer = 1; R = 1;
        for (int d = 0; d < kept; ++d)        outer *= x->dims[d];
        for (int d = kept; d < x->ndim; ++d)  R     *= x->dims[d];
        if (R <= 0 || outer <= 0) return true;

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec_common(CUfunction f, void* out_ptr, size_t out_type_size) {
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        if (!d_x || !out_ptr) { return fallback->exec(); }

        unsigned block = 256;
        while ((int)block > R && block > 32) block >>= 1;
        if (block < 32) block = 32;
        unsigned shared = block * (unsigned)out_type_size;
        int _o = outer, _R = R;
        void* args[] = { &d_x, &out_ptr, &_o, &_R };
        if (!gpu::nvrtc_launch(be->device, f, (unsigned)outer, 1, 1, block, 1, 1, args, shared)) {
            return fallback->exec();
        }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

// ---- ReduceSum / Mean / Max / Min ----

struct ReduceSimple_cuda : Reduce_cuda_base {
    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }

        const char* kn;
        size_t elem_size;
        switch (inputs[0]->type) {
            case NNR_DATA_TYPE_INT64: kn = kernel_name_i64; elem_size = sizeof(long long); break;
            case NNR_DATA_TYPE_INT32: kn = kernel_name_i32; elem_size = sizeof(int);       break;
            default:                  kn = kernel_name;     elem_size = sizeof(float);     break;
        }
        CUfunction f = get_reduce_kernel(be, kn);
        if (!f) { return fallback->exec(); }

        void* d_x = be->cache->ensure_device(inputs[0]);
        void* d_y = be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        unsigned block = 256;
        while ((int)block > R && block > 32) block >>= 1;
        if (block < 32) block = 32;
        unsigned shared = block * (unsigned)elem_size;
        int _o = outer, _R = R;
        void* args[] = { &d_x, &d_y, &_o, &_R };
        if (!gpu::nvrtc_launch(be->device, f, (unsigned)outer, 1, 1, block, 1, 1, args, shared)) {
            return fallback->exec();
        }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

struct ReduceSum_cuda : ReduceSimple_cuda {
    ReduceSum_cuda() {
        kernel_name     = "reduce_sum_trailing_f32";
        kernel_name_i64 = "reduce_sum_trailing_i64";
        kernel_name_i32 = "reduce_sum_trailing_i32";
        axes_since_opset = 13;
    }
    operator_t* make_fallback() override { return resolver_default_op_ReduceSum(opset, ctx->attr_pool); }
};
struct ReduceMean_cuda : ReduceSimple_cuda {
    // No int paths (mean of ints would need integer division semantics — unused on bench models).
    ReduceMean_cuda() { kernel_name = "reduce_mean_trailing_f32"; }
    operator_t* make_fallback() override { return resolver_default_op_ReduceMean(opset, ctx->attr_pool); }
};
struct ReduceMax_cuda : ReduceSimple_cuda {
    ReduceMax_cuda() {
        kernel_name     = "reduce_max_trailing_f32";
        kernel_name_i64 = "reduce_max_trailing_i64";
        kernel_name_i32 = "reduce_max_trailing_i32";
    }
    operator_t* make_fallback() override { return resolver_default_op_ReduceMax(opset, ctx->attr_pool); }
};
struct ReduceMin_cuda : ReduceSimple_cuda {
    ReduceMin_cuda() {
        kernel_name     = "reduce_min_trailing_f32";
        kernel_name_i64 = "reduce_min_trailing_i64";
        kernel_name_i32 = "reduce_min_trailing_i32";
    }
    operator_t* make_fallback() override { return resolver_default_op_ReduceMin(opset, ctx->attr_pool); }
};

// ---- ArgMax / ArgMin: reduce over a single axis (attribute), output int64 ----

struct Arg_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    const char* kernel_name = nullptr;
    int outer = 0, R = 0;
    int axis = 0;

    virtual operator_t* make_fallback() = 0;

    bool init() override {
        if (!(inputs.size() == 1 && outputs.size() == 1)) return false;
        fallback = make_fallback();
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        axis = (int)attribute(attr_key_t::axis, (int64_t)0);
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* x = inputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return true;
        if (outputs[0]->type != NNR_DATA_TYPE_INT64) return true;

        int caxis = axis < 0 ? axis + x->ndim : axis;
        if (caxis != x->ndim - 1) return true;  // only last-axis supported

        outer = 1;
        for (int d = 0; d < caxis; ++d) outer *= x->dims[d];
        R = x->dims[caxis];

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        CUfunction f = get_reduce_kernel(be, kernel_name);
        if (!f) { return fallback->exec(); }

        float* d_x     = (float*)be->cache->ensure_device(inputs[0]);
        long long* d_y = (long long*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        unsigned block = 256;
        while ((int)block > R && block > 32) block >>= 1;
        if (block < 32) block = 32;
        unsigned shared = block * (sizeof(float) + sizeof(int));
        int _o = outer, _R = R;
        void* args[] = { &d_x, &d_y, &_o, &_R };
        if (!gpu::nvrtc_launch(be->device, f, (unsigned)outer, 1, 1, block, 1, 1, args, shared)) {
            return fallback->exec();
        }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

struct ArgMax_cuda : Arg_cuda {
    ArgMax_cuda() { kernel_name = "argmax_trailing_i64"; }
    operator_t* make_fallback() override { return resolver_default_op_ArgMax(opset, ctx->attr_pool); }
};
struct ArgMin_cuda : Arg_cuda {
    ArgMin_cuda() { kernel_name = "argmin_trailing_i64"; }
    operator_t* make_fallback() override { return resolver_default_op_ArgMin(opset, ctx->attr_pool); }
};

// ---- LogSoftmax (last axis) ----

struct LogSoftmax_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int axis = -1, D = 0, outer = 0;

    bool init() override {
        if (!(inputs.size() == 1 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_LogSoftmax(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        axis = (int)attribute(attr_key_t::axis, (int64_t)-1);
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* x = inputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return true;
        int caxis = axis < 0 ? axis + x->ndim : axis;
        if (caxis != x->ndim - 1) return true;
        D = x->dims[caxis];
        outer = 1; for (int d = 0; d < caxis; ++d) outer *= x->dims[d];
        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        CUfunction f = get_reduce_kernel(be, "logsoftmax_lastaxis_f32");
        if (!f) { return fallback->exec(); }

        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        unsigned block = 256;
        while ((int)block > D && block > 32) block >>= 1;
        if (block < 32) block = 32;
        unsigned shared = block * sizeof(float);
        int _D = D, _outer = outer;
        void* args[] = { &d_x, &d_y, &_D, &_outer };
        if (!gpu::nvrtc_launch(be->device, f, (unsigned)outer, 1, 1, block, 1, 1, args, shared)) {
            return fallback->exec();
        }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_ReduceSum (int opset, pool_t& pool) { return pool_new<ReduceSum_cuda> (pool); }
operator_t* resolver_cuda_op_ReduceMean(int opset, pool_t& pool) { return pool_new<ReduceMean_cuda>(pool); }
operator_t* resolver_cuda_op_ReduceMax (int opset, pool_t& pool) { return pool_new<ReduceMax_cuda> (pool); }
operator_t* resolver_cuda_op_ReduceMin (int opset, pool_t& pool) { return pool_new<ReduceMin_cuda> (pool); }
operator_t* resolver_cuda_op_ArgMax    (int opset, pool_t& pool) { return pool_new<ArgMax_cuda>    (pool); }
operator_t* resolver_cuda_op_ArgMin    (int opset, pool_t& pool) { return pool_new<ArgMin_cuda>    (pool); }
operator_t* resolver_cuda_op_LogSoftmax(int opset, pool_t& pool) { return pool_new<LogSoftmax_cuda>(pool); }

} // namespace nnr

#endif // NNR_USE_CUDA
