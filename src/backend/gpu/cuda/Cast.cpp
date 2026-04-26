#if defined(NNR_USE_CUDA)

// Cast + Where via NVRTC.
// Cast: convert between dtypes (limited set; f32↔i32↔i64, f32↔f16 not covered).
// Where: select(condition, x, y) — same-shape f32 only.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "attr_key.h"

namespace nnr {

operator_t* resolver_default_op_Cast (int opset, pool_t& pool);
operator_t* resolver_default_op_Where(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* cast_source() {
    return R"CUDA(
extern "C" {

// f32 <-> i32
__global__ void cast_f32_to_i32(const float* __restrict__ x, int* __restrict__ y, unsigned long long n) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = (int)x[i];
}
__global__ void cast_i32_to_f32(const int* __restrict__ x, float* __restrict__ y, unsigned long long n) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = (float)x[i];
}
// f32 <-> i64
__global__ void cast_f32_to_i64(const float* __restrict__ x, long long* __restrict__ y, unsigned long long n) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = (long long)x[i];
}
__global__ void cast_i64_to_f32(const long long* __restrict__ x, float* __restrict__ y, unsigned long long n) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = (float)x[i];
}
// i32 <-> i64
__global__ void cast_i32_to_i64(const int* __restrict__ x, long long* __restrict__ y, unsigned long long n) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = (long long)x[i];
}
__global__ void cast_i64_to_i32(const long long* __restrict__ x, int* __restrict__ y, unsigned long long n) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = (int)x[i];
}

// Where (same-shape f32 path): cond is int8/bool-as-byte.
__global__ void where_f32(const unsigned char* __restrict__ cond,
                          const float* __restrict__ x,
                          const float* __restrict__ y,
                          float* __restrict__ out,
                          unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) out[i] = cond[i] ? x[i] : y[i];
}

} // extern "C"
)CUDA";
}

static CUfunction get_cast_kernel(gpu::cuda_backend_t* be, const char* name) {
    return be->nvrtc.get("nnr_cast_f32", cast_source(), name,
                         gpu::nvrtc_arch_option(be->device));
}

static bool launch_1d(gpu::cuda_backend_t* be, CUfunction f, void** args, unsigned long long n) {
    constexpr unsigned BLK = 256;
    unsigned grid = (unsigned)((n + BLK - 1) / BLK);
    return gpu::nvrtc_launch(be->device, f, grid, 1, 1, BLK, 1, 1, args);
}

// -------- Cast --------

struct Cast_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    const char* kernel_name = nullptr;
    int   identity_elem_size = 0;  // != 0 → input/output type identical, do d2d copy

    bool init() override {
        if (!(inputs.size() == 1 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_Cast(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        identity_elem_size = 0;
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        if (x->ndata != y->ndata) return true;

        auto it  = x->type;
        auto ot  = y->type;
        kernel_name = nullptr;
        // Identity cast (in.type == out.type) → device-to-device memcpy.
        // Common in shape-arithmetic chains (e.g. int64 → int64).
        if (it == ot) {
            switch (it) {
                case NNR_DATA_TYPE_FLOAT32: case NNR_DATA_TYPE_INT32: case NNR_DATA_TYPE_UINT32:
                    identity_elem_size = 4; break;
                case NNR_DATA_TYPE_INT64: case NNR_DATA_TYPE_UINT64: case NNR_DATA_TYPE_FLOAT64:
                    identity_elem_size = 8; break;
                case NNR_DATA_TYPE_INT16: case NNR_DATA_TYPE_UINT16: case NNR_DATA_TYPE_FLOAT16:
                    identity_elem_size = 2; break;
                case NNR_DATA_TYPE_INT8:  case NNR_DATA_TYPE_UINT8:  case NNR_DATA_TYPE_BOOL:
                    identity_elem_size = 1; break;
                default: return true;
            }
        }
        else if (it == NNR_DATA_TYPE_FLOAT32 && ot == NNR_DATA_TYPE_INT32) kernel_name = "cast_f32_to_i32";
        else if (it == NNR_DATA_TYPE_INT32 && ot == NNR_DATA_TYPE_FLOAT32) kernel_name = "cast_i32_to_f32";
        else if (it == NNR_DATA_TYPE_FLOAT32 && ot == NNR_DATA_TYPE_INT64) kernel_name = "cast_f32_to_i64";
        else if (it == NNR_DATA_TYPE_INT64 && ot == NNR_DATA_TYPE_FLOAT32) kernel_name = "cast_i64_to_f32";
        else if (it == NNR_DATA_TYPE_INT32 && ot == NNR_DATA_TYPE_INT64)   kernel_name = "cast_i32_to_i64";
        else if (it == NNR_DATA_TYPE_INT64 && ot == NNR_DATA_TYPE_INT32)   kernel_name = "cast_i64_to_i32";
        else return true;

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }

        void* d_x = be->cache->ensure_device(inputs[0]);
        void* d_y = be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        unsigned long long n = outputs[0]->ndata;
        if (identity_elem_size > 0) {
            // Identity cast: device-to-device memcpy on the compute stream
            // (captured into the graph during replay setup).
            cudaError_t err = cudaMemcpyAsync(d_y, d_x,
                                              n * (size_t)identity_elem_size,
                                              cudaMemcpyDeviceToDevice,
                                              be->device->compute_stream());
            if (err != cudaSuccess) return fallback->exec();
            be->cache->mark_written(outputs[0]);
            return true;
        }

        CUfunction f = get_cast_kernel(be, kernel_name);
        if (!f) { return fallback->exec(); }
        void* args[] = { &d_x, &d_y, &n };
        if (!launch_1d(be, f, args, n)) { return fallback->exec(); }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

// -------- Where --------

struct Where_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;

    bool init() override {
        if (!(inputs.size() == 3 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_Where(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* c = inputs[0];
        const tensor_t* x = inputs[1];
        const tensor_t* y = inputs[2];
        tensor_t* o = outputs[0];
        if (c->type != NNR_DATA_TYPE_BOOL) return true;
        if (x->type != NNR_DATA_TYPE_FLOAT32 || y->type != NNR_DATA_TYPE_FLOAT32) return true;
        // Same-shape only (no broadcasting in CUDA path).
        if (c->ndata != x->ndata || x->ndata != y->ndata || o->ndata != x->ndata) return true;

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        CUfunction f = get_cast_kernel(be, "where_f32");
        if (!f) { return fallback->exec(); }

        unsigned char* d_c = (unsigned char*)be->cache->ensure_device(inputs[0]);
        float*         d_x = (float*)        be->cache->ensure_device(inputs[1]);
        float*         d_y = (float*)        be->cache->ensure_device(inputs[2]);
        float*         d_o = (float*)        be->cache->alloc_output (outputs[0]);
        if (!d_c || !d_x || !d_y || !d_o) { return fallback->exec(); }

        unsigned long long n = outputs[0]->ndata;
        void* args[] = { &d_c, &d_x, &d_y, &d_o, &n };
        if (!launch_1d(be, f, args, n)) { return fallback->exec(); }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_Cast (int opset, pool_t& pool) { return pool_new<Cast_cuda> (pool); }
operator_t* resolver_cuda_op_Where(int opset, pool_t& pool) { return pool_new<Where_cuda>(pool); }

} // namespace nnr

#endif // NNR_USE_CUDA
