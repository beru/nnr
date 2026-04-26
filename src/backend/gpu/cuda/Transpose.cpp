#if defined(NNR_USE_CUDA)

// Generic N-D Transpose via NVRTC.
// Thread per output element; decode linear output index into multi-index,
// then reindex source using src_strides[perm[d]].
// Up to 8 dims (covers all common models).

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "attr_key.h"

namespace nnr {

operator_t* resolver_default_op_Transpose(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

// Bitwise transpose kernels parameterized by element size in bytes.
// Element types are punned to uintN_t — only the byte width matters for a copy.
// 1/2/4/8-byte variants cover every numeric dtype in NNR (int8/uint8/bool,
// fp16/bf16/int16, fp32/int32, fp64/int64).
static const char* transpose_source() {
    return R"CUDA(
#define TRANSPOSE_KERNEL(NAME, T)                                                \
extern "C" __global__                                                            \
void NAME(const T* __restrict__ x,                                               \
          T* __restrict__ y,                                                     \
          unsigned long long n,                                                  \
          int ndim,                                                              \
          int out_dims_0, int out_dims_1, int out_dims_2, int out_dims_3,        \
          int out_dims_4, int out_dims_5, int out_dims_6, int out_dims_7,        \
          int sstr_0, int sstr_1, int sstr_2, int sstr_3,                        \
          int sstr_4, int sstr_5, int sstr_6, int sstr_7)                        \
{                                                                                \
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; \
    if (i >= n) return;                                                          \
    int out_dims[8] = { out_dims_0, out_dims_1, out_dims_2, out_dims_3,          \
                        out_dims_4, out_dims_5, out_dims_6, out_dims_7 };        \
    int sstr[8]     = { sstr_0, sstr_1, sstr_2, sstr_3,                          \
                        sstr_4, sstr_5, sstr_6, sstr_7 };                        \
    unsigned long long rem = i;                                                  \
    unsigned long long src_off = 0;                                              \
    for (int d = ndim - 1; d >= 0; --d) {                                        \
        unsigned long long idx = rem % (unsigned long long)out_dims[d];          \
        rem /= (unsigned long long)out_dims[d];                                  \
        src_off += idx * (unsigned long long)sstr[d];                            \
    }                                                                            \
    y[i] = x[src_off];                                                           \
}

typedef unsigned char      u8_t;
typedef unsigned short     u16_t;
typedef unsigned int       u32_t;
typedef unsigned long long u64_t;

TRANSPOSE_KERNEL(transpose_b1, u8_t)
TRANSPOSE_KERNEL(transpose_b2, u16_t)
TRANSPOSE_KERNEL(transpose_b4, u32_t)
TRANSPOSE_KERNEL(transpose_b8, u64_t)
)CUDA";
}

struct Transpose_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;

    int ndim = 0;
    int elem_size = 0;  // 1, 2, 4, or 8 — picks the kernel variant
    int out_dims[8] = {1,1,1,1,1,1,1,1};
    int sstr_by_out[8] = {0,0,0,0,0,0,0,0};

    bool init() override {
        if (!(inputs.size() == 1 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_Transpose(opset, ctx->attr_pool);
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
        const tensor_t* x = inputs[0];
        const tensor_t* y = outputs[0];

        prim_valid = false;
        device_tag = 0;
        if (x->ndim < 1 || x->ndim > 8) return true;
        if (y->ndim != x->ndim) return true;
        size_t esz = data_type_sizeof(x);
        if (esz != 1 && esz != 2 && esz != 4 && esz != 8) return true;
        elem_size = (int)esz;

        int64_t* ints = nullptr;
        int n = attribute(attr_key_t::perm, ints);
        int perm_buf[8];
        if (n == x->ndim) {
            for (int i = 0; i < x->ndim; ++i) perm_buf[i] = (int)ints[i];
        } else {
            // Default: reverse dims.
            for (int i = 0; i < x->ndim; ++i) perm_buf[i] = x->ndim - i - 1;
        }

        // Compute row-major strides for source (x).
        int src_stride[8];
        src_stride[x->ndim - 1] = 1;
        for (int i = x->ndim - 2; i >= 0; --i) {
            src_stride[i] = src_stride[i + 1] * x->dims[i + 1];
        }

        ndim = x->ndim;
        for (int d = 0; d < ndim; ++d) {
            out_dims[d]    = y->dims[d];
            sstr_by_out[d] = src_stride[perm_buf[d]];
        }
        for (int d = ndim; d < 8; ++d) { out_dims[d] = 1; sstr_by_out[d] = 0; }

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }

        const char* kern_name = nullptr;
        const char* cache_key = nullptr;
        switch (elem_size) {
            case 1: kern_name = "transpose_b1"; cache_key = "nnr_transpose_b1"; break;
            case 2: kern_name = "transpose_b2"; cache_key = "nnr_transpose_b2"; break;
            case 4: kern_name = "transpose_b4"; cache_key = "nnr_transpose_b4"; break;
            case 8: kern_name = "transpose_b8"; cache_key = "nnr_transpose_b8"; break;
            default: return fallback->exec();
        }
        CUfunction f = be->nvrtc.get(cache_key,
                                     transpose_source(),
                                     kern_name,
                                     gpu::nvrtc_arch_option(be->device));
        if (!f) { return fallback->exec(); }

        void* d_x = be->cache->ensure_device(inputs[0]);
        void* d_y = be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        unsigned long long n = outputs[0]->ndata;
        void* args[] = {
            &d_x, &d_y, &n, &ndim,
            &out_dims[0], &out_dims[1], &out_dims[2], &out_dims[3],
            &out_dims[4], &out_dims[5], &out_dims[6], &out_dims[7],
            &sstr_by_out[0], &sstr_by_out[1], &sstr_by_out[2], &sstr_by_out[3],
            &sstr_by_out[4], &sstr_by_out[5], &sstr_by_out[6], &sstr_by_out[7],
        };
        unsigned block = 256;
        unsigned grid = (unsigned)((n + block - 1) / block);
        if (!gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args)) {
            return fallback->exec();
        }

        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_Transpose(int opset, pool_t& pool) {
    return pool_new<Transpose_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
