#if defined(NNR_USE_CUDA)

// QLinearAdd on CUDA — element-wise quantized add for the SAME_SHAPE case.
// Formula (matches ORT / CPU): C = sat(A*sa + B*sb + fixed) where
//   sa = a_scale / y_scale, sb = b_scale / y_scale,
//   fixed = y_zp - a_zp*sa - b_zp*sb.
//
// Broadcasted cases (scalar, per-channel) fall back to CPU — they're rare on
// the residual-heavy int8 models (ssd-12-int8, vgg16-int8) where nearly every
// QLinearAdd is SAME_SHAPE. Keeping this op on-device is what unlocks CUDA
// Graph replay for full int8 model inference.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"

#include <cstdint>

namespace nnr {

operator_t* resolver_default_op_QLinearAdd(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* qadd_source() {
    return R"CUDA(
extern "C" {

#define QADD_BODY(ELT, Y_MIN, Y_MAX)                                           \
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;                  \
    if (i >= total) return;                                                    \
    float v = (float)A[i] * sa + (float)B[i] * sb + fixed;                     \
    int q = (int)rintf(v);                                                     \
    if (q < (Y_MIN)) q = (Y_MIN);                                              \
    if (q > (Y_MAX)) q = (Y_MAX);                                              \
    Y[i] = (ELT)q;

__global__ void qadd_u8(const unsigned char* __restrict__ A,
                        const unsigned char* __restrict__ B,
                        unsigned char* __restrict__ Y,
                        size_t total, float sa, float sb, float fixed)
{
    QADD_BODY(unsigned char, 0, 255)
}

__global__ void qadd_s8(const signed char* __restrict__ A,
                        const signed char* __restrict__ B,
                        signed char* __restrict__ Y,
                        size_t total, float sa, float sb, float fixed)
{
    QADD_BODY(signed char, -128, 127)
}

// Per-channel broadcast: A is [..., C, inner], B is [C]. B[c] is reused
// across inner spatial positions.
#define QADD_PC_BODY(ELT, Y_MIN, Y_MAX)                                        \
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;                  \
    if (i >= total) return;                                                    \
    int c = (int)((i / inner) % C);                                            \
    float v = (float)A[i] * sa + (float)B[c] * sb + fixed;                     \
    int q = (int)rintf(v);                                                     \
    if (q < (Y_MIN)) q = (Y_MIN);                                              \
    if (q > (Y_MAX)) q = (Y_MAX);                                              \
    Y[i] = (ELT)q;

__global__ void qadd_pc_u8(const unsigned char* __restrict__ A,
                           const unsigned char* __restrict__ B,
                           unsigned char* __restrict__ Y,
                           size_t total, size_t inner, int C,
                           float sa, float sb, float fixed)
{ QADD_PC_BODY(unsigned char, 0, 255) }

__global__ void qadd_pc_s8(const signed char* __restrict__ A,
                           const signed char* __restrict__ B,
                           signed char* __restrict__ Y,
                           size_t total, size_t inner, int C,
                           float sa, float sb, float fixed)
{ QADD_PC_BODY(signed char, -128, 127) }
#undef QADD_PC_BODY

// NHWC per-channel: A is [..., H, W, C], B is [C]. Channel index is i % C.
#define QADD_PC_NHWC_BODY(ELT, Y_MIN, Y_MAX)                                   \
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;                  \
    if (i >= total) return;                                                    \
    int c = (int)(i % (size_t)C);                                              \
    float v = (float)A[i] * sa + (float)B[c] * sb + fixed;                     \
    int q = (int)rintf(v);                                                     \
    if (q < (Y_MIN)) q = (Y_MIN);                                              \
    if (q > (Y_MAX)) q = (Y_MAX);                                              \
    Y[i] = (ELT)q;

__global__ void qadd_pc_nhwc_u8(const unsigned char* __restrict__ A,
                                const unsigned char* __restrict__ B,
                                unsigned char* __restrict__ Y,
                                size_t total, int C,
                                float sa, float sb, float fixed)
{ QADD_PC_NHWC_BODY(unsigned char, 0, 255) }

__global__ void qadd_pc_nhwc_s8(const signed char* __restrict__ A,
                                const signed char* __restrict__ B,
                                signed char* __restrict__ Y,
                                size_t total, int C,
                                float sa, float sb, float fixed)
{ QADD_PC_NHWC_BODY(signed char, -128, 127) }
#undef QADD_BODY
#undef QADD_PC_NHWC_BODY

} // extern "C"
)CUDA";
}

struct QLinearAdd_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    bool is_uint8 = true;
    bool is_per_channel = false;
    size_t inner = 1;
    int C_channels = 0;
    float sa = 0.f, sb = 0.f, fixed = 0.f;
    size_t total = 0;
    const char* kernel_name = nullptr;

    bool init() override {
        if (!(inputs.size() == 8 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_QLinearAdd(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        layout_mask = fallback->layout_mask;
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;

        const tensor_t* A = inputs[0];
        const tensor_t* B = inputs[3];
        tensor_t* Y = outputs[0];

        // Fast path: equal-shape (SAME_SHAPE) or per-channel broadcast
        // (B is the channel dim of A, matching dims[1]). Other broadcasts
        // fall back to CPU.
        if (A->type != B->type || A->type != Y->type) return true;
        if (A->type != NNR_DATA_TYPE_UINT8 && A->type != NNR_DATA_TYPE_INT8) return true;
        if (A->ndata != Y->ndata) return true;
        is_uint8 = (A->type == NNR_DATA_TYPE_UINT8);
        is_per_channel = false;
        if (A->ndata == B->ndata) {
            // SAME_SHAPE.
        } else if (A->ndim >= 2 && (int64_t)B->ndata == A->dims[1]) {
            is_per_channel = true;
            C_channels = A->dims[1];
            inner = 1;
            for (int d = 2; d < A->ndim; ++d) inner *= (size_t)A->dims[d];
        } else {
            return true;
        }

        const tensor_t* as_t = inputs[1];
        const tensor_t* bs_t = inputs[4];
        const tensor_t* ys_t = inputs[6];
        if (as_t->type != NNR_DATA_TYPE_FLOAT32 || as_t->ndata != 1 || !as_t->data) return true;
        if (bs_t->type != NNR_DATA_TYPE_FLOAT32 || bs_t->ndata != 1 || !bs_t->data) return true;
        if (ys_t->type != NNR_DATA_TYPE_FLOAT32 || ys_t->ndata != 1 || !ys_t->data) return true;
        float a_scale = *(const float*)as_t->data;
        float b_scale = *(const float*)bs_t->data;
        float y_scale = *(const float*)ys_t->data;
        if (y_scale == 0.f) return true;

        auto read_zp = [](const tensor_t* t) -> int {
            if (!t || t->ndata == 0 || !t->data) return 0;
            if (t->type == NNR_DATA_TYPE_UINT8) return (int)*(const uint8_t*)t->data;
            if (t->type == NNR_DATA_TYPE_INT8)  return (int)*(const int8_t*)t->data;
            return INT32_MIN;
        };
        int a_zp = read_zp(inputs[2]);
        int b_zp = read_zp(inputs[5]);
        int y_zp = read_zp(inputs[7]);
        if (a_zp == INT32_MIN || b_zp == INT32_MIN || y_zp == INT32_MIN) return true;

        sa = a_scale / y_scale;
        sb = b_scale / y_scale;
        fixed = (float)y_zp - (sa * (float)a_zp + sb * (float)b_zp);
        total = Y->ndata;
        // Kernel selected at exec() based on Y->format. Keep three names alive.
        kernel_name = nullptr; // resolved in exec() based on layout

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        // SAME_SHAPE elementwise is layout-agnostic (just bytes through bytes).
        // Per-channel needs the NHWC kernel when format is NHWC; both are
        // available so we can opt in to LAYOUT_NHWC unconditionally here.
        layout_mask = LAYOUT_NCHW | LAYOUT_NHWC;
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) return fallback->exec();

        bool is_nhwc = (outputs[0]->format == memory_layout_t::NHWC);
        const char* kname;
        if (is_per_channel) {
            kname = is_nhwc
                ? (is_uint8 ? "qadd_pc_nhwc_u8" : "qadd_pc_nhwc_s8")
                : (is_uint8 ? "qadd_pc_u8"      : "qadd_pc_s8");
        } else {
            // SAME_SHAPE elementwise: same kernel for both layouts (no
            // dim-aware indexing, just byte-by-byte).
            kname = is_uint8 ? "qadd_u8" : "qadd_s8";
        }
        CUfunction f = be->nvrtc.get("nnr_qadd", qadd_source(),
                                     kname,
                                     gpu::nvrtc_arch_option(be->device));
        if (!f) return fallback->exec();

        void* d_a = be->cache->ensure_device(inputs[0]);
        void* d_b = be->cache->ensure_device(inputs[3]);
        void* d_y = be->cache->alloc_output(outputs[0]);
        if (!d_a || !d_b || !d_y) return fallback->exec();

        size_t _t = total;
        float _sa = sa, _sb = sb, _f = fixed;
        unsigned block = 256;
        unsigned grid  = (unsigned)((total + block - 1) / block);
        if (is_per_channel) {
            int _C = C_channels;
            if (is_nhwc) {
                void* args[] = { &d_a, &d_b, &d_y, &_t, &_C, &_sa, &_sb, &_f };
                if (!gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args))
                    return fallback->exec();
            } else {
                size_t _in = inner;
                void* args[] = { &d_a, &d_b, &d_y, &_t, &_in, &_C, &_sa, &_sb, &_f };
                if (!gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args))
                    return fallback->exec();
            }
        } else {
            void* args[] = { &d_a, &d_b, &d_y, &_t, &_sa, &_sb, &_f };
            if (!gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args))
                return fallback->exec();
        }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_QLinearAdd(int opset, pool_t& pool) {
    return pool_new<QLinearAdd_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
