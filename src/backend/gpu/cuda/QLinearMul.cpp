#if defined(NNR_USE_CUDA)

// QLinearMul on CUDA — element-wise quantized multiply for the SAME_SHAPE
// case. Formula:
//   C = sat(round(cs * (A − a_zp) * (B − b_zp) + y_zp))  where cs = a_scale * b_scale / y_scale.
// Broadcast cases (scalar / per-channel / general) fall back to CPU.
//
// Targets densenet-12-int8's QLinearMul ops on dense-block residual paths —
// same-shape int8 multiplies that would otherwise block CUDA Graph replay.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"

#include <cstdint>

namespace nnr {

operator_t* resolver_default_op_QLinearMul(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* qmul_source() {
    return R"CUDA(
extern "C" {

#define QMUL_BODY(ELT, Y_MIN, Y_MAX)                                           \
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;                  \
    if (i >= total) return;                                                    \
    float fa = (float)((int)A[i] - a_zp);                                      \
    float fb = (float)((int)B[i] - b_zp);                                      \
    float v = cs * fa * fb + y_zp_f;                                           \
    int q = (int)rintf(v);                                                     \
    if (q < (Y_MIN)) q = (Y_MIN);                                              \
    if (q > (Y_MAX)) q = (Y_MAX);                                              \
    Y[i] = (ELT)q;

__global__ void qmul_u8(const unsigned char* __restrict__ A,
                        const unsigned char* __restrict__ B,
                        unsigned char* __restrict__ Y,
                        size_t total, float cs, int a_zp, int b_zp, float y_zp_f)
{ QMUL_BODY(unsigned char, 0, 255) }

__global__ void qmul_s8(const signed char* __restrict__ A,
                        const signed char* __restrict__ B,
                        signed char* __restrict__ Y,
                        size_t total, float cs, int a_zp, int b_zp, float y_zp_f)
{ QMUL_BODY(signed char, -128, 127) }

// Per-channel broadcast: A is [..., C, inner], B is [C]. B[c] is reused
// across inner spatial positions. total = outer * C * inner.
#define QMUL_PC_BODY(ELT, Y_MIN, Y_MAX)                                        \
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;                  \
    if (i >= total) return;                                                    \
    int c = (int)((i / inner) % C);                                            \
    float fa = (float)((int)A[i] - a_zp);                                      \
    float fb = (float)((int)B[c] - b_zp);                                      \
    float v = cs * fa * fb + y_zp_f;                                           \
    int q = (int)rintf(v);                                                     \
    if (q < (Y_MIN)) q = (Y_MIN);                                              \
    if (q > (Y_MAX)) q = (Y_MAX);                                              \
    Y[i] = (ELT)q;

__global__ void qmul_pc_u8(const unsigned char* __restrict__ A,
                           const unsigned char* __restrict__ B,
                           unsigned char* __restrict__ Y,
                           size_t total, size_t inner, int C,
                           float cs, int a_zp, int b_zp, float y_zp_f)
{ QMUL_PC_BODY(unsigned char, 0, 255) }

__global__ void qmul_pc_s8(const signed char* __restrict__ A,
                           const signed char* __restrict__ B,
                           signed char* __restrict__ Y,
                           size_t total, size_t inner, int C,
                           float cs, int a_zp, int b_zp, float y_zp_f)
{ QMUL_PC_BODY(signed char, -128, 127) }
#undef QMUL_PC_BODY

// NHWC per-channel: A is [..., H, W, C], B is [C]. Channel index is i % C.
#define QMUL_PC_NHWC_BODY(ELT, Y_MIN, Y_MAX)                                   \
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;                  \
    if (i >= total) return;                                                    \
    int c = (int)(i % (size_t)C);                                              \
    float fa = (float)((int)A[i] - a_zp);                                      \
    float fb = (float)((int)B[c] - b_zp);                                      \
    float v = cs * fa * fb + y_zp_f;                                           \
    int q = (int)rintf(v);                                                     \
    if (q < (Y_MIN)) q = (Y_MIN);                                              \
    if (q > (Y_MAX)) q = (Y_MAX);                                              \
    Y[i] = (ELT)q;

__global__ void qmul_pc_nhwc_u8(const unsigned char* __restrict__ A,
                                const unsigned char* __restrict__ B,
                                unsigned char* __restrict__ Y,
                                size_t total, int C,
                                float cs, int a_zp, int b_zp, float y_zp_f)
{ QMUL_PC_NHWC_BODY(unsigned char, 0, 255) }

__global__ void qmul_pc_nhwc_s8(const signed char* __restrict__ A,
                                const signed char* __restrict__ B,
                                signed char* __restrict__ Y,
                                size_t total, int C,
                                float cs, int a_zp, int b_zp, float y_zp_f)
{ QMUL_PC_NHWC_BODY(signed char, -128, 127) }
#undef QMUL_BODY
#undef QMUL_PC_NHWC_BODY

} // extern "C"
)CUDA";
}

struct QLinearMul_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    bool is_uint8 = true;
    bool is_per_channel = false;
    size_t inner = 1;
    int C_channels = 0;
    float cs = 0.f;
    int a_zp = 0, b_zp = 0;
    float y_zp_f = 0.f;
    size_t total = 0;
    const char* kernel_name = nullptr;

    bool init() override {
        if (!(inputs.size() == 8 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_QLinearMul(opset, ctx->attr_pool);
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

        if (A->type != B->type || A->type != Y->type) return true;
        if (A->type != NNR_DATA_TYPE_UINT8 && A->type != NNR_DATA_TYPE_INT8) return true;
        if (A->ndata != Y->ndata) return true;   // Y must match A shape (no output broadcast)
        is_uint8 = (A->type == NNR_DATA_TYPE_UINT8);
        is_per_channel = false;
        if (A->ndata == B->ndata) {
            // SAME_SHAPE fast path.
        } else if (A->ndim >= 2 && (int64_t)B->ndata == A->dims[1]) {
            // Per-channel broadcast along dims[1] (NCHW channel axis).
            is_per_channel = true;
            C_channels = A->dims[1];
            inner = 1;
            for (int d = 2; d < A->ndim; ++d) inner *= (size_t)A->dims[d];
        } else {
            return true;   // general broadcast → CPU fallback
        }

        auto is_scalar_f32 = [](const tensor_t* t) {
            return t && t->type == NNR_DATA_TYPE_FLOAT32 && t->ndata == 1 && t->data;
        };
        if (!is_scalar_f32(inputs[1]) || !is_scalar_f32(inputs[4]) || !is_scalar_f32(inputs[6]))
            return true;

        float a_scale = *(const float*)inputs[1]->data;
        float b_scale = *(const float*)inputs[4]->data;
        float y_scale = *(const float*)inputs[6]->data;
        if (y_scale == 0.f) return true;

        auto read_zp = [](const tensor_t* t) -> int {
            if (!t || t->ndata == 0 || !t->data) return 0;
            if (t->type == NNR_DATA_TYPE_UINT8) return (int)*(const uint8_t*)t->data;
            if (t->type == NNR_DATA_TYPE_INT8)  return (int)*(const int8_t*)t->data;
            return INT32_MIN;
        };
        a_zp = read_zp(inputs[2]);
        b_zp = read_zp(inputs[5]);
        int y_zp = read_zp(inputs[7]);
        if (a_zp == INT32_MIN || b_zp == INT32_MIN || y_zp == INT32_MIN) return true;

        cs = a_scale * b_scale / y_scale;
        y_zp_f = (float)y_zp;
        total = Y->ndata;
        kernel_name = nullptr; // resolved at exec() based on layout

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
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
                ? (is_uint8 ? "qmul_pc_nhwc_u8" : "qmul_pc_nhwc_s8")
                : (is_uint8 ? "qmul_pc_u8"      : "qmul_pc_s8");
        } else {
            kname = is_uint8 ? "qmul_u8" : "qmul_s8";
        }
        CUfunction f = be->nvrtc.get("nnr_qmul", qmul_source(),
                                     kname, gpu::nvrtc_arch_option(be->device));
        if (!f) return fallback->exec();

        void* d_a = be->cache->ensure_device(inputs[0]);
        void* d_b = be->cache->ensure_device(inputs[3]);
        void* d_y = be->cache->alloc_output(outputs[0]);
        if (!d_a || !d_b || !d_y) return fallback->exec();

        size_t _t = total;
        float _cs = cs, _yz = y_zp_f;
        int _az = a_zp, _bz = b_zp;
        unsigned block = 256;
        unsigned grid  = (unsigned)((total + block - 1) / block);
        if (is_per_channel) {
            int _C = C_channels;
            if (is_nhwc) {
                void* args[] = { &d_a, &d_b, &d_y, &_t, &_C,
                                 &_cs, &_az, &_bz, &_yz };
                if (!gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args))
                    return fallback->exec();
            } else {
                size_t _in = inner;
                void* args[] = { &d_a, &d_b, &d_y, &_t, &_in, &_C,
                                 &_cs, &_az, &_bz, &_yz };
                if (!gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args))
                    return fallback->exec();
            }
        } else {
            void* args[] = { &d_a, &d_b, &d_y, &_t, &_cs, &_az, &_bz, &_yz };
            if (!gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args))
                return fallback->exec();
        }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_QLinearMul(int opset, pool_t& pool) {
    return pool_new<QLinearMul_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
