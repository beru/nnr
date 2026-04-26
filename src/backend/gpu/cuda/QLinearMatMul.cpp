#if defined(NNR_USE_CUDA)

// QLinearMatMul on CUDA — per-tensor quantized GEMM, naive one-output-per-thread.
//
// Primary target is vgg16-12-int8's 3 dense layers (25088→4096, 4096→4096,
// 4096→1000) where M=1 reduces the GEMM to a GEMV — WMMA doesn't help a
// 1-row output, and the bandwidth/compute is small enough that a simple
// scalar accumulate kernel is plenty fast and unblocks CUDA Graph replay.
//
// Supports all 8 A/B sign combinations via a shift-to-signed unification
// (A' = A − a_shift, a_zp_eff = a_zp − a_shift), identical to the trick
// used in QLinearConv.cpp. Output can be uint8 or int8. Scales/zps must be
// per-tensor scalars; broadcast / per-channel / ndim > 2 cases fall back
// to CPU.
//
// Batch (ndim > 2) also falls back — vgg16's dense layers are 2D.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"

#include <cstdint>

namespace nnr {

operator_t* resolver_default_op_QLinearMatMul(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* qmatmul_source() {
    return R"CUDA(
extern "C" {

// One output element per thread. AT/BT are input byte types (signed char
// or unsigned char). The host computes a_zp_eff / b_zp_eff as the
// already-shifted zero points; the kernel applies the shift inline so
// uint8/int8 mixes compute in one math path.
// Original 2D kernel — fast path for the M×N×K case (e.g. VGG dense layers).
#define QMATMUL_BODY(AT, BT, YT, Y_MIN, Y_MAX)                               \
    int m = blockIdx.y * blockDim.y + threadIdx.y;                           \
    int n = blockIdx.x * blockDim.x + threadIdx.x;                           \
    if (m >= M || n >= N) return;                                            \
    int acc = 0;                                                             \
    for (int kk = 0; kk < K; ++kk) {                                         \
        int av = (int)A[m * K + kk] - a_shift;                               \
        int bv = (int)B[kk * N + n] - b_shift;                               \
        acc += (av - a_zp_eff) * (bv - b_zp_eff);                            \
    }                                                                        \
    float f = (float)acc * combined_scale + (float)y_zp;                     \
    int q = (int)rintf(f);                                                   \
    if (q < (Y_MIN)) q = (Y_MIN);                                            \
    if (q > (Y_MAX)) q = (Y_MAX);                                            \
    Y[m * N + n] = (YT)q;

// Batched kernel — same body but with per-batch offsets via blockIdx.z.
#define QMATMUL_BATCH_BODY(AT, BT, YT, Y_MIN, Y_MAX)                         \
    int m = blockIdx.y * blockDim.y + threadIdx.y;                           \
    int n = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int b = blockIdx.z;                                                      \
    if (m >= M || n >= N) return;                                            \
    long long a_off = (long long)b * (long long)M * (long long)K;            \
    long long b_off = (long long)b * (long long)K * (long long)N;            \
    long long y_off = (long long)b * (long long)M * (long long)N;            \
    int acc = 0;                                                             \
    for (int kk = 0; kk < K; ++kk) {                                         \
        int av = (int)A[a_off + m * K + kk] - a_shift;                       \
        int bv = (int)B[b_off + kk * N + n] - b_shift;                       \
        acc += (av - a_zp_eff) * (bv - b_zp_eff);                            \
    }                                                                        \
    float f = (float)acc * combined_scale + (float)y_zp;                     \
    int q = (int)rintf(f);                                                   \
    if (q < (Y_MIN)) q = (Y_MIN);                                            \
    if (q > (Y_MAX)) q = (Y_MAX);                                            \
    Y[y_off + m * N + n] = (YT)q;

__global__ void qmatmul_uuu(const unsigned char* __restrict__ A,
                            const unsigned char* __restrict__ B,
                            unsigned char* __restrict__ Y,
                            int M, int N, int K,
                            int a_shift, int b_shift,
                            int a_zp_eff, int b_zp_eff,
                            float combined_scale, int y_zp)
{ QMATMUL_BODY(unsigned char, unsigned char, unsigned char, 0, 255) }

__global__ void qmatmul_usu(const unsigned char* __restrict__ A,
                            const signed char*   __restrict__ B,
                            unsigned char* __restrict__ Y,
                            int M, int N, int K,
                            int a_shift, int b_shift,
                            int a_zp_eff, int b_zp_eff,
                            float combined_scale, int y_zp)
{ QMATMUL_BODY(unsigned char, signed char, unsigned char, 0, 255) }

__global__ void qmatmul_suu(const signed char*   __restrict__ A,
                            const unsigned char* __restrict__ B,
                            unsigned char* __restrict__ Y,
                            int M, int N, int K,
                            int a_shift, int b_shift,
                            int a_zp_eff, int b_zp_eff,
                            float combined_scale, int y_zp)
{ QMATMUL_BODY(signed char, unsigned char, unsigned char, 0, 255) }

__global__ void qmatmul_ssu(const signed char*   __restrict__ A,
                            const signed char*   __restrict__ B,
                            unsigned char* __restrict__ Y,
                            int M, int N, int K,
                            int a_shift, int b_shift,
                            int a_zp_eff, int b_zp_eff,
                            float combined_scale, int y_zp)
{ QMATMUL_BODY(signed char, signed char, unsigned char, 0, 255) }

__global__ void qmatmul_uus(const unsigned char* __restrict__ A,
                            const unsigned char* __restrict__ B,
                            signed char*   __restrict__ Y,
                            int M, int N, int K,
                            int a_shift, int b_shift,
                            int a_zp_eff, int b_zp_eff,
                            float combined_scale, int y_zp)
{ QMATMUL_BODY(unsigned char, unsigned char, signed char, -128, 127) }

__global__ void qmatmul_uss(const unsigned char* __restrict__ A,
                            const signed char*   __restrict__ B,
                            signed char*   __restrict__ Y,
                            int M, int N, int K,
                            int a_shift, int b_shift,
                            int a_zp_eff, int b_zp_eff,
                            float combined_scale, int y_zp)
{ QMATMUL_BODY(unsigned char, signed char, signed char, -128, 127) }

__global__ void qmatmul_sus(const signed char*   __restrict__ A,
                            const unsigned char* __restrict__ B,
                            signed char*   __restrict__ Y,
                            int M, int N, int K,
                            int a_shift, int b_shift,
                            int a_zp_eff, int b_zp_eff,
                            float combined_scale, int y_zp)
{ QMATMUL_BODY(signed char, unsigned char, signed char, -128, 127) }

__global__ void qmatmul_sss(const signed char*   __restrict__ A,
                            const signed char*   __restrict__ B,
                            signed char*   __restrict__ Y,
                            int M, int N, int K,
                            int a_shift, int b_shift,
                            int a_zp_eff, int b_zp_eff,
                            float combined_scale, int y_zp)
{ QMATMUL_BODY(signed char, signed char, signed char, -128, 127) }
#undef QMATMUL_BODY

// Batched variants (grid.z = batch). Per-batch offset in long-long.
__global__ void qmatmul_b_uuu(const unsigned char* A, const unsigned char* B, unsigned char* Y,
                              int M, int N, int K,
                              int a_shift, int b_shift, int a_zp_eff, int b_zp_eff,
                              float combined_scale, int y_zp)
{ QMATMUL_BATCH_BODY(unsigned char, unsigned char, unsigned char, 0, 255) }
__global__ void qmatmul_b_usu(const unsigned char* A, const signed char* B, unsigned char* Y,
                              int M, int N, int K,
                              int a_shift, int b_shift, int a_zp_eff, int b_zp_eff,
                              float combined_scale, int y_zp)
{ QMATMUL_BATCH_BODY(unsigned char, signed char, unsigned char, 0, 255) }
__global__ void qmatmul_b_suu(const signed char* A, const unsigned char* B, unsigned char* Y,
                              int M, int N, int K,
                              int a_shift, int b_shift, int a_zp_eff, int b_zp_eff,
                              float combined_scale, int y_zp)
{ QMATMUL_BATCH_BODY(signed char, unsigned char, unsigned char, 0, 255) }
__global__ void qmatmul_b_ssu(const signed char* A, const signed char* B, unsigned char* Y,
                              int M, int N, int K,
                              int a_shift, int b_shift, int a_zp_eff, int b_zp_eff,
                              float combined_scale, int y_zp)
{ QMATMUL_BATCH_BODY(signed char, signed char, unsigned char, 0, 255) }
__global__ void qmatmul_b_uus(const unsigned char* A, const unsigned char* B, signed char* Y,
                              int M, int N, int K,
                              int a_shift, int b_shift, int a_zp_eff, int b_zp_eff,
                              float combined_scale, int y_zp)
{ QMATMUL_BATCH_BODY(unsigned char, unsigned char, signed char, -128, 127) }
__global__ void qmatmul_b_uss(const unsigned char* A, const signed char* B, signed char* Y,
                              int M, int N, int K,
                              int a_shift, int b_shift, int a_zp_eff, int b_zp_eff,
                              float combined_scale, int y_zp)
{ QMATMUL_BATCH_BODY(unsigned char, signed char, signed char, -128, 127) }
__global__ void qmatmul_b_sus(const signed char* A, const unsigned char* B, signed char* Y,
                              int M, int N, int K,
                              int a_shift, int b_shift, int a_zp_eff, int b_zp_eff,
                              float combined_scale, int y_zp)
{ QMATMUL_BATCH_BODY(signed char, unsigned char, signed char, -128, 127) }
__global__ void qmatmul_b_sss(const signed char* A, const signed char* B, signed char* Y,
                              int M, int N, int K,
                              int a_shift, int b_shift, int a_zp_eff, int b_zp_eff,
                              float combined_scale, int y_zp)
{ QMATMUL_BATCH_BODY(signed char, signed char, signed char, -128, 127) }
#undef QMATMUL_BODY
#undef QMATMUL_BATCH_BODY

} // extern "C"
)CUDA";
}

static inline int type_char(data_type_t t) {
    return (t == NNR_DATA_TYPE_UINT8) ? 'u' : 's';
}

struct QLinearMatMul_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;

    int M = 0, N = 0, K = 0, batch = 1;
    int a_zp = 0, b_zp = 0, y_zp = 0;
    float combined_scale = 0.f;
    char kn_buf[16] = {};   // "qmatmul_uuu" etc.

    bool init() override {
        if (!(inputs.size() == 8 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_QLinearMatMul(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;

        const tensor_t* A = inputs[0];
        const tensor_t* B = inputs[3];
        tensor_t* Y = outputs[0];

        // ndim >= 2; for ndim > 2 require matching leading dims on A, B, Y
        // (no broadcast). Common batched-MatMul shape `[B, M, K] × [B, K, N]`.
        if (A->ndim < 2 || A->ndim != B->ndim || A->ndim != Y->ndim) return true;
        if ((A->type != NNR_DATA_TYPE_UINT8 && A->type != NNR_DATA_TYPE_INT8) ||
            (B->type != NNR_DATA_TYPE_UINT8 && B->type != NNR_DATA_TYPE_INT8) ||
            (Y->type != NNR_DATA_TYPE_UINT8 && Y->type != NNR_DATA_TYPE_INT8))
            return true;

        const int nd = A->ndim;
        batch = 1;
        for (int d = 0; d < nd - 2; ++d) {
            if (A->dims[d] != B->dims[d] || A->dims[d] != Y->dims[d]) return true;
            batch *= A->dims[d];
        }
        M = A->dims[nd - 2];
        K = A->dims[nd - 1];
        if (B->dims[nd - 2] != K) return true;
        N = B->dims[nd - 1];
        if (Y->dims[nd - 2] != M || Y->dims[nd - 1] != N) return true;

        // Scales / zero-points must be per-tensor scalars.
        auto is_scalar_f32 = [](const tensor_t* t) {
            return t && t->type == NNR_DATA_TYPE_FLOAT32 && t->ndata == 1 && t->data;
        };
        auto is_scalar_u8_or_s8 = [](const tensor_t* t) {
            return t && (t->type == NNR_DATA_TYPE_UINT8 || t->type == NNR_DATA_TYPE_INT8)
                     && t->ndata == 1 && t->data;
        };
        if (!is_scalar_f32(inputs[1]) || !is_scalar_f32(inputs[4]) || !is_scalar_f32(inputs[6]))
            return true;
        if (!is_scalar_u8_or_s8(inputs[2]) || !is_scalar_u8_or_s8(inputs[5]) || !is_scalar_u8_or_s8(inputs[7]))
            return true;

        float a_scale = *(const float*)inputs[1]->data;
        float b_scale = *(const float*)inputs[4]->data;
        float y_scale = *(const float*)inputs[6]->data;
        if (y_scale == 0.f) return true;

        auto read_zp = [](const tensor_t* t) -> int {
            if (t->type == NNR_DATA_TYPE_UINT8) return (int)*(const uint8_t*)t->data;
            return (int)*(const int8_t*)t->data;
        };
        a_zp = read_zp(inputs[2]);
        b_zp = read_zp(inputs[5]);
        y_zp = read_zp(inputs[7]);
        combined_scale = a_scale * b_scale / y_scale;

        // Build kernel name "qmatmul_<sigA><sigB><sigY>" for 2D, or
        // "qmatmul_b_<sigA><sigB><sigY>" for batched (ndim>2).
        int p = 0;
        kn_buf[p++] = 'q'; kn_buf[p++] = 'm'; kn_buf[p++] = 'a'; kn_buf[p++] = 't';
        kn_buf[p++] = 'm'; kn_buf[p++] = 'u'; kn_buf[p++] = 'l'; kn_buf[p++] = '_';
        if (batch > 1) { kn_buf[p++] = 'b'; kn_buf[p++] = '_'; }
        kn_buf[p++] = (char)type_char(A->type);
        kn_buf[p++] = (char)type_char(B->type);
        kn_buf[p++] = (char)type_char(Y->type);
        kn_buf[p] = 0;

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) return fallback->exec();

        CUfunction f = be->nvrtc.get("nnr_qmatmul", qmatmul_source(),
                                     kn_buf,
                                     gpu::nvrtc_arch_option(be->device));
        if (!f) return fallback->exec();

        void* d_a = be->cache->ensure_device(inputs[0]);
        void* d_b = be->cache->ensure_device(inputs[3]);
        void* d_y = be->cache->alloc_output(outputs[0]);
        if (!d_a || !d_b || !d_y) return fallback->exec();

        const int a_shift = (inputs[0]->type == NNR_DATA_TYPE_UINT8) ? 128 : 0;
        const int b_shift = (inputs[3]->type == NNR_DATA_TYPE_UINT8) ? 128 : 0;
        const int a_zp_eff = a_zp - a_shift;
        const int b_zp_eff = b_zp - b_shift;

        int _M = M, _N = N, _K = K;
        int _as = a_shift, _bs = b_shift, _az = a_zp_eff, _bz = b_zp_eff;
        float _cs = combined_scale;
        int _yz = y_zp;
        void* args[] = { &d_a, &d_b, &d_y, &_M, &_N, &_K,
                         &_as, &_bs, &_az, &_bz, &_cs, &_yz };

        unsigned block_x = 32, block_y = 8;
        unsigned grid_x  = (unsigned)((N + block_x - 1) / block_x);
        unsigned grid_y  = (unsigned)((M + block_y - 1) / block_y);
        unsigned grid_z  = (unsigned)batch;
        if (!gpu::nvrtc_launch(be->device, f, grid_x, grid_y, grid_z,
                               block_x, block_y, 1, args))
            return fallback->exec();

        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_QLinearMatMul(int opset, pool_t& pool) {
    return pool_new<QLinearMatMul_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
