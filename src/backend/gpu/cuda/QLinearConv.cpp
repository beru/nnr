#if defined(NNR_USE_CUDA)

// QLinearConv on CUDA — int8/uint8 convolution via im2col + WMMA int8 GEMM.
// Supports per-tensor OR per-output-channel w_scale / w_zp (ssd-12-int8 and
// densenet-int8 both use per-channel weight quantization).
//
// Mixed-sign unification: WMMA only supports same-sign s8×s8 or u8×u8, but
// ORT's QDQ models mix uint8 activations with int8 weights. We unify by
// shifting any uint8 input by −128 to signed int8; the zero-points become
// x_zp_eff = x_zp − x_shift and w_zp_eff[oc] = w_zp[oc] − w_shift.
// Math identity: sum((X−x_zp)(W−w_zp)) = sum((X'−x_zp_eff)(W'−w_zp_eff)).
//
// Pipeline per batch:
//   1. im2col_qs  : X → col[K, M] int8, shift+pad-with-x_zp_eff on OOB.
//   2. col_sum_s8 : col_sum[pixel] = sum_k col[k, pixel].
//   3. s8s32 WMMA : Acc[Cout, M] = W' · col  (int32 accum).
//   4. requant    : Y = sat((Acc − w_zp_eff[oc]·col_sum − x_zp_eff·row_sum_W'[oc]
//                           + K·x_zp_eff·w_zp_eff[oc] + bias[oc])
//                           · combined_scale[oc] + y_zp).
//
// Per-op device-side constants uploaded once at reshape:
//   row_sum_W      (int32[Cout])
//   w_zp_eff_arr   (int32[Cout])  — per-channel w_zp, shift-adjusted
//   combined_scale (f32[Cout])    — (x_scale · w_scale[oc]) / y_scale
// If w_scale / w_zp are scalar, we still expand to Cout-sized arrays for a
// single kernel path. The waste is Cout·12 bytes per op — negligible.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "attr_key.h"

#include <cstdint>
#include <vector>

namespace nnr {

operator_t* resolver_default_op_QLinearConv(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* qconv_kernels_source() {
    return R"CUDA(
#include <mma.h>
#include <cuda_pipeline_primitives.h>
using namespace nvcuda;

extern "C" {

__global__ void im2col_qs(const unsigned char* __restrict__ X,
                          signed char* __restrict__ col,
                          int x_shift, int x_zp_eff,
                          int Cin, int Hi, int Wi, int Ho, int Wo,
                          int kH, int kW, int sH, int sW,
                          int pT, int pL, int dH, int dW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cols = Ho * Wo;
    int rows = Cin * kH * kW;
    if (idx >= rows * cols) return;
    int col_idx = idx % cols;  int row = idx / cols;
    int kc = row % kW;         int t = row / kW;
    int kr = t % kH;           int ci = t / kH;
    int oh = col_idx / Wo;
    int ow = col_idx % Wo;
    int ih = oh * sH - pT + kr * dH;
    int iw = ow * sW - pL + kc * dW;
    int v = x_zp_eff;
    if ((unsigned)ih < (unsigned)Hi && (unsigned)iw < (unsigned)Wi) {
        int raw = (int)X[((size_t)ci * Hi + ih) * Wi + iw];
        v = raw - x_shift;
    }
    col[idx] = (signed char)v;
}

// Requantize with per-channel w_zp and combined_scale. K_total = Cin·kH·kW.
// Note: this standalone requantize kernel is no longer wired — the fused
// GEMM kernel below does the epilogue inline. Kept in-source as a simpler
// reference for correctness debugging / A/B testing.
#define REQUANT_BODY(YT, Y_MIN, Y_MAX)                                         \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int total = Cout * M;                                                      \
    if (idx >= total) return;                                                  \
    int pixel = idx % M;                                                       \
    int oc = idx / M;                                                          \
    int wzp = w_zp_eff_arr[oc];                                                \
    int acc = Acc[idx];                                                        \
    acc -= wzp * col_sum[pixel];                                               \
    acc -= x_zp_eff * row_sum_W[oc];                                           \
    acc += K_total * x_zp_eff * wzp;                                           \
    if (has_bias) acc += Bias[oc];                                             \
    float f = (float)acc * combined_scale[oc] + (float)y_zp;                   \
    int q = (int)rintf(f);                                                     \
    if (q < (Y_MIN)) q = (Y_MIN);                                              \
    if (q > (Y_MAX)) q = (Y_MAX);                                              \
    Y[idx] = (YT)q;

__global__ void qconv_requant_u8(const int* __restrict__ Acc,
                                 unsigned char* __restrict__ Y,
                                 const int* __restrict__ col_sum,
                                 const int* __restrict__ row_sum_W,
                                 const int* __restrict__ w_zp_eff_arr,
                                 const float* __restrict__ combined_scale,
                                 const int* __restrict__ Bias,
                                 int Cout, int M,
                                 int x_zp_eff, int K_total, int y_zp,
                                 int has_bias)
{
    REQUANT_BODY(unsigned char, 0, 255)
}

__global__ void qconv_requant_s8(const int* __restrict__ Acc,
                                 signed char* __restrict__ Y,
                                 const int* __restrict__ col_sum,
                                 const int* __restrict__ row_sum_W,
                                 const int* __restrict__ w_zp_eff_arr,
                                 const float* __restrict__ combined_scale,
                                 const int* __restrict__ Bias,
                                 int Cout, int M,
                                 int x_zp_eff, int K_total, int y_zp,
                                 int has_bias)
{
    REQUANT_BODY(signed char, -128, 127)
}
#undef REQUANT_BODY

// Fused GEMM + requantize. Equivalent to gemm_tc_s8s32_blocked followed by
// qconv_requant_{u8,s8}, but the int32 Acc tile lives only in shared memory —
// never round-trips through HBM. Saves one full MN·4-byte pass per conv.
//
// Each thread, after store_matrix_sync lands Acc in C_scratch, reads 32 ints
// from its assigned slab, applies the full quant epilogue, and writes 32
// int8/uint8 elements to Y. Drop-in replacement for the 2-launch sequence.
//
// Layout:
//   A[M=Cout, K] row-major        (signed int8; weight)
//   B[K, N=M_spatial] row-major   (signed int8; im2col output)
//   Y[M, N] row-major             (int8 or uint8)
// col_sum[gc] is folded into the kernel: each block accumulates its own
// column-wise sum across the full K-loop into shared memory. The epilogue
// then subtracts `w_zp_eff[oc] * col_sum_tile[col_in_block]` inline. This
// eliminates both the separate col_sum kernel launch and the 52 MB HBM pass
// that kernel made over B = im2col output for large Block-1-style convs.
#define GEMM_QCONV_FUSED_BODY(YT, Y_MIN, Y_MAX)                                \
    constexpr int BM = 64, BN = 64, BK = 16;                                   \
    __shared__ signed char A_tile[BM][BK];                                     \
    __shared__ signed char B_tile[BK][BN];                                     \
    __shared__ int         C_scratch[BM][BN];                                  \
    __shared__ int         col_sum_tile[BN];                                   \
    const int tid = threadIdx.x;                                               \
    const int warp_id = tid / 32;                                              \
    const int warp_m  = warp_id / 2;                                           \
    const int warp_n  = warp_id % 2;                                           \
    const int block_m = blockIdx.y * BM;                                       \
    const int block_n = blockIdx.x * BN;                                       \
    if (tid < BN) col_sum_tile[tid] = 0;                                       \
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag[2][2];           \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < 2; ++i)                                                \
        _Pragma("unroll")                                                      \
        for (int j = 0; j < 2; ++j)                                            \
            wmma::fill_fragment(c_frag[i][j], 0);                              \
    for (int k = 0; k < K; k += BK) {                                          \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < 8; ++i) {                                          \
            int lin = tid * 8 + i;                                             \
            int row = lin / BK, col = lin % BK;                                \
            int gr = block_m + row, gc = k + col;                              \
            A_tile[row][col] = (gr < M && gc < K)                              \
                ? A[(size_t)gr * K + gc] : (signed char)0;                     \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < 8; ++i) {                                          \
            int lin = tid * 8 + i;                                             \
            int row = lin / BN, col = lin % BN;                                \
            int gr = k + row, gc = block_n + col;                              \
            B_tile[row][col] = (gr < K && gc < N)                              \
                ? B[(size_t)gr * N + gc] : (signed char)0;                     \
        }                                                                      \
        __syncthreads();                                                       \
        /* per-K-chunk column sums: 64 threads, one column each. */            \
        if (tid < BN) {                                                        \
            int csum = 0;                                                      \
            _Pragma("unroll")                                                  \
            for (int r = 0; r < BK; ++r) csum += (int)B_tile[r][tid];          \
            col_sum_tile[tid] += csum;                                         \
        }                                                                      \
        wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag[2]; \
        wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major> b_frag[2]; \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < 2; ++i)                                            \
            wmma::load_matrix_sync(a_frag[i],                                  \
                &A_tile[warp_m * 32 + i * 16][0], BK);                         \
        _Pragma("unroll")                                                      \
        for (int j = 0; j < 2; ++j)                                            \
            wmma::load_matrix_sync(b_frag[j],                                  \
                &B_tile[0][warp_n * 32 + j * 16], BN);                         \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < 2; ++i)                                            \
            _Pragma("unroll")                                                  \
            for (int j = 0; j < 2; ++j)                                        \
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]); \
        __syncthreads();                                                       \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < 2; ++i) {                                              \
        _Pragma("unroll")                                                      \
        for (int j = 0; j < 2; ++j) {                                          \
            int sm_row = warp_m * 32 + i * 16;                                 \
            int sm_col = warp_n * 32 + j * 16;                                 \
            wmma::store_matrix_sync(&C_scratch[sm_row][sm_col],                \
                                    c_frag[i][j], BN, wmma::mem_row_major);   \
        }                                                                      \
    }                                                                          \
    __syncthreads();                                                           \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < 32; ++i) {                                             \
        int lin = tid * 32 + i;                                                \
        int row = lin / BN, col = lin % BN;                                    \
        int gr = block_m + row, gc = block_n + col;                            \
        if (gr < M && gc < N) {                                                \
            int wzp = w_zp_eff_arr[gr];                                        \
            int acc = C_scratch[row][col];                                     \
            acc -= wzp * col_sum_tile[col];                                    \
            acc -= x_zp_eff * row_sum_W[gr];                                   \
            acc += K_total * x_zp_eff * wzp;                                   \
            if (has_bias) acc += Bias[gr];                                     \
            float f = (float)acc * combined_scale[gr] + (float)y_zp;           \
            int q = (int)rintf(f);                                             \
            if (q < (Y_MIN)) q = (Y_MIN);                                      \
            if (q > (Y_MAX)) q = (Y_MAX);                                      \
            Y[(size_t)gr * N + gc] = (YT)q;                                    \
        }                                                                      \
    }

)CUDA"
R"CUDA(
// Implicit GEMM — the B_tile (im2col column slab) is computed on-the-fly from
// X instead of materialized to HBM. Eliminates the ~52 MB/call d_col buffer on
// ssd-12 Block-1 shapes. Each thread that would normally load from B[gr,gc]
// decodes the im2col index instead:
//   gr → (ci, kr, kc)   via two integer divisions
//   gc → (oh, ow)       via Wo division
//   ih = oh·sH − pT + kr·dH,  iw = ow·sW − pL + kc·dW
// and reads X[ci, ih, iw] with OOB padding = x_zp_eff (shifted domain).
//
// XT = input byte type (signed char or unsigned char — controls sign of the
// X read). x_shift handles the uint8→int8 unification at the same time.
#define GEMM_QCONV_IMPLICIT_BODY(XT, YT, Y_MIN, Y_MAX)                         \
    constexpr int BM = 64, BN = 64, BK = 16;                                   \
    __shared__ signed char A_tile[2][BM][BK];                                  \
    __shared__ signed char B_tile[BK][BN];                                     \
    __shared__ int         C_scratch[BM][BN];                                  \
    __shared__ int         col_sum_tile[BN];                                   \
    const int tid = threadIdx.x;                                               \
    const int warp_id = tid / 32;                                              \
    const int warp_m  = warp_id / 2;                                           \
    const int warp_n  = warp_id % 2;                                           \
    const int block_m = blockIdx.y * BM;                                       \
    const int block_n = blockIdx.x * BN;                                       \
    if (tid < BN) col_sum_tile[tid] = 0;                                       \
    const int K_total = Cin * kH * kW;                                         \
    const int M_spatial = Ho * Wo;                                             \
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag[2][2];           \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < 2; ++i)                                                \
        _Pragma("unroll")                                                      \
        for (int j = 0; j < 2; ++j)                                            \
            wmma::fill_fragment(c_frag[i][j], 0);                              \
    /* A_tile issue: 1 row (16 B = BK bytes) per thread via cp.async.16 when   \
     * source is 16B-aligned + fully in bounds; scalar fallback otherwise.    \
     * 64 active threads of 128 (tid < BM). */                                \
    auto issue_A = [&](int k_base, int buf) {                                  \
        if (tid < BM) {                                                        \
            int row = tid;                                                     \
            int gr  = block_m + row;                                           \
            int gc  = k_base;                                                  \
            signed char*       dst = &A_tile[buf][row][0];                     \
            bool in_row = (gr < Cout);                                         \
            const signed char* src = in_row                                    \
                ? &A[(size_t)gr * K_total + gc] : &A[0];                       \
            bool all_in  = in_row && (gc + BK <= K_total);                     \
            bool aligned = all_in &&                                           \
                ((reinterpret_cast<uintptr_t>(src) & 15u) == 0);               \
            if (aligned) {                                                     \
                __pipeline_memcpy_async(dst, src, 16);                         \
            } else {                                                           \
                _Pragma("unroll")                                              \
                for (int j = 0; j < BK; ++j)                                   \
                    dst[j] = (in_row && (gc + j) < K_total)                    \
                           ? src[j] : (signed char)0;                          \
            }                                                                  \
        }                                                                      \
    };                                                                         \
    /* Kick off tile 0. */                                                     \
    issue_A(0, 0);                                                             \
    __pipeline_commit();                                                       \
    int cur = 0;                                                               \
    for (int k = 0; k < K_total; k += BK) {                                    \
        int nxt = 1 - cur;                                                     \
        int k_next = k + BK;                                                   \
        /* Prefetch A[k+BK] into buffer[nxt] while consuming buffer[cur]. */   \
        if (k_next < K_total) {                                                \
            issue_A(k_next, nxt);                                              \
            __pipeline_commit();                                               \
            __pipeline_wait_prior(1);                                          \
        } else {                                                               \
            __pipeline_wait_prior(0);                                          \
        }                                                                      \
        /* B_tile computed implicitly from X — no d_col buffer. */             \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < 8; ++i) {                                          \
            int lin = tid * 8 + i;                                             \
            int row = lin / BN, col = lin % BN;                                \
            int gr = k + row, gc = block_n + col;                              \
            signed char v = 0;                                                 \
            if (gr < K_total && gc < M_spatial) {                              \
                int kc = gr % kW;                                              \
                int t  = gr / kW;                                              \
                int kr = t % kH;                                               \
                int ci = t / kH;                                               \
                int oh = gc / Wo;                                              \
                int ow = gc - oh * Wo;                                         \
                int ih = oh * sH - pT + kr * dH;                               \
                int iw = ow * sW - pL + kc * dW;                               \
                if ((unsigned)ih < (unsigned)Hi                                \
                 && (unsigned)iw < (unsigned)Wi) {                             \
                    int raw = (int)X[((size_t)ci * Hi + ih) * Wi + iw];        \
                    v = (signed char)(raw - x_shift);                          \
                } else {                                                       \
                    v = (signed char)x_zp_eff;                                 \
                }                                                              \
            }                                                                  \
            B_tile[row][col] = v;                                              \
        }                                                                      \
        __syncthreads();                                                       \
        if (tid < BN) {                                                        \
            int csum = 0;                                                      \
            _Pragma("unroll")                                                  \
            for (int r = 0; r < BK; ++r) csum += (int)B_tile[r][tid];          \
            col_sum_tile[tid] += csum;                                         \
        }                                                                      \
        wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag[2]; \
        wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major> b_frag[2]; \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < 2; ++i)                                            \
            wmma::load_matrix_sync(a_frag[i],                                  \
                &A_tile[cur][warp_m * 32 + i * 16][0], BK);                    \
        _Pragma("unroll")                                                      \
        for (int j = 0; j < 2; ++j)                                            \
            wmma::load_matrix_sync(b_frag[j],                                  \
                &B_tile[0][warp_n * 32 + j * 16], BN);                         \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < 2; ++i)                                            \
            _Pragma("unroll")                                                  \
            for (int j = 0; j < 2; ++j)                                        \
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]); \
        __syncthreads();                                                       \
        cur = nxt;                                                             \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < 2; ++i) {                                              \
        _Pragma("unroll")                                                      \
        for (int j = 0; j < 2; ++j) {                                          \
            int sm_row = warp_m * 32 + i * 16;                                 \
            int sm_col = warp_n * 32 + j * 16;                                 \
            wmma::store_matrix_sync(&C_scratch[sm_row][sm_col],                \
                                    c_frag[i][j], BN, wmma::mem_row_major);   \
        }                                                                      \
    }                                                                          \
    __syncthreads();                                                           \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < 32; ++i) {                                             \
        int lin = tid * 32 + i;                                                \
        int row = lin / BN, col = lin % BN;                                    \
        int gr = block_m + row, gc = block_n + col;                            \
        if (gr < Cout && gc < M_spatial) {                                     \
            int wzp = w_zp_eff_arr[gr];                                        \
            int acc = C_scratch[row][col];                                     \
            acc -= wzp * col_sum_tile[col];                                    \
            acc -= x_zp_eff * row_sum_W[gr];                                   \
            acc += K_total * x_zp_eff * wzp;                                   \
            if (has_bias) acc += Bias[gr];                                     \
            float f = (float)acc * combined_scale[gr] + (float)y_zp;           \
            int q = (int)rintf(f);                                             \
            if (q < (Y_MIN)) q = (Y_MIN);                                      \
            if (q > (Y_MAX)) q = (Y_MAX);                                      \
            Y[(size_t)gr * M_spatial + gc] = (YT)q;                            \
        }                                                                      \
    }

__global__ void gemm_qconv_implicit_u8(const signed char* __restrict__ A,
                                       const unsigned char* __restrict__ X,
                                       unsigned char* __restrict__ Y,
                                       const int* __restrict__ row_sum_W,
                                       const int* __restrict__ w_zp_eff_arr,
                                       const float* __restrict__ combined_scale,
                                       const int* __restrict__ Bias,
                                       int Cout, int Cin, int Hi, int Wi, int Ho, int Wo,
                                       int kH, int kW, int sH, int sW,
                                       int pT, int pL, int dH, int dW,
                                       int x_shift, int x_zp_eff, int y_zp,
                                       int has_bias)
{
    GEMM_QCONV_IMPLICIT_BODY(unsigned char, unsigned char, 0, 255)
}

__global__ void gemm_qconv_implicit_s8(const signed char* __restrict__ A,
                                       const signed char* __restrict__ X,
                                       signed char* __restrict__ Y,
                                       const int* __restrict__ row_sum_W,
                                       const int* __restrict__ w_zp_eff_arr,
                                       const float* __restrict__ combined_scale,
                                       const int* __restrict__ Bias,
                                       int Cout, int Cin, int Hi, int Wi, int Ho, int Wo,
                                       int kH, int kW, int sH, int sW,
                                       int pT, int pL, int dH, int dW,
                                       int x_shift, int x_zp_eff, int y_zp,
                                       int has_bias)
{
    GEMM_QCONV_IMPLICIT_BODY(signed char, signed char, -128, 127)
}
#undef GEMM_QCONV_IMPLICIT_BODY

)CUDA"
R"CUDA(

// NHWC variant of the implicit-GEMM kernel.
//   X   : [N, Hi, Wi, Cin]  row-major (NHWC)
//   A   : [Cout, kH, kW, Cin] row-major  (weights pre-packed in NHWC K-order)
//   Y   : [N, Ho, Wo, Cout] row-major (NHWC)
// K-decoder differs from NCHW: ci varies fastest in K so a 16-wide WMMA
// k-step reads X[ih][iw][ci..ci+16] — 16 contiguous bytes, fully coalesced.
// Output epilogue writes to Y in NHWC layout: Y[oh][ow][oc].
#define GEMM_QCONV_IMPLICIT_NHWC_BODY(XT, YT, Y_MIN, Y_MAX)                    \
    constexpr int BM = 64, BN = 64, BK = 16;                                   \
    __shared__ signed char A_tile[2][BM][BK];                                  \
    __shared__ signed char B_tile[BK][BN];                                     \
    __shared__ int         C_scratch[BM][BN];                                  \
    __shared__ int         col_sum_tile[BN];                                   \
    const int tid = threadIdx.x;                                               \
    const int warp_id = tid / 32;                                              \
    const int warp_m  = warp_id / 2;                                           \
    const int warp_n  = warp_id % 2;                                           \
    const int block_m = blockIdx.y * BM;                                       \
    const int block_n = blockIdx.x * BN;                                       \
    if (tid < BN) col_sum_tile[tid] = 0;                                       \
    const int K_total = Cin * kH * kW;                                         \
    const int M_spatial = Ho * Wo;                                             \
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag[2][2];           \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < 2; ++i)                                                \
        _Pragma("unroll")                                                      \
        for (int j = 0; j < 2; ++j)                                            \
            wmma::fill_fragment(c_frag[i][j], 0);                              \
    auto issue_A = [&](int k_base, int buf) {                                  \
        if (tid < BM) {                                                        \
            int row = tid;                                                     \
            int gr  = block_m + row;                                           \
            int gc  = k_base;                                                  \
            signed char*       dst = &A_tile[buf][row][0];                     \
            bool in_row = (gr < Cout);                                         \
            const signed char* src = in_row                                    \
                ? &A[(size_t)gr * K_total + gc] : &A[0];                       \
            bool all_in  = in_row && (gc + BK <= K_total);                     \
            bool aligned = all_in &&                                           \
                ((reinterpret_cast<uintptr_t>(src) & 15u) == 0);               \
            if (aligned) {                                                     \
                __pipeline_memcpy_async(dst, src, 16);                         \
            } else {                                                           \
                _Pragma("unroll")                                              \
                for (int j = 0; j < BK; ++j)                                   \
                    dst[j] = (in_row && (gc + j) < K_total)                    \
                           ? src[j] : (signed char)0;                          \
            }                                                                  \
        }                                                                      \
    };                                                                         \
    issue_A(0, 0);                                                             \
    __pipeline_commit();                                                       \
    int cur = 0;                                                               \
    for (int k = 0; k < K_total; k += BK) {                                    \
        int nxt = 1 - cur;                                                     \
        int k_next = k + BK;                                                   \
        if (k_next < K_total) {                                                \
            issue_A(k_next, nxt);                                              \
            __pipeline_commit();                                               \
            __pipeline_wait_prior(1);                                          \
        } else {                                                               \
            __pipeline_wait_prior(0);                                          \
        }                                                                      \
        /* B_tile from X (NHWC). Coalesced layout: each "chunk" of 16        \
         * threads reads 16 contiguous ci bytes from X[ih][iw][ci..ci+16]    \
         * for one spatial col. 128 threads/block = 8 chunks per pass × 8    \
         * passes = 64 spatial cols × BK=16 = 1024 cells = full B_tile.      \
         * Each warp (32 threads) covers 2 chunks → 2 coalesced 16-B reads. */\
        {                                                                      \
            int row = tid & 15;             /* 0..15: ci within K-step */     \
            _Pragma("unroll")                                                  \
            for (int pass = 0; pass < 8; ++pass) {                             \
                int chunk = (tid >> 4) & 7; /* 0..7 chunks/pass */             \
                int col = pass * 8 + chunk; /* 0..63 spatial col */            \
                int gr = k + row, gc = block_n + col;                          \
                signed char v = 0;                                             \
                if (gr < K_total && gc < M_spatial) {                          \
                    int ci = gr % Cin;                                         \
                    int t  = gr / Cin;                                         \
                    int kc = t % kW;                                           \
                    int kr = t / kW;                                           \
                    int oh = gc / Wo;                                          \
                    int ow = gc - oh * Wo;                                     \
                    int ih = oh * sH - pT + kr * dH;                           \
                    int iw = ow * sW - pL + kc * dW;                           \
                    if ((unsigned)ih < (unsigned)Hi                            \
                     && (unsigned)iw < (unsigned)Wi) {                         \
                        int raw = (int)X[(((size_t)ih * Wi) + iw) * Cin + ci]; \
                        v = (signed char)(raw - x_shift);                      \
                    } else {                                                   \
                        v = (signed char)x_zp_eff;                             \
                    }                                                          \
                }                                                              \
                B_tile[row][col] = v;                                          \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        if (tid < BN) {                                                        \
            int csum = 0;                                                      \
            _Pragma("unroll")                                                  \
            for (int r = 0; r < BK; ++r) csum += (int)B_tile[r][tid];          \
            col_sum_tile[tid] += csum;                                         \
        }                                                                      \
        wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag[2]; \
        wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major> b_frag[2]; \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < 2; ++i)                                            \
            wmma::load_matrix_sync(a_frag[i],                                  \
                &A_tile[cur][warp_m * 32 + i * 16][0], BK);                    \
        _Pragma("unroll")                                                      \
        for (int j = 0; j < 2; ++j)                                            \
            wmma::load_matrix_sync(b_frag[j],                                  \
                &B_tile[0][warp_n * 32 + j * 16], BN);                         \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < 2; ++i)                                            \
            _Pragma("unroll")                                                  \
            for (int j = 0; j < 2; ++j)                                        \
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]); \
        __syncthreads();                                                       \
        cur = nxt;                                                             \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < 2; ++i) {                                              \
        _Pragma("unroll")                                                      \
        for (int j = 0; j < 2; ++j) {                                          \
            int sm_row = warp_m * 32 + i * 16;                                 \
            int sm_col = warp_n * 32 + j * 16;                                 \
            wmma::store_matrix_sync(&C_scratch[sm_row][sm_col],                \
                                    c_frag[i][j], BN, wmma::mem_row_major);   \
        }                                                                      \
    }                                                                          \
    __syncthreads();                                                           \
    /* NHWC Y write: Y[oh][ow][oc] = Y[gc * Cout + gr]. */                     \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < 32; ++i) {                                             \
        int lin = tid * 32 + i;                                                \
        int row = lin / BN, col = lin % BN;                                    \
        int gr = block_m + row, gc = block_n + col;                            \
        if (gr < Cout && gc < M_spatial) {                                     \
            int wzp = w_zp_eff_arr[gr];                                        \
            int acc = C_scratch[row][col];                                     \
            acc -= wzp * col_sum_tile[col];                                    \
            acc -= x_zp_eff * row_sum_W[gr];                                   \
            acc += K_total * x_zp_eff * wzp;                                   \
            if (has_bias) acc += Bias[gr];                                     \
            float f = (float)acc * combined_scale[gr] + (float)y_zp;           \
            int q = (int)rintf(f);                                             \
            if (q < (Y_MIN)) q = (Y_MIN);                                      \
            if (q > (Y_MAX)) q = (Y_MAX);                                      \
            Y[(size_t)gc * Cout + gr] = (YT)q;                                 \
        }                                                                      \
    }

__global__ void gemm_qconv_implicit_nhwc_u8(const signed char* __restrict__ A,
                                            const unsigned char* __restrict__ X,
                                            unsigned char* __restrict__ Y,
                                            const int* __restrict__ row_sum_W,
                                            const int* __restrict__ w_zp_eff_arr,
                                            const float* __restrict__ combined_scale,
                                            const int* __restrict__ Bias,
                                            int Cout, int Cin, int Hi, int Wi, int Ho, int Wo,
                                            int kH, int kW, int sH, int sW,
                                            int pT, int pL, int dH, int dW,
                                            int x_shift, int x_zp_eff, int y_zp,
                                            int has_bias)
{
    GEMM_QCONV_IMPLICIT_NHWC_BODY(unsigned char, unsigned char, 0, 255)
}

__global__ void gemm_qconv_implicit_nhwc_s8(const signed char* __restrict__ A,
                                            const signed char* __restrict__ X,
                                            signed char* __restrict__ Y,
                                            const int* __restrict__ row_sum_W,
                                            const int* __restrict__ w_zp_eff_arr,
                                            const float* __restrict__ combined_scale,
                                            const int* __restrict__ Bias,
                                            int Cout, int Cin, int Hi, int Wi, int Ho, int Wo,
                                            int kH, int kW, int sH, int sW,
                                            int pT, int pL, int dH, int dW,
                                            int x_shift, int x_zp_eff, int y_zp,
                                            int has_bias)
{
    GEMM_QCONV_IMPLICIT_NHWC_BODY(signed char, signed char, -128, 127)
}
#undef GEMM_QCONV_IMPLICIT_NHWC_BODY

__global__ void gemm_qconv_fused_u8(const signed char* __restrict__ A,
                                    const signed char* __restrict__ B,
                                    unsigned char* __restrict__ Y,
                                    const int* __restrict__ row_sum_W,
                                    const int* __restrict__ w_zp_eff_arr,
                                    const float* __restrict__ combined_scale,
                                    const int* __restrict__ Bias,
                                    int M, int N, int K,
                                    int x_zp_eff, int K_total, int y_zp,
                                    int has_bias)
{
    GEMM_QCONV_FUSED_BODY(unsigned char, 0, 255)
}

__global__ void gemm_qconv_fused_s8(const signed char* __restrict__ A,
                                    const signed char* __restrict__ B,
                                    signed char* __restrict__ Y,
                                    const int* __restrict__ row_sum_W,
                                    const int* __restrict__ w_zp_eff_arr,
                                    const float* __restrict__ combined_scale,
                                    const int* __restrict__ Bias,
                                    int M, int N, int K,
                                    int x_zp_eff, int K_total, int y_zp,
                                    int has_bias)
{
    GEMM_QCONV_FUSED_BODY(signed char, -128, 127)
}
#undef GEMM_QCONV_FUSED_BODY

} // extern "C"
)CUDA";
}

struct QLinearConv_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;

    int N=0, Cin=0, Hi=0, Wi=0, Cout=0, Ho=0, Wo=0;
    int kH=0, kW=0, sH=1, sW=1, pT=0, pL=0, dH=1, dW=1;
    int K_gemm = 0, M_gemm = 0;
    int x_zp_eff = 0, y_zp = 0;
    int x_shift = 0, w_shift = 0;
    bool has_bias = false;
    bool y_is_uint8 = true;
    bool w_shifted = false;

    int*   d_row_sum_W = nullptr;       // int32[Cout]
    int*   d_w_zp_eff_arr = nullptr;    // int32[Cout]
    float* d_combined_scale = nullptr;  // f32[Cout]
    void*  d_w_signed = nullptr;        // int8[Cout×K_gemm] (only if W is uint8)
    void*  d_w_nhwc   = nullptr;        // int8[Cout × kH×kW×Cin] — weights repacked
                                        // in NHWC K-order: [Cout][kH][kW][Cin].
                                        // Allocated lazily when inputs[0]->format
                                        // becomes NHWC at first exec.
    void*  d_col = nullptr;             // int8[K_gemm×M_gemm]
    size_t d_col_bytes = 0;

    const char* fused_name = nullptr;
    const char* implicit_name = nullptr;
    bool use_implicit = false;
    bool x_is_uint8 = true;

    bool init() override {
        if (!(inputs.size() >= 8 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_QLinearConv(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    size_t workspace_size() const override { return fallback ? fallback->workspace_size() : 0; }

    static bool is_q8(data_type_t t) {
        return t == NNR_DATA_TYPE_UINT8 || t == NNR_DATA_TYPE_INT8;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;

        const tensor_t* X  = inputs[0];
        const tensor_t* xs = inputs[1];
        const tensor_t* xz = inputs[2];
        const tensor_t* Wt = inputs[3];
        const tensor_t* ws = inputs[4];
        const tensor_t* wz = inputs[5];
        const tensor_t* ys = inputs[6];
        const tensor_t* yz = inputs[7];
        const tensor_t* Bi = (inputs.size() > 8 && inputs[8] && inputs[8]->ndata > 0) ? inputs[8] : nullptr;
        tensor_t* Y = outputs[0];

        if (X->ndim != 4 || Wt->ndim != 4 || Y->ndim != 4) return true;
        if (!is_q8(X->type) || !is_q8(Wt->type) || !is_q8(Y->type)) return true;
        x_shift = (X->type  == NNR_DATA_TYPE_UINT8) ? 128 : 0;
        w_shift = (Wt->type == NNR_DATA_TYPE_UINT8) ? 128 : 0;
        y_is_uint8 = (Y->type == NNR_DATA_TYPE_UINT8);
        w_shifted = (w_shift != 0);

        int64_t* ints = nullptr;
        int n = 0;
        n = attribute(attr_key_t::pads, ints);      pT = (n>=2)?(int)ints[0]:0; pL = (n>=2)?(int)ints[1]:0;
        n = attribute(attr_key_t::strides, ints);   sH = (n>=1)?(int)ints[0]:1; sW = (n>=2)?(int)ints[1]:1;
        n = attribute(attr_key_t::dilations, ints); dH = (n>=1)?(int)ints[0]:1; dW = (n>=2)?(int)ints[1]:1;
        int group = (int)attribute(attr_key_t::group, (int64_t)1);
        if (group != 1) return true;

        N = X->dims[0]; Cin = X->dims[1]; Hi = X->dims[2]; Wi = X->dims[3];
        Cout = Wt->dims[0]; kH = Wt->dims[2]; kW = Wt->dims[3];
        if (Wt->dims[1] != Cin) return true;
        Ho = Y->dims[2]; Wo = Y->dims[3];
        if (Y->dims[1] != Cout) return true;
        K_gemm = Cin * kH * kW;
        M_gemm = Ho * Wo;

        // Scales: x_scale scalar, y_scale scalar, w_scale scalar OR per-channel Cout.
        if (xs->type != NNR_DATA_TYPE_FLOAT32 || xs->ndata != 1 || !xs->data) return true;
        if (ws->type != NNR_DATA_TYPE_FLOAT32 || !ws->data) return true;
        if (!(ws->ndata == 1 || (int64_t)ws->ndata == Cout)) return true;
        if (ys->type != NNR_DATA_TYPE_FLOAT32 || ys->ndata != 1 || !ys->data) return true;
        float x_scale = *(const float*)xs->data;
        float y_scale = *(const float*)ys->data;
        if (y_scale == 0.f) return true;
        const float* w_scale_arr = (const float*)ws->data;
        bool w_scale_pc = (ws->ndata == Cout);

        // Zero-points: x_zp / y_zp scalar; w_zp scalar or per-channel Cout.
        auto read_scalar_zp = [](const tensor_t* t) -> int {
            if (!t || t->ndata == 0 || !t->data) return 0;
            if (t->ndata != 1) return INT32_MIN;
            if (t->type == NNR_DATA_TYPE_UINT8) return (int)*(const uint8_t*)t->data;
            if (t->type == NNR_DATA_TYPE_INT8)  return (int)*(const int8_t*)t->data;
            return INT32_MIN;
        };
        int x_zp = read_scalar_zp(xz);
        y_zp = read_scalar_zp(yz);
        if (x_zp == INT32_MIN || y_zp == INT32_MIN) return true;
        x_zp_eff = x_zp - x_shift;

        // w_zp: allow scalar or (Cout,) array.
        bool w_zp_pc = false;
        int  w_zp_scalar = 0;
        if (!wz || wz->ndata == 0 || !wz->data) {
            w_zp_scalar = 0;
        } else if (wz->ndata == 1) {
            if (wz->type == NNR_DATA_TYPE_UINT8)      w_zp_scalar = (int)*(const uint8_t*)wz->data;
            else if (wz->type == NNR_DATA_TYPE_INT8)  w_zp_scalar = (int)*(const int8_t*)wz->data;
            else return true;
        } else if ((int64_t)wz->ndata == Cout) {
            if (wz->type != Wt->type) return true;
            w_zp_pc = true;
        } else {
            return true;
        }

        has_bias = (Bi && Bi->type == NNR_DATA_TYPE_INT32
                    && (int64_t)Bi->ndata == Cout && Bi->data);

        fused_name = y_is_uint8 ? "gemm_qconv_fused_u8" : "gemm_qconv_fused_s8";
        // Implicit-GEMM path: skip the d_col materialization entirely when the
        // X input type matches Y. Computes im2col inline in the WMMA mainloop.
        x_is_uint8 = (X->type == NNR_DATA_TYPE_UINT8);
        use_implicit = (X->type == Y->type);
        implicit_name = use_implicit
            ? (x_is_uint8 ? "gemm_qconv_implicit_u8" : "gemm_qconv_implicit_s8")
            : nullptr;

        // Build host-side per-channel tables and (if needed) a shifted weight copy.
        // Row_sum_W is computed on the *shifted* weight — matches epilogue algebra.
        std::vector<int32_t> host_row_sum((size_t)Cout, 0);
        std::vector<int32_t> host_w_zp_eff((size_t)Cout, 0);
        std::vector<float>   host_combined_scale((size_t)Cout, 0.f);
        std::vector<int8_t>  host_w_shifted;
        if (w_shifted) host_w_shifted.resize((size_t)Cout * K_gemm);

        const uint8_t* wu = (const uint8_t*)Wt->data;
        const int8_t*  ws8 = (const int8_t*)Wt->data;
        const uint8_t* wzu = w_zp_pc ? (const uint8_t*)wz->data : nullptr;
        const int8_t*  wzs = w_zp_pc ? (const int8_t*)wz->data  : nullptr;

        for (int oc = 0; oc < Cout; ++oc) {
            int64_t s = 0;
            if (Wt->type == NNR_DATA_TYPE_UINT8) {
                for (int k = 0; k < K_gemm; ++k) {
                    int v = (int)wu[(size_t)oc * K_gemm + k] - w_shift;
                    host_w_shifted[(size_t)oc * K_gemm + k] = (int8_t)v;
                    s += v;
                }
            } else {
                for (int k = 0; k < K_gemm; ++k)
                    s += (int64_t)ws8[(size_t)oc * K_gemm + k];
            }
            host_row_sum[oc] = (int32_t)s;

            int w_zp_this;
            if (w_zp_pc) {
                w_zp_this = (Wt->type == NNR_DATA_TYPE_UINT8)
                          ? (int)wzu[oc] : (int)wzs[oc];
            } else {
                w_zp_this = w_zp_scalar;
            }
            host_w_zp_eff[oc] = w_zp_this - w_shift;

            float w_sc_this = w_scale_pc ? w_scale_arr[oc] : w_scale_arr[0];
            host_combined_scale[oc] = (x_scale * w_sc_this) / y_scale;
        }

        // NHWC weight pack: reorder [Cout, Cin, kH, kW] (NCHW K-order) to
        // [Cout, kH, kW, Cin] (NHWC K-order). Source values come from
        // host_w_shifted (uint8 weights that were already shifted to int8) or
        // directly from Wt->data (already int8). Both signed-int8 in dst.
        std::vector<int8_t> host_w_nhwc((size_t)Cout * K_gemm);
        {
            const int8_t* src_w = w_shifted ? host_w_shifted.data()
                                            : (const int8_t*)Wt->data;
            for (int oc = 0; oc < Cout; ++oc) {
                for (int ci = 0; ci < Cin; ++ci) {
                    for (int kr = 0; kr < kH; ++kr) {
                        for (int kc = 0; kc < kW; ++kc) {
                            // Source: oc, ci, kr, kc → linear (oc*Cin + ci)*kH*kW + kr*kW + kc
                            size_t s = ((size_t)oc * Cin + ci) * kH * kW
                                     + (size_t)kr * kW + kc;
                            // Dest: oc, kr, kc, ci → linear oc*kH*kW*Cin + (kr*kW + kc)*Cin + ci
                            size_t d = (size_t)oc * kH * kW * Cin
                                     + ((size_t)kr * kW + kc) * Cin + ci;
                            host_w_nhwc[d] = src_w[s];
                        }
                    }
                }
            }
        }

        // (Re)allocate device buffers. Per-op; kept alive across inferences.
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        // If backend not yet created, defer. Uploads happen at first exec.
        if (be) {
            auto* dev = be->device;
            auto upload = [&](void*& dst, size_t bytes, const void* src) -> bool {
                if (!dst) dst = dev->alloc(bytes);
                if (!dst) return false;
                auto* evt = dev->copy_h2d_async(dst, src, bytes);
                if (evt) dev->compute_wait(evt);
                return true;
            };
            if (!upload(*(void**)&d_row_sum_W,       Cout * sizeof(int32_t), host_row_sum.data())) return true;
            if (!upload(*(void**)&d_w_zp_eff_arr,    Cout * sizeof(int32_t), host_w_zp_eff.data())) return true;
            if (!upload(*(void**)&d_combined_scale,  Cout * sizeof(float),   host_combined_scale.data())) return true;
            if (w_shifted && !upload(d_w_signed,     (size_t)Cout * K_gemm,  host_w_shifted.data())) return true;
            if (!upload(d_w_nhwc,                    (size_t)Cout * K_gemm,  host_w_nhwc.data())) return true;
        } else {
            // Backend not yet ready — cache host tables so first exec uploads.
            host_pending_row_sum_   = std::move(host_row_sum);
            host_pending_w_zp_eff_  = std::move(host_w_zp_eff);
            host_pending_cscale_    = std::move(host_combined_scale);
            host_pending_w_shifted_ = std::move(host_w_shifted);
            host_pending_w_nhwc_    = std::move(host_w_nhwc);
            have_host_pending_ = true;
        }

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        // Advertise NHWC support so assign_layouts.cpp will propagate NHWC
        // through chains of CUDA QLinearConv ops. The all-int8 force-accept
        // path in assign_layouts (see line ~253) takes effect because every
        // Conv in the chain is a QLinearConv. The actual NHWC kernel path
        // is not yet wired here — until it is, exec() will detect NHWC
        // input format and fall back to the CPU op (correct but slow,
        // breaks graph replay for that chain).
        layout_mask = LAYOUT_NCHW | LAYOUT_NHWC;
        return true;
    }

    // Placeholder cost model: NHWC is preferred over NCHW by 2× for int8
    // QLinearConv on CUDA. The proper formula will land with the WMMA NHWC
    // kernel (task #2). For now this just biases assign_layouts toward
    // accepting NHWC chains — the all-int8 force-accept already does this,
    // but we provide a real cost so non-int8 chains (currently none on
    // CUDA, but future) get a sensible signal.
    float layout_cost(memory_layout_t layout, bool input_nhwc) const override {
        // Use op count as a coarse cost (matches default operator_t logic).
        float ops = (float)((int64_t)Cout * Ho * Wo * K_gemm);
        if (layout == memory_layout_t::NHWC) return ops * 0.5f;
        return ops;
    }

    // First-exec upload path for reshapes that ran before the backend was created.
    std::vector<int32_t> host_pending_row_sum_;
    std::vector<int32_t> host_pending_w_zp_eff_;
    std::vector<float>   host_pending_cscale_;
    std::vector<int8_t>  host_pending_w_shifted_;
    std::vector<int8_t>  host_pending_w_nhwc_;     // [Cout][kH][kW][Cin] signed
    bool have_host_pending_ = false;

    bool drain_host_pending(gpu::cuda_backend_t* be) {
        if (!have_host_pending_) return true;
        auto* dev = be->device;
        auto upload = [&](void*& dst, size_t bytes, const void* src) -> bool {
            if (!dst) dst = dev->alloc(bytes);
            if (!dst) return false;
            auto* evt = dev->copy_h2d_async(dst, src, bytes);
            if (evt) dev->compute_wait(evt);
            return true;
        };
        if (!upload(*(void**)&d_row_sum_W,      Cout * sizeof(int32_t), host_pending_row_sum_.data())) return false;
        if (!upload(*(void**)&d_w_zp_eff_arr,   Cout * sizeof(int32_t), host_pending_w_zp_eff_.data())) return false;
        if (!upload(*(void**)&d_combined_scale, Cout * sizeof(float),   host_pending_cscale_.data())) return false;
        if (w_shifted && !upload(d_w_signed,    (size_t)Cout * K_gemm,  host_pending_w_shifted_.data())) return false;
        if (!upload(d_w_nhwc,                   (size_t)Cout * K_gemm,  host_pending_w_nhwc_.data())) return false;
        host_pending_row_sum_.clear();
        host_pending_w_zp_eff_.clear();
        host_pending_cscale_.clear();
        host_pending_w_shifted_.clear();
        host_pending_w_nhwc_.clear();
        have_host_pending_ = false;
        return true;
    }

    bool ensure_scratch(gpu::cuda_backend_t* be) {
        if (use_implicit) return true;   // no d_col needed
        auto* dev = be->device;
        size_t col_bytes = (size_t)K_gemm * M_gemm;
        if (col_bytes > d_col_bytes) {
            if (d_col) dev->free(d_col);
            d_col = dev->alloc(col_bytes);
            d_col_bytes = col_bytes;
        }
        return d_col != nullptr;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) return fallback->exec();
        if (!drain_host_pending(be)) return fallback->exec();
        if (!ensure_scratch(be)) return fallback->exec();

        // Detect NHWC input layout (set by graph optimizer's assign_layouts).
        // Output format follows input (already propagated by the optimizer).
        bool is_nhwc = (inputs[0]->format == memory_layout_t::NHWC);

        // NHWC path requires the implicit kernel (NHWC explicit-im2col is
        // not implemented). If a chain ended up requesting NHWC for a node
        // that can't use implicit GEMM (currently no such case for int8),
        // fall back to CPU. Implicit kernel is enabled when X.type == Y.type.
        if (is_nhwc && !use_implicit) return fallback->exec();

        const char* arch = gpu::nvrtc_arch_option(be->device);
        CUfunction f_im2col = nullptr;
        CUfunction f_fused  = nullptr;
        CUfunction f_implicit = nullptr;
        if (use_implicit) {
            const char* kname = is_nhwc
                ? (x_is_uint8 ? "gemm_qconv_implicit_nhwc_u8"
                              : "gemm_qconv_implicit_nhwc_s8")
                : implicit_name;
            f_implicit = be->nvrtc.get("nnr_qlconv", qconv_kernels_source(), kname, arch);
            if (!f_implicit) return fallback->exec();
        } else {
            f_im2col = be->nvrtc.get("nnr_qlconv", qconv_kernels_source(), "im2col_qs", arch);
            f_fused  = be->nvrtc.get("nnr_qlconv", qconv_kernels_source(), fused_name,  arch);
            if (!f_im2col || !f_fused) return fallback->exec();
        }

        void* d_x = be->cache->ensure_device(inputs[0]);
        // NHWC kernel reads weights from d_w_nhwc ([Cout][kH][kW][Cin]).
        // NCHW kernel reads from d_w_signed (if uint8 weight) or original.
        void* d_w = is_nhwc
            ? d_w_nhwc
            : (w_shifted ? d_w_signed : be->cache->ensure_device(inputs[3]));
        void* d_bias = has_bias ? be->cache->ensure_device(inputs[8]) : nullptr;
        void* d_y = be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_w || !d_y) return fallback->exec();

        int _xzp = x_zp_eff, _yzp = y_zp;
        int _xshift = x_shift;
        int _hb = has_bias ? 1 : 0;
        int _Cin = Cin, _Hi = Hi, _Wi = Wi, _Ho = Ho, _Wo = Wo;
        int _kH = kH, _kW = kW, _sH = sH, _sW = sW;
        int _pT = pT, _pL = pL, _dH = dH, _dW = dW;
        int _K = K_gemm, _M = M_gemm, _Cout = Cout, _Kt = K_gemm;

        for (int b = 0; b < N; ++b) {
            const void* d_xb = (const char*)d_x + (size_t)b * Cin * Hi * Wi;
            void*       d_yb =       (char*)d_y + (size_t)b * Cout * Ho * Wo;

            if (use_implicit) {
                // Implicit GEMM: the B_tile is computed from X inline. No
                // im2col launch, no d_col buffer. One launch per conv.
                void* args[] = {
                    (void*)&d_w, (void*)&d_xb, (void*)&d_yb,
                    (void*)&d_row_sum_W,
                    (void*)&d_w_zp_eff_arr, (void*)&d_combined_scale,
                    (void*)&d_bias,
                    &_Cout, &_Cin, &_Hi, &_Wi, &_Ho, &_Wo,
                    &_kH, &_kW, &_sH, &_sW, &_pT, &_pL, &_dH, &_dW,
                    &_xshift, &_xzp, &_yzp, &_hb,
                };
                unsigned grid_x = (unsigned)((M_gemm + 63) / 64);
                unsigned grid_y = (unsigned)((Cout   + 63) / 64);
                if (!gpu::nvrtc_launch(be->device, f_implicit,
                                       grid_x, grid_y, 1, 128, 1, 1, args))
                    return fallback->exec();
            } else {
                // Fallback: explicit im2col into d_col, then fused GEMM.
                {
                    void* args[] = {
                        (void*)&d_xb, (void*)&d_col,
                        &_xshift, &_xzp,
                        &_Cin, &_Hi, &_Wi, &_Ho, &_Wo,
                        &_kH, &_kW, &_sH, &_sW, &_pT, &_pL, &_dH, &_dW,
                    };
                    unsigned long long total = (unsigned long long)K_gemm * M_gemm;
                    unsigned block = 256;
                    unsigned grid  = (unsigned)((total + block - 1) / block);
                    if (!gpu::nvrtc_launch(be->device, f_im2col, grid, 1, 1, block, 1, 1, args))
                        return fallback->exec();
                }
                {
                    void* args[] = {
                        (void*)&d_w, (void*)&d_col, (void*)&d_yb,
                        (void*)&d_row_sum_W,
                        (void*)&d_w_zp_eff_arr, (void*)&d_combined_scale,
                        (void*)&d_bias,
                        &_Cout, &_M, &_Kt, &_xzp, &_Kt, &_yzp, &_hb,
                    };
                    unsigned grid_x = (unsigned)((M_gemm + 63) / 64);
                    unsigned grid_y = (unsigned)((Cout   + 63) / 64);
                    if (!gpu::nvrtc_launch(be->device, f_fused,
                                           grid_x, grid_y, 1, 128, 1, 1, args))
                        return fallback->exec();
                }
            }
        }

        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_QLinearConv(int opset, pool_t& pool) {
    return pool_new<QLinearConv_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
