#if defined(NNR_USE_CUDA)

// TopK (last-axis, small K) via NVRTC.
// Thread per row — each maintains a locally-sorted top-K array on stack.
// K <= 32 accelerated; larger falls back.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "attr_key.h"

namespace nnr {

operator_t* resolver_default_op_TopK(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static constexpr int TOPK_MAX        = 32;     // register-resident fast path
static constexpr int TOPK_LARGE_MAX  = 4096;   // shared-mem block kernel

static const char* topk_source() {
    return R"CUDA(
#ifndef INFINITY
#define INFINITY __int_as_float(0x7F800000)
#endif
#define K_MAX 32
extern "C" __global__
void topk_lastaxis_f32(const float* __restrict__ x,
                       float* __restrict__ vals,
                       long long* __restrict__ idxs,
                       int outer,
                       int N,
                       int K,
                       int largest)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outer) return;
    const float* xr = x + (size_t)row * N;

    float b_v[K_MAX];
    int   b_i[K_MAX];
    float initval = largest ? -INFINITY : INFINITY;
    #pragma unroll
    for (int i = 0; i < K_MAX; ++i) { b_v[i] = initval; b_i[i] = 0; }

    if (largest) {
        for (int j = 0; j < N; ++j) {
            float v = xr[j];
            if (v > b_v[K-1]) {
                int p = K - 1;
                while (p > 0 && b_v[p-1] < v) {
                    b_v[p] = b_v[p-1]; b_i[p] = b_i[p-1]; --p;
                }
                b_v[p] = v; b_i[p] = j;
            }
        }
    } else {
        for (int j = 0; j < N; ++j) {
            float v = xr[j];
            if (v < b_v[K-1]) {
                int p = K - 1;
                while (p > 0 && b_v[p-1] > v) {
                    b_v[p] = b_v[p-1]; b_i[p] = b_i[p-1]; --p;
                }
                b_v[p] = v; b_i[p] = j;
            }
        }
    }

    for (int i = 0; i < K; ++i) {
        vals[(size_t)row * K + i] = b_v[i];
        idxs[(size_t)row * K + i] = (long long)b_i[i];
    }
}

#define I64_MIN ((long long)0x8000000000000000LL)
#define I64_MAX ((long long)0x7fffffffffffffffLL)

extern "C" __global__
void topk_lastaxis_i64(const long long* __restrict__ x,
                       long long* __restrict__ vals,
                       long long* __restrict__ idxs,
                       int outer,
                       int N,
                       int K,
                       int largest)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outer) return;
    const long long* xr = x + (size_t)row * N;

    long long b_v[K_MAX];
    int       b_i[K_MAX];
    long long initval = largest ? I64_MIN : I64_MAX;
    #pragma unroll
    for (int i = 0; i < K_MAX; ++i) { b_v[i] = initval; b_i[i] = 0; }

    if (largest) {
        for (int j = 0; j < N; ++j) {
            long long v = xr[j];
            if (v > b_v[K-1]) {
                int p = K - 1;
                while (p > 0 && b_v[p-1] < v) {
                    b_v[p] = b_v[p-1]; b_i[p] = b_i[p-1]; --p;
                }
                b_v[p] = v; b_i[p] = j;
            }
        }
    } else {
        for (int j = 0; j < N; ++j) {
            long long v = xr[j];
            if (v < b_v[K-1]) {
                int p = K - 1;
                while (p > 0 && b_v[p-1] > v) {
                    b_v[p] = b_v[p-1]; b_i[p] = b_i[p-1]; --p;
                }
                b_v[p] = v; b_i[p] = j;
            }
        }
    }

    for (int i = 0; i < K; ++i) {
        vals[(size_t)row * K + i] = b_v[i];
        idxs[(size_t)row * K + i] = (long long)b_i[i];
    }
}

// ---------- Block-level TopK for arbitrary K (K > K_MAX) ----------
// P-batched two-stage. One block per row. Each pass produces P winners:
//   1. Each thread keeps the top-P of its strided slice in registers (insertion
//      sort, skipping bitmap-marked indices).
//   2. Per-thread top-P written to shared mem; pair-wise tree merge of sorted
//      P-lists across log2(nth) levels.
//   3. Threads 0..p-1 emit one winner each into top_v/top_i and mark the
//      bitmap. Loop until K winners produced.
// Correctness: per-thread top-P merged across all threads ⊇ global top-P.
// Shared mem: top_v[K] | top_i[K] | bm[(N+31)/32] | mrg_v[nth*P] | mrg_i[nth*P].
#define P_BATCH 32

extern "C" __global__
void topk_lastaxis_large_f32(const float* __restrict__ x,
                             float* __restrict__ vals,
                             long long* __restrict__ idxs,
                             int outer, int N, int K, int largest)
{
    int row = blockIdx.x;
    if (row >= outer) return;
    int tid = threadIdx.x;
    int nth = blockDim.x;
    const float* xr = x + (size_t)row * N;

    extern __shared__ unsigned char shmem[];
    float*        top_v = (float*)shmem;
    int*          top_i = (int*)(top_v + K);
    unsigned int* bm    = (unsigned int*)(top_i + K);
    int bm_words = (N + 31) / 32;
    float* mrg_v = (float*)(bm + bm_words);
    int*   mrg_i = (int*)(mrg_v + (size_t)nth * P_BATCH);

    for (int i = tid; i < bm_words; i += nth) bm[i] = 0u;
    __syncthreads();

    float init_v = largest ? -INFINITY : INFINITY;

    int produced = 0;
    while (produced < K) {
        int p_this = (K - produced < P_BATCH) ? (K - produced) : P_BATCH;

        // Phase 1: per-thread top-P over strided slice (descending if largest).
        float lv[P_BATCH];
        int   li[P_BATCH];
        #pragma unroll
        for (int i = 0; i < P_BATCH; ++i) { lv[i] = init_v; li[i] = -1; }

        for (int j = tid; j < N; j += nth) {
            if ((bm[j >> 5] >> (j & 31)) & 1u) continue;
            float v = xr[j];
            // Compare against worst slot lv[P-1]. Skip if not better.
            bool insert;
            if (li[P_BATCH-1] < 0)   insert = true;
            else if (largest)        insert = (v > lv[P_BATCH-1]) || (v == lv[P_BATCH-1] && j < li[P_BATCH-1]);
            else                     insert = (v < lv[P_BATCH-1]) || (v == lv[P_BATCH-1] && j < li[P_BATCH-1]);
            if (!insert) continue;
            int p = P_BATCH - 1;
            while (p > 0) {
                bool prev_loses;
                if (li[p-1] < 0) prev_loses = true;
                else if (largest) prev_loses = (lv[p-1] < v) || (lv[p-1] == v && li[p-1] > j);
                else              prev_loses = (lv[p-1] > v) || (lv[p-1] == v && li[p-1] > j);
                if (!prev_loses) break;
                lv[p] = lv[p-1]; li[p] = li[p-1];
                --p;
            }
            lv[p] = v; li[p] = j;
        }

        // Phase 2: dump per-thread top-P into shared mem.
        #pragma unroll
        for (int i = 0; i < P_BATCH; ++i) {
            mrg_v[(size_t)tid * P_BATCH + i] = lv[i];
            mrg_i[(size_t)tid * P_BATCH + i] = li[i];
        }
        __syncthreads();

        // Phase 3: pair-wise tree merge of sorted P-lists, log2(nth) levels.
        for (int off = nth >> 1; off > 0; off >>= 1) {
            float a_v_reg[P_BATCH], b_v_reg[P_BATCH];
            int   a_i_reg[P_BATCH], b_i_reg[P_BATCH];
            if (tid < off) {
                int a_base = tid * P_BATCH;
                int b_base = (tid + off) * P_BATCH;
                #pragma unroll
                for (int i = 0; i < P_BATCH; ++i) {
                    a_v_reg[i] = mrg_v[a_base + i];
                    a_i_reg[i] = mrg_i[a_base + i];
                    b_v_reg[i] = mrg_v[b_base + i];
                    b_i_reg[i] = mrg_i[b_base + i];
                }
                int ai = 0, bi = 0;
                #pragma unroll
                for (int o = 0; o < P_BATCH; ++o) {
                    bool take_a;
                    if (ai >= P_BATCH)            take_a = false;
                    else if (bi >= P_BATCH)       take_a = true;
                    else if (a_i_reg[ai] < 0)     take_a = false;
                    else if (b_i_reg[bi] < 0)     take_a = true;
                    else if (largest)             take_a = (a_v_reg[ai] > b_v_reg[bi]) || (a_v_reg[ai] == b_v_reg[bi] && a_i_reg[ai] < b_i_reg[bi]);
                    else                          take_a = (a_v_reg[ai] < b_v_reg[bi]) || (a_v_reg[ai] == b_v_reg[bi] && a_i_reg[ai] < b_i_reg[bi]);
                    if (take_a) {
                        mrg_v[a_base + o] = a_v_reg[ai];
                        mrg_i[a_base + o] = a_i_reg[ai];
                        ++ai;
                    } else {
                        mrg_v[a_base + o] = b_v_reg[bi];
                        mrg_i[a_base + o] = b_i_reg[bi];
                        ++bi;
                    }
                }
            }
            __syncthreads();
        }

        // Phase 4: emit p_this winners and update bitmap.
        if (tid < p_this) {
            int slot = produced + tid;
            int win_i = mrg_i[tid];
            top_v[slot] = mrg_v[tid];
            top_i[slot] = (win_i < 0) ? 0 : win_i;
            if (win_i >= 0) atomicOr(&bm[win_i >> 5], 1u << (win_i & 31));
        }
        __syncthreads();
        produced += p_this;
    }

    for (int i = tid; i < K; i += nth) {
        vals[(size_t)row * K + i] = top_v[i];
        idxs[(size_t)row * K + i] = (long long)top_i[i];
    }
}

extern "C" __global__
void topk_lastaxis_large_i64(const long long* __restrict__ x,
                             long long* __restrict__ vals,
                             long long* __restrict__ idxs,
                             int outer, int N, int K, int largest)
{
    int row = blockIdx.x;
    if (row >= outer) return;
    int tid = threadIdx.x;
    int nth = blockDim.x;
    const long long* xr = x + (size_t)row * N;

    extern __shared__ unsigned char shmem[];
    long long*    top_v = (long long*)shmem;
    int*          top_i = (int*)(top_v + K);
    unsigned int* bm    = (unsigned int*)(top_i + K);
    int bm_words = (N + 31) / 32;
    long long* mrg_v = (long long*)(bm + bm_words);
    int*       mrg_i = (int*)(mrg_v + (size_t)nth * P_BATCH);

    for (int i = tid; i < bm_words; i += nth) bm[i] = 0u;
    __syncthreads();

    long long init_v = largest ? I64_MIN : I64_MAX;

    int produced = 0;
    while (produced < K) {
        int p_this = (K - produced < P_BATCH) ? (K - produced) : P_BATCH;

        long long lv[P_BATCH];
        int       li[P_BATCH];
        #pragma unroll
        for (int i = 0; i < P_BATCH; ++i) { lv[i] = init_v; li[i] = -1; }

        for (int j = tid; j < N; j += nth) {
            if ((bm[j >> 5] >> (j & 31)) & 1u) continue;
            long long v = xr[j];
            bool insert;
            if (li[P_BATCH-1] < 0)   insert = true;
            else if (largest)        insert = (v > lv[P_BATCH-1]) || (v == lv[P_BATCH-1] && j < li[P_BATCH-1]);
            else                     insert = (v < lv[P_BATCH-1]) || (v == lv[P_BATCH-1] && j < li[P_BATCH-1]);
            if (!insert) continue;
            int p = P_BATCH - 1;
            while (p > 0) {
                bool prev_loses;
                if (li[p-1] < 0) prev_loses = true;
                else if (largest) prev_loses = (lv[p-1] < v) || (lv[p-1] == v && li[p-1] > j);
                else              prev_loses = (lv[p-1] > v) || (lv[p-1] == v && li[p-1] > j);
                if (!prev_loses) break;
                lv[p] = lv[p-1]; li[p] = li[p-1];
                --p;
            }
            lv[p] = v; li[p] = j;
        }

        #pragma unroll
        for (int i = 0; i < P_BATCH; ++i) {
            mrg_v[(size_t)tid * P_BATCH + i] = lv[i];
            mrg_i[(size_t)tid * P_BATCH + i] = li[i];
        }
        __syncthreads();

        for (int off = nth >> 1; off > 0; off >>= 1) {
            long long a_v_reg[P_BATCH], b_v_reg[P_BATCH];
            int       a_i_reg[P_BATCH], b_i_reg[P_BATCH];
            if (tid < off) {
                int a_base = tid * P_BATCH;
                int b_base = (tid + off) * P_BATCH;
                #pragma unroll
                for (int i = 0; i < P_BATCH; ++i) {
                    a_v_reg[i] = mrg_v[a_base + i];
                    a_i_reg[i] = mrg_i[a_base + i];
                    b_v_reg[i] = mrg_v[b_base + i];
                    b_i_reg[i] = mrg_i[b_base + i];
                }
                int ai = 0, bi = 0;
                #pragma unroll
                for (int o = 0; o < P_BATCH; ++o) {
                    bool take_a;
                    if (ai >= P_BATCH)            take_a = false;
                    else if (bi >= P_BATCH)       take_a = true;
                    else if (a_i_reg[ai] < 0)     take_a = false;
                    else if (b_i_reg[bi] < 0)     take_a = true;
                    else if (largest)             take_a = (a_v_reg[ai] > b_v_reg[bi]) || (a_v_reg[ai] == b_v_reg[bi] && a_i_reg[ai] < b_i_reg[bi]);
                    else                          take_a = (a_v_reg[ai] < b_v_reg[bi]) || (a_v_reg[ai] == b_v_reg[bi] && a_i_reg[ai] < b_i_reg[bi]);
                    if (take_a) {
                        mrg_v[a_base + o] = a_v_reg[ai];
                        mrg_i[a_base + o] = a_i_reg[ai];
                        ++ai;
                    } else {
                        mrg_v[a_base + o] = b_v_reg[bi];
                        mrg_i[a_base + o] = b_i_reg[bi];
                        ++bi;
                    }
                }
            }
            __syncthreads();
        }

        if (tid < p_this) {
            int slot = produced + tid;
            int win_i = mrg_i[tid];
            top_v[slot] = mrg_v[tid];
            top_i[slot] = (win_i < 0) ? 0 : win_i;
            if (win_i >= 0) atomicOr(&bm[win_i >> 5], 1u << (win_i & 31));
        }
        __syncthreads();
        produced += p_this;
    }

    for (int i = tid; i < K; i += nth) {
        vals[(size_t)row * K + i] = top_v[i];
        idxs[(size_t)row * K + i] = (long long)top_i[i];
    }
}
)CUDA";
}

struct TopK_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int axis_attr = -1;
    int largest = 1;
    int outer = 0, N = 0, K = 0;

    bool init() override {
        if (inputs.size() < 1 || outputs.size() < 2) return false;
        fallback = resolver_default_op_TopK(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        axis_attr = (int)attribute(attr_key_t::axis, (int64_t)-1);
        largest   = (int)attribute(attr_key_t::largest, (int64_t)1);
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* x = inputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32 && x->type != NNR_DATA_TYPE_INT64) return true;
        if (outputs[0]->type != x->type) return true;
        if (outputs[1]->type != NNR_DATA_TYPE_INT64) return true;

        int caxis = axis_attr < 0 ? axis_attr + x->ndim : axis_attr;
        if (caxis != x->ndim - 1) return true;   // last axis only

        // K from input[1] (opset 10+)
        if (inputs.size() < 2 || !inputs[1] || inputs[1]->type != NNR_DATA_TYPE_INT64
            || !inputs[1]->data || inputs[1]->ndata != 1) return true;
        int64_t kv = *(const int64_t*)inputs[1]->data;
        if (kv <= 0 || kv > TOPK_LARGE_MAX) return true;
        K = (int)kv;
        N = x->dims[caxis];
        outer = 1; for (int d = 0; d < caxis; ++d) outer *= x->dims[d];

        // Large-K shared-mem budget (must match exec() launch shared bytes).
        // Both f32 and i64 use the P-batched layout:
        //   K*(elem+4) + ((N+31)/32)*4 + nth*P*(elem+4)            (nth=128, P=32)
        // Reject if it exceeds 46 KB (RTX 3090 default static smem cap is 48 KB).
        if (K > TOPK_MAX) {
            int elem = (x->type == NNR_DATA_TYPE_INT64) ? 8 : 4;
            int nth = 128, P = 32;
            long long need = (long long)K * (elem + 4)
                           + (long long)((N + 31) / 32) * 4
                           + (long long)nth * P * (elem + 4);
            if (need > 46 * 1024) return true;
        }

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }

        const bool large = (K > TOPK_MAX);
        const bool i64   = (inputs[0]->type == NNR_DATA_TYPE_INT64);
        const char* kn;
        if (large) kn = i64 ? "topk_lastaxis_large_i64" : "topk_lastaxis_large_f32";
        else       kn = i64 ? "topk_lastaxis_i64"       : "topk_lastaxis_f32";
        CUfunction f = be->nvrtc.get("nnr_topk",
                                     topk_source(),
                                     kn,
                                     gpu::nvrtc_arch_option(be->device));
        if (!f) { return fallback->exec(); }

        void*      d_x    = be->cache->ensure_device(inputs[0]);
        void*      d_vals = be->cache->alloc_output (outputs[0]);
        long long* d_idxs = (long long*)be->cache->alloc_output (outputs[1]);
        if (!d_x || !d_vals || !d_idxs) { return fallback->exec(); }

        int _o = outer, _N = N, _K = K, _lg = largest;
        void* args[] = { &d_x, &d_vals, &d_idxs, &_o, &_N, &_K, &_lg };

        if (!large) {
            unsigned block = 128;
            unsigned grid = (unsigned)((outer + block - 1) / block);
            if (!gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args)) {
                return fallback->exec();
            }
        } else {
            // Block-per-row. Shared-memory layout must match the kernel
            // (P-batched two-stage; f32 and i64 share the layout shape).
            unsigned nth = 128;
            int P = 32;
            int elem = i64 ? 8 : 4;
            unsigned shared = (unsigned)((long long)K * (elem + 4)
                                       + (long long)((N + 31) / 32) * 4
                                       + (long long)nth * P * (elem + 4));
            if (!gpu::nvrtc_launch(be->device, f, (unsigned)outer, 1, 1, nth, 1, 1, args, shared)) {
                return fallback->exec();
            }
        }
        be->cache->mark_written(outputs[0]);
        be->cache->mark_written(outputs[1]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_TopK(int opset, pool_t& pool) {
    return pool_new<TopK_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
