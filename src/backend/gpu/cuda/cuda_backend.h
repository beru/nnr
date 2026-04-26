#pragma once
// Per-context CUDA backend state. Stored in context_t::backends[CUDA].data.
//
// Owns:
//   - cuda_device_t*        — device handle (streams, mem pool)
//   - gpu::gpu_cache_t      — per-tensor device buffer registry (upload/alloc/writeback)
//   - nvrtc_cache_t         — runtime-compiled kernel cache (shared across ops)
//   - cudaGraphExec_t       — optional captured exec-graph for the GPU subsequence
//     (nullptr until first capture; caller manages invalidation by shape)
//
// Created lazily on first CUDA op exec(). Freed via the free_fn set on
// context_t::backend_state_t.

#if defined(NNR_USE_CUDA)

#include "cuda_device.h"
#include "nvrtc.h"
#include "../gpu_cache.h"

namespace nnr {
struct context_t;
}

namespace nnr::gpu {

// Launch C = A * B via NVRTC WMMA/scalar kernels (no cuBLAS).
// A: (M, K) row-major if transA=0; (K, M) if transA=1.
// B: (K, N) row-major if transB=0; (N, K) if transB=1.
// C: (M, N) row-major.
// Uses TF32 tensor cores when transA=0, transB=0, and M%16==0, N%16==0, K%8==0;
// otherwise a scalar CUDA-core kernel. Returns false on launch error.
bool gemm_device_f32(struct cuda_backend_t* be,
                     const float* A, const float* B, float* C,
                     int M, int N, int K,
                     int transA = 0, int transB = 0);

// Y[i] = alpha * Y[i] + beta * bias[...]. Launched after a GEMM. `bias_kind`:
// 0=none, 1=elementwise, 2=row-broadcast ((N,)), 3=col-broadcast ((M,)).
bool gemm_epilogue_f32(struct cuda_backend_t* be,
                       float* Y, const float* bias,
                       int M, int N, float alpha, float beta, int bias_kind);

// Int8 WMMA GEMM: C[M,N] = A[M,K] @ B[K,N], int32 accumulator, row-major.
// `is_signed` picks the signed-char vs unsigned-char WMMA kernel. No transpose
// variants; QLinearConv's im2col already produces the expected layout. Returns
// false on launch error. Falls through silently to caller for CPU fallback.
bool gemm_device_s8s32(struct cuda_backend_t* be,
                       const void* A, const void* B, int* C,
                       int M, int N, int K, bool is_signed);

struct cuda_backend_t {
    cuda_device_t*  device = nullptr;
    gpu_cache_t*    cache  = nullptr;
    nvrtc_cache_t   nvrtc;
    cudaGraphExec_t graph_exec = nullptr;   // captured ops (nullptr until capture)

    // Execution mode for CUDA Graph capture/replay lifecycle.
    //   normal     — op-by-op launches on compute_stream (no capture active)
    //   capturing  — compute_stream is in capture mode; launches record into graph
    //   replaying  — the exec loop is skipped; launch_graph(graph_exec) is used
    enum class exec_mode : uint8_t { normal, capturing, replaying };
    exec_mode mode = exec_mode::normal;
    int       run_count = 0;     // number of successful runs (used to pick eligibility)

    cuda_backend_t() = default;

    ~cuda_backend_t() {
        if (graph_exec && device) device->destroy_graph(graph_exec);
        delete cache;       // frees device buffers first
        delete device;      // then destroys streams, mem pool
    }

    cuda_backend_t(const cuda_backend_t&) = delete;
    cuda_backend_t& operator=(const cuda_backend_t&) = delete;

    // Called once per run_graph at start (after reshape, before exec loop).
    //   all_ops_gpu  — whether every non-skip/non-folded op is device_tag==CUDA
    //   shapes_dirty — the graph must be rebuilt; invalidates any existing graph_exec
    // Returns the mode the caller should operate in.
    //   normal     — run ops one at a time (first run, CPU ops present, or ineligible)
    //   capturing  — run ops one at a time on compute_stream; graph captured in background
    //   replaying  — skip exec loop; caller will call replay() + device->sync()
    exec_mode plan_run(bool all_ops_gpu, bool shapes_dirty);

    // Called once per run_graph at end (after exec loop, but only if plan_run
    // returned `capturing`). Ends capture, instantiates, stores graph_exec.
    void finalize_capture();

    // Launch the previously-instantiated graph on compute_stream. Non-blocking;
    // caller should device->sync() before reading outputs (or writeback handles it).
    bool replay();

    // Invalidate any captured graph (frees graph_exec). Call when the memory plan
    // changes or shapes change such that captured pointers are no longer valid.
    void invalidate_graph();
};

// Helper used by every CUDA op's exec() to lazily create and retrieve the
// per-context backend state. Returns nullptr on device creation failure —
// callers should fall back to CPU in that case.
//
// On success, the state is stored in ctx->backends[CUDA].data and freed
// automatically by ~context_t.
struct context_t_fwd;  // avoid pulling nnr.h here — implementation casts

} // namespace nnr::gpu

#endif // NNR_USE_CUDA
