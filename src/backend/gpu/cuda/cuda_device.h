#pragma once
// CUDA device implementation — wraps the CUDA runtime + driver APIs. No cuBLAS,
// no cuDNN. All kernels (including GEMM via WMMA TensorCores) are compiled at
// runtime by NVRTC — see nvrtc.h and gemm_wmma.h.
//
// Owns:
//   3 cudaStream_t (compute, h2d, d2h)
//   cudaMemPool_t (stream-ordered alloc/free)
//
// Event pool recycles cudaEvent_t objects to avoid create/destroy overhead.

#if defined(NNR_USE_CUDA)

#include "../gpu_device.h"

#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <cstring>

namespace nnr::gpu {

// --- CUDA event wrapper ---

struct cuda_event_t : gpu_event_t {
    cudaEvent_t evt = nullptr;

    explicit cuda_event_t(cudaEvent_t e) : evt(e) {}

    void wait() override {
        cudaEventSynchronize(evt);
    }

    bool query() override {
        return cudaEventQuery(evt) == cudaSuccess;
    }
};

// --- CUDA device ---

struct cuda_device_t : gpu_device_t {

    // Construction / destruction
    explicit cuda_device_t(int id);
    ~cuda_device_t() override;

    cuda_device_t(const cuda_device_t&) = delete;
    cuda_device_t& operator=(const cuda_device_t&) = delete;

    bool valid() const { return valid_; }

    // --- Identity ---
    gpu_backend_t kind() const override { return gpu_backend_t::CUDA; }
    int device_id() const override { return device_id_; }
    const char* name() const override { return name_; }
    bool is_uma() const override { return uma_; }

    // --- Streams & sync ---
    gpu_event_t* record_compute_event() override;
    gpu_event_t* record_transfer_event() override;
    void compute_wait(gpu_event_t* evt) override;
    void transfer_wait(gpu_event_t* evt) override;
    void sync() override;

    // --- Memory pool ---
    void* alloc(size_t bytes) override;
    void  free(void* ptr) override;

    // --- Async transfers ---
    gpu_event_t* copy_h2d_async(void* dst_dev, const void* src_host, size_t bytes) override;
    gpu_event_t* copy_d2h_async(void* dst_host, const void* src_dev, size_t bytes) override;
    void copy_d2d(void* dst, const void* src, size_t bytes) override;

    // --- Pinned host memory ---
    void* alloc_pinned(size_t bytes) override;
    void  free_pinned(void* ptr) override;
    bool  pin(void* ptr, size_t bytes) override;
    void  unpin(void* ptr) override;

    // --- Unified/managed memory ---
    void* alloc_managed(size_t bytes) override;
    void  free_managed(void* ptr) override;

    // --- Runtime compilation ---
    // Raw NVRTC is used directly; see nvrtc.h. These stubs remain for the
    // virtual gpu_device_t interface.
    void* compile_kernel(const char* source, const char* name,
                         const char* options = nullptr) override;
    void  launch_kernel(void* kernel, const int grid[3], const int block[3],
                        void** args, size_t shared_mem = 0) override;

    // --- Profiling ---
    gpu_event_t* record_profile_start() override;
    gpu_event_t* record_profile_stop() override;
    float event_elapsed_us(gpu_event_t* start, gpu_event_t* stop) override;

    // --- Direct stream access (for vendor library calls) ---
    cudaStream_t compute_stream() const { return compute_stream_; }

    // --- CUDA Graph capture / replay ---
    // Capture a sequence of compute-stream launches into a replayable graph.
    // Typical use (per-context, after shapes stabilize):
    //   dev->begin_capture();
    //   ... run all CUDA ops on compute_stream ...
    //   auto exec = dev->end_capture();
    //   ... on subsequent runs with same shapes: dev->launch_graph(exec)
    //
    // Capture uses cudaStreamCaptureModeThreadLocal so concurrent host threads
    // are not blocked. Allocations during capture MUST use cudaMallocAsync on
    // compute_stream_ (alloc() below already does this).
    //
    // Returns false on error (capture not started, invalid state, etc).
    bool begin_capture();
    // Ends capture and instantiates. Returns the executable graph (owned by caller;
    // destroy with destroy_graph()). Returns nullptr on error.
    cudaGraphExec_t end_capture();
    // Replay a previously instantiated graph on compute_stream_. Non-blocking.
    bool launch_graph(cudaGraphExec_t exec);
    // Destroy an instantiated graph (releases CUDA resources).
    void destroy_graph(cudaGraphExec_t exec);

private:
    int             device_id_ = 0;
    char            name_[256] = {};
    bool            valid_ = false;
    bool            uma_ = false;

    cudaStream_t    compute_stream_ = nullptr;
    cudaStream_t    h2d_stream_     = nullptr;
    cudaStream_t    d2h_stream_     = nullptr;

    cudaMemPool_t   mem_pool_       = nullptr;

    // Event pool: recycle events to avoid create/destroy overhead.
    std::vector<cudaEvent_t> event_pool_;
    size_t event_pool_idx_ = 0;
    cuda_event_t* acquire_event();
};

// Factory — called by gpu_cache.cpp via create_device().
gpu_device_t* create_cuda_device(int id);

// Enumeration — called by gpu_cache.cpp via enumerate_devices().
int enumerate_cuda_devices(device_info_t* out, int max_count);

} // namespace nnr::gpu

#endif // NNR_USE_CUDA
