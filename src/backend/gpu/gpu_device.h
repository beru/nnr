#pragma once
// GPU device abstraction — one instance per physical GPU.
//
// Virtual interface: one per context, called per-operator (not per-element).
// Virtual dispatch overhead is negligible vs kernel launch latency.
//
// Backend implementations: cuda/cuda_device.h, hip/hip_device.h,
// vulkan/vk_device.h, webgpu/wgpu_device.h

#include "gpu_common.h"

namespace nnr::gpu {

// Completion marker on a stream. Other streams can wait on it.
struct gpu_event_t {
    virtual ~gpu_event_t() = default;
    virtual void wait() = 0;        // host blocks until event fires
    virtual bool query() = 0;       // non-blocking poll (true = complete)
};

struct gpu_device_t {
    virtual ~gpu_device_t() = default;

    // --- Identity ---
    virtual gpu_backend_t kind() const = 0;
    virtual int device_id() const = 0;
    virtual const char* name() const = 0;
    virtual bool is_uma() const = 0;

    // --- Streams & synchronization ---
    //
    // Each device owns internal streams:
    //   compute  — kernel launches, vendor library calls
    //   h2d      — host-to-device async memcpy
    //   d2h      — device-to-host async memcpy
    //
    // record_*_event() captures the current point on a stream.
    // compute_wait(evt) makes the compute stream wait for a transfer event.
    // transfer_wait(evt) makes the d2h stream wait for a compute event.
    // sync() blocks the host until all streams drain.

    virtual gpu_event_t* record_compute_event() = 0;
    virtual gpu_event_t* record_transfer_event() = 0;
    virtual void compute_wait(gpu_event_t* evt) = 0;
    virtual void transfer_wait(gpu_event_t* evt) = 0;
    virtual void sync() = 0;

    // --- Device memory pool ---
    //
    // Sub-allocates from a device memory pool (cudaMemPool_t / VMA / etc).
    // alloc() is async-safe — no implicit synchronization.

    virtual void* alloc(size_t bytes) = 0;
    virtual void  free(void* ptr) = 0;

    // --- Async transfers ---
    //
    // Enqueued on h2d / d2h transfer streams. Return event for ordering.
    // IMPORTANT: Host memory must be pinned for true async DMA.
    // Pageable host memory with cudaMemcpyAsync silently falls back to sync.

    virtual gpu_event_t* copy_h2d_async(void* dst_dev, const void* src_host, size_t bytes) = 0;
    virtual gpu_event_t* copy_d2h_async(void* dst_host, const void* src_dev, size_t bytes) = 0;
    virtual void copy_d2d(void* dst, const void* src, size_t bytes) = 0;  // on compute stream

    // --- Pinned host memory ---
    //
    // Page-locked host memory enables true async DMA transfers.
    // alloc_pinned() allocates new pinned memory.
    // pin()/unpin() register existing host allocations for DMA.

    virtual void* alloc_pinned(size_t bytes) = 0;
    virtual void  free_pinned(void* ptr) = 0;
    virtual bool  pin(void* ptr, size_t bytes) { return false; }
    virtual void  unpin(void* ptr) {}

    // --- Unified/managed memory ---
    //
    // Optional. Returns pointer accessible from both CPU and GPU.
    // Default on UMA devices (Jetson, DGX Spark, APU).
    // CUDA: cudaMallocManaged. HIP: hipMallocManaged.
    // Vulkan/WebGPU: not supported (returns nullptr).

    virtual void* alloc_managed(size_t bytes) { return nullptr; }
    virtual void  free_managed(void* ptr) {}

    // --- Peer-to-peer (multi-device, future) ---
    //
    // Direct GPU-to-GPU transfer. Falls back to D2H + H2D if not supported.
    // Returns nullptr if P2P is not available.

    virtual gpu_event_t* copy_p2p_async(void* dst_dev, gpu_device_t* dst_device,
                                        const void* src_dev, size_t bytes)
    { return nullptr; }

    // --- Runtime compilation ---
    //
    // Compile kernel source to device binary. Cached by (source, options) key.
    // CUDA: Jitify (wraps NVRTC). HIP: hipRTC. Vulkan: shaderc. WebGPU: native WGSL.
    // Returns opaque handle passed to launch_kernel().

    virtual void* compile_kernel(const char* source, const char* name,
                                 const char* options = nullptr) { return nullptr; }
    virtual void  launch_kernel(void* kernel, const int grid[3], const int block[3],
                                void** args, size_t shared_mem = 0) {}

    // --- Profiling ---
    //
    // GPU-side timing via events. No host synchronization during recording.
    // Resolve elapsed time after graph execution in a single sync pass.
    // CUDA: cudaEventRecord + cudaEventElapsedTime (~0.5 us resolution).

    virtual gpu_event_t* record_profile_start() = 0;
    virtual gpu_event_t* record_profile_stop() = 0;
    virtual float event_elapsed_us(gpu_event_t* start, gpu_event_t* stop) = 0;
};

// Factory: creates the correct device for the compiled backend.
// Returns nullptr if the backend is unavailable (no GPU, driver issue).
gpu_device_t* create_device(gpu_backend_t backend, int device_id = 0);

} // namespace nnr::gpu
