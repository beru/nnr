#if defined(NNR_USE_CUDA)

#include "cuda_device.h"
#include <cuda.h>      // driver API — needed for cuInit / primary context
#include <cstdio>

#define CUDA_CHECK(call) do {                                              \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                __FILE__, __LINE__, cudaGetErrorString(err));              \
        return;                                                            \
    }                                                                      \
} while(0)

#define CUDA_CHECK_RET(call, ret) do {                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                __FILE__, __LINE__, cudaGetErrorString(err));              \
        return (ret);                                                      \
    }                                                                      \
} while(0)

namespace nnr::gpu {

// --- Construction / destruction ---

cuda_device_t::cuda_device_t(int id) : device_id_(id) {
    // Set device context for this thread
    if (cudaSetDevice(id) != cudaSuccess)
        return;

    // Force CUDA runtime to realize the primary context — cudaFree(0) triggers
    // lazy context init and pushes the primary as current on this thread (both
    // runtime and driver views). Needed before cuLaunchKernel.
    cudaFree(0);

    // Query device properties
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, id) != cudaSuccess)
        return;
    std::strncpy(name_, prop.name, sizeof(name_) - 1);

    // Detect UMA: integrated GPU (Jetson, Tegra), or Grace Blackwell (NVLink-C2C).
    int integrated = 0;
    cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, id);
    if (integrated) {
        uma_ = true;
    } else {
        // Grace Blackwell: not "integrated" but has coherent pageable memory access
        int pageable = 0, host_atomic = 0;
        cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, id);
        cudaDeviceGetAttribute(&host_atomic, cudaDevAttrHostNativeAtomicSupported, id);
        uma_ = (pageable && host_atomic);
    }

    // Create streams (non-blocking — no implicit sync with NULL stream)
    if (cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking) != cudaSuccess)
        return;
    if (cudaStreamCreateWithFlags(&h2d_stream_, cudaStreamNonBlocking) != cudaSuccess)
        return;
    if (cudaStreamCreateWithFlags(&d2h_stream_, cudaStreamNonBlocking) != cudaSuccess)
        return;

    // Get default memory pool and configure release threshold
    if (cudaDeviceGetDefaultMemPool(&mem_pool_, id) != cudaSuccess)
        return;
    uint64_t threshold = UINT64_MAX;  // keep memory warm, don't return to OS
    cudaMemPoolSetAttribute(mem_pool_, cudaMemPoolAttrReleaseThreshold, &threshold);

    // Pre-allocate event pool (64 events — enough for one inference pass)
    event_pool_.resize(64);
    for (auto& e : event_pool_)
        cudaEventCreateWithFlags(&e, cudaEventDisableTiming);

    valid_ = true;
}

cuda_device_t::~cuda_device_t() {
    if (!valid_) return;

    cudaSetDevice(device_id_);

    for (auto& e : event_pool_)
        if (e) cudaEventDestroy(e);

    if (compute_stream_) cudaStreamDestroy(compute_stream_);
    if (h2d_stream_)     cudaStreamDestroy(h2d_stream_);
    if (d2h_stream_)     cudaStreamDestroy(d2h_stream_);
}

// --- Event pool ---

cuda_event_t* cuda_device_t::acquire_event() {
    if (event_pool_idx_ >= event_pool_.size()) {
        // Grow pool
        size_t old_size = event_pool_.size();
        event_pool_.resize(old_size * 2);
        for (size_t i = old_size; i < event_pool_.size(); i++)
            cudaEventCreateWithFlags(&event_pool_[i], cudaEventDisableTiming);
    }
    // We reuse cuda_event_t objects by wrapping the pool's cudaEvent_t.
    // Caller should not delete these — they're managed by the pool.
    // For simplicity, allocate wrapper on heap. TODO: pool these too.
    return new cuda_event_t(event_pool_[event_pool_idx_++]);
}

// --- Streams & sync ---

gpu_event_t* cuda_device_t::record_compute_event() {
    auto* evt = acquire_event();
    cudaEventRecord(evt->evt, compute_stream_);
    return evt;
}

gpu_event_t* cuda_device_t::record_transfer_event() {
    auto* evt = acquire_event();
    cudaEventRecord(evt->evt, h2d_stream_);
    return evt;
}

void cuda_device_t::compute_wait(gpu_event_t* evt) {
    auto* ce = static_cast<cuda_event_t*>(evt);
    cudaStreamWaitEvent(compute_stream_, ce->evt, 0);
}

void cuda_device_t::transfer_wait(gpu_event_t* evt) {
    auto* ce = static_cast<cuda_event_t*>(evt);
    cudaStreamWaitEvent(d2h_stream_, ce->evt, 0);
}

void cuda_device_t::sync() {
    cudaStreamSynchronize(compute_stream_);
    cudaStreamSynchronize(h2d_stream_);
    cudaStreamSynchronize(d2h_stream_);
    event_pool_idx_ = 0;  // all events are now complete — recycle pool
}

// --- Memory pool ---

void* cuda_device_t::alloc(size_t bytes) {
    void* ptr = nullptr;
    CUDA_CHECK_RET(cudaMallocAsync(&ptr, bytes, compute_stream_), nullptr);
    return ptr;
}

void cuda_device_t::free(void* ptr) {
    if (ptr)
        cudaFreeAsync(ptr, compute_stream_);
}

// --- Async transfers ---

gpu_event_t* cuda_device_t::copy_h2d_async(void* dst_dev, const void* src_host, size_t bytes) {
    cudaMemcpyAsync(dst_dev, src_host, bytes, cudaMemcpyHostToDevice, h2d_stream_);
    auto* evt = acquire_event();
    cudaEventRecord(evt->evt, h2d_stream_);
    return evt;
}

gpu_event_t* cuda_device_t::copy_d2h_async(void* dst_host, const void* src_dev, size_t bytes) {
    cudaMemcpyAsync(dst_host, src_dev, bytes, cudaMemcpyDeviceToHost, d2h_stream_);
    auto* evt = acquire_event();
    cudaEventRecord(evt->evt, d2h_stream_);
    return evt;
}

void cuda_device_t::copy_d2d(void* dst, const void* src, size_t bytes) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, compute_stream_);
}

// --- Pinned host memory ---

void* cuda_device_t::alloc_pinned(size_t bytes) {
    void* ptr = nullptr;
    CUDA_CHECK_RET(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault), nullptr);
    return ptr;
}

void cuda_device_t::free_pinned(void* ptr) {
    if (ptr)
        cudaFreeHost(ptr);
}

bool cuda_device_t::pin(void* ptr, size_t bytes) {
    return cudaHostRegister(ptr, bytes, cudaHostRegisterDefault) == cudaSuccess;
}

void cuda_device_t::unpin(void* ptr) {
    if (ptr)
        cudaHostUnregister(ptr);
}

// --- Unified/managed memory ---

void* cuda_device_t::alloc_managed(size_t bytes) {
    void* ptr = nullptr;
    CUDA_CHECK_RET(cudaMallocManaged(&ptr, bytes, cudaMemAttachGlobal), nullptr);
    // Pin weights on-device to avoid migration thrashing
    cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, device_id_);
    return ptr;
}

void cuda_device_t::free_managed(void* ptr) {
    if (ptr)
        cudaFree(ptr);
}

// --- Runtime compilation ---
// The virtual gpu_device_t hooks are thin shims around the NVRTC cache in
// nvrtc.cpp; most callers use nvrtc_cache_t directly (per-backend state).

void* cuda_device_t::compile_kernel(const char* /*source*/, const char* /*name*/,
                                     const char* /*options*/) {
    // Direct callers use nvrtc_cache_t (see nvrtc.h); this override is unused.
    return nullptr;
}

void cuda_device_t::launch_kernel(void* /*kernel*/, const int /*grid*/[3],
                                   const int /*block*/[3], void** /*args*/,
                                   size_t /*shared_mem*/) {
    // See nvrtc_launch() in nvrtc.cpp.
}

// --- CUDA Graph capture / replay ---

bool cuda_device_t::begin_capture() {
    if (!valid_) return false;
    cudaError_t err = cudaStreamBeginCapture(compute_stream_,
                                             cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaStreamBeginCapture: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

cudaGraphExec_t cuda_device_t::end_capture() {
    cudaGraph_t graph = nullptr;
    cudaError_t err = cudaStreamEndCapture(compute_stream_, &graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaStreamEndCapture: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    cudaGraphExec_t exec = nullptr;
    err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return exec;
}

bool cuda_device_t::launch_graph(cudaGraphExec_t exec) {
    if (!exec) return false;
    cudaError_t err = cudaGraphLaunch(exec, compute_stream_);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGraphLaunch: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

void cuda_device_t::destroy_graph(cudaGraphExec_t exec) {
    if (exec) cudaGraphExecDestroy(exec);
}

// --- Profiling ---

gpu_event_t* cuda_device_t::record_profile_start() {
    // Profiling events need timing enabled — allocate separately.
    cudaEvent_t e;
    cudaEventCreate(&e);  // default flags (timing enabled)
    cudaEventRecord(e, compute_stream_);
    return new cuda_event_t(e);
}

gpu_event_t* cuda_device_t::record_profile_stop() {
    cudaEvent_t e;
    cudaEventCreate(&e);
    cudaEventRecord(e, compute_stream_);
    return new cuda_event_t(e);
}

float cuda_device_t::event_elapsed_us(gpu_event_t* start, gpu_event_t* stop) {
    auto* cs = static_cast<cuda_event_t*>(start);
    auto* ce = static_cast<cuda_event_t*>(stop);
    cudaEventSynchronize(ce->evt);
    float ms = 0;
    cudaEventElapsedTime(&ms, cs->evt, ce->evt);
    return ms * 1000.0f;  // convert ms → us
}

// --- Factory ---

gpu_device_t* create_cuda_device(int id) {
    auto* dev = new cuda_device_t(id);
    if (!dev->valid()) {
        delete dev;
        return nullptr;
    }
    return dev;
}

int enumerate_cuda_devices(device_info_t* out, int max_count) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess)
        return 0;
    int n = (count < max_count) ? count : max_count;
    for (int i = 0; i < n; i++) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess)
            continue;
        auto& d = out[i];
        d.id = i;
        d.backend = gpu_backend_t::CUDA;
        std::strncpy(d.name, prop.name, sizeof(d.name) - 1);
        d.total_mem = prop.totalGlobalMem;
        d.free_mem = 0;  // requires cudaMemGetInfo after cudaSetDevice
        d.p2p_capable = false;
        d.uma = (prop.integrated != 0);
        d.compute_capability[0] = prop.major;
        d.compute_capability[1] = prop.minor;
    }
    return n;
}

} // namespace nnr::gpu

#endif // NNR_USE_CUDA
