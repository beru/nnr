#pragma once
// NVRTC kernel cache — runtime-compile CUDA C++ source, cache CUmodule+CUfunction.
//
// Usage:
//   CUfunction f = nvrtc_cache.get(key, source, "kernel_name", "-arch=compute_86");
//   nvrtc_launch(device, f, grid, block, args, shared);
//
// Modules/functions are cached per (key) string; compile happens once.
// Kernels launch on the device's compute stream so they're safe to capture
// into a CUDA graph.

#if defined(NNR_USE_CUDA)

#include <cuda.h>         // driver API: CUmodule, CUfunction, cuLaunchKernel
#include <cuda_runtime.h> // cudaStream_t
#include <string>
#include <unordered_map>

namespace nnr::gpu {

struct cuda_device_t;

struct nvrtc_kernel_t {
    CUmodule   module = nullptr;
    CUfunction func   = nullptr;
};

struct nvrtc_cache_t {
    // Compile `source` and look up kernel `name`, or return cached entry.
    // `key` is the cache key (any stable string — typically source+name+options).
    // `options` may be nullptr. Returns nullptr on failure (logs to stderr).
    CUfunction get(const std::string& key,
                   const char* source,
                   const char* name,
                   const char* options = nullptr);

    ~nvrtc_cache_t();

private:
    std::unordered_map<std::string, nvrtc_kernel_t> cache_;
};

// Launch `func` on `dev`'s compute stream. `args` is an array of void* pointing
// to each argument's storage (matching cuLaunchKernel conventions).
// Returns false on launch error.
bool nvrtc_launch(cuda_device_t* dev,
                  CUfunction func,
                  unsigned grid_x, unsigned grid_y, unsigned grid_z,
                  unsigned block_x, unsigned block_y, unsigned block_z,
                  void** args,
                  unsigned shared_bytes = 0);

// Returns the arch option string for the current device (e.g. "-arch=compute_86").
// Cached per-device.
const char* nvrtc_arch_option(cuda_device_t* dev);

} // namespace nnr::gpu

#endif // NNR_USE_CUDA
