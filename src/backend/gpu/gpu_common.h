#pragma once
// GPU backend shared utilities.

#include <cstdint>
#include <cstddef>
#include <cstdio>

namespace nnr::gpu {

enum class gpu_backend_t : uint8_t {
    CUDA,
    HIP,
    VULKAN,
    WEBGPU,
};

// Device capability summary — returned by enumerate_devices().
struct device_info_t {
    int             id;                     // ordinal within backend
    gpu_backend_t   backend;
    char            name[256];              // "NVIDIA GeForce RTX 5090"
    size_t          total_mem;              // device memory (bytes)
    size_t          free_mem;
    bool            p2p_capable;
    bool            uma;                    // unified memory architecture
    int             compute_capability[2];  // {major, minor} — CUDA/HIP; {0,0} for Vulkan/WebGPU
};

// Discover all devices for a given backend.
// Writes up to max_count entries into out[]. Returns number found (may be 0).
int enumerate_devices(gpu_backend_t backend, device_info_t* out, int max_count);

// Check if two devices can do direct P2P transfers (NVLink, PCIe BAR, etc).
bool can_peer_access(struct gpu_device_t* src, struct gpu_device_t* dst);

} // namespace nnr::gpu
