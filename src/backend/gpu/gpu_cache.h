#pragma once
// GPU device memory cache — manages tensor data on GPU devices.
//
// Parallels oneDNN's format_cache_t (src/backend/onednn/dnnl_common.h):
//   format_cache_t::get_blocked()  →  gpu_cache_t::ensure_device()
//   format_cache_t::get_output()   →  gpu_cache_t::alloc_output()
//   format_cache_t::writeback()    →  gpu_cache_t::writeback()
//
// Stored in context_t::accel_data (same slot as oneDNN).
// accel_writeback calls gpu_cache_t::writeback().

#include "gpu_device.h"
#include <unordered_map>

namespace nnr {
struct tensor_t;  // forward — defined in nnr.h
}

namespace nnr::gpu {

// One entry per tensor that has a device-side copy.
struct device_buffer_t {
    void*           dev_ptr         = nullptr;
    size_t          bytes           = 0;
    gpu_device_t*   owner           = nullptr;  // which physical device holds this buffer
    gpu_event_t*    last_write_evt  = nullptr;  // event after last kernel that wrote this buffer
    void*           pinned_host_ptr = nullptr;  // exact host pointer passed to pin(); used on unpin
    bool            pinned_host     = false;    // host memory was pinned by us (unpin on evict)
    bool            is_weight       = false;    // persistent across inference runs
    bool            is_alias        = false;    // dev_ptr is owned by another entry (Reshape/Flatten)
};

struct gpu_cache_t {
    gpu_device_t* device = nullptr;
    std::unordered_map<const tensor_t*, device_buffer_t> entries;

    explicit gpu_cache_t(gpu_device_t* dev) : device(dev) {}
    ~gpu_cache_t();

    gpu_cache_t(const gpu_cache_t&) = delete;
    gpu_cache_t& operator=(const gpu_cache_t&) = delete;

    // Upload host → device (async).
    // Returns device pointer. Skips upload if already on device.
    // On UMA: returns tensor_t::data directly (zero-copy).
    //
    // Makes the compute stream wait for the transfer to complete
    // so the kernel can safely read the data.
    void* ensure_device(const tensor_t* t);

    // Allocate device output buffer for a tensor. No data transfer.
    // On UMA: returns tensor_t::data directly.
    void* alloc_output(tensor_t* t);

    // Download device → host. Blocks host until complete.
    // Called by accel_writeback when a CPU op needs device-written data.
    // On UMA: just waits for the last kernel to finish (no copy).
    void writeback(tensor_t* t);

    // Record that a GPU kernel has written to this tensor's device buffer.
    // Captures a compute event for future D2H ordering.
    void mark_written(tensor_t* t);

    // Alias: output tensor shares the input's device buffer (zero-copy reshape
    // family: Reshape, Flatten, Squeeze, Unsqueeze — view ops that only rename
    // dimensions). No kernel launch, no allocation. Returns the shared dev ptr.
    // The alias entry is skipped by writeback/clear so the buffer is freed
    // exactly once (when the input's entry is freed).
    void* alias(const tensor_t* src, tensor_t* dst);

    // Query whether a tensor has a device-side copy.
    bool has_device(const tensor_t* t) const;

    // Evict all entries. Frees device memory, unpins host memory.
    void clear();

private:
    // Staging buffer pool for pinned H2D transfers (discrete GPU only).
    // Reuses pinned buffers across ensure_device() calls to avoid
    // repeated cudaHostAlloc/cudaFreeHost overhead.
    struct staging_pool_t {
        struct entry { void* ptr; size_t bytes; };
        std::vector<entry> pool;
        void* acquire(gpu_device_t* dev, size_t bytes);
        void release_all(gpu_device_t* dev);
    };
    staging_pool_t staging_;
};

} // namespace nnr::gpu
