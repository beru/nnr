#include "gpu_cache.h"
#include "nnr.h"
#include <cstring>

namespace nnr::gpu {

// --- staging_pool_t ---

void* gpu_cache_t::staging_pool_t::acquire(gpu_device_t* dev, size_t bytes) {
    // Find a buffer large enough in the pool.
    for (auto& e : pool) {
        if (e.bytes >= bytes)
            return e.ptr;
    }
    // Allocate a new pinned staging buffer.
    void* ptr = dev->alloc_pinned(bytes);
    if (ptr)
        pool.push_back({ptr, bytes});
    return ptr;
}

void gpu_cache_t::staging_pool_t::release_all(gpu_device_t* dev) {
    for (auto& e : pool)
        dev->free_pinned(e.ptr);
    pool.clear();
}

// --- gpu_cache_t ---

gpu_cache_t::~gpu_cache_t() {
    clear();
}

void* gpu_cache_t::ensure_device(const tensor_t* t) {
    if (!t || !t->data || t->ndata == 0)
        return nullptr;

    // Already cached?
    auto it = entries.find(t);
    if (it != entries.end())
        return it->second.dev_ptr;

    const size_t bytes = t->ndata * data_type_sizeof(t->type);

    // UMA: GPU reads host memory directly. No alloc, no copy.
    if (device->is_uma()) {
        entries[t] = {
            .dev_ptr = t->data,
            .bytes   = bytes,
            .owner   = device,
        };
        return t->data;
    }

    // Discrete GPU: allocate device buffer + async upload.
    void* dev_ptr = device->alloc(bytes);
    if (!dev_ptr)
        return nullptr;

    // Pin host memory for weights (large, uploaded once).
    // Use staging pool for activations (moderate, uploaded every run).
    const void* src = t->data;
    bool pinned = false;

    // Try to pin the host allocation directly for large tensors (>64KB).
    // Small tensors go through the staging pool to avoid pin/unpin overhead.
    if (bytes > 64 * 1024) {
        pinned = device->pin(const_cast<void*>(t->data), bytes);
    }

    if (!pinned) {
        // Staging path: memcpy host → pinned staging, then async DMA.
        void* stage = staging_.acquire(device, bytes);
        if (stage) {
            std::memcpy(stage, t->data, bytes);
            src = stage;
        }
        // If staging alloc fails, fall through with unpinned host ptr.
        // cudaMemcpyAsync will degrade to sync — acceptable as fallback.
    }

    gpu_event_t* evt = device->copy_h2d_async(dev_ptr, src, bytes);
    device->compute_wait(evt);  // compute stream waits for upload

    entries[t] = {
        .dev_ptr         = dev_ptr,
        .bytes           = bytes,
        .owner           = device,
        .pinned_host_ptr = pinned ? const_cast<void*>(t->data) : nullptr,
        .pinned_host     = pinned,
    };

    return dev_ptr;
}

void* gpu_cache_t::alloc_output(tensor_t* t) {
    if (!t)
        return nullptr;

    const size_t bytes = t->ndata * data_type_sizeof(t->type);

    // Reuse existing buffer if sizes match. Keeps device pointers stable across
    // inference runs, which is required for CUDA Graph replay (captured graphs
    // hardcode device pointers). invalidate_shape() clears the cache when the
    // memory plan changes.
    auto it = entries.find(t);
    if (it != entries.end() && it->second.bytes >= bytes && !it->second.is_alias) {
        return it->second.dev_ptr;
    }
    if (it != entries.end()) {
        // Size changed — free the old buffer.
        if (!it->second.is_alias && !device->is_uma() && it->second.dev_ptr)
            device->free(it->second.dev_ptr);
        entries.erase(it);
    }

    // Concat alias: this tensor is a sub-region of `concat_parent`'s buffer.
    // Resolve the parent's device pointer (recursively, in case parents chain)
    // and return parent_dev_ptr + offset. The producer kernel then writes
    // directly into the Concat output, and the Concat op becomes a no-op.
    if (t->concat_parent && !device->is_uma()) {
        void* parent_dev = alloc_output(t->concat_parent);
        if (!parent_dev)
            return nullptr;
        void* dev_ptr = (char*)parent_dev + t->concat_offset;
        entries[t] = {
            .dev_ptr  = dev_ptr,
            .bytes    = bytes,
            .owner    = device,
            .is_alias = true,
        };
        return dev_ptr;
    }

    // UMA: kernel writes directly to host memory.
    if (device->is_uma()) {
        entries[t] = {
            .dev_ptr = t->data,
            .bytes   = bytes,
            .owner   = device,
        };
        return t->data;
    }

    // Discrete GPU: allocate device buffer.
    void* dev_ptr = device->alloc(bytes);
    if (!dev_ptr)
        return nullptr;

    entries[t] = {
        .dev_ptr = dev_ptr,
        .bytes   = bytes,
        .owner   = device,
    };

    return dev_ptr;
}

void gpu_cache_t::writeback(tensor_t* t) {
    if (!t)
        return;

    auto it = entries.find(t);
    if (it == entries.end())
        return;  // not on device — nothing to write back

    auto& buf = it->second;

    if (device->is_uma()) {
        // UMA: data is already in host memory. Just wait for GPU to finish.
        if (buf.last_write_evt)
            buf.last_write_evt->wait();
        return;
    }

    // Discrete GPU: D2H copy.
    if (buf.last_write_evt)
        device->transfer_wait(buf.last_write_evt);  // d2h stream waits for kernel

    gpu_event_t* evt = device->copy_d2h_async(t->data, buf.dev_ptr, buf.bytes);
    evt->wait();  // host blocks for readback

    // Do NOT free the device buffer here. CUDA Graph replay hardcodes output
    // pointers into the captured graph — freeing between runs breaks replay
    // (cudaGraphLaunch → "invalid argument"). The buffer stays in the cache
    // and is reused by the next alloc_output() for this tensor. clear() and
    // alloc_output()'s size-mismatch branch handle the actual frees.
}

void gpu_cache_t::mark_written(tensor_t* t) {
    auto it = entries.find(t);
    if (it == entries.end())
        return;
    it->second.last_write_evt = device->record_compute_event();
}

bool gpu_cache_t::has_device(const tensor_t* t) const {
    return entries.count(t) > 0;
}

void gpu_cache_t::clear() {
    if (!device)
        return;

    for (auto& [tensor, buf] : entries) {
        if (buf.is_alias) continue;  // aliases don't own the buffer
        if (!device->is_uma() && buf.dev_ptr) {
            device->free(buf.dev_ptr);
        }
        // Unpin using the cached host pointer, not tensor->data — the tensor's
        // data pointer can be reassigned by the memory planner between pin
        // and destruction (intermediate tensors get pool-allocated slots).
        if (buf.pinned_host && buf.pinned_host_ptr) {
            device->unpin(buf.pinned_host_ptr);
        }
    }
    entries.clear();
    staging_.release_all(device);
}

void* gpu_cache_t::alias(const tensor_t* src, tensor_t* dst) {
    if (!src || !dst) return nullptr;
    auto it = entries.find(src);
    if (it == entries.end()) return nullptr;  // src not on device
    const auto& sbuf = it->second;
    entries[dst] = device_buffer_t{
        .dev_ptr        = sbuf.dev_ptr,
        .bytes          = sbuf.bytes,
        .owner          = sbuf.owner,
        .last_write_evt = sbuf.last_write_evt,
        .pinned_host    = false,
        .is_weight      = false,
        .is_alias       = true,
    };
    return sbuf.dev_ptr;
}

// --- Factory ---

gpu_device_t* create_device(gpu_backend_t backend, int device_id) {
    switch (backend) {
#if defined(NNR_USE_CUDA)
    case gpu_backend_t::CUDA:
        extern gpu_device_t* create_cuda_device(int id);
        return create_cuda_device(device_id);
#endif
#if defined(NNR_USE_HIP)
    case gpu_backend_t::HIP:
        extern gpu_device_t* create_hip_device(int id);
        return create_hip_device(device_id);
#endif
#if defined(NNR_USE_VULKAN)
    case gpu_backend_t::VULKAN:
        extern gpu_device_t* create_vulkan_device(int id);
        return create_vulkan_device(device_id);
#endif
#if defined(NNR_USE_WEBGPU)
    case gpu_backend_t::WEBGPU:
        extern gpu_device_t* create_webgpu_device(int id);
        return create_webgpu_device(device_id);
#endif
    default:
        return nullptr;
    }
}

int enumerate_devices(gpu_backend_t backend, device_info_t* out, int max_count) {
    switch (backend) {
#if defined(NNR_USE_CUDA)
    case gpu_backend_t::CUDA:
        extern int enumerate_cuda_devices(device_info_t* out, int max_count);
        return enumerate_cuda_devices(out, max_count);
#endif
#if defined(NNR_USE_VULKAN)
    case gpu_backend_t::VULKAN:
        extern int enumerate_vulkan_devices(device_info_t* out, int max_count);
        return enumerate_vulkan_devices(out, max_count);
#endif
    default:
        return 0;
    }
}

bool can_peer_access(gpu_device_t* /*src*/, gpu_device_t* /*dst*/) {
    return false;  // TODO: implement in Phase 8 (multi-device)
}

} // namespace nnr::gpu
