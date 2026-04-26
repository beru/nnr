#pragma once
// gpu_array<T> — typed device array with pool allocation.
//
// Design:
//   - Thin wrapper: pointer + size + device reference
//   - Pool-allocated: no cudaMalloc per array (uses gpu_device_t::alloc)
//   - Non-owning view mode: wrap existing device pointers without ownership
//   - Host↔device transfer via explicit copy methods
//   - CUDA Graph safe: no allocation during execution, only at setup time
//   - SoA-friendly: multiple gpu_array<float> for x[], y[], z[]
//
// Usage:
//   auto device = gpu::create_device(gpu::gpu_backend_t::CUDA);
//   gpu::gpu_array<float> a(device, 1024);           // allocate 1024 floats
//   gpu::gpu_array<float> b(device, ptr, 1024);      // view existing pointer
//   a.copy_from_host(host_data, 1024);                // upload
//   a.copy_to_host(host_data, 1024);                  // download
//   float* raw = a.data();                            // raw device pointer
//   size_t n = a.size();                              // element count

#include "gpu_device.h"
#include <cstddef>
#include <utility>

namespace nnr::gpu {

template <typename T>
struct gpu_array {
    // --- Constructors ---

    // Empty
    gpu_array() = default;

    // Allocate N elements on device
    gpu_array(gpu_device_t* dev, size_t count)
        : device_(dev), data_(nullptr), size_(count), owns_(true)
    {
        if (count > 0)
            data_ = static_cast<T*>(dev->alloc(count * sizeof(T)));
    }

    // View existing device pointer (non-owning)
    gpu_array(gpu_device_t* dev, T* ptr, size_t count)
        : device_(dev), data_(ptr), size_(count), owns_(false)
    {}

    // Move
    gpu_array(gpu_array&& other) noexcept
        : device_(other.device_), data_(other.data_),
          size_(other.size_), owns_(other.owns_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
        other.owns_ = false;
    }

    gpu_array& operator=(gpu_array&& other) noexcept {
        if (this != &other) {
            free();
            device_ = other.device_;
            data_ = other.data_;
            size_ = other.size_;
            owns_ = other.owns_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.owns_ = false;
        }
        return *this;
    }

    // No copy
    gpu_array(const gpu_array&) = delete;
    gpu_array& operator=(const gpu_array&) = delete;

    ~gpu_array() { free(); }

    // --- Access ---

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    size_t bytes() const { return size_ * sizeof(T); }
    bool empty() const { return size_ == 0 || data_ == nullptr; }
    bool owns() const { return owns_; }
    gpu_device_t* device() const { return device_; }

    // --- Transfers (async on device streams) ---

    // Upload: host → device
    gpu_event_t* copy_from_host(const T* host_src, size_t count) {
        size_t n = (count < size_) ? count : size_;
        return device_->copy_h2d_async(data_, host_src, n * sizeof(T));
    }

    gpu_event_t* copy_from_host(const T* host_src) {
        return copy_from_host(host_src, size_);
    }

    // Download: device → host
    gpu_event_t* copy_to_host(T* host_dst, size_t count) const {
        size_t n = (count < size_) ? count : size_;
        return device_->copy_d2h_async(host_dst, data_, n * sizeof(T));
    }

    gpu_event_t* copy_to_host(T* host_dst) const {
        return copy_to_host(host_dst, size_);
    }

    // Device → device copy
    void copy_from(const gpu_array& src) {
        size_t n = (src.size_ < size_) ? src.size_ : size_;
        device_->copy_d2d(data_, src.data_, n * sizeof(T));
    }

    // --- Fill ---

    // Zero-fill (async, on compute stream)
    void zero() {
        if (data_ && size_ > 0) {
            // cudaMemsetAsync wraps to device's compute stream
            device_->copy_d2d(data_, nullptr, 0);  // TODO: add memset to gpu_device_t
        }
    }

    // --- Resize ---

    // Reallocate (frees old, allocates new — NOT preserving data)
    void resize(size_t new_count) {
        if (new_count == size_) return;
        free();
        size_ = new_count;
        owns_ = true;
        if (new_count > 0)
            data_ = static_cast<T*>(device_->alloc(new_count * sizeof(T)));
    }

    // --- Subview ---

    // Non-owning view into a slice of this array
    gpu_array<T> view(size_t offset, size_t count) {
        return gpu_array<T>(device_, data_ + offset, count);
    }

private:
    void free() {
        if (owns_ && data_ && device_) {
            device_->free(data_);
        }
        data_ = nullptr;
        size_ = 0;
        owns_ = false;
    }

    gpu_device_t* device_ = nullptr;
    T* data_ = nullptr;
    size_t size_ = 0;
    bool owns_ = false;
};

} // namespace nnr::gpu
