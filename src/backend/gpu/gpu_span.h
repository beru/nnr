#pragma once
// gpu_span<T> — non-owning view into device memory.
//
// Like std::span but for GPU. Trivially copyable — safe to pass
// to kernels by value. No ownership, no device reference.
//
// Usage:
//   gpu_array<float> arr(device, N);
//   gpu_span<float> s = arr;                      // implicit conversion
//   gpu_span<float> sub = s.subspan(10, 100);     // slice
//   kernel<<<grid, block>>>(s.data(), s.size());   // pass to kernel

#include <cstddef>

namespace nnr::gpu {

template <typename T>
struct gpu_span {
    // --- Constructors ---

    gpu_span() = default;
    gpu_span(T* ptr, size_t count) : data_(ptr), size_(count) {}

    // From gpu_array (implicit)
    template <typename U>
    gpu_span(U& arr) : data_(arr.data()), size_(arr.size()) {}

    // --- Access ---

    T* data() const { return data_; }
    size_t size() const { return size_; }
    size_t bytes() const { return size_ * sizeof(T); }
    bool empty() const { return size_ == 0; }

    // --- Slicing ---

    gpu_span<T> subspan(size_t offset, size_t count) const {
        return {data_ + offset, count};
    }

    gpu_span<T> first(size_t count) const {
        return {data_, count};
    }

    gpu_span<T> last(size_t count) const {
        return {data_ + size_ - count, count};
    }

private:
    T* data_ = nullptr;
    size_t size_ = 0;
};

} // namespace nnr::gpu
