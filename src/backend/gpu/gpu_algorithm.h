#pragma once
// gpu_algorithm.h — device-side parallel algorithms.
//
// All algorithms:
//   - Take gpu_span<T> inputs (non-owning views)
//   - Execute on device's compute stream (async, no host sync)
//   - Are CUDA Graph capturable (no allocation, no host callback)
//   - Can be JIT-compiled for specific types/sizes via gpu_device_t::compile_kernel
//
// Usage:
//   gpu_array<float> a(device, N), b(device, N), c(device, N);
//
//   gpu::for_each(device, a, [] __device__ (float x) { return x * 2; });
//   gpu::transform(device, a, b, c, [] __device__ (float a, float b) { return a + b; });
//   float sum = gpu::reduce(device, a, 0.0f, gpu::plus<float>{});
//   gpu::sort(device, a);
//   gpu::scan(device, a, b, gpu::plus<float>{});  // inclusive prefix sum

#include "gpu_span.h"
#include "gpu_device.h"

namespace nnr::gpu {

// --- Reduction operators ---

template <typename T> struct plus     { T operator()(T a, T b) const { return a + b; } };
template <typename T> struct multiply { T operator()(T a, T b) const { return a * b; } };
template <typename T> struct min_op   { T operator()(T a, T b) const { return a < b ? a : b; } };
template <typename T> struct max_op   { T operator()(T a, T b) const { return a > b ? a : b; } };

// --- Algorithms ---

// Apply unary function to each element in-place: a[i] = f(a[i])
template <typename T, typename F>
void for_each(gpu_device_t* device, gpu_span<T> a, F fn);

// Unary transform: out[i] = f(in[i])
template <typename T, typename U, typename F>
void transform(gpu_device_t* device, gpu_span<const T> in, gpu_span<U> out, F fn);

// Binary transform: out[i] = f(a[i], b[i])
template <typename T, typename U, typename V, typename F>
void transform(gpu_device_t* device,
               gpu_span<const T> a, gpu_span<const U> b,
               gpu_span<V> out, F fn);

// JIT transform: compile kernel from expression string at runtime
// Expression uses 'x' for unary, 'a','b' for binary.
//   jit_transform(device, in, out, "x * 2.0f + 1.0f");
//   jit_transform(device, a, b, out, "a * b + 1.0f");
void jit_transform(gpu_device_t* device,
                   gpu_span<const float> in, gpu_span<float> out,
                   const char* expr);

void jit_transform(gpu_device_t* device,
                   gpu_span<const float> a, gpu_span<const float> b,
                   gpu_span<float> out,
                   const char* expr);

// Reduce: result = reduce(a, init, op)
// Synchronous — returns result to host.
template <typename T, typename Op>
T reduce(gpu_device_t* device, gpu_span<const T> a, T init, Op op);

// Reduce async — result written to device memory
template <typename T, typename Op>
void reduce_async(gpu_device_t* device, gpu_span<const T> a, T* d_result, T init, Op op);

// Inclusive prefix sum (scan): out[i] = sum(a[0..i])
template <typename T, typename Op>
void inclusive_scan(gpu_device_t* device,
                    gpu_span<const T> in, gpu_span<T> out, Op op);

// Exclusive prefix sum: out[i] = sum(a[0..i-1]), out[0] = init
template <typename T, typename Op>
void exclusive_scan(gpu_device_t* device,
                    gpu_span<const T> in, gpu_span<T> out, T init, Op op);

// Sort in-place (ascending)
template <typename T>
void sort(gpu_device_t* device, gpu_span<T> a);

// Sort by key: reorder values according to sorted keys
template <typename K, typename V>
void sort_by_key(gpu_device_t* device, gpu_span<K> keys, gpu_span<V> values);

// Fill: a[i] = value
template <typename T>
void fill(gpu_device_t* device, gpu_span<T> a, T value);

// Copy: dst[i] = src[i]
template <typename T>
void copy(gpu_device_t* device, gpu_span<const T> src, gpu_span<T> dst);

// Count elements matching predicate
template <typename T, typename Pred>
size_t count_if(gpu_device_t* device, gpu_span<const T> a, Pred pred);

// Gather: out[i] = in[indices[i]]
template <typename T>
void gather(gpu_device_t* device,
            gpu_span<const T> in, gpu_span<const int> indices,
            gpu_span<T> out);

// Scatter: out[indices[i]] = in[i]
template <typename T>
void scatter(gpu_device_t* device,
             gpu_span<const T> in, gpu_span<const int> indices,
             gpu_span<T> out);

} // namespace nnr::gpu
