#pragma once
// gpu_plan — deferred execution plan (CUDA Graph capturable).
//
// Collects GPU operations without executing them. When finalized,
// the plan can be captured as a CUDA Graph for near-zero replay cost.
//
// This is the key difference from Thrust: operations are recorded first,
// then executed as a batch. No per-op launch overhead, no per-op sync.
//
// Usage:
//   auto device = gpu::create_device(gpu::gpu_backend_t::CUDA);
//   gpu_array<float> a(device, N), b(device, N), c(device, N);
//
//   // Record operations
//   gpu_plan plan(device);
//   plan.transform(a, b, "x * 2.0f");
//   plan.reduce(b, &sum, gpu::plus<float>{});
//   plan.sort(c);
//
//   // Option A: execute immediately (stream-ordered, no graph)
//   plan.execute();
//
//   // Option B: capture as CUDA Graph, replay many times
//   auto graph = plan.capture();
//   for (int i = 0; i < 1000; i++)
//       graph.launch();    // ~1us per launch

#include "gpu_span.h"
#include "gpu_device.h"
#include <vector>
#include <functional>
#include <memory>

namespace nnr::gpu {

// Opaque captured graph handle
struct gpu_graph {
    gpu_graph() = default;
    gpu_graph(gpu_device_t* dev, void* handle) : device_(dev), handle_(handle) {}
    ~gpu_graph() { if (handle_) device_->graph_destroy(handle_); }

    gpu_graph(gpu_graph&& o) noexcept : device_(o.device_), handle_(o.handle_) {
        o.handle_ = nullptr;
    }
    gpu_graph& operator=(gpu_graph&& o) noexcept {
        if (this != &o) {
            if (handle_) device_->graph_destroy(handle_);
            device_ = o.device_; handle_ = o.handle_;
            o.handle_ = nullptr;
        }
        return *this;
    }

    gpu_graph(const gpu_graph&) = delete;
    gpu_graph& operator=(const gpu_graph&) = delete;

    bool valid() const { return handle_ != nullptr; }
    bool launch() { return device_->graph_launch(handle_); }

private:
    gpu_device_t* device_ = nullptr;
    void* handle_ = nullptr;
};

// Operation record — one step in the plan
struct plan_op {
    enum class kind_t {
        TRANSFORM_UNARY,
        TRANSFORM_BINARY,
        TRANSFORM_JIT,
        REDUCE,
        SCAN,
        SORT,
        FILL,
        COPY,
        CUSTOM,       // user-provided function
    };

    kind_t kind;
    std::function<void()> execute_fn;   // captures all args by value
};

struct gpu_plan {
    explicit gpu_plan(gpu_device_t* dev) : device_(dev) {}

    // --- Record operations ---

    // JIT unary transform: out[i] = f(in[i]) where f is an expression string
    void transform(gpu_span<const float> in, gpu_span<float> out, const char* expr) {
        ops_.push_back({plan_op::kind_t::TRANSFORM_JIT, [=, dev=device_]() {
            jit_transform(dev, in, out, expr);
        }});
    }

    // JIT binary transform
    void transform(gpu_span<const float> a, gpu_span<const float> b,
                   gpu_span<float> out, const char* expr) {
        ops_.push_back({plan_op::kind_t::TRANSFORM_JIT, [=, dev=device_]() {
            jit_transform(dev, a, b, out, expr);
        }});
    }

    // Sort
    template <typename T>
    void sort(gpu_span<T> a) {
        ops_.push_back({plan_op::kind_t::SORT, [=, dev=device_]() {
            gpu::sort(dev, a);
        }});
    }

    // Fill
    template <typename T>
    void fill(gpu_span<T> a, T value) {
        ops_.push_back({plan_op::kind_t::FILL, [=, dev=device_]() {
            gpu::fill(dev, a, value);
        }});
    }

    // Custom operation (user provides lambda)
    void custom(std::function<void()> fn) {
        ops_.push_back({plan_op::kind_t::CUSTOM, std::move(fn)});
    }

    // --- Execute ---

    // Execute all recorded ops immediately (stream-ordered)
    void execute() {
        for (auto& op : ops_)
            op.execute_fn();
    }

    // Capture as CUDA Graph for replay
    gpu_graph capture() {
        if (!device_->supports_graph_capture())
            return {};

        device_->begin_capture();
        execute();
        void* handle = device_->end_capture();
        return gpu_graph(device_, handle);
    }

    // --- Properties ---

    size_t num_ops() const { return ops_.size(); }
    void clear() { ops_.clear(); }

private:
    gpu_device_t* device_;
    std::vector<plan_op> ops_;
};

} // namespace nnr::gpu
