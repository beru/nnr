#pragma once

#include "nnr.h"
#include <map>
#include <mutex>

namespace nnr {

enum class backend_t : uint8_t { CPU = 0, CUDA, SYCL, VULKAN, WEBGPU };

using resolver_fn = operator_t* (*)(int opset, pool_t& pool);

struct registry_t {
    void register_op(std::string_view name, backend_t backend, resolver_fn fn);

    // Resolve: try preferred backend, fall back to CPU
    operator_t* solve(std::string_view op_type, int opset, pool_t& pool,
                      backend_t preferred = backend_t::CPU);

private:
    struct key_t {
        std::string_view name;
        backend_t        backend;
        auto operator<=>(const key_t&) const = default;
    };
    // `ops` is mutated from static-init registrars across translation units
    // (and potentially DSOs) and read concurrently by load() threads calling
    // solve(). Every access below holds `mtx`. Contention is negligible:
    // register_op runs once per op at startup, solve runs once per graph
    // node at load — never on the per-inference hot path.
    mutable std::mutex mtx;
    std::map<key_t, resolver_fn> ops;
};

registry_t& global_registry();

} // namespace nnr
