#pragma once

#include "nnr.h"
#include <map>

namespace nnr {

enum class backend_t : uint8_t { CPU = 0, CUDA, SYCL, VULKAN };

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
    std::map<key_t, resolver_fn> ops;
};

registry_t& global_registry();

} // namespace nnr
