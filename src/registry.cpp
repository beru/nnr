#include "registry.h"

namespace nnr {

void registry_t::register_op(std::string_view name, backend_t backend, resolver_fn fn)
{
    std::lock_guard<std::mutex> lock(mtx);
    ops[{name, backend}] = fn;
}

operator_t* registry_t::solve(std::string_view op_type, int opset, pool_t& pool,
                               backend_t preferred)
{
    // Resolve the entries under the lock, then release before invoking the
    // resolver — the resolver allocates through `pool` and can be long-running,
    // and we don't want to hold the map lock across it.
    resolver_fn preferred_fn = nullptr;
    resolver_fn cpu_fn = nullptr;
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (preferred != backend_t::CPU) {
            auto it = ops.find({op_type, preferred});
            if (it != ops.end()) preferred_fn = it->second;
        }
        auto it = ops.find({op_type, backend_t::CPU});
        if (it != ops.end()) cpu_fn = it->second;
    }

    // 1. Try preferred backend (skip if CPU — that's the fallback)
    //    If the resolver returns nullptr, fall through to CPU.
    if (preferred_fn) {
        operator_t* op = preferred_fn(opset, pool);
        if (op) {
            op->resolved_backend = static_cast<uint8_t>(preferred);
            return op;
        }
    }
    // 2. Fall back to CPU
    if (cpu_fn) {
        operator_t* op = cpu_fn(opset, pool);
        if (op) op->resolved_backend = static_cast<uint8_t>(backend_t::CPU);
        return op;
    }
    return nullptr;
}

registry_t& global_registry()
{
    static registry_t instance;
    return instance;
}

} // namespace nnr
