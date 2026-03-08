#include "registry.h"

namespace nnr {

void registry_t::register_op(std::string_view name, backend_t backend, resolver_fn fn)
{
    ops[{name, backend}] = fn;
}

operator_t* registry_t::solve(std::string_view op_type, int opset, pool_t& pool,
                               backend_t preferred)
{
    // 1. Try preferred backend (skip if CPU — that's the fallback)
    //    If the resolver returns nullptr, fall through to CPU.
    if (preferred != backend_t::CPU) {
        auto it = ops.find({op_type, preferred});
        if (it != ops.end()) {
            operator_t* op = it->second(opset, pool);
            if (op) return op;
        }
    }
    // 2. Fall back to CPU
    auto it = ops.find({op_type, backend_t::CPU});
    if (it != ops.end())
        return it->second(opset, pool);
    return nullptr;
}

registry_t& global_registry()
{
    static registry_t instance;
    return instance;
}

} // namespace nnr
