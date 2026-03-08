#include "solve_operator.h"
#include "registry.h"

namespace nnr {

// Forward-declare all CPU resolver functions
#define X(name) operator_t* resolver_default_op_##name(int opset, pool_t& pool);
#include "ops.h"
#undef X

namespace {

// Auto-register all CPU operators at static-init time.
// IMPORTANT: this static must stay in this TU, and solve_operator() below
// must remain the entry point called by onnx_loader — that reference keeps
// the whole TU alive in MSVC static-lib builds (otherwise dead-stripping
// would silently remove cpu_registrar_instance before it runs).
struct cpu_registrar {
    cpu_registrar() {
        auto& r = global_registry();
        #define X(name) r.register_op(#name, backend_t::CPU, resolver_default_op_##name);
        #include "ops.h"
        #undef X
    }
} cpu_registrar_instance;

} // namespace

operator_t* solve_operator(std::string_view op_type, int opset, pool_t& pool,
                           backend_t preferred)
{
    return global_registry().solve(op_type, opset, pool, preferred);
}

} // namespace nnr
