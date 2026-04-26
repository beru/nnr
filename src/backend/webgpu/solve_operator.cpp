#include "solve_operator.h"

#include "registry.h"

namespace nnr {

// Forward-declare all WebGPU resolver functions.
#define X(name) operator_t* resolver_default_op_##name##_webgpu(int opset, pool_t& pool);
#include "ops.h"
#undef X

namespace {

// Auto-register all WebGPU operators at static-init time.
// The ensure_registered() anchor keeps this TU alive in MSVC static-lib
// builds (dead-stripping would otherwise drop webgpu_registrar_instance
// before it runs).
struct webgpu_registrar {
    webgpu_registrar() {
        auto& r = global_registry();
        #define X(name) r.register_op(#name, backend_t::WEBGPU, resolver_default_op_##name##_webgpu);
        #include "ops.h"
        #undef X
    }
} webgpu_registrar_instance;

} // namespace

namespace webgpu {
void ensure_registered() {
    // Reference the registrar so the linker keeps this TU.
    (void)&webgpu_registrar_instance;
}
} // namespace webgpu

} // namespace nnr
