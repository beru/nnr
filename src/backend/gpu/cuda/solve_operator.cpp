#include "registry.h"

#if defined(NNR_USE_CUDA)

namespace nnr {

// Forward-declare CUDA resolver functions
#define X(name) operator_t* resolver_cuda_op_##name(int opset, pool_t& pool);
#include "ops.h"
#undef X

namespace {

// Auto-register CUDA operators at static-init time.
// Covers the hot-path operators (MatMul, Gemm, Conv).
// All other ops fall back to CPU via the registry.
struct cuda_registrar {
    cuda_registrar() {
        auto& r = global_registry();
        #define X(name) r.register_op(#name, backend_t::CUDA, resolver_cuda_op_##name);
        #include "ops.h"
        #undef X
    }
} cuda_registrar_instance;

} // namespace

// Anchor function: called from onnx_loader to prevent MSVC dead-stripping.
void cuda_anchor() { (void)cuda_registrar_instance; }

} // namespace nnr

#endif // NNR_USE_CUDA
