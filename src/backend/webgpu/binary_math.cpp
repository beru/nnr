#include "elementwise.h"
#include "pool.h"

#include <string_view>

// Binary float32 elementwise ops. NumPy/ONNX broadcasting is handled in the
// shared `binary_elementwise_t` base (rank up to 8); each subclass only
// supplies the per-element WGSL expression.
//
// The op list + WGSL expressions are in binary_exprs.inc so the fusion pass
// (fuse_webgpu_elementwise.cpp) can query the same table for pipe-first
// chain composition via fusable_binary_op_pattern().

namespace nnr {

#define DEFINE_BINARY(name, expr)                                              \
    namespace {                                                                \
    struct name##_op_webgpu : webgpu::binary_elementwise_t {                   \
        const char* op_expr() const override { return expr; }                  \
    };                                                                         \
    }                                                                          \
    operator_t* resolver_default_op_##name##_webgpu(int, pool_t& pool) {       \
        return pool_new<name##_op_webgpu>(pool);                               \
    }

DEFINE_BINARY(Add, "a + b")
DEFINE_BINARY(Sub, "a - b")
DEFINE_BINARY(Mul, "a * b")
DEFINE_BINARY(Div, "a / b")
DEFINE_BINARY(Pow, "pow(a, b)")
DEFINE_BINARY(Max, "max(a, b)")
DEFINE_BINARY(Min, "min(a, b)")
// PRelu(x, slope): x if x>=0 else slope*x. Slope is unidirectionally
// broadcast into x (so output shape == x shape as long as the model uses a
// slope that's NumPy-broadcastable to x's shape — e.g. [C,1,1] for NCHW).
DEFINE_BINARY(PRelu, "select(b * a, a, a >= 0.0)")

#undef DEFINE_BINARY

} // namespace nnr

namespace nnr::webgpu {

// Returns the pipe-first WGSL pattern for a same-shape f32 binary op_type,
// or nullptr if the op isn't in the fusable-binary set. Pattern uses `v` as
// the pipe variable (prior stage output) and `$s` as a placeholder for
// the side input access, which the fusion pass substitutes with `S<k>[i]`.
const char* fusable_binary_op_pattern(std::string_view op_type)
{
#define BINARY(name, pattern) if (op_type == #name) return pattern;
#include "binary_exprs.inc"
#undef BINARY
    return nullptr;
}

} // namespace nnr::webgpu
