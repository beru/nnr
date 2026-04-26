#include "elementwise.h"
#include "pool.h"

#include <string_view>

// Unary float32 elementwise ops. Each op is a subclass of
// webgpu::unary_elementwise_t that only overrides op_expr().
//
// The op list + WGSL expressions are in unary_exprs.inc so that the fusion
// pass (fuse_webgpu_elementwise.cpp) can query the same table via
// fusable_unary_op_expr().

namespace nnr {

#define DEFINE_UNARY(name, expr)                                               \
    namespace {                                                                \
    struct name##_op_webgpu : webgpu::unary_elementwise_t {                    \
        const char* op_expr() const override { return expr; }                  \
    };                                                                         \
    }                                                                          \
    operator_t* resolver_default_op_##name##_webgpu(int, pool_t& pool) {       \
        return pool_new<name##_op_webgpu>(pool);                               \
    }

#define UNARY(name, expr) DEFINE_UNARY(name, expr)
#include "unary_exprs.inc"
#undef UNARY

#undef DEFINE_UNARY

} // namespace nnr

namespace nnr::webgpu {

// Returns the WGSL expression for a same-shape f32 unary op type, or nullptr
// if the op isn't a member of the fusable-unary set. The fusion pass uses
// this to (a) decide whether an op is a valid chain member and (b) extract
// its per-stage expression for the composed shader.
const char* fusable_unary_op_expr(std::string_view op_type)
{
#define UNARY(name, expr) if (op_type == #name) return expr;
#include "unary_exprs.inc"
#undef UNARY
    return nullptr;
}

} // namespace nnr::webgpu
