// WebGPU comparison / logical ops producing u32 masks.
//
// Two families share this file because they share the u32-mask encoding:
//
//   * Comparisons: (f32, f32) → u32   — Equal, Greater, Less, GEQ, LEQ
//   * Logical:     (u32, u32) → u32   — And, Or, Xor  (broadcast)
//                  u32        → u32   — Not
//
// ONNX represents comparison / logical results as BOOL (u8 on CPU); on WebGPU
// we standardize on u32 because storage buffers can't hold u8 and this is
// the same convention used by Where's cond input. A Compare → And → Where
// chain therefore runs entirely on GPU with no CPU round-trip — the mask
// lives only in VRAM as u32, 1=true/0=false.
//
// Logical ops tolerate any non-zero value as "true" (via `!= 0u` guards)
// and always emit canonical 0u / 1u, matching ONNX bool semantics.
//
// Limitations (fall back to CPU via reshape returning false):
//   - Comparisons currently take f32 inputs only. Extending to i32/u32 is
//     trivial; other dtypes need the broader dtype story (see
//     docs/webgpu-continuation.md "Finish the dtype/bool story").
//   - Logical ops expect u32 masks. An ONNX graph carrying bool tensors as
//     u8 on CPU must go through a Cast or bool-upload path (not yet wired).
//   - BOOL-typed outputs consumed by a CPU op. We emit u32, which has a
//     different storage size than BOOL (u8). Safe for pure-GPU chains
//     but a cross-backend chain where a CPU op consumes a WebGPU mask
//     would need bool widening — out of scope for this pass.

#include "elementwise.h"
#include "pool.h"

namespace nnr {

// Comparisons: f32 in, u32 out.
#define DEFINE_COMPARE(name, expr)                                              \
    namespace {                                                                 \
    struct name##_op_webgpu : webgpu::binary_elementwise_t {                    \
        const char* op_expr()        const override { return expr; }            \
        const char* output_wgsl_ty() const override { return "u32"; }           \
        data_type_t output_dtype()   const override { return NNR_DATA_TYPE_UINT32; } \
    };                                                                          \
    }                                                                           \
    operator_t* resolver_default_op_##name##_webgpu(int, pool_t& pool) {        \
        return pool_new<name##_op_webgpu>(pool);                                \
    }

DEFINE_COMPARE(Equal,          "select(0u, 1u, a == b)")
DEFINE_COMPARE(Greater,        "select(0u, 1u, a >  b)")
DEFINE_COMPARE(Less,           "select(0u, 1u, a <  b)")
DEFINE_COMPARE(GreaterOrEqual, "select(0u, 1u, a >= b)")
DEFINE_COMPARE(LessOrEqual,    "select(0u, 1u, a <= b)")

#undef DEFINE_COMPARE

// Logical binary: u32 in, u32 out, broadcast (rank ≤ 8 via the base).
#define DEFINE_LOGICAL_BIN(name, expr)                                          \
    namespace {                                                                 \
    struct name##_op_webgpu : webgpu::binary_elementwise_t {                    \
        const char* op_expr()        const override { return expr; }            \
        const char* input_wgsl_ty()  const override { return "u32"; }           \
        const char* output_wgsl_ty() const override { return "u32"; }           \
        data_type_t input_dtype()    const override { return NNR_DATA_TYPE_UINT32; } \
        data_type_t output_dtype()   const override { return NNR_DATA_TYPE_UINT32; } \
    };                                                                          \
    }                                                                           \
    operator_t* resolver_default_op_##name##_webgpu(int, pool_t& pool) {        \
        return pool_new<name##_op_webgpu>(pool);                                \
    }

DEFINE_LOGICAL_BIN(And, "select(0u, 1u, (a != 0u) && (b != 0u))")
DEFINE_LOGICAL_BIN(Or,  "select(0u, 1u, (a != 0u) || (b != 0u))")
DEFINE_LOGICAL_BIN(Xor, "select(0u, 1u, (a != 0u) != (b != 0u))")

#undef DEFINE_LOGICAL_BIN

// Logical unary: Not — u32 mask inversion. 0 → 1, non-zero → 0.
namespace {
struct Not_op_webgpu : webgpu::unary_elementwise_t {
    const char* op_expr()        const override { return "select(1u, 0u, v != 0u)"; }
    const char* input_wgsl_ty()  const override { return "u32"; }
    const char* output_wgsl_ty() const override { return "u32"; }
    data_type_t input_dtype()    const override { return NNR_DATA_TYPE_UINT32; }
    data_type_t output_dtype()   const override { return NNR_DATA_TYPE_UINT32; }
};
}
operator_t* resolver_default_op_Not_webgpu(int, pool_t& pool) {
    return pool_new<Not_op_webgpu>(pool);
}

// ---------------------------------------------------------------------------
// Bitwise family: true bitwise ops on u32 (vs the logical/mask ops above).
// ONNX BitwiseAnd/Or/Xor/Not are opset-18+ ops. Scope: u32 only — extend
// to i32 when needed by adding input_dtype() override. BitShift uses the
// same base with a direction-dependent expression chosen at init time.

#define DEFINE_BITWISE_BIN(name, expr)                                          \
    namespace {                                                                 \
    struct name##_op_webgpu : webgpu::binary_elementwise_t {                    \
        const char* op_expr()        const override { return expr; }            \
        const char* input_wgsl_ty()  const override { return "u32"; }           \
        const char* output_wgsl_ty() const override { return "u32"; }           \
        data_type_t input_dtype()    const override { return NNR_DATA_TYPE_UINT32; } \
        data_type_t output_dtype()   const override { return NNR_DATA_TYPE_UINT32; } \
    };                                                                          \
    }                                                                           \
    operator_t* resolver_default_op_##name##_webgpu(int, pool_t& pool) {        \
        return pool_new<name##_op_webgpu>(pool);                                \
    }

DEFINE_BITWISE_BIN(BitwiseAnd, "a & b")
DEFINE_BITWISE_BIN(BitwiseOr,  "a | b")
DEFINE_BITWISE_BIN(BitwiseXor, "a ^ b")

#undef DEFINE_BITWISE_BIN

// Unary bitwise NOT on u32.
namespace {
struct BitwiseNot_op_webgpu : webgpu::unary_elementwise_t {
    const char* op_expr()        const override { return "~v"; }
    const char* input_wgsl_ty()  const override { return "u32"; }
    const char* output_wgsl_ty() const override { return "u32"; }
    data_type_t input_dtype()    const override { return NNR_DATA_TYPE_UINT32; }
    data_type_t output_dtype()   const override { return NNR_DATA_TYPE_UINT32; }
};
}
operator_t* resolver_default_op_BitwiseNot_webgpu(int, pool_t& pool) {
    return pool_new<BitwiseNot_op_webgpu>(pool);
}

// BitShift needs the direction attribute to pick left-shift (`<<`) vs
// right-shift (`>>`) at graph-build time. The expression is chosen once in
// init() (before the base compiles the shader) and pinned for this op
// instance. WGSL shift operators require u32 RHS so we're already typed
// correctly via the u32/u32 base.
namespace {
struct BitShift_op_webgpu : webgpu::binary_elementwise_t {
    const char* expr = "a << b";   // default: LEFT

    bool init() override {
        std::string_view dir = attribute(attr_key_t::direction, "LEFT");
        if (dir == "RIGHT")      expr = "a >> b";
        else if (dir == "LEFT")  expr = "a << b";
        else                     return false;   // unknown direction → CPU
        return webgpu::binary_elementwise_t::init();
    }
    const char* op_expr()        const override { return expr; }
    const char* input_wgsl_ty()  const override { return "u32"; }
    const char* output_wgsl_ty() const override { return "u32"; }
    data_type_t input_dtype()    const override { return NNR_DATA_TYPE_UINT32; }
    data_type_t output_dtype()   const override { return NNR_DATA_TYPE_UINT32; }
};
}
operator_t* resolver_default_op_BitShift_webgpu(int, pool_t& pool) {
    return pool_new<BitShift_op_webgpu>(pool);
}

} // namespace nnr
