// X-macro list of same-shape f32->f32 unary elementwise ops and their WGSL
// expressions. Single source of truth shared by:
//   - unary_math.cpp   — DEFINE_UNARY macro expands these into resolver ops
//   - unary_math.cpp   — fusable_unary_op_expr() switch table
//   - fuse_webgpu_elementwise.cpp — via fusable_unary_op_expr()
//
// Define UNARY(name, expr) before including.
// Expressions use `v` as the input variable. Each expression must be a pure
// function of `v` (no side effects, no extra state), so the fusion framework
// can compose them by reassigning `v` between stages.

UNARY(Relu,        "max(v, 0.0)")
UNARY(Sigmoid,     "1.0 / (1.0 + exp(-v))")
UNARY(Tanh,        "tanh(v)")
UNARY(Abs,         "abs(v)")
UNARY(Neg,         "-v")
UNARY(Exp,         "exp(v)")
UNARY(Log,         "log(v)")
UNARY(Sqrt,        "sqrt(v)")
UNARY(Ceil,        "ceil(v)")
UNARY(Floor,       "floor(v)")
UNARY(Round,       "round(v)")
UNARY(Sin,         "sin(v)")
UNARY(Cos,         "cos(v)")
UNARY(Reciprocal,  "1.0 / v")
UNARY(Softplus,    "log(1.0 + exp(v))")
UNARY(Softsign,    "v / (1.0 + abs(v))")
UNARY(Gelu,        "0.5 * v * (1.0 + tanh(0.7978845608 * (v + 0.044715 * v * v * v)))")
UNARY(HardSigmoid, "clamp(0.2 * v + 0.5, 0.0, 1.0)")
UNARY(HardSwish,   "v * clamp(v / 6.0 + 0.5, 0.0, 1.0)")
UNARY(Elu,         "select(exp(v) - 1.0, v, v >= 0.0)")
UNARY(Mish,        "v * tanh(log(1.0 + exp(v)))")
UNARY(Sign,        "sign(v)")
UNARY(Sinh,        "sinh(v)")
UNARY(Cosh,        "cosh(v)")
UNARY(Tan,         "tan(v)")
UNARY(Asin,        "asin(v)")
UNARY(Acos,        "acos(v)")
UNARY(Atan,        "atan(v)")
UNARY(Asinh,       "asinh(v)")
UNARY(Acosh,       "acosh(v)")
UNARY(Atanh,       "atanh(v)")
UNARY(Erf,
    "sign(v) * (1.0 - exp(-v*v) * ("
        "(1.0 / (1.0 + 0.3275911 * abs(v))) * (0.254829592 + "
        "(1.0 / (1.0 + 0.3275911 * abs(v))) * (-0.284496736 + "
        "(1.0 / (1.0 + 0.3275911 * abs(v))) * (1.421413741 + "
        "(1.0 / (1.0 + 0.3275911 * abs(v))) * (-1.453152027 + "
        "(1.0 / (1.0 + 0.3275911 * abs(v))) *  1.061405429"
        "))))))")
