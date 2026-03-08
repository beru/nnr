// Consolidated simple unary element-wise operators.
// Each uses unary_foreach_op_t CRTP base: init=is_inout_size(1,1), exec=foreach_tensor.
// Complex unary ops with fusable_apply/scroll (Relu, Sigmoid, Tanh) remain in their own files.

#include <cmath>
#include <algorithm>
#include "nnr.h"
#include "util.h"

#define NNR_UNARY_OP(Name, fn_body, ...) \
namespace { struct Name##_op : public unary_foreach_op_t<Name##_op, __VA_ARGS__> { \
    static auto fn(auto x) { return fn_body; } \
}; } \
operator_t* resolver_default_op_##Name(int opset, pool_t& pool) { return pool_new<Name##_op>(pool); }

namespace nnr {

// Trig
// @nnr-meta-op op=Sin mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Sin,   sin(x),   opset_t<7, float16_t, float, double>)
// @nnr-meta-op op=Cos mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Cos,   cos(x),   opset_t<7, float16_t, float, double>)
// @nnr-meta-op op=Tan mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Tan,   tan(x),   opset_t<7, float16_t, float, double>)
// @nnr-meta-op op=Asin mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Asin,  asin(x),  opset_t<7, float16_t, float, double>)
// @nnr-meta-op op=Acos mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Acos,  acos(x),  opset_t<7, float16_t, float, double>)
// @nnr-meta-op op=Atan mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Atan,  atan(x),  opset_t<7, float16_t, float, double>)

// Hyperbolic
// @nnr-meta-op op=Sinh mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Sinh,  sinh(x),  opset_t<9, float16_t, float, double>)
// @nnr-meta-op op=Cosh mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Cosh,  cosh(x),  opset_t<9, float16_t, float, double>)
// @nnr-meta-op op=Asinh mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Asinh, asinh(x), opset_t<9, float16_t, float, double>)
// @nnr-meta-op op=Acosh mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Acosh, acosh(x), opset_t<9, float16_t, float, double>)
// @nnr-meta-op op=Atanh mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Atanh, atanh(x), opset_t<9, float16_t, float, double>)

// Exponential / logarithmic
// @nnr-meta-op op=Exp mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Exp,  exp(x),  opset_t<13, bfloat16_t, float16_t, float, double>, opset_t<1, float16_t, float, double>)
// @nnr-meta-op op=Log mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Log,  log(x),  opset_t<13, bfloat16_t, float16_t, float, double>, opset_t<1, float16_t, float, double>)
// @nnr-meta-op op=Sqrt mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Sqrt, sqrt(x), opset_t<13, bfloat16_t, float16_t, float, double>, opset_t<1, float16_t, float, double>)

// Rounding
// @nnr-meta-op op=Ceil mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Ceil,  ceil(x),  opset_t<13, bfloat16_t, float16_t, float, double>, opset_t<1, float16_t, float, double>)
// @nnr-meta-op op=Floor mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Floor, floor(x), opset_t<13, bfloat16_t, float16_t, float, double>, opset_t<1, float16_t, float, double>)

// Arithmetic
// @nnr-meta-op op=Neg mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Neg, -x,
    opset_t<13, int8_t, int16_t, int32_t, int64_t, bfloat16_t, float16_t, float, double>,
    opset_t<6, int8_t, int16_t, int32_t, int64_t, float16_t, float, double>,
    opset_t<1, float16_t, float, double>)
// @nnr-meta-op op=Reciprocal mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Reciprocal, decltype(x)(1.0) / x,
    opset_t<13, bfloat16_t, float16_t, float, double>,
    opset_t<1, float16_t, float, double>)

// Special
// @nnr-meta-op op=Erf mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Erf, erf(x),
    opset_t<13, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, bfloat16_t, float16_t, float, double>,
    opset_t<9, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float16_t, float, double>)
// @nnr-meta-op op=Softplus mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Softplus, log(exp(x) + 1), opset_t<1, float16_t, float, double>)
// @nnr-meta-op op=Softsign mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Softsign, x / (1 + fabs(x)), opset_t<1, float16_t, float, double>)

// Activation
// HardSwish moved to HardSwish.cpp (needs fusable_apply for Conv fusion)
// @nnr-meta-op op=Mish mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Mish, (double)x * std::tanh(std::log1p(std::exp((double)x))),
    opset_t<1, float16_t, float, double>)
// @nnr-meta-op op=Swish mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Swish, (double)x / (1.0 + std::exp(-(double)x)),
    opset_t<1, float16_t, float, double, bfloat16_t>)

// Bitwise / sign
// @nnr-meta-op op=Sign mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(Sign, (x > 0) - (x < 0),
    opset_t<13, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, bfloat16_t, float16_t, float, double>,
    opset_t<9, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float16_t, float, double>)
// @nnr-meta-op op=BitwiseNot mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
NNR_UNARY_OP(BitwiseNot, ~x,
    opset_t<1, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t>)

#undef NNR_UNARY_OP

} // namespace nnr
