// Consolidated comparison operators.
// Each uses comparison_op_t CRTP base: init=is_inout_size(2,1), reshape=multi_broadcast(BOOL), exec=binary_broadcast_exec<T,bool_t>.

#include "nnr.h"
#include "util.h"

#define NNR_COMPARISON_OP(Name, fn_body, ...) \
namespace { struct Name##_op : public comparison_op_t<Name##_op, __VA_ARGS__> { \
    static bool fn(auto a, auto b) { return fn_body; } \
}; } \
operator_t* resolver_default_op_##Name(int opset, pool_t& pool) { return pool_new<Name##_op>(pool); }

namespace nnr {

// @nnr-meta-op op=Equal mt=no
NNR_COMPARISON_OP(Equal, a == b,
    opset_t<13, bool_t, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, bfloat16_t, float16_t, float, double, std::string>,
    opset_t<11, bool_t, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float16_t, float, double, std::string>,
    opset_t<7, bool_t, int32_t, int64_t>)
// @nnr-meta-op op=Greater mt=no
NNR_COMPARISON_OP(Greater, a > b,
    opset_t<13, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, bfloat16_t, float16_t, float, double>,
    opset_t<9, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float16_t, float, double>,
    opset_t<7, float16_t, float, double>)
// @nnr-meta-op op=GreaterOrEqual mt=no
NNR_COMPARISON_OP(GreaterOrEqual, a >= b,
    opset_t<12, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float16_t, float, double>)
// @nnr-meta-op op=Less mt=no
NNR_COMPARISON_OP(Less, a < b,
    opset_t<13, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, bfloat16_t, float16_t, float, double>,
    opset_t<9, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float16_t, float, double>,
    opset_t<7, float16_t, float, double>)
// @nnr-meta-op op=LessOrEqual mt=no
NNR_COMPARISON_OP(LessOrEqual, a <= b,
    opset_t<12, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float16_t, float, double>)

// Logical (bool_t → bool_t)
// @nnr-meta-op op=And mt=no
NNR_COMPARISON_OP(And, a && b, opset_t<7, bool_t>)
// @nnr-meta-op op=Or mt=no
NNR_COMPARISON_OP(Or,  a || b, opset_t<7, bool_t>)
// @nnr-meta-op op=Xor mt=no
NNR_COMPARISON_OP(Xor, a != b, opset_t<7, bool_t>)

#undef NNR_COMPARISON_OP

} // namespace nnr
