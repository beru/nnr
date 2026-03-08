#pragma once

#include "tflite_schema.h"
#include "nnr.h"
#include <cstdint>

// TFLite → NNR mapping tables: operator names and data types.

namespace nnr {
namespace tflite {

// Map TFLite BuiltinOperator enum → ONNX op_type string.
// Returns nullptr for unsupported / unmapped operators.
inline const char* builtin_to_onnx(int32_t op) {
    switch (op) {
    case BuiltinOperator_ADD:                         return "Add";
    case BuiltinOperator_AVERAGE_POOL_2D:             return "AveragePool";
    case BuiltinOperator_CONCATENATION:               return "Concat";
    case BuiltinOperator_CONV_2D:                     return "Conv";
    case BuiltinOperator_DEPTHWISE_CONV_2D:           return "Conv"; // group = input_channels
    case BuiltinOperator_FULLY_CONNECTED:             return "Gemm";
    case BuiltinOperator_MAX_POOL_2D:                 return "MaxPool";
    case BuiltinOperator_MUL:                         return "Mul";
    case BuiltinOperator_RELU:                        return "Relu";
    case BuiltinOperator_RELU6:                       return "Clip";
    case BuiltinOperator_RESHAPE:                     return "Reshape";
    case BuiltinOperator_SOFTMAX:                     return "Softmax";
    case BuiltinOperator_LOGISTIC:                    return "Sigmoid";
    case BuiltinOperator_TANH:                        return "Tanh";
    case BuiltinOperator_PAD:                         return "Pad";
    case BuiltinOperator_PADV2:                       return "Pad";
    case BuiltinOperator_MEAN:                        return "ReduceMean";
    case BuiltinOperator_SUB:                         return "Sub";
    case BuiltinOperator_DIV:                         return "Div";
    case BuiltinOperator_TRANSPOSE:                   return "Transpose";
    case BuiltinOperator_RESIZE_BILINEAR:             return "Resize";
    case BuiltinOperator_RESIZE_NEAREST_NEIGHBOR:     return "Resize";
    case BuiltinOperator_SQUEEZE:                     return "Squeeze";
    case BuiltinOperator_STRIDED_SLICE:               return "Slice"; // approximate
    case BuiltinOperator_EXP:                         return "Exp";
    case BuiltinOperator_LOG:                         return "Log";
    case BuiltinOperator_SIN:                         return "Sin";
    case BuiltinOperator_COS:                         return "Cos";
    case BuiltinOperator_SQRT:                        return "Sqrt";
    case BuiltinOperator_RSQRT:                       return "Rsqrt";
    case BuiltinOperator_ABS:                         return "Abs";
    case BuiltinOperator_NEG:                         return "Neg";
    case BuiltinOperator_FLOOR:                       return "Floor";
    case BuiltinOperator_CEIL:                        return "Ceil";
    case BuiltinOperator_ROUND:                       return "Round";
    case BuiltinOperator_CAST:                        return "Cast";
    case BuiltinOperator_GATHER:                      return "Gather";
    case BuiltinOperator_GATHER_ND:                   return "GatherND";
    case BuiltinOperator_SCATTER_ND:                  return "ScatterND";
    case BuiltinOperator_SPLIT:                       return "Split";
    case BuiltinOperator_SPLIT_V:                     return "Split";
    case BuiltinOperator_TILE:                        return "Tile";
    case BuiltinOperator_EXPAND_DIMS:                 return "Unsqueeze";
    case BuiltinOperator_PACK:                        return "Concat"; // pack = stack = concat after unsqueeze
    case BuiltinOperator_UNPACK:                      return "Split";  // unpack along axis
    case BuiltinOperator_SLICE:                       return "Slice";
    case BuiltinOperator_SHAPE:                       return "Shape";
    case BuiltinOperator_MAXIMUM:                     return "Max";
    case BuiltinOperator_MINIMUM:                     return "Min";
    case BuiltinOperator_LESS:                        return "Less";
    case BuiltinOperator_LESS_EQUAL:                  return "LessOrEqual";
    case BuiltinOperator_GREATER:                     return "Greater";
    case BuiltinOperator_GREATER_EQUAL:               return "GreaterOrEqual";
    case BuiltinOperator_EQUAL:                       return "Equal";
    case BuiltinOperator_NOT_EQUAL:                   return "NotEqual";
    case BuiltinOperator_LOGICAL_AND:                 return "And";
    case BuiltinOperator_LOGICAL_OR:                  return "Or";
    case BuiltinOperator_LOGICAL_NOT:                 return "Not";
    case BuiltinOperator_SELECT:                      return "Where";
    case BuiltinOperator_SELECT_V2:                   return "Where";
    case BuiltinOperator_POW:                         return "Pow";
    case BuiltinOperator_ARG_MAX:                     return "ArgMax";
    case BuiltinOperator_ARG_MIN:                     return "ArgMin";
    case BuiltinOperator_TOPK_V2:                     return "TopK";
    case BuiltinOperator_SUM:                         return "ReduceSum";
    case BuiltinOperator_REDUCE_PROD:                 return "ReduceProd";
    case BuiltinOperator_REDUCE_MAX:                  return "ReduceMax";
    case BuiltinOperator_REDUCE_MIN:                  return "ReduceMin";
    case BuiltinOperator_REDUCE_ANY:                  return "ReduceSum"; // approximate
    case BuiltinOperator_RANGE:                       return "Range";
    case BuiltinOperator_FILL:                        return "Expand";
    case BuiltinOperator_BATCH_MATMUL:                return "MatMul";
    case BuiltinOperator_ONE_HOT:                     return "OneHot";
    case BuiltinOperator_DEPTH_TO_SPACE:              return "DepthToSpace";
    case BuiltinOperator_SPACE_TO_DEPTH:              return "SpaceToDepth";
    case BuiltinOperator_TRANSPOSE_CONV:              return "ConvTranspose";
    case BuiltinOperator_MIRROR_PAD:                  return "Pad";
    case BuiltinOperator_LEAKY_RELU:                  return "LeakyRelu";
    case BuiltinOperator_PRELU:                       return "PRelu";
    case BuiltinOperator_ELU:                         return "Elu";
    case BuiltinOperator_HARD_SWISH:                  return "HardSwish";
    case BuiltinOperator_DEQUANTIZE:                  return "DequantizeLinear";
    case BuiltinOperator_QUANTIZE:                    return "QuantizeLinear";
    case BuiltinOperator_BATCH_TO_SPACE_ND:           return "BatchToSpace";
    case BuiltinOperator_SPACE_TO_BATCH_ND:           return "SpaceToBatch";
    case BuiltinOperator_CUMSUM:                      return "CumSum";
    case BuiltinOperator_REVERSE_V2:                  return "ReverseSequence";
    case BuiltinOperator_SQUARED_DIFFERENCE:          return "SquaredDifference";
    case BuiltinOperator_ADD_N:                       return "Sum";
    case BuiltinOperator_L2_NORMALIZATION:            return "LpNormalization";
    case BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION:return "LRN";
    case BuiltinOperator_SIGN:                        return "Sign";
    case BuiltinOperator_GELU:                        return "Gelu";
    default:                                          return nullptr;
    }
}

// Map TFLite TensorType → NNR data_type_t.
// Returns NNR_DATA_TYPE_UNDEFINED for unmapped types.
inline data_type_t tflite_type_to_nnr(int8_t tt) {
    switch (tt) {
    case TensorType_FLOAT32:    return NNR_DATA_TYPE_FLOAT32;
    case TensorType_FLOAT16:    return NNR_DATA_TYPE_FLOAT16;
    case TensorType_FLOAT64:    return NNR_DATA_TYPE_FLOAT64;
    case TensorType_INT8:       return NNR_DATA_TYPE_INT8;
    case TensorType_INT16:      return NNR_DATA_TYPE_INT16;
    case TensorType_INT32:      return NNR_DATA_TYPE_INT32;
    case TensorType_INT64:      return NNR_DATA_TYPE_INT64;
    case TensorType_UINT8:      return NNR_DATA_TYPE_UINT8;
    case TensorType_UINT16:     return NNR_DATA_TYPE_UINT16;
    case TensorType_UINT32:     return NNR_DATA_TYPE_UINT32;
    case TensorType_UINT64:     return NNR_DATA_TYPE_UINT64;
    case TensorType_BOOL:       return NNR_DATA_TYPE_BOOL;
    case TensorType_STRING:     return NNR_DATA_TYPE_STRING;
    case TensorType_COMPLEX64:  return NNR_DATA_TYPE_COMPLEX64;
    case TensorType_COMPLEX128: return NNR_DATA_TYPE_COMPLEX128;
    case TensorType_BFLOAT16:   return NNR_DATA_TYPE_BFLOAT16;
    case TensorType_INT4:       return NNR_DATA_TYPE_INT4;
    default:                    return NNR_DATA_TYPE_UNDEFINED;
    }
}

// Map TFLite fused activation function to ONNX op_type for post-op insertion.
// Returns nullptr for NONE (no activation).
inline const char* fused_activation_op(int8_t act) {
    switch (act) {
    case ActivationFunctionType_RELU:          return "Relu";
    case ActivationFunctionType_RELU6:         return "Clip"; // Clip(0, 6)
    case ActivationFunctionType_RELU_N1_TO_1:  return "Clip"; // Clip(-1, 1)
    case ActivationFunctionType_TANH:          return "Tanh";
    default:                                   return nullptr;
    }
}

} // namespace tflite
} // namespace nnr
