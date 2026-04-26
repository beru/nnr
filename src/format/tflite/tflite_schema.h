#pragma once

#include <cstdint>
#include <cstring>
#include <string_view>

// Hand-rolled read-only FlatBuffer accessors for the TFLite schema.
// FlatBuffers are little-endian, zero-copy: every accessor reads directly
// from the mmap'd / loaded buffer. No allocation, no codegen dependency.
//
// Only the subset needed by the loader is implemented.

namespace nnr {
namespace tflite {

// --- Buffer bounds context ---------------------------------------------------
// Created on the stack by the loader. Stored by pointer in every table/vector
// so that all pointer arithmetic is bounds-checked without global state.

struct fb_ctx {
    const uint8_t* start = nullptr;
    const uint8_t* end   = nullptr;

    fb_ctx() = default;
    fb_ctx(const uint8_t* buf, size_t size) : start(buf), end(buf + size) {}

    bool in_bounds(const void* p, size_t n = 1) const {
        auto* bp = (const uint8_t*)p;
        return bp >= start && bp + n >= bp && bp + n <= end;
    }
};

// --- Low-level FlatBuffer primitives (raw, unchecked — callers validate) -----

inline uint8_t  fb_u8 (const void* p) { uint8_t  v; memcpy(&v, p, 1); return v; }
inline int8_t   fb_i8 (const void* p) { int8_t   v; memcpy(&v, p, 1); return v; }
inline uint16_t fb_u16(const void* p) { uint16_t v; memcpy(&v, p, 2); return v; }
inline int16_t  fb_i16(const void* p) { int16_t  v; memcpy(&v, p, 2); return v; }
inline uint32_t fb_u32(const void* p) { uint32_t v; memcpy(&v, p, 4); return v; }
inline int32_t  fb_i32(const void* p) { int32_t  v; memcpy(&v, p, 4); return v; }
inline float    fb_f32(const void* p) { float    v; memcpy(&v, p, 4); return v; }

// --- fb_table ----------------------------------------------------------------

struct fb_table {
    const uint8_t* buf;
    const fb_ctx*  ctx;

    fb_table() : buf(nullptr), ctx(nullptr) {}
    fb_table(const uint8_t* p, const fb_ctx* c) : buf(p), ctx(c) {}
    explicit operator bool() const { return buf != nullptr; }

    bool ok(const void* p, size_t n) const { return ctx && ctx->in_bounds(p, n); }

    const uint8_t* vtable() const {
        if (!buf || !ok(buf, 4)) return nullptr;
        int32_t vt_off = fb_i32(buf);
        const uint8_t* vt = buf - vt_off;
        if (!ok(vt, 4)) return nullptr;
        return vt;
    }

    uint16_t field_off(int field_index) const {
        const uint8_t* vt = vtable();
        if (!vt) return 0;
        uint16_t vt_size = fb_u16(vt);
        uint16_t byte_off = 4 + field_index * 2;
        if (byte_off >= vt_size) return 0;
        if (!ok(vt + byte_off, 2)) return 0;
        return fb_u16(vt + byte_off);
    }

    int32_t field_i32(int fi, int32_t def = 0) const {
        uint16_t off = field_off(fi);
        if (!off) return def;
        const uint8_t* p = buf + off;
        return ok(p, 4) ? fb_i32(p) : def;
    }
    int8_t field_i8(int fi, int8_t def = 0) const {
        uint16_t off = field_off(fi);
        if (!off) return def;
        const uint8_t* p = buf + off;
        return ok(p, 1) ? fb_i8(p) : def;
    }
    float field_f32(int fi, float def = 0.0f) const {
        uint16_t off = field_off(fi);
        if (!off) return def;
        const uint8_t* p = buf + off;
        return ok(p, 4) ? fb_f32(p) : def;
    }

    const uint8_t* field_ptr(int fi) const {
        uint16_t off = field_off(fi);
        if (!off) return nullptr;
        const uint8_t* p = buf + off;
        if (!ok(p, 4)) return nullptr;
        uint32_t rel = fb_u32(p);
        const uint8_t* target = p + rel;
        if (!ok(target, 1)) return nullptr;
        return target;
    }

    // Returns a string_view bounded by the FlatBuffer stored length. Does not
    // assume a null terminator exists or is in bounds — FlatBuffer writers are
    // supposed to include one, but a crafted buffer may omit it, so callers
    // must not pass the .data() pointer to strlen or C string APIs.
    std::string_view field_str(int fi) const {
        const uint8_t* p = field_ptr(fi);
        if (!p || !ok(p, 4)) return {};
        uint32_t len = fb_u32(p);
        if (!ok(p + 4, len)) return {};
        return std::string_view((const char*)(p + 4), len);
    }

    int32_t field_str_len(int fi) const {
        const uint8_t* p = field_ptr(fi);
        if (!p || !ok(p, 4)) return 0;
        return fb_i32(p);
    }

    fb_table field_table(int fi) const {
        return fb_table(field_ptr(fi), ctx);
    }
};

// --- fb_vec (vector of offset tables/strings) --------------------------------

struct fb_vec {
    const uint8_t* buf;
    const fb_ctx*  ctx;

    fb_vec() : buf(nullptr), ctx(nullptr) {}
    fb_vec(const uint8_t* p, const fb_ctx* c) : buf(p), ctx(c) {}

    bool ok(const void* p, size_t n) const { return ctx && ctx->in_bounds(p, n); }

    uint32_t size() const {
        if (!buf || !ok(buf, 4)) return 0;
        return fb_u32(buf);
    }

    fb_table operator[](uint32_t i) const {
        if (i >= size()) return fb_table(nullptr, ctx);
        const uint8_t* p = buf + 4 + (size_t)i * 4;
        if (!ok(p, 4)) return fb_table(nullptr, ctx);
        uint32_t off = fb_u32(p);
        const uint8_t* target = p + off;
        if (!ok(target, 4)) return fb_table(nullptr, ctx);
        return fb_table(target, ctx);
    }

    // Returns a string_view bounded by the FlatBuffer stored length. Does not
    // assume a trailing null terminator is in bounds — callers must not treat
    // .data() as a C string.
    std::string_view str(uint32_t i) const {
        if (i >= size()) return {};
        const uint8_t* p = buf + 4 + (size_t)i * 4;
        if (!ok(p, 4)) return {};
        const uint8_t* s = p + fb_u32(p);
        if (!ok(s, 4)) return {};
        uint32_t len = fb_u32(s);
        if (!ok(s + 4, len)) return {};
        return std::string_view((const char*)(s + 4), len);
    }
};

// --- fb_vec_i32 (vector of int32 scalars) ------------------------------------

struct fb_vec_i32 {
    const uint8_t* buf;
    const fb_ctx*  ctx;

    fb_vec_i32() : buf(nullptr), ctx(nullptr) {}
    fb_vec_i32(const uint8_t* p, const fb_ctx* c) : buf(p), ctx(c) {}

    bool ok(const void* p, size_t n) const { return ctx && ctx->in_bounds(p, n); }

    uint32_t size() const {
        if (!buf || !ok(buf, 4)) return 0;
        return fb_u32(buf);
    }
    int32_t operator[](uint32_t i) const {
        if (i >= size()) return 0;
        const uint8_t* p = buf + 4 + (size_t)i * 4;
        return ok(p, 4) ? fb_i32(p) : 0;
    }
};

// --- fb_vec_u8 (vector of uint8 scalars) -------------------------------------

struct fb_vec_u8 {
    const uint8_t* buf;
    const fb_ctx*  ctx;

    fb_vec_u8() : buf(nullptr), ctx(nullptr) {}
    fb_vec_u8(const uint8_t* p, const fb_ctx* c) : buf(p), ctx(c) {}

    bool ok(const void* p, size_t n) const { return ctx && ctx->in_bounds(p, n); }

    uint32_t size() const {
        if (!buf || !ok(buf, 4)) return 0;
        return fb_u32(buf);
    }
    uint8_t operator[](uint32_t i) const {
        if (i >= size()) return 0;
        const uint8_t* p = buf + 4 + i;
        return ok(p, 1) ? *p : 0;
    }
    const uint8_t* data() const {
        if (!buf) return nullptr;
        uint32_t sz = size();
        const uint8_t* p = buf + 4;
        return (sz > 0 && ok(p, sz)) ? p : nullptr;
    }
};

// --- TFLite schema tables ---------------------------------------------------

// BuiltinOperator enum (subset — extend as needed)
enum BuiltinOperator : int32_t {
    BuiltinOperator_ADD = 0,
    BuiltinOperator_AVERAGE_POOL_2D = 1,
    BuiltinOperator_CONCATENATION = 2,
    BuiltinOperator_CONV_2D = 3,
    BuiltinOperator_DEPTHWISE_CONV_2D = 4,
    BuiltinOperator_DEPTH_TO_SPACE = 5,
    BuiltinOperator_DEQUANTIZE = 6,
    BuiltinOperator_EMBEDDING_LOOKUP = 7,
    BuiltinOperator_FLOOR = 8,
    BuiltinOperator_FULLY_CONNECTED = 9,
    BuiltinOperator_HASHTABLE_LOOKUP = 10,
    BuiltinOperator_L2_NORMALIZATION = 11,
    BuiltinOperator_L2_POOL_2D = 12,
    BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION = 13,
    BuiltinOperator_LOGISTIC = 14,
    BuiltinOperator_LSH_PROJECTION = 15,
    BuiltinOperator_LSTM = 16,
    BuiltinOperator_MAX_POOL_2D = 17,
    BuiltinOperator_MUL = 18,
    BuiltinOperator_RELU = 19,
    BuiltinOperator_RELU_N1_TO_1 = 20,
    BuiltinOperator_RELU6 = 21,
    BuiltinOperator_RESHAPE = 22,
    BuiltinOperator_RESIZE_BILINEAR = 23,
    BuiltinOperator_RNN = 24,
    BuiltinOperator_SOFTMAX = 25,
    BuiltinOperator_SPACE_TO_DEPTH = 26,
    BuiltinOperator_SVDF = 27,
    BuiltinOperator_TANH = 28,
    BuiltinOperator_CONCAT_EMBEDDINGS = 29,
    BuiltinOperator_SKIP_GRAM = 30,
    BuiltinOperator_CALL = 31,
    BuiltinOperator_CUSTOM = 32,
    BuiltinOperator_EMBEDDING_LOOKUP_SPARSE = 33,
    BuiltinOperator_PAD = 34,
    BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN = 35,
    BuiltinOperator_GATHER = 36,
    BuiltinOperator_BATCH_TO_SPACE_ND = 37,
    BuiltinOperator_SPACE_TO_BATCH_ND = 38,
    BuiltinOperator_TRANSPOSE = 39,
    BuiltinOperator_MEAN = 40,
    BuiltinOperator_SUB = 41,
    BuiltinOperator_DIV = 42,
    BuiltinOperator_SQUEEZE = 43,
    BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM = 44,
    BuiltinOperator_STRIDED_SLICE = 45,
    BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN = 46,
    BuiltinOperator_EXP = 47,
    BuiltinOperator_TOPK_V2 = 48,
    BuiltinOperator_SPLIT = 49,
    BuiltinOperator_LOG_SOFTMAX = 50,
    BuiltinOperator_DELEGATE = 51,
    BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM = 52,
    BuiltinOperator_CAST = 53,
    BuiltinOperator_PRELU = 54,
    BuiltinOperator_MAXIMUM = 55,
    BuiltinOperator_ARG_MAX = 56,
    BuiltinOperator_MINIMUM = 57,
    BuiltinOperator_LESS = 58,
    BuiltinOperator_NEG = 59,
    BuiltinOperator_PADV2 = 60,
    BuiltinOperator_GREATER = 61,
    BuiltinOperator_GREATER_EQUAL = 62,
    BuiltinOperator_LESS_EQUAL = 63,
    BuiltinOperator_SELECT = 64,
    BuiltinOperator_SLICE = 65,
    BuiltinOperator_SIN = 66,
    BuiltinOperator_TRANSPOSE_CONV = 67,
    BuiltinOperator_SPARSE_TO_DENSE = 68,
    BuiltinOperator_TILE = 69,
    BuiltinOperator_EXPAND_DIMS = 70,
    BuiltinOperator_EQUAL = 71,
    BuiltinOperator_NOT_EQUAL = 72,
    BuiltinOperator_LOG = 73,
    BuiltinOperator_SUM = 74,
    BuiltinOperator_SQRT = 75,
    BuiltinOperator_RSQRT = 76,
    BuiltinOperator_SHAPE = 77,
    BuiltinOperator_POW = 78,
    BuiltinOperator_ARG_MIN = 79,
    BuiltinOperator_FAKE_QUANT = 80,
    BuiltinOperator_REDUCE_PROD = 81,
    BuiltinOperator_REDUCE_MAX = 82,
    BuiltinOperator_PACK = 83,
    BuiltinOperator_LOGICAL_OR = 84,
    BuiltinOperator_ONE_HOT = 85,
    BuiltinOperator_LOGICAL_AND = 86,
    BuiltinOperator_LOGICAL_NOT = 87,
    BuiltinOperator_UNPACK = 88,
    BuiltinOperator_REDUCE_MIN = 89,
    BuiltinOperator_FLOOR_DIV = 90,
    BuiltinOperator_REDUCE_ANY = 91,
    BuiltinOperator_SQUARE = 92,
    BuiltinOperator_ZEROS_LIKE = 93,
    BuiltinOperator_FILL = 94,
    BuiltinOperator_FLOOR_MOD = 95,
    BuiltinOperator_RANGE = 96,
    BuiltinOperator_RESIZE_NEAREST_NEIGHBOR = 97,
    BuiltinOperator_LEAKY_RELU = 98,
    BuiltinOperator_SQUARED_DIFFERENCE = 99,
    BuiltinOperator_MIRROR_PAD = 100,
    BuiltinOperator_ABS = 101,
    BuiltinOperator_SPLIT_V = 102,
    BuiltinOperator_UNIQUE = 103,
    BuiltinOperator_CEIL = 104,
    BuiltinOperator_REVERSE_V2 = 105,
    BuiltinOperator_ADD_N = 106,
    BuiltinOperator_GATHER_ND = 107,
    BuiltinOperator_COS = 108,
    BuiltinOperator_WHERE = 109,
    BuiltinOperator_RANK = 110,
    BuiltinOperator_ELU = 111,
    BuiltinOperator_REVERSE_SEQUENCE = 112,
    BuiltinOperator_MATRIX_DIAG = 113,
    BuiltinOperator_QUANTIZE = 114,
    BuiltinOperator_MATRIX_SET_DIAG = 115,
    BuiltinOperator_ROUND = 116,
    BuiltinOperator_HARD_SWISH = 117,
    BuiltinOperator_IF = 118,
    BuiltinOperator_WHILE = 119,
    BuiltinOperator_NON_MAX_SUPPRESSION_V4 = 120,
    BuiltinOperator_NON_MAX_SUPPRESSION_V5 = 121,
    BuiltinOperator_SCATTER_ND = 122,
    BuiltinOperator_SELECT_V2 = 123,
    BuiltinOperator_DENSIFY = 124,
    BuiltinOperator_SEGMENT_SUM = 125,
    BuiltinOperator_BATCH_MATMUL = 126,
    BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES = 127,
    BuiltinOperator_CUMSUM = 128,
    BuiltinOperator_CALL_ONCE = 129,
    BuiltinOperator_BROADCAST_TO = 130,
    BuiltinOperator_RFFT2D = 131,
    BuiltinOperator_CONV_3D = 132,
    BuiltinOperator_IMAG = 133,
    BuiltinOperator_REAL = 134,
    BuiltinOperator_COMPLEX_ABS = 135,
    BuiltinOperator_HASHTABLE = 136,
    BuiltinOperator_HASHTABLE_FIND = 137,
    BuiltinOperator_HASHTABLE_IMPORT = 138,
    BuiltinOperator_HASHTABLE_SIZE = 139,
    BuiltinOperator_REDUCE_ALL = 140,
    BuiltinOperator_CONV_3D_TRANSPOSE = 141,
    BuiltinOperator_VAR_HANDLE = 142,
    BuiltinOperator_READ_VARIABLE = 143,
    BuiltinOperator_ASSIGN_VARIABLE = 144,
    BuiltinOperator_BROADCAST_ARGS = 145,
    BuiltinOperator_RANDOM_STANDARD_NORMAL = 146,
    BuiltinOperator_BUCKETIZE = 147,
    BuiltinOperator_RANDOM_UNIFORM = 148,
    BuiltinOperator_MULTINOMIAL = 149,
    BuiltinOperator_GELU = 150,
    BuiltinOperator_DYNAMIC_UPDATE_SLICE = 151,
    BuiltinOperator_RELU_0_TO_1 = 152,
    BuiltinOperator_UNSORTED_SEGMENT_PROD = 153,
    BuiltinOperator_UNSORTED_SEGMENT_MAX = 154,
    BuiltinOperator_UNSORTED_SEGMENT_SUM = 155,
    BuiltinOperator_ATAN2 = 156,
    BuiltinOperator_UNSORTED_SEGMENT_MIN = 157,
    BuiltinOperator_SIGN = 158,
};

// Padding enum
enum Padding : int8_t {
    Padding_SAME = 0,
    Padding_VALID = 1,
};

// ActivationFunctionType enum
enum ActivationFunctionType : int8_t {
    ActivationFunctionType_NONE = 0,
    ActivationFunctionType_RELU = 1,
    ActivationFunctionType_RELU_N1_TO_1 = 2,
    ActivationFunctionType_RELU6 = 3,
    ActivationFunctionType_TANH = 4,
    ActivationFunctionType_SIGN_BIT = 5,
};

// TensorType enum
enum TensorType : int8_t {
    TensorType_FLOAT32 = 0, TensorType_FLOAT16 = 1, TensorType_INT32 = 2,
    TensorType_UINT8 = 3, TensorType_INT64 = 4, TensorType_STRING = 5,
    TensorType_BOOL = 6, TensorType_INT16 = 7, TensorType_COMPLEX64 = 8,
    TensorType_INT8 = 9, TensorType_FLOAT64 = 10, TensorType_COMPLEX128 = 11,
    TensorType_UINT64 = 12, TensorType_RESOURCE = 13, TensorType_UINT16 = 14,
    TensorType_UINT32 = 15, TensorType_INT4 = 16, TensorType_BFLOAT16 = 17,
};

// BuiltinOptions type enum
enum BuiltinOptions : uint8_t {
    BuiltinOptions_NONE = 0,
    BuiltinOptions_Conv2DOptions = 1,
    BuiltinOptions_DepthwiseConv2DOptions = 2,
    BuiltinOptions_ConcatenationOptions = 3,
    BuiltinOptions_SoftmaxOptions = 4,
    BuiltinOptions_Pool2DOptions = 5,
    BuiltinOptions_ReshapeOptions = 6,
    BuiltinOptions_FullyConnectedOptions = 8,
    BuiltinOptions_AddOptions = 18,
    BuiltinOptions_MulOptions = 19,
    BuiltinOptions_PadOptions = 22,
    BuiltinOptions_SubOptions = 28,
    BuiltinOptions_DivOptions = 29,
    BuiltinOptions_TransposeOptions = 35,
    BuiltinOptions_MeanOptions = 37,
    BuiltinOptions_SqueezeOptions = 38,
    BuiltinOptions_StridedSliceOptions = 39,
    BuiltinOptions_CastOptions = 43,
    BuiltinOptions_ResizeBilinearOptions = 48,
    BuiltinOptions_ResizeNearestNeighborOptions = 75,
    BuiltinOptions_TransposeConvOptions = 52,
    BuiltinOptions_SliceOptions = 55,
    BuiltinOptions_TileOptions = 56,
    BuiltinOptions_ExpandDimsOptions = 57,
    BuiltinOptions_SplitOptions = 46,
    BuiltinOptions_SplitVOptions = 68,
    BuiltinOptions_ShapeOptions = 62,
    BuiltinOptions_PackOptions = 63,
    BuiltinOptions_UnpackOptions = 65,
    BuiltinOptions_GatherOptions = 36,
    BuiltinOptions_BatchMatMulOptions = 86,
    BuiltinOptions_LeakyReluOptions = 69,
};

// --- High-level table accessors (field indices from TFLite schema.fbs) ------

struct Model {
    fb_table t;
    explicit Model(const uint8_t* buf, const fb_ctx& c) {
        if (!c.in_bounds(buf, 8)) return;
        uint32_t root_off = fb_u32(buf);
        const uint8_t* root = buf + root_off;
        if (c.in_bounds(root, 4))
            t = fb_table(root, &c);
    }
    int32_t version() const { return t.field_i32(0, 0); }
    fb_vec operator_codes() const { return fb_vec(t.field_ptr(1), t.ctx); }
    fb_vec subgraphs() const { return fb_vec(t.field_ptr(2), t.ctx); }
    std::string_view description() const { return t.field_str(3); }
    fb_vec buffers() const { return fb_vec(t.field_ptr(4), t.ctx); }
};

struct OperatorCode {
    fb_table t;
    explicit OperatorCode(fb_table tab) : t(tab) {}
    int32_t builtin_code() const {
        int32_t code = t.field_i32(4, -1);
        if (code >= 0) return code;
        return (int32_t)t.field_i8(0, 0);
    }
    std::string_view custom_code() const { return t.field_str(1); }
    int32_t version() const { return t.field_i32(2, 1); }
};

struct SubGraph {
    fb_table t;
    explicit SubGraph(fb_table tab) : t(tab) {}
    fb_vec tensors() const { return fb_vec(t.field_ptr(0), t.ctx); }
    fb_vec_i32 inputs() const { return fb_vec_i32(t.field_ptr(1), t.ctx); }
    fb_vec_i32 outputs() const { return fb_vec_i32(t.field_ptr(2), t.ctx); }
    fb_vec operators() const { return fb_vec(t.field_ptr(3), t.ctx); }
    std::string_view name() const { return t.field_str(4); }
};

struct Tensor {
    fb_table t;
    explicit Tensor(fb_table tab) : t(tab) {}
    fb_vec_i32 shape() const { return fb_vec_i32(t.field_ptr(0), t.ctx); }
    int8_t type() const { return t.field_i8(1, 0); }
    uint32_t buffer() const { return (uint32_t)t.field_i32(2, 0); }
    std::string_view name() const { return t.field_str(3); }
    fb_table quantization() const { return t.field_table(4); }
    fb_vec_i32 shape_signature() const { return fb_vec_i32(t.field_ptr(11), t.ctx); }
};

struct QuantizationParameters {
    fb_table t;
    explicit QuantizationParameters(fb_table tab) : t(tab) {}
    const uint8_t* scale_vec() const { return t.field_ptr(2); }
    const uint8_t* zero_point_vec() const { return t.field_ptr(3); }
    int32_t quantized_dimension() const { return t.field_i32(6, 0); }
    uint32_t scale_count() const {
        auto p = scale_vec();
        return (p && t.ok(p, 4)) ? fb_u32(p) : 0;
    }
    float scale(uint32_t i) const {
        auto p = scale_vec();
        if (!p || !t.ok(p, 4)) return 0.0f;
        uint32_t count = fb_u32(p);
        if (i >= count) return 0.0f;
        const uint8_t* elem = p + 4 + (size_t)i * 4;
        return t.ok(elem, 4) ? fb_f32(elem) : 0.0f;
    }
    uint32_t zero_point_count() const {
        auto p = zero_point_vec();
        return (p && t.ok(p, 4)) ? fb_u32(p) : 0;
    }
    int64_t zero_point(uint32_t i) const {
        auto p = zero_point_vec();
        if (!p || !t.ok(p, 4)) return 0;
        uint32_t count = fb_u32(p);
        if (i >= count) return 0;
        const uint8_t* elem = p + 4 + (size_t)i * 8;
        if (!t.ok(elem, 8)) return 0;
        int64_t v; memcpy(&v, elem, 8); return v;
    }
};

struct Buffer {
    fb_table t;
    explicit Buffer(fb_table tab) : t(tab) {}
    fb_vec_u8 data() const { return fb_vec_u8(t.field_ptr(0), t.ctx); }
};

struct Operator {
    fb_table t;
    explicit Operator(fb_table tab) : t(tab) {}
    uint32_t opcode_index() const { return (uint32_t)t.field_i32(0, 0); }
    fb_vec_i32 inputs() const { return fb_vec_i32(t.field_ptr(1), t.ctx); }
    fb_vec_i32 outputs() const { return fb_vec_i32(t.field_ptr(2), t.ctx); }
    uint8_t builtin_options_type() const {
        uint16_t off = t.field_off(3);
        if (!off) return 0;
        const uint8_t* p = t.buf + off;
        return t.ok(p, 1) ? fb_u8(p) : 0;
    }
    fb_table builtin_options() const { return t.field_table(4); }
};

// --- Option table wrappers ---------------------------------------------------

struct Conv2DOptions {
    fb_table t;
    explicit Conv2DOptions(fb_table tab) : t(tab) {}
    int8_t padding() const { return t.field_i8(0, 0); }
    int32_t stride_w() const { return t.field_i32(1, 1); }
    int32_t stride_h() const { return t.field_i32(2, 1); }
    int8_t fused_activation() const { return t.field_i8(3, 0); }
    int32_t dilation_w() const { return t.field_i32(4, 1); }
    int32_t dilation_h() const { return t.field_i32(5, 1); }
};

struct DepthwiseConv2DOptions {
    fb_table t;
    explicit DepthwiseConv2DOptions(fb_table tab) : t(tab) {}
    int8_t padding() const { return t.field_i8(0, 0); }
    int32_t stride_w() const { return t.field_i32(1, 1); }
    int32_t stride_h() const { return t.field_i32(2, 1); }
    int32_t depth_multiplier() const { return t.field_i32(3, 1); }
    int8_t fused_activation() const { return t.field_i8(4, 0); }
    int32_t dilation_w() const { return t.field_i32(5, 1); }
    int32_t dilation_h() const { return t.field_i32(6, 1); }
};

struct Pool2DOptions {
    fb_table t;
    explicit Pool2DOptions(fb_table tab) : t(tab) {}
    int8_t padding() const { return t.field_i8(0, 0); }
    int32_t stride_w() const { return t.field_i32(1, 1); }
    int32_t stride_h() const { return t.field_i32(2, 1); }
    int32_t filter_width() const { return t.field_i32(3, 0); }
    int32_t filter_height() const { return t.field_i32(4, 0); }
    int8_t fused_activation() const { return t.field_i8(5, 0); }
};

struct FullyConnectedOptions {
    fb_table t;
    explicit FullyConnectedOptions(fb_table tab) : t(tab) {}
    int8_t fused_activation() const { return t.field_i8(0, 0); }
};

struct ConcatenationOptions {
    fb_table t;
    explicit ConcatenationOptions(fb_table tab) : t(tab) {}
    int32_t axis() const { return t.field_i32(0, 0); }
    int8_t fused_activation() const { return t.field_i8(1, 0); }
};

struct SoftmaxOptions {
    fb_table t;
    explicit SoftmaxOptions(fb_table tab) : t(tab) {}
    float beta() const { return t.field_f32(0, 0.0f); }
};

struct ReshapeOptions {
    fb_table t;
    explicit ReshapeOptions(fb_table tab) : t(tab) {}
    fb_vec_i32 new_shape() const { return fb_vec_i32(t.field_ptr(0), t.ctx); }
};

struct ArithOptions {
    fb_table t;
    explicit ArithOptions(fb_table tab) : t(tab) {}
    int8_t fused_activation() const { return t.field_i8(0, 0); }
};

struct ReducerOptions {
    fb_table t;
    explicit ReducerOptions(fb_table tab) : t(tab) {}
    bool keep_dims() const { return t.field_i8(0, 0) != 0; }
};

struct ResizeBilinearOptions {
    fb_table t;
    explicit ResizeBilinearOptions(fb_table tab) : t(tab) {}
    bool align_corners() const { return t.field_i8(0, 0) != 0; }
    bool half_pixel_centers() const { return t.field_i8(1, 0) != 0; }
};

struct ResizeNNOptions {
    fb_table t;
    explicit ResizeNNOptions(fb_table tab) : t(tab) {}
    bool align_corners() const { return t.field_i8(0, 0) != 0; }
    bool half_pixel_centers() const { return t.field_i8(1, 0) != 0; }
};

struct TransposeConvOptions {
    fb_table t;
    explicit TransposeConvOptions(fb_table tab) : t(tab) {}
    int8_t padding() const { return t.field_i8(0, 0); }
    int32_t stride_w() const { return t.field_i32(1, 1); }
    int32_t stride_h() const { return t.field_i32(2, 1); }
};

struct StridedSliceOptions {
    fb_table t;
    explicit StridedSliceOptions(fb_table tab) : t(tab) {}
    int32_t begin_mask() const { return t.field_i32(0, 0); }
    int32_t end_mask() const { return t.field_i32(1, 0); }
    int32_t ellipsis_mask() const { return t.field_i32(2, 0); }
    int32_t new_axis_mask() const { return t.field_i32(3, 0); }
    int32_t shrink_axis_mask() const { return t.field_i32(4, 0); }
};

struct PackOptions {
    fb_table t;
    explicit PackOptions(fb_table tab) : t(tab) {}
    int32_t values_count() const { return t.field_i32(0, 0); }
    int32_t axis() const { return t.field_i32(1, 0); }
};

struct GatherOptions {
    fb_table t;
    explicit GatherOptions(fb_table tab) : t(tab) {}
    int32_t axis() const { return t.field_i32(0, 0); }
    int32_t batch_dims() const { return t.field_i32(1, 0); }
};

struct LeakyReluOptions {
    fb_table t;
    explicit LeakyReluOptions(fb_table tab) : t(tab) {}
    float alpha() const { return t.field_f32(0, 0.0f); }
};

struct BatchMatMulOptions {
    fb_table t;
    explicit BatchMatMulOptions(fb_table tab) : t(tab) {}
    bool adj_x() const { return t.field_i8(0, 0) != 0; }
    bool adj_y() const { return t.field_i8(1, 0) != 0; }
};

} // namespace tflite
} // namespace nnr
