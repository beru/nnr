#include <array>
#include <charconv>

#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

void Cast_from_string(
    const std::string* from_data,
    data_type_t to_type, void* to_data,
    size_t ndata)
{
    size_t i;

    switch (to_type) {
    case NNR_DATA_TYPE_BOOL:
        {
            bool_t* py = (bool_t*)to_data;
            for (i = 0; i < ndata; ++i) {
                py[i] = strtoul(from_data[i].c_str(), 0, 0);
            }
        }
        break;
    case NNR_DATA_TYPE_INT8:
        {
            int8_t* py = (int8_t*)to_data;
            for (i = 0; i < ndata; ++i) {
                py[i] = (int8_t)strtol(from_data[i].c_str(), 0, 0);
            }
        }
        break;
    case NNR_DATA_TYPE_INT16:
        {
            int16_t* py = (int16_t*)to_data;
            for (i = 0; i < ndata; ++i) {
                py[i] = (int16_t)strtol(from_data[i].c_str(), 0, 0);
            }
        }
        break;
    case NNR_DATA_TYPE_INT32:
        {
            int32_t* py = (int32_t*)to_data;
            for (i = 0; i < ndata; ++i) {
                py[i] = (int32_t)strtol(from_data[i].c_str(), 0, 0);
            }
        }
        break;
    case NNR_DATA_TYPE_INT64:
        {
            int64_t* py = (int64_t*)to_data;
            for (i = 0; i < ndata; ++i) {
                py[i] = (int64_t)strtoll(from_data[i].c_str(), 0, 0);
            }
        }
        break;
    case NNR_DATA_TYPE_UINT8:
        {
            uint8_t* py = (uint8_t*)to_data;
            for (i = 0; i < ndata; ++i) {
                py[i] = (uint8_t)strtoul(from_data[i].c_str(), 0, 0);
            }
        }
        break;
    case NNR_DATA_TYPE_UINT16:
        {
            uint16_t* py = (uint16_t*)to_data;
            for (i = 0; i < ndata; ++i) {
                py[i] = (uint16_t)strtoul(from_data[i].c_str(), 0, 0);
            }
        }
        break;
    case NNR_DATA_TYPE_UINT32:
        {
            uint32_t* py = (uint32_t*)to_data;
            for (i = 0; i < ndata; ++i) {
                py[i] = (uint32_t)strtoul(from_data[i].c_str(), 0, 0);
            }
        }
        break;
    case NNR_DATA_TYPE_UINT64:
        {
            uint64_t* py = (uint64_t*)to_data;
            for (i = 0; i < ndata; ++i) {
                py[i] = (uint64_t)strtoull(from_data[i].c_str(), 0, 0);
            }
        }
        break;
    case NNR_DATA_TYPE_BFLOAT16:
        {
            bfloat16_t* py = (bfloat16_t*)to_data;
            for (i = 0; i < ndata; ++i) {
                py[i] = (float)strtod(from_data[i].c_str(), nullptr);
            }
        }
        break;
    case NNR_DATA_TYPE_FLOAT16:
        {
            float16_t* py = (float16_t*)to_data;
            for (i = 0; i < ndata; ++i) {
                py[i] = (float)strtod(from_data[i].c_str(), nullptr);
            }
        }
        break;
    case NNR_DATA_TYPE_FLOAT8E4M3FN:
        {
            uint8_t* py = (uint8_t*)to_data;
            for (i = 0; i < ndata; ++i)
                py[i] = float32_to_float8e4m3fn((float)strtod(from_data[i].c_str(), nullptr));
        }
        break;
    case NNR_DATA_TYPE_FLOAT8E4M3FNUZ:
        {
            uint8_t* py = (uint8_t*)to_data;
            for (i = 0; i < ndata; ++i)
                py[i] = float32_to_float8e4m3fnuz((float)strtod(from_data[i].c_str(), nullptr));
        }
        break;
    case NNR_DATA_TYPE_FLOAT8E5M2:
        {
            uint8_t* py = (uint8_t*)to_data;
            for (i = 0; i < ndata; ++i)
                py[i] = float32_to_float8e5m2((float)strtod(from_data[i].c_str(), nullptr));
        }
        break;
    case NNR_DATA_TYPE_FLOAT8E5M2FNUZ:
        {
            uint8_t* py = (uint8_t*)to_data;
            for (i = 0; i < ndata; ++i)
                py[i] = float32_to_float8e5m2fnuz((float)strtod(from_data[i].c_str(), nullptr));
        }
        break;
    case NNR_DATA_TYPE_FLOAT32:
        {
            float* py = (float*)to_data;
            for (i = 0; i < ndata; ++i) {
                py[i] = (float)strtod(from_data[i].c_str(), nullptr);
            }
        }
        break;
    case NNR_DATA_TYPE_FLOAT64:
        {
            double* py = (double*)to_data;
            for (i = 0; i < ndata; ++i) {
                py[i] = (double)strtod(from_data[i].c_str(), nullptr);
            }
        }
        break;
    default:
        break;
    }
}

template <typename TreatT, typename DataT>
void Cast_to_string(const DataT* from_data, std::string* to_data, size_t ndata)
{
    if constexpr (std::is_floating_point_v<TreatT>) {
        std::array<char, 32> str;
        for (size_t i=0; i<ndata; ++i) {
            auto result = std::to_chars(str.data(), str.data() + str.size(), (TreatT)from_data[i]);
            to_data[i] = std::string(str.data(), result.ptr);
        }
    } else {
        for (size_t i=0; i<ndata; ++i) {
            to_data[i] = std::to_string((TreatT)from_data[i]);
        }
    }
}

void Cast_to_string(
    data_type_t from_type, const void* from_data,
    std::string* to_data,
    size_t ndata)
{
    switch (from_type) {
#define X(enum_type, data_type, treat_type) case enum_type: Cast_to_string<treat_type>((const data_type*)from_data, to_data, ndata); return;
    X(NNR_DATA_TYPE_BOOL, bool_t, bool)
    X(NNR_DATA_TYPE_INT8, int8_t, int32_t)
    X(NNR_DATA_TYPE_INT16, int16_t, int32_t)
    X(NNR_DATA_TYPE_INT32, int32_t, int32_t)
    X(NNR_DATA_TYPE_INT64, int64_t, int64_t)
    X(NNR_DATA_TYPE_UINT8, uint8_t, uint32_t)
    X(NNR_DATA_TYPE_UINT16, uint16_t, uint32_t)
    X(NNR_DATA_TYPE_UINT32, uint32_t, uint32_t)
    X(NNR_DATA_TYPE_UINT64, uint64_t, uint64_t)
    X(NNR_DATA_TYPE_BFLOAT16, bfloat16_t, float)
    X(NNR_DATA_TYPE_FLOAT16, float16_t, float)
    X(NNR_DATA_TYPE_FLOAT32, float, float)
    X(NNR_DATA_TYPE_FLOAT64, double, double)
#undef X
    case NNR_DATA_TYPE_FLOAT8E4M3FN:
    case NNR_DATA_TYPE_FLOAT8E4M3FNUZ:
    case NNR_DATA_TYPE_FLOAT8E5M2:
    case NNR_DATA_TYPE_FLOAT8E5M2FNUZ: {
        const uint8_t* px = (const uint8_t*)from_data;
        std::array<char, 32> str;
        for (size_t i = 0; i < ndata; ++i) {
            float f = (from_type == NNR_DATA_TYPE_FLOAT8E4M3FN)   ? float8e4m3fn_to_float32(px[i]) :
                      (from_type == NNR_DATA_TYPE_FLOAT8E4M3FNUZ)  ? float8e4m3fnuz_to_float32(px[i]) :
                      (from_type == NNR_DATA_TYPE_FLOAT8E5M2)      ? float8e5m2_to_float32(px[i]) :
                                                                          float8e5m2fnuz_to_float32(px[i]);
            auto result = std::to_chars(str.data(), str.data() + str.size(), f);
            to_data[i] = std::string(str.data(), result.ptr);
        }
        return;
    }
    }
}

void Copy_array(
    data_type_t enum_type,
    const void* from_data, void* to_data,
    size_t ndata)
{
    switch (enum_type) {
#define X(enum_type, data_type) case enum_type: std::copy_n((const data_type*)from_data, ndata, (data_type*)to_data); return;
    X(NNR_DATA_TYPE_BOOL, bool_t)
    X(NNR_DATA_TYPE_INT8, int8_t)
    X(NNR_DATA_TYPE_INT16, int16_t)
    X(NNR_DATA_TYPE_INT32, int32_t)
    X(NNR_DATA_TYPE_INT64, int64_t)
    X(NNR_DATA_TYPE_UINT8, uint8_t)
    X(NNR_DATA_TYPE_UINT16, uint16_t)
    X(NNR_DATA_TYPE_UINT32, uint32_t)
    X(NNR_DATA_TYPE_UINT64, uint64_t)
    X(NNR_DATA_TYPE_BFLOAT16, bfloat16_t)
    X(NNR_DATA_TYPE_FLOAT16, float16_t)
    X(NNR_DATA_TYPE_FLOAT32, float)
    X(NNR_DATA_TYPE_FLOAT64, double)
    X(NNR_DATA_TYPE_STRING, std::string)
    // float4/float8 types stored as uint8_t (1 byte each)
    case NNR_DATA_TYPE_FLOAT8E4M3FN:
    case NNR_DATA_TYPE_FLOAT8E4M3FNUZ:
    case NNR_DATA_TYPE_FLOAT8E5M2:
    case NNR_DATA_TYPE_FLOAT8E5M2FNUZ:
    case NNR_DATA_TYPE_FLOAT4E2M1:
    case NNR_DATA_TYPE_FLOAT8E8M0:
        std::copy_n((const uint8_t*)from_data, ndata, (uint8_t*)to_data);
        return;
#undef X
    }
}

template <typename FromT, typename ToT>
void Cast_array(
    const FromT* from_data,
    ToT* to_data,
    size_t ndata)
{
    for (size_t i=0; i<ndata; ++i) {
        to_data[i] = (ToT)from_data[i];
    }
}

// saturate flag for float8 encoding (set by Cast_operator before calling Cast_array)
static bool g_cast_saturate = true;

template <typename FromT>
void Cast_array(
    const FromT* from_data,
    data_type_t to_type, void* to_data,
    size_t ndata)
{
    switch (to_type) {
#define X(enum_type, data_type) case enum_type: Cast_array(from_data, (data_type*)to_data, ndata); return;
    X(NNR_DATA_TYPE_BOOL, bool_t)
    X(NNR_DATA_TYPE_INT8, int8_t)
    X(NNR_DATA_TYPE_INT16, int16_t)
    X(NNR_DATA_TYPE_INT32, int32_t)
    X(NNR_DATA_TYPE_INT64, int64_t)
    X(NNR_DATA_TYPE_UINT8, uint8_t)
    X(NNR_DATA_TYPE_UINT16, uint16_t)
    X(NNR_DATA_TYPE_UINT32, uint32_t)
    X(NNR_DATA_TYPE_UINT64, uint64_t)
    X(NNR_DATA_TYPE_BFLOAT16, bfloat16_t)
    X(NNR_DATA_TYPE_FLOAT16, float16_t)
    X(NNR_DATA_TYPE_FLOAT32, float)
    X(NNR_DATA_TYPE_FLOAT64, double)
#undef X
    case NNR_DATA_TYPE_FLOAT8E4M3FN: {
        uint8_t* p = (uint8_t*)to_data;
        for (size_t i = 0; i < ndata; ++i) p[i] = float32_to_float8e4m3fn((float)from_data[i], g_cast_saturate);
        return;
    }
    case NNR_DATA_TYPE_FLOAT8E4M3FNUZ: {
        uint8_t* p = (uint8_t*)to_data;
        for (size_t i = 0; i < ndata; ++i) p[i] = float32_to_float8e4m3fnuz((float)from_data[i], g_cast_saturate);
        return;
    }
    case NNR_DATA_TYPE_FLOAT8E5M2: {
        uint8_t* p = (uint8_t*)to_data;
        for (size_t i = 0; i < ndata; ++i) p[i] = float32_to_float8e5m2((float)from_data[i], g_cast_saturate);
        return;
    }
    case NNR_DATA_TYPE_FLOAT8E5M2FNUZ: {
        uint8_t* p = (uint8_t*)to_data;
        for (size_t i = 0; i < ndata; ++i) p[i] = float32_to_float8e5m2fnuz((float)from_data[i], g_cast_saturate);
        return;
    }
    case NNR_DATA_TYPE_FLOAT4E2M1: {
        uint8_t* p = (uint8_t*)to_data;
        for (size_t i = 0; i < ndata; ++i) p[i] = float32_to_float4e2m1((float)from_data[i]);
        return;
    }
    case NNR_DATA_TYPE_FLOAT8E8M0: {
        uint8_t* p = (uint8_t*)to_data;
        for (size_t i = 0; i < ndata; ++i) p[i] = float32_to_float8e8m0((float)from_data[i]);
        return;
    }
    case NNR_DATA_TYPE_INT4: {
        // Truncate to lower 4 bits, then sign-extend (same as two's complement wrap)
        int8_t* p = (int8_t*)to_data;
        for (size_t i = 0; i < ndata; ++i) {
            int8_t nibble = (int8_t)((int32_t)from_data[i] & 0xF);
            p[i] = (nibble & 0x8) ? (nibble | (int8_t)0xF0) : nibble;
        }
        return;
    }
    case NNR_DATA_TYPE_UINT4: {
        // Truncate to lower 4 bits
        uint8_t* p = (uint8_t*)to_data;
        for (size_t i = 0; i < ndata; ++i)
            p[i] = (uint8_t)((int32_t)from_data[i] & 0xF);
        return;
    }
    case NNR_DATA_TYPE_INT2: {
        // Truncate to lower 2 bits, sign-extend from 2-bit two's complement
        int8_t* p = (int8_t*)to_data;
        for (size_t i = 0; i < ndata; ++i) {
            int8_t bits = (int8_t)((int32_t)from_data[i] & 0x3);
            p[i] = (bits & 0x2) ? (bits | (int8_t)0xFC) : bits;
        }
        return;
    }
    case NNR_DATA_TYPE_UINT2: {
        // Truncate to lower 2 bits
        uint8_t* p = (uint8_t*)to_data;
        for (size_t i = 0; i < ndata; ++i)
            p[i] = (uint8_t)((int32_t)from_data[i] & 0x3);
        return;
    }
    default: break;
    }
}

} // namespace

void Cast_set_saturate(bool sat) { g_cast_saturate = sat; }

void Cast_array(
    data_type_t from_type, const void* from_data,
    data_type_t to_type, void* to_data,
    size_t ndata)
{
    if (from_type == to_type) {
        Copy_array(from_type, from_data, to_data, ndata);
    }else if (to_type == NNR_DATA_TYPE_STRING) {
        Cast_to_string(from_type, from_data, (std::string*)to_data, ndata);
    }else if (from_type == NNR_DATA_TYPE_STRING) {
        Cast_from_string((const std::string*)from_data, to_type, to_data, ndata);
    }else {
        switch (from_type) {
#define X(enum_type, data_type) case enum_type: Cast_array((const data_type*)from_data, to_type, to_data, ndata); return;
        X(NNR_DATA_TYPE_BOOL, bool_t)
        X(NNR_DATA_TYPE_INT8, int8_t)
        X(NNR_DATA_TYPE_INT16, int16_t)
        X(NNR_DATA_TYPE_INT32, int32_t)
        X(NNR_DATA_TYPE_INT64, int64_t)
        X(NNR_DATA_TYPE_UINT8, uint8_t)
        X(NNR_DATA_TYPE_UINT16, uint16_t)
        X(NNR_DATA_TYPE_UINT32, uint32_t)
        X(NNR_DATA_TYPE_UINT64, uint64_t)
        X(NNR_DATA_TYPE_BFLOAT16, bfloat16_t)
        X(NNR_DATA_TYPE_FLOAT16, float16_t)
        X(NNR_DATA_TYPE_FLOAT32, float)
        X(NNR_DATA_TYPE_FLOAT64, double)
        X(NNR_DATA_TYPE_INT4, int8_t)
        X(NNR_DATA_TYPE_UINT4, uint8_t)
        X(NNR_DATA_TYPE_INT2, int8_t)
        X(NNR_DATA_TYPE_UINT2, uint8_t)
#undef X
        case NNR_DATA_TYPE_FLOAT8E4M3FN:
        case NNR_DATA_TYPE_FLOAT8E4M3FNUZ:
        case NNR_DATA_TYPE_FLOAT8E5M2:
        case NNR_DATA_TYPE_FLOAT8E5M2FNUZ:
        case NNR_DATA_TYPE_FLOAT4E2M1:
        case NNR_DATA_TYPE_FLOAT8E8M0: {
            // Decode float4/float8 → float32 per element, cast directly to target (no buffer)
            const uint8_t* px = (const uint8_t*)from_data;
            auto decode = [&](size_t i) -> float {
                switch (from_type) {
                case NNR_DATA_TYPE_FLOAT8E4M3FN:   return float8e4m3fn_to_float32(px[i]);
                case NNR_DATA_TYPE_FLOAT8E4M3FNUZ: return float8e4m3fnuz_to_float32(px[i]);
                case NNR_DATA_TYPE_FLOAT8E5M2:     return float8e5m2_to_float32(px[i]);
                case NNR_DATA_TYPE_FLOAT8E5M2FNUZ: return float8e5m2fnuz_to_float32(px[i]);
                case NNR_DATA_TYPE_FLOAT4E2M1:     return float4e2m1_to_float32(px[i]);
                default:                              return float8e8m0_to_float32(px[i]);
                }
            };
            switch (to_type) {
#define X(et, dt) case et: { dt* py = (dt*)to_data; for (size_t i = 0; i < ndata; ++i) py[i] = (dt)decode(i); return; }
            X(NNR_DATA_TYPE_BOOL,     bool_t)
            X(NNR_DATA_TYPE_INT8,     int8_t)
            X(NNR_DATA_TYPE_INT16,    int16_t)
            X(NNR_DATA_TYPE_INT32,    int32_t)
            X(NNR_DATA_TYPE_INT64,    int64_t)
            X(NNR_DATA_TYPE_UINT8,    uint8_t)
            X(NNR_DATA_TYPE_UINT16,   uint16_t)
            X(NNR_DATA_TYPE_UINT32,   uint32_t)
            X(NNR_DATA_TYPE_UINT64,   uint64_t)
            X(NNR_DATA_TYPE_BFLOAT16, bfloat16_t)
            X(NNR_DATA_TYPE_FLOAT16,  float16_t)
            X(NNR_DATA_TYPE_FLOAT32,  float)
            X(NNR_DATA_TYPE_FLOAT64,  double)
#undef X
            case NNR_DATA_TYPE_FLOAT8E4M3FN:   { uint8_t* py = (uint8_t*)to_data; for (size_t i = 0; i < ndata; ++i) py[i] = float32_to_float8e4m3fn(decode(i),   g_cast_saturate); return; }
            case NNR_DATA_TYPE_FLOAT8E4M3FNUZ: { uint8_t* py = (uint8_t*)to_data; for (size_t i = 0; i < ndata; ++i) py[i] = float32_to_float8e4m3fnuz(decode(i), g_cast_saturate); return; }
            case NNR_DATA_TYPE_FLOAT8E5M2:     { uint8_t* py = (uint8_t*)to_data; for (size_t i = 0; i < ndata; ++i) py[i] = float32_to_float8e5m2(decode(i),     g_cast_saturate); return; }
            case NNR_DATA_TYPE_FLOAT8E5M2FNUZ: { uint8_t* py = (uint8_t*)to_data; for (size_t i = 0; i < ndata; ++i) py[i] = float32_to_float8e5m2fnuz(decode(i), g_cast_saturate); return; }
            case NNR_DATA_TYPE_FLOAT4E2M1:     { uint8_t* py = (uint8_t*)to_data; for (size_t i = 0; i < ndata; ++i) py[i] = float32_to_float4e2m1(decode(i));                      return; }
            case NNR_DATA_TYPE_FLOAT8E8M0:     { uint8_t* py = (uint8_t*)to_data; for (size_t i = 0; i < ndata; ++i) py[i] = float32_to_float8e8m0(decode(i));                      return; }
            default: break;
            }
            return;
        }
        }
    }
}

namespace {

struct Cast_operator : public operator_t {
    data_type_t to;
    bool saturate = true;

    bool init() override {
        if (!is_inout_size(1, 1)) {
            return false;
        }
        to = (data_type_t)attribute(attr_key_t::to, inputs[0]->type);
        saturate = attribute(attr_key_t::saturate, (int32_t)1) != 0;
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        return y->reshape_identity(x, to);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        Cast_array(x->type, x->data, y->type, y->data, y->ndata);
        return true;
    }

    bool exec() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        data_type_t type = x->type;

        // Re-derive y's shape from x's live dims — an upstream
        // NonMaxSuppression can shrink x->dims during exec(), and the
        // fast path skips reshape(). Without this, y->ndata stays at the
        // prepare-time upper bound and Cast_array reads past x->data
        // (yolov3-tiny NMS→Unsqueeze→Cast crash).
        if (x->ndata != y->ndata)
            if (!reshape()) return false;

        // Float4/Float8 source: decode first, then cast
        if (type == NNR_DATA_TYPE_FLOAT8E4M3FN || type == NNR_DATA_TYPE_FLOAT8E4M3FNUZ
         || type == NNR_DATA_TYPE_FLOAT8E5M2 || type == NNR_DATA_TYPE_FLOAT8E5M2FNUZ
         || type == NNR_DATA_TYPE_FLOAT4E2M1 || type == NNR_DATA_TYPE_FLOAT8E8M0) {
            g_cast_saturate = saturate;
            Cast_array(type, x->data, y->type, y->data, y->ndata);
            return true;
        }
        // Float4/Float8 target: cast source, then encode
        data_type_t totype = y->type;
        if (totype == NNR_DATA_TYPE_FLOAT8E4M3FN || totype == NNR_DATA_TYPE_FLOAT8E4M3FNUZ
         || totype == NNR_DATA_TYPE_FLOAT8E5M2 || totype == NNR_DATA_TYPE_FLOAT8E5M2FNUZ
         || totype == NNR_DATA_TYPE_FLOAT4E2M1 || totype == NNR_DATA_TYPE_FLOAT8E8M0) {
            g_cast_saturate = saturate;
            Cast_array(type, x->data, totype, y->data, y->ndata);
            return true;
        }

        if (type == NNR_DATA_TYPE_INT4 || type == NNR_DATA_TYPE_UINT4
         || type == NNR_DATA_TYPE_INT2 || type == NNR_DATA_TYPE_UINT2) {
            Cast_array(type, x->data, y->type, y->data, y->ndata);
            return true;
        }
        if (opset >= 13) {
            return typed_exec<Cast_operator,
                bool_t,
                uint8_t, uint16_t, uint32_t, uint64_t,
                int8_t, int16_t, int32_t, int64_t,
                float16_t, float, double, bfloat16_t,
                std::string
            >(this, type);
        }else if (opset >= 9) {
            return typed_exec<Cast_operator,
                bool_t,
                uint8_t, uint16_t, uint32_t, uint64_t,
                int8_t, int16_t, int32_t, int64_t,
                float16_t, float, double,
                std::string
            >(this, type);
        }else if (opset >= 6) {
            return typed_exec<Cast_operator,
                bool_t,
                uint8_t, uint16_t, uint32_t, uint64_t,
                int8_t, int16_t, int32_t, int64_t,
                float16_t, float, double
            >(this, type);
        }else if (opset >= 1) {
        }
        return false;
    }

};

} // namespace {

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Cast(int opset, pool_t& pool) { return pool_new<Cast_operator>(pool); }

} // namespace nnr
