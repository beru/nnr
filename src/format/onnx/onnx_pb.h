// Auto-generated from onnx.proto3 by gen_onnx_pb.py -- do not edit

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>
#include <memory>
#include "pb.h"

namespace onnx_pb {

struct GraphProto;
struct TypeProto;

struct StringStringEntryProto {
    std::string_view key;
    std::string_view value;
};

struct IntIntListEntryProto {
    int64_t key = 0;
    std::vector<int64_t> value;
};

struct OperatorSetIdProto {
    std::string_view domain;
    int64_t version = 0;
};

struct TensorShapeProto {
    struct Dimension {
        int64_t dim_value = 0;
        std::string_view dim_param;
        std::string_view denotation;
    };

    std::vector<TensorShapeProto::Dimension> dim;
};

struct TensorProto {
    enum DataType {
        UNDEFINED = 0,
        FLOAT = 1,
        UINT8 = 2,
        INT8 = 3,
        UINT16 = 4,
        INT16 = 5,
        INT32 = 6,
        INT64 = 7,
        STRING = 8,
        BOOL = 9,
        FLOAT16 = 10,
        DOUBLE = 11,
        UINT32 = 12,
        UINT64 = 13,
        COMPLEX64 = 14,
        COMPLEX128 = 15,
        BFLOAT16 = 16,
        FLOAT8E4M3FN = 17,
        FLOAT8E4M3FNUZ = 18,
        FLOAT8E5M2 = 19,
        FLOAT8E5M2FNUZ = 20,
        UINT4 = 21,
        INT4 = 22,
        FLOAT4E2M1 = 23,
        FLOAT8E8M0 = 24,
        UINT2 = 25,
        INT2 = 26,
    };

    enum DataLocation {
        DEFAULT = 0,
        EXTERNAL = 1,
    };

    struct Segment {
        int64_t begin = 0;
        int64_t end = 0;
    };

    std::vector<int64_t> dims;
    int32_t data_type = 0;
    TensorProto::Segment segment;
    std::vector<float> float_data;
    std::vector<int32_t> int32_data;
    std::vector<std::string_view> string_data;
    std::vector<int64_t> int64_data;
    std::string_view name;
    std::string_view doc_string;
    std::string_view raw_data;
    std::vector<StringStringEntryProto> external_data;
    int32_t data_location = 0;
    std::vector<double> double_data;
    std::vector<uint64_t> uint64_data;
    std::vector<StringStringEntryProto> metadata_props;
};

struct SparseTensorProto {
    TensorProto values;
    TensorProto indices;
    std::vector<int64_t> dims;
};

struct TypeProto {
    struct Tensor {
        int32_t elem_type = 0;
        TensorShapeProto shape;
    };

    struct Sequence {
        std::unique_ptr<TypeProto> elem_type;
    };

    struct Map {
        int32_t key_type = 0;
        std::unique_ptr<TypeProto> value_type;
    };

    struct Optional {
        std::unique_ptr<TypeProto> elem_type;
    };

    struct SparseTensor {
        int32_t elem_type = 0;
        TensorShapeProto shape;
    };

    TypeProto::Tensor tensor_type;
    std::unique_ptr<TypeProto::Sequence> sequence_type;
    std::unique_ptr<TypeProto::Map> map_type;
    std::unique_ptr<TypeProto::Optional> optional_type;
    TypeProto::SparseTensor sparse_tensor_type;
    std::string_view denotation;
};

struct ValueInfoProto {
    std::string_view name;
    TypeProto type;
    std::string_view doc_string;
    std::vector<StringStringEntryProto> metadata_props;
};

struct TensorAnnotation {
    std::string_view tensor_name;
    std::vector<StringStringEntryProto> quant_parameter_tensor_names;
};

struct SimpleShardedDimProto {
    int64_t dim_value = 0;
    std::string_view dim_param;
    int64_t num_shards = 0;
};

struct ShardedDimProto {
    int64_t axis = 0;
    std::vector<SimpleShardedDimProto> simple_sharding;
};

struct ShardingSpecProto {
    std::string_view tensor_name;
    std::vector<int64_t> device;
    std::vector<IntIntListEntryProto> index_to_device_group_map;
    std::vector<ShardedDimProto> sharded_dim;
};

struct NodeDeviceConfigurationProto {
    std::string_view configuration_id;
    std::vector<ShardingSpecProto> sharding_spec;
    int32_t pipeline_stage = 0;
};

struct DeviceConfigurationProto {
    std::string_view name;
    int32_t num_devices = 0;
    std::vector<std::string_view> device;
};

struct AttributeProto {
    enum AttributeType {
        UNDEFINED = 0,
        FLOAT = 1,
        INT = 2,
        STRING = 3,
        TENSOR = 4,
        GRAPH = 5,
        SPARSE_TENSOR = 11,
        TYPE_PROTO = 13,
        FLOATS = 6,
        INTS = 7,
        STRINGS = 8,
        TENSORS = 9,
        GRAPHS = 10,
        SPARSE_TENSORS = 12,
        TYPE_PROTOS = 14,
    };

    std::string_view name;
    std::string_view ref_attr_name;
    std::string_view doc_string;
    int32_t type = 0;
    float f = 0.0f;
    int64_t i = 0;
    std::string_view s;
    TensorProto t;
    std::unique_ptr<GraphProto> g;
    SparseTensorProto sparse_tensor;
    TypeProto tp;
    std::vector<float> floats;
    std::vector<int64_t> ints;
    std::vector<std::string_view> strings;
    std::vector<TensorProto> tensors;
    std::vector<GraphProto> graphs;
    std::vector<SparseTensorProto> sparse_tensors;
    std::vector<TypeProto> type_protos;
};

struct NodeProto {
    std::vector<std::string_view> input;
    std::vector<std::string_view> output;
    std::string_view name;
    std::string_view op_type;
    std::string_view domain;
    std::string_view overload;
    std::vector<AttributeProto> attribute;
    std::string_view doc_string;
    std::vector<StringStringEntryProto> metadata_props;
    std::vector<NodeDeviceConfigurationProto> device_configurations;
};

struct GraphProto {
    std::vector<NodeProto> node;
    std::string_view name;
    std::vector<TensorProto> initializer;
    std::vector<SparseTensorProto> sparse_initializer;
    std::string_view doc_string;
    std::vector<ValueInfoProto> input;
    std::vector<ValueInfoProto> output;
    std::vector<ValueInfoProto> value_info;
    std::vector<TensorAnnotation> quantization_annotation;
    std::vector<StringStringEntryProto> metadata_props;
};

struct FunctionProto {
    std::string_view name;
    std::vector<std::string_view> input;
    std::vector<std::string_view> output;
    std::vector<std::string_view> attribute;
    std::vector<AttributeProto> attribute_proto;
    std::vector<NodeProto> node;
    std::string_view doc_string;
    std::vector<OperatorSetIdProto> opset_import;
    std::string_view domain;
    std::string_view overload;
    std::vector<ValueInfoProto> value_info;
    std::vector<StringStringEntryProto> metadata_props;
};

struct TrainingInfoProto {
    GraphProto initialization;
    GraphProto algorithm;
    std::vector<StringStringEntryProto> initialization_binding;
    std::vector<StringStringEntryProto> update_binding;
};

struct ModelProto {
    int64_t ir_version = 0;
    std::vector<OperatorSetIdProto> opset_import;
    std::string_view producer_name;
    std::string_view producer_version;
    std::string_view domain;
    int64_t model_version = 0;
    std::string_view doc_string;
    GraphProto graph;
    std::vector<StringStringEntryProto> metadata_props;
    std::vector<TrainingInfoProto> training_info;
    std::vector<FunctionProto> functions;
    std::vector<DeviceConfigurationProto> configuration;
};

// ---------------------------------------------------------------------------
// onnx_ctx — ONNX-specific parse context. Inherits primitive readers from
// pb::ctx and adds readField overloads + message-level readers as methods,
// so no function needs an explicit ctx parameter.
// ---------------------------------------------------------------------------

struct onnx_ctx : pb::ctx {
    using pb::ctx::ctx; // inherit constructors

    static constexpr int max_depth = 64;
    int depth = 0;

    // Per-type field dispatchers (defined in onnx_pb.cpp)
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, StringStringEntryProto& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, IntIntListEntryProto& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, OperatorSetIdProto& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TensorShapeProto::Dimension& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TensorShapeProto& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TensorProto::Segment& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TensorProto& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, SparseTensorProto& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TypeProto::Tensor& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TypeProto::Sequence& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TypeProto::Map& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TypeProto::Optional& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TypeProto::SparseTensor& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TypeProto& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, ValueInfoProto& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TensorAnnotation& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, SimpleShardedDimProto& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, ShardedDimProto& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, ShardingSpecProto& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, NodeDeviceConfigurationProto& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, DeviceConfigurationProto& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, AttributeProto& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, NodeProto& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, GraphProto& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, FunctionProto& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TrainingInfoProto& t);
    void readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, ModelProto& t);

    // Message-level readers (call readField via overload resolution)

    template <typename T>
    void readMessage(const uint8_t* p, const uint8_t* msg_end, T& t) {
        while (p < msg_end && !error) {
            auto [wt, fn] = readTag(p);
            if (error) return;
            readField(p, wt, fn, t);
        }
    }

    template <typename T>
    void readEmbeddedMessage(const uint8_t*& p, T& dst) {
        if (++depth > max_depth) { error = true; return; }
        uint64_t len = readVarint(p);
        if (error) { --depth; return; }
        if (out_of_bounds(p, (size_t)len)) { error = true; --depth; return; }
        const uint8_t* msg_end = p + len;
        readMessage(p, msg_end, dst);
        p = msg_end;
        --depth;
    }

    template <typename T>
    void readRepeatedMessage(const uint8_t*& p, std::vector<T>& dst) {
        dst.emplace_back();
        readEmbeddedMessage(p, dst.back());
        if (error) dst.pop_back();
    }

    template <typename T>
    void readEmbeddedMessage(const uint8_t*& p, std::unique_ptr<T>& dst) {
        if (!dst) dst = std::make_unique<T>();
        readEmbeddedMessage(p, *dst);
    }
};

bool read(const uint8_t* p, size_t sz, ModelProto& model);
bool read(const uint8_t* p, size_t sz, TensorProto& tensor);

} // namespace onnx_pb
