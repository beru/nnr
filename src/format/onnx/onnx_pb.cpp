// Auto-generated from onnx.proto3 by gen_onnx_pb.py -- do not edit

#include "onnx_pb.h"

namespace onnx_pb {

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, StringStringEntryProto& t) {
    switch (fn) {
    case 1: t.key = readString(p); break;
    case 2: t.value = readString(p); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, IntIntListEntryProto& t) {
    switch (fn) {
    case 1: t.key = (int64_t)readVarint(p); break;
    case 2:
        if (wt == pb::WT_LengthLimited) readPackedVarint(p, t.value);
        else t.value.push_back((int64_t)readVarint(p));
        break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, OperatorSetIdProto& t) {
    switch (fn) {
    case 1: t.domain = readString(p); break;
    case 2: t.version = (int64_t)readVarint(p); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TensorShapeProto::Dimension& t) {
    switch (fn) {
    case 1: t.dim_value = (int64_t)readVarint(p); break;
    case 2: t.dim_param = readString(p); break;
    case 3: t.denotation = readString(p); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TensorShapeProto& t) {
    switch (fn) {
    case 1: readRepeatedMessage(p, t.dim); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TensorProto::Segment& t) {
    switch (fn) {
    case 1: t.begin = (int64_t)readVarint(p); break;
    case 2: t.end = (int64_t)readVarint(p); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TensorProto& t) {
    switch (fn) {
    case 1:
        if (wt == pb::WT_LengthLimited) readPackedVarint(p, t.dims);
        else t.dims.push_back((int64_t)readVarint(p));
        break;
    case 2: t.data_type = (int32_t)readVarint(p); break;
    case 3: readEmbeddedMessage(p, t.segment); break;
    case 4:
        if (wt == pb::WT_LengthLimited) readPackedFixed32(p, t.float_data);
        else { float v; readFixed32(p, &v); t.float_data.push_back(v); }
        break;
    case 5:
        if (wt == pb::WT_LengthLimited) readPackedVarint(p, t.int32_data);
        else t.int32_data.push_back((int32_t)readVarint(p));
        break;
    case 6: t.string_data.push_back(readString(p)); break;
    case 7:
        if (wt == pb::WT_LengthLimited) readPackedVarint(p, t.int64_data);
        else t.int64_data.push_back((int64_t)readVarint(p));
        break;
    case 8: t.name = readString(p); break;
    case 12: t.doc_string = readString(p); break;
    case 9: t.raw_data = readString(p); break;
    case 13: readRepeatedMessage(p, t.external_data); break;
    case 14: t.data_location = (int32_t)readVarint(p); break;
    case 10:
        if (wt == pb::WT_LengthLimited) readPackedFixed64(p, t.double_data);
        else { double v; readFixed64(p, &v); t.double_data.push_back(v); }
        break;
    case 11:
        if (wt == pb::WT_LengthLimited) readPackedVarint(p, t.uint64_data);
        else t.uint64_data.push_back(readVarint(p));
        break;
    case 16: readRepeatedMessage(p, t.metadata_props); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, SparseTensorProto& t) {
    switch (fn) {
    case 1: readEmbeddedMessage(p, t.values); break;
    case 2: readEmbeddedMessage(p, t.indices); break;
    case 3:
        if (wt == pb::WT_LengthLimited) readPackedVarint(p, t.dims);
        else t.dims.push_back((int64_t)readVarint(p));
        break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TypeProto::Tensor& t) {
    switch (fn) {
    case 1: t.elem_type = (int32_t)readVarint(p); break;
    case 2: readEmbeddedMessage(p, t.shape); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TypeProto::Sequence& t) {
    switch (fn) {
    case 1: readEmbeddedMessage(p, t.elem_type); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TypeProto::Map& t) {
    switch (fn) {
    case 1: t.key_type = (int32_t)readVarint(p); break;
    case 2: readEmbeddedMessage(p, t.value_type); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TypeProto::Optional& t) {
    switch (fn) {
    case 1: readEmbeddedMessage(p, t.elem_type); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TypeProto::SparseTensor& t) {
    switch (fn) {
    case 1: t.elem_type = (int32_t)readVarint(p); break;
    case 2: readEmbeddedMessage(p, t.shape); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TypeProto& t) {
    switch (fn) {
    case 1: readEmbeddedMessage(p, t.tensor_type); break;
    case 4: readEmbeddedMessage(p, t.sequence_type); break;
    case 5: readEmbeddedMessage(p, t.map_type); break;
    case 9: readEmbeddedMessage(p, t.optional_type); break;
    case 8: readEmbeddedMessage(p, t.sparse_tensor_type); break;
    case 6: t.denotation = readString(p); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, ValueInfoProto& t) {
    switch (fn) {
    case 1: t.name = readString(p); break;
    case 2: readEmbeddedMessage(p, t.type); break;
    case 3: t.doc_string = readString(p); break;
    case 4: readRepeatedMessage(p, t.metadata_props); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TensorAnnotation& t) {
    switch (fn) {
    case 1: t.tensor_name = readString(p); break;
    case 2: readRepeatedMessage(p, t.quant_parameter_tensor_names); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, SimpleShardedDimProto& t) {
    switch (fn) {
    case 1: t.dim_value = (int64_t)readVarint(p); break;
    case 2: t.dim_param = readString(p); break;
    case 3: t.num_shards = (int64_t)readVarint(p); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, ShardedDimProto& t) {
    switch (fn) {
    case 1: t.axis = (int64_t)readVarint(p); break;
    case 2: readRepeatedMessage(p, t.simple_sharding); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, ShardingSpecProto& t) {
    switch (fn) {
    case 1: t.tensor_name = readString(p); break;
    case 2:
        if (wt == pb::WT_LengthLimited) readPackedVarint(p, t.device);
        else t.device.push_back((int64_t)readVarint(p));
        break;
    case 3: readRepeatedMessage(p, t.index_to_device_group_map); break;
    case 4: readRepeatedMessage(p, t.sharded_dim); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, NodeDeviceConfigurationProto& t) {
    switch (fn) {
    case 1: t.configuration_id = readString(p); break;
    case 2: readRepeatedMessage(p, t.sharding_spec); break;
    case 3: t.pipeline_stage = (int32_t)readVarint(p); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, DeviceConfigurationProto& t) {
    switch (fn) {
    case 1: t.name = readString(p); break;
    case 2: t.num_devices = (int32_t)readVarint(p); break;
    case 3: t.device.push_back(readString(p)); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, AttributeProto& t) {
    switch (fn) {
    case 1: t.name = readString(p); break;
    case 21: t.ref_attr_name = readString(p); break;
    case 13: t.doc_string = readString(p); break;
    case 20: t.type = (int32_t)readVarint(p); break;
    case 2: readFixed32(p, &t.f); break;
    case 3: t.i = (int64_t)readVarint(p); break;
    case 4: t.s = readString(p); break;
    case 5: readEmbeddedMessage(p, t.t); break;
    case 6: readEmbeddedMessage(p, t.g); break;
    case 22: readEmbeddedMessage(p, t.sparse_tensor); break;
    case 14: readEmbeddedMessage(p, t.tp); break;
    case 7:
        if (wt == pb::WT_LengthLimited) readPackedFixed32(p, t.floats);
        else { float v; readFixed32(p, &v); t.floats.push_back(v); }
        break;
    case 8:
        if (wt == pb::WT_LengthLimited) readPackedVarint(p, t.ints);
        else t.ints.push_back((int64_t)readVarint(p));
        break;
    case 9: t.strings.push_back(readString(p)); break;
    case 10: readRepeatedMessage(p, t.tensors); break;
    case 11: readRepeatedMessage(p, t.graphs); break;
    case 23: readRepeatedMessage(p, t.sparse_tensors); break;
    case 15: readRepeatedMessage(p, t.type_protos); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, NodeProto& t) {
    switch (fn) {
    case 1: t.input.push_back(readString(p)); break;
    case 2: t.output.push_back(readString(p)); break;
    case 3: t.name = readString(p); break;
    case 4: t.op_type = readString(p); break;
    case 7: t.domain = readString(p); break;
    case 8: t.overload = readString(p); break;
    case 5: readRepeatedMessage(p, t.attribute); break;
    case 6: t.doc_string = readString(p); break;
    case 9: readRepeatedMessage(p, t.metadata_props); break;
    case 10: readRepeatedMessage(p, t.device_configurations); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, GraphProto& t) {
    switch (fn) {
    case 1: readRepeatedMessage(p, t.node); break;
    case 2: t.name = readString(p); break;
    case 5: readRepeatedMessage(p, t.initializer); break;
    case 15: readRepeatedMessage(p, t.sparse_initializer); break;
    case 10: t.doc_string = readString(p); break;
    case 11: readRepeatedMessage(p, t.input); break;
    case 12: readRepeatedMessage(p, t.output); break;
    case 13: readRepeatedMessage(p, t.value_info); break;
    case 14: readRepeatedMessage(p, t.quantization_annotation); break;
    case 16: readRepeatedMessage(p, t.metadata_props); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, FunctionProto& t) {
    switch (fn) {
    case 1: t.name = readString(p); break;
    case 4: t.input.push_back(readString(p)); break;
    case 5: t.output.push_back(readString(p)); break;
    case 6: t.attribute.push_back(readString(p)); break;
    case 11: readRepeatedMessage(p, t.attribute_proto); break;
    case 7: readRepeatedMessage(p, t.node); break;
    case 8: t.doc_string = readString(p); break;
    case 9: readRepeatedMessage(p, t.opset_import); break;
    case 10: t.domain = readString(p); break;
    case 13: t.overload = readString(p); break;
    case 12: readRepeatedMessage(p, t.value_info); break;
    case 14: readRepeatedMessage(p, t.metadata_props); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, TrainingInfoProto& t) {
    switch (fn) {
    case 1: readEmbeddedMessage(p, t.initialization); break;
    case 2: readEmbeddedMessage(p, t.algorithm); break;
    case 3: readRepeatedMessage(p, t.initialization_binding); break;
    case 4: readRepeatedMessage(p, t.update_binding); break;
    default: skip(p, wt); break;
    }
}

void onnx_ctx::readField(const uint8_t*& p, pb::WireType wt, uint64_t fn, ModelProto& t) {
    switch (fn) {
    case 1: t.ir_version = (int64_t)readVarint(p); break;
    case 8: readRepeatedMessage(p, t.opset_import); break;
    case 2: t.producer_name = readString(p); break;
    case 3: t.producer_version = readString(p); break;
    case 4: t.domain = readString(p); break;
    case 5: t.model_version = (int64_t)readVarint(p); break;
    case 6: t.doc_string = readString(p); break;
    case 7: readEmbeddedMessage(p, t.graph); break;
    case 14: readRepeatedMessage(p, t.metadata_props); break;
    case 20: readRepeatedMessage(p, t.training_info); break;
    case 25: readRepeatedMessage(p, t.functions); break;
    case 26: readRepeatedMessage(p, t.configuration); break;
    default: skip(p, wt); break;
    }
}

bool read(const uint8_t* p, size_t sz, ModelProto& model) {
    const uint8_t* end = p + sz;
    onnx_ctx c(end);
    c.readMessage(p, end, model);
    return !c.error;
}

bool read(const uint8_t* p, size_t sz, TensorProto& tensor) {
    const uint8_t* end = p + sz;
    onnx_ctx c(end);
    c.readMessage(p, end, tensor);
    return !c.error;
}

} // namespace onnx_pb
