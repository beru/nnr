#pragma once
// ONNX format loader for NNR.
// Parses ONNX protobuf and populates a context_t.

#include "nnr.h"
#include "format/onnx/onnx_pb.h"

namespace nnr {

// Main entry point: load an ONNX model from raw bytes into ctx.
// Returns false on parse or init failure.
bool load_onnx(context_t* ctx, const void* data, size_t size);

// Tensor allocation from ONNX tensor proto (for tensor_t::alloc_from_file).
tensor_t* onnx_tensor_alloc_from_file(std::string_view filename);
tensor_t* onnx_tensor_alloc_optional_from_file(std::string_view filename);
bool onnx_tensor_load_sequence_from_file(tensor_t* t, std::string_view filename);
bool onnx_tensor_sequence_equal_file(const tensor_t* t, std::string_view filename);

// Build a tensor_t from an onnx_pb::TensorProto (data is copied in).
tensor_t* onnx_tensor_alloc_from_proto(const onnx_pb::TensorProto& pb);

// Copy onnx_pb::TensorProto data into an existing tensor_t.
void onnx_tensor_copy_from_proto(tensor_t* t, const onnx_pb::TensorProto& o);

} // namespace nnr
