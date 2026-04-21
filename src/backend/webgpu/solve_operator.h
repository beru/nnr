#pragma once

#include "nnr.h"
#include "registry.h"

namespace nnr::webgpu {

// Anchor function: calling this forces backend/webgpu/solve_operator.cpp
// (and its webgpu_registrar static initializer) to be linked.
// The ONNX loader calls this when preferred backend == WEBGPU.
void ensure_registered();

} // namespace nnr::webgpu
