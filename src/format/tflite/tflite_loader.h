#pragma once

#include "nnr.h"

namespace nnr {

// Load a TFLite model from raw bytes into ctx.
// The buffer must start with the FlatBuffer header (offset 4 = "TFL3").
// Returns false on parse failure.
bool load_tflite(context_t* ctx, const void* data, size_t size);

} // namespace nnr
