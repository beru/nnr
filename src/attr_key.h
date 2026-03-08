#pragma once
#include <string_view>

namespace nnr {

enum class attr_key_t : uint16_t {
    unknown = 0,
#define X(name) name,
#include "attr_keys.def"
#undef X
    _COUNT  // sentinel
};

// Convert ONNX attribute name string -> attr_key_t (called once at load time).
// Returns attr_key_t::unknown for unrecognized names.
attr_key_t attr_key_from_string(std::string_view name);

} // namespace nnr
