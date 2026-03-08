#include "attr_key.h"
#include <algorithm>

namespace nnr {

// kAttrTable entries are generated from attr_keys.def — must match its alphabetical order.
// #name stringifies each identifier, giving the ONNX attribute name string.
static constexpr std::pair<std::string_view, attr_key_t> kAttrTable[] = {
#define X(name) {#name, attr_key_t::name},
#include "attr_keys.def"
#undef X
};

attr_key_t attr_key_from_string(std::string_view name)
{
    auto it = std::lower_bound(
        std::begin(kAttrTable), std::end(kAttrTable), name,
        [](const auto& e, std::string_view n) { return e.first < n; });
    if (it != std::end(kAttrTable) && it->first == name)
        return it->second;
    return attr_key_t::unknown;
}

} // namespace nnr
