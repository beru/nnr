#pragma once

#include "nnr.h"
#include "registry.h"

namespace nnr {

// Anchor function: calling this forces solve_operator.cpp (and its cpu_registrar
// static initializer) to be linked from nnr.lib on all platforms including MSVC.
operator_t* solve_operator(std::string_view op_type, int opset, pool_t& pool,
                           backend_t preferred = backend_t::CPU);

}
