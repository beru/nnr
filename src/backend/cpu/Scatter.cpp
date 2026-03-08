#include "nnr.h"
#include "util.h"

namespace nnr {

// Scatter is deprecated since opset 11, replaced by ScatterElements
// They have the same semantics (axis attribute, no reduction)
operator_t* resolver_default_op_ScatterElements(int opset, pool_t& pool);

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Scatter(int opset, pool_t& pool)
{
    return resolver_default_op_ScatterElements(opset, pool);
}

} // namespace nnr
