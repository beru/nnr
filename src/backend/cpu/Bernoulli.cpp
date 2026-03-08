#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {
}

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Bernoulli(int opset, pool_t& pool)
{
    //if (n->opset >= 15) {
    //}
    return nullptr;
}

} // namespace nnr
