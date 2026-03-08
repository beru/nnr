#include "ArgReduce.h"

namespace nnr {

// @nnr-meta-op mt=no
operator_t* resolver_default_op_ArgMin(int opset, pool_t& pool) { return pool_new<ArgReduce_operator<false>>(pool); }

} // namespace nnr
