#include "ArgReduce.h"

namespace nnr {

// @nnr-meta-op mt=no
operator_t* resolver_default_op_ArgMax(int opset, pool_t& pool) { return pool_new<ArgReduce_operator<true>>(pool); }

} // namespace nnr
