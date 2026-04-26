#include "reduce.h"
#include "pool.h"

// Concrete reductions. Each subclass supplies the WGSL expressions that
// parameterize the shared kernel.
//
//   init:       starting value of the accumulator.
//   transform:  applied to each element `v` before accumulation.
//               For "sum-of-x" reductions this is just `v`; for
//               sum-of-squares / L1 / L2 / log-sum etc. it's the pointwise
//               transformation.
//   merge:      associative combiner of two accumulators `a` and `b`.
//               Used both in the per-thread stream step and in the
//               workgroup tree reduction.
//   finalize:   post-processing (e.g. mean divides by `n`, L2 takes sqrt).
//
// Large sentinels (-3.4e38 / 3.4e38) are used instead of f32 MIN/MAX
// literals since Tint rejects `-3.4028235e38` as out of range.

namespace nnr {

#define DEFINE_REDUCE(name, init, transform, merge, finalize)                   \
    namespace {                                                                 \
    struct name##_op_webgpu : webgpu::reduce_elementwise_t {                    \
        const char* init_expr()      const override { return init; }            \
        const char* transform_expr() const override { return transform; }       \
        const char* merge_expr()     const override { return merge; }           \
        const char* finalize_expr()  const override { return finalize; }        \
    };                                                                          \
    }                                                                           \
    operator_t* resolver_default_op_##name##_webgpu(int, pool_t& pool) {        \
        return pool_new<name##_op_webgpu>(pool);                                \
    }

DEFINE_REDUCE(ReduceSum,        "0.0",     "v",        "a + b",      "acc")
DEFINE_REDUCE(ReduceMean,       "0.0",     "v",        "a + b",      "acc / f32(n)")
DEFINE_REDUCE(ReduceMax,        "-3.4e38", "v",        "max(a, b)",  "acc")
DEFINE_REDUCE(ReduceMin,        "3.4e38",  "v",        "min(a, b)",  "acc")
DEFINE_REDUCE(ReduceProd,       "1.0",     "v",        "a * b",      "acc")
DEFINE_REDUCE(ReduceSumSquare,  "0.0",     "v * v",    "a + b",      "acc")
DEFINE_REDUCE(ReduceL1,         "0.0",     "abs(v)",   "a + b",      "acc")
DEFINE_REDUCE(ReduceL2,         "0.0",     "v * v",    "a + b",      "sqrt(acc)")
DEFINE_REDUCE(ReduceLogSum,     "0.0",     "v",        "a + b",      "log(acc)")

#undef DEFINE_REDUCE

} // namespace nnr
