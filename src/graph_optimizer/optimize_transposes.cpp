// Transpose optimizer — M7+ #2.
//
// Runs at LAYOUT level after assign_layouts. Detects and eliminates redundant
// NCHW<->NHWC reorders left behind by layout assignment:
//   - cancellation: two adjacent reorders that round-trip the format
//   - push-across-elementwise: move reorders across LAYOUT_ALL ops so that
//     two reorders can meet and cancel further down the graph
//
// Modelled on ORT's `transpose_optimizer.cc`.
//
// STATUS: scaffold only. apply() is currently a no-op; logic lands in
// follow-up commits.

#include "graph_optimizer/graph_optimizer_internal.h"

namespace nnr {

void optimize_transposes(context_t* ctx)
{
    (void)ctx;
    // TODO(M7+ #2):
    //   1. Walk nodes, find reorder pairs X -> reorder(A->B) -> reorder(B->A) -> Y
    //      and fold them away.
    //   2. For LAYOUT_ALL ops sandwiched between opposite reorders, push the
    //      reorder past the op so two reorders become adjacent.
}

} // namespace nnr
