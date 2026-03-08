#include "trace.h"

#ifdef NNR_ENABLE_TRACE

namespace nnr {

static trace::event_t g_trace_events[trace::MAX_EVENTS];
static std::atomic<int> g_trace_count{0};

trace::event_t* trace::events() { return g_trace_events; }
std::atomic<int>& trace::count() { return g_trace_count; }

} // namespace nnr

#endif // NNR_ENABLE_TRACE
