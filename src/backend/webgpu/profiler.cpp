#include "profiler.h"

#include "buffer.h"  // shared_encoder()
#include "device.h"

#include <cstdlib>
#include <vector>

namespace nnr::webgpu {

namespace {

struct op_profiler_state_t {
    bool                active = false;
    int                 capacity = 0;          // ops the QuerySet can hold
    int                 n_ops_this_run = 0;
    wgpu::QuerySet      qs;
    wgpu::Buffer        resolve_buf;           // QueryResolve | CopySrc
    wgpu::Buffer        read_buf;              // MapRead | CopyDst
    std::vector<double> last_times_us;
};

op_profiler_state_t g_prof;

// Env-var check cached at first call. Re-reads on each call are fine too
// but this matches the style of NNR_LOG_WEBGPU_FUSION.
bool env_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = std::getenv("NNR_WEBGPU_OP_TIMINGS");
        cached = (v && *v && *v != '0') ? 1 : 0;
    }
    return cached != 0;
}

// Grow QuerySet + staging buffers to hold at least `needed_ops` × 2 timestamps.
// Timestamps are 8 bytes (u64) each.
void ensure_capacity(int needed_ops) {
    if (g_prof.capacity >= needed_ops) return;
    int cap = g_prof.capacity > 0 ? g_prof.capacity : 16;
    while (cap < needed_ops) cap *= 2;

    auto& dev = get_device();

    wgpu::QuerySetDescriptor qsd = {};
    qsd.type  = wgpu::QueryType::Timestamp;
    qsd.count = (uint32_t)(cap * 2);
    g_prof.qs = dev.device.CreateQuerySet(&qsd);

    const size_t bytes = (size_t)cap * 2 * 8;

    wgpu::BufferDescriptor rbd = {};
    rbd.size  = bytes;
    rbd.usage = wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc;
    g_prof.resolve_buf = dev.device.CreateBuffer(&rbd);

    wgpu::BufferDescriptor mbd = {};
    mbd.size  = bytes;
    mbd.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    g_prof.read_buf = dev.device.CreateBuffer(&mbd);

    g_prof.capacity = cap;
}

// Emit a one-off empty compute pass whose sole purpose is to write a single
// timestamp at `qs_index`. Dawn requires at least one of begin/end indices
// on timestampWrites; we set only beginningOfPassWriteIndex and leave
// endOfPassWriteIndex as kQuerySetIndexUndefined.
void emit_timestamp(uint32_t qs_index) {
    wgpu::PassTimestampWrites tsw = {};
    tsw.querySet                  = g_prof.qs;
    tsw.beginningOfPassWriteIndex = qs_index;
    // endOfPassWriteIndex stays at the kQuerySetIndexUndefined default.

    wgpu::ComputePassDescriptor cpd = {};
    cpd.timestampWrites = &tsw;

    auto pass = shared_encoder().BeginComputePass(&cpd);
    pass.End();
}

} // namespace

bool op_profiling_enabled() {
    if (!env_enabled())        return false;
    if (!has_timestamp_query()) return false;
    return true;
}

void op_profiler_begin_run(int n_ops) {
    if (!op_profiling_enabled()) {
        g_prof.active = false;
        return;
    }
    g_prof.active          = true;
    g_prof.n_ops_this_run  = n_ops;
    ensure_capacity(n_ops);
    g_prof.last_times_us.assign((size_t)n_ops, 0.0);
}

void op_profiler_op_begin(int op_idx) {
    if (!g_prof.active)                                  return;
    if (op_idx < 0 || op_idx >= g_prof.n_ops_this_run)   return;
    emit_timestamp((uint32_t)(op_idx * 2));
}

void op_profiler_op_end(int op_idx) {
    if (!g_prof.active)                                  return;
    if (op_idx < 0 || op_idx >= g_prof.n_ops_this_run)   return;
    emit_timestamp((uint32_t)(op_idx * 2 + 1));
}

void op_profiler_pre_flush() {
    if (!g_prof.active)                  return;
    if (g_prof.n_ops_this_run <= 0)      return;

    const uint32_t n_ts = (uint32_t)g_prof.n_ops_this_run * 2;
    auto& enc = shared_encoder();
    enc.ResolveQuerySet(g_prof.qs, 0, n_ts, g_prof.resolve_buf, 0);
    enc.CopyBufferToBuffer(g_prof.resolve_buf, 0, g_prof.read_buf, 0,
                           (uint64_t)n_ts * 8);
}

void op_profiler_post_flush() {
    if (!g_prof.active)                  return;
    if (g_prof.n_ops_this_run <= 0)      return;

    const size_t n_ts  = (size_t)g_prof.n_ops_this_run * 2;
    const size_t bytes = n_ts * 8;

    auto& dev = get_device();
    wgpu::Future f = g_prof.read_buf.MapAsync(
        wgpu::MapMode::Read, 0, bytes,
        wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::MapAsyncStatus, wgpu::StringView) {});
    dev.instance.WaitAny(f, UINT64_MAX);

    const uint64_t* ts =
        (const uint64_t*)g_prof.read_buf.GetConstMappedRange(0, bytes);
    if (ts) {
        // Per WebGPU spec, timestamps are delivered as nanoseconds. No
        // period conversion needed on a standards-conformant runtime.
        for (int i = 0; i < g_prof.n_ops_this_run; ++i) {
            const uint64_t t0 = ts[(size_t)i * 2];
            const uint64_t t1 = ts[(size_t)i * 2 + 1];
            const double us = (t1 > t0) ? (double)(t1 - t0) / 1000.0 : 0.0;
            g_prof.last_times_us[(size_t)i] = us;
        }
    }
    g_prof.read_buf.Unmap();

    // Clear for the next run so stale indices don't silently bracket the
    // wrong ops if op_profiler_begin_run() is skipped.
    g_prof.active         = false;
    g_prof.n_ops_this_run = 0;
}

const std::vector<double>& op_profiler_last_times_us() {
    return g_prof.last_times_us;
}

} // namespace nnr::webgpu
