#pragma once
// Lightweight Chrome-trace emitter for Perfetto UI / chrome://tracing.
//
// Usage:
//   NNR_TRACE_SCOPE("conv", "conv1x1_nchwc");                    // scoped slice
//   NNR_TRACE_SCOPE("gemm", "dgemm_nhwc", "tiles", 6, "nt", 12); // with args
//   NNR_TRACE_BEGIN("io", "load_model");                          // manual begin
//   NNR_TRACE_END("io", "load_model");                            // manual end
//   NNR_TRACE_COUNTER("pool", "active_threads", 12);              // counter
//   nnr::trace::dump("trace.json");                               // write file
//   nnr::trace::reset();                                          // clear
//
// Define NNR_ENABLE_TRACE to activate. Otherwise macros are no-ops.
// Output loads in ui.perfetto.dev or chrome://tracing.

#ifdef NNR_ENABLE_TRACE

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <string_view>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace nnr {

struct trace {
    // Key-value arg pair (up to 4 per event)
    struct arg_t { const char* key; int64_t val; };

    struct event_t {
        std::string_view name;
        std::string_view cat;
        int64_t ts_ns;     // timestamp (ns from epoch)
        int64_t dur_ns;    // duration (0 for instant/begin/end)
        int32_t tid;
        char ph;           // 'X'=complete, 'B'=begin, 'E'=end, 'C'=counter
        uint8_t nargs;
        arg_t args[4];
    };

    static constexpr int MAX_EVENTS = 1 << 20;  // ~1M events, ~48MB

    // Defined in trace.cpp to guarantee single instance across static lib + exe.
    static event_t* events();
    static std::atomic<int>& count();

    static int64_t now_ns() {
#ifdef _WIN32
        static int64_t freq = 0;
        if (!freq) { LARGE_INTEGER f; QueryPerformanceFrequency(&f); freq = f.QuadPart; }
        LARGE_INTEGER t; QueryPerformanceCounter(&t);
        return t.QuadPart * 1000000000LL / freq;
#else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return ts.tv_sec * 1000000000LL + ts.tv_nsec;
#endif
    }

    static int32_t this_tid() {
        thread_local int32_t id = -1;
        if (id < 0) {
#ifdef _WIN32
            id = (int32_t)GetCurrentThreadId();
#else
            id = (int32_t)std::hash<std::thread::id>{}(std::this_thread::get_id());
#endif
        }
        return id;
    }

    static event_t* alloc() {
        int idx = count().fetch_add(1, std::memory_order_relaxed);
        if (idx >= MAX_EVENTS) { count().store(MAX_EVENTS, std::memory_order_relaxed); return nullptr; }
        return &events()[idx];
    }

    static void emit_complete(std::string_view cat, std::string_view name, int64_t ts, int64_t dur,
                              const arg_t* args = nullptr, int nargs = 0) {
        auto* e = alloc();
        if (!e) return;
        e->name = name; e->cat = cat; e->ts_ns = ts; e->dur_ns = dur;
        e->tid = this_tid(); e->ph = 'X'; e->nargs = (uint8_t)nargs;
        for (int i = 0; i < nargs && i < 4; i++) e->args[i] = args[i];
    }

    static void emit_begin(std::string_view cat, std::string_view name) {
        auto* e = alloc();
        if (!e) return;
        e->name = name; e->cat = cat; e->ts_ns = now_ns(); e->dur_ns = 0;
        e->tid = this_tid(); e->ph = 'B'; e->nargs = 0;
    }

    static void emit_end(std::string_view cat, std::string_view name) {
        auto* e = alloc();
        if (!e) return;
        e->name = name; e->cat = cat; e->ts_ns = now_ns(); e->dur_ns = 0;
        e->tid = this_tid(); e->ph = 'E'; e->nargs = 0;
    }

    static void emit_counter(std::string_view cat, std::string_view name, int64_t value) {
        auto* e = alloc();
        if (!e) return;
        e->name = name; e->cat = cat; e->ts_ns = now_ns(); e->dur_ns = 0;
        e->tid = this_tid(); e->ph = 'C'; e->nargs = 1;
        e->args[0] = {"value", value};
    }

    static void dump(const char* path) {
        FILE* f = fopen(path, "w");
        if (!f) return;
        int n = std::min((int)count().load(), MAX_EVENTS);
        auto* ev = events();

        // Find min timestamp for relative times
        int64_t t0 = n > 0 ? ev[0].ts_ns : 0;
        for (int i = 1; i < n; i++)
            if (ev[i].ts_ns < t0) t0 = ev[i].ts_ns;

        fprintf(f, "[");
        bool first = true;
        for (int i = 0; i < n; i++) {
            auto& e = ev[i];
            if (!first) fprintf(f, ",");
            first = false;

            double ts_us = (e.ts_ns - t0) / 1000.0;

            if (e.ph == 'X') {
                double dur_us = e.dur_ns / 1000.0;
                fprintf(f, "\n{\"ph\":\"X\",\"cat\":\"%.*s\",\"name\":\"%.*s\","
                    "\"ts\":%.3f,\"dur\":%.3f,\"tid\":%d,\"pid\":0",
                    (int)e.cat.size(), e.cat.data(), (int)e.name.size(), e.name.data(),
                    ts_us, dur_us, e.tid);
            } else if (e.ph == 'C') {
                fprintf(f, "\n{\"ph\":\"C\",\"cat\":\"%.*s\",\"name\":\"%.*s\","
                    "\"ts\":%.3f,\"tid\":%d,\"pid\":0",
                    (int)e.cat.size(), e.cat.data(), (int)e.name.size(), e.name.data(),
                    ts_us, e.tid);
            } else {
                fprintf(f, "\n{\"ph\":\"%c\",\"cat\":\"%.*s\",\"name\":\"%.*s\","
                    "\"ts\":%.3f,\"tid\":%d,\"pid\":0",
                    e.ph, (int)e.cat.size(), e.cat.data(), (int)e.name.size(), e.name.data(),
                    ts_us, e.tid);
            }

            // Args
            if (e.nargs > 0) {
                fprintf(f, ",\"args\":{");
                for (int a = 0; a < e.nargs; a++) {
                    if (a > 0) fprintf(f, ",");
                    fprintf(f, "\"%s\":%lld", e.args[a].key, (long long)e.args[a].val);
                }
                fprintf(f, "}");
            }
            fprintf(f, "}");
        }
        fprintf(f, "\n]\n");
        fclose(f);
        fprintf(stderr, "[trace] wrote %d events to %s\n", n, path);
    }

    static void reset() { count().store(0, std::memory_order_relaxed); }
};

// RAII scoped trace event
struct trace_scope {
    std::string_view cat;
    std::string_view name;
    int64_t ts;
    trace::arg_t args[4];
    int nargs;

    trace_scope(std::string_view c, std::string_view n)
        : cat(c), name(n), ts(trace::now_ns()), nargs(0) {}

    trace_scope(std::string_view c, std::string_view n,
                const char* k0, int64_t v0)
        : cat(c), name(n), ts(trace::now_ns()), nargs(1) { args[0] = {k0, v0}; }

    trace_scope(std::string_view c, std::string_view n,
                const char* k0, int64_t v0, const char* k1, int64_t v1)
        : cat(c), name(n), ts(trace::now_ns()), nargs(2) { args[0] = {k0, v0}; args[1] = {k1, v1}; }

    trace_scope(std::string_view c, std::string_view n,
                const char* k0, int64_t v0, const char* k1, int64_t v1,
                const char* k2, int64_t v2)
        : cat(c), name(n), ts(trace::now_ns()), nargs(3) {
        args[0] = {k0, v0}; args[1] = {k1, v1}; args[2] = {k2, v2};
    }

    ~trace_scope() {
        trace::emit_complete(cat, name, ts, trace::now_ns() - ts, args, nargs);
    }
};

} // namespace nnr

#define NNR_TRACE_SCOPE(cat, name, ...) \
    nnr::trace_scope _trace_##__LINE__(cat, name, ##__VA_ARGS__)

#define NNR_TRACE_BEGIN(cat, name)   nnr::trace::emit_begin(cat, name)
#define NNR_TRACE_END(cat, name)     nnr::trace::emit_end(cat, name)
#define NNR_TRACE_COUNTER(cat, name, val) nnr::trace::emit_counter(cat, name, val)
#define NNR_TRACE_DUMP(path)         nnr::trace::dump(path)
#define NNR_TRACE_RESET()            nnr::trace::reset()

#else // !NNR_ENABLE_TRACE

#define NNR_TRACE_SCOPE(cat, name, ...) ((void)0)
#define NNR_TRACE_BEGIN(cat, name)      ((void)0)
#define NNR_TRACE_END(cat, name)        ((void)0)
#define NNR_TRACE_COUNTER(cat, name, val) ((void)0)
#define NNR_TRACE_DUMP(path)            ((void)0)
#define NNR_TRACE_RESET()               ((void)0)

#endif // NNR_ENABLE_TRACE
