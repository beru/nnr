#pragma once
// Lightweight instrumentation profiler.
// Accumulates time per named region, prints summary on demand.
//
// Usage:
//   NNR_PROFILE_SCOPE("gemm_pack_a");          // times until end of scope
//   NNR_PROFILE_BEGIN("fma_loop");              // manual begin
//   NNR_PROFILE_END("fma_loop");                // manual end
//   nnr::profiler::report();                    // print summary to stderr
//   nnr::profiler::reset();                     // clear all counters
//
// Define NNR_ENABLE_PROFILER to activate. Otherwise macros are no-ops.

#ifdef NNR_ENABLE_PROFILER

#include <cstdio>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
#include <intrin.h>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <time.h>
#endif

namespace nnr {

struct profiler {
    struct entry {
        const char* name;
        int64_t total_ns;
        int64_t count;
    };

    static constexpr int MAX_ENTRIES = 128;

    static entry* entries() { static entry e[MAX_ENTRIES] = {}; return e; }
    static int& num_entries() { static int n = 0; return n; }

    static entry& find_or_create(const char* name) {
        entry* e = entries();
        int& n = num_entries();
        for (int i = 0; i < n; i++) {
            if (e[i].name == name || strcmp(e[i].name, name) == 0)
                return e[i];
        }
        if (n < MAX_ENTRIES) {
            e[n].name = name;
            e[n].total_ns = 0;
            e[n].count = 0;
            return e[n++];
        }
        // overflow — reuse last slot
        return e[MAX_ENTRIES - 1];
    }

    static int64_t now_ns() {
#ifdef _WIN32
        static int64_t freq = 0;
        if (!freq) {
            LARGE_INTEGER f;
            QueryPerformanceFrequency(&f);
            freq = f.QuadPart;
        }
        LARGE_INTEGER t;
        QueryPerformanceCounter(&t);
        return t.QuadPart * 1000000000LL / freq;
#else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return ts.tv_sec * 1000000000LL + ts.tv_nsec;
#endif
    }

    static void record(const char* name, int64_t elapsed_ns) {
        auto& e = find_or_create(name);
        e.total_ns += elapsed_ns;
        e.count++;
    }

    // Lightweight counter-only: no timing overhead, just count executions.
    static void count(const char* name) {
        auto& e = find_or_create(name);
        e.count++;
    }

    static void report() {
        entry* e = entries();
        int n = num_entries();
        if (n == 0) return;

        // Sort by total_ns descending
        std::sort(e, e + n, [](const entry& a, const entry& b) {
            return a.total_ns > b.total_ns;
        });

        int64_t grand_total = 0;
        for (int i = 0; i < n; i++) grand_total += e[i].total_ns;

        // Separate timed entries from counter-only entries
        int n_timed = 0, n_count = 0;
        for (int i = 0; i < n; i++) {
            if (e[i].count == 0) continue;
            if (e[i].total_ns > 0) n_timed++; else n_count++;
        }

        fprintf(stderr, "\n=== NNR Profile ===\n");
        if (n_timed) {
            fprintf(stderr, "%-40s %12s %10s %10s %6s\n", "Region", "Total(ms)", "Count", "Avg(us)", "%");
            fprintf(stderr, "%-40s %12s %10s %10s %6s\n", "------", "--------", "-----", "------", "-");
            for (int i = 0; i < n; i++) {
                if (e[i].count == 0 || e[i].total_ns == 0) continue;
                double total_ms = e[i].total_ns / 1e6;
                double avg_us = e[i].total_ns / (double)e[i].count / 1e3;
                double pct = grand_total > 0 ? 100.0 * e[i].total_ns / grand_total : 0;
                fprintf(stderr, "%-40s %12.3f %10lld %10.1f %5.1f%%\n",
                        e[i].name, total_ms, (long long)e[i].count, avg_us, pct);
            }
            fprintf(stderr, "%-40s %12.3f\n", "TOTAL", grand_total / 1e6);
        }
        if (n_count) {
            fprintf(stderr, "\n%-40s %10s\n", "Counter", "Count");
            fprintf(stderr, "%-40s %10s\n", "-------", "-----");
            for (int i = 0; i < n; i++) {
                if (e[i].count == 0 || e[i].total_ns != 0) continue;
                fprintf(stderr, "%-40s %10lld\n", e[i].name, (long long)e[i].count);
            }
        }
        fprintf(stderr, "===================\n\n");
    }

    static void reset() {
        num_entries() = 0;
    }

    struct scope_guard {
        const char* name;
        int64_t t0;
        scope_guard(const char* n) : name(n), t0(now_ns()) {}
        ~scope_guard() { record(name, now_ns() - t0); }
    };
};

} // namespace nnr

#define NNR_PROFILE_SCOPE(name) ::nnr::profiler::scope_guard _prof_##__LINE__(name)
#define NNR_PROFILE_BEGIN(name) auto _prof_t_##__LINE__ = ::nnr::profiler::now_ns()
#define NNR_PROFILE_END(name) ::nnr::profiler::record(name, ::nnr::profiler::now_ns() - _prof_t_##__LINE__)
#define NNR_PROFILE_REPORT() ::nnr::profiler::report()
#define NNR_PROFILE_RESET() ::nnr::profiler::reset()
#define NNR_PROFILE_COUNT(name) ::nnr::profiler::count(name)

#else // !NNR_ENABLE_PROFILER

#define NNR_PROFILE_SCOPE(name) ((void)0)
#define NNR_PROFILE_BEGIN(name) ((void)0)
#define NNR_PROFILE_END(name) ((void)0)
#define NNR_PROFILE_REPORT() ((void)0)
#define NNR_PROFILE_RESET() ((void)0)
#define NNR_PROFILE_COUNT(name) ((void)0)

#endif // NNR_ENABLE_PROFILER
