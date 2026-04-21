#pragma once
// Persistent thread pool replacing OpenMP. Zero external dependencies.
//
// Design: grouped gen counters + adaptive spin + MONITORX/MWAITX hardware wait.
// All workers use the same wait strategy (no leader/follower distinction):
//   1. Adaptive spin (catches rapid back-to-back dispatches, ~65ns wake)
//   2. MONITORX/MWAITX (hardware cache-line monitor, ~460ns wake, zero CPU)
// gen++ from main thread writes the cache line, which automatically wakes
// all cores monitoring it via hardware — no Wake* kernel calls needed.
//
// Grouping (default 8 threads/group, matching Zen4 CCD topology):
// Small dispatches only increment group 0's gen, other groups stay asleep.
// SMT-aware affinity spreads threads across physical cores first.
//
// Define NNR_NO_THREAD_POOL to compile in single-threaded mode.
// In that mode only a minimal serial stub is compiled; no threading headers
// are pulled in and no worker threads are created.

#include "cpu_features.h"  // always needed (defines NNR_ARCH_X64 / NNR_ARCH_ARM64)

#ifndef NNR_NO_THREAD_POOL // full multi-threaded implementation below

#if defined(NNR_USE_POOL_V2)
// v2 adaptive dispatch: static path is v1-equivalent (zero shared atomics);
// dynamic path uses 8-way sharded LoopCounter + stealing for range > 48,
// falls back to static below. See thread_pool_v2.h header for full design
// notes and kb/thread_pool_rewrite_session1.md for the iteration record.
#include "thread_pool_v2.h"
#else  // v1 body

#include <atomic>
#include <thread>
#include <vector>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include "cpu_features.h"
#include "aligned_alloc.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#elif defined(__APPLE__)
// __ulock_wait / __ulock_wake: lightweight kernel sleep (used by libdispatch)
extern "C" int __ulock_wait(uint32_t op, void* addr, uint64_t value, uint32_t timeout_us);
extern "C" int __ulock_wake(uint32_t op, void* addr, uint64_t wake_value);
#define UL_COMPARE_AND_WAIT 1
#define ULF_WAKE_ALL        0x00000100
#endif

#if defined(_M_AMD64) || defined(_M_X64) || defined(__x86_64__) || defined(__SSE2__)
#include <immintrin.h>
#define POOL_SPIN_PAUSE() _mm_pause()
#else
#define POOL_SPIN_PAUSE() std::this_thread::yield()
#endif

namespace nnr {

class thread_pool_t {
    static inline int configured_threads_ = 0;  // 0 = auto (physical cores)
public:
    // Set thread count before first use. 0 = auto (physical cores).
    // Must be called before get() — typically via context_t::set_num_threads().
    static void configure(int num_threads) { configured_threads_ = num_threads; }

    static thread_pool_t& get() {
        static thread_pool_t instance;
        return instance;
    }

    int num_threads() const { return nthreads_; }
    int num_physical() const { return std::max(1, nthreads_ / smt_stride_); }

    // Physical cores sharing one L3 cache (one CCD on AMD Zen, one package on
    // monolithic parts). Bandwidth-sensitive kernels cap parallelism at this
    // value to avoid cross-CCD wake/barrier overhead and Infinity-Fabric
    // coherence traffic. Falls back to num_physical() when detection fails
    // (non-Windows/Linux, or monolithic topology).
    int num_l3_physical() const {
        int p = num_physical();
        if (l3_physical_ > 0 && l3_physical_ < p) return l3_physical_;
        return p;
    }

    // Put all workers into deep sleep (skip adaptive spin, go straight to OS sleep).
    // Call after model load or when no inference is expected for a while.
    void sleep() { sleeping_.store(true, std::memory_order_release); }

    // Allow workers to resume adaptive spinning on next dispatch.
    // Not strictly required — dispatch() auto-wakes — but can be called
    // before a burst of inference to pre-warm the spin budget.
    void wake() { sleeping_.store(false, std::memory_order_release); }

    bool is_sleeping() const { return sleeping_.load(std::memory_order_acquire); }

    template <typename Fn>
    void for_static(int begin, int end, bool parallel, Fn&& fn) {
        int range = end - begin;
        if (range <= 0) return;
        if (!parallel || nthreads_ <= 1) {
            for (int i = begin; i < end; ++i) fn(i);
            return;
        }
        auto wrapper = [&fn](int /*tid*/, int i) { fn(i); };
        // `bool par` overload is used by element-wise and generic parallel
        // loops; cap at physical cores so small ops don't pay the SMT
        // dispatch-overhead penalty. Kernels that genuinely benefit from
        // SMT (int8 VNNI conv/gemm) must pass an explicit thread count.
        dispatch(wrapper, begin, end, false, num_physical());
    }

    // Thread-count-limited static scheduling.
    template <typename Fn>
    void for_static(int begin, int end, int num_threads, Fn&& fn) {
        int range = end - begin;
        if (range <= 0) return;
        if (num_threads <= 1 || nthreads_ <= 1) {
            for (int i = begin; i < end; ++i) fn(i);
            return;
        }
        auto wrapper = [&fn](int /*tid*/, int i) { fn(i); };
        dispatch(wrapper, begin, end, false, num_threads);
    }

    template <typename Fn>
    void for_dynamic(int begin, int end, bool parallel, Fn&& fn) {
        int range = end - begin;
        if (range <= 0) return;
        if (!parallel || nthreads_ <= 1) {
            for (int i = begin; i < end; ++i) fn(0, i);
            return;
        }
        // Same rationale as for_static(bool): cap at physical cores.
        dispatch(fn, begin, end, true, num_physical());
    }

    // Thread-count-limited dynamic scheduling (ORT-style cost-based parallelization).
    // Wakes exactly min(num_threads, range) threads. Serial if num_threads <= 1.
    template <typename Fn>
    void for_dynamic(int begin, int end, int num_threads, Fn&& fn) {
        int range = end - begin;
        if (range <= 0) return;
        if (num_threads <= 1 || nthreads_ <= 1) {
            for (int i = begin; i < end; ++i) fn(0, i);
            return;
        }
        dispatch(fn, begin, end, true, num_threads);
    }

    // Cost-based parallelization: caller provides estimated cost per work item
    // in "elements processed" units. The pool decides whether to parallelize
    // and how many threads to use based on the total element count.
    //
    // cost_per_item: elements processed per iteration (0 = always sequential).
    //   e.g. for a per-channel affine over W elements, pass cost_per_item = W.
    //
    // Thresholds are sized for the measured ~50µs thread-dispatch overhead on
    // Zen4 (see binary_math.cpp:100 comment). Below MIN_PARALLEL_COST total
    // elements, serial is strictly faster. Each additional thread needs
    // MIN_COST_PER_THREAD elements to justify its dispatch cost.
    //
    // Uses static scheduling (even chunks).
    template <typename Fn>
    void for_cost(int begin, int end, int64_t cost_per_item, Fn&& fn) {
        int range = end - begin;
        if (range <= 0) return;
        if (nthreads_ <= 1 || cost_per_item <= 0) {
            for (int i = begin; i < end; ++i) fn(i);
            return;
        }
        // Minimum total element count to justify parallelism. 500k floats ≈
        // 2 MB, which takes ~70µs single-threaded memory bandwidth on Zen4 —
        // the break-even point against ~50µs dispatch overhead.
        constexpr int64_t MIN_PARALLEL_COST = 500000;
        // Minimum elements per thread to justify the extra worker.
        constexpr int64_t MIN_COST_PER_THREAD = 100000;

        int64_t total_cost = (int64_t)range * cost_per_item;
        if (total_cost < MIN_PARALLEL_COST) {
            for (int i = begin; i < end; ++i) fn(i);
            return;
        }
        int max_threads = std::max(1, (int)(total_cost / MIN_COST_PER_THREAD));
        int nthreads = std::min({nthreads_, max_threads, range});
        auto wrapper = [&fn](int /*tid*/, int i) { fn(i); };
        dispatch(wrapper, begin, end, false, nthreads);
    }

    void ensure_scratch(size_t bytes) {
        if (bytes <= scratch_size_) return;
        for (int i = 0; i < nthreads_; ++i) {
            nnr_aligned_free(scratch_[i]);
            scratch_[i] = nnr_aligned_alloc(bytes, 64);
        }
        scratch_size_ = bytes;
    }

    void* scratch(int tid) const { return scratch_[tid]; }

private:
    struct alignas(64) group_t {
        std::atomic<unsigned> gen{0};
    };

    // Per-worker done flag: each on its own cache line, only written by its owner.
    // Zero contention — no fetch_add serialization between workers.
    struct alignas(64) worker_done_t {
        std::atomic<unsigned> epoch{0};
    };

    int nthreads_;
    int nworkers_;
    int group_size_;
    int ngroups_;
    int smt_stride_;
    int l3_physical_ = 0;   // physical cores sharing one L3 (CCD on AMD)
    bool has_monitorx_;
    std::vector<std::jthread> threads_;
    std::unique_ptr<group_t[]> groups_;
    std::unique_ptr<worker_done_t[]> worker_done_; // nworkers_ entries, indexed [tid-1]
    std::vector<void*> scratch_;
    size_t scratch_size_ = 0;

    alignas(64) void (*work_fn_)(thread_pool_t* pool, int tid) = nullptr;
    void* task_ctx_ = nullptr;
    int task_begin_ = 0;
    int task_end_ = 0;
    int task_nactive_ = 0;
    int nwoken_ = 0;
    // exit_ / dispatch_epoch_ are atomic: both are read by workers concurrently
    // with main-thread writes. Plain-type reads across threads are a C++ data
    // race (UB) and TSan flags them; x86 TSO papers over it in practice.
    std::atomic<unsigned> dispatch_epoch_{0};
    std::atomic<bool> exit_{false};

    alignas(64) std::atomic<int> startup_count_{0};
    alignas(64) std::atomic<int> dynamic_counter_{0};
    alignas(64) std::atomic<bool> sleeping_{true};

    // Adaptive spin: active workers spin longer, idle workers ramp down.
    // SPIN_MAX=512 iterations ≈ 10-20µs on modern CPUs (covers typical dispatch gap).
    // After IDLE_RAMPDOWN=4 consecutive non-participating dispatches, a worker
    // stops spinning entirely and falls through to MONITORX/WaitOnAddress sleep.
    static constexpr int SPIN_MAX = 512;
    static constexpr int IDLE_RAMPDOWN = 4;

    // Dispatch tracer (M1). Env-gated, zero cost when disabled.
    // NNR_POOL_TRACE=1 to enable, NNR_POOL_TRACE_FILE=<path> to redirect (default stderr).
    // Row format: range,nactive,nwoken,dynamic,total_ns,main_ns,barrier_ns
    // Main-thread only: dispatch() is never called concurrently, so no locking.
    bool trace_enabled_ = false;
    FILE* trace_fp_ = nullptr;
    bool trace_fp_owned_ = false;
#ifdef _WIN32
    double trace_ns_per_tick_ = 0.0;
    static inline int64_t trace_now() {
        LARGE_INTEGER t; QueryPerformanceCounter(&t); return (int64_t)t.QuadPart;
    }
#else
    static inline int64_t trace_now() {
        struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
        return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
    }
#endif

    static bool detect_monitorx() {
        return cpu_features().monitorx;
    }

    static int detect_smt_stride() {
#ifdef _WIN32
        DWORD len = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
        if (len == 0) return 1;
        std::vector<char> buf(len);
        if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf.data(), &len))
            return 1;
        int physical = 0;
        DWORD offset = 0;
        while (offset < len) {
            auto* info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf.data() + offset);
            if (info->Relationship == RelationProcessorCore)
                physical++;
            offset += info->Size;
        }
        int logical = (int)std::thread::hardware_concurrency();
        return (logical > physical && physical > 0) ? (logical / physical) : 1;
#else
        return 1;
#endif
    }

    // Physical cores sharing one L3 cache. Returns minimum across detected
    // L3 groups (safe cap for dispatches distributed round-robin). 0 when
    // detection fails or topology is flat; caller treats 0 as "no cap".
    // See num_l3_physical() in the public API for the usage pattern.
    //
    // Intel chiplet parts (Meteor/Arrow/Lunar Lake) expose the SoC-tile LPE
    // island as a separate 2-core L3 domain. Taking the raw MIN would cap
    // every L3-gated kernel at 2 threads, wiping out performance on the
    // compute-tile P-cores. Floor the candidate groups at MIN_L3_GROUP_CORES
    // physical cores so LPE islands are ignored.
    static int detect_l3_physical(int smt_stride) {
        constexpr int MIN_L3_GROUP_CORES = 4;
        const int min_logical_floor = MIN_L3_GROUP_CORES * (smt_stride > 0 ? smt_stride : 1);
#ifdef _WIN32
        DWORD len = 0;
        GetLogicalProcessorInformationEx(RelationCache, nullptr, &len);
        if (len == 0) return 0;
        std::vector<char> buf(len);
        if (!GetLogicalProcessorInformationEx(RelationCache,
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf.data(), &len))
            return 0;
        int min_logical = 0;
        DWORD offset = 0;
        while (offset < len) {
            auto* info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf.data() + offset);
            if (info->Relationship == RelationCache && info->Cache.Level == 3) {
                KAFFINITY mask = info->Cache.GroupMask.Mask;
                int count = (int)__popcnt64((uint64_t)mask);
                if (count >= min_logical_floor
                    && (min_logical == 0 || count < min_logical))
                    min_logical = count;
            }
            offset += info->Size;
        }
        return (smt_stride > 0) ? (min_logical / smt_stride) : min_logical;
#elif defined(__linux__)
        FILE* f = std::fopen("/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list", "r");
        if (!f) return 0;
        int a = 0, b = 0;
        int n = std::fscanf(f, "%d-%d", &a, &b);
        std::fclose(f);
        if (n < 1) return 0;
        int logical = (n == 2) ? (b - a + 1) : 1;
        if (logical < min_logical_floor) return 0;
        return (smt_stride > 0) ? (logical / smt_stride) : logical;
#else
        (void)smt_stride;
        (void)min_logical_floor;
        return 0;
#endif
    }

    void trace_init() {
        const char* enabled = std::getenv("NNR_POOL_TRACE");
        if (!enabled || enabled[0] == '\0' || enabled[0] == '0') return;
        const char* path = std::getenv("NNR_POOL_TRACE_FILE");
        if (path && path[0] != '\0') {
            trace_fp_ = std::fopen(path, "w");
            trace_fp_owned_ = (trace_fp_ != nullptr);
        }
        if (!trace_fp_) trace_fp_ = stderr;
        trace_enabled_ = true;
#ifdef _WIN32
        LARGE_INTEGER freq; QueryPerformanceFrequency(&freq);
        trace_ns_per_tick_ = 1.0e9 / (double)freq.QuadPart;
#endif
        std::fprintf(trace_fp_, "range,nactive,nwoken,dynamic,total_ns,main_ns,barrier_ns\n");
    }

    thread_pool_t() {
        has_monitorx_ = detect_monitorx();
        smt_stride_ = detect_smt_stride();
        l3_physical_ = detect_l3_physical(smt_stride_);
        trace_init();

        // Default to all logical cores. Individual call sites control SMT
        // exposure: fp32 compute-bound ops (Conv, GEMM, Pool) cap themselves
        // at num_physical() via compute_threads(); int8 VNNI ops use all
        // logical threads via `for_dynamic(..., bool par=true)` because their
        // weight-streaming traffic benefits from SMT latency hiding.
        // Override via context_t::set_num_threads() before prepare()/run().
        int logical = std::max(1, (int)std::thread::hardware_concurrency());
        nthreads_ = (configured_threads_ > 0) ? configured_threads_
                                               : logical;
        nworkers_ = nthreads_ - 1;

        group_size_ = 8;
        ngroups_ = (nthreads_ + group_size_ - 1) / group_size_;

        scratch_.resize(nthreads_, nullptr);
        if (nworkers_ > 0) {
            groups_ = std::make_unique<group_t[]>(ngroups_);
            worker_done_ = std::make_unique<worker_done_t[]>(nworkers_);
#ifdef _WIN32
            SetThreadAffinityMask(GetCurrentThread(), 1ULL << 0);
#endif
            for (int i = 1; i < nthreads_; ++i)
                threads_.emplace_back([this, i](std::stop_token) { worker_main(i); });
            while (startup_count_.load(std::memory_order_acquire) < nworkers_)
                POOL_SPIN_PAUSE();
        }
    }

    ~thread_pool_t() {
        if (nworkers_ > 0) {
            // Wait for the last dispatch's workers to finish before tearing down.
            // dispatch() already barrier-waits workers [1..nwoken_); this is a
            // defensive no-op in well-formed code that closes process-exit
            // races where a worker is still mid-store in worker_done_ or
            // otherwise touching shared state (scratch, task_ctx_) when the
            // Meyers singleton dtor runs.
            unsigned last_epoch = dispatch_epoch_.load(std::memory_order_acquire);
            if (last_epoch > 0) {
                for (int t = 1; t < nwoken_; ++t) {
                    while (worker_done_[t - 1].epoch.load(std::memory_order_acquire) < last_epoch)
                        POOL_SPIN_PAUSE();
                }
            }
            exit_.store(true, std::memory_order_release);
            for (int g = 0; g < ngroups_; ++g)
                groups_[g].gen.store(UINT_MAX, std::memory_order_release);
            // Wake all workers (MONITORX workers may be in WaitOnAddress fallback)
            {
                for (int g = 0; g < ngroups_; ++g) {
#ifdef _WIN32
                    WakeByAddressAll((void*)&groups_[g].gen);
#elif defined(__linux__)
                    syscall(SYS_futex, &groups_[g].gen, FUTEX_WAKE_PRIVATE, INT_MAX,
                        nullptr, nullptr, 0);
#elif defined(__APPLE__)
                    __ulock_wake(UL_COMPARE_AND_WAIT | ULF_WAKE_ALL, (void*)&groups_[g].gen, 0);
#endif
                }
            }
            threads_.clear();
        }
        for (int i = 0; i < nthreads_; ++i)
            nnr_aligned_free(scratch_[i]);
        if (trace_fp_ && trace_fp_owned_) std::fclose(trace_fp_);
    }

    thread_pool_t(const thread_pool_t&) = delete;
    thread_pool_t& operator=(const thread_pool_t&) = delete;

    int tid_to_processor(int tid) const {
        int nphysical = nthreads_ / smt_stride_;
        if (nphysical <= 0) nphysical = nthreads_;
        if (tid < nphysical)
            return tid * smt_stride_;
        else
            return (tid - nphysical) * smt_stride_ + 1;
    }

    enum class wait_mode { monitorx, wait_on_address, spin };

    template <wait_mode Mode>
    void worker_main_impl(int tid) {
#ifdef _WIN32
        SetThreadAffinityMask(GetCurrentThread(), 1ULL << tid_to_processor(tid));
#endif
        int grp = tid / group_size_;
        unsigned my_gen = 0;
        int idle_count = 0;
        startup_count_.fetch_add(1, std::memory_order_release);
        for (;;) {
            unsigned gen;

            // Phase 1: adaptive spin (catches rapid dispatches in ~65ns)
            // Skipped entirely when pool is in sleep mode.
            int spin_budget = sleeping_.load(std::memory_order_acquire) ? 0
                : (idle_count >= IDLE_RAMPDOWN) ? 0
                : SPIN_MAX - idle_count * (SPIN_MAX / IDLE_RAMPDOWN);
            bool caught = false;
            for (int s = 0; s < spin_budget; ++s) {
                gen = groups_[grp].gen.load(std::memory_order_acquire);
                if (gen != my_gen) { caught = true; break; }
                POOL_SPIN_PAUSE();
            }

            // Phase 2: sleep until gen changes
            if (!caught) {
                for (;;) {
                    gen = groups_[grp].gen.load(std::memory_order_acquire);
                    if (gen != my_gen) break;
                    if constexpr (Mode == wait_mode::monitorx) {
#ifdef NNR_ARCH_X64
                        _mm_monitorx((void*)&groups_[grp].gen, 0, 0);
                        gen = groups_[grp].gen.load(std::memory_order_acquire);
                        if (gen != my_gen) break;
                        _mm_mwaitx(0, 0, 0);
                        gen = groups_[grp].gen.load(std::memory_order_acquire);
                        if (gen != my_gen) break;
                        // MWAITX wakes on timer interrupts (~1-15ms on Windows).
                        // In sleep mode, fall back to OS sleep to avoid hot loop.
                        // In active mode, retry MONITORX (quick wake needed).
                        if (sleeping_.load(std::memory_order_acquire)) {
#ifdef _WIN32
                            unsigned cmp = my_gen;
                            WaitOnAddress((volatile void*)&groups_[grp].gen, &cmp, sizeof(unsigned), INFINITE);
#elif defined(__linux__)
                            syscall(SYS_futex, &groups_[grp].gen, FUTEX_WAIT_PRIVATE,
                                my_gen, nullptr, nullptr, 0);
#endif
                        }
#else
                        POOL_SPIN_PAUSE();
#endif
                    } else if constexpr (Mode == wait_mode::wait_on_address) {
#ifdef _WIN32
                        unsigned cmp = my_gen;
                        WaitOnAddress((volatile void*)&groups_[grp].gen, &cmp, sizeof(unsigned), INFINITE);
#elif defined(__linux__)
                        syscall(SYS_futex, &groups_[grp].gen, FUTEX_WAIT_PRIVATE,
                            my_gen, nullptr, nullptr, 0);
#elif defined(__APPLE__)
                        __ulock_wait(UL_COMPARE_AND_WAIT, (void*)&groups_[grp].gen,
                            my_gen, 0);
#else
                        POOL_SPIN_PAUSE();
#endif
                    } else {
                        POOL_SPIN_PAUSE();
                    }
                }
            }

            my_gen = gen;
            if (exit_.load(std::memory_order_acquire)) return;

            if (tid < task_nactive_) {
                work_fn_(this, tid);
                idle_count = 0;
            } else {
                idle_count = std::min(idle_count + 1, IDLE_RAMPDOWN);
            }

            // Sense-reversing: each worker writes only its own cache line (zero contention).
            // dispatch_epoch_ was released by main prior to the gen increment that woke us,
            // so the acquire load below sees the value paired with this dispatch.
            worker_done_[tid - 1].epoch.store(
                dispatch_epoch_.load(std::memory_order_acquire),
                std::memory_order_release);
        }
    }

    void worker_main(int tid) {
#ifdef NNR_ARCH_X64
        if (has_monitorx_)
            worker_main_impl<wait_mode::monitorx>(tid);
        else
#endif
            worker_main_impl<wait_mode::wait_on_address>(tid);
    }

    template <typename Fn, bool Dynamic>
    static void work_impl(thread_pool_t* pool, int tid) {
        Fn& fn = *static_cast<Fn*>(pool->task_ctx_);
        if constexpr (Dynamic) {
            int end = pool->task_end_;
            for (;;) {
                int i = pool->dynamic_counter_.fetch_add(1, std::memory_order_relaxed);
                if (i >= end) break;
                fn(tid, i);
            }
        } else {
            int nactive = pool->task_nactive_;
            int range = pool->task_end_ - pool->task_begin_;
            int chunk = (range + nactive - 1) / nactive;
            int lo = pool->task_begin_ + tid * chunk;
            int hi = std::min(lo + chunk, pool->task_end_);
            for (int i = lo; i < hi; ++i)
                fn(tid, i);
        }
    }

    template <typename Fn>
    void dispatch(Fn& fn, int begin, int end, bool dynamic, int max_threads = 0) {
        int range = end - begin;
        int thread_limit = max_threads > 0 ? std::min(nthreads_, max_threads) : nthreads_;
        int nactive = std::min(thread_limit, range);
        int groups_needed = (nactive + group_size_ - 1) / group_size_;
        int nwoken = std::min(nthreads_, groups_needed * group_size_);

        int64_t t_begin = 0, t_main_done = 0, t_end = 0;
        if (trace_enabled_) t_begin = trace_now();

        task_ctx_ = &fn;
        task_begin_ = begin;
        task_end_ = end;
        task_nactive_ = nactive;
        nwoken_ = nwoken;
        // acq_rel so workers observe the new dispatch_epoch_ when they acquire
        // the following gen increment; the barrier below acquires this value.
        unsigned epoch = dispatch_epoch_.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (dynamic) {
            work_fn_ = &work_impl<Fn, true>;
            dynamic_counter_.store(begin, std::memory_order_relaxed);
        } else {
            work_fn_ = &work_impl<Fn, false>;
        }

        // Don't clear sleeping_ here — workers are woken explicitly via
        // WakeByAddressAll/MONITORX below. Clearing sleeping_ would let workers
        // spin at SPIN_MAX after the dispatch completes, burning CPU until
        // pool_sleep() is called again (race condition: workers re-enter their
        // loop before the caller can call pool_sleep()).

        // gen++ writes the cache line — MONITORX cores wake via hardware,
        // spinning cores see the change via acquire load.
        for (int g = 0; g < groups_needed; ++g)
            groups_[g].gen.fetch_add(1, std::memory_order_release);
        // Explicitly wake workers via OS primitives. Required even on MONITORX
        // platforms because workers in sleep mode may have fallen through to
        // WaitOnAddress after spurious MWAITX wakes.
        {
            for (int g = 0; g < groups_needed; ++g) {
#ifdef _WIN32
                WakeByAddressAll((void*)&groups_[g].gen);
#elif defined(__linux__)
                syscall(SYS_futex, &groups_[g].gen, FUTEX_WAKE_PRIVATE, INT_MAX,
                    nullptr, nullptr, 0);
#elif defined(__APPLE__)
                __ulock_wake(UL_COMPARE_AND_WAIT | ULF_WAKE_ALL, (void*)&groups_[g].gen, 0);
#endif
            }
        }

        work_fn_(this, 0);

        if (trace_enabled_) t_main_done = trace_now();

        // Per-worker scan: each worker wrote its own cache line (zero contention).
        // Main reads nwoken-1 cache lines — all shared-reads, no bus writes.
        for (int t = 1; t < nwoken; ++t) {
            while (worker_done_[t - 1].epoch.load(std::memory_order_acquire) < epoch)
                POOL_SPIN_PAUSE();
        }

        if (trace_enabled_) {
            t_end = trace_now();
#ifdef _WIN32
            double nsp = trace_ns_per_tick_;
            int64_t total_ns   = (int64_t)((t_end - t_begin) * nsp);
            int64_t main_ns    = (int64_t)((t_main_done - t_begin) * nsp);
            int64_t barrier_ns = (int64_t)((t_end - t_main_done) * nsp);
#else
            int64_t total_ns   = t_end - t_begin;
            int64_t main_ns    = t_main_done - t_begin;
            int64_t barrier_ns = t_end - t_main_done;
#endif
            std::fprintf(trace_fp_, "%d,%d,%d,%d,%lld,%lld,%lld\n",
                range, nactive, nwoken, dynamic ? 1 : 0,
                (long long)total_ns, (long long)main_ns, (long long)barrier_ns);
        }
    }
};

} // namespace nnr

#endif // NNR_USE_POOL_V2

#endif // !NNR_NO_THREAD_POOL

// ── Free function wrappers ────────────────────────────────────────────────────
// Template replacements for NNR_FOR_STATIC / NNR_FOR_DYNAMIC macros.
// Two compile modes: serial (NNR_NO_THREAD_POOL) and thread pool (default).

namespace nnr {

template <typename Fn>
inline void for_static(int begin, int end, bool parallel, Fn&& fn) {
#ifdef NNR_NO_THREAD_POOL
    (void)parallel;
    for (int i = begin; i < end; ++i) fn(i);
#else
    thread_pool_t::get().for_static(begin, end, parallel, std::forward<Fn>(fn));
#endif
}

template <typename Fn>
inline void for_static(int begin, int end, int num_threads, Fn&& fn) {
#ifdef NNR_NO_THREAD_POOL
    (void)num_threads;
    for (int i = begin; i < end; ++i) fn(i);
#else
    thread_pool_t::get().for_static(begin, end, num_threads, std::forward<Fn>(fn));
#endif
}

template <typename Fn>
inline void for_dynamic(int begin, int end, bool parallel, Fn&& fn) {
#ifdef NNR_NO_THREAD_POOL
    (void)parallel;
    for (int i = begin; i < end; ++i) fn(0, i);
#else
    thread_pool_t::get().for_dynamic(begin, end, parallel, std::forward<Fn>(fn));
#endif
}

template <typename Fn>
inline void for_dynamic(int begin, int end, int num_threads, Fn&& fn) {
#ifdef NNR_NO_THREAD_POOL
    (void)num_threads;
    for (int i = begin; i < end; ++i) fn(0, i);
#else
    thread_pool_t::get().for_dynamic(begin, end, num_threads, std::forward<Fn>(fn));
#endif
}

// Cost-based parallelization: auto-selects thread count from work estimate.
// cost_per_item: estimated work per iteration (see thread_pool_t::for_cost).
template <typename Fn>
inline void for_cost(int begin, int end, int64_t cost_per_item, Fn&& fn) {
#ifdef NNR_NO_THREAD_POOL
    (void)cost_per_item;
    for (int i = begin; i < end; ++i) fn(i);
#else
    thread_pool_t::get().for_cost(begin, end, cost_per_item, std::forward<Fn>(fn));
#endif
}

// Thread count for compute-bound fp32 ops (GEMM, Conv, Pool).
// SMT siblings share FMA/ALU units — zero throughput gain for compute-bound
// fp32 work, but waking 2× threads adds dispatch overhead that dominates
// small ops. Cap at physical cores to avoid SMT contention.
inline int compute_threads(int work_items) {
#ifndef NNR_NO_THREAD_POOL
    return std::clamp(work_items, 1, thread_pool_t::get().num_physical());
#else
    return 1;
#endif
}

// Thread count for int8 VNNI compute-bound ops (QLinearConv, QLinearMatMul,
// int8 GEMM). Cost-aware:
//   - total_ops < 1M:        serial (dispatch overhead dominates)
//   - total_ops < phys × 4M: up to num_physical (SMT not worth it for
//                            modest work — 12T covers small-layer models)
//   - total_ops ≥ phys × 4M: up to num_threads (large layers benefit from
//                            SMT latency hiding during weight streaming)
// Measured on Zen4 Ryzen 9 7900X: large-layer int8 models gain 22-35%
// from SMT (vgg16-12-int8, ssd-12-int8, vgg16-12-qdq); small-layer models
// (densenet-12-int8) stay at 12T because 24T regresses on small work.
inline int int8_compute_threads(int work_items, int64_t total_ops) {
#ifndef NNR_NO_THREAD_POOL
    auto& pool = thread_pool_t::get();
    if (total_ops < (1 << 20)) return 1;           // < 1M ops
    int phys = pool.num_physical();
    int64_t smt_threshold = (int64_t)phys * (1 << 22);  // phys × 4M
    int cap = (total_ops >= smt_threshold) ? pool.num_threads() : phys;
    return std::clamp(work_items, 1, cap);
#else
    (void)total_ops;
    return 1;
#endif
}

// ORT-style cost-based thread count for element-wise ops.
// Returns how many threads are justified by the total work.
// bytes_in/bytes_out: bytes per element loaded/stored.
// compute: estimated cycles per element beyond memory access.
inline int elementwise_threads(size_t N, float bytes_in, float bytes_out, float compute) {
    constexpr float kLoadCycles = 0.172f;   // ORT L2 cache cost: 1/64 * 11
    constexpr float kStartupCycles = 100000.0f;  // min total cost to justify parallelism
    constexpr float kPerThreadCycles = 100000.0f; // cost threshold per additional thread
    float cost = N * (bytes_in * kLoadCycles + bytes_out * kLoadCycles + compute);
    if (cost <= kStartupCycles) return 1;
    int nt = (int)((cost - kStartupCycles) / kPerThreadCycles + 0.9f);
#ifndef NNR_NO_THREAD_POOL
    return std::clamp(nt, 1, thread_pool_t::get().num_threads());
#else
    return 1;
#endif
}

// Put all worker threads into deep sleep (zero CPU when idle).
// Call after load/prepare when no inference is expected for a while.
// Workers auto-wake on next dispatch (run/for_static/for_dynamic).
inline void pool_sleep() {
#ifndef NNR_NO_THREAD_POOL
    thread_pool_t::get().sleep();
#endif
}

// Pre-warm workers back to adaptive spin mode before an inference burst.
// Optional — dispatch() auto-wakes, but this avoids the first-dispatch latency hit.
inline void pool_wake() {
#ifndef NNR_NO_THREAD_POOL
    thread_pool_t::get().wake();
#endif
}

} // namespace nnr

// ── Convenience macros ────────────────────────────────────────────────────────
// NNR_POOL_ENSURE_SCRATCH / NNR_POOL_SCRATCH — per-thread scratch (mode-specific)
// __VA_ARGS__ lets commas inside the lambda body pass through unharmed.

#if defined(NNR_NO_THREAD_POOL)

#include <cstdlib>

namespace nnr::detail {
inline void*& serial_scratch() { static void* p = nullptr; return p; }
inline size_t& serial_scratch_size() { static size_t s = 0; return s; }
} // namespace nnr::detail

#  define NNR_POOL_ENSURE_SCRATCH(bytes) \
     do { if ((size_t)(bytes) > ::nnr::detail::serial_scratch_size()) { \
         free(::nnr::detail::serial_scratch()); \
         ::nnr::detail::serial_scratch() = malloc(bytes); \
         ::nnr::detail::serial_scratch_size() = (size_t)(bytes); \
     } } while (0)
#  define NNR_POOL_SCRATCH(tid) (::nnr::detail::serial_scratch())

#else // default: custom thread pool

#  define NNR_POOL_ENSURE_SCRATCH(bytes)  ::nnr::thread_pool_t::get().ensure_scratch(bytes)
#  define NNR_POOL_SCRATCH(tid)           ::nnr::thread_pool_t::get().scratch(tid)
#  define NNR_POOL                        ::nnr::thread_pool_t::get()

#endif

// Helper for scrollable unary elementwise exec_strip implementations.
// Applies fn(src[i]) -> dst[i] over the strip rows, parallelized across outer dims.
// When the scroll executor sets up ring buffers, input and output tensors may have
// different H (dims[2]): the input uses the full tensor H while the output uses ring_H.
// The orig_H parameter overrides boundary clamping (real height, not ring_H).
// The out_dims/out_ndata parameters allow different output tensor dimensions when
// the output is ring-buffered with a different H than the input.
template <typename Fn>
inline bool exec_strip_elementwise(const float* px, float* py, size_t ndata,
    const int* dims, int ndim, int out_row_start, int out_rows, Fn fn,
    int orig_H = 0, const int* out_dims = nullptr, size_t out_ndata = 0)
{
    if (ndim < 3) return false;
    int iH = dims[ndim - 2];
    int W = dims[ndim - 1];
    int outer = (int)(ndata / (iH * W));
    int oH = (out_dims && out_dims[ndim - 2] != iH) ? out_dims[ndim - 2] : iH;
    int clamp_H = orig_H > 0 ? orig_H : oH;
    int out_end = std::min(out_row_start + out_rows, clamp_H);
    int count = (out_end - out_row_start) * W;
    if (count <= 0) return true;
    nnr::for_static(0, outer, outer > 4, [&](int nc) {
        const float* src = px + (size_t)nc * iH * W + (size_t)out_row_start * W;
        float* dst = py + (size_t)nc * oH * W + (size_t)out_row_start * W;
        for (int i = 0; i < count; ++i)
            dst[i] = fn(src[i]);
    });
    return true;
}
