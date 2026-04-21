#pragma once
// thread_pool_v2 — adaptive dispatch pool (M5 production port).
//
// Selected at compile time via NNR_USE_POOL_V2 in thread_pool.h. Defines the
// same nnr::thread_pool_t class as the v1 path; free-function wrappers and
// macros (NNR_POOL, NNR_POOL_ENSURE_SCRATCH, NNR_POOL_SCRATCH) live in
// thread_pool.h and resolve identically against either implementation.
//
// Adaptive dispatch design (from playground M3e):
//
//   for_static / for_cost  → v1-style pre-sliced static partition with
//                            per-worker epoch barrier. Zero shared atomics
//                            on the fast path. Matches v1 on the common
//                            case (balanced kernels, tiny ranges).
//
//   for_dynamic            → range ≤ FALLBACK_THRESHOLD (48): falls back to
//                            static (v1's single dynamic_counter pays 12-way
//                            fetch_add contention exceeding imbalance cost
//                            at small range).
//                            range > 48: 8-way sharded LoopCounter + round-
//                            robin work stealing. Block = shard_size / 4.
//                            Per-shard `drained` flag lets stealers skip
//                            empty shards via cheap acquire load.
//
// Shared infrastructure (both paths, identical to v1):
//   - Per-group gen counter (group_size=8) for wake broadcast via one
//     WakeByAddressAll per group.
//   - MONITORX/MWAITX hardware wait guarded by has_monitorx_.
//     sleeping_{true} preserved for Zen 4 boost (see
//     kb/thread_pool_zen4_boost.md).
//   - Per-worker epoch counters (zero contention barrier).
//   - SMT-stride-aware tid_to_processor (physical cores first).

#include "cpu_features.h"

#include <atomic>
#include <thread>
#include <vector>
#include <algorithm>
#include <climits>
#include <memory>
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
    static inline int configured_threads_ = 0;
public:
    static void configure(int num_threads) { configured_threads_ = num_threads; }

    static thread_pool_t& get() {
        static thread_pool_t instance;
        return instance;
    }

    int num_threads() const { return nthreads_; }
    int num_physical() const { return std::max(1, nthreads_ / smt_stride_); }

    // Physical cores sharing one L3 cache (= one CCD on AMD, one package on
    // monolithic parts). Bandwidth-sensitive kernels (Winograd, weight-heavy
    // convs) see wake/barrier overhead blow up once a dispatch crosses the
    // L3 boundary, because worker wakes now pay Infinity-Fabric latency and
    // shared data bounces through DRAM. Capping at this size keeps the
    // parallel region inside one L3 domain. Never larger than `num_physical()`;
    // on parts where detection fails or the package is monolithic, equals
    // `num_physical()`.
    int num_l3_physical() const {
        int p = num_physical();
        if (l3_physical_ > 0 && l3_physical_ < p) return l3_physical_;
        return p;
    }

    void sleep()  { sleeping_.store(true,  std::memory_order_release); }
    void wake()   { sleeping_.store(false, std::memory_order_release); }
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
        dispatch_static(wrapper, begin, end, num_physical());
    }

    template <typename Fn>
    void for_static(int begin, int end, int num_threads, Fn&& fn) {
        int range = end - begin;
        if (range <= 0) return;
        if (num_threads <= 1 || nthreads_ <= 1) {
            for (int i = begin; i < end; ++i) fn(i);
            return;
        }
        auto wrapper = [&fn](int /*tid*/, int i) { fn(i); };
        dispatch_static(wrapper, begin, end, num_threads);
    }

    template <typename Fn>
    void for_dynamic(int begin, int end, bool parallel, Fn&& fn) {
        int range = end - begin;
        if (range <= 0) return;
        if (!parallel || nthreads_ <= 1) {
            for (int i = begin; i < end; ++i) fn(0, i);
            return;
        }
        dispatch_dynamic(fn, begin, end, num_physical());
    }

    template <typename Fn>
    void for_dynamic(int begin, int end, int num_threads, Fn&& fn) {
        int range = end - begin;
        if (range <= 0) return;
        if (num_threads <= 1 || nthreads_ <= 1) {
            for (int i = begin; i < end; ++i) fn(0, i);
            return;
        }
        dispatch_dynamic(fn, begin, end, num_threads);
    }

    template <typename Fn>
    void for_cost(int begin, int end, int64_t cost_per_item, Fn&& fn) {
        int range = end - begin;
        if (range <= 0) return;
        if (nthreads_ <= 1 || cost_per_item <= 0) {
            for (int i = begin; i < end; ++i) fn(i);
            return;
        }
        constexpr int64_t MIN_PARALLEL_COST = 500000;
        constexpr int64_t MIN_COST_PER_THREAD = 100000;
        int64_t total_cost = (int64_t)range * cost_per_item;
        if (total_cost < MIN_PARALLEL_COST) {
            for (int i = begin; i < end; ++i) fn(i);
            return;
        }
        int max_threads = std::max(1, (int)(total_cost / MIN_COST_PER_THREAD));
        int nthreads = std::min({nthreads_, max_threads, range});
        auto wrapper = [&fn](int /*tid*/, int i) { fn(i); };
        // for_cost is cost-bucketed static — route to static path.
        dispatch_static(wrapper, begin, end, nthreads);
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
    static constexpr int GROUP_SIZE  = 8;
    static constexpr int MAX_SHARDS  = 8;
    static constexpr int SPIN_MAX    = 512;

    struct alignas(64) group_t {
        std::atomic<unsigned> gen{0};
    };

    struct alignas(64) worker_done_t {
        std::atomic<unsigned> epoch{0};
    };

    struct alignas(64) shard_t {
        std::atomic<int> next{0};
        int end{0};
        int block{1};
        // Set by any worker whose fetch_add first returns >= end. Stealers
        // check this first with an acquire load (cached cheap) before paying
        // the cost of a contended fetch_add on a drained shard. Reduces
        // stealing overhead for range=140-252 where shards drain quickly.
        std::atomic<bool> drained{false};
    };

    int nthreads_   = 1;
    int nworkers_   = 0;
    int ngroups_    = 0;
    int smt_stride_ = 1;
    int l3_physical_ = 0;   // physical cores sharing one L3 cache (one CCD on AMD)
    bool has_monitorx_ = false;
    std::vector<std::jthread> threads_;
    std::unique_ptr<group_t[]> groups_;
    std::unique_ptr<worker_done_t[]> worker_done_;  // nworkers_ entries, indexed [tid-1]
    std::vector<void*> scratch_;
    size_t scratch_size_ = 0;

    alignas(64) void (*work_fn_)(thread_pool_t* pool, int tid) = nullptr;
    void* task_ctx_ = nullptr;
    int task_begin_ = 0;
    int task_end_ = 0;
    int task_nactive_ = 0;
    int task_nshards_ = 0;
    int nwoken_ = 0;
    // exit_ / dispatch_epoch_ are atomic: both are read by workers concurrently
    // with main-thread writes. Plain-type reads across threads are a C++ data
    // race (UB) and TSan flags them; x86 TSO papers over it in practice.
    std::atomic<unsigned> dispatch_epoch_{0};

    shard_t shards_[MAX_SHARDS];

    alignas(64) std::atomic<int> startup_count_{0};
    alignas(64) std::atomic<bool> sleeping_{true};
    std::atomic<bool> exit_{false};

    static bool detect_monitorx() {
#ifdef NNR_ARCH_X64
        return nnr::cpu_features().monitorx;
#else
        return false;
#endif
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

    // Returns the number of physical cores sharing one L3 cache (= one CCD
    // on AMD Zen, one package on monolithic parts). 0 if detection fails or
    // topology is flat. Uses RelationCache/Level=3 on Windows and walks
    // /sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list on Linux.
    //
    // Returns the MINIMUM L3 group size across detected groups — that's the
    // safe cap when workers are distributed round-robin: once a dispatch
    // exceeds it, at least one worker lives on a different CCD than main.
    // Intel chiplet parts (Meteor/Arrow/Lunar Lake) expose the SoC-tile LPE
    // island as a separate 2-core L3 domain. Taking the raw MIN would cap
    // every L3-gated kernel at 2 threads. Floor candidate groups at
    // MIN_L3_GROUP_CORES physical cores so LPE islands are ignored.
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
        FILE* f = fopen("/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list", "r");
        if (!f) return 0;
        int a = 0, b = 0;
        int n = fscanf(f, "%d-%d", &a, &b);
        fclose(f);
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

    int tid_to_processor(int tid) const {
        int nphysical = nthreads_ / smt_stride_;
        if (nphysical <= 0) nphysical = nthreads_;
        if (tid < nphysical) return tid * smt_stride_;
        return (tid - nphysical) * smt_stride_ + 1;
    }

    thread_pool_t() {
        has_monitorx_ = detect_monitorx();
        smt_stride_ = detect_smt_stride();
        l3_physical_ = detect_l3_physical(smt_stride_);
        int logical = std::max(1, (int)std::thread::hardware_concurrency());
        nthreads_ = (configured_threads_ > 0) ? configured_threads_ : logical;
        nworkers_ = nthreads_ - 1;
        ngroups_ = (nthreads_ + GROUP_SIZE - 1) / GROUP_SIZE;

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
            // dispatch_*()'s barrier already covers workers [1..nwoken_); this is
            // a defensive no-op in well-formed code that closes process-exit
            // races where a worker is still mid-store in worker_done_ or
            // otherwise touching shared state (scratch, task_ctx_, shards_)
            // when the Meyers singleton dtor runs.
            unsigned last_epoch = dispatch_epoch_.load(std::memory_order_acquire);
            if (last_epoch > 0) {
                for (int t = 1; t < nwoken_; ++t) {
                    while (worker_done_[t - 1].epoch.load(std::memory_order_acquire) < last_epoch)
                        POOL_SPIN_PAUSE();
                }
            }
            exit_.store(true, std::memory_order_release);
            for (int g = 0; g < ngroups_; ++g) {
                groups_[g].gen.store(UINT_MAX, std::memory_order_release);
#ifdef _WIN32
                WakeByAddressAll((void*)&groups_[g].gen);
#elif defined(__linux__)
                syscall(SYS_futex, &groups_[g].gen, FUTEX_WAKE_PRIVATE, INT_MAX,
                    nullptr, nullptr, 0);
#elif defined(__APPLE__)
                __ulock_wake(UL_COMPARE_AND_WAIT | ULF_WAKE_ALL, (void*)&groups_[g].gen, 0);
#endif
            }
            threads_.clear();
        }
        for (int i = 0; i < nthreads_; ++i)
            nnr_aligned_free(scratch_[i]);
    }

    thread_pool_t(const thread_pool_t&) = delete;
    thread_pool_t& operator=(const thread_pool_t&) = delete;

    enum class wait_mode { monitorx, wait_on_address };

    template <wait_mode Mode>
    void worker_main_impl(int tid) {
#ifdef _WIN32
        SetThreadAffinityMask(GetCurrentThread(), 1ULL << tid_to_processor(tid));
#endif
        int grp = tid / GROUP_SIZE;
        unsigned my_gen = 0;
        startup_count_.fetch_add(1, std::memory_order_release);
        for (;;) {
            int spin_budget = sleeping_.load(std::memory_order_acquire) ? 0 : SPIN_MAX;
            bool caught = false;
            for (int s = 0; s < spin_budget; ++s) {
                unsigned g = groups_[grp].gen.load(std::memory_order_acquire);
                if (g != my_gen) { my_gen = g; caught = true; break; }
                POOL_SPIN_PAUSE();
            }

            if (!caught) {
                for (;;) {
                    unsigned g = groups_[grp].gen.load(std::memory_order_acquire);
                    if (g != my_gen) { my_gen = g; break; }
                    if constexpr (Mode == wait_mode::monitorx) {
#ifdef NNR_ARCH_X64
                        _mm_monitorx((void*)&groups_[grp].gen, 0, 0);
                        g = groups_[grp].gen.load(std::memory_order_acquire);
                        if (g != my_gen) { my_gen = g; break; }
                        _mm_mwaitx(0, 0, 0);
                        g = groups_[grp].gen.load(std::memory_order_acquire);
                        if (g != my_gen) { my_gen = g; break; }
                        if (sleeping_.load(std::memory_order_acquire)) {
#ifdef _WIN32
                            unsigned cmp = my_gen;
                            WaitOnAddress((volatile void*)&groups_[grp].gen,
                                          &cmp, sizeof(unsigned), INFINITE);
#elif defined(__linux__)
                            syscall(SYS_futex, &groups_[grp].gen, FUTEX_WAIT_PRIVATE,
                                my_gen, nullptr, nullptr, 0);
#endif
                        }
#endif
                    } else {
#ifdef _WIN32
                        unsigned cmp = my_gen;
                        WaitOnAddress((volatile void*)&groups_[grp].gen,
                                      &cmp, sizeof(unsigned), INFINITE);
#elif defined(__linux__)
                        syscall(SYS_futex, &groups_[grp].gen, FUTEX_WAIT_PRIVATE,
                            my_gen, nullptr, nullptr, 0);
#elif defined(__APPLE__)
                        __ulock_wait(UL_COMPARE_AND_WAIT, (void*)&groups_[grp].gen,
                            my_gen, 0);
#else
                        POOL_SPIN_PAUSE();
#endif
                    }
                }
            }

            if (exit_.load(std::memory_order_acquire)) return;

            if (tid < task_nactive_) {
                work_fn_(this, tid);
            }
            // Unified barrier via per-worker epoch (matches v1). Each worker
            // writes its own cache line — zero contention. Main reads O(nwoken)
            // cache lines. The acquire load on dispatch_epoch_ pairs with main's
            // acq_rel fetch_add that ran before the gen increment that woke us.
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

    // ── Static work kernel (v1-equivalent fast path) ─────────────────────────
    // Each worker processes its own pre-sliced range [tid*chunk, ...). Zero
    // shared-atomic touches. Matches v1's work_impl<Fn, false>.
    template <typename Fn>
    static void work_impl_static(thread_pool_t* pool, int tid) {
        Fn& fn = *static_cast<Fn*>(pool->task_ctx_);
        int nactive = pool->task_nactive_;
        int range = pool->task_end_ - pool->task_begin_;
        int chunk = (range + nactive - 1) / nactive;
        int lo = pool->task_begin_ + tid * chunk;
        int hi = std::min(lo + chunk, pool->task_end_);
        for (int i = lo; i < hi; ++i)
            fn(tid, i);
    }

    // ── Dynamic work kernel (sharded LoopCounter + stealing) ─────────────────
    // Each worker claims iters from home shard (tid % nshards), then round-
    // robin-steals from siblings when home drains.
    template <typename Fn>
    static void work_impl_sharded(thread_pool_t* pool, int tid) {
        Fn& fn = *static_cast<Fn*>(pool->task_ctx_);
        int ns = pool->task_nshards_;
        int home = tid % ns;

        // Drain home. First-seeing-empty sets `drained` so stealers skip it.
        {
            shard_t& s = pool->shards_[home];
            int blk = s.block;
            for (;;) {
                int i = s.next.fetch_add(blk, std::memory_order_relaxed);
                if (i >= s.end) {
                    s.drained.store(true, std::memory_order_release);
                    break;
                }
                int hi = std::min(i + blk, s.end);
                for (int k = i; k < hi; ++k) fn(tid, k);
            }
        }
        // Steal round-robin. Skip shards already flagged drained — cheap
        // acquire load on a shard-local cache line instead of a contended
        // fetch_add that returns nothing.
        for (int off = 1; off < ns; ++off) {
            int sid = (home + off) % ns;
            shard_t& s = pool->shards_[sid];
            if (s.drained.load(std::memory_order_acquire)) continue;
            int blk = s.block;
            for (;;) {
                int i = s.next.fetch_add(blk, std::memory_order_relaxed);
                if (i >= s.end) {
                    s.drained.store(true, std::memory_order_release);
                    break;
                }
                int hi = std::min(i + blk, s.end);
                for (int k = i; k < hi; ++k) fn(tid, k);
            }
        }
    }

    // ── Static dispatch (v1-equivalent) ──────────────────────────────────────
    template <typename Fn>
    void dispatch_static(Fn& fn, int begin, int end, int max_threads) {
        int range = end - begin;
        int thread_limit = std::min(nthreads_, max_threads);
        int nactive = std::min(thread_limit, range);
        int groups_needed = (nactive + GROUP_SIZE - 1) / GROUP_SIZE;
        int nwoken = std::min(nthreads_, groups_needed * GROUP_SIZE);

        task_ctx_     = &fn;
        task_begin_   = begin;
        task_end_     = end;
        task_nactive_ = nactive;
        nwoken_       = nwoken;
        work_fn_      = &work_impl_static<Fn>;
        // acq_rel so workers observe dispatch_epoch_ via the subsequent gen
        // release increment that wakes them.
        unsigned epoch = dispatch_epoch_.fetch_add(1, std::memory_order_acq_rel) + 1;

        for (int g = 0; g < groups_needed; ++g)
            groups_[g].gen.fetch_add(1, std::memory_order_release);
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

        // Main's share (tid=0).
        work_fn_(this, 0);

        // Per-worker epoch scan — v1-equivalent O(nwoken) reads, zero contention.
        for (int t = 1; t < nwoken; ++t) {
            while (worker_done_[t - 1].epoch.load(std::memory_order_acquire) < epoch)
                POOL_SPIN_PAUSE();
        }
    }

    // ── Dynamic dispatch (sharded LoopCounter + stealing) ────────────────────
    // For ranges ≤ FALLBACK_THRESHOLD the shard setup + stealing overhead
    // dominates the load-balancing benefit (measured +100–200% vs v1 on
    // range≤48). The static path is strictly faster in that regime because
    // (a) zero shared-atomic touches on the fast path, and (b) even v1's
    // own dynamic path pays 12-way fetch_add contention per iter, which
    // for tiny ranges exceeds the imbalance cost.
    template <typename Fn>
    void dispatch_dynamic(Fn& fn, int begin, int end, int max_threads) {
        int range = end - begin;
        constexpr int FALLBACK_THRESHOLD = 48;
        if (range <= FALLBACK_THRESHOLD) {
            // dispatch_static's work_impl_static calls fn(tid, i) — same
            // signature as the dynamic caller's fn, so forwarding is direct.
            dispatch_static(fn, begin, end, max_threads);
            return;
        }

        int thread_limit = std::min(nthreads_, max_threads);
        int nactive = std::min(thread_limit, range);
        int groups_needed = (nactive + GROUP_SIZE - 1) / GROUP_SIZE;
        int nwoken = std::min(nthreads_, groups_needed * GROUP_SIZE);

        int nshards = std::min(nactive, MAX_SHARDS);
        if (nshards < 1) nshards = 1;
        int shard_size = (range + nshards - 1) / nshards;

        // Dynamic block size: claim `shard_size / DYN_CHUNKS_PER_SHARD` items
        // per fetch_add, amortizing atomic contention. For range=140, nshards=8,
        // shard_size=18, block=4 → ~4 fetch_adds per worker instead of 17.
        // K=2 was too aggressive (broke load balance at range=648, +30%).
        constexpr int DYN_CHUNKS_PER_SHARD = 4;
        int block = std::max(1, shard_size / DYN_CHUNKS_PER_SHARD);

        int cur = begin;
        for (int s = 0; s < nshards; ++s) {
            int lo = cur;
            int hi = std::min(lo + shard_size, end);
            shards_[s].next.store(lo, std::memory_order_relaxed);
            shards_[s].end   = hi;
            shards_[s].block = block;
            shards_[s].drained.store(false, std::memory_order_relaxed);
            cur = hi;
        }

        task_ctx_     = &fn;
        task_begin_   = begin;
        task_end_     = end;
        task_nactive_ = nactive;
        task_nshards_ = nshards;
        nwoken_       = nwoken;
        work_fn_      = &work_impl_sharded<Fn>;
        // acq_rel so workers observe dispatch_epoch_ via the subsequent gen
        // release increment that wakes them.
        unsigned epoch = dispatch_epoch_.fetch_add(1, std::memory_order_acq_rel) + 1;

        for (int g = 0; g < groups_needed; ++g)
            groups_[g].gen.fetch_add(1, std::memory_order_release);
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

        work_fn_(this, 0);

        // Per-worker epoch scan. Workers finish in indeterminate order under
        // stealing, but each still writes its own epoch cache line exactly once.
        for (int t = 1; t < nwoken; ++t) {
            while (worker_done_[t - 1].epoch.load(std::memory_order_acquire) < epoch)
                POOL_SPIN_PAUSE();
        }
    }
};

} // namespace nnr
