#if defined(NNR_USE_CUDA)

#include "cupti_profiler.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#if defined(NNR_USE_CUPTI)

#include <cupti.h>
#include <cupti_activity.h>

#endif // NNR_USE_CUPTI

namespace nnr::gpu {

#if defined(NNR_USE_CUPTI)

namespace {

// Buffer size for CUPTI activity records. CUPTI fills these and hands them
// back via the buffer_completed callback. 16 KB is the recommended baseline.
constexpr size_t kBufferBytes  = 16 * 1024;
constexpr size_t kBufferAlign  = 8;

// Raw kernel record carrying the CUPTI correlationId — op_idx resolved later.
struct pending_kernel_t {
    uint64_t start_ns;
    uint64_t end_ns;
    uint32_t correlation_id;  // CUPTI internal correlation
    const char* kernel;       // CUPTI-owned, lifetime = subscriber session
};

// All shared state is protected by g_mtx.
// We can't resolve op_idx inside the callback because CUPTI delivers KERNEL
// and EXTERNAL_CORRELATION records via different buffers in unpredictable
// order. Resolution happens in drain() once both kinds have been flushed.
std::mutex g_mtx;
std::vector<pending_kernel_t> g_pending;
std::unordered_map<uint32_t, uint32_t> g_corr_to_op;  // cupti corrId → op_idx

// Canonical kernel sequence: (kernel_name, op_idx) per kernel launch in the
// order they fire during a single inference. Populated during a "good"
// drain where most records resolve (typically the first warmup pass through
// the slow path, where push/pop fires per op and externalCorrelationId is
// captured directly on the launch). Reused in subsequent drains where
// records arrive without external correlation (e.g., CUDA Graph replay) to
// recover op_idx from the kernel name's position.
struct canonical_entry_t {
    const char* kernel;
    uint32_t    op_idx;
};
std::vector<canonical_entry_t> g_canonical_seq;

void* aligned_alloc_buffer(size_t size, size_t align) {
#ifdef _WIN32
    return _aligned_malloc(size, align);
#else
    void* p = nullptr;
    if (posix_memalign(&p, align, size) != 0) return nullptr;
    return p;
#endif
}

void aligned_free_buffer(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

void CUPTIAPI buffer_requested(uint8_t** buffer,
                               size_t* size,
                               size_t* maxNumRecords) {
    *buffer = static_cast<uint8_t*>(aligned_alloc_buffer(kBufferBytes, kBufferAlign));
    *size = kBufferBytes;
    *maxNumRecords = 0;  // 0 = fill the entire buffer
}

void CUPTIAPI buffer_completed(CUcontext /*ctx*/,
                               uint32_t /*streamId*/,
                               uint8_t* buffer,
                               size_t /*size*/,
                               size_t validSize) {
    if (validSize == 0) {
        aligned_free_buffer(buffer);
        return;
    }
    CUpti_Activity* record = nullptr;
    while (true) {
        CUptiResult res = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (res == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
        if (res != CUPTI_SUCCESS) break;
        if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL ||
            record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            const auto* k = reinterpret_cast<const CUpti_ActivityKernel9*>(record);
            pending_kernel_t pk;
            pk.start_ns = k->start;
            pk.end_ns   = k->end;
            pk.correlation_id = k->correlationId;
            pk.kernel   = k->name;
            std::lock_guard<std::mutex> lk(g_mtx);
            g_pending.push_back(pk);
            if (const char* dbg = std::getenv("NNR_CUPTI_DEBUG")) {
                if (dbg[0] >= '2') {
                    fprintf(stderr, "[cupti] KERN corr=%u name=%s\n",
                        k->correlationId, k->name ? k->name : "?");
                }
            }
        } else if (record->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
            const auto* ec = reinterpret_cast<const CUpti_ActivityExternalCorrelation*>(record);
            std::lock_guard<std::mutex> lk(g_mtx);
            g_corr_to_op[ec->correlationId] = static_cast<uint32_t>(ec->externalId);
            if (const char* dbg = std::getenv("NNR_CUPTI_DEBUG")) {
                if (dbg[0] >= '2') {
                    fprintf(stderr, "[cupti] ECORR corr=%u ext=%llu\n",
                        ec->correlationId, (unsigned long long)ec->externalId);
                }
            }
        } else if (record->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
            const auto* rt = reinterpret_cast<const CUpti_ActivityAPI*>(record);
            if (const char* dbg = std::getenv("NNR_CUPTI_DEBUG")) {
                if (dbg[0] >= '2') {
                    fprintf(stderr, "[cupti] RUNT corr=%u cbid=%u\n",
                        rt->correlationId, rt->cbid);
                }
            }
        }
    }
    size_t dropped = 0;
    cuptiActivityGetNumDroppedRecords(nullptr, 0, &dropped);
    if (dropped) {
        fprintf(stderr, "[cupti] dropped %zu activity records\n", dropped);
    }
    aligned_free_buffer(buffer);
}

bool g_initialized = false;

bool do_init() {
    if (g_initialized) return true;
    CUptiResult r = cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
    if (r != CUPTI_SUCCESS) {
        const char* msg = nullptr;
        cuptiGetResultString(r, &msg);
        fprintf(stderr, "[cupti] registerCallbacks failed: %s\n", msg ? msg : "?");
        return false;
    }
    // Enable kernel records (works on all consumer GPUs without elevated perms).
    CUptiResult ck = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (ck != CUPTI_SUCCESS) {
        ck = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    }
    // RUNTIME and DRIVER records are required for EXTERNAL_CORRELATION
    // records to be emitted at API call boundaries. The KERNEL record's
    // correlationId matches the DRIVER call (cuLaunchKernel), not the
    // RUNTIME call (cudaLaunchKernel) — so DRIVER must be enabled for the
    // ECORR record at the driver level to be emitted with the kernel's
    // correlationId.
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER);
    CUptiResult ec = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
    if (const char* dbg = std::getenv("NNR_CUPTI_DEBUG")) {
        if (dbg[0] && dbg[0] != '0') {
            const char *ck_msg = nullptr, *ec_msg = nullptr;
            cuptiGetResultString(ck, &ck_msg);
            cuptiGetResultString(ec, &ec_msg);
            fprintf(stderr, "[cupti] init: kernel-enable=%s ext-corr-enable=%s\n",
                ck_msg ? ck_msg : "?", ec_msg ? ec_msg : "?");
        }
    }
    g_initialized = true;
    return true;
}

void do_shutdown() {
    if (!g_initialized) return;
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
    cuptiActivityFlushAll(0);
    g_initialized = false;
}

} // namespace

cupti_profiler_t::cupti_profiler_t() = default;
cupti_profiler_t::~cupti_profiler_t() { shutdown(); }

bool cupti_profiler_t::init() {
    if (initialized_) return true;
    initialized_ = do_init();
    return initialized_;
}

void cupti_profiler_t::push_op(uint32_t op_idx) {
    if (!initialized_) return;
    cuptiActivityPushExternalCorrelationId(
        CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN,
        static_cast<uint64_t>(op_idx));
}

void cupti_profiler_t::pop_op() {
    if (!initialized_) return;
    uint64_t popped = 0;
    cuptiActivityPopExternalCorrelationId(
        CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &popped);
}

void cupti_profiler_t::flush() {
    if (!initialized_) return;
    cuptiActivityFlushAll(0);
}

std::vector<cupti_record_t> cupti_profiler_t::drain() {
    std::vector<pending_kernel_t> pending;
    std::unordered_map<uint32_t, uint32_t> corr_to_op;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        pending.swap(g_pending);
        corr_to_op.swap(g_corr_to_op);
    }
    std::vector<cupti_record_t> out;
    out.reserve(pending.size());
    for (auto& pk : pending) {
        cupti_record_t r;
        r.start_ns = pk.start_ns;
        r.end_ns   = pk.end_ns;
        r.kernel   = pk.kernel;
        auto it = corr_to_op.find(pk.correlation_id);
        r.op_idx = (it != corr_to_op.end()) ? it->second : UINT32_MAX;
        out.push_back(r);
    }
    // Sort by GPU start time so we can match against the canonical kernel
    // sequence from a prior good drain (kernel launch order in a captured
    // graph is deterministic across replays).
    std::sort(out.begin(), out.end(),
        [](const cupti_record_t& a, const cupti_record_t& b) {
            return a.start_ns < b.start_ns;
        });

    // Decide whether this drain is "good enough" to refresh canonical_seq_.
    // We require at least 80% of records to resolve. The first non-replay
    // run (slow path during warmup) typically hits 100% — that's the canonical.
    int resolved = 0;
    for (auto& r : out) if (r.op_idx != UINT32_MAX) ++resolved;
    bool good_drain = !out.empty() && (resolved * 5 >= (int)out.size() * 4);

    {
        std::lock_guard<std::mutex> lk(g_mtx);
        if (good_drain) {
            // Snapshot kernel-name + op_idx for every resolved record.
            // This becomes the template for matching unresolved replay-mode
            // records by name in their launch order.
            g_canonical_seq.clear();
            g_canonical_seq.reserve(out.size());
            for (auto& r : out) {
                if (r.op_idx == UINT32_MAX || !r.kernel) continue;
                g_canonical_seq.push_back({r.kernel, r.op_idx});
            }
        } else if (!g_canonical_seq.empty()) {
            // Replay-mode style drain: most records have op_idx == UINT32_MAX.
            // Walk records in start order, advance through canonical entries
            // matching by kernel name. A canonical entry may be skipped if
            // the replay doesn't include that kernel (e.g., capture-time-only
            // setup), and a record without a canonical match keeps op_idx=
            // UINT32_MAX (gets dropped by the caller).
            size_t ci = 0;
            for (auto& r : out) {
                if (r.op_idx != UINT32_MAX) continue;
                if (!r.kernel) continue;
                while (ci < g_canonical_seq.size() &&
                       std::strcmp(g_canonical_seq[ci].kernel, r.kernel) != 0) {
                    ++ci;
                }
                if (ci < g_canonical_seq.size()) {
                    r.op_idx = g_canonical_seq[ci].op_idx;
                    ++ci;
                }
            }
        }
    }

    if (const char* dbg = std::getenv("NNR_CUPTI_DEBUG")) {
        if (dbg[0] && dbg[0] != '0') {
            int final_resolved = 0;
            for (auto& r : out) if (r.op_idx != UINT32_MAX) ++final_resolved;
            fprintf(stderr,
                "[cupti] drain: %zu records, %d direct, %d after-canonical "
                "(%zu corr-map, %zu canonical%s)\n",
                out.size(), resolved, final_resolved,
                corr_to_op.size(), g_canonical_seq.size(),
                good_drain ? ", refreshed" : "");
        }
    }
    return out;
}

void cupti_profiler_t::shutdown() {
    if (!initialized_) return;
    do_shutdown();
    initialized_ = false;
}

#else // !NNR_USE_CUPTI — stubs

cupti_profiler_t::cupti_profiler_t() = default;
cupti_profiler_t::~cupti_profiler_t() = default;
bool cupti_profiler_t::init()                     { return false; }
void cupti_profiler_t::push_op(uint32_t)          {}
void cupti_profiler_t::pop_op()                   {}
void cupti_profiler_t::flush()                    {}
std::vector<cupti_record_t> cupti_profiler_t::drain() { return {}; }
void cupti_profiler_t::shutdown()                 {}

#endif // NNR_USE_CUPTI

cupti_profiler_t& cupti_profiler() {
    static cupti_profiler_t inst;
    return inst;
}

} // namespace nnr::gpu

#endif // NNR_USE_CUDA
