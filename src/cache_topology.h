#pragma once
// Runtime CPU cache topology detection.
//
// Exposes L1d, L2, L3-per-domain sizes (KB) and the number of L3 domains
// (CCDs on AMD Zen, one per socket on Intel monolithic, dies on Apple Silicon).
// Populated once at startup via the OS topology API — Windows
// GetLogicalProcessorInformationEx(RelationCache) or Linux
// /sys/devices/system/cpu/cpu0/cache/. We intentionally avoid CPUID leaves
// 4 / 0x8000001D: they don't report the shared-domain grouping needed for
// workspace budgets, and they behave inconsistently across vendors/VMs.
//
// Detection failure yields all-zero sizes. Callers MUST provide a safe
// fallback (see cpu_features_t::l{1d,2,l3}_kb accessors). A one-time stderr
// warning is emitted when detection returns zeros.

#include <cstdint>
#include <cstdio>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <vector>
#elif defined(__linux__)
#include <cstdio>
#include <cstring>
#include <cstdlib>
#endif

namespace nnr {

struct cache_topology_t {
    uint32_t l1d_kb           = 0;  // per-core L1 data cache (KB)
    uint32_t l2_kb            = 0;  // per-core or per-cluster L2 (KB)
    uint32_t l3_kb_per_domain = 0;  // shared L3 within one domain (KB)
    uint32_t l3_domains       = 1;  // number of L3 domains on the package
    uint32_t cache_line       = 64; // line size in bytes
};

namespace detail {

// Emit the "cache detection failed" warning at most once.
inline void cache_topology_warn_once() {
    static bool warned = false;
    if (!warned) {
        warned = true;
        std::fprintf(stderr,
            "[nnr] cache topology detection failed; using conservative defaults "
            "(L1d=32KB, L2=512KB, L3=8MB, 1 domain)\n");
    }
}

inline cache_topology_t cache_topology_fallback() {
    cache_topology_warn_once();
    cache_topology_t t;
    t.l1d_kb           = 32;
    t.l2_kb            = 512;
    t.l3_kb_per_domain = 8 * 1024;
    t.l3_domains       = 1;
    t.cache_line       = 64;
    return t;
}

} // namespace detail

inline cache_topology_t detect_cache_topology() {
    cache_topology_t t;

#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationCache, nullptr, &len);
    if (len == 0) return detail::cache_topology_fallback();
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationCache,
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf.data(), &len))
        return detail::cache_topology_fallback();

    uint64_t seen_l3_masks[64] = {0};
    int n_l3 = 0;

    DWORD offset = 0;
    while (offset < len) {
        auto* info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf.data() + offset);
        if (info->Relationship == RelationCache) {
            const auto& c = info->Cache;
            uint32_t size_kb = (uint32_t)(c.CacheSize / 1024);
            if (c.LineSize > 0) t.cache_line = c.LineSize;
            if (c.Level == 1 && (c.Type == CacheData || c.Type == CacheUnified)) {
                if (t.l1d_kb == 0) t.l1d_kb = size_kb;
            } else if (c.Level == 2) {
                if (t.l2_kb == 0) t.l2_kb = size_kb;
            } else if (c.Level == 3) {
                if (t.l3_kb_per_domain == 0) t.l3_kb_per_domain = size_kb;
                uint64_t mask = (uint64_t)c.GroupMask.Mask;
                bool dup = false;
                for (int i = 0; i < n_l3; ++i)
                    if (seen_l3_masks[i] == mask) { dup = true; break; }
                if (!dup && n_l3 < (int)(sizeof(seen_l3_masks)/sizeof(seen_l3_masks[0])))
                    seen_l3_masks[n_l3++] = mask;
            }
        }
        offset += info->Size;
    }

    if (n_l3 > 0) t.l3_domains = (uint32_t)n_l3;

#elif defined(__linux__)
    auto read_size_kb = [](const char* path) -> uint32_t {
        FILE* f = std::fopen(path, "r");
        if (!f) return 0;
        char buf[64] = {0};
        size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
        std::fclose(f);
        if (n == 0) return 0;
        unsigned long val = std::strtoul(buf, nullptr, 10);
        // Size is like "32K" / "1024K" / "32M" / "32768". Detect suffix.
        char suffix = '\0';
        for (size_t i = 0; i < n; ++i) {
            char ch = buf[i];
            if (ch == 'K' || ch == 'M' || ch == 'G') { suffix = ch; break; }
        }
        if (suffix == 'M') val *= 1024;
        else if (suffix == 'G') val *= 1024 * 1024;
        else if (suffix == '\0') val /= 1024;  // raw bytes → KB
        return (uint32_t)val;
    };
    auto read_int = [](const char* path) -> int {
        FILE* f = std::fopen(path, "r");
        if (!f) return 0;
        int val = 0;
        if (std::fscanf(f, "%d", &val) != 1) val = 0;
        std::fclose(f);
        return val;
    };
    auto read_line = [](const char* path, char* out, size_t cap) -> bool {
        FILE* f = std::fopen(path, "r");
        if (!f) return false;
        size_t n = std::fread(out, 1, cap - 1, f);
        std::fclose(f);
        out[n] = '\0';
        return n > 0;
    };

    // Iterate index0..index7; stop at first gap.
    char path[256];
    char line[128];
    int  l3_level_idx = -1;
    for (int i = 0; i < 8; ++i) {
        std::snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu0/cache/index%d/level", i);
        int level = read_int(path);
        if (level <= 0) break;

        std::snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu0/cache/index%d/type", i);
        if (!read_line(path, line, sizeof(line))) continue;
        bool is_data = (std::strncmp(line, "Data", 4) == 0
                     || std::strncmp(line, "Unified", 7) == 0);
        bool is_unified = (std::strncmp(line, "Unified", 7) == 0);

        std::snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu0/cache/index%d/size", i);
        uint32_t size_kb = read_size_kb(path);

        std::snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu0/cache/index%d/coherency_line_size", i);
        int ls = read_int(path);
        if (ls > 0) t.cache_line = (uint32_t)ls;

        if (level == 1 && is_data) {
            if (t.l1d_kb == 0) t.l1d_kb = size_kb;
        } else if (level == 2 && (is_data || is_unified)) {
            if (t.l2_kb == 0) t.l2_kb = size_kb;
        } else if (level == 3) {
            if (t.l3_kb_per_domain == 0) {
                t.l3_kb_per_domain = size_kb;
                l3_level_idx = i;
            }
        }
    }

    // Count distinct L3 domains by comparing shared_cpu_list across all CPUs.
    // Approximation: iterate cpu0..cpu127 reading cache/index<N>/shared_cpu_list.
    if (l3_level_idx >= 0) {
        char seen[64][64] = {{0}};
        int n_l3 = 0;
        for (int cpu = 0; cpu < 256; ++cpu) {
            std::snprintf(path, sizeof(path),
                "/sys/devices/system/cpu/cpu%d/cache/index%d/shared_cpu_list",
                cpu, l3_level_idx);
            if (!read_line(path, line, sizeof(line))) break;
            // strip trailing newline
            size_t ll = std::strlen(line);
            while (ll && (line[ll-1] == '\n' || line[ll-1] == ' ')) { line[--ll] = 0; }
            bool dup = false;
            for (int j = 0; j < n_l3; ++j)
                if (std::strcmp(seen[j], line) == 0) { dup = true; break; }
            if (!dup && n_l3 < 64) {
                std::strncpy(seen[n_l3], line, sizeof(seen[0]) - 1);
                ++n_l3;
            }
        }
        if (n_l3 > 0) t.l3_domains = (uint32_t)n_l3;
    }
#endif

    if (t.l1d_kb == 0 || t.l2_kb == 0 || t.l3_kb_per_domain == 0)
        return detail::cache_topology_fallback();
    return t;
}

} // namespace nnr
