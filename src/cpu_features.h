#pragma once
// Runtime CPU feature detection.
// x64: Detects ISA extensions (AVX-512, AVX2, SSE4, etc.) via CPUID.
// ARM: NEON is mandatory on AArch64.
// Caches the results for fast repeated queries.

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define NNR_ARCH_X64 1
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#define NNR_ARCH_ARM64 1
#endif

#include <cstring>
#include <utility>
#include <cstdint>

#include "cache_topology.h"

#if defined(NNR_ARCH_ARM64) && defined(_WIN32)
// Forward-declare to avoid pulling in the full <windows.h>. The real symbol lives in kernel32
// and is exported as an import entry, hence __declspec(dllimport) to match the SDK header.
extern "C" __declspec(dllimport) int __stdcall IsProcessorFeaturePresent(unsigned long ProcessorFeature);
#  ifndef PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE
#    define PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE 43
#  endif
#endif

#if defined(NNR_ARCH_ARM64) && defined(__linux__)
#  include <sys/auxv.h>
#  include <asm/hwcap.h>
// Older kernel headers may be missing newer HWCAP2 bits; define defensively.
#  ifndef HWCAP_ASIMDHP
#    define HWCAP_ASIMDHP  (1 << 10)
#  endif
#  ifndef HWCAP_ASIMDDP
#    define HWCAP_ASIMDDP  (1 << 20)
#  endif
#  ifndef HWCAP2_I8MM
#    define HWCAP2_I8MM    (1 << 13)
#  endif
#  ifndef HWCAP2_BF16
#    define HWCAP2_BF16    (1 << 14)
#  endif
#endif

namespace nnr {

struct cpu_features_t {
    // Vendor (x64)
    bool is_intel = false;
    bool is_amd = false;

    // SSE family (x64)
    bool sse3 = false;
    bool ssse3 = false;
    bool sse41 = false;
    bool sse42 = false;

    // AVX family (x64)
    bool avx = false;
    bool avx2 = false;
    bool fma = false;
    bool f16c = false;

    // AVX-512 (x64)
    bool avx512f = false;
    bool avx512bw = false;
    bool avx512vl = false;
    bool avx512dq = false;
    bool avx512vnni = false;
    bool avx512_bf16 = false;  // VDPBF16PS (Zen4+, Cooper Lake+)

    // Misc (x64)
    bool popcnt = false;
    bool bmi1 = false;
    bool bmi2 = false;
    bool monitorx = false;  // AMD MONITORX/MWAITX

    // ARM
    bool neon    = false;   // NEON SIMD (mandatory on AArch64)
    bool fp16    = false;   // ARMv8.2-A FP16 arithmetic (vfmaq_f16 etc.)
    bool dotprod = false;   // ARMv8.4-A SDOT/UDOT (int8 dot-product)
    bool i8mm    = false;   // ARMv8.6-A SMMLA/UMMLA/USMMLA (int8 matrix-multiply)
    bool bf16    = false;   // ARMv8.6-A BFDOT/BFMMLA (bf16 dot-product)

    // Cache topology (sizes in KB; 0 means "unknown"). Callers that derive
    // workspace/tile budgets MUST substitute a safe fallback when these are
    // zero — detect_cache_topology() in cache_topology.h already populates
    // conservative defaults on detection failure and emits a one-time
    // stderr warning, so under normal use these fields are always nonzero.
    uint32_t l1d_kb           = 0;  // per-core L1 data
    uint32_t l2_kb            = 0;  // per-core (or per-cluster) L2
    uint32_t l3_kb_per_domain = 0;  // shared L3 within one domain (CCD / socket)
    uint32_t l3_domains       = 1;  // number of L3 domains on the package
    uint32_t cache_line       = 64; // line size in bytes
};

inline const cpu_features_t& cpu_features() {
    static const cpu_features_t f = []() {
        cpu_features_t f;

#ifdef NNR_ARCH_X64
        int info[4]{};

        auto cpuid = [&](int leaf, int sub = 0) {
#ifdef _MSC_VER
            __cpuidex(info, leaf, sub);
#else
            __cpuid_count(leaf, sub, info[0], info[1], info[2], info[3]);
#endif
        };

        cpuid(0);
        int max_leaf = info[0];

        // Vendor string
        char vendor[13]{};
        std::memcpy(vendor, &info[1], 4);
        std::memcpy(vendor + 4, &info[3], 4);
        std::memcpy(vendor + 8, &info[2], 4);
        f.is_intel = !std::strcmp(vendor, "GenuineIntel");
        f.is_amd   = !std::strcmp(vendor, "AuthenticAMD");

        // Leaf 1: basic features (ECX, EDX)
        if (max_leaf >= 1) {
            cpuid(1);
            auto ecx = info[2];
            f.sse3   = (ecx >> 0) & 1;
            f.ssse3  = (ecx >> 9) & 1;
            f.fma    = (ecx >> 12) & 1;
            f.sse41  = (ecx >> 19) & 1;
            f.sse42  = (ecx >> 20) & 1;
            f.popcnt = (ecx >> 23) & 1;
            f.avx    = (ecx >> 28) & 1;
            f.f16c   = (ecx >> 29) & 1;
        }

        // Leaf 7: extended features (EBX, ECX)
        if (max_leaf >= 7) {
            cpuid(7, 0);
            auto ebx = info[1];
            auto ecx = info[2];
            f.bmi1       = (ebx >> 3) & 1;
            f.avx2       = (ebx >> 5) & 1;
            f.bmi2       = (ebx >> 8) & 1;
            f.avx512f    = (ebx >> 16) & 1;
            f.avx512dq   = (ebx >> 17) & 1;
            f.avx512bw   = (ebx >> 30) & 1;
            f.avx512vl   = (ebx >> 31) & 1;
            f.avx512vnni = (ecx >> 11) & 1;
            // Subleaf 1: EAX
            cpuid(7, 1);
            f.avx512_bf16 = (info[0] >> 5) & 1;
        }

        // Extended leaf 0x80000001: AMD features
        cpuid(0x80000000);
        if (info[0] >= (int)0x80000001) {
            cpuid(0x80000001);
            f.monitorx = (info[2] >> 29) & 1;
        }

#elifdef NNR_ARCH_ARM64
        // NEON is mandatory on AArch64 — always present.
        f.neon = true;
  #ifdef _WIN32
        // Windows-on-ARM exposes dotprod via IsProcessorFeaturePresent(43).
        f.dotprod = IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) != 0;
        // i8mm/bf16/fp16 have no universal Windows probe as of 2026-04; detect via SEH illegal-instruction
        // probe (see arm_feature_probe.h). For MVP, conservatively assume these track dotprod on
        // Oryon-class chips — Oryon Gen 1 has all of dotprod/i8mm/bf16/fp16. Refine when other ARM
        // Windows chips appear without this grouping.
        if (f.dotprod) {
            f.fp16 = true;
            f.i8mm = true;
            f.bf16 = true;
        }
  #elif defined(__linux__)
        // Linux/Android: per-feature detection via ELF auxiliary vector HWCAP bits.
        unsigned long hwcap  = getauxval(AT_HWCAP);
        unsigned long hwcap2 = getauxval(AT_HWCAP2);
        // Scalar FP16 is FPHP; vector FP16 (the one we care about for vfmaq_f16 etc.) is ASIMDHP.
        f.fp16    = (hwcap  & HWCAP_ASIMDHP) != 0;
        f.dotprod = (hwcap  & HWCAP_ASIMDDP) != 0;
        f.i8mm    = (hwcap2 & HWCAP2_I8MM)   != 0;
        f.bf16    = (hwcap2 & HWCAP2_BF16)   != 0;
  #endif
#endif

        // Cache topology detection (OS API; fallback to conservative defaults).
        cache_topology_t t = detect_cache_topology();
        f.l1d_kb           = t.l1d_kb;
        f.l2_kb            = t.l2_kb;
        f.l3_kb_per_domain = t.l3_kb_per_domain;
        f.l3_domains       = t.l3_domains;
        f.cache_line       = t.cache_line;

        return f;
    }();
    return f;
}

// ISA levels for template dispatch.
// x64: scalar → avx2 → avx512 (runtime CPUID selects best within x64).
// ARM: scalar → neon (NEON is mandatory on AArch64).
// x64 vs ARM selection is compile-time only (#if guards in kernel headers).
// The enum contains all values so unguarded code can reference them without error;
// on each architecture, only the relevant values are returned by detect_isa().
enum class isa_t { scalar, neon, avx2, avx512 };

// Maximum ISA level to use. Set via set_max_isa() to force a lower ISA
// (e.g., avx2 on an avx512-capable machine for testing/benchmarking,
//  or scalar on ARM to bypass NEON).
inline isa_t& max_isa() {
#ifdef NNR_ARCH_ARM64
    static isa_t level = isa_t::neon;
#else
    static isa_t level = isa_t::avx512;
#endif
    return level;
}

inline void set_max_isa(isa_t level) { max_isa() = level; }

// True if all AVX-512 features used by NNR kernels are present
// AND AVX-512 is not capped by set_max_isa().
// Always false on non-x64 (compile-time constant).
inline bool has_avx512() {
#ifdef NNR_ARCH_X64
    auto& f = cpu_features();
    return f.avx512f && f.avx512bw && max_isa() >= isa_t::avx512;
#else
    return false;
#endif
}

inline bool has_avx512_bf16() {
#ifdef NNR_ARCH_X64
    return has_avx512() && cpu_features().avx512_bf16;
#else
    return false;
#endif
}

// True if NEON is not capped by set_max_isa(scalar).
// NEON is always present on AArch64; this only checks the ISA cap.
// Always false on non-ARM (compile-time constant).
inline bool has_neon() {
#ifdef NNR_ARCH_ARM64
    return max_isa() >= isa_t::neon;
#else
    return false;
#endif
}

// ARM ISA extensions beyond base NEON. Runtime-gated via cpu_features().
// Always false on non-ARM (compile-time constant).
inline bool has_neon_fp16() {
#ifdef NNR_ARCH_ARM64
    return has_neon() && cpu_features().fp16;
#else
    return false;
#endif
}

inline bool has_neon_dotprod() {
#ifdef NNR_ARCH_ARM64
    return has_neon() && cpu_features().dotprod;
#else
    return false;
#endif
}

inline bool has_neon_i8mm() {
#ifdef NNR_ARCH_ARM64
    return has_neon() && cpu_features().i8mm;
#else
    return false;
#endif
}

inline bool has_neon_bf16() {
#ifdef NNR_ARCH_ARM64
    return has_neon() && cpu_features().bf16;
#else
    return false;
#endif
}

inline isa_t detect_isa() {
    static const isa_t hw = []() {
#ifdef NNR_ARCH_X64
        auto& f = cpu_features();
        if (f.avx512f && f.avx512bw) return isa_t::avx512;
        if (f.avx2 && f.fma)         return isa_t::avx2;
#elifdef NNR_ARCH_ARM64
        return isa_t::neon;
#endif
        return isa_t::scalar;
    }();
    return hw <= max_isa() ? hw : max_isa();
}

// Dispatch helper: calls Fn<isa_t>::call(args...) based on runtime ISA detection.
// x64: dispatches avx512 → avx2 → scalar.
// ARM: dispatches neon → scalar.
//
// Usage:
//   struct my_gemm {
//       template <isa_t ISA>
//       static void call(int n, int m, ...);
//   };
//   isa_dispatch<my_gemm>(n, m, ...);
//
template <typename Fn, typename... Args>
inline auto isa_dispatch(Args&&... args) {
    switch (detect_isa()) {
#ifdef NNR_ARCH_X64
    case isa_t::avx512: return Fn::template call<isa_t::avx512>(std::forward<Args>(args)...);
    case isa_t::avx2:   return Fn::template call<isa_t::avx2>(std::forward<Args>(args)...);
#elifdef NNR_ARCH_ARM64
    case isa_t::neon:   return Fn::template call<isa_t::neon>(std::forward<Args>(args)...);
#endif
    default:            return Fn::template call<isa_t::scalar>(std::forward<Args>(args)...);
    }
}

} // namespace nnr
