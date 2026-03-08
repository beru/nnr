#pragma once
// JIT kernel base class — wraps Xbyak::CodeGenerator.
// Only jit_*.cpp files include <xbyak/xbyak.h>; everyone else sees this header.

#include "cpu_features.h"

#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <memory>
#include <functional>
#include <utility>
#include <mutex>

namespace Xbyak { class CodeGenerator; }

namespace nnr {

// Base class for all JIT-compiled kernels.
// Derived classes emit code via gen() in their constructor, then call finalize().
class jit_kernel_t {
public:
    explicit jit_kernel_t(size_t code_size = 4096);
    // @nnr-meta isa=scalar special=JIT
    ~jit_kernel_t();

    jit_kernel_t(const jit_kernel_t&) = delete;
    jit_kernel_t& operator=(const jit_kernel_t&) = delete;
    // @nnr-meta isa=scalar special=JIT
    jit_kernel_t(jit_kernel_t&&) noexcept;
    jit_kernel_t& operator=(jit_kernel_t&&) noexcept;

    // Get typed function pointer. Call after finalize().
    template <typename T>
    // @nnr-meta isa=scalar special=JIT
    T fn() const { return reinterpret_cast<T>(entry_); }

    // @nnr-meta isa=scalar special=JIT
    size_t code_size() const;

protected:
    // Call after emitting all code. Sets memory to R+X, stores entry point.
    // @nnr-meta isa=scalar special=JIT
    void finalize();

    // Access the Xbyak code generator (only usable from .cpp files that include xbyak.h).
    Xbyak::CodeGenerator& gen();
    // @nnr-meta isa=scalar special=JIT
    const Xbyak::CodeGenerator& gen() const;

    // --- ABI prologue/epilogue (call from derived constructor) ---

    // Emit prologue: push callee-saved GPRs + save xmm6-15 on Win64.
    // used_gprs: bitmask of GPR codes (e.g. (1<<3)|(1<<12) for rbx+r12).
    //   Only callee-saved registers are actually pushed; others are silently ignored.
    //   On Win64, rdi/rsi are always saved (they're callee-saved and used as working regs).
    // xmm_save_mask: bitmask of xmm6-xmm15 to save on Win64 (bit 0 = xmm6, bit 9 = xmm15).
    //   Use XMM_SAVE_ALL (0x3FF) to save all 10, 0 for none.
    // Returns total stack adjustment in bytes (for manual offset computation).
    static constexpr uint32_t XMM_SAVE_ALL = 0x3FF;
    // @nnr-meta isa=scalar special=JIT
    int emit_prologue(uint32_t used_gprs, uint32_t xmm_save_mask = XMM_SAVE_ALL);

    // Emit epilogue: restore xmm + pop GPRs (reverse order) + ret.
    // @nnr-meta isa=scalar special=JIT
    void emit_epilogue();

    // Load function argument into a register (64-bit mov).
    // Handles register args (platform-dependent) and stack args automatically.
    // Skips emit if src == dst register.
    // @nnr-meta isa=scalar special=JIT
    void load_arg(int arg_index, int dst_reg);

    // Load function argument with 32→64 sign extension (movsxd).
    // Use for int parameters that need 64-bit register width.
    // @nnr-meta isa=scalar special=JIT
    void load_arg_i32(int arg_index, int dst_reg);

    // Stack offset from current rsp to the Nth function argument.
    // Only valid for stack arguments (arg_index >= num_register_args).
    // Call after emit_prologue().
    // @nnr-meta isa=scalar special=JIT
    int arg_stack_offset(int arg_index) const;

private:
    Xbyak::CodeGenerator* gen_;
    const void* entry_ = nullptr;
    // Prologue state (set by emit_prologue, used by emit_epilogue)
    int prologue_stack_adj_ = 0;
    int prologue_xmm_bytes_ = 0;
    uint32_t prologue_xmm_save_mask_ = 0;
    int prologue_push_count_ = 0;
    int prologue_pushed_[16] = {};
};

// Kernel cache: maps a key to a compiled JIT kernel.
// Thread-safe for reads (kernels are immutable once compiled).
// Not thread-safe for concurrent get_or_create with the same key — caller
// must ensure serialization (reshape() is single-threaded).
template <typename Key, typename Kernel, typename Hash = std::hash<Key>>
class jit_cache_t {
    std::unordered_map<Key, std::unique_ptr<Kernel>, Hash> cache_;
    std::mutex mu_;
public:
    template <typename... Args>
    // @nnr-meta isa=scalar special=JIT
    Kernel* get_or_create(const Key& key, Args&&... args) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = cache_.find(key);
        if (it != cache_.end()) return it->second.get();
        auto k = std::make_unique<Kernel>(std::forward<Args>(args)...);
        auto* p = k.get();
        cache_.emplace(key, std::move(k));
        return p;
    }
    void clear() { std::lock_guard<std::mutex> lock(mu_); cache_.clear(); }
    // @nnr-meta isa=scalar special=JIT
    size_t size() const { return cache_.size(); }
};

// ABI parameter register indices (Xbyak::Operand::Code values).
// Use with Xbyak::Reg64(abi_param::p0) etc.
struct abi_param {
#ifdef _WIN32
    // Win64: rcx=1, rdx=2, r8=8, r9=9
    static constexpr int p0 = 1;   // rcx
    static constexpr int p1 = 2;   // rdx
    static constexpr int p2 = 8;   // r8
    static constexpr int p3 = 9;   // r9
    static constexpr int shadow_space = 32;
#else
    // SysV: rdi=7, rsi=6, rdx=2, rcx=1, r8=8, r9=9
    static constexpr int p0 = 7;   // rdi
    static constexpr int p1 = 6;   // rsi
    static constexpr int p2 = 2;   // rdx
    static constexpr int p3 = 1;   // rcx
    static constexpr int shadow_space = 0;
#endif
};

// Resolve-once JIT dispatch: maps Key → function pointer with zero overhead
// after first call.  Wraps jit_cache_t + std::call_once per slot.
//
// Usage:
//   static jit_dispatch_t<key_t, jit_my_kernel_t, fn_t, hash_t, 4> dispatch;
//   auto fn = dispatch.resolve(key, ctor_arg1, ctor_arg2, ...);
//   fn(runtime_arg1, ...);
//
// MaxEntries must cover all possible key values (e.g. 4 for 2 bools).
template <typename Key, typename Kernel, typename Fn, typename Hash, int MaxEntries>
class jit_dispatch_t {
    jit_cache_t<Key, Kernel, Hash> cache_;
    Fn fns_[MaxEntries] = {};
    std::once_flag flags_[MaxEntries];
public:
    template <typename... Args>
    // @nnr-meta isa=scalar special=JIT
    Fn resolve(const Key& key, Args&&... args) {
        int idx = static_cast<int>(Hash{}(key));
        std::call_once(flags_[idx], [&]() {
            auto* k = cache_.get_or_create(key, std::forward<Args>(args)...);
            fns_[idx] = k->template fn<Fn>();
        });
        return fns_[idx];
    }
    // @nnr-meta isa=scalar special=JIT
    void clear() { cache_.clear(); }
};

// ---------------------------------------------------------------------------
// jit_execute: unified JIT dispatch with eligibility check and fallback.
//
// Usage:
//   jit_execute(
//       jit_nchwc_eligible(KH, KW, strideW, tile),   // eligibility predicate
//       [&] { return resolve_jit_nchwc(KH, KW, strideW, tile); },  // resolve
//       [&](auto fn) { fn(out, in, w, ...); },        // JIT body
//       [&] { conv_nchwc_tile<true>(...); }            // intrinsics fallback
//   );
//
// Centralizes: env var disable, eligibility gate, try/catch safeguard,
// error reporting. The try/catch is a defensive last resort — the
// eligibility predicate should prevent all failures in practice.
// ---------------------------------------------------------------------------
template <typename ResolveFn, typename JitBody, typename Fallback>
// @nnr-meta isa=scalar special=JIT
inline void jit_execute(bool eligible, ResolveFn&& resolve,
                        JitBody&& jit_body, Fallback&& fallback)
{
    static const bool disabled = (std::getenv("NNR_DISABLE_JIT") != nullptr);
    if (!disabled && eligible) {
        try {
            auto fn = resolve();
            jit_body(fn);
            return;
        } catch (const std::exception& e) {
            // Predicate was wrong — log so we can fix it
            fprintf(stderr, "[jit] unexpected failure (predicate bug): %s\n", e.what());
        }
    }
    fallback();
}

} // namespace nnr

#endif // NNR_ARCH_X64 && NNR_USE_XBYAK
