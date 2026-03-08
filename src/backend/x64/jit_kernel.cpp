#include "jit_kernel.h"

#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)

#include <xbyak/xbyak.h>
#include <cassert>
#include <utility>

namespace nnr {

jit_kernel_t::jit_kernel_t(size_t code_size)
    : gen_(new Xbyak::CodeGenerator(
          static_cast<int>(code_size), Xbyak::DontSetProtectRWE)) {}

jit_kernel_t::~jit_kernel_t() { delete gen_; }

jit_kernel_t::jit_kernel_t(jit_kernel_t&& o) noexcept
    : gen_(std::exchange(o.gen_, nullptr))
    , entry_(std::exchange(o.entry_, nullptr)) {}

jit_kernel_t& jit_kernel_t::operator=(jit_kernel_t&& o) noexcept {
    if (this != &o) {
        delete gen_;
        gen_ = std::exchange(o.gen_, nullptr);
        entry_ = std::exchange(o.entry_, nullptr);
    }
    return *this;
}

void jit_kernel_t::finalize() {
    assert(gen_ && "finalize() called on moved-from kernel");
    assert(!gen_->hasUndefinedLabel() && "JIT kernel has undefined labels");
    gen_->setProtectModeRE();
    entry_ = gen_->getCode<const void*>();
}

Xbyak::CodeGenerator& jit_kernel_t::gen() {
    assert(gen_);
    return *gen_;
}

const Xbyak::CodeGenerator& jit_kernel_t::gen() const {
    assert(gen_);
    return *gen_;
}

size_t jit_kernel_t::code_size() const {
    return gen_ ? gen_->getSize() : 0;
}

// ---------------------------------------------------------------------------
// ABI prologue/epilogue
// ---------------------------------------------------------------------------

int jit_kernel_t::emit_prologue(uint32_t used_gprs, uint32_t xmm_save_mask) {
    using namespace Xbyak;
    auto& c = gen();
    prologue_push_count_ = 0;
    prologue_xmm_bytes_ = 0;
    prologue_xmm_save_mask_ = 0;

    // Callee-saved register masks per platform.
    // SysV:  rbx(3), rbp(5), r12-r15(12-15)
    // Win64: same + rdi(7), rsi(6)
    constexpr uint32_t cs_sysv = (1u<<3)|(1u<<5)|(1u<<12)|(1u<<13)|(1u<<14)|(1u<<15);
#ifdef _WIN32
    constexpr uint32_t cs_mask = cs_sysv | (1u<<6) | (1u<<7);
    // Always save rdi/rsi on Win64 — kernels use them as working registers.
    used_gprs |= (1u<<6) | (1u<<7);
#else
    constexpr uint32_t cs_mask = cs_sysv;
#endif
    used_gprs &= cs_mask;

    // Push in ascending register code order (epilogue pops in reverse).
    for (int r = 0; r < 16; r++) {
        if (used_gprs & (1u << r)) {
            c.push(Reg64(r));
            prologue_pushed_[prologue_push_count_++] = r;
        }
    }

#ifdef _WIN32
    if (xmm_save_mask) {
        // Win64: xmm6-xmm15 are callee-saved. Mask bit i → xmm(6+i).
        prologue_xmm_save_mask_ = xmm_save_mask & 0x3FF;
        int count = 0;
        for (int i = 0; i < 10; i++)
            if (prologue_xmm_save_mask_ & (1u << i)) count++;
        prologue_xmm_bytes_ = count * 16;
        c.sub(c.rsp, prologue_xmm_bytes_);
        int slot = 0;
        for (int i = 0; i < 10; i++)
            if (prologue_xmm_save_mask_ & (1u << i))
                c.vmovups(c.ptr[c.rsp + (slot++) * 16], Xmm(6 + i));
    }
#else
    (void)xmm_save_mask;
#endif

    prologue_stack_adj_ = prologue_push_count_ * 8 + prologue_xmm_bytes_;
    return prologue_stack_adj_;
}

void jit_kernel_t::emit_epilogue() {
    using namespace Xbyak;
    auto& c = gen();

#ifdef _WIN32
    if (prologue_xmm_bytes_) {
        int slot = 0;
        for (int i = 0; i < 10; i++)
            if (prologue_xmm_save_mask_ & (1u << i))
                c.vmovups(Xmm(6 + i), c.ptr[c.rsp + (slot++) * 16]);
        c.add(c.rsp, prologue_xmm_bytes_);
    }
#endif

    // Pop in reverse push order.
    for (int i = prologue_push_count_ - 1; i >= 0; i--)
        c.pop(Reg64(prologue_pushed_[i]));

    c.ret();
}

// Arg register tables.
#ifdef _WIN32
static constexpr int arg_regs[] = {1, 2, 8, 9};  // rcx, rdx, r8, r9
static constexpr int num_reg_args = 4;
static constexpr int shadow = 32;
#else
static constexpr int arg_regs[] = {7, 6, 2, 1, 8, 9};  // rdi, rsi, rdx, rcx, r8, r9
static constexpr int num_reg_args = 6;
static constexpr int shadow = 0;
#endif

void jit_kernel_t::load_arg(int arg_index, int dst_reg) {
    using namespace Xbyak;
    auto& c = gen();
    if (arg_index < num_reg_args) {
        int src = arg_regs[arg_index];
        if (src != dst_reg)
            c.mov(Reg64(dst_reg), Reg64(src));
    } else {
        int off = prologue_stack_adj_ + 8 + shadow + (arg_index - num_reg_args) * 8;
        c.mov(Reg64(dst_reg), c.ptr[c.rsp + off]);
    }
}

void jit_kernel_t::load_arg_i32(int arg_index, int dst_reg) {
    using namespace Xbyak;
    auto& c = gen();
    if (arg_index < num_reg_args) {
        c.movsxd(Reg64(dst_reg), Reg32(arg_regs[arg_index]));
    } else {
        int off = prologue_stack_adj_ + 8 + shadow + (arg_index - num_reg_args) * 8;
        c.movsxd(Reg64(dst_reg), c.dword[c.rsp + off]);
    }
}

int jit_kernel_t::arg_stack_offset(int arg_index) const {
    return prologue_stack_adj_ + 8 + shadow + (arg_index - num_reg_args) * 8;
}

} // namespace nnr

#endif // NNR_ARCH_X64 && NNR_USE_XBYAK
