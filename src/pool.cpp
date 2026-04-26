#include "pool.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <Windows.h>
#  define NNR_VMALLOC(sz)   VirtualAlloc(nullptr, (sz), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
#  define NNR_VMFREE(p)     VirtualFree((p), 0, MEM_RELEASE)
#else
#  include <sys/mman.h>
#  define NNR_VMALLOC(sz)   mmap(nullptr, (sz), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)
#  define NNR_VMFREE(p)     munmap((p), pool_t::BlockSize)
#endif

namespace nnr {

pool_t::pool_t()
{
    alloc_block();
}

pool_t::~pool_t()
{
    Block* b = head_;
    while (b) {
        Block* next = b->next;
        NNR_VMFREE(b);
        b = next;
    }
}

void pool_t::alloc_block()
{
    void* p = NNR_VMALLOC(BlockSize);
#ifndef _WIN32
    // mmap returns MAP_FAILED (((void*)-1) on all known kernels) on failure,
    // not nullptr. Without this branch, an OOM pool would silently
    // dereference an invalid pointer on the next alloc.
    if (p == MAP_FAILED) p = nullptr;
#endif
    if (!p) throw std::bad_alloc();
    Block* b = static_cast<Block*>(p);
    b->next = head_;
    head_ = b;
    used_ = 0;
}

} // namespace nnr

#undef NNR_VMALLOC
#undef NNR_VMFREE
