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
    if (!p) throw std::bad_alloc();
    Block* b = static_cast<Block*>(p);
    b->next = head_;
    head_ = b;
    used_ = 0;
}

} // namespace nnr

#undef NNR_VMALLOC
#undef NNR_VMFREE
