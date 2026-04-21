#pragma once

// Portable aligned allocator wrappers.
//
// All 64-byte aligned heap allocations in nnr go through these helpers so
// that the allocator pair matches across every translation unit. The raw
// platform APIs (`_aligned_malloc`/`_aligned_free` on MSVC, `posix_memalign`/
// `free` elsewhere) must never be called directly from the library — a
// mismatched alloc/free pair is undefined behavior.

#include <cstddef>

#ifdef _WIN32
#  include <malloc.h>
#else
#  include <cstdlib>
#endif

namespace nnr {

inline void* nnr_aligned_alloc(size_t size, size_t align = 64)
{
    if (size == 0) return nullptr;
#ifdef _WIN32
    return _aligned_malloc(size, align);
#else
    void* p = nullptr;
    if (posix_memalign(&p, align, size) != 0) return nullptr;
    return p;
#endif
}

inline void nnr_aligned_free(void* p)
{
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

} // namespace nnr
