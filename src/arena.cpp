#include "arena.h"
#include "aligned_alloc.h"

#include <cstring>

namespace nnr {

void* arena_t::alloc(size_t size, size_t align)
{
    if (size == 0) return nullptr;

    // Round current offset up to requested alignment.
    size_t offset = (used_ + align - 1) & ~(align - 1);
    size_t end    = offset + size;

    if (end > cap_) {
        // Grow: at least double, or exactly enough — whichever is larger.
        size_t new_cap = cap_ ? cap_ * 2 : 4096;
        while (new_cap < end) new_cap *= 2;

        char* new_buf = (char*)nnr_aligned_alloc(new_cap);
        if (!new_buf) return nullptr;

        if (buf_) {
            memcpy(new_buf, buf_, used_);
            nnr_aligned_free(buf_);
        }
        buf_ = new_buf;
        cap_ = new_cap;
    }

    used_ = end;
    return buf_ + offset;
}

arena_t::~arena_t()
{
    nnr_aligned_free(buf_);
}

} // namespace nnr
