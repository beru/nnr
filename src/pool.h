#pragma once

#include <cstddef>
#include <cassert>
#include <new>

namespace nnr {

// Stable bump-pointer pool allocator.
//
// Memory is allocated from 64 KB OS pages via VirtualAlloc/mmap.
// New blocks are prepended to a singly-linked list; existing allocations
// are never moved, so all returned pointers remain valid for the lifetime
// of the pool. No individual deallocation — all memory is freed at once
// when the pool is destroyed.
struct pool_t {
    static constexpr size_t BlockSize = 64 * 1024;  // 64 KB

    pool_t();
    pool_t(const pool_t&) = delete;
    pool_t& operator=(const pool_t&) = delete;
    ~pool_t();

    void* alloc(size_t size, size_t align = 8)
    {
        if (size == 0) return nullptr;
        // Alignment and oversize checks must be hard runtime guards, not
        // release-build-elided asserts: both failure modes return a pointer
        // that a caller writes through, stomping the allocator's own state.
        if (align > BlockSize) return nullptr;
        constexpr size_t kMaxPayload = BlockSize - sizeof(Block);
        if (size > kMaxPayload) return nullptr;

        char* body = reinterpret_cast<char*>(head_) + sizeof(Block);
        size_t off = (used_ + align - 1) & ~(align - 1);

        if (off <= kMaxPayload && off + size <= kMaxPayload) {
            used_ = off + size;
            return body + off;
        }

        // Current block is full — allocate a new one.
        alloc_block();
        // Fresh block starts at offset 0, which is already `alignof(Block)`-
        // aligned. Round up to the requested alignment before placing the
        // first allocation so callers with `align > alignof(Block)` stay
        // honored (the previous code skipped this round-up).
        size_t start = (align > alignof(Block))
                     ? ((alignof(Block) + align - 1) & ~(align - 1)) - alignof(Block)
                     : 0;
        if (start + size > kMaxPayload) return nullptr;
        used_ = start + size;
        return reinterpret_cast<char*>(head_) + sizeof(Block) + start;
    }

    template <typename T>
    T* alloc_arr(size_t n)
    {
        if (n == 0) return nullptr;
        return static_cast<T*>(alloc(n * sizeof(T), alignof(T)));
    }

private:
    struct Block { Block* next; };

    void alloc_block();  // implemented in pool.cpp (uses VirtualAlloc/mmap)

    Block* head_ = nullptr;
    size_t used_ = 0;  // bytes used in the body of head_ (after the Block header)
};

template <typename T, typename... Args>
T* pool_new(pool_t& pool, Args&&... args)
{
    return new (pool.alloc(sizeof(T), alignof(T))) T(std::forward<Args>(args)...);
}

} // namespace nnr
