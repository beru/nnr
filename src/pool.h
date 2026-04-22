#pragma once

#include <cstddef>
#include <cassert>
#include <new>
#include <utility>

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
        assert(align <= BlockSize);

        char* body = reinterpret_cast<char*>(head_) + sizeof(Block);
        size_t off = (used_ + align - 1) & ~(align - 1);

        if (off + size <= BlockSize - sizeof(Block)) {
            used_ = off + size;
            return body + off;
        }

        // Current block is full — allocate a new one.
        alloc_block();
        assert(size <= BlockSize - sizeof(Block));
        used_ = size;
        return reinterpret_cast<char*>(head_) + sizeof(Block);
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
