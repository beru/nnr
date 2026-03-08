#pragma once

#include "arena.h"
#include "pool.h"

namespace nnr {

// ---------------------------------------------------------------------------
// arena_allocator<T>
//
// STL-compatible allocator backed by arena_t. Allocates from the arena;
// deallocate() is a no-op. Use arena_t::save()/restore() to reclaim all
// memory allocated through this allocator since the checkpoint.
//
// Intended use: short-lived exec-time containers (Loop/Scan scratch vectors)
// that are freed in bulk at the end of the exec call.
// ---------------------------------------------------------------------------
template <typename T>
struct arena_allocator {
    using value_type = T;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;

    arena_t* arena;

    explicit arena_allocator(arena_t& a) noexcept : arena(&a) {}

    template <typename U>
    arena_allocator(const arena_allocator<U>& other) noexcept : arena(other.arena) {}

    T* allocate(size_t n)
    {
        return static_cast<T*>(arena->alloc(n * sizeof(T), alignof(T)));
    }

    void deallocate(T*, size_t) noexcept {}  // arena frees in bulk via restore()

    template <typename U>
    bool operator==(const arena_allocator<U>& o) const noexcept { return arena == o.arena; }
    template <typename U>
    bool operator!=(const arena_allocator<U>& o) const noexcept { return arena != o.arena; }
};

// ---------------------------------------------------------------------------
// pool_allocator<T>
//
// STL-compatible allocator backed by pool_t. Allocates from the stable pool;
// deallocate() is a no-op. Memory lives for the lifetime of the pool.
//
// Intended use: permanent containers whose contents must remain stable
// (e.g., operator inputs/outputs/attrs that are set once at graph build).
// ---------------------------------------------------------------------------
template <typename T>
struct pool_allocator {
    using value_type = T;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;

    pool_t* pool;

    explicit pool_allocator(pool_t& p) noexcept : pool(&p) {}

    template <typename U>
    pool_allocator(const pool_allocator<U>& other) noexcept : pool(other.pool) {}

    T* allocate(size_t n)
    {
        return static_cast<T*>(pool->alloc(n * sizeof(T), alignof(T)));
    }

    void deallocate(T*, size_t) noexcept {}  // pool frees in bulk at destruction

    template <typename U>
    bool operator==(const pool_allocator<U>& o) const noexcept { return pool == o.pool; }
    template <typename U>
    bool operator!=(const pool_allocator<U>& o) const noexcept { return pool != o.pool; }
};

// Convenience aliases
template <typename T> using arena_vector = std::vector<T, arena_allocator<T>>;
template <typename T> using pool_vector  = std::vector<T, pool_allocator<T>>;

} // namespace nnr
