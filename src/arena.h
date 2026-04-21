#pragma once

#include <cstddef>

namespace nnr {

/// Bump-pointer arena allocator.
///
/// Lifetime contract:
///   - A pointer returned by `alloc()` is valid until the NEXT `alloc()` call
///     that triggers growth, OR until `reset()` / destruction — whichever
///     comes first. Callers MUST NOT hold an arena pointer across a
///     subsequent `alloc()` unless they can prove no growth will occur
///     (e.g. sized by `reserve()` ahead of time).
///   - Successfully-returned pointers stay valid across allocs that fit in
///     the current block; growth memcpys existing content into a new block
///     and frees the old one, invalidating every prior pointer.
///
/// The backing block is retained across reset() calls and grows on demand
/// (only during reshape(), never during exec()).
///
/// Intended use: control-flow operators (Loop, Scan, SequenceMap) request
/// arena space in reshape() to pre-size their scratch buffers so that exec()
/// can run allocation-free.
struct arena_t {
    arena_t() = default;
    arena_t(const arena_t&) = delete;
    arena_t& operator=(const arena_t&) = delete;
    ~arena_t();

    /// Allocate `size` bytes aligned to `align` bytes.
    /// Returns nullptr on failure.
    void* alloc(size_t size, size_t align = 8);

    /// Invalidate all allocations and reset to the start of the backing block.
    /// The backing block is kept for reuse. O(1).
    void reset() { used_ = 0; }

    /// Save the current allocation cursor and return it as a checkpoint.
    size_t save() const { return used_; }

    /// Restore the allocation cursor to a previous checkpoint, freeing all
    /// allocations made since save() was called. O(1).
    void restore(size_t checkpoint) { used_ = checkpoint; }

    template <typename T>
    T* alloc_arr(size_t n) { return static_cast<T*>(alloc(n * sizeof(T), alignof(T))); }

    size_t capacity() const { return cap_; }
    size_t used()     const { return used_; }

private:
    char*  buf_  = nullptr;
    size_t cap_  = 0;
    size_t used_ = 0;
};

/// RAII scope guard for arena_t.
///
/// Saves the arena cursor on construction and restores it on destruction,
/// freeing all allocations made within the scope — even on early return.
struct arena_scope_t {
    arena_t& arena;

    explicit arena_scope_t(arena_t& a) : arena(a), mark_(a.save()) {}
    ~arena_scope_t() { arena.restore(mark_); }

    arena_scope_t(const arena_scope_t&) = delete;
    arena_scope_t& operator=(const arena_scope_t&) = delete;

    template <typename T>
    T* alloc_arr(size_t n) { return arena.alloc_arr<T>(n); }

private:
    size_t mark_;
};

} // namespace nnr
