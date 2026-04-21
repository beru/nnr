#pragma once

#include <vector>
#include <cstddef>
#include <unordered_set>
#include <string_view>

namespace nnr {

struct tensor_t;
struct context_t;

struct memory_planner_t {
    void analyze(context_t* ctx);
    void plan();
    void apply();
    void zero_pool();
    void release();
    ~memory_planner_t();

    int num_intermediates = 0;
    int num_slots = 0;
    int num_inplace = 0;
    size_t total_pool_bytes = 0;
    size_t total_unpooled_bytes = 0;

private:
    struct tensor_lifetime {
        tensor_t* tensor;
        int producer;
        int last_consumer;
        size_t size_bytes;
        int slot_id = -1;
        bool inplace = false;
        int inplace_parent = -1;
        // Concat alias: this tensor is a sub-region of a Concat output.
        // The producer writes directly into the Concat output buffer.
        int concat_parent = -1;      // index of Concat output's lifetime entry
        size_t concat_offset = 0;    // byte offset within parent's slot
    };

    struct buffer_slot {
        size_t size = 0;
        int free_after = -1;
    };

    context_t* ctx_ = nullptr;
    std::vector<tensor_lifetime> lifetimes_;
    std::vector<buffer_slot> slots_;
    std::vector<void*> pool_;
    // Cleared on apply() and set after the first zero_pool() call. Lets run()
    // skip the per-inference memset of the whole activation pool once the
    // underlying buffers have been zeroed at least once (F-PHP-001). Ops
    // write their outputs before reading them, so leaving stale bytes in the
    // pool across runs is safe.
    bool pool_zeroed_ = false;
public:
    bool is_pool_zeroed() const { return pool_zeroed_; }
    void mark_pool_zeroed() { pool_zeroed_ = true; }
};

} // namespace nnr
