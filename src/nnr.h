#pragma once

#include "nnrconf.h"
#include "arena.h"
#include "pool.h"
#include "memory_planner.h"
#include "graph_optimizer.h"
#include "attr_key.h"

#define NNR_MAJOR       (1)
#define NNR_MINOR       (0)
#define NNR_PATCH       (0)
#define NNR_VERSION     ((NNR_MAJOR * 10000) + (NNR_MINOR * 100) + NNR_PATCH)

namespace nnr {

struct tensor_t;
struct graph_t;
struct context_t;

enum data_type_t : uint8_t {
    NNR_DATA_TYPE_UNDEFINED     = 0,
    // floating point
    NNR_DATA_TYPE_FLOAT32,
    NNR_DATA_TYPE_FLOAT64,
    NNR_DATA_TYPE_FLOAT16,
    NNR_DATA_TYPE_BFLOAT16,
    // signed integer
    NNR_DATA_TYPE_INT8,
    NNR_DATA_TYPE_INT16,
    NNR_DATA_TYPE_INT32,
    NNR_DATA_TYPE_INT64,
    // unsigned integer
    NNR_DATA_TYPE_UINT8,
    NNR_DATA_TYPE_UINT16,
    NNR_DATA_TYPE_UINT32,
    NNR_DATA_TYPE_UINT64,
    // other
    NNR_DATA_TYPE_BOOL,
    NNR_DATA_TYPE_STRING,
    NNR_DATA_TYPE_COMPLEX64,
    NNR_DATA_TYPE_COMPLEX128,
    // fp8
    NNR_DATA_TYPE_FLOAT8E4M3FN,
    NNR_DATA_TYPE_FLOAT8E4M3FNUZ,
    NNR_DATA_TYPE_FLOAT8E5M2,
    NNR_DATA_TYPE_FLOAT8E5M2FNUZ,
    NNR_DATA_TYPE_FLOAT8E8M0,
    // sub-byte
    NNR_DATA_TYPE_UINT4,
    NNR_DATA_TYPE_INT4,
    NNR_DATA_TYPE_FLOAT4E2M1,
    NNR_DATA_TYPE_UINT2,
    NNR_DATA_TYPE_INT2,
    // internal
    NNR_DATA_TYPE_SEQUENCE,     // data -> sequence_t*
    NNR_DATA_TYPE_COUNT,
};

// Memory layout format
enum class memory_layout_t : uint8_t {
    NCHW = 0,       // Standard C-contiguous, channels-first (ONNX default)
    NHWC,           // Channels-last (TFLite, ARM backends)
    BLOCKED_8,      // nChw8c — used by ARM NEON native blocked layout
    BLOCKED_16,     // nChw16c — used by x64 AVX-512 native blocked layout
};

// Layout capability bitmask — which memory layouts an operator supports.
static constexpr uint8_t LAYOUT_NCHW        = 1 << (int)memory_layout_t::NCHW;
static constexpr uint8_t LAYOUT_NHWC        = 1 << (int)memory_layout_t::NHWC;
static constexpr uint8_t LAYOUT_BLOCKED_8   = 1 << (int)memory_layout_t::BLOCKED_8;
static constexpr uint8_t LAYOUT_BLOCKED_16  = 1 << (int)memory_layout_t::BLOCKED_16;
static constexpr uint8_t LAYOUT_ALL         = 0xFF;

// Native NCHWc layout for the current build target.
// x64: BLOCKED_16 / 16-lane (AVX-512 ZMM width)
// ARM64: BLOCKED_8 / 8-lane (2 NEON qregs pair)
// Everything else: NCHW / 0 (blocked layout disabled)
//
// Callers that produce or consume blocked tensors must use these constants
// rather than literal BLOCKED_16/16 so the same code path handles both
// architectures. The two enum values coexist in the enum but never coexist
// at runtime in a single build.
//
// We test compiler-intrinsic arch macros directly (NOT NNR_ARCH_X64/ARM64),
// because cpu_features.h — which defines those NNR_ARCH_* macros — may not
// have been included at the point where nnr.h is first pulled in. Using
// the raw __x86_64__/_M_X64/__aarch64__/_M_ARM64 tests keeps these
// constants correctly defined from the very first include of nnr.h.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
static constexpr memory_layout_t NATIVE_BLOCKED_FMT = memory_layout_t::BLOCKED_16;
static constexpr uint8_t         LAYOUT_NATIVE_BLOCKED = LAYOUT_BLOCKED_16;
static constexpr int             NATIVE_BLOCK = 16;
#elif defined(__aarch64__) || defined(_M_ARM64)
static constexpr memory_layout_t NATIVE_BLOCKED_FMT = memory_layout_t::BLOCKED_8;
static constexpr uint8_t         LAYOUT_NATIVE_BLOCKED = LAYOUT_BLOCKED_8;
static constexpr int             NATIVE_BLOCK = 8;
#else
static constexpr memory_layout_t NATIVE_BLOCKED_FMT = memory_layout_t::NCHW;
static constexpr uint8_t         LAYOUT_NATIVE_BLOCKED = 0;
static constexpr int             NATIVE_BLOCK = 0;
#endif

// NCHWc block size for a given layout format.
inline constexpr int nchwc_block_size(memory_layout_t fmt) {
    return fmt == memory_layout_t::BLOCKED_16 ? 16
         : fmt == memory_layout_t::BLOCKED_8  ? 8
         : 0;
}

// Padded channel count for NCHWc layout.
inline constexpr int nchwc_padded_channels(int C, int block) {
    return (C + block - 1) / block * block;
}

// Structured cost report for one execution candidate.
struct op_cost_t {
    memory_layout_t layout = memory_layout_t::NCHW;

    // Memory traffic
    float read_bytes = 0;
    float write_bytes = 0;
    float read_sequential = 1.0f;   // 0..1: fraction of sequential reads
    float write_sequential = 1.0f;  // 0..1: fraction of sequential writes

    // Compute
    float compute_ops = 0;          // FLOPs (FMA = 2)

    // Working set
    float working_set_bytes = 0;    // peak cache-resident data

    // Parallelism
    int min_threads = 1;
    int max_threads = 1;

    // Capabilities
    bool scrollable = false;
    bool fusable_post_op = false;
    bool fusable_binary = false;
};

std::string_view data_type_tostring(data_type_t type);
size_t data_type_sizeof(data_type_t type);
size_t data_type_sizeof(const tensor_t* tensor);
bool tensor_equal(const tensor_t* a, const tensor_t* b);

// Sequence type: tensor_t with type==NNR_DATA_TYPE_SEQUENCE, data->sequence_t
struct sequence_t {
    data_type_t elem_type = NNR_DATA_TYPE_UNDEFINED;
    std::vector<tensor_t*> tensors; // owned
    ~sequence_t();
};

struct tensor_t {
    tensor_t() = default;
    tensor_t(std::string_view name, data_type_t type, std::span<const int> dims);
    ~tensor_t();

    // `tensor_t` owns `data` when `owns_data == true`. A default copy would
    // silently alias the buffer and then double-free on destruction, so the
    // copy operations are deleted. Move operations are also disabled because
    // several callers hold tensor_t* references that must not be invalidated
    // by a sudden address change; explicit helpers like `reinit`/`apply` are
    // the supported way to transfer data.
    tensor_t(const tensor_t&) = delete;
    tensor_t& operator=(const tensor_t&) = delete;
    tensor_t(tensor_t&&) = delete;
    tensor_t& operator=(tensor_t&&) = delete;

    static tensor_t* alloc_from_file(std::string_view filename);

    // Resizes the tensor. Returns false when the allocation request overflows
    // SIZE_MAX or when operator new fails; in both cases `data == nullptr` and
    // `allocation_failed` is set. Callers that reach for `apply()` afterwards
    // will silently no-op, so propagate the return value rather than assuming
    // success.
    [[nodiscard]] bool reinit(data_type_t type, std::span<const int> dims);
    void apply(const void* buf, size_t len);
    [[nodiscard]] bool apply(const tensor_t& t);

    void dump(bool detail) const;

    std::span<const int> dim_span() const { return {dims, static_cast<size_t>(ndim)}; }
    std::span<const int64_t> stride_span() const { return {strides, static_cast<size_t>(ndim)}; }

    int64_t indices_to_offset(std::span<const int> indices) const
    {
        int64_t offset = 0;
        for (int i = 0; i < ndim; ++i) {
            offset += static_cast<int64_t>(indices[i]) * strides[i];
        }
        return offset;
    }

    void offset_to_indices(int64_t offset, std::span<int> indices) const
    {
        for (int i = ndim - 1; i >= 0; i--) {
            indices[i] = static_cast<int>(offset % dims[i]);
            offset /= dims[i];
        }
    }

    // Return false on allocation failure propagated from reinit(); callers
    // should abort their reshape() handler in that case.
    bool reshape(std::span<const int> dims, data_type_t type);

    bool reshape_identity(const tensor_t* x, data_type_t type);

    bool reshape_identity(const tensor_t* x)
    {
        return reshape_identity(x, x->type);
    }

    bool reshape_multi_broadcast(const tensor_t* a, const tensor_t* b, data_type_t type);

    bool is_scalar() const
    {
        return ((ndim == 0) && (ndata == 1));
    }

    bool broadcast_is_valid(std::span<const int> dims) const
    {
        const int ndim2 = static_cast<int>(dims.size());
        if (this->ndim > ndim2) {
            return false;
        }
        for (int i = 1; i <= this->ndim; ++i) {
            if ((this->dims[this->ndim - i] != 1) && (this->dims[this->ndim - i] != dims[ndim2 - i])) {
                return false;
            }
        }
        return true;
    }

    void* broadcast_map_address(const tensor_t* y, int64_t offset);

    const void* broadcast_map_address(const tensor_t* y, int64_t offset) const
    {
        return (const void*) const_cast<tensor_t*>(this)->broadcast_map_address(y, offset);
    }

    std::string_view name;
    data_type_t type = NNR_DATA_TYPE_UNDEFINED;
    memory_layout_t format = memory_layout_t::NCHW;
    int64_t strides[MAX_NDIM] = {};
    int dims[MAX_NDIM] = {};
    int ndim = 0;
    void* data = nullptr;
    bool owns_data = true;
    // Set by reinit() (and therefore by the sizing ctor) when an allocation
    // failed or the size calculation overflowed. Cleared on a successful
    // reinit. Use this to detect ctor-path failures — new (std::nothrow)
    // only catches the outer tensor_t allocation, not the inner data buffer.
    bool allocation_failed = false;
    size_t ndata = 0;

    // Quantization metadata (populated by QDQ fusion or loaders).
    // quant_scale == 0 means "not quantized" — all existing code paths unaffected.
    float quant_scale = 0.0f;
    int32_t quant_zero_point = 0;
    int quant_axis = -1;                  // -1 = per-tensor, >= 0 = per-channel axis
    float* quant_scales = nullptr;        // per-channel scales (when axis >= 0)
    int32_t* quant_zero_points = nullptr; // per-channel zero-points (when axis >= 0)

    bool is_quantized() const { return quant_scale != 0.0f || quant_scales != nullptr; }

    void set_quant(float scale, int32_t zp) {
        quant_scale = scale;
        quant_zero_point = zp;
        quant_axis = -1;
        quant_scales = nullptr;
        quant_zero_points = nullptr;
    }

    // Defined in nnr.cpp: frees the per-channel scales/zero-points using
    // the portable aligned-allocation wrappers. Kept out of the public
    // header so downstream users never need the MSVC-only `_aligned_free`.
    void clear_quant();

    void propagate_quant(const tensor_t* src) {
        quant_scale = src->quant_scale;
        quant_zero_point = src->quant_zero_point;
        quant_axis = src->quant_axis;
        quant_scales = src->quant_scales;       // shared, not copied
        quant_zero_points = src->quant_zero_points;
    }
};

void copy_data(tensor_t* y, const tensor_t* x);

inline sequence_t* tensor_get_sequence(tensor_t* t) {
    return (t && t->type == NNR_DATA_TYPE_SEQUENCE) ? (sequence_t*)t->data : nullptr;
}
inline const sequence_t* tensor_get_sequence(const tensor_t* t) {
    return (t && t->type == NNR_DATA_TYPE_SEQUENCE) ? (const sequence_t*)t->data : nullptr;
}

bool tensor_load_sequence_from_file(tensor_t* t, std::string_view filename);
tensor_t* tensor_alloc_optional_from_file(std::string_view filename);
bool tensor_sequence_equal_file(const tensor_t* t, std::string_view filename);
bool tensor_sequence_equal(const tensor_t* a, const tensor_t* b);

// Format-agnostic operator attribute.
// Non-owning: all variable-length data lives in context_t::attr_pool.
// Owned resources (tensor, subgraph) are tracked by context_t.
struct attr_t {
    enum class kind_t : uint8_t {
        INT = 0, FLOAT, STRING, INTS, FLOATS, TENSOR, GRAPH, STRINGS,
    };

    kind_t kind = kind_t::INT;
    int64_t i = 0;
    float f = 0.0f;
    std::string_view s;
    // Spans into the model proto buffer (ints/floats) or attr_pool (strings/subgraph names)
    std::span<const int64_t> ints;
    std::span<const float>   floats;
    std::span<const std::string_view> strings;
    // Non-owning pointers (owned by context_t::attr_tensors_ / attr_subgraphs_)
    tensor_t* tensor   = nullptr;
    graph_t*  subgraph = nullptr;
    std::span<const std::string_view> subgraph_inputs;
    std::span<const std::string_view> subgraph_outputs;
    // Opaque format-specific data (non-owning; points into model proto lifetime)
    void* raw = nullptr;
};

// Scrolling tiling: per-operator descriptor for strip-based execution.
// Operators that support row-strip decomposition override scroll_info()
// to declare their halo requirements. The scheduler uses this to compute
// compound halos and circular buffer sizes for pipelined execution.
struct scroll_info_t {
    bool scrollable = false;   // false for global ops, reshapes, etc.
    bool needs_pre_pass = false; // run scroll_pre_exec() before strip loop (e.g., InstanceNorm stats)
    int halo_top = 0;          // extra input rows needed above output strip
    int halo_bottom = 0;       // extra input rows needed below output strip
    int stride_h = 1;          // output-to-input row ratio (stride)
};

// Ring buffer state for scroll execution. Set by exec_scroll_segment on each
// op before calling exec_strip, cleared after the segment completes.
// When active (ring_H > 0), the tensor's data pointer is replaced with a small
// ring buffer and exec_strip uses ring_H as the per-channel row stride while
// still using orig_H for boundary/padding checks.
struct ring_buf_info_t {
    int ring_H = 0;       // ring buffer height (0 = inactive, use normal tensor)
    int base_row = 0;     // first logical row stored at physical position 0
    int orig_H = 0;       // original tensor height (for padding/boundary checks)
};

struct operator_t {
    operator_t() = default;
    virtual ~operator_t() = default;
    // Polymorphic and owns nothing copyable; delete copy/move so a subclass
    // slice can't sneak in via a pass-by-value. Matches graph_t/context_t/
    // tensor_t, which all delete these for the same reason (stable addresses,
    // span/pool-backed members). Placement-new into the attr_pool is still
    // fine — it goes through the default ctor, not a copy.
    operator_t(const operator_t&) = delete;
    operator_t& operator=(const operator_t&) = delete;
    operator_t(operator_t&&) = delete;
    operator_t& operator=(operator_t&&) = delete;
    // Set by registry_t::solve() to the backend this op was resolved on.
    // context_t::run() uses this to sync cross-backend tensors (e.g.,
    // download GPU-resident inputs before a CPU op). Stored as uint8_t so
    // nnr.h doesn't need to include registry.h (circular).
    uint8_t resolved_backend = 0;  // backend_t::CPU
    void dump(bool detail) const;

    // Attribute accessors — format-agnostic, work on pre-parsed attr store
    attr_t* find_attr(attr_key_t key);
    attr_t* find_attr(std::string_view name) { return find_attr(attr_key_from_string(name)); }
    float attribute(attr_key_t key, float def);
    float attribute(std::string_view name, float def) { return attribute(attr_key_from_string(name), def); }
    int32_t attribute(attr_key_t key, int32_t def);
    int32_t attribute(std::string_view name, int32_t def) { return attribute(attr_key_from_string(name), def); }
    int64_t attribute(attr_key_t key, int64_t def);
    int64_t attribute(std::string_view name, int64_t def) { return attribute(attr_key_from_string(name), def); }
    std::string_view attribute(attr_key_t key, std::string_view def);
    std::string_view attribute(std::string_view name, std::string_view def) { return attribute(attr_key_from_string(name), def); }
    std::string_view attribute(attr_key_t key, const char* def) { return attribute(key, std::string_view(def)); }
    std::string_view attribute(std::string_view name, const char* def) { return attribute(attr_key_from_string(name), std::string_view(def)); }
    int attribute(attr_key_t key, int64_t*& ints);
    int attribute(std::string_view name, int64_t*& ints) { return attribute(attr_key_from_string(name), ints); }
    int attribute(attr_key_t key, float*& floats);
    int attribute(std::string_view name, float*& floats) { return attribute(attr_key_from_string(name), floats); }
    int attribute(attr_key_t key, tensor_t* t);
    int attribute(std::string_view name, tensor_t* t) { return attribute(attr_key_from_string(name), t); }
    graph_t* attribute_subgraph(std::string_view name);
    // Returns opaque raw attribute data (for format-specific operators like Loop/Scan)
    void* attribute_raw(std::string_view name);

    context_t* ctx = nullptr;            // non-owning — set by onnx_loader at graph build, valid during run()
    int opset = 0;
    std::string_view op_type;
    std::string_view node_name;
    std::string_view domain;
    // Spans into context_t::attr_pool — set once at graph build, never reallocated.
    std::span<tensor_t*> inputs;         // non-owning — tensors owned by context_t::map
    std::span<tensor_t*> outputs;        // non-owning — tensors owned by context_t::map
    std::span<std::pair<attr_key_t, attr_t>> attrs;

    virtual bool init() { return true; }
    virtual bool reshape()
    {
        if (inputs.empty() || outputs.empty()) {
            return false;
        }
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        if (x && y) {
            return y->reshape_identity(x);
        }else {
            return false;
        }
    }
    virtual bool exec() = 0;
    virtual int64_t num_ops() const { return 0; }
    virtual size_t workspace_size() const { return 0; }

    // Cost model: operators return structured cost reports for all execution
    // candidates they support. Each candidate describes a (layout, strategy)
    // combination with memory traffic, compute, working set, parallelism,
    // and capability flags. The optimizer filters by capabilities and picks
    // the cheapest candidate.
    // Empty return = layout-neutral (Relu, Add, etc.): zero cost in all layouts.
    virtual small_vector<op_cost_t, 8> estimate_costs(bool input_nhwc) const { return {}; }

    // Scalar cost for layout comparison. Used by assign_layouts() and
    // assign_blocked_layouts(). Default: picks cheapest estimate_costs()
    // candidate for the layout and reduces to scalar. Operators with
    // proven hand-tuned formulas (Conv) override this directly.
    virtual float layout_cost(memory_layout_t layout, bool input_nhwc) const;

    // View ops: output shares input's memory (no copy needed).
    // Returns the input index whose data the output aliases, or -1 if not a view.
    // The memory planner skips allocation for view outputs and extends
    // the source tensor's lifetime instead.
    virtual int view_input_index() const { return -1; }

    // Scrolling tiling support
    virtual scroll_info_t scroll_info() const { return {}; }
    // Compute output rows [out_row_start, out_row_start + out_rows) given
    // input data starting at in_row_start with in_rows valid rows.
    // Default returns false (not implemented). Operators that support
    // scrolling override this to work on sub-regions of their tensors.
    virtual bool exec_strip(int out_row_start, int out_rows,
        int in_row_start, int in_rows) { return false; }
    // Pre-pass for scroll execution: called before the strip loop on the full
    // input tensor. Used by ops that need global statistics (e.g., InstanceNorm
    // computes per-channel mean/variance here).
    virtual bool scroll_pre_exec() { return false; }

    // Ring buffer state (set temporarily by exec_scroll_segment)
    ring_buf_info_t ring_in;   // input tensor ring buffer info
    ring_buf_info_t ring_out;  // output tensor ring buffer info

    bool skip = false;
    bool folded = false;      // set by constant folding — exec() skipped, output data retained
    bool accelerated = false; // set by oneDNN ops when prim_valid — outputs may be in blocked format
    bool is_fused_silu = false; // Sigmoid fused with following Mul into SiLU (graph optimizer)
    uint8_t layout_mask = LAYOUT_NCHW;  // bitmask of supported memory layouts (default: NCHW only)

    // Fused post-op (set by fusion pass in graph_optimizer.cpp)
    // Any unary/binary operator that sets fusable_apply in init() can be fused
    // into a preceding Conv/Gemm/MatMul. The function is called on L1-hot
    // tile data inside the GEMM/depthwise kernel.
    // Multi-row signature: processes rows×cols block with stride between rows.
    // bias: per-row bias pointer (bias[r] for row r), or nullptr for no bias.
    // offset: linear index of data[0] within full output tensor (used by binary
    //         ops like Add to index their second operand).
    using post_fn_t = void (*)(float* data, int rows, int cols, int stride,
                               const operator_t* fused_op, const float* bias, int offset);
    post_fn_t  fusable_apply = nullptr;  // set by unary ops in init() — declares this op can be fused
    post_fn_t  post_fn  = nullptr;       // set by fusion pass on the producer — calls consumer's apply
    operator_t* fused_op = nullptr;      // non-owning — points to fused consumer in graph (for param access)
    const tensor_t* fused_tensor = nullptr; // non-owning — skip connection tensor (for binary Add fusion)

    bool is_inout_size(size_t in_size, size_t out_size) const
    {
        return (inputs.size() == in_size) && (outputs.size() == out_size);
    }

    template <typename T, typename FuncT>
    void foreach_tensor(FuncT func)
    {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;

        for (size_t i = 0, l = y->ndata; i < l; ++i) {
            py[i] = (T)func(px[i]);
        }
    }
};

struct graph_t {
    graph_t() = default;
    graph_t(const graph_t&) = delete;
    graph_t& operator=(const graph_t&) = delete;
    ~graph_t() {
        for (auto* n : nodes) n->~operator_t();
        // Memory is owned by context_t::attr_pool — freed when context_t is destroyed.
    }

    void dump(bool detail) const;

    std::vector<operator_t*> nodes;
};

struct node_profile_t {
    int64_t total_ticks = 0;  // raw QPC ticks (converted to ns at dump time)
    uint32_t calls = 0;
};

struct context_t {
    context_t() = default;
    context_t(const context_t&) = delete;
    context_t& operator=(const context_t&) = delete;
    ~context_t();

    // Load a model — auto-detects format by file extension or magic bytes
    bool load(const void* buf, size_t len);
    bool load_from_file(std::string_view filename);

    void dump(bool detail) const;
    bool run();
    // Run all nodes once without profiling or post-run optimization.
    // Used by graph_optimizer_t::prune_segments() to warm up tensor data
    // before timing trials when prepare() used fold_run (no full exec).
    bool run_for_warmup();
    // Eagerly run reshape + optimization + memory planning so that
    // subsequent run() calls execute the optimized path from the start.
    // Call once after load() and enable_*() configuration.
    bool prepare();
    tensor_t* search_tensor(std::string_view name);

    // Call after reinit()'ing graph input tensors to new shapes.
    // Marks shapes dirty and forces the next run()/prepare() to
    // re-reshape all intermediate tensors.
    void invalidate_shapes();

    // Thread count: default = physical cores (hardware_concurrency / smt_stride).
    // Set before prepare()/run(). Use all logical threads (e.g., 24 on 12-core
    // SMT CPU) only for int8 GEMM workloads that benefit from latency hiding.
    //
    // PROCESS-GLOBAL: despite being a member of context_t, this mutates the
    // singleton thread pool's static `configured_threads_`. Call once at
    // program start, before any context_t::prepare()/run() on any thread.
    // Concurrent calls race. Prefer the free function nnr::set_global_thread_count().
    [[deprecated("use nnr::set_global_thread_count(); this is process-global, not per-context")]]
    static void set_num_threads(int n);

    // Operator fusion (Conv+BN, Conv+Activation, etc.)
    void enable_fusion(bool enable = true) { optimizer->enable_fusion(enable); }

    // Profiling
    void enable_profiling(bool enable = true);
    void reset_profile();
    void dump_profile(FILE* f) const;

    // Memory planning
    void enable_memory_planning(bool enable = true) {
        memory_planning_enabled = enable;
        if (!enable && memory_planned) {
            planner.release();
            memory_planned = false;
        }
    }
    void dump_memory_stats(FILE* f) const;

    // Thread pool sleep/wake: put workers into deep sleep (zero CPU when idle).
    // Workers auto-wake on next run(). Call sleep() after load/prepare when no
    // inference is expected for a while.
    void pool_sleep();
    void pool_wake();

    // Scrolling tiling (strip-based pipelined execution)
    // Default AUTO: detects and enables only when beneficial.
    // enable_scrolling(true) forces ON, enable_scrolling(false) forces OFF.
    void enable_scrolling(bool enable = true) { optimizer->enable_scrolling(enable); }
    void set_scroll_mode(scroll_mode_t mode) { optimizer->set_scroll_mode(mode); }

    // Layout controls
    void set_debug_layout(bool enable = true) { optimizer->debug_layout = enable; }
    void set_no_blocked(bool enable = true) { optimizer->no_blocked = enable; }
    void set_no_nhwc(bool enable = true) { optimizer->no_nhwc = enable; }
    void set_force_nchwc(bool enable = true) { optimizer->force_nchwc = enable; }

    // Public state
    std::vector<std::pair<std::string_view, tensor_t*>> map;
    pool_t  attr_pool;  // arena for op inputs/outputs/attrs spans and inline attr data.
                        // Declared before graph so it outlives graph_t (operators are placement-new'd here).
    std::unique_ptr<graph_t> graph;
    std::vector<node_profile_t> profile;
    void* workspace = nullptr;
    size_t workspace_size = 0;
    bool shapes_dirty = true;
    bool profiling_enabled = false;
    bool memory_planned = false;
    bool memory_planning_enabled = false;
    std::unique_ptr<graph_optimizer_t> optimizer = make_pass_graph_optimizer();
    memory_planner_t planner;
    arena_t arena;      // scratch memory for control-flow ops (Loop, Scan, SequenceMap)

    // Owned attr resources: attr_t fields (tensor, subgraph) are non-owning pointers;
    // these vectors hold the real ownership. Freed in ~context_t().
    std::vector<tensor_t*> attr_tensors_;     // tensors created for Constant/initializer attrs
    std::vector<graph_t*>  attr_subgraphs_;   // subgraphs for control-flow ops (If, Loop, Scan)

    // Format-agnostic metadata (populated by format loaders)
    std::string_view meta_producer_name;
    std::string_view meta_producer_version;
    std::string_view meta_domain;
    int64_t meta_ir_version = 0;
    std::vector<std::pair<std::string_view, int64_t>> meta_opsets;

    // Set by format loaders: graph input/output names in declaration order
    std::vector<std::string_view> graph_inputs;
    std::vector<std::string_view> graph_outputs;
    // Resolved tensor_t* for every entry in graph_outputs — populated on the
    // cold path (first `run()` / `prepare()`) and reused on every subsequent
    // run so `run_graph_impl` does not linear-scan `map` per output per
    // inference. Invalidated (cleared) only when the map itself changes.
    std::vector<tensor_t*> graph_output_tensors;
    // Set by format loaders: tensor names that must not be pooled by memory planner
    std::unordered_set<std::string_view> memory_plan_excluded;
    // Set by format loaders: names of weight/initializer tensors (for graph optimizations)
    std::unordered_set<std::string_view> initializer_names;

    // Preferred execution backend — CPU (0) by default.
    // Other backends fall back to CPU for ops they don't implement.
    // Use backend_t values from registry.h to set this field.
    uint8_t preferred_backend = 0; // backend_t::CPU

    // Called by format loaders to register format-specific model data
    void set_model_handle(void* data, void (*free_fn)(void*)) {
        if (model_data_ && model_free_) model_free_(model_data_);
        model_data_ = data;
        model_free_ = free_fn;
    }
    void* model_handle() const { return model_data_; }

    // Memory-mapped file data (set by load_from_file, freed in ~context_t).
    // When non-null, format loaders can alias data instead of copying.
    bool is_mmap() const { return mmap_data_ != nullptr; }
    void* mmap_data_ = nullptr;
    void (*mmap_free_)(void*) = nullptr;

    // Model directory path — set by load_from_file() for resolving external data.
    std::string model_dir;

private:
    // Opaque model data holder (format-specific, freed by model_free_)
    void* model_data_ = nullptr;
    void (*model_free_)(void*) = nullptr;
};

// Configure the singleton thread pool worker count.
// Process-global: affects all context_t instances in this process.
// Call once at program start, before any context_t::prepare()/run() runs.
// Concurrent calls race on the underlying static storage.
// 0 = auto-detect (hardware_concurrency / smt_stride, i.e., physical cores).
void set_global_thread_count(int n);

static inline int dim_next(std::span<int> dims, std::span<const int> dim_max)
{
    int ndim = static_cast<int>(dims.size());
    if (ndim == 0) {
        return 0;
    }
    while (1) {
        ndim = ndim - 1;
        dims[ndim] += 1;
        if (dims[ndim] < dim_max[ndim]) {
            return 1;
        }else {
            if (ndim == 0) {
                return 0;
            }
            dims[ndim] = 0;
        }
    }
}

static inline int dim_offset(std::span<const int> dims, std::span<const int> dim_max)
{
    int o = 0, s = 1;
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; i--) {
        o += dims[i] * s;
        s *= dim_max[i];
    }
    return o;
}

static inline int stride_offset(std::span<const int> indices, std::span<const int> strides)
{
    int o = 0;
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        o += indices[i] * strides[i];
    }
    return o;
}

template <typename T>
T string2enum(std::string_view value, T def) {
    static_assert(std::is_enum_v<T>);
    auto v0 = enchantum::cast<T>(value);
    if (v0.has_value()) {
        return v0.value();
    }else {
        return def;
    }
}

} // namespace nnr
