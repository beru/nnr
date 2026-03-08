#pragma once

#include <complex>
#include <functional>
#include <memory>
#include <numeric>
#include "nnr.h"

#include "bool.h"
#include "float16.h"
#include "bfloat16.h"

namespace nnr {

// Type-to-data_type_t mapping (used at compile time for bitmask/table construction).
template<typename T> struct type2int {};
template<> struct type2int<bool_t>                 { static constexpr int v = NNR_DATA_TYPE_BOOL; };
template<> struct type2int<int8_t>                 { static constexpr int v = NNR_DATA_TYPE_INT8; };
template<> struct type2int<int16_t>                { static constexpr int v = NNR_DATA_TYPE_INT16; };
template<> struct type2int<int32_t>                { static constexpr int v = NNR_DATA_TYPE_INT32; };
template<> struct type2int<int64_t>                { static constexpr int v = NNR_DATA_TYPE_INT64; };
template<> struct type2int<uint8_t>                { static constexpr int v = NNR_DATA_TYPE_UINT8; };
template<> struct type2int<uint16_t>               { static constexpr int v = NNR_DATA_TYPE_UINT16; };
template<> struct type2int<uint32_t>               { static constexpr int v = NNR_DATA_TYPE_UINT32; };
template<> struct type2int<uint64_t>               { static constexpr int v = NNR_DATA_TYPE_UINT64; };
template<> struct type2int<bfloat16_t>             { static constexpr int v = NNR_DATA_TYPE_BFLOAT16; };
template<> struct type2int<float16_t>              { static constexpr int v = NNR_DATA_TYPE_FLOAT16; };
template<> struct type2int<float>                  { static constexpr int v = NNR_DATA_TYPE_FLOAT32; };
template<> struct type2int<double>                 { static constexpr int v = NNR_DATA_TYPE_FLOAT64; };
template<> struct type2int<std::complex<float>>    { static constexpr int v = NNR_DATA_TYPE_COMPLEX64; };
template<> struct type2int<std::complex<double>>   { static constexpr int v = NNR_DATA_TYPE_COMPLEX128; };
template<> struct type2int<std::string>            { static constexpr int v = NNR_DATA_TYPE_STRING; };

enum class broadcast_kind : uint8_t {
    SAME_SHAPE, A_SCALAR, B_SCALAR, PER_CHANNEL, GENERAL
};

// Check if tensor t is a per-channel broadcast source relative to 4D output y.
// Matches shapes [C], [1,C,1,1], [C,1,1] where C = y->dims[1].
inline bool is_per_channel_4d(const tensor_t* t, const tensor_t* y) {
    if (y->ndim != 4) return false;
    int C = y->dims[1];
    if ((int)t->ndata != C) return false;
    if (t->ndim == 1 && t->dims[0] == C) return true;
    if (t->ndim == 4 && t->dims[0] == 1 && t->dims[1] == C && t->dims[2] == 1 && t->dims[3] == 1) return true;
    if (t->ndim == 3 && t->dims[0] == C && t->dims[1] == 1 && t->dims[2] == 1) return true;
    return false;
}

inline broadcast_kind classify_broadcast(const tensor_t* a, const tensor_t* b, const tensor_t* y) {
    if (a->ndata == y->ndata && b->ndata == y->ndata
        && a->ndim == y->ndim && b->ndim == y->ndim
        && memcmp(a->dims, y->dims, y->ndim * sizeof(int)) == 0
        && memcmp(b->dims, y->dims, y->ndim * sizeof(int)) == 0)
        return broadcast_kind::SAME_SHAPE;
    if (a->ndata == 1) return broadcast_kind::A_SCALAR;
    if (b->ndata == 1) return broadcast_kind::B_SCALAR;
    // Per-channel: one input has full output shape, the other is [1,C,1,1] or [C].
    if (y->ndim == 4) {
        auto same_shape = [](const tensor_t* t, const tensor_t* y) {
            return t->ndim == y->ndim && memcmp(t->dims, y->dims, y->ndim * sizeof(int)) == 0;
        };
        if (is_per_channel_4d(b, y) && same_shape(a, y)) return broadcast_kind::PER_CHANNEL;
        if (is_per_channel_4d(a, y) && same_shape(b, y)) return broadcast_kind::PER_CHANNEL;
    }
    return broadcast_kind::GENERAL;
}

// Compute broadcast strides for tensor t relative to output shape y.
// bstr[d] = 0 if t broadcasts (dim==1) along that axis, else contiguous stride.
// This avoids per-element broadcast_map_address() calls by precomputing offsets.
inline void compute_broadcast_strides(const tensor_t* t, const tensor_t* y, int* bstr) {
    int yndim = y->ndim, tndim = t->ndim;
    int off = yndim - tndim;
    for (int d = 0; d < off; d++) bstr[d] = 0;
    int s = 1;
    for (int d = tndim - 1; d >= 0; d--) {
        bstr[off + d] = (t->dims[d] == 1) ? 0 : s;
        s *= t->dims[d];
    }
}

template <typename T, typename OutT = T, typename Op>
inline bool binary_broadcast_exec(const tensor_t* a, const tensor_t* b, tensor_t* y, Op op) {
    OutT* py = (OutT*)y->data;
    size_t l = y->ndata;
    switch (classify_broadcast(a, b, y)) {
    case broadcast_kind::SAME_SHAPE: {
        const T* pa = (const T*)a->data;
        const T* pb = (const T*)b->data;
        for (size_t i = 0; i < l; ++i) py[i] = op(pa[i], pb[i]);
        return true;
    }
    case broadcast_kind::A_SCALAR: {
        T sa = *(const T*)a->data;
        const T* pb = (const T*)b->data;
        for (size_t i = 0; i < l; ++i) py[i] = op(sa, pb[i]);
        return true;
    }
    case broadcast_kind::B_SCALAR: {
        const T* pa = (const T*)a->data;
        T sb = *(const T*)b->data;
        for (size_t i = 0; i < l; ++i) py[i] = op(pa[i], sb);
        return true;
    }
    case broadcast_kind::PER_CHANNEL: {
        const T* pa = (const T*)a->data;
        const T* pb = (const T*)b->data;
        int C = y->dims[1];
        bool a_per_ch = ((int)a->ndata == C);
        if (y->format == memory_layout_t::NHWC) {
            // NHWC: channel is innermost dim, repeats every C elements
            if (a_per_ch) {
                for (size_t i = 0; i < l; ++i)
                    py[i] = op(pa[i % C], pb[i]);
            } else {
                for (size_t i = 0; i < l; ++i)
                    py[i] = op(pa[i], pb[i % C]);
            }
        } else {
            // NCHW: channel plane is contiguous (HW elements per channel)
            int HW = 1;
            for (int d = 2; d < y->ndim; d++) HW *= y->dims[d];
            int NC = y->dims[0] * C;
            size_t idx = 0;
            if (a_per_ch) {
                for (int nc = 0; nc < NC; ++nc) {
                    T cv = pa[nc % C];
                    for (int hw = 0; hw < HW; ++hw, ++idx)
                        py[idx] = op(cv, pb[idx]);
                }
            } else {
                for (int nc = 0; nc < NC; ++nc) {
                    T cv = pb[nc % C];
                    for (int hw = 0; hw < HW; ++hw, ++idx)
                        py[idx] = op(pa[idx], cv);
                }
            }
        }
        return true;
    }
    default: {
        const T* pa = (const T*)a->data;
        const T* pb = (const T*)b->data;
        int yndim = y->ndim;
        int a_bstr[MAX_NDIM], b_bstr[MAX_NDIM];
        compute_broadcast_strides(a, y, a_bstr);
        compute_broadcast_strides(b, y, b_bstr);

        // Fast 4D path: ONNX models are almost always 4D (N,C,H,W).
        // Nested loops avoid per-element modulo/division of the generic path.
        if (yndim == 4) {
            int D0 = y->dims[0], D1 = y->dims[1], D2 = y->dims[2], D3 = y->dims[3];
            size_t idx = 0;
            for (int i0 = 0; i0 < D0; i0++) {
                int a0 = i0 * a_bstr[0], b0 = i0 * b_bstr[0];
                for (int i1 = 0; i1 < D1; i1++) {
                    int a1 = a0 + i1 * a_bstr[1], b1 = b0 + i1 * b_bstr[1];
                    for (int i2 = 0; i2 < D2; i2++) {
                        int a2 = a1 + i2 * a_bstr[2], b2 = b1 + i2 * b_bstr[2];
                        int as3 = a_bstr[3], bs3 = b_bstr[3];
                        for (int i3 = 0; i3 < D3; i3++)
                            py[idx++] = op(pa[a2 + i3 * as3], pb[b2 + i3 * bs3]);
                    }
                }
            }
            return true;
        }

        // Generic broadcast with precomputed strides (avoids broadcast_map_address)
        for (size_t i = 0; i < l; ++i) {
            int a_off = 0, b_off = 0;
            size_t rem = i;
            for (int d = yndim - 1; d >= 0; d--) {
                int idx = (int)(rem % y->dims[d]);
                rem /= y->dims[d];
                a_off += idx * a_bstr[d];
                b_off += idx * b_bstr[d];
            }
            py[i] = op(pa[a_off], pb[b_off]);
        }
        return true;
    }
    }
}

// Describes how to decompose a reduction into batch × reduce × tail loops.
// contiguous=true means the reduced axes form a contiguous block in the tensor,
// so data layout is [batch, reduce, tail] and a simple triple-loop can be used.
struct reduce_plan_t {
    int batch_size = 1;
    int reduce_size = 1;
    int tail_size = 1;
    bool contiguous = false;
};

inline reduce_plan_t plan_reduce(const int* dims, int ndim, const int* axes, int naxes)
{
    reduce_plan_t plan;
    if (naxes == 0 || ndim == 0) return plan;
    int sorted[MAX_NDIM];
    for (int i = 0; i < naxes; ++i) sorted[i] = axes[i];
    std::sort(sorted, sorted + naxes);
    for (int i = 1; i < naxes; ++i)
        if (sorted[i] != sorted[i-1] + 1) return plan;
    plan.contiguous = true;
    int first = sorted[0], last = sorted[naxes-1];
    for (int i = 0; i < first; ++i) plan.batch_size *= dims[i];
    for (int i = first; i <= last; ++i) plan.reduce_size *= dims[i];
    for (int i = last + 1; i < ndim; ++i) plan.tail_size *= dims[i];
    return plan;
}

using resolved_exec_fn = bool(*)(operator_t*);

template <typename Self, typename T>
bool typed_exec_thunk(operator_t* op) {
    return static_cast<Self*>(op)->template exec<T>();
}

// Bitmask of supported data_type_t values + compressed function pointer table.
// Table has exactly popcount(mask) entries, indexed by popcount(mask & ((1<<dt)-1)).
template <typename... Types>
constexpr uint32_t type_mask = ((1u << type2int<Types>::v) | ...);

template <typename Self, typename... Types>
consteval auto make_compressed_table() {
    constexpr int N = sizeof...(Types);
    std::array<resolved_exec_fn, N> table{};
    constexpr uint32_t mask = type_mask<Types...>;
    int i = 0;
    ((table[std::popcount(mask & ((1u << type2int<Types>::v) - 1))] = &typed_exec_thunk<Self, Types>, ++i), ...);
    return table;
}

template <int MinOpset, typename... Types>
struct opset_t {
    static constexpr int min_opset = MinOpset;
    static constexpr uint32_t mask = type_mask<Types...>;
    template <typename Self>
    static resolved_exec_fn resolve(uint32_t dt) {
        if (!(mask & (1u << dt))) return nullptr;
        static constexpr auto table = make_compressed_table<Self, Types...>();
        return table[std::popcount(mask & ((1u << dt) - 1))];
    }
};

// Merge multiple opset entries into one bitmask + compressed table.
template <typename... OpsetEntries>
constexpr uint32_t merged_mask = (OpsetEntries::mask | ...);

template <typename Self, uint32_t Mask, typename... OpsetEntries>
consteval auto make_merged_compressed_table() {
    constexpr int N = std::popcount(Mask);
    std::array<resolved_exec_fn, N> table{};
    // Fill from each opset entry's types
    auto fill_one = [&]<int MinOpset, typename... Types>(opset_t<MinOpset, Types...>) {
        ((table[std::popcount(Mask & ((1u << type2int<Types>::v) - 1))] = &typed_exec_thunk<Self, Types>), ...);
    };
    (fill_one(OpsetEntries{}), ...);
    return table;
}

template <typename Self, typename... OpsetEntries>
resolved_exec_fn resolve_opset_type([[maybe_unused]] int opset, data_type_t type) {
    uint32_t dt = (uint32_t)type;
#ifdef NNR_STRICT_OPSET_TYPES
    resolved_exec_fn fn = nullptr;
    ((fn == nullptr && opset >= OpsetEntries::min_opset &&
        (fn = OpsetEntries::template resolve<Self>(dt), true)), ...);
    return fn;
#else
    constexpr uint32_t mask = merged_mask<OpsetEntries...>;
    if (!(mask & (1u << dt))) return nullptr;
    static constexpr auto table = make_merged_compressed_table<Self, mask, OpsetEntries...>();
    return table[std::popcount(mask & ((1u << dt) - 1))];
#endif
}

// typed_exec: plain type list
template <typename ExecT, typename... Types>
inline bool typed_exec(ExecT* self, data_type_t type) {
    auto fn = opset_t<0, Types...>::template resolve<ExecT>((uint32_t)type);
    return fn ? fn(self) : false;
}

// typed_exec: with opset entries
template <typename ExecT, typename... OpsetEntries>
inline bool typed_exec(ExecT* self, int opset, data_type_t type) {
    auto fn = resolve_opset_type<ExecT, OpsetEntries...>(opset, type);
    return fn ? fn(self) : false;
}

// CRTP base for simple unary element-wise ops.
// Derived must provide: static auto fn(auto x) { ... }
template <typename Derived, typename... OpsetEntries>
struct unary_foreach_op_t : public operator_t {
    bool init() override { layout_mask = LAYOUT_ALL; return is_inout_size(1, 1); }
    template <typename T>
    bool exec() {
        foreach_tensor<T>([](auto x) { return Derived::fn(x); });
        return true;
    }
    bool exec() override {
        auto fn = resolve_opset_type<Derived, OpsetEntries...>(opset, inputs[0]->type);
        return fn ? fn(this) : false;
    }
};

// CRTP base for binary arithmetic ops (2 inputs, 1 output, same-type broadcast).
// Derived must provide: static auto fn(auto a, auto b) { ... }
template <typename Derived, typename... OpsetEntries>
struct binary_arith_op_t : public operator_t {
    bool init() override { layout_mask = LAYOUT_ALL; return is_inout_size(2, 1); }
    bool reshape() override {
        if (!outputs[0]->reshape_multi_broadcast(inputs[0], inputs[1], inputs[0]->type))
            return false;
        // General broadcast uses NCHW dimension ordering — not safe for NHWC data.
        // Same-shape, scalar, and per-channel broadcasts are layout-agnostic.
        auto kind = classify_broadcast(inputs[0], inputs[1], outputs[0]);
        layout_mask = (kind == broadcast_kind::GENERAL) ? LAYOUT_NCHW : LAYOUT_ALL;
        return true;
    }
    template <typename T>
    bool exec() {
        return binary_broadcast_exec<T>(inputs[0], inputs[1], outputs[0],
            [](T a, T b) -> T { return (T)Derived::fn(a, b); });
    }
    bool exec() override {
        auto fn = resolve_opset_type<Derived, OpsetEntries...>(opset, inputs[0]->type);
        return fn ? fn(this) : false;
    }
};

// CRTP base for comparison ops (2 inputs, 1 bool output, broadcast).
// Derived must provide: static bool fn(auto a, auto b) { ... }
template <typename Derived, typename... OpsetEntries>
struct comparison_op_t : public operator_t {
    bool init() override { return is_inout_size(2, 1); }
    bool reshape() override {
        return outputs[0]->reshape_multi_broadcast(inputs[0], inputs[1], NNR_DATA_TYPE_BOOL);
    }
    template <typename T>
    bool exec() {
        return binary_broadcast_exec<T, bool_t>(inputs[0], inputs[1], outputs[0],
            [](T a, T b) -> bool_t { return bool_t(Derived::fn(a, b)); });
    }
    bool exec() override {
        auto fn = resolve_opset_type<Derived, OpsetEntries...>(opset, inputs[0]->type);
        return fn ? fn(this) : false;
    }
};

} // namespace nnr
