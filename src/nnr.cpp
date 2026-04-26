#include <complex>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <cinttypes>
#include "nnr.h"
#include "aligned_alloc.h"
#include "layout_cost.h"
#include "registry.h"
#include "backend/cpu/solve_operator.h"
#include "graph_optimizer.h"
#include "backend/cpu/util.h"
#include "backend/cpu/kernel/layout.h"
#include "format/onnx/onnx_loader.h"
#include "format/tflite/tflite_loader.h"
#include "thread_pool.h"
#include "trace.h"
#include "no_popup.h"
#if defined(NNR_USE_CUPTI)
#include "backend/gpu/cuda/cupti_profiler.h"
#endif

#ifdef NNR_ENABLE_WEBGPU
#include "backend/webgpu/buffer.h"
#include "backend/webgpu/profiler.h"
#endif

#ifdef _WIN32
#include <sys/stat.h>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace nnr {

// Forward declaration – implemented in format/onnx/onnx_loader.cpp
tensor_t* onnx_tensor_alloc_from_file(std::string_view filename);

// ---------------------------------------------------------------------------
// attr_t
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// operator_t attribute accessors
// ---------------------------------------------------------------------------

attr_t* operator_t::find_attr(attr_key_t key)
{
    for (auto& [k, v] : attrs) {
        if (k == key) return &v;
    }
    return nullptr;
}

float operator_t::attribute(attr_key_t key, float def)
{
    auto* a = find_attr(key);
    if (a && a->kind == attr_t::kind_t::FLOAT) return a->f;
    return def;
}

int32_t operator_t::attribute(attr_key_t key, int32_t def)
{
    return (int32_t)attribute(key, (int64_t)def);
}

int64_t operator_t::attribute(attr_key_t key, int64_t def)
{
    auto* a = find_attr(key);
    if (a && a->kind == attr_t::kind_t::INT) return a->i;
    return def;
}

std::string_view operator_t::attribute(attr_key_t key, std::string_view def)
{
    auto* a = find_attr(key);
    if (a && a->kind == attr_t::kind_t::STRING) return a->s;
    return def;
}

int operator_t::attribute(attr_key_t key, float*& floats)
{
    auto* a = find_attr(key);
    if (a && a->kind == attr_t::kind_t::FLOATS) {
        floats = const_cast<float*>(a->floats.data());
        return (int)a->floats.size();
    }
    return 0;
}

int operator_t::attribute(attr_key_t key, int64_t*& ints)
{
    auto* a = find_attr(key);
    if (a && a->kind == attr_t::kind_t::INTS) {
        ints = const_cast<int64_t*>(a->ints.data());
        return (int)a->ints.size();
    }
    return 0;
}

int operator_t::attribute(attr_key_t key, tensor_t* t)
{
    auto* a = find_attr(key);
    if (a && a->kind == attr_t::kind_t::TENSOR && a->tensor) {
        tensor_t* src = a->tensor;
        if (t->type != src->type || t->ndim != src->ndim ||
            memcmp(t->dims, src->dims, src->ndim * sizeof(int)) != 0) {
            if (!t->reinit(src->type, src->dim_span())) return 0;
        }
        if (!t->apply(*src)) return 0;
        return 1;
    }
    return 0;
}

graph_t* operator_t::attribute_subgraph(std::string_view name)
{
    auto* a = find_attr(attr_key_from_string(name));
    if (a && a->kind == attr_t::kind_t::GRAPH) return a->subgraph;
    return nullptr;
}

void* operator_t::attribute_raw(std::string_view name)
{
    auto* a = find_attr(attr_key_from_string(name));
    return a ? a->raw : nullptr;
}

float operator_t::layout_cost(memory_layout_t layout, bool input_nhwc) const
{
    auto candidates = estimate_costs(input_nhwc);
    if (candidates.empty()) return 0;
    float best = 0;
    bool found = false;
    for (auto& c : candidates) {
        if (c.layout != layout) continue;
        float cost = reduce_to_scalar(c);
        if (!found || cost < best) { best = cost; found = true; }
    }
    return best;
}

// ---------------------------------------------------------------------------
// sequence_t
// ---------------------------------------------------------------------------

sequence_t::~sequence_t()
{
    for (auto* t : tensors) delete t;
}

// ---------------------------------------------------------------------------
// tensor_t
// ---------------------------------------------------------------------------

void delete_data(void* data, data_type_t type)
{
    if (type == NNR_DATA_TYPE_STRING) {
        delete[] (std::string*)data;
    } else if (type == NNR_DATA_TYPE_SEQUENCE) {
        delete (sequence_t*)data;
    } else {
        delete[] (char*)data;
    }
}

tensor_t::tensor_t(std::string_view name, data_type_t type, std::span<const int> dims)
    : name(name)
{
    // reinit's return is captured in the `allocation_failed` member — ctors
    // can't propagate a bool, and callers that `new` a tensor_t check that
    // field after construction. Ignore [[nodiscard]] warning here since the
    // member already captures the same signal.
    (void)reinit(type, dims);
}

tensor_t::~tensor_t()
{
    if (owns_data && (ndata > 0) && data) {
        delete_data(data, type);
    }
}

void tensor_t::clear_quant()
{
    quant_scale = 0.0f;
    quant_zero_point = 0;
    quant_axis = -1;
    nnr_aligned_free(quant_scales);
    quant_scales = nullptr;
    nnr_aligned_free(quant_zero_points);
    quant_zero_points = nullptr;
}

bool tensor_t::reinit(data_type_t type, std::span<const int> dims)
{
    const int ndim = static_cast<int>(dims.size());
    size_t n;

    if (ndim > 0) {
        this->ndim = 0;
    }
    // Preserve view state: when the memory planner has marked this tensor as
    // a non-owning view (owns_data=false), reinit must update dims/strides
    // without allocating fresh storage and without flipping ownership. The
    // view executor (e.g. Slice) will assign data=view_pointer after reshape.
    // Without this, ssd-12's NMS→Slice path reseats the view as an owning
    // tensor whose `data` points into another buffer, producing a heap-free
    // crash at teardown.
    //
    // Two non-owning sub-cases share owns_data=false:
    //   - pool-backed: data points at a pool slot sized for the planned max
    //     shape; a smaller reshape (e.g. NMS shrinking to actual results)
    //     must keep that pointer or the op's exec writes to nullptr.
    //   - view-aliased: data was zeroed by the planner's third pass, exec
    //     will reassign from input.
    // Preserving the existing `data` value handles both: pool slot stays
    // valid, view-aliased stays null until exec assigns.
    const bool was_owning = owns_data;
    if (was_owning) {
        if ((ndata > 0) && data) {
            delete_data(data, this->type);
        }
        data = nullptr;
    }
    // Non-owning: leave `data` untouched. Pool slots stay valid for the
    // planned max size; view-aliased tensors were already nulled by the
    // planner and stay null.
#ifdef NNR_ENABLE_WEBGPU
    // reinit frees t->data and re-allocates at a new address/size. Any GPU
    // residency record we built previously is now semantically stale; drop
    // it so the next ensure_buffer rebuilds cleanly. Without this, callers
    // had to remember to manually webgpu::mark_cpu_written(t) after reinit,
    // which is an easy-to-miss footgun when models use dynamic shapes.
    webgpu::forget(this);
#endif
    ndata = 0;
    owns_data = was_owning;
    allocation_failed = false;
    this->type = type;
    if (type == NNR_DATA_TYPE_UNDEFINED) {
        return true;
    }
    if ((ndim > 0) && !dims.empty()) {
        assert(ndim <= MAX_NDIM);
        this->strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--) {
            this->strides[i] = dims[i + 1] * this->strides[i + 1];
        }
        std::copy(dims.begin(), dims.end(), this->dims);
        this->ndim = ndim;
        // Overflow-checked multiplication of dimensions
        n = 1;
        bool overflow = false;
        for (int i = 0; i < ndim; ++i) {
            if (dims[i] < 0) { n = 0; break; }
            size_t d = (size_t)dims[i];
            if (d != 0 && n > SIZE_MAX / d) { overflow = true; break; }
            n *= d;
        }
        if (overflow) {
            data = nullptr;
            ndata = 0;
            allocation_failed = true;
            return false;
        }
    }else {
        n = 1;
    }
    if (n == 0) {
        // Valid zero-sized tensor — not a failure, just no storage.
        // For non-owning tensors keep `data` so a subsequent reshape back
        // to non-zero dims can reuse the pool slot.
        if (was_owning) data = nullptr;
        ndata = 0;
        return true;
    }
    if (!was_owning) {
        // Non-owning tensor: dims/strides updated, consumer-visible ndata
        // reflects logical extent. `data` was preserved above — pool slots
        // remain valid for the planned max size; view-aliased tensors stay
        // null until their producing op's exec assigns from input.
        ndata = n;
        return true;
    }
    switch (type) {
    case NNR_DATA_TYPE_UNDEFINED: break;
    case NNR_DATA_TYPE_BOOL: data = new (std::nothrow) bool[n](); break;
    case NNR_DATA_TYPE_INT8: data = new (std::nothrow) int8_t[n](); break;
    case NNR_DATA_TYPE_INT16: data = new (std::nothrow) int16_t[n](); break;
    case NNR_DATA_TYPE_INT32: data = new (std::nothrow) int32_t[n](); break;
    case NNR_DATA_TYPE_INT64: data = new (std::nothrow) int64_t[n](); break;
    case NNR_DATA_TYPE_UINT8: data = new (std::nothrow) uint8_t[n](); break;
    case NNR_DATA_TYPE_UINT16: data = new (std::nothrow) uint16_t[n](); break;
    case NNR_DATA_TYPE_UINT32: data = new (std::nothrow) uint32_t[n](); break;
    case NNR_DATA_TYPE_UINT64: data = new (std::nothrow) uint64_t[n](); break;
    case NNR_DATA_TYPE_BFLOAT16: data = new (std::nothrow) uint16_t[n](); break;
    case NNR_DATA_TYPE_FLOAT16: data = new (std::nothrow) uint16_t[n](); break;
    case NNR_DATA_TYPE_FLOAT32: data = new (std::nothrow) float[n](); break;
    case NNR_DATA_TYPE_FLOAT64: data = new (std::nothrow) double[n](); break;
    case NNR_DATA_TYPE_COMPLEX64: data = new (std::nothrow) std::complex<float>[n](); break;
    case NNR_DATA_TYPE_COMPLEX128: data = new (std::nothrow) std::complex<double>[n](); break;
    case NNR_DATA_TYPE_STRING: data = new (std::nothrow) std::string[n]; break;
    case NNR_DATA_TYPE_INT4:  data = new (std::nothrow) int8_t[n]();  break;
    case NNR_DATA_TYPE_UINT4: data = new (std::nothrow) uint8_t[n](); break;
    case NNR_DATA_TYPE_INT2:  data = new (std::nothrow) int8_t[n]();  break;
    case NNR_DATA_TYPE_UINT2: data = new (std::nothrow) uint8_t[n](); break;
    case NNR_DATA_TYPE_FLOAT8E4M3FN:
    case NNR_DATA_TYPE_FLOAT8E4M3FNUZ:
    case NNR_DATA_TYPE_FLOAT8E5M2:
    case NNR_DATA_TYPE_FLOAT8E5M2FNUZ:
    case NNR_DATA_TYPE_FLOAT4E2M1:
    case NNR_DATA_TYPE_FLOAT8E8M0: data = new (std::nothrow) uint8_t[n](); break;
    case NNR_DATA_TYPE_SEQUENCE:
        data = new (std::nothrow) sequence_t();
        ndata = 1;
        if (!data) {
            ndata = 0;
            allocation_failed = true;
            return false;
        }
        return true;
    default: break;
    }
    ndata = n;
    if (!data) {
        // operator new returned nullptr — ndata reflects the requested size
        // so callers can still reason about "wanted X, got nothing".
        allocation_failed = true;
        return false;
    }
    return true;
}

void tensor_t::apply(const void* buf, size_t len)
{
    if (!data || !buf || (len == 0)) {
        return;
    }
    size_t sz = data_type_sizeof(type);
    if (sz <= 0) {
        return;
    }
    if (type == NNR_DATA_TYPE_STRING) {
        std::string* p = (std::string*)data;
        const std::string* q = (const std::string*)buf;
        for (size_t idx = 0; idx < ndata; ++idx) {
            p[idx].clear();
        }
        size_t l = min(ndata, (size_t)len);
        for (size_t idx = 0; idx < l; ++idx) {
            p[idx] = q[idx];
        }
    }else {
        size_t l = ndata * sz;
        if (l > 0) {
            memcpy(data, buf, min(l, len));
        }
    }
}

bool tensor_t::apply(const tensor_t& t)
{
    if (ndim != t.ndim || type != t.type || memcmp(dims, t.dims, sizeof(int) * t.ndim) != 0) {
        bool need_reinit = (type == NNR_DATA_TYPE_UNDEFINED || t.type == NNR_DATA_TYPE_UNDEFINED
            || type == NNR_DATA_TYPE_SEQUENCE || t.type == NNR_DATA_TYPE_SEQUENCE);
        if (!need_reinit && ndim == t.ndim && memcmp(dims, t.dims, sizeof(int) * t.ndim) == 0
            && data_type_sizeof(type) == data_type_sizeof(t.type)
            && type != NNR_DATA_TYPE_STRING && t.type != NNR_DATA_TYPE_STRING) {
            // compatible sizes, keep original type
        } else {
            if (!reinit(t.type, t.dim_span())) return false;
        }
    }
    if (data && t.data && ndata > 0) {
        if (t.type == NNR_DATA_TYPE_SEQUENCE) {
            copy_data(this, &t);
        } else {
        size_t sz = data_type_sizeof(t.type);
        if (t.type == NNR_DATA_TYPE_STRING) {
            std::string* p = (std::string*)data;
            const std::string* q = (const std::string*)t.data;
            for (size_t i = 0; i < ndata; ++i) {
                p[i] = q[i];
            }
        }else {
            memcpy(data, t.data, sz * ndata);
        }
        }
    }
    return true;
}

bool tensor_t::reshape(std::span<const int> dims, data_type_t type)
{
    const int ndim = static_cast<int>(dims.size());
    if ((this->ndim != ndim) || (!dims.empty() && (memcmp(&this->dims[0], dims.data(), sizeof(int) * ndim) != 0)) || (this->type != type)) {
        if (!reinit(type, dims)) return false;
    }
    return true;
}

bool tensor_t::reshape_identity(const tensor_t* x, data_type_t type)
{
    if (x->ndim > 0 || x->ndata > 0) {
        if ((this->ndim != x->ndim) || (memcmp(this->dims, x->dims, sizeof(int) * this->ndim) != 0) || (this->type != type)) {
            if (!reinit(type, x->dim_span())) return false;
        }
    }
    // Propagate quantization metadata from input to output
    if (x->is_quantized())
        propagate_quant(x);
    return true;
}

bool tensor_t::reshape_multi_broadcast(const tensor_t* a, const tensor_t* b, data_type_t type)
{
    const int ndim = max(a->ndim, b->ndim);
    small_vector<int> dims(ndim);
    if (ndim > 0) {
        for (int i = a->ndim - 1, j = b->ndim - 1, k = ndim - 1; k >= 0; k--) {
            if (i < 0) {
                dims[k] = b->dims[j--];
            }else if (j < 0) {
                dims[k] = a->dims[i--];
            }else {
                if (a->dims[i] == b->dims[j]) {
                    dims[k] = a->dims[i];
                }else if ((a->dims[i] == 1) || (b->dims[j] == 1)) {
                    dims[k] = (a->dims[i] > b->dims[j]) ? a->dims[i] : b->dims[j];
                }else {
                    return false;
                }
                i--;
                j--;
            }
        }
    }
    if ((this->type != type) || (this->ndim != ndim) || (memcmp(this->dims, dims.data(), sizeof(int) * ndim) != 0)) {
        if (!reinit(type, dims)) return false;
    }
    return true;
}

void* tensor_t::broadcast_map_address(const tensor_t* y, int64_t offset)
{
    int xndim = this->ndim;
    int yndim = y->ndim;

    if ((xndim > 0) && (yndim > 0)) {
        int dndim = yndim - xndim;
        small_vector<int> ix(xndim);
        small_vector<int> iy(yndim);
        y->offset_to_indices(offset, iy);
        for (int i = 0; i < xndim; ++i) {
            ix[i] = iy[dndim + i] % this->dims[i];
        }
        return (char*)this->data + this->indices_to_offset(ix) * data_type_sizeof(this);
    }
    return this->data;
}

tensor_t* tensor_t::alloc_from_file(std::string_view filename)
{
    return onnx_tensor_alloc_from_file(filename);
}

void copy_data(tensor_t* y, const tensor_t* x)
{
    if (x->type == NNR_DATA_TYPE_STRING) {
        const std::string* px = (const std::string*)x->data;
        std::string* py = (std::string*)y->data;
        for (size_t i = 0, l = y->ndata; i < l; ++i) {
            py[i] = px[i];
        }
    } else if (x->type == NNR_DATA_TYPE_SEQUENCE) {
        const sequence_t* src = (const sequence_t*)x->data;
        sequence_t* dst = (sequence_t*)y->data;
        if (!src || !dst) return;
        for (auto* t : dst->tensors) delete t;
        dst->tensors.clear();
        dst->elem_type = src->elem_type;
        for (const auto* t : src->tensors) {
            tensor_t* copy = new (std::nothrow) tensor_t("", t->type, t->dim_span());
            if (copy && !copy->allocation_failed && copy->apply(*t)) {
                dst->tensors.push_back(copy);
            } else {
                delete copy;
            }
        }
    } else {
        size_t sz = data_type_sizeof(x->type);
        if (sz > 0) {
            memcpy(y->data, x->data, sz * y->ndata);
        }
    }
}

// ---------------------------------------------------------------------------
// tensor_type helpers
// ---------------------------------------------------------------------------

std::string_view data_type_tostring(data_type_t type)
{
    switch (type) {
    case NNR_DATA_TYPE_FLOAT32:         return "float32";
    case NNR_DATA_TYPE_FLOAT64:         return "float64";
    case NNR_DATA_TYPE_FLOAT16:         return "float16";
    case NNR_DATA_TYPE_BFLOAT16:        return "bfloat16";
    case NNR_DATA_TYPE_INT8:            return "int8";
    case NNR_DATA_TYPE_INT16:           return "int16";
    case NNR_DATA_TYPE_INT32:           return "int32";
    case NNR_DATA_TYPE_INT64:           return "int64";
    case NNR_DATA_TYPE_UINT8:           return "uint8";
    case NNR_DATA_TYPE_UINT16:          return "uint16";
    case NNR_DATA_TYPE_UINT32:          return "uint32";
    case NNR_DATA_TYPE_UINT64:          return "uint64";
    case NNR_DATA_TYPE_BOOL:            return "bool";
    case NNR_DATA_TYPE_STRING:          return "string";
    case NNR_DATA_TYPE_COMPLEX64:       return "complex64";
    case NNR_DATA_TYPE_COMPLEX128:      return "complex128";
    case NNR_DATA_TYPE_FLOAT8E4M3FN:    return "float8e4m3fn";
    case NNR_DATA_TYPE_FLOAT8E4M3FNUZ:  return "float8e4m3fnuz";
    case NNR_DATA_TYPE_FLOAT8E5M2:      return "float8e5m2";
    case NNR_DATA_TYPE_FLOAT8E5M2FNUZ:  return "float8e5m2fnuz";
    case NNR_DATA_TYPE_FLOAT8E8M0:      return "float8e8m0";
    case NNR_DATA_TYPE_UINT4:           return "uint4";
    case NNR_DATA_TYPE_INT4:            return "int4";
    case NNR_DATA_TYPE_FLOAT4E2M1:      return "float4e2m1";
    case NNR_DATA_TYPE_UINT2:           return "uint2";
    case NNR_DATA_TYPE_INT2:            return "int2";
    case NNR_DATA_TYPE_SEQUENCE:        return "sequence";
    default:                            return "undefined";
    }
}

size_t data_type_sizeof(data_type_t type)
{
    switch (type) {
    case NNR_DATA_TYPE_FLOAT32:         return sizeof(float);
    case NNR_DATA_TYPE_FLOAT64:         return sizeof(double);
    case NNR_DATA_TYPE_FLOAT16:         return sizeof(float16_t);
    case NNR_DATA_TYPE_BFLOAT16:        return sizeof(bfloat16_t);
    case NNR_DATA_TYPE_INT8:            return sizeof(int8_t);
    case NNR_DATA_TYPE_INT16:           return sizeof(int16_t);
    case NNR_DATA_TYPE_INT32:           return sizeof(int32_t);
    case NNR_DATA_TYPE_INT64:           return sizeof(int64_t);
    case NNR_DATA_TYPE_UINT8:           return sizeof(uint8_t);
    case NNR_DATA_TYPE_UINT16:          return sizeof(uint16_t);
    case NNR_DATA_TYPE_UINT32:          return sizeof(uint32_t);
    case NNR_DATA_TYPE_UINT64:          return sizeof(uint64_t);
    case NNR_DATA_TYPE_BOOL:            return sizeof(bool_t);
    case NNR_DATA_TYPE_STRING:          return sizeof(std::string);
    case NNR_DATA_TYPE_COMPLEX64:       return sizeof(std::complex<float>);
    case NNR_DATA_TYPE_COMPLEX128:      return sizeof(std::complex<double>);
    case NNR_DATA_TYPE_FLOAT8E4M3FN:
    case NNR_DATA_TYPE_FLOAT8E4M3FNUZ:
    case NNR_DATA_TYPE_FLOAT8E5M2:
    case NNR_DATA_TYPE_FLOAT8E5M2FNUZ:
    case NNR_DATA_TYPE_FLOAT8E8M0:      return sizeof(uint8_t);
    case NNR_DATA_TYPE_UINT4:           return sizeof(uint8_t);
    case NNR_DATA_TYPE_INT4:            return sizeof(int8_t);
    case NNR_DATA_TYPE_FLOAT4E2M1:      return sizeof(uint8_t);
    case NNR_DATA_TYPE_UINT2:           return sizeof(uint8_t);
    case NNR_DATA_TYPE_INT2:            return sizeof(int8_t);
    default:                            return 0;
    }
}

size_t data_type_sizeof(const tensor_t* tensor)
{
    return data_type_sizeof(tensor->type);
}

// ---------------------------------------------------------------------------
// tensor_equal
// ---------------------------------------------------------------------------

bool tensor_equal(const tensor_t* a, const tensor_t* b)
{
    if (!a || !b) return false;
    if (a->type == NNR_DATA_TYPE_SEQUENCE || b->type == NNR_DATA_TYPE_SEQUENCE) return false;
    if (a->type != b->type) {
        bool compatible = (data_type_sizeof(a->type) == data_type_sizeof(b->type))
            && a->type != NNR_DATA_TYPE_STRING && b->type != NNR_DATA_TYPE_STRING;
        if (!compatible) return false;
    }
    if (a->ndim != b->ndim) return false;
    if (a->ndata != b->ndata) return false;
    if (a->ndim > 0) {
        if (memcmp(&a->dims[0], &b->dims[0], sizeof(int) * a->ndim) != 0) return false;
    }
    switch (a->type) {
    case NNR_DATA_TYPE_BOOL:
    case NNR_DATA_TYPE_INT8:
    case NNR_DATA_TYPE_INT16:
    case NNR_DATA_TYPE_INT32:
    case NNR_DATA_TYPE_INT64:
    case NNR_DATA_TYPE_UINT8:
    case NNR_DATA_TYPE_UINT16:
    case NNR_DATA_TYPE_UINT32:
    case NNR_DATA_TYPE_UINT64:
    case NNR_DATA_TYPE_INT4:
    case NNR_DATA_TYPE_UINT4:
    case NNR_DATA_TYPE_INT2:
    case NNR_DATA_TYPE_UINT2:
    case NNR_DATA_TYPE_FLOAT8E4M3FN:
    case NNR_DATA_TYPE_FLOAT8E4M3FNUZ:
    case NNR_DATA_TYPE_FLOAT8E5M2:
    case NNR_DATA_TYPE_FLOAT8E5M2FNUZ:
    case NNR_DATA_TYPE_FLOAT4E2M1:
    case NNR_DATA_TYPE_FLOAT8E8M0:
        if (memcmp(a->data, b->data, a->ndata * data_type_sizeof(a)) != 0) return false;
        break;
    case NNR_DATA_TYPE_BFLOAT16:
    {
        const bfloat16_t* p = (const bfloat16_t*)a->data;
        const bfloat16_t* q = (const bfloat16_t*)b->data;
        for (size_t i = 0; i < a->ndata; ++i) {
            if (fabsf(p[i] - q[i]) > 1e-3) return false;
        }
    }
    break;
    case NNR_DATA_TYPE_FLOAT16:
    {
        const float16_t* p = (const float16_t*)a->data;
        const float16_t* q = (const float16_t*)b->data;
        for (size_t i = 0; i < a->ndata; ++i) {
            if (fabsf(p[i] - q[i]) > 1e-3) return false;
        }
    }
    break;
    case NNR_DATA_TYPE_FLOAT32:
    {
        const float* p = (const float*)a->data;
        const float* q = (const float*)b->data;
        for (size_t i = 0; i < a->ndata; ++i) {
            if (fabsf(p[i] - q[i]) > 1e-3) return false;
        }
    }
    break;
    case NNR_DATA_TYPE_FLOAT64:
    {
        const double* p = (const double*)a->data;
        const double* q = (const double*)b->data;
        for (size_t i = 0; i < a->ndata; ++i) {
            if (fabs(p[i] - q[i]) > 1e-3) return false;
        }
    }
    break;
    case NNR_DATA_TYPE_COMPLEX64:
    {
        const std::complex<float>* p = (const std::complex<float>*)a->data;
        const std::complex<float>* q = (const std::complex<float>*)b->data;
        for (size_t i = 0; i < a->ndata; ++i) {
            if (std::abs(p[i] - q[i]) > 1e-3) return false;
        }
    }
    break;
    case NNR_DATA_TYPE_COMPLEX128:
    {
        const std::complex<double>* p = (const std::complex<double>*)a->data;
        const std::complex<double>* q = (const std::complex<double>*)b->data;
        for (size_t i = 0; i < a->ndata; ++i) {
            if (std::abs(p[i] - q[i]) > 1e-3) return false;
        }
    }
    break;
    case NNR_DATA_TYPE_STRING:
    {
        const std::string* p = (const std::string*)a->data;
        const std::string* q = (const std::string*)b->data;
        for (size_t i = 0; i < a->ndata; ++i) {
            if (!p[i].empty() && !q[i].empty() && (p[i] != q[i])) return false;
        }
    }
    break;
    default:
        break;
    }
    return true;
}

// ---------------------------------------------------------------------------
// operator_t::dump
// ---------------------------------------------------------------------------

void operator_t::dump(bool detail) const
{
    std::println("{}: {}-{} ({})", node_name, op_type, opset,
        domain.empty() ? "ai.onnx" : domain);
    if (inputs.size() > 0) {
        std::println("\tInputs:");
        for (auto input : inputs) {
            std::print("\t\t");
            if (input) input->dump(detail);
        }
    }
    if (outputs.size() > 0) {
        std::println("\tOutputs:");
        for (auto output : outputs) {
            std::print("\t\t");
            if (output) output->dump(detail);
        }
    }
}

// ---------------------------------------------------------------------------
// graph_t
// ---------------------------------------------------------------------------

void graph_t::dump(bool detail) const
{
    for (auto node : nodes) {
        node->dump(detail);
    }
}

// ---------------------------------------------------------------------------
// context_t
// ---------------------------------------------------------------------------

context_t::~context_t()
{
    if (memory_planned) {
        planner.release();
        memory_planned = false;
    }
#ifdef NNR_ENABLE_WEBGPU
    // Release the WebGPU residency records keyed on these tensor pointers
    // BEFORE delete, or the next context's tensor allocations may reuse an
    // address and inherit a stale record (e.g. gpu_valid=true pointing at a
    // prior run's buffer), silently skipping the upload of fresh CPU data.
    for (auto& [key, value] : map)    webgpu::forget(value);
    for (auto* t : attr_tensors_)     webgpu::forget(t);
#endif
    for (auto& [key, value] : map) {
        delete value;
    }
    for (auto* t : attr_tensors_)    delete t;
    for (auto* g : attr_subgraphs_)  delete g;
    nnr_aligned_free(workspace);
    // Free multi-backend state (GPU caches, etc.)
    for (auto& b : backends) {
        if (b.free_fn && b.data) b.free_fn(b.data);
    }
    if (model_free_ && model_data_) model_free_(model_data_);
    // Free mmap AFTER model_data_ — model's string_views may alias into mmap'd region
    if (mmap_free_ && mmap_data_) mmap_free_(mmap_data_);
}

void context_t::pool_sleep() { nnr::pool_sleep(); }
void context_t::pool_wake()  { nnr::pool_wake(); }

void set_global_thread_count(int n) {
#ifndef NNR_NO_THREAD_POOL
    thread_pool_t::configure(std::max(1, n));
#else
    (void)n;
#endif
}

// Deprecated: forwards to set_global_thread_count. Declared [[deprecated]]
// in the header; suppress the warning on the definition itself.
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable : 4996)
#endif
void context_t::set_num_threads(int n) {
    set_global_thread_count(n);
}
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#  pragma warning(pop)
#endif

void context_t::invalidate_shapes()
{
    shapes_dirty = true;
    if (memory_planned) {
        planner.release();
        memory_planned = false;
    }
}

tensor_t* context_t::search_tensor(std::string_view name)
{
    for (auto& [k, v] : map) {
        if (k == name) return v;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Cross-platform memory-mapped file
// ---------------------------------------------------------------------------
struct mapped_file_t {
    void*  data = nullptr;
    size_t size = 0;
#ifdef _WIN32
    HANDLE file_handle = INVALID_HANDLE_VALUE;
    HANDLE mapping     = nullptr;
#else
    int    fd = -1;
#endif

    ~mapped_file_t() { close(); }
    mapped_file_t() = default;
    mapped_file_t(const mapped_file_t&) = delete;
    mapped_file_t& operator=(const mapped_file_t&) = delete;

    bool open(const char* path) {
#ifdef _WIN32
        file_handle = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                                  nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_handle == INVALID_HANDLE_VALUE) return false;
        LARGE_INTEGER li;
        if (!GetFileSizeEx(file_handle, &li) || li.QuadPart == 0) { close(); return false; }
        size = (size_t)li.QuadPart;
        mapping = CreateFileMappingA(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping) { close(); return false; }
        data = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        if (!data) { close(); return false; }
#else
        fd = ::open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size == 0) { close(); return false; }
        size = (size_t)st.st_size;
        data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) { data = nullptr; close(); return false; }
#endif
        return true;
    }

    void close() {
#ifdef _WIN32
        if (data) { UnmapViewOfFile(data); data = nullptr; }
        if (mapping) { CloseHandle(mapping); mapping = nullptr; }
        if (file_handle != INVALID_HANDLE_VALUE) { CloseHandle(file_handle); file_handle = INVALID_HANDLE_VALUE; }
#else
        if (data) { munmap(data, size); data = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
#endif
        size = 0;
    }
};

bool context_t::load_from_file(std::string_view filename)
{
    std::string path(filename);

    // Store model directory for resolving external data files
    {
        size_t pos = path.find_last_of("/\\");
        model_dir = (pos != std::string::npos) ? path.substr(0, pos + 1) : "./";
    }

    // Memory-map the file — zero-copy, OS manages paging.
    auto* mf = new (std::nothrow) mapped_file_t;
    if (!mf) return false;
    if (!mf->open(path.c_str())) {
        delete mf;
        return false;
    }

    // Store the mapping on context_t so it lives for the model's lifetime.
    // Both ONNX and TFLite loaders check mmap_data_ to skip their internal copy.
    mmap_data_ = mf;
    mmap_free_ = [](void* p) { delete (mapped_file_t*)p; };

    return load(mf->data, mf->size);
}

bool context_t::load(const void* buf, size_t len)
{
    // TFLite FlatBuffer: identifier "TFL3" at offset 4
    if (len >= 8 && memcmp((const uint8_t*)buf + 4, "TFL3", 4) == 0)
        return load_tflite(this, buf, len);
    return load_onnx(this, buf, len); // default (ONNX protobuf)
}

// ---------------------------------------------------------------------------
// context_t::run and supporting functions
// ---------------------------------------------------------------------------

static void ensure_workspace(context_t* ctx, size_t ws)
{
    if (ws > ctx->workspace_size) {
        nnr_aligned_free(ctx->workspace);
        ctx->workspace = nnr_aligned_alloc(ws, 64);
        ctx->workspace_size = ws;
    }
}

#if defined(NNR_USE_CUDA)
// CUDA Graph capture/replay hooks — defined in backend/gpu/cuda/cuda_backend.cpp.
// Opaque ints used to avoid leaking CUDA headers into nnr.cpp.
int  cuda_begin_run(context_t* ctx, bool all_ops_gpu, bool shapes_dirty);
void cuda_end_run  (context_t* ctx, int mode);
bool cuda_replay_and_sync(context_t* ctx);
#endif

#if defined(NNR_USE_CUPTI)
// Drain CUPTI kernel records and accumulate per-op GPU time into profile[].
// Called after each inference run (after cudaStreamSynchronize). Records that
// can't be assigned to an op (no canonical sequence yet) are dropped.
//
// host_clock_fired tells us whether the host-side prof.end() incremented
// calls for the just-completed run. In normal/capture mode it did, so CUPTI
// should not bump calls or it would double-count. In replay mode the fast
// path is skipped, so CUPTI must bump calls = 1 per op exec'd this run.
static void drain_cupti_into_profile(std::vector<node_profile_t>& profile,
                                     bool host_clock_fired) {
    auto& cp = nnr::gpu::cupti_profiler();
    cp.flush();
    auto recs = cp.drain();
#ifdef _WIN32
    int64_t qpc_freq;
    QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&qpc_freq));
    // Convert ns → QPC ticks so the values share node_profile_t::total_ticks
    // semantics with the host-clock profiler. ticks = ns * qpc_freq / 1e9.
    const double ns_to_ticks = (double)qpc_freq / 1e9;
#else
    const double ns_to_ticks = 1.0;  // already ns
#endif
    // Each op may launch multiple kernels — their times sum into the same
    // op_idx so per-op total reflects total GPU work for that op invocation.
    // calls is incremented once per (op_idx, drain) when host-clock didn't
    // already do it for this run.
    std::unordered_set<uint32_t> bumped;
    for (auto& r : recs) {
        if (r.op_idx == UINT32_MAX) continue;
        if (r.op_idx >= profile.size()) continue;
        int64_t dur_ns = (int64_t)(r.end_ns - r.start_ns);
        profile[r.op_idx].total_ticks += (int64_t)(dur_ns * ns_to_ticks);
        if (!host_clock_fired && bumped.insert(r.op_idx).second) {
            profile[r.op_idx].calls++;
        }
    }
}
#endif

// If this op runs on CPU but reads tensors that were last written by a
// GPU backend (device buffer present + last_write_evt set), download them.
// Cheap when no GPU backend is active (inner loop is empty). Idempotent.
static inline void sync_inputs_to_host(context_t* ctx, operator_t* op) {
    if (op->device_tag != 0) return;          // GPU op — input stays on device
    for (auto& b : ctx->backends) {
        if (!b.data || !b.writeback_fn) continue;
        for (auto* in : op->inputs)
            if (in) b.writeback_fn(b.data, in);
    }
}

// After run_graph completes, make sure all graph outputs are on the host.
static inline void sync_graph_outputs_to_host(context_t* ctx) {
    bool any_backend = false;
    for (auto& b : ctx->backends)
        if (b.data && b.writeback_fn) { any_backend = true; break; }
    if (!any_backend) return;
    for (auto& name : ctx->graph_outputs) {
        tensor_t* t = ctx->search_tensor(name);
        if (!t) continue;
        for (auto& b : ctx->backends)
            if (b.data && b.writeback_fn) b.writeback_fn(b.data, t);
    }
}

// Profiler policies for run_graph: real measurement vs no-op.
struct profiler_noop {
    void begin() {}
    void end(int) {}
};

// Platform-specific high-resolution timer with overhead calibration.
// Uses raw QPC on Windows, steady_clock elsewhere.
// Subtracts the measured per-call overhead so per-op sums match wall-clock.
static int64_t calibrate_timer_overhead() {
    int64_t samples[101];
    for (int i = 0; i < 101; i++) {
#ifdef _WIN32
        int64_t a, b;
        QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&a));
        QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&b));
        samples[i] = b - a;
#else
        auto a = std::chrono::steady_clock::now();
        auto b = std::chrono::steady_clock::now();
        samples[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
#endif
    }
    std::sort(samples, samples + 101);
    return samples[50];
}

struct profiler_clock {
    std::vector<node_profile_t>& profile;
    int64_t overhead;
    int64_t t0;

    profiler_clock(std::vector<node_profile_t>& p)
        : profile(p), overhead(calibrate_timer_overhead()), t0(0) {}

    void begin() {
#ifdef _WIN32
        QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&t0));
#else
        t0 = std::chrono::steady_clock::now().time_since_epoch().count();
#endif
    }
    void end(int i) {
#ifdef _WIN32
        int64_t t1;
        QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&t1));
#else
        int64_t t1 = std::chrono::steady_clock::now().time_since_epoch().count();
#endif
        int64_t ticks = (t1 - t0) - overhead;
        if (ticks < 0) ticks = 0;
        profile[i].total_ticks += ticks;
        profile[i].calls++;
    }
};

#ifdef NNR_PROFILE_REORDERS
struct reorder_stats_t {
    int nhwc_count = 0;
    int blocked_count = 0;
    size_t nhwc_bytes = 0;
    size_t blocked_bytes = 0;
    int64_t nhwc_ticks = 0;
    int64_t blocked_ticks = 0;
    int64_t inferences = 0;
    ~reorder_stats_t() {
        if (inferences == 0) return;
#ifdef _WIN32
        int64_t qpc_freq;
        QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&qpc_freq));
        double ns_per_tick = 1e9 / qpc_freq;
#else
        double ns_per_tick = 1.0;
#endif
        double nhwc_ms = (nhwc_ticks * ns_per_tick) / 1e6 / inferences;
        double blocked_ms = (blocked_ticks * ns_per_tick) / 1e6 / inferences;
        fprintf(stderr,
            "[reorder] infs=%lld  NHWC->NCHW: %d calls/inf, %.2f MB/inf, %.3f ms/inf  "
            "BLOCKED->NCHW: %d calls/inf, %.2f MB/inf, %.3f ms/inf  total %.3f ms/inf\n",
            (long long)inferences,
            (int)(nhwc_count / inferences), nhwc_bytes/1024.0/1024.0/inferences, nhwc_ms,
            (int)(blocked_count / inferences), blocked_bytes/1024.0/1024.0/inferences, blocked_ms,
            nhwc_ms + blocked_ms);
    }
};
static reorder_stats_t g_reorder_stats;
static inline int64_t now_tick() {
#ifdef _WIN32
    LARGE_INTEGER li; QueryPerformanceCounter(&li);
    return li.QuadPart;
#else
    return std::chrono::steady_clock::now().time_since_epoch().count();
#endif
}
#endif

template <typename Profiler>
static bool run_graph_impl(context_t* ctx, Profiler prof)
{
    auto& nodes = ctx->graph->nodes;
    const int num_nodes = static_cast<int>(nodes.size());
    const bool have_plan = ctx->optimizer->plan_built;

#ifdef NNR_ENABLE_WEBGPU
    // Allocate / resize the per-op timestamp QuerySet to cover this graph's
    // node count. No-op unless NNR_WEBGPU_OP_TIMINGS is set and the device
    // exposes TimestampQuery. See backend/webgpu/profiler.cpp.
    nnr::webgpu::op_profiler_begin_run(num_nodes);
#endif
#ifdef NNR_PROFILE_REORDERS
    g_reorder_stats.inferences++;
#endif

    // Reset NHWC/BLOCKED_16 formats that boundary reorder may have changed
    // to NCHW. Only reset once the plan is built — before that, tensor data
    // is still NCHW and resetting formats to BLOCKED_16 would cause NCHWc
    // Conv to misinterpret NCHW data as BLOCKED_16.
    if (have_plan)
        ctx->optimizer->reset_formats();

    // CUDA Graph: opportunistic capture + replay. When the plan is stable and
    // every actionable op is on CUDA, replay a previously-captured cudaGraphExec
    // instead of the op-by-op exec loop — or start capturing on this run.
    // no-op when NNR_USE_CUDA=OFF or no CUDA backend is active.
#if defined(NNR_USE_CUDA)
    int cuda_mode = 0;  // 0=normal, 1=replaying, 2=capturing
    if (have_plan && !ctx->optimizer->exec_steps.empty() && !ctx->shapes_dirty) {
        bool all_gpu = true;
        for (auto& step : ctx->optimizer->exec_steps) {
            auto* n = step.op;
            if (step.scroll_seg == -2) continue;
            if (n->folded) continue;
            if (n->device_tag == 0) { all_gpu = false; break; }
        }
        cuda_mode = cuda_begin_run(ctx, all_gpu, ctx->shapes_dirty);
#if defined(NNR_USE_CUPTI)
        nnr::gpu::cupti_profiler().init();
#endif
        if (cuda_mode == 1) {
            cuda_replay_and_sync(ctx);
            cuda_end_run(ctx, cuda_mode);
#if defined(NNR_USE_CUPTI)
            // Replay mode: fast-path skipped, so host-clock prof.end never
            // fired for this run. CUPTI canonical-sequence matching (built
            // up during the slow-path warmup) recovers op_idx, and we bump
            // calls inside the drain to keep accounting consistent.
            drain_cupti_into_profile(ctx->profile, /*host_clock_fired=*/false);
#endif
            goto graph_output_reorder;
        }
    }
#endif

    // Fast path: pre-compiled execution steps (after plan is built).
    // Iterates only actionable steps — no FOLDED/SCROLL_INSIDE scanning.
    // Skip fast path when shapes are dirty — need slow path for reshape.
    if (have_plan && !ctx->optimizer->exec_steps.empty() && !ctx->shapes_dirty) {
        using EF = graph_optimizer_t;
        for (auto& step : ctx->optimizer->exec_steps) {
            auto* n = step.op;

            // SKIP: forward data pointer
            if (step.scroll_seg == -2) {
                n->outputs[0]->data = n->inputs[0]->data;
                n->outputs[0]->owns_data = false;
                n->outputs[0]->format = n->inputs[0]->format;
                continue;
            }

            // SCROLL_START: execute scroll segment
            if (step.scroll_seg >= 0) {
                auto& seg = ctx->optimizer->scroll_segments[step.scroll_seg];
                arena_scope_t scroll_scope(ctx->arena);
                prof.begin();
                { NNR_TRACE_SCOPE("scroll", "scroll_segment");
                if (!graph_optimizer_t::exec_scroll_segment(ctx, seg.start, seg.end, seg.strip_height)) {
                    // Fallback: execute segment nodes individually
                    auto& nodes = ctx->graph->nodes;
                    for (int si = seg.start; si < seg.end; ++si) {
                        auto* sn = nodes[si];
                        if (sn->skip) {
                            if (!sn->inputs.empty() && !sn->outputs.empty() && sn->inputs[0] && sn->outputs[0]) {
                                sn->outputs[0]->data = sn->inputs[0]->data;
                                sn->outputs[0]->owns_data = false;
                            }
                        } else if (!sn->folded) {
                            if (!sn->exec()) return false;
                        }
                    }
                }
                } prof.end(step.node_idx);
                continue;
            }

            // EXEC: layout boundary checks + execute
            bool wants_nhwc = step.flags & EF::FLAG_WANTS_NHWC;
            bool wants_blocked = step.flags & EF::FLAG_WANTS_BLOCKED;
            bool is_layout_all = step.flags & EF::FLAG_LAYOUT_ALL;
            bool has_nhwc_input = false;
            bool has_blocked_input = false;

            // NHWC boundary reorder
            if (!wants_nhwc) {
                for (auto* t : n->inputs) {
                    if (t && t->format == memory_layout_t::NHWC
                        && t->ndim == 4 && t->type == NNR_DATA_TYPE_FLOAT32) {
                        has_nhwc_input = true;
                        break;
                    }
                }
                if (has_nhwc_input && !is_layout_all) {
                    NNR_TRACE_SCOPE("reorder", "NHWC->NCHW");
#ifdef NNR_PROFILE_REORDERS
                    int64_t rt0 = now_tick();
#endif
                    for (auto* t : n->inputs) {
                        if (t && t->format == memory_layout_t::NHWC
                            && t->ndim == 4 && t->type == NNR_DATA_TYPE_FLOAT32) {
                            size_t sz = t->ndata * sizeof(float);
                            if (sz > ctx->workspace_size) ensure_workspace(ctx, sz);
                            reorder_inplace((float*)t->data, t->dims[0], t->dims[1],
                                t->dims[2], t->dims[3],
                                memory_layout_t::NHWC, memory_layout_t::NCHW,
                                (float*)ctx->workspace);
                            t->format = memory_layout_t::NCHW;
#ifdef NNR_PROFILE_REORDERS
                            g_reorder_stats.nhwc_count++;
                            g_reorder_stats.nhwc_bytes += sz;
#endif
                        }
                    }
#ifdef NNR_PROFILE_REORDERS
                    g_reorder_stats.nhwc_ticks += now_tick() - rt0;
#endif
                }
            }

            // BLOCKED_16 boundary reorder
            if (!wants_blocked) {
                for (auto* t : n->inputs) {
                    if (t && t->format == NATIVE_BLOCKED_FMT && NATIVE_BLOCKED_FMT != memory_layout_t::NCHW
                        && t->ndim == 4 && t->type == NNR_DATA_TYPE_FLOAT32) {
                        has_blocked_input = true;
                        break;
                    }
                }
                if (has_blocked_input) {
                    bool propagate = is_layout_all && !(step.flags & EF::FLAG_HAS_BROADCAST);
                    // Terminal blocked consumer: node advertises NCHWc but its
                    // output is NCHW. Skip the input reorder — the node's own
                    // exec() (exec_nchwc_blocked) handles the blocked input
                    // and emits NCHW output via workspace + nchwc_to_nchw.
                    bool consumes_blocked = (n->layout_mask & LAYOUT_NATIVE_BLOCKED) != 0;
                    if (!propagate && !consumes_blocked) {
                        NNR_TRACE_SCOPE("reorder", "BLOCKED16->NCHW");
#ifdef NNR_PROFILE_REORDERS
                        int64_t rt0 = now_tick();
#endif
                        for (auto* t : n->inputs) {
                            if (t && t->format == NATIVE_BLOCKED_FMT && NATIVE_BLOCKED_FMT != memory_layout_t::NCHW
                                && t->ndim == 4 && t->type == NNR_DATA_TYPE_FLOAT32) {
                                int N = t->dims[0], C = t->dims[1], H = t->dims[2], W = t->dims[3];
                                size_t sz = (size_t)N * nchwc_padded_channels(C, NATIVE_BLOCK) * H * W * sizeof(float);
                                if (sz > ctx->workspace_size) ensure_workspace(ctx, sz);
                                nchwc_to_nchw((float*)ctx->workspace, (float*)t->data, N, C, H, W, NATIVE_BLOCK);
                                memcpy(t->data, ctx->workspace, t->ndata * sizeof(float));
                                t->format = memory_layout_t::NCHW;
#ifdef NNR_PROFILE_REORDERS
                                g_reorder_stats.blocked_count++;
                                g_reorder_stats.blocked_bytes += sz;
#endif
                            }
                        }
#ifdef NNR_PROFILE_REORDERS
                        g_reorder_stats.blocked_ticks += now_tick() - rt0;
#endif
                        has_blocked_input = false;  // reordered, don't propagate
                    }
                }
            }

            sync_inputs_to_host(ctx, n);
#if defined(NNR_USE_CUPTI)
            nnr::gpu::cupti_profiler().push_op((uint32_t)step.node_idx);
#endif
            prof.begin();
#ifdef NNR_ENABLE_WEBGPU
            nnr::webgpu::op_profiler_op_begin(step.node_idx);
#endif
            { NNR_TRACE_SCOPE("op", n->op_type);
#ifdef NNR_ENABLE_WEBGPU
            nnr::webgpu::sync_inputs_if_cpu_op(n);
#endif
            if (!n->exec()) {
                fprintf(stderr, "[run_graph] exec() failed at node %d (%.*s)\n",
                        step.node_idx, (int)n->op_type.size(), n->op_type.data());
                return false;
            }
#ifdef NNR_ENABLE_WEBGPU
            nnr::webgpu::sync_outputs_if_cpu_op(n);
#endif
            }
#ifdef NNR_ENABLE_WEBGPU
            nnr::webgpu::op_profiler_op_end(step.node_idx);
#endif
            prof.end(step.node_idx);
#if defined(NNR_USE_CUPTI)
            nnr::gpu::cupti_profiler().pop_op();
#endif

            // Propagate NHWC/BLOCKED_16 through layout-agnostic ops
            if (has_nhwc_input && !wants_nhwc && is_layout_all) {
                for (auto* t : n->outputs)
                    if (t && t->ndim == 4 && t->type == NNR_DATA_TYPE_FLOAT32)
                        t->format = memory_layout_t::NHWC;
            }
            if (has_blocked_input && !wants_blocked && is_layout_all) {
                for (auto* t : n->outputs)
                    if (t && t->ndim == 4 && t->type == NNR_DATA_TYPE_FLOAT32)
                        t->format = NATIVE_BLOCKED_FMT;
            }
        }
        goto graph_output_reorder;
    }

    // Slow path: first run before plan is built, or shapes_dirty.
    {
    auto action_for = [&](int i) -> node_action_t {
        if (have_plan) return ctx->optimizer->plan[i];
        auto* n = nodes[i];
        if (n->skip) return node_action_t::SKIP;
        if (n->folded) return node_action_t::FOLDED;
        return node_action_t::EXEC;
    };

    for (int i = 0; i < num_nodes; ++i) {
        auto* n = nodes[i];
        if (ctx->shapes_dirty && !n->skip) {
            if (!n->reshape()) return false;
            size_t ws = n->workspace_size();
            if (ws > ctx->workspace_size)
                ensure_workspace(ctx, ws);
        }
        auto action = action_for(i);

        bool wants_nhwc = !n->outputs.empty() && n->outputs[0]
            && n->outputs[0]->format == memory_layout_t::NHWC;
        bool wants_blocked = !n->outputs.empty() && n->outputs[0]
            && n->outputs[0]->format == NATIVE_BLOCKED_FMT
            && NATIVE_BLOCKED_FMT != memory_layout_t::NCHW;
        bool has_nhwc_input = false;
        bool has_blocked_input = false;
        if (action == node_action_t::EXEC && !wants_nhwc) {
            for (auto* t : n->inputs) {
                if (t && t->format == memory_layout_t::NHWC && t->ndim == 4) {
                    has_nhwc_input = true;
                    break;
                }
            }
            if (has_nhwc_input && n->layout_mask == LAYOUT_ALL) {
            } else if (has_nhwc_input) {
                for (auto* t : n->inputs) {
                    if (t && t->format == memory_layout_t::NHWC
                        && t->ndim == 4 && t->type == NNR_DATA_TYPE_FLOAT32) {
                        size_t sz = t->ndata * sizeof(float);
                        if (sz > ctx->workspace_size)
                            ensure_workspace(ctx, sz);
                        reorder_inplace((float*)t->data, t->dims[0], t->dims[1],
                            t->dims[2], t->dims[3],
                            memory_layout_t::NHWC, memory_layout_t::NCHW,
                            (float*)ctx->workspace);
                        t->format = memory_layout_t::NCHW;
                    }
                }
            }
        }

        if (action == node_action_t::EXEC && !wants_blocked) {
            for (auto* t : n->inputs) {
                if (t && t->format == NATIVE_BLOCKED_FMT && NATIVE_BLOCKED_FMT != memory_layout_t::NCHW
                    && t->ndim == 4 && t->type == NNR_DATA_TYPE_FLOAT32) {
                    has_blocked_input = true;
                    break;
                }
            }
            if (has_blocked_input && n->layout_mask == LAYOUT_ALL) {
                bool has_broadcast = false;
                for (auto* t : n->inputs) {
                    if (t && t->ndim > 0 && t->ndim < 4 && t->ndata > 1) {
                        has_broadcast = true;
                        break;
                    }
                }
                if (has_broadcast) {
                    has_blocked_input = false;
                    for (auto* t : n->inputs) {
                        if (t && t->format == NATIVE_BLOCKED_FMT && NATIVE_BLOCKED_FMT != memory_layout_t::NCHW
                            && t->ndim == 4 && t->type == NNR_DATA_TYPE_FLOAT32) {
                            int N = t->dims[0], C = t->dims[1], H = t->dims[2], W = t->dims[3];
                            size_t sz = (size_t)N * nchwc_padded_channels(C, NATIVE_BLOCK) * H * W * sizeof(float);
                            if (sz > ctx->workspace_size)
                                ensure_workspace(ctx, sz);
                            nchwc_to_nchw((float*)ctx->workspace, (float*)t->data, N, C, H, W, NATIVE_BLOCK);
                            memcpy(t->data, ctx->workspace, t->ndata * sizeof(float));
                            t->format = memory_layout_t::NCHW;
                        }
                    }
                }
            } else if (has_blocked_input && !(n->layout_mask & LAYOUT_NATIVE_BLOCKED)) {
                // Terminal blocked consumers (layout_mask has LAYOUT_NATIVE_BLOCKED)
                // skip this reorder — exec_nchwc_blocked handles the input directly.
                for (auto* t : n->inputs) {
                    if (t && t->format == NATIVE_BLOCKED_FMT && NATIVE_BLOCKED_FMT != memory_layout_t::NCHW
                        && t->ndim == 4 && t->type == NNR_DATA_TYPE_FLOAT32) {
                        int N = t->dims[0], C = t->dims[1], H = t->dims[2], W = t->dims[3];
                        size_t sz = (size_t)N * nchwc_padded_channels(C, NATIVE_BLOCK) * H * W * sizeof(float);
                        if (sz > ctx->workspace_size)
                            ensure_workspace(ctx, sz);
                        nchwc_to_nchw((float*)ctx->workspace, (float*)t->data, N, C, H, W, NATIVE_BLOCK);
                        memcpy(t->data, ctx->workspace, t->ndata * sizeof(float));
                        t->format = memory_layout_t::NCHW;
                    }
                }
            }
        }

        switch (action) {
        case node_action_t::SKIP:
            if (!n->inputs.empty() && !n->outputs.empty() && n->inputs[0] && n->outputs[0]) {
                n->outputs[0]->data = n->inputs[0]->data;
                n->outputs[0]->owns_data = false;
                n->outputs[0]->format = n->inputs[0]->format;
            }
            break;
        case node_action_t::FOLDED:
            break;
        case node_action_t::SCROLL_START: {
            int seg_idx = ctx->optimizer->plan_scroll_seg[i];
            auto& seg = ctx->optimizer->scroll_segments[seg_idx];
            arena_scope_t scroll_scope(ctx->arena);
            prof.begin();
            if (!graph_optimizer_t::exec_scroll_segment(ctx, seg.start, seg.end, seg.strip_height)) {
                // Fallback: execute segment nodes individually
                for (int si = seg.start; si < seg.end; ++si) {
                    auto* sn = nodes[si];
                    if (sn->skip) {
                        if (!sn->inputs.empty() && !sn->outputs.empty() && sn->inputs[0] && sn->outputs[0]) {
                            sn->outputs[0]->data = sn->inputs[0]->data;
                            sn->outputs[0]->owns_data = false;
                        }
                    } else if (!sn->folded) {
                        if (!sn->exec()) return false;
                    }
                }
            }
            prof.end(i);
            i = seg.end - 1;
            break;
        }
        case node_action_t::SCROLL_INSIDE:
            break;
        case node_action_t::EXEC:
            sync_inputs_to_host(ctx, n);
            prof.begin();
#ifdef NNR_ENABLE_WEBGPU
            nnr::webgpu::op_profiler_op_begin(i);
#endif
            { NNR_TRACE_SCOPE("op", n->op_type);
#ifdef NNR_ENABLE_WEBGPU
            nnr::webgpu::sync_inputs_if_cpu_op(n);
#endif
            if (!n->exec()) {
                fprintf(stderr, "[run_graph] exec() failed at node %d (%.*s)\n",
                        i, (int)n->op_type.size(), n->op_type.data());
                return false;
            }
#ifdef NNR_ENABLE_WEBGPU
            nnr::webgpu::sync_outputs_if_cpu_op(n);
#endif
            }
#ifdef NNR_ENABLE_WEBGPU
            nnr::webgpu::op_profiler_op_end(i);
#endif
            prof.end(i);
            if (has_nhwc_input && !wants_nhwc && n->layout_mask == LAYOUT_ALL) {
                for (auto* t : n->outputs)
                    if (t && t->ndim == 4)
                        t->format = memory_layout_t::NHWC;
            }
            if (has_blocked_input && !wants_blocked && n->layout_mask == LAYOUT_ALL) {
                for (auto* t : n->outputs)
                    if (t && t->ndim == 4 && t->type == NNR_DATA_TYPE_FLOAT32)
                        t->format = NATIVE_BLOCKED_FMT;
            }
            break;
        }
    }
    } // end slow path

graph_output_reorder:

#if defined(NNR_USE_CUDA)
    // If we were capturing this run, finalize the graph now, then replay it once
    // to actually produce outputs (capture only records — kernel launches under
    // capture are not executed).
    if (cuda_mode == 2) {
        cuda_end_run(ctx, cuda_mode);
        cuda_replay_and_sync(ctx);
    } else if (cuda_mode != 1) {
        // normal mode — still increment run_count for capture eligibility
        cuda_end_run(ctx, cuda_mode);
    }
#if defined(NNR_USE_CUPTI)
    // Drain CUPTI records for the just-completed run (normal or capture).
    // Fast/slow path ran exec() with prof.begin/end and CUPTI push/pop, so
    // host-clock already bumped calls — the drain only adds GPU time.
    if (cuda_mode != 1) {
        drain_cupti_into_profile(ctx->profile, /*host_clock_fired=*/true);
    }
#endif
#endif

    // Bring GPU-written outputs back to host memory before the caller sees them.
    // No-op when no GPU backend is active.
    sync_graph_outputs_to_host(ctx);

    // Resolve output tensors once, lazily. `search_tensor` does a linear
    // scan over `ctx->map`; caching the pointer eliminates O(|outputs| ×
    // |map|) work on every inference.
    if (ctx->graph_output_tensors.size() != ctx->graph_outputs.size()) {
        ctx->graph_output_tensors.clear();
        ctx->graph_output_tensors.reserve(ctx->graph_outputs.size());
        for (auto& name : ctx->graph_outputs)
            ctx->graph_output_tensors.push_back(ctx->search_tensor(name));
    }

    // Reorder NHWC graph outputs to NCHW so caller always sees NCHW.
    // Internal ops may leave outputs in NHWC for efficiency, but the
    // public API contract is NCHW output.  Non-float types (uint8/int8
    // for QDQ graphs, fp16 for attention) go through the byte-generic
    // path since the SIMD nhwc_to_nchw is float-only.
    for (tensor_t* t : ctx->graph_output_tensors) {
        if (t && t->format == memory_layout_t::NHWC && t->ndim == 4) {
            size_t elem_sz = data_type_sizeof(t->type);
            size_t sz = t->ndata * elem_sz;
            if (sz > ctx->workspace_size)
                ensure_workspace(ctx, sz);
#ifdef DEBUG_LAYOUT
            auto rt0 = std::chrono::high_resolution_clock::now();
#endif
            if (t->type == NNR_DATA_TYPE_FLOAT32) {
                reorder_inplace((float*)t->data, t->dims[0], t->dims[1],
                    t->dims[2], t->dims[3],
                    memory_layout_t::NHWC, memory_layout_t::NCHW,
                    (float*)ctx->workspace);
            } else {
                reorder_inplace_bytes(t->data, t->dims[0], t->dims[1],
                    t->dims[2], t->dims[3],
                    memory_layout_t::NHWC, memory_layout_t::NCHW,
                    ctx->workspace, elem_sz);
            }
#ifdef DEBUG_LAYOUT
            auto rt1 = std::chrono::high_resolution_clock::now();
            fprintf(stderr, "[reorder] graph output NHWC->NCHW [%d,%d,%d,%d] type=%d %.0f us\n",
                t->dims[0], t->dims[1], t->dims[2], t->dims[3], (int)t->type,
                std::chrono::duration<double, std::micro>(rt1 - rt0).count());
#endif
            t->format = memory_layout_t::NCHW;
        }
    }

    // Reorder BLOCKED_16 graph outputs to NCHW.
    for (tensor_t* t : ctx->graph_output_tensors) {
        if (t && t->format == NATIVE_BLOCKED_FMT && NATIVE_BLOCKED_FMT != memory_layout_t::NCHW
            && t->ndim == 4 && t->type == NNR_DATA_TYPE_FLOAT32) {
            int N = t->dims[0], C = t->dims[1], H = t->dims[2], W = t->dims[3];
            size_t sz = (size_t)N * nchwc_padded_channels(C, NATIVE_BLOCK) * H * W * sizeof(float);
            if (sz > ctx->workspace_size)
                ensure_workspace(ctx, sz);
            nchwc_to_nchw((float*)ctx->workspace, (float*)t->data, N, C, H, W, NATIVE_BLOCK);
            memcpy(t->data, ctx->workspace, t->ndata * sizeof(float));
            t->format = memory_layout_t::NCHW;
        }
    }

    return true;
}

static bool run_graph(context_t* ctx)
{
    // The optimizer can splice nodes into ctx->graph->nodes after
    // enable_profiling() sized the profile vector (e.g. insert_reorders
    // synthesizes Reorder ops at layout boundaries). Resize defensively
    // so profile[i] stays in bounds for every active node.
    if (!ctx->profile.empty() && ctx->graph
            && ctx->profile.size() < ctx->graph->nodes.size()) {
        ctx->profile.resize(ctx->graph->nodes.size());
    }
    if (ctx->profile.empty())
        return run_graph_impl(ctx, profiler_noop{});
    else
        return run_graph_impl(ctx, profiler_clock(ctx->profile));
}

// First-pass for prepare(): reshape all nodes, exec cheap ops for shape propagation,
// exec and fold only fully-constant nodes.
//
// Some ops' reshape() reads input DATA (not just shape): Reshape reads its shape
// input, Slice reads starts/ends/axes, etc.  These are often fed by Shape→Gather
// chains that compute static values from known tensor shapes.  We exec all cheap
// (non-compute-intensive) ops so that such chains produce valid data for reshape(),
// while skipping expensive ops (Conv, ConvTranspose, MatMul, Gemm, LSTM, GRU, RNN).
//
// Nodes with ALL inputs from the constants set are also marked folded so that
// subsequent run() calls skip them entirely.
//
// prune_segments() calls run_for_warmup() to exec the skipped expensive ops before
// timing scroll-vs-layer trials.
static bool fold_run(context_t* ctx)
{
    auto& nodes = ctx->graph->nodes;

    // Ops that must not be executed (non-deterministic / side effects)
    auto is_unsafe = [](std::string_view op) -> bool {
        return op == "RandomNormal" || op == "RandomNormalLike" ||
               op == "RandomUniform" || op == "RandomUniformLike" ||
               op == "Multinomial" || op == "Bernoulli";
    };
    // Ops whose exec() is expensive enough to skip.  Their shapes are
    // still propagated via reshape(); run_for_warmup() fills their data
    // before prune_segments() needs it.
    auto is_expensive = [](std::string_view op) -> bool {
        return op == "Conv"        || op == "ConvTranspose"  ||
               op == "MatMul"      || op == "Gemm"           ||
               op == "ConvInteger" || op == "MatMulInteger"  ||
               op == "QLinearConv" || op == "QLinearMatMul"  ||
               op == "LSTM"        || op == "GRU"            || op == "RNN" ||
               op == "Loop"        || op == "If"             || op == "Scan" ||
               op == "NonMaxSuppression" || op == "TopK"     || op == "RoiAlign";
    };

    std::unordered_set<const tensor_t*> constants;
    constants.reserve(ctx->map.size());
    for (auto& [name, tensor] : ctx->map)
        if (ctx->initializer_names.count(name))
            constants.insert(tensor);

    // Helper: build a transient CPU operator that mirrors `n`'s identity.
    // Used as a per-iteration shadow for non-CPU ops so reshape() and
    // exec() during fold_run write to host-side tensor->data, which
    // downstream reshape() (TopK's K, Reshape's shape, Slice's bounds…)
    // and constant folding read on the CPU side. Returns nullptr if no
    // CPU resolver exists or init() fails. Pool-allocated; not freed
    // here (lifetime tied to ctx->attr_pool).
    auto build_cpu_shadow = [&](operator_t* n) -> operator_t* {
        operator_t* s = solve_operator(n->op_type, n->opset,
                                       ctx->attr_pool, backend_t::CPU);
        if (!s) return nullptr;
        s->ctx       = ctx;
        s->opset     = n->opset;
        s->op_type   = n->op_type;
        s->node_name = n->node_name;
        s->domain    = n->domain;
        s->inputs    = n->inputs;
        s->outputs   = n->outputs;
        s->attrs     = n->attrs;
        if (!s->init()) return nullptr;
        return s;
    };

    for (auto* n : nodes) {
        // Skip fused/folded/inactive nodes — their inputs may be invalid
        if (n->skip || n->folded) continue;

        // Mirror CUDA's "no GPU during fold_run" gate
        // (cuda_backend.cpp:14-21): GPU exec writes only to device memory,
        // but fold_run's downstream consumers — reshape() data reads and
        // constant folding — read tensor->data on the host. CUDA achieves
        // this via a per-op CPU `fallback` field; for backends without one
        // (today: WebGPU), we shadow non-CPU ops with a transient CPU
        // operator for the fold_run pass. The original GPU op stays in
        // nodes[] for runtime use.
        operator_t* eff = n;
        if (n->resolved_backend != static_cast<uint8_t>(backend_t::CPU)) {
            if (operator_t* shadow = build_cpu_shadow(n))
                eff = shadow;
        }

        if (!eff->reshape()) {
            // If the original was non-CPU and the shadow path didn't apply
            // (no CPU resolver, or init failed), try the legacy reshape-
            // failure CPU fallback once and swap permanently. Some reshape
            // failures are backend-specific (WebGPU Gather rejecting int64
            // data, for instance, where the type is only known at reshape
            // time because the producer is a Shape op). Replacing the op
            // with a CPU equivalent keeps prepare() going instead of
            // killing the whole graph.
            bool fell_back = false;
            if (eff == n && n->resolved_backend != static_cast<uint8_t>(backend_t::CPU)) {
                if (operator_t* cpu_n = build_cpu_shadow(n)) {
                    if (cpu_n->reshape()) {
                        // Swap in place; preserves iteration order.
                        for (size_t ni = 0; ni < nodes.size(); ++ni) {
                            if (nodes[ni] == n) { nodes[ni] = cpu_n; n = cpu_n; eff = cpu_n; break; }
                        }
                        fell_back = true;
                    }
                }
            }
            if (!fell_back) {
                std::fprintf(stderr, "nnr: reshape failed: %.*s (%.*s)\n",
                             (int)n->op_type.size(), n->op_type.data(),
                             (int)n->node_name.size(), n->node_name.data());
                for (size_t ii = 0; ii < n->inputs.size(); ii++) {
                    auto* t = n->inputs[ii];
                    if (!t) { std::fprintf(stderr, "  input[%zu]: null\n", ii); continue; }
                    std::fprintf(stderr, "  input[%zu]: [", ii);
                    for (int d = 0; d < t->ndim; d++) std::fprintf(stderr, "%s%d", d?",":"", t->dims[d]);
                    auto tn = data_type_tostring(t->type);
                    std::fprintf(stderr, "] %.*s%s\n",
                        (int)tn.size(), tn.data(),
                        t->data ? "" : " (no data)");
                }
                for (size_t ii = 0; ii < n->outputs.size(); ii++) {
                    auto* t = n->outputs[ii];
                    if (!t) continue;
                    std::fprintf(stderr, "  output[%zu]: [", ii);
                    for (int d = 0; d < t->ndim; d++) std::fprintf(stderr, "%s%d", d?",":"", t->dims[d]);
                    auto tn = data_type_tostring(t->type);
                    std::fprintf(stderr, "] %.*s\n", (int)tn.size(), tn.data());
                }
                return false;
            }
        }
        // If we ran a CPU shadow above, the original GPU op never had its
        // own reshape() called — backend-specific state (WebGPU pipelines,
        // uniform buffers, output buffer allocations) wouldn't be set up
        // for runtime exec(). The shadow's reshape has populated output
        // dims/types so the GPU op's reshape can complete; do it now,
        // before any control-flow `continue` skips past it.
        if (eff != n) {
            if (!n->reshape()) {
                // Backend reshape rejected something the shadow accepted
                // (e.g. dtype/layout the GPU kernel doesn't support).
                // Demote permanently to the shadow so runtime is correct.
                for (size_t ni = 0; ni < nodes.size(); ++ni) {
                    if (nodes[ni] == n) { nodes[ni] = eff; n = eff; break; }
                }
            }
        }

        // Workspace must satisfy both shadow (CPU exec, this pass) and
        // original (GPU exec, runtime). ensure_workspace is grow-only.
        size_t ws = eff->workspace_size();
        if (eff != n) {
            size_t ws_n = n->workspace_size();
            if (ws_n > ws) ws = ws_n;
        }
        if (ws > ctx->workspace_size)
            ensure_workspace(ctx, ws);

        if (is_unsafe(n->op_type)) continue;
        if (is_expensive(n->op_type)) continue;

        bool all_const = true;
        for (auto* t : n->inputs)
            if (t && !constants.count(t)) { all_const = false; break; }

        // Check for null/uninitialized data on any input or output — skip exec entirely.
        // Catches: un-executed expensive ops (data=nullptr), control-flow outputs (type=UNDEFINED).
        bool has_null_data = false;
        for (auto* t : n->inputs)
            if (t && (!t->data || t->type == NNR_DATA_TYPE_UNDEFINED)) { has_null_data = true; break; }
        for (auto* t : n->outputs)
            if (t && !t->data && t->type != NNR_DATA_TYPE_UNDEFINED) { has_null_data = true; break; }

        if (all_const && !has_null_data) {
            // Fully constant: exec, fold, exclude from memory planning.
            if (!eff->exec()) {
                std::fprintf(stderr, "nnr: constant fold failed: %.*s (%.*s)\n",
                             (int)n->op_type.size(), n->op_type.data(),
                             (int)n->node_name.size(), n->node_name.data());
                return false;
            }
            n->folded = true;
            for (auto* t : n->outputs)
                if (t) {
                    constants.insert(t);
                    ctx->memory_plan_excluded.insert(t->name);
                }
        } else if (!has_null_data) {
            // Non-constant cheap op: exec for shape-data propagation
            // (e.g. Shape→Gather→ReduceMin chains that feed into Reshape/TopK).
            // Output may be garbage for activation paths but correct for
            // shape paths.  Failures are ignored — garbage floats are fine.
            eff->exec();
        }
    }
    return true;
}

bool context_t::prepare()
{
    if (!graph) return false;

    // PREPROCESS-level passes (graph rewrites like LayerNorm + Gelu fusion)
    // must run before fold_run so constant folding sees the fused graph.
    // Without this, bench.exe misses graph rewrites that run() would apply
    // via its first-run preprocess call.
    optimizer->preprocess(this);

    // Reshape all nodes and exec only constant-foldable nodes.
    // Non-constant ops are skipped here; prune_segments() will warm them up
    // via run_for_warmup() if scroll timing trials require valid tensor data.
    if (shapes_dirty) {
        if (!fold_run(this))
            return false;
        shapes_dirty = false;
    }

    // Run optimization passes (fusion, scroll detection, prune_segments, plan)
    optimizer->optimize(this);
    optimizer->build_plan(this);
    // Ensure workspace covers layout reorder needs
    if (optimizer->layout_reorder_ws > workspace_size)
        ensure_workspace(this, optimizer->layout_reorder_ws);

    // Memory planning
    if (memory_planning_enabled && !memory_planned) {
        planner.analyze(this);
        planner.plan();
        planner.apply();
        memory_planned = true;
    }

    return true;
}

bool context_t::run_for_warmup()
{
    // Execute graph for warmup (fills tensor data for scroll timing).
    // Skip nodes whose inputs have null data or undefined type — these
    // depend on un-executed expensive/control-flow ops and would crash.
    auto& nodes = graph->nodes;
    for (auto* n : nodes) {
        if (n->skip || n->folded) continue;
        bool safe = true;
        for (auto* t : n->inputs)
            if (t && (!t->data || t->type == NNR_DATA_TYPE_UNDEFINED)) { safe = false; break; }
        for (auto* t : n->outputs)
            if (t && !t->data && t->type != NNR_DATA_TYPE_UNDEFINED) { safe = false; break; }
        if (safe) n->exec();
    }
#ifdef NNR_ENABLE_WEBGPU
    // Flush any WebGPU dispatches queued during warmup so subsequent
    // scroll-timing trials see populated tensor data, not pending work.
    nnr::webgpu::flush_encoder();
#endif
    return true;
}

bool context_t::run()
{
    if (shapes_dirty && memory_planned) {
        planner.release();
        memory_planned = false;
    }

    // first_run: very first execution (graph not yet optimized).
    // shapes_changed: inputs were re-shaped but plan already exists.
    bool first_run = shapes_dirty && !optimizer->plan_built;
    bool shapes_changed = shapes_dirty && optimizer->plan_built;

    // Pre-execution graph rewriting (decompose unsupported composite ops)
    if (first_run)
        optimizer->preprocess(this);

    // Dynamic shape change: re-run fold_run to reshape all nodes and
    // re-execute folded ops (Shape, Gather, etc.) that produce shape data.
    if (shapes_changed) {
        shapes_dirty = true; // fold_run checks this
        if (!fold_run(this))
            return false;
        shapes_dirty = false;
        // Re-apply memory plan with new sizes
        if (memory_planning_enabled) {
            planner.analyze(this);
            planner.plan();
            planner.apply();
            memory_planned = true;
        }
    }

    // Zero the activation pool only the first time after (re)allocation.
    // Operators write every cell of their output before reading it, so
    // stale bytes from a previous run are overwritten; blanket-memset on
    // every run was the memory-bandwidth cost flagged by F-PHP-001.
    if (memory_planned && !planner.is_pool_zeroed())
        planner.zero_pool();

    if (!run_graph(this))
        return false;
    shapes_dirty = false;

    // --- Post-first-run optimization ---
    if (first_run) {
        optimizer->optimize(this);
        optimizer->build_plan(this);
        // Ensure workspace covers layout reorder needs
        if (optimizer->layout_reorder_ws > workspace_size)
            ensure_workspace(this, optimizer->layout_reorder_ws);
    }

    if (memory_planning_enabled && first_run && !memory_planned) {
        planner.analyze(this);
        planner.plan();
        planner.apply();
        memory_planned = true;
    }

#ifdef NNR_ENABLE_WEBGPU
    // Submit any WebGPU dispatches still pending in the shared encoder.
    // Without this, ops that don't trigger a CPU download (e.g. when no
    // CPU op consumes their output and the user doesn't read the graph
    // output mid-loop) would silently not run until the next download —
    // breaking timing measurements and benchmark semantics. No-op when
    // no encoder is active (CPU-only graphs, or already flushed by a
    // boundary download).
    //
    // When per-op profiling is active, append ResolveQuerySet + readback
    // copy commands before the flush submits, then map+read after.
    nnr::webgpu::op_profiler_pre_flush();
    nnr::webgpu::flush_encoder();
    nnr::webgpu::op_profiler_post_flush();
#endif

    return true;
}

void context_t::dump(bool detail) const
{
    if (meta_ir_version > 0) {
        std::println("IR Version: v{}", meta_ir_version);
        std::println("Producer: {} {}", meta_producer_name, meta_producer_version);
        std::println("Domain: {}", meta_domain);
        std::println("Imports:");
        for (auto& [domain, version] : meta_opsets) {
            std::println("\t{} v{}", domain.empty() ? "ai.onnx" : domain, version);
        }
    }
    if (graph) {
        graph->dump(detail);
    }
}

void context_t::enable_profiling(bool enable)
{
    profiling_enabled = enable;
    if (enable && graph) {
        profile.assign(graph->nodes.size(), {});
    } else if (!enable) {
        profile.clear();
    }
}

void context_t::reset_profile()
{
    std::fill(profile.begin(), profile.end(), node_profile_t{});
}

void context_t::dump_profile(FILE* f) const
{
    if (profile.empty() || !graph) return;
    // Convert ticks → nanoseconds
#ifdef _WIN32
    int64_t qpc_freq;
    QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&qpc_freq));
    const double ns_per_tick = 1e9 / qpc_freq;
#else
    const double ns_per_tick = 1.0;  // steady_clock already in ns
#endif
    std::println(f, "op_type,node_name,total_ns,calls,ops");
    for (size_t i = 0; i < profile.size(); ++i) {
        auto* n = graph->nodes[i];
        int64_t node_ops = n->num_ops();
        uint64_t total_ns = static_cast<uint64_t>(profile[i].total_ticks * ns_per_tick);
        std::println(f, "{},{},{},{},{}", n->op_type, n->node_name,
            total_ns, profile[i].calls, node_ops);
    }
}

void context_t::dump_memory_stats(FILE* f) const
{
    if (!memory_planned) {
        fprintf(f, "memory_planning: off\n");
        return;
    }
    fprintf(f, "memory_planning: on\n");
    fprintf(f, "  intermediates:  %d tensors\n", planner.num_intermediates);
    fprintf(f, "  pool_slots:     %d\n", planner.num_slots);
    fprintf(f, "  inplace_ops:    %d\n", planner.num_inplace);
    fprintf(f, "  pool_bytes:     %zu (%.2f KB)\n", planner.total_pool_bytes, planner.total_pool_bytes / 1024.0);
    fprintf(f, "  unpooled_bytes: %zu (%.2f KB)\n", planner.total_unpooled_bytes, planner.total_unpooled_bytes / 1024.0);
    double saved = (planner.total_unpooled_bytes > 0)
        ? 100.0 * (1.0 - (double)planner.total_pool_bytes / (double)planner.total_unpooled_bytes)
        : 0.0;
    fprintf(f, "  savings:        %.1f%%\n", saved);
}

// ---------------------------------------------------------------------------
// tensor_t::dump
// ---------------------------------------------------------------------------

void tensor_t::dump(bool detail) const
{
    std::print("{}: {}", name, data_type_tostring(type));
    if (ndim > 0) {
        std::print("[");
        for (int i = 0; i < ndim; ++i) {
            std::print("{}", dims[i]);
            if (i != ndim - 1) {
                std::print(" x ");
            }
        }
        std::print("]");
        if (detail) {
            std::print(" = \r\n");
            for (int i = 0; i < ndim; ++i) {
                if (dims[i] <= 0) return;
            }
            int sizes[MAX_NDIM] = {};
            int levels[MAX_NDIM] = {};
            sizes[ndim - 1] = dims[ndim - 1];
            levels[ndim - 1] = 0;
            char lbuf[MAX_NDIM + 1] = {};
            char rbuf[MAX_NDIM + 1] = {};
            char* lp = lbuf;
            char* rp = rbuf;
            for (int i = ndim - 2; i >= 0; i--) {
                sizes[i] = dims[i] * sizes[i + 1];
                levels[i] = 0;
            }
            for (size_t idx = 0; idx < ndata; ++idx) {
                for (int j = 0; j < ndim; ++j) {
                    if ((idx % sizes[j]) == 0) levels[j]++;
                    if (levels[j] == 1) { *lp++ = '['; levels[j]++; }
                    if (levels[j] == 3) {
                        *rp++ = ']';
                        if ((j != 0) && (levels[j] > levels[j - 1])) { *lp++ = '['; levels[j] = 2; }
                        else levels[j] = 0;
                    }
                }
                *lp = *rp = '\0';
                std::print("{}", rbuf);
                if (rbuf[0] != '\0') {
                    std::print("\r\n");
                    for (int k = ndim - static_cast<int>(strlen(rbuf)); k > 0; k--)
                        std::print(" ");
                }
                std::print("{}", lbuf);
                if (lbuf[0] == '\0') std::print(" ");
                void* p = (void*)((char*)data + data_type_sizeof(type) * idx);
                switch (type) {
                case NNR_DATA_TYPE_BOOL:    std::print("{},", *((uint8_t*)p) ? "true" : "false"); break;
                case NNR_DATA_TYPE_INT8:    std::print("{},", *((int8_t*)p)); break;
                case NNR_DATA_TYPE_INT16:   std::print("{},", *((int16_t*)p)); break;
                case NNR_DATA_TYPE_INT32:   std::print("{},", *((int32_t*)p)); break;
                case NNR_DATA_TYPE_INT64:   std::print("{},", *((int64_t*)p)); break;
                case NNR_DATA_TYPE_UINT8:   std::print("{},", *((uint8_t*)p)); break;
                case NNR_DATA_TYPE_UINT16:  std::print("{},", *((uint16_t*)p)); break;
                case NNR_DATA_TYPE_UINT32:  std::print("{},", *((uint32_t*)p)); break;
                case NNR_DATA_TYPE_UINT64:  std::print("{},", *((uint64_t*)p)); break;
                case NNR_DATA_TYPE_BFLOAT16: std::print("{:g},", bfloat16_to_float32(*((uint16_t*)p))); break;
                case NNR_DATA_TYPE_FLOAT16:  std::print("{:g},", float16_to_float32(*((uint16_t*)p))); break;
                case NNR_DATA_TYPE_FLOAT32:  std::print("{:g},", *((float*)p)); break;
                case NNR_DATA_TYPE_FLOAT64:  std::print("{:g},", *((double*)p)); break;
                case NNR_DATA_TYPE_COMPLEX64: std::print("{:g} + {:g}i,", *((float*)p), *((float*)((char*)p + sizeof(float)))); break;
                case NNR_DATA_TYPE_COMPLEX128: std::print("{:g} + {:g}i,", *((double*)p), *((double*)((char*)p + sizeof(double)))); break;
                case NNR_DATA_TYPE_STRING:   std::print("{}", (*(std::string*)p)); break;
                default: std::print("?,"); break;
                }
                lp = lbuf;
                rp = rbuf;
            }
            for (int j = 0; j < ndim; ++j) std::print("]");
            std::println("");
        }else {
            std::print(" = [...]");
            std::println("");
        }
    }else if (ndata == 1) {
        std::print(" = ");
        void* p = data;
        switch (type) {
        case NNR_DATA_TYPE_BOOL:    std::print("{}", *((uint8_t*)p) ? "true" : "false"); break;
        case NNR_DATA_TYPE_INT8:    std::print("{}", *((int8_t*)p)); break;
        case NNR_DATA_TYPE_INT16:   std::print("{}", *((int16_t*)p)); break;
        case NNR_DATA_TYPE_INT32:   std::print("{}", *((int32_t*)p)); break;
        case NNR_DATA_TYPE_INT64:   std::print("{}", *((int64_t*)p)); break;
        case NNR_DATA_TYPE_UINT8:   std::print("{}", *((uint8_t*)p)); break;
        case NNR_DATA_TYPE_UINT16:  std::print("{}", *((uint16_t*)p)); break;
        case NNR_DATA_TYPE_UINT32:  std::print("{}", *((uint32_t*)p)); break;
        case NNR_DATA_TYPE_UINT64:  std::print("{}", *((uint64_t*)p)); break;
        case NNR_DATA_TYPE_BFLOAT16: std::print("{:g}", bfloat16_to_float32(*((uint16_t*)p))); break;
        case NNR_DATA_TYPE_FLOAT16:  std::print("{:g}", float16_to_float32(*((uint16_t*)p))); break;
        case NNR_DATA_TYPE_FLOAT32:  std::print("{:g}", *((float*)p)); break;
        case NNR_DATA_TYPE_FLOAT64:  std::print("{:g}", *((double*)p)); break;
        case NNR_DATA_TYPE_COMPLEX64: std::print("{:g} + {:g}i", *((float*)p), *((float*)((char*)p + sizeof(float)))); break;
        case NNR_DATA_TYPE_COMPLEX128: std::print("{:g} + {:g}i", *((double*)p), *((double*)((char*)p + sizeof(double)))); break;
        case NNR_DATA_TYPE_STRING:   std::print("{}", (*(std::string*)p)); break;
        default: std::print("?"); break;
        }
        std::println("");
    }else {
        std::print(" = null");
        std::println("");
    }
}

// Stubs for sequence file I/O — implemented by ONNX loader
bool tensor_load_sequence_from_file(tensor_t* t, std::string_view filename)
{
    extern bool onnx_tensor_load_sequence_from_file(tensor_t* t, std::string_view filename);
    return onnx_tensor_load_sequence_from_file(t, filename);
}

tensor_t* tensor_alloc_optional_from_file(std::string_view filename)
{
    extern tensor_t* onnx_tensor_alloc_optional_from_file(std::string_view filename);
    return onnx_tensor_alloc_optional_from_file(filename);
}

bool tensor_sequence_equal_file(const tensor_t* t, std::string_view filename)
{
    extern bool onnx_tensor_sequence_equal_file(const tensor_t* t, std::string_view filename);
    return onnx_tensor_sequence_equal_file(t, filename);
}

bool tensor_sequence_equal(const tensor_t* a, const tensor_t* b)
{
    if (!a || !b) return false;
    const sequence_t* sa = tensor_get_sequence(a);
    const sequence_t* sb = tensor_get_sequence(b);
    if (!sa || !sb) return false;
    if (sa->elem_type != sb->elem_type) return false;
    if (sa->tensors.size() != sb->tensors.size()) return false;
    for (size_t i = 0; i < sa->tensors.size(); ++i) {
        if (!tensor_equal(sa->tensors[i], sb->tensors[i])) return false;
    }
    return true;
}

} // namespace nnr
