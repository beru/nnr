#include "onnx_loader.h"
#include "nnr.h"
#include "attr_key.h"
#include "nnrconf.h"
#include "backend/cpu/solve_operator.h"
#include "backend/cpu/util.h"
#include <cstring>
#include <cstdio>
#include <complex>
#include <unordered_map>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#if defined(NNR_USE_CUDA)
namespace nnr { void cuda_anchor(); }
static struct _cuda_anchor_t { _cuda_anchor_t() { nnr::cuda_anchor(); } } _cuda_anchor;
#endif

namespace nnr {

// ---------------------------------------------------------------------------
// ONNX data type -> NNR data type mapping
// ---------------------------------------------------------------------------

static data_type_t onnx_to_nnr_dtype(int32_t onnx_dt)
{
    switch (onnx_dt) {
    case onnx_pb::TensorProto::FLOAT:           return NNR_DATA_TYPE_FLOAT32;
    case onnx_pb::TensorProto::DOUBLE:          return NNR_DATA_TYPE_FLOAT64;
    case onnx_pb::TensorProto::FLOAT16:         return NNR_DATA_TYPE_FLOAT16;
    case onnx_pb::TensorProto::BFLOAT16:        return NNR_DATA_TYPE_BFLOAT16;
    case onnx_pb::TensorProto::INT8:            return NNR_DATA_TYPE_INT8;
    case onnx_pb::TensorProto::INT16:           return NNR_DATA_TYPE_INT16;
    case onnx_pb::TensorProto::INT32:           return NNR_DATA_TYPE_INT32;
    case onnx_pb::TensorProto::INT64:           return NNR_DATA_TYPE_INT64;
    case onnx_pb::TensorProto::UINT8:           return NNR_DATA_TYPE_UINT8;
    case onnx_pb::TensorProto::UINT16:          return NNR_DATA_TYPE_UINT16;
    case onnx_pb::TensorProto::UINT32:          return NNR_DATA_TYPE_UINT32;
    case onnx_pb::TensorProto::UINT64:          return NNR_DATA_TYPE_UINT64;
    case onnx_pb::TensorProto::BOOL:            return NNR_DATA_TYPE_BOOL;
    case onnx_pb::TensorProto::STRING:          return NNR_DATA_TYPE_STRING;
    case onnx_pb::TensorProto::COMPLEX64:       return NNR_DATA_TYPE_COMPLEX64;
    case onnx_pb::TensorProto::COMPLEX128:      return NNR_DATA_TYPE_COMPLEX128;
    case onnx_pb::TensorProto::FLOAT8E4M3FN:    return NNR_DATA_TYPE_FLOAT8E4M3FN;
    case onnx_pb::TensorProto::FLOAT8E4M3FNUZ:  return NNR_DATA_TYPE_FLOAT8E4M3FNUZ;
    case onnx_pb::TensorProto::FLOAT8E5M2:      return NNR_DATA_TYPE_FLOAT8E5M2;
    case onnx_pb::TensorProto::FLOAT8E5M2FNUZ:  return NNR_DATA_TYPE_FLOAT8E5M2FNUZ;
    case onnx_pb::TensorProto::FLOAT8E8M0:      return NNR_DATA_TYPE_FLOAT8E8M0;
    case onnx_pb::TensorProto::UINT4:           return NNR_DATA_TYPE_UINT4;
    case onnx_pb::TensorProto::INT4:            return NNR_DATA_TYPE_INT4;
    case onnx_pb::TensorProto::FLOAT4E2M1:      return NNR_DATA_TYPE_FLOAT4E2M1;
    case onnx_pb::TensorProto::UINT2:           return NNR_DATA_TYPE_UINT2;
    case onnx_pb::TensorProto::INT2:            return NNR_DATA_TYPE_INT2;
    default:                                    return NNR_DATA_TYPE_UNDEFINED;
    }
}

// ---------------------------------------------------------------------------
// Tensor data copy from onnx_pb::TensorProto
// ---------------------------------------------------------------------------

void onnx_tensor_copy_from_proto(tensor_t* t, const onnx_pb::TensorProto& o)
{
    if (!t) return;
    if (t->type != onnx_to_nnr_dtype(o.data_type)) return;
    if (t->ndata <= 0 || !t->data) return;

    size_t sz = data_type_sizeof(t->type);
    if (sz <= 0) return;

    // ONNX spec forbids raw_data for STRING tensors. If a malicious model
    // sets it anyway, treating raw_data as a blob and memcpy'ing over
    // `std::string` objects overwrites their internal SSO pointers / flags
    // — subsequent destruction invokes arbitrary frees. Fall through to
    // string_data instead.
    const bool raw_valid_for_type = !o.raw_data.empty() &&
                                    t->type != NNR_DATA_TYPE_STRING;
    if (raw_valid_for_type) {
        const uint8_t* src = (const uint8_t*)o.raw_data.data();
        size_t len = o.raw_data.size();
#ifdef NNR_LITTLE_ENDIAN
        if (t->type == NNR_DATA_TYPE_INT4) {
            size_t n = std::min(t->ndata, len * 2);  // 2 values per byte
            int4_unpack(src, (int8_t*)t->data, n);
        } else if (t->type == NNR_DATA_TYPE_UINT4) {
            size_t n = std::min(t->ndata, len * 2);
            uint4_unpack(src, (uint8_t*)t->data, n);
        } else if (t->type == NNR_DATA_TYPE_FLOAT4E2M1) {
            size_t n = std::min(t->ndata, len * 2);
            uint4_unpack(src, (uint8_t*)t->data, n);
        } else if (t->type == NNR_DATA_TYPE_INT2) {
            size_t n = std::min(t->ndata, len * 4);  // 4 values per byte
            int2_unpack(src, (int8_t*)t->data, n);
        } else if (t->type == NNR_DATA_TYPE_UINT2) {
            size_t n = std::min(t->ndata, len * 4);
            uint2_unpack(src, (uint8_t*)t->data, n);
        } else {
            memcpy(t->data, src, std::min(len, t->ndata * sz));
        }
#else
        size_t n = std::min(t->ndata, len / sz);
        switch (o.data_type) {
        case onnx_pb::TensorProto::FLOAT: {
            float* p = (float*)t->data; uint32_t* q = (uint32_t*)src;
            for (size_t i = 0; i < n; ++i) p[i] = std::bit_cast<float>(le32_to_cpu(q[i]));
        } break;
        case onnx_pb::TensorProto::UINT8: case onnx_pb::TensorProto::INT8:
        case onnx_pb::TensorProto::BOOL:
            memcpy(t->data, src, n); break;
        case onnx_pb::TensorProto::UINT16: case onnx_pb::TensorProto::INT16:
        case onnx_pb::TensorProto::FLOAT16: case onnx_pb::TensorProto::BFLOAT16: {
            uint16_t* p = (uint16_t*)t->data; uint16_t* q = (uint16_t*)src;
            for (size_t i = 0; i < n; ++i) p[i] = le16_to_cpu(q[i]);
        } break;
        case onnx_pb::TensorProto::INT32: case onnx_pb::TensorProto::UINT32: {
            uint32_t* p = (uint32_t*)t->data; uint32_t* q = (uint32_t*)src;
            for (size_t i = 0; i < n; ++i) p[i] = le32_to_cpu(q[i]);
        } break;
        case onnx_pb::TensorProto::INT64: case onnx_pb::TensorProto::UINT64: {
            uint64_t* p = (uint64_t*)t->data; uint64_t* q = (uint64_t*)src;
            for (size_t i = 0; i < n; ++i) p[i] = le64_to_cpu(q[i]);
        } break;
        case onnx_pb::TensorProto::DOUBLE: {
            double* p = (double*)t->data; uint64_t* q = (uint64_t*)src;
            for (size_t i = 0; i < n; ++i) p[i] = std::bit_cast<double>(le64_to_cpu(q[i]));
        } break;
        case onnx_pb::TensorProto::COMPLEX64: {
            float* p = (float*)t->data; uint32_t* q = (uint32_t*)src;
            for (size_t i = 0; i < 2*n; ++i) p[i] = std::bit_cast<float>(le32_to_cpu(q[i]));
        } break;
        case onnx_pb::TensorProto::COMPLEX128: {
            double* p = (double*)t->data; uint64_t* q = (uint64_t*)src;
            for (size_t i = 0; i < 2*n; ++i) p[i] = std::bit_cast<double>(le64_to_cpu(q[i]));
        } break;
        default: break;
        }
#endif
    } else {
        switch (o.data_type) {
        case onnx_pb::TensorProto::FLOAT: {
            size_t n = std::min(t->ndata, o.float_data.size());
            if (n) memcpy(t->data, o.float_data.data(), sizeof(float) * n);
        } break;
        case onnx_pb::TensorProto::INT32: {
            size_t n = std::min(t->ndata, o.int32_data.size());
            if (n) memcpy(t->data, o.int32_data.data(), sizeof(int32_t) * n);
        } break;
        case onnx_pb::TensorProto::UINT8: case onnx_pb::TensorProto::INT8:
        case onnx_pb::TensorProto::UINT16: case onnx_pb::TensorProto::INT16:
        case onnx_pb::TensorProto::BOOL:
        case onnx_pb::TensorProto::FLOAT16:
        case onnx_pb::TensorProto::BFLOAT16: {
            // Each sub-int32 value is stored in its own int32_t element —
            // extract element-wise to avoid reading interleaved padding bytes.
            size_t n = std::min(t->ndata, o.int32_data.size());
            for (size_t i = 0; i < n; ++i)
                memcpy((uint8_t*)t->data + i * sz, &o.int32_data[i], sz);
        } break;
        case onnx_pb::TensorProto::STRING: {
            size_t n = std::min(t->ndata, o.string_data.size());
            std::string* str = (std::string*)t->data;
            for (size_t i = 0; i < n; ++i)
                str[i].assign(o.string_data[i].data(), o.string_data[i].size());
        } break;
        case onnx_pb::TensorProto::INT64: {
            size_t n = std::min(t->ndata, o.int64_data.size());
            if (n) memcpy(t->data, o.int64_data.data(), sizeof(int64_t) * n);
        } break;
        case onnx_pb::TensorProto::DOUBLE: {
            size_t n = std::min(t->ndata, o.double_data.size());
            if (n) memcpy(t->data, o.double_data.data(), sizeof(double) * n);
        } break;
        case onnx_pb::TensorProto::UINT64: {
            size_t n = std::min(t->ndata, o.uint64_data.size());
            if (n) memcpy(t->data, o.uint64_data.data(), sizeof(uint64_t) * n);
        } break;
        case onnx_pb::TensorProto::UINT32: {
            // Each uint32 value is stored in its own uint64_t element.
            size_t n = std::min(t->ndata, o.uint64_data.size());
            uint32_t* dst = (uint32_t*)t->data;
            for (size_t i = 0; i < n; ++i)
                dst[i] = (uint32_t)o.uint64_data[i];
        } break;
        case onnx_pb::TensorProto::COMPLEX64: {
            size_t n = std::min(t->ndata, o.float_data.size() / 2);
            if (n) memcpy(t->data, o.float_data.data(), sizeof(float) * 2 * n);
        } break;
        case onnx_pb::TensorProto::COMPLEX128: {
            size_t n = std::min(t->ndata, o.double_data.size() / 2);
            if (n) memcpy(t->data, o.double_data.data(), sizeof(double) * 2 * n);
        } break;
        case onnx_pb::TensorProto::UINT4: {
            size_t n2 = std::min((t->ndata+1)/2, o.int32_data.size());
            if (n2) {
                auto p = std::make_unique<uint8_t[]>(n2);
                for (size_t i = 0; i < n2; ++i) p[i] = (uint8_t)o.int32_data[i];
                uint4_unpack(p.get(), (uint8_t*)t->data, t->ndata);
            }
        } break;
        case onnx_pb::TensorProto::INT4: {
            size_t n2 = std::min((t->ndata+1)/2, o.int32_data.size());
            if (n2) {
                auto p = std::make_unique<uint8_t[]>(n2);
                for (size_t i = 0; i < n2; ++i) p[i] = (uint8_t)o.int32_data[i];
                int4_unpack(p.get(), (int8_t*)t->data, t->ndata);
            }
        } break;
        case onnx_pb::TensorProto::FLOAT8E4M3FN:
        case onnx_pb::TensorProto::FLOAT8E4M3FNUZ:
        case onnx_pb::TensorProto::FLOAT8E5M2:
        case onnx_pb::TensorProto::FLOAT8E5M2FNUZ:
        case onnx_pb::TensorProto::FLOAT8E8M0: {
            size_t n = std::min(t->ndata, o.int32_data.size());
            if (n) {
                uint8_t* d = (uint8_t*)t->data;
                for (size_t i = 0; i < n; ++i) d[i] = (uint8_t)(o.int32_data[i] & 0xFF);
            }
        } break;
        case onnx_pb::TensorProto::FLOAT4E2M1: {
            size_t n2 = std::min((t->ndata+1)/2, o.int32_data.size());
            if (n2) {
                auto p = std::make_unique<uint8_t[]>(n2);
                for (size_t i = 0; i < n2; ++i) p[i] = (uint8_t)o.int32_data[i];
                uint4_unpack(p.get(), (uint8_t*)t->data, t->ndata);
            }
        } break;
        case onnx_pb::TensorProto::UINT2: {
            size_t n2 = std::min((t->ndata+3)/4, o.int32_data.size());
            if (n2) {
                auto p = std::make_unique<uint8_t[]>(n2);
                for (size_t i = 0; i < n2; ++i) p[i] = (uint8_t)o.int32_data[i];
                uint2_unpack(p.get(), (uint8_t*)t->data, t->ndata);
            }
        } break;
        case onnx_pb::TensorProto::INT2: {
            size_t n2 = std::min((t->ndata+3)/4, o.int32_data.size());
            if (n2) {
                auto p = std::make_unique<uint8_t[]>(n2);
                for (size_t i = 0; i < n2; ++i) p[i] = (uint8_t)o.int32_data[i];
                int2_unpack(p.get(), (int8_t*)t->data, t->ndata);
            }
        } break;
        default: break;
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor allocation helpers
// ---------------------------------------------------------------------------

// Load tensor data from ONNX external data file.
// External data entries: "location" (filename), "offset" (byte offset), "length" (byte count).
static bool onnx_tensor_load_external(tensor_t* t, const onnx_pb::TensorProto& o,
                                       const std::string& model_dir)
{
    if (!t || !t->data || t->ndata == 0) return false;
    if (o.data_location != onnx_pb::TensorProto::EXTERNAL) return false;

    std::string_view location;
    int64_t offset = 0;
    int64_t length = -1;

    for (auto& entry : o.external_data) {
        if (entry.key == "location") location = entry.value;
        else if (entry.key == "offset") offset = std::atoll(std::string(entry.value).c_str());
        else if (entry.key == "length") length = std::atoll(std::string(entry.value).c_str());
    }

    if (location.empty()) return false;

    // Reject path-traversal and absolute paths. `location` is attacker-
    // controlled and is concatenated with `model_dir`; without this guard,
    // values like "../../../etc/passwd" or "/etc/passwd" / "C:\..." resolve
    // into arbitrary files on the host.
    auto has_traversal = [](std::string_view s) {
        if (s.empty()) return true;
        if (s.find("..") != std::string_view::npos) return true;
        if (s.front() == '/' || s.front() == '\\') return true;
        // Reject Windows drive-letter absolute paths like "C:\..." or "C:/...".
        if (s.size() >= 2 && s[1] == ':') return true;
        return false;
    };
    if (has_traversal(location)) {
        std::fprintf(stderr, "nnr: rejecting external-data location with "
                             "path traversal: %.*s\n",
                     (int)location.size(), location.data());
        return false;
    }

    std::string path = model_dir + std::string(location);
    size_t elem_size = data_type_sizeof(t->type);
    // Compute the maximum number of bytes the destination tensor can receive
    // safely (ndata * elem_size, saturating on overflow). `length` from the
    // model is attacker-controlled; clamp it to that ceiling so a malicious
    // oversized value cannot drive `ReadFile`/`fread` past the heap buffer.
    size_t tensor_cap = (elem_size && t->ndata <= SIZE_MAX / elem_size)
                      ? t->ndata * elem_size
                      : SIZE_MAX;
    size_t byte_count = (length > 0) ? (size_t)length : tensor_cap;
    if (byte_count > tensor_cap) byte_count = tensor_cap;
    if (offset < 0) return false;

#ifdef _WIN32
    HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr, "nnr: external data file not found: %s\n", path.c_str());
        return false;
    }
    LARGE_INTEGER li;
    li.QuadPart = offset;
    SetFilePointerEx(fh, li, nullptr, FILE_BEGIN);
    DWORD bytes_read = 0;
    // Read in chunks (ReadFile DWORD limit)
    size_t remaining = byte_count;
    uint8_t* dst = (uint8_t*)t->data;
    while (remaining > 0) {
        DWORD chunk = (DWORD)std::min(remaining, (size_t)0x40000000);
        if (!ReadFile(fh, dst, chunk, &bytes_read, nullptr) || bytes_read == 0) break;
        dst += bytes_read;
        remaining -= bytes_read;
    }
    CloseHandle(fh);
    return remaining == 0;
#else
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "nnr: external data file not found: %s\n", path.c_str());
        return false;
    }
    fseeko(fp, offset, SEEK_SET);
    size_t read = fread(t->data, 1, byte_count, fp);
    fclose(fp);
    return read == byte_count;
#endif
}

tensor_t* onnx_tensor_alloc_from_proto(const onnx_pb::TensorProto& pb)
{
    if (pb.dims.size() > MAX_NDIM) return nullptr;
    int ndim = (int)pb.dims.size();
    small_vector<int> dims(ndim);
    for (int i = 0; i < ndim; ++i) {
        int64_t d = pb.dims[i];
        if (d < 0 || d > INT32_MAX) return nullptr;
        dims[i] = (int)d;
    }
    tensor_t* t = new (std::nothrow) tensor_t("", onnx_to_nnr_dtype(pb.data_type), dims);
    onnx_tensor_copy_from_proto(t, pb);
    return t;
}

tensor_t* onnx_tensor_alloc_from_file(std::string_view filename)
{
    std::string path(filename);
    tensor_t* t = nullptr;
#ifdef _WIN32
    HANDLE hf = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hf == INVALID_HANDLE_VALUE) return nullptr;
    LARGE_INTEGER sz;
    if (GetFileSizeEx(hf, &sz) && sz.QuadPart > 0) {
        HANDLE hm = CreateFileMappingA(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
        CloseHandle(hf);
        const uint8_t* ptr = hm ? (const uint8_t*)MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0) : nullptr;
        if (ptr) {
            onnx_pb::TensorProto pb;
            if (onnx_pb::read(ptr, (size_t)sz.QuadPart, pb))
                t = onnx_tensor_alloc_from_proto(pb);
            UnmapViewOfFile(ptr);
        }
        if (hm) CloseHandle(hm);
    } else {
        CloseHandle(hf);
    }
#else
    struct stat st;
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;
    if (fstat(fd, &st) == 0 && st.st_size > 0) {
        size_t l = (size_t)st.st_size;
        void* ptr = mmap(nullptr, l, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (ptr != MAP_FAILED) {
            onnx_pb::TensorProto pb;
            if (onnx_pb::read((const uint8_t*)ptr, l, pb))
                t = onnx_tensor_alloc_from_proto(pb);
            munmap(ptr, l);
        }
    } else {
        close(fd);
    }
#endif
    return t;
}

tensor_t* onnx_tensor_alloc_optional_from_file(std::string_view filename)
{
#ifdef _WIN32
    std::string path(filename);
    if (GetFileAttributesA(path.c_str()) == INVALID_FILE_ATTRIBUTES) return nullptr;
#else
    std::string path(filename);
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return nullptr;
#endif
    return onnx_tensor_alloc_from_file(filename);
}

bool onnx_tensor_load_sequence_from_file(tensor_t* t, std::string_view filename)
{
    (void)t; (void)filename;
    return false; // sequence file format not yet implemented
}

bool onnx_tensor_sequence_equal_file(const tensor_t* t, std::string_view filename)
{
    (void)t; (void)filename;
    return false; // sequence file format not yet implemented
}

// ---------------------------------------------------------------------------
// ValueInfoProto -> tensor_t
// ---------------------------------------------------------------------------

static tensor_t* tensor_from_value_info(const onnx_pb::ValueInfoProto& v)
{
    if (v.name.empty()) return nullptr;

    if (v.type.tensor_type.elem_type != 0) {
        data_type_t type = onnx_to_nnr_dtype(v.type.tensor_type.elem_type);
        small_vector<int> dims;
        // Symbolic / unspecified dims (dim_param or no dim_value) load as 0.
        // Bench harness's -d N path and user code rely on 0 as the
        // "needs concrete value" sentinel (see tests/bench/bench.cpp -d loop).
        // Defaulting to 1 silently runs models at degenerate spatial sizes.
        for (auto& d : v.type.tensor_type.shape.dim)
            dims.push_back(d.dim_value > 0 ? (int)d.dim_value : 0);
        return new (std::nothrow) tensor_t(v.name, type, dims);
    }
    if (v.type.sequence_type)
        return new (std::nothrow) tensor_t(v.name, NNR_DATA_TYPE_SEQUENCE, {});

    return new (std::nothrow) tensor_t(v.name, NNR_DATA_TYPE_UNDEFINED, {});
}

// ---------------------------------------------------------------------------
// Attribute parsing: onnx_pb::AttributeProto -> attr_t
// ---------------------------------------------------------------------------

static bool onnx_build_graph(context_t* ctx, graph_t* graph, const onnx_pb::GraphProto& gp, int default_opset); // forward decl

static void parse_attr(attr_t& a, const onnx_pb::AttributeProto& ap, context_t* ctx, int default_opset)
{
    switch (ap.type) {
    case onnx_pb::AttributeProto::FLOAT:
        a.kind = attr_t::kind_t::FLOAT; a.f = ap.f; break;
    case onnx_pb::AttributeProto::INT:
        a.kind = attr_t::kind_t::INT; a.i = ap.i; break;
    case onnx_pb::AttributeProto::STRING:
        a.kind = attr_t::kind_t::STRING; a.s = ap.s; break;
    case onnx_pb::AttributeProto::FLOATS:
        a.kind = attr_t::kind_t::FLOATS;
        a.floats = std::span<const float>(ap.floats.data(), ap.floats.size()); break;
    case onnx_pb::AttributeProto::INTS:
        a.kind = attr_t::kind_t::INTS;
        a.ints = std::span<const int64_t>(ap.ints.data(), ap.ints.size()); break;
    case onnx_pb::AttributeProto::TENSOR:
        a.kind = attr_t::kind_t::TENSOR;
        if (ap.t.data_type != 0) {
            a.tensor = onnx_tensor_alloc_from_proto(ap.t);
            if (a.tensor) ctx->attr_tensors_.push_back(a.tensor);
        }
        break;
    case onnx_pb::AttributeProto::GRAPH:
        a.kind = attr_t::kind_t::GRAPH;
        if (ap.g) {
            // Store raw proto pointer (lifetime = model proto, no separate free)
            a.raw = ap.g.get();
            // Pre-compile subgraph into the same context (outer tensors already in ctx->map)
            a.subgraph = new (std::nothrow) graph_t;
            if (a.subgraph) {
                if (!onnx_build_graph(ctx, a.subgraph, *ap.g, default_opset)) {
                    delete a.subgraph;
                    a.subgraph = nullptr;
                } else {
                    ctx->attr_subgraphs_.push_back(a.subgraph);
                    // Build subgraph_inputs span from attr_pool
                    size_t n_in = 0;
                    for (auto& vi : ap.g->input) if (!vi.name.empty()) ++n_in;
                    if (n_in > 0) {
                        std::string_view* sv = ctx->attr_pool.alloc_arr<std::string_view>(n_in);
                        size_t idx = 0;
                        for (auto& vi : ap.g->input) if (!vi.name.empty()) sv[idx++] = vi.name;
                        a.subgraph_inputs = std::span<const std::string_view>(sv, n_in);
                    }
                    // Build subgraph_outputs span from attr_pool
                    size_t n_out = 0;
                    for (auto& vi : ap.g->output) if (!vi.name.empty()) ++n_out;
                    if (n_out > 0) {
                        std::string_view* sv = ctx->attr_pool.alloc_arr<std::string_view>(n_out);
                        size_t idx = 0;
                        for (auto& vi : ap.g->output) if (!vi.name.empty()) sv[idx++] = vi.name;
                        a.subgraph_outputs = std::span<const std::string_view>(sv, n_out);
                    }
                }
            }
        }
        break;
    case onnx_pb::AttributeProto::STRINGS:
        a.kind = attr_t::kind_t::STRINGS;
        if (!ap.strings.empty()) {
            std::string_view* sv = ctx->attr_pool.alloc_arr<std::string_view>(ap.strings.size());
            for (size_t i = 0; i < ap.strings.size(); ++i) sv[i] = ap.strings[i];
            a.strings = std::span<const std::string_view>(sv, ap.strings.size());
        }
        break;
    default:
        a.kind = attr_t::kind_t::INT; break;
    }
}

// ---------------------------------------------------------------------------
// Graph builder (internal)
// ---------------------------------------------------------------------------

static bool onnx_build_graph(context_t* ctx, graph_t* graph, const onnx_pb::GraphProto& gp, int default_opset)
{
    if (!ctx || !graph) return false;

    // Local index for O(1) tensor lookup. Pre-populated from ctx->map so that
    // tensors from an outer scope are visible to subgraph builds.
    std::unordered_map<std::string_view, tensor_t*> tmap;
    tmap.reserve(ctx->map.size() + gp.input.size() + gp.output.size()
                 + gp.value_info.size() + gp.node.size() * 2);
    for (auto& [k, v] : ctx->map) tmap[k] = v;

    auto find_t = [&](std::string_view name) -> tensor_t* {
        auto it = tmap.find(name);
        return it != tmap.end() ? it->second : nullptr;
    };
    auto add_t = [&](std::string_view name, tensor_t* t) {
        ctx->map.emplace_back(name, t);
        tmap[name] = t;
    };

    // Phase 1: add all tensors declared in this graph proto to ctx->map.
    // Tensors already present (from outer scope) are skipped.

    for (auto& v : gp.input) {
        if (v.name.empty() || find_t(v.name)) continue;
        tensor_t* t = tensor_from_value_info(v);
        if (!t) return false;
        for (auto& ini : gp.initializer)
            if (t->name == ini.name) {
                onnx_tensor_copy_from_proto(t, ini);
                if (ini.data_location == onnx_pb::TensorProto::EXTERNAL)
                    onnx_tensor_load_external(t, ini, ctx->model_dir);
                break;
            }
        add_t(t->name, t);
    }

    for (auto& v : gp.output) {
        if (v.name.empty() || find_t(v.name)) continue;
        tensor_t* t = tensor_from_value_info(v);
        if (!t) return false;
        add_t(t->name, t);
    }

    for (auto& v : gp.value_info) {
        if (v.name.empty() || find_t(v.name)) continue;
        tensor_t* t = tensor_from_value_info(v);
        if (!t) continue;
        // Load initializer data when tensor appears in both value_info and initializer
        // (e.g. ssd-12-int8 stores Reshape shape constants as initializers with value_info)
        for (auto& ini : gp.initializer)
            if (t->name == ini.name) {
                onnx_tensor_copy_from_proto(t, ini);
                if (ini.data_location == onnx_pb::TensorProto::EXTERNAL)
                    onnx_tensor_load_external(t, ini, ctx->model_dir);
                break;
            }
        add_t(t->name, t);
    }

    for (auto& np : gp.node)
        for (auto& name : np.output) {
            if (name.empty() || find_t(name)) continue;
            tensor_t* t = new (std::nothrow) tensor_t(name, NNR_DATA_TYPE_UNDEFINED, {});
            if (!t) return false;
            add_t(name, t);
        }

    for (auto& np : gp.node)
        for (auto& name : np.input) {
            if (name.empty() || find_t(name)) continue;
            bool found = false;
            for (auto& o : gp.initializer) {
                if (o.name != name) continue;
                if (o.dims.size() > MAX_NDIM) continue;
                int ndim = (int)o.dims.size();
                small_vector<int> dims(ndim);
                bool bad_dim = false;
                for (int l = 0; l < ndim; ++l) {
                    if (o.dims[l] < 0 || o.dims[l] > INT32_MAX) { bad_dim = true; break; }
                    dims[l] = (int)o.dims[l];
                }
                if (bad_dim) continue;
                tensor_t* t = new (std::nothrow) tensor_t(name, onnx_to_nnr_dtype(o.data_type), dims);
                if (!t) return false;
                onnx_tensor_copy_from_proto(t, o);
                if (o.data_location == onnx_pb::TensorProto::EXTERNAL)
                    onnx_tensor_load_external(t, o, ctx->model_dir);
                add_t(name, t);
                found = true; break;
            }
            if (!found) {
                tensor_t* t = new (std::nothrow) tensor_t(name, NNR_DATA_TYPE_UNDEFINED, {});
                if (!t) return false;
                add_t(name, t);
            }
        }

    // Pre-load any initializers not yet referenced by this scope's nodes.
    // Subgraph bodies (Loop/If/Scan) can consume an outer initializer that
    // has no outer-scope consumer; without loading here the body's
    // pre-populated tmap finds no tensor and falls through to creating an
    // UNDEFINED placeholder, leaving body ops to dereference null data
    // (e.g. ssd_mobilenet_v1_10's Loop body Add reads
    // 'Postprocessor/.../range/delta:0' which only the body uses).
    for (auto& o : gp.initializer) {
        if (o.name.empty() || find_t(o.name)) continue;
        if (o.dims.size() > MAX_NDIM) continue;
        int ndim = (int)o.dims.size();
        small_vector<int> dims(ndim);
        bool bad_dim = false;
        for (int l = 0; l < ndim; ++l) {
            if (o.dims[l] < 0 || o.dims[l] > INT32_MAX) { bad_dim = true; break; }
            dims[l] = (int)o.dims[l];
        }
        if (bad_dim) continue;
        tensor_t* t = new (std::nothrow) tensor_t(o.name, onnx_to_nnr_dtype(o.data_type), dims);
        if (!t) return false;
        onnx_tensor_copy_from_proto(t, o);
        if (o.data_location == onnx_pb::TensorProto::EXTERNAL)
            onnx_tensor_load_external(t, o, ctx->model_dir);
        add_t(o.name, t);
    }

    // Phase 2: build operator nodes
    graph->nodes.resize(gp.node.size());
    for (size_t i = 0; i < gp.node.size(); ++i) {
        const onnx_pb::NodeProto& np = gp.node[i];
        std::string_view domain = !np.domain.empty() ? np.domain : "ai.onnx";

        operator_t* n = solve_operator(np.op_type, default_opset, ctx->attr_pool,
                                       static_cast<backend_t>(ctx->preferred_backend));
        struct operator_dummy : public operator_t { bool exec() override { return false; } };
        if (!n) {
            std::fprintf(stderr, "nnr: unsupported op: %.*s (opset %d)\n",
                         (int)np.op_type.size(), np.op_type.data(), default_opset);
            n = pool_new<operator_dummy>(ctx->attr_pool);
        }
        graph->nodes[i] = n;

        n->ctx       = ctx;
        n->opset     = default_opset;
        n->op_type   = np.op_type;
        n->node_name = np.name;
        n->domain    = domain;

        size_t n_inputs  = np.input.size();
        size_t n_outputs = np.output.size();

        if (n_inputs > 0) {
            tensor_t** buf = ctx->attr_pool.alloc_arr<tensor_t*>(n_inputs);
            for (size_t j = 0; j < n_inputs; ++j)
                buf[j] = !np.input[j].empty() ? find_t(np.input[j]) : nullptr;
            n->inputs = std::span<tensor_t*>(buf, n_inputs);
        }
        if (n_outputs > 0) {
            tensor_t** buf = ctx->attr_pool.alloc_arr<tensor_t*>(n_outputs);
            for (size_t j = 0; j < n_outputs; ++j)
                buf[j] = !np.output[j].empty() ? find_t(np.output[j]) : nullptr;
            n->outputs = std::span<tensor_t*>(buf, n_outputs);
        }

        // Count valid attrs, then allocate from pool (placement-new each pair)
        size_t n_valid_attrs = 0;
        for (auto& ap : np.attribute) if (!ap.name.empty()) ++n_valid_attrs;
        if (n_valid_attrs > 0) {
            using pair_t = std::pair<attr_key_t, attr_t>;
            pair_t* abuf = static_cast<pair_t*>(
                ctx->attr_pool.alloc(n_valid_attrs * sizeof(pair_t), alignof(pair_t)));
            size_t idx = 0;
            for (auto& ap : np.attribute) {
                if (ap.name.empty()) continue;
                attr_key_t key = attr_key_from_string(ap.name);
                new (&abuf[idx]) pair_t(key, attr_t{});
                parse_attr(abuf[idx].second, ap, ctx, default_opset);
                // Convert ONNX data type integers to NNR data_type_t at load time
                if (abuf[idx].second.kind == attr_t::kind_t::INT &&
                    (key == attr_key_t::to || key == attr_key_t::dtype ||
                     key == attr_key_t::output_datatype ||
                     key == attr_key_t::output_dtype)) {
                    abuf[idx].second.i = onnx_to_nnr_dtype((int32_t)abuf[idx].second.i);
                }
                ++idx;
            }
            n->attrs = std::span<std::pair<attr_key_t, attr_t>>(abuf, n_valid_attrs);
        }

        if (!n->init()) {
            // Fall back to the CPU backend if the preferred backend's init
            // rejects this op/opset/shape combination (e.g. WebGPU Pad
            // requires opset ≥ 11; opset-9 Pad reads pads from attributes
            // and only CPU handles that path). Without this, onnx_loader
            // leaves a broken op in the graph and prepare()'s reshape()
            // fails. The fallback only kicks in when a non-CPU backend
            // already picked an op; CPU init failures are hard errors.
            bool fell_back = false;
            if (n->resolved_backend != static_cast<uint8_t>(backend_t::CPU)) {
                operator_t* cpu_n = solve_operator(np.op_type, default_opset,
                                                   ctx->attr_pool, backend_t::CPU);
                if (cpu_n) {
                    cpu_n->ctx       = ctx;
                    cpu_n->opset     = default_opset;
                    cpu_n->op_type   = np.op_type;
                    cpu_n->node_name = np.name;
                    cpu_n->domain    = domain;
                    cpu_n->inputs    = n->inputs;
                    cpu_n->outputs   = n->outputs;
                    cpu_n->attrs     = n->attrs;
                    if (cpu_n->init()) {
                        graph->nodes[i] = cpu_n;
                        n = cpu_n;
                        fell_back = true;
                    }
                }
            }
            if (!fell_back) {
                std::fprintf(stderr, "nnr: op init failed: %.*s (opset %d)\n",
                             (int)np.op_type.size(), np.op_type.data(), default_opset);
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Main loader
// ---------------------------------------------------------------------------

bool load_onnx(context_t* ctx, const void* data, size_t size)
{
    struct onnx_model_data_t {
        size_t bytes_size = 0;
        const uint8_t* bytes_ptr = nullptr;  // always valid: points to owned or mmap'd bytes
        std::unique_ptr<uint8_t[]> bytes;     // owning copy (null when mmap'd)
        onnx_pb::ModelProto model;
    };

    auto* md = new (std::nothrow) onnx_model_data_t;
    if (!md) return false;
    md->bytes_size = size;

    if (ctx->mmap_data_) {
        // Data is mmap'd by load_from_file() — alias directly, no copy.
        md->bytes_ptr = (const uint8_t*)data;
    } else {
        // Data passed via load() from user buffer — must copy for ownership.
        md->bytes = std::make_unique<uint8_t[]>(size);
        std::memcpy(md->bytes.get(), data, size);
        md->bytes_ptr = md->bytes.get();
    }

    if (!onnx_pb::read(md->bytes_ptr, md->bytes_size, md->model)) {
        delete md;
        return false;
    }

    ctx->set_model_handle(md, [](void* p) { delete (onnx_model_data_t*)p; });

    const onnx_pb::ModelProto& model = md->model;

    ctx->meta_ir_version = model.ir_version;
    if (!model.producer_name.empty())    ctx->meta_producer_name    = model.producer_name;
    if (!model.producer_version.empty()) ctx->meta_producer_version = model.producer_version;
    if (!model.domain.empty())           ctx->meta_domain            = model.domain;
    for (auto& op : model.opset_import) {
        std::string_view d = op.domain.empty() ? "ai.onnx" : op.domain;
        ctx->meta_opsets.emplace_back(d, op.version);
    }

    int default_opset = -1;
    for (auto& op : model.opset_import) {
        if (op.domain.empty() || op.domain == "ai.onnx") {
            default_opset = (int)op.version; break;
        }
    }

    const onnx_pb::GraphProto& gp = model.graph;

    for (auto& ini : gp.initializer)
        if (!ini.name.empty()) ctx->initializer_names.insert(ini.name);

    for (auto& v : gp.input) {
        if (v.name.empty()) continue;
        if (ctx->initializer_names.count(v.name))
            ctx->memory_plan_excluded.insert(v.name);
        else
            ctx->graph_inputs.push_back(v.name);
    }
    for (auto& v : gp.output)
        if (!v.name.empty()) {
            ctx->memory_plan_excluded.insert(v.name);
            ctx->graph_outputs.push_back(v.name);
        }

    ctx->graph = std::make_unique<graph_t>();
    return onnx_build_graph(ctx, ctx->graph.get(), gp, default_opset);
}

} // namespace nnr
