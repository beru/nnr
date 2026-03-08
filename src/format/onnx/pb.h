#pragma once

#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>
#include <memory>

namespace pb {

enum WireType {
    WT_Varint         = 0,
    WT_64bit          = 1,
    WT_LengthLimited  = 2,
    WT_StartGroup     = 3,
    WT_EndGroup       = 4,
    WT_32bit          = 5,
};

// ---------------------------------------------------------------------------
// Parse context — carries buffer bounds and error state. All read operations
// are methods so callers never need to pass ctx explicitly.
// Primitive readers return values instead of using out-parameters to reduce
// template instantiations and simplify call sites.
// ---------------------------------------------------------------------------

struct ctx {
    const uint8_t* end = nullptr;
    bool error = false;

    ctx() = default;
    explicit ctx(const uint8_t* buf_end) : end(buf_end) {}

    bool out_of_bounds(const uint8_t* p, size_t n = 1) const {
        return end && (p + n > end || p + n < p);
    }

    // --- Tag ---

    std::pair<WireType, uint64_t> readTag(const uint8_t*& p) {
        uint64_t tag = 0;
        int shift = 0;
        for (int i = 0; i < 10; ++i) {
            if (out_of_bounds(p)) { error = true; return {WT_Varint, 0}; }
            uint8_t b = *p++;
            tag |= (uint64_t)(b & 0x7F) << shift;
            shift += 7;
            if (!(b & 0x80)) break;
        }
        return {(WireType)(tag & 7), tag >> 3};
    }

    // --- Skip unknown field ---

    void skip(const uint8_t*& p, WireType wt) {
        switch (wt) {
        case WT_Varint:
            for (int i = 0; i < 10; ++i) {
                if (out_of_bounds(p)) { error = true; return; }
                if (!(*p++ & 0x80)) break;
            }
            break;
        case WT_64bit:
            if (out_of_bounds(p, 8)) { error = true; return; }
            p += 8;
            break;
        case WT_LengthLimited: {
            uint64_t len = readVarint(p);
            if (error) return;
            if (out_of_bounds(p, (size_t)len)) { error = true; return; }
            p += len;
            break;
        }
        case WT_32bit:
            if (out_of_bounds(p, 4)) { error = true; return; }
            p += 4;
            break;
        default: break;
        }
    }

    // --- Varint (non-template: one implementation, caller casts) ---

    uint64_t readVarint(const uint8_t*& p) {
        uint64_t n = 0;
        int shift = 0;
        for (int i = 0; i < 10; ++i) {
            if (out_of_bounds(p)) { error = true; return 0; }
            uint8_t b = *p++;
            n |= (uint64_t)(b & 0x7F) << shift;
            shift += 7;
            if (!(b & 0x80)) break;
        }
        return n;
    }

    // --- Fixed width (type-erased: one function per width) ---

    void readFixed32(const uint8_t*& p, void* dst) {
        if (out_of_bounds(p, 4)) { error = true; memset(dst, 0, 4); return; }
        memcpy(dst, p, 4); p += 4;
    }

    void readFixed64(const uint8_t*& p, void* dst) {
        if (out_of_bounds(p, 8)) { error = true; memset(dst, 0, 8); return; }
        memcpy(dst, p, 8); p += 8;
    }

    // --- Length-delimited string ---

    std::string_view readString(const uint8_t*& p) {
        uint64_t len = readVarint(p);
        if (error) return {};
        if (out_of_bounds(p, (size_t)len)) { error = true; return {}; }
        auto sv = std::string_view((const char*)p, (size_t)len);
        p += len;
        return sv;
    }

    // --- Packed repeated (varint-encoded scalars) ---

    template <typename T>
    void readPackedVarint(const uint8_t*& p, std::vector<T>& dst) {
        uint64_t len = readVarint(p);
        if (error) return;
        if (out_of_bounds(p, (size_t)len)) { error = true; return; }
        const uint8_t* e = p + len;
        while (p < e) {
            if (error) return;
            dst.push_back((T)readVarint(p));
        }
    }

    // --- Packed repeated (fixed 32-bit) ---

    template <typename T>
    void readPackedFixed32(const uint8_t*& p, std::vector<T>& dst) {
        uint64_t len = readVarint(p);
        if (error) return;
        if (out_of_bounds(p, (size_t)len)) { error = true; return; }
        size_t count = (size_t)len / 4;
        size_t old_sz = dst.size();
        if (count > (size_t)256 * 1024 * 1024) { error = true; return; }
        dst.resize(old_sz + count);
        memcpy(&dst[old_sz], p, count * 4);
        p += count * 4;
    }

    // --- Packed repeated (fixed 64-bit) ---

    template <typename T>
    void readPackedFixed64(const uint8_t*& p, std::vector<T>& dst) {
        uint64_t len = readVarint(p);
        if (error) return;
        if (out_of_bounds(p, (size_t)len)) { error = true; return; }
        size_t count = (size_t)len / 8;
        size_t old_sz = dst.size();
        if (count > (size_t)256 * 1024 * 1024) { error = true; return; }
        dst.resize(old_sz + count);
        memcpy(&dst[old_sz], p, count * 8);
        p += count * 8;
    }

};

} // namespace pb
