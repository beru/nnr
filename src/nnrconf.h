#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <complex>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <cfenv>
#include <memory>
#include <numeric>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <list>
#include <map>

#ifdef __linux__
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "enchantum.hpp"

// Compiler hints
#ifdef _MSC_VER
#define NNR_NOINLINE __declspec(noinline)
#else
#define NNR_NOINLINE __attribute__((noinline))
#endif

// ONNX spec supports up to 8 tensor dimensions (N, C, D1..D6).
static constexpr int MAX_NDIM = 8;

template<typename T, int N = MAX_NDIM>
struct small_vector {
    T data_[N] = {};
    int8_t size_ = 0;

    small_vector() = default;
    explicit small_vector(int n) : size_(static_cast<int8_t>(n)) { assert(n >= 0 && n <= N); }
    small_vector(std::initializer_list<T> il) : size_(static_cast<int8_t>(il.size())) {
        assert((int)il.size() <= N);
        std::copy(il.begin(), il.end(), data_);
    }

    T& operator[](int i) { return data_[i]; }
    const T& operator[](int i) const { return data_[i]; }
    T* data() { return data_; }
    const T* data() const { return data_; }
    static constexpr int capacity() { return N; }
    int size() const { return size_; }
    bool empty() const { return size_ == 0; }
    T* begin() { return data_; }
    T* end() { return data_ + size_; }
    const T* begin() const { return data_; }
    const T* end() const { return data_ + size_; }
    T& back() { return data_[size_ - 1]; }
    const T& back() const { return data_[size_ - 1]; }
    // resize with optional fill value; asserts n fits within fixed capacity.
    void resize(int n, const T& val = T{}) {
        assert(n >= 0 && n <= N);
        for (int i = size_; i < n; ++i) data_[i] = val;
        size_ = static_cast<int8_t>(n);
    }
    void push_back(const T& v) { assert(size_ < N); data_[size_++] = v; }
    void clear() { size_ = 0; }
    void assign(const T* first, const T* last) {
        int n = static_cast<int>(last - first);
        assert(n >= 0 && n <= N);
        size_ = static_cast<int8_t>(n);
        std::copy(first, last, data_);
    }
    operator std::span<const T>() const { return {data_, static_cast<size_t>(size_)}; }
    operator std::span<T>() { return {data_, static_cast<size_t>(size_)}; }
};

using std::min;
using std::max;
using std::clamp;

/*
 * little or big endian
 */
#define NNR_LITTLE_ENDIAN   (1)

#ifdef NNR_LITTLE_ENDIAN
#undef WORDS_BIGENDIAN
#else
#define WORDS_BIGENDIAN
#endif

static inline constexpr uint16_t __swab16(uint16_t x)
{
    return std::byteswap(x);
}

static inline constexpr uint32_t __swab32(uint32_t x)
{
    return std::byteswap(x);
}

static inline constexpr uint64_t __swab64(uint64_t x)
{
    return std::byteswap(x);
}

#ifdef NNR_LITTLE_ENDIAN
#define cpu_to_le64(x)      ((uint64_t)(x))
#define le64_to_cpu(x)      ((uint64_t)(x))
#define cpu_to_le32(x)      ((uint32_t)(x))
#define le32_to_cpu(x)      ((uint32_t)(x))
#define cpu_to_le16(x)      ((uint16_t)(x))
#define le16_to_cpu(x)      ((uint16_t)(x))
#define cpu_to_be64(x)      (__swab64((uint64_t)(x)))
#define be64_to_cpu(x)      (__swab64((uint64_t)(x)))
#define cpu_to_be32(x)      (__swab32((uint32_t)(x)))
#define be32_to_cpu(x)      (__swab32((uint32_t)(x)))
#define cpu_to_be16(x)      (__swab16((uint16_t)(x)))
#define be16_to_cpu(x)      (__swab16((uint16_t)(x)))
#else
#define cpu_to_le64(x)      (__swab64((uint64_t)(x)))
#define le64_to_cpu(x)      (__swab64((uint64_t)(x)))
#define cpu_to_le32(x)      (__swab32((uint32_t)(x)))
#define le32_to_cpu(x)      (__swab32((uint32_t)(x)))
#define cpu_to_le16(x)      (__swab16((uint16_t)(x)))
#define le16_to_cpu(x)      (__swab16((uint16_t)(x)))
#define cpu_to_be64(x)      ((uint64_t)(x))
#define be64_to_cpu(x)      ((uint64_t)(x))
#define cpu_to_be32(x)      ((uint32_t)(x))
#define be32_to_cpu(x)      ((uint32_t)(x))
#define cpu_to_be16(x)      ((uint16_t)(x))
#define be16_to_cpu(x)      ((uint16_t)(x))
#endif

/*
 * float16, bfloat16 and float32 conversion
 */
static inline uint16_t float32_to_float16(float v)
{
    uint32_t u = std::bit_cast<uint32_t>(v);
    uint16_t sign = (u >> 16) & 0x8000;
    int exp = (int)((u >> 23) & 0xff) - 127;
    uint32_t mant = u & 0x7fffff;

    if (exp > 15) {
        return sign | 0x7c00; // infinity
    } else if (exp >= -14) {
        uint32_t round = mant & 0x1fff;
        uint16_t y = sign | ((uint16_t)(exp + 15) << 10) | (uint16_t)(mant >> 13);
        if (round > 0x1000 || (round == 0x1000 && (y & 1)))
            y++;
        return y;
    } else if (exp >= -24) {
        mant |= 0x800000;
        int shift = -exp - 14 + 13;
        uint32_t round = mant & ((1u << shift) - 1);
        uint32_t half = 1u << (shift - 1);
        uint16_t y = sign | (uint16_t)(mant >> shift);
        if (round > half || (round == half && (y & 1)))
            y++;
        return y;
    }
    return sign;
}

static inline float float16_to_float32(uint16_t v)
{
    uint32_t sign = (v & 0x8000) << 16;
    uint32_t exp = (v >> 10) & 0x1f;
    uint32_t mant = v & 0x3ff;

    if (exp == 0) {
        if (mant == 0) {
            return std::bit_cast<float>(sign);
        }
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        mant &= 0x3ff;
        exp++;
        uint32_t u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        return std::bit_cast<float>(u);
    } else if (exp == 31) {
        uint32_t u = sign | 0x7f800000 | (mant << 13);
        return std::bit_cast<float>(u);
    }
    uint32_t u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    return std::bit_cast<float>(u);
}

static inline uint16_t float32_to_bfloat16(float v)
{
    uint32_t u = std::bit_cast<uint32_t>(v);
    uint32_t rounding_bias = ((u >> 16) & 1) + 0x7FFF;
    return (uint16_t)((u + rounding_bias) >> 16);
}

static inline float bfloat16_to_float32(uint16_t v)
{
    uint32_t u = static_cast<uint32_t>(v) << 16;
    return std::bit_cast<float>(u);
}

/*
 * INT4/UINT4 nibble pack/unpack
 */
static inline void int4_unpack(const uint8_t* packed, int8_t* out, size_t ndata)
{
    for (size_t i = 0; i < ndata; ++i) {
        uint8_t byte = packed[i >> 1];
        int8_t nibble = (int8_t)((i & 1) ? (byte >> 4) : (byte & 0x0F));
        out[i] = (nibble & 0x08) ? (nibble | (int8_t)0xF0) : nibble;
    }
}

static inline void uint4_unpack(const uint8_t* packed, uint8_t* out, size_t ndata)
{
    for (size_t i = 0; i < ndata; ++i) {
        uint8_t byte = packed[i >> 1];
        out[i] = (i & 1) ? (byte >> 4) : (byte & 0x0F);
    }
}

static inline void int2_unpack(const uint8_t* packed, int8_t* out, size_t ndata)
{
    for (size_t i = 0; i < ndata; ++i) {
        uint8_t byte = packed[i >> 2];
        int shift = (int)((i & 3) * 2);
        int8_t bits = (int8_t)((byte >> shift) & 0x3);
        out[i] = (bits & 0x2) ? (bits | (int8_t)0xFC) : bits;
    }
}

static inline void uint2_unpack(const uint8_t* packed, uint8_t* out, size_t ndata)
{
    for (size_t i = 0; i < ndata; ++i) {
        uint8_t byte = packed[i >> 2];
        int shift = (int)((i & 3) * 2);
        out[i] = (byte >> shift) & 0x3;
    }
}

static inline void int2_pack(const int8_t* in, uint8_t* packed, size_t ndata)
{
    size_t nbytes = (ndata + 3) / 4;
    memset(packed, 0, nbytes);
    for (size_t i = 0; i < ndata; ++i) {
        int shift = (int)((i & 3) * 2);
        packed[i >> 2] |= (uint8_t)((in[i] & 0x3) << shift);
    }
}

static inline void uint2_pack(const uint8_t* in, uint8_t* packed, size_t ndata)
{
    size_t nbytes = (ndata + 3) / 4;
    memset(packed, 0, nbytes);
    for (size_t i = 0; i < ndata; ++i) {
        int shift = (int)((i & 3) * 2);
        packed[i >> 2] |= (uint8_t)((in[i] & 0x3) << shift);
    }
}

static inline void int4_pack(const int8_t* in, uint8_t* packed, size_t ndata)
{
    size_t nbytes = (ndata + 1) / 2;
    memset(packed, 0, nbytes);
    for (size_t i = 0; i < ndata; ++i) {
        uint8_t nibble = (uint8_t)(in[i]) & 0x0F;
        if (i & 1)
            packed[i >> 1] |= (uint8_t)(nibble << 4);
        else
            packed[i >> 1] = nibble;
    }
}

static inline void uint4_pack(const uint8_t* in, uint8_t* packed, size_t ndata)
{
    size_t nbytes = (ndata + 1) / 2;
    memset(packed, 0, nbytes);
    for (size_t i = 0; i < ndata; ++i) {
        uint8_t nibble = in[i] & 0x0F;
        if (i & 1)
            packed[i >> 1] |= (uint8_t)(nibble << 4);
        else
            packed[i >> 1] = nibble;
    }
}

/*
 * Float8 encode/decode
 */
static inline uint32_t rte_shift(uint32_t val, int rshift) {
    uint32_t half = 1u << (rshift - 1);
    uint32_t rem = val & ((1u << rshift) - 1);
    uint32_t trunc = val >> rshift;
    if (rem > half) return trunc + 1;
    if (rem == half) return trunc + (trunc & 1);
    return trunc;
}

static inline float float8e4m3fn_to_float32(uint8_t v)
{
    uint32_t sign = (uint32_t)(v & 0x80) << 24;
    uint8_t exp8 = (v >> 3) & 0xF;
    uint8_t mant8 = v & 0x7;
    if (exp8 == 0xF && mant8 == 7)
        return std::bit_cast<float>(sign | 0x7FC00000u);
    if (exp8 == 0) {
        if (mant8 == 0) return std::bit_cast<float>(sign);
        int sh = std::bit_width((unsigned)mant8) - 1;
        uint32_t f32_mant = (uint32_t)(mant8 ^ (1u << sh)) << (23 - sh);
        uint32_t f32_exp = (uint32_t)(118 + sh) << 23;
        return std::bit_cast<float>(sign | f32_exp | f32_mant);
    }
    return std::bit_cast<float>(sign | ((uint32_t)(exp8 + 120) << 23) | ((uint32_t)mant8 << 20));
}

static inline uint8_t float32_to_float8e4m3fn(float v, bool saturate = true)
{
    if (std::isnan(v)) return 0x7F;
    uint32_t u = std::bit_cast<uint32_t>(v);
    uint8_t sign = (u >> 24) & 0x80;
    u &= 0x7FFFFFFFu;
    if (u == 0) return sign;
    int f32_biased = (int)(u >> 23);
    if (f32_biased == 255)
        return saturate ? (sign | 0x7Eu) : (sign | 0x7Fu);
    uint32_t f32_mant = u & 0x7FFFFFu;
    int f8_biased = f32_biased - 120;
    if (f8_biased > 15)
        return saturate ? (sign | 0x7Eu) : (sign | 0x7Fu);
    if (f8_biased <= 0) {
        int rshift = 21 - f8_biased;
        if (rshift >= 25) return sign;
        uint32_t sig = 0x800000u | f32_mant;
        uint32_t m = rte_shift(sig, rshift);
        if (m >= 8u) return sign | (1u << 3);
        return sign | (uint8_t)m;
    }
    uint8_t exp8 = (uint8_t)f8_biased;
    uint32_t m = rte_shift(f32_mant, 20);
    if (m >= 8u) { m = 0; exp8++; }
    if (exp8 > 15u) return saturate ? (sign | 0x7Eu) : (sign | 0x7Fu);
    if (exp8 == 15u && m == 7u)
        return saturate ? (sign | 0x7Eu) : (sign | 0x7Fu);
    return sign | (exp8 << 3) | (uint8_t)m;
}

static inline float float8e4m3fnuz_to_float32(uint8_t v)
{
    if (v == 0x80) return std::bit_cast<float>(0x7FC00000u);
    uint32_t sign = (uint32_t)(v & 0x80) << 24;
    uint8_t exp8 = (v >> 3) & 0xF;
    uint8_t mant8 = v & 0x7;
    if (exp8 == 0) {
        if (mant8 == 0) return std::bit_cast<float>(sign);
        int sh = std::bit_width((unsigned)mant8) - 1;
        uint32_t f32_mant = (uint32_t)(mant8 ^ (1u << sh)) << (23 - sh);
        uint32_t f32_exp = (uint32_t)(117 + sh) << 23;
        return std::bit_cast<float>(sign | f32_exp | f32_mant);
    }
    return std::bit_cast<float>(sign | ((uint32_t)(exp8 + 119) << 23) | ((uint32_t)mant8 << 20));
}

static inline uint8_t float32_to_float8e4m3fnuz(float v, bool saturate = true)
{
    if (std::isnan(v)) return 0x80;
    uint32_t u = std::bit_cast<uint32_t>(v);
    uint8_t sign = (u >> 24) & 0x80;
    u &= 0x7FFFFFFFu;
    if (u == 0) return 0x00;
    int f32_biased = (int)(u >> 23);
    if (f32_biased == 255)
        return saturate ? (sign | 0x7Fu) : 0x80u;
    uint32_t f32_mant = u & 0x7FFFFFu;
    int f8_biased = f32_biased - 119;
    if (f8_biased > 15)
        return saturate ? (sign | 0x7Fu) : 0x80u;
    if (f8_biased <= 0) {
        int rshift = 21 - f8_biased;
        if (rshift >= 25) return 0x00;
        uint32_t sig = 0x800000u | f32_mant;
        uint32_t m = rte_shift(sig, rshift);
        if (m >= 8u) return sign | (1u << 3);
        return sign | (uint8_t)m;
    }
    uint8_t exp8 = (uint8_t)f8_biased;
    uint32_t m = rte_shift(f32_mant, 20);
    if (m >= 8u) { m = 0; exp8++; }
    if (exp8 > 15u) return saturate ? (sign | 0x7Fu) : 0x80u;
    return sign | (exp8 << 3) | (uint8_t)m;
}

static inline float float8e5m2_to_float32(uint8_t v)
{
    uint32_t sign = (uint32_t)(v & 0x80) << 24;
    uint8_t exp8 = (v >> 2) & 0x1F;
    uint8_t mant8 = v & 0x3;
    if (exp8 == 0x1F) {
        if (mant8 == 0) return std::bit_cast<float>(sign | 0x7F800000u);
        return std::bit_cast<float>(sign | 0x7FC00000u);
    }
    if (exp8 == 0) {
        if (mant8 == 0) return std::bit_cast<float>(sign);
        int sh = std::bit_width((unsigned)mant8) - 1;
        uint32_t f32_mant = (uint32_t)(mant8 ^ (1u << sh)) << (23 - sh);
        uint32_t f32_exp = (uint32_t)(111 + sh) << 23;
        return std::bit_cast<float>(sign | f32_exp | f32_mant);
    }
    return std::bit_cast<float>(sign | ((uint32_t)(exp8 + 112) << 23) | ((uint32_t)mant8 << 21));
}

static inline uint8_t float32_to_float8e5m2(float v, bool saturate = true)
{
    if (std::isnan(v)) return 0x7Eu;
    uint32_t u = std::bit_cast<uint32_t>(v);
    uint8_t sign = (u >> 24) & 0x80;
    u &= 0x7FFFFFFFu;
    if (u == 0) return sign;
    int f32_biased = (int)(u >> 23);
    if (f32_biased == 255)
        return saturate ? (sign | 0x7Bu) : (sign | 0x7Cu);
    uint32_t f32_mant = u & 0x7FFFFFu;
    int f8_biased = f32_biased - 112;
    if (f8_biased > 30)
        return saturate ? (sign | 0x7Bu) : (sign | 0x7Cu);
    if (f8_biased <= 0) {
        int rshift = 22 - f8_biased;
        if (rshift >= 26) return sign;
        uint32_t sig = 0x800000u | f32_mant;
        uint32_t m = rte_shift(sig, rshift);
        if (m >= 4u) return sign | (1u << 2);
        return sign | (uint8_t)m;
    }
    uint8_t exp8 = (uint8_t)f8_biased;
    uint32_t m = rte_shift(f32_mant, 21);
    if (m >= 4u) { m = 0; exp8++; }
    if (exp8 > 30u)
        return saturate ? (sign | 0x7Bu) : (sign | 0x7Cu);
    return sign | (exp8 << 2) | (uint8_t)m;
}

static inline float float4e2m1_to_float32(uint8_t v)
{
    v &= 0x0F;
    uint8_t s = v >> 3;
    uint8_t e = (v >> 1) & 0x3;
    uint8_t m = v & 0x1;
    if (e == 0) {
        float val = m ? 0.5f : 0.0f;
        return s ? -val : val;
    }
    float val = (1.0f + m * 0.5f) * (float)(1 << (e - 1));
    return s ? -val : val;
}

static inline uint8_t float32_to_float4e2m1(float v)
{
    if (std::isnan(v)) return 0x8;
    uint8_t sign = std::signbit(v) ? 0x8 : 0;
    if (v == 0.0f) return sign;
    float av = std::abs(v);
    uint8_t code;
    if      (av < 0.25f)  code = 0;
    else if (av <= 0.25f) code = 0;
    else if (av < 0.75f)  code = 1;
    else if (av <= 0.75f) code = 2;
    else if (av < 1.25f)  code = 2;
    else if (av <= 1.25f) code = 2;
    else if (av < 1.75f)  code = 3;
    else if (av <= 1.75f) code = 4;
    else if (av < 2.5f)   code = 4;
    else if (av <= 2.5f)  code = 4;
    else if (av < 3.5f)   code = 5;
    else if (av <= 3.5f)  code = 6;
    else if (av < 5.0f)   code = 6;
    else if (av <= 5.0f)  code = 6;
    else if (av < 7.0f)   code = 7;
    else                  code = 7;
    return sign | code;
}

static inline float float8e8m0_to_float32(uint8_t v)
{
    if (v == 0xFF) return std::bit_cast<float>(0x7FC00000u);
    if (v == 0)   return std::bit_cast<float>(0x00400000u);
    return std::bit_cast<float>((uint32_t)v << 23);
}

static inline uint8_t float32_to_float8e8m0(float v)
{
    if (std::isnan(v)) return 0xFF;
    uint32_t bits = std::bit_cast<uint32_t>(std::abs(v));
    uint8_t biased_exp = (uint8_t)((bits >> 23) & 0xFF);
    uint32_t mant = bits & 0x7FFFFF;
    if (mant == 0) return biased_exp;
    return biased_exp < 0xFE ? (uint8_t)(biased_exp + 1) : (uint8_t)0xFE;
}

static inline float float8e5m2fnuz_to_float32(uint8_t v)
{
    if (v == 0x80) return std::bit_cast<float>(0x7FC00000u);
    uint32_t sign = (uint32_t)(v & 0x80) << 24;
    uint8_t exp8 = (v >> 2) & 0x1F;
    uint8_t mant8 = v & 0x3;
    if (exp8 == 0) {
        if (mant8 == 0) return std::bit_cast<float>(sign);
        int sh = std::bit_width((unsigned)mant8) - 1;
        uint32_t f32_mant = (uint32_t)(mant8 ^ (1u << sh)) << (23 - sh);
        uint32_t f32_exp = (uint32_t)(110 + sh) << 23;
        return std::bit_cast<float>(sign | f32_exp | f32_mant);
    }
    return std::bit_cast<float>(sign | ((uint32_t)(exp8 + 111) << 23) | ((uint32_t)mant8 << 21));
}

static inline uint8_t float32_to_float8e5m2fnuz(float v, bool saturate = true)
{
    if (std::isnan(v)) return 0x80;
    uint32_t u = std::bit_cast<uint32_t>(v);
    uint8_t sign = (u >> 24) & 0x80;
    u &= 0x7FFFFFFFu;
    if (u == 0) return 0x00;
    int f32_biased = (int)(u >> 23);
    if (f32_biased == 255)
        return saturate ? (sign | 0x7Fu) : 0x80u;
    uint32_t f32_mant = u & 0x7FFFFFu;
    int f8_biased = f32_biased - 111;
    if (f8_biased > 31)
        return saturate ? (sign | 0x7Fu) : 0x80u;
    if (f8_biased <= 0) {
        int rshift = 22 - f8_biased;
        if (rshift >= 26) return 0x00;
        uint32_t sig = 0x800000u | f32_mant;
        uint32_t m = rte_shift(sig, rshift);
        if (m >= 4u) return sign | (1u << 2);
        return sign | (uint8_t)m;
    }
    uint8_t exp8 = (uint8_t)f8_biased;
    uint32_t m = rte_shift(f32_mant, 21);
    if (m >= 4u) { m = 0; exp8++; }
    if (exp8 > 31u) return saturate ? (sign | 0x7Fu) : 0x80u;
    return sign | (exp8 << 2) | (uint8_t)m;
}
