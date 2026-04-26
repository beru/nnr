#pragma once

#include <stdint.h>
#include "nnrconf.h"

// On GCC/Clang aarch64 the compiler provides a native IEEE-754 binary16 type
// (`__fp16`), and `<arm_neon.h>` already uses the name `float16_t` as a typedef
// for it — defining a conflicting struct here causes a redeclaration error the
// moment any NEON header is pulled in. Use the native type on those platforms
// and rely on built-in arithmetic (implicit promotion to float) instead of the
// struct's software-conversion wrappers. MSVC (x64 and ARM64) lacks a usable
// `__fp16` so it continues to use the software struct.
#if defined(__aarch64__) && !defined(_MSC_VER)

using float16_t = __fp16;

#else

struct float16_t final
{
    float16_t(){}
    float16_t(float f) : val(float32_to_float16(f)) {}
    operator float() const { return float16_to_float32(val); }

    float16_t& operator += (float16_t v) {
        float f = *this;
        f += v;
        *this = f;
        return *this;
    }

    float16_t& operator -= (float16_t v) {
        float f = *this;
        f -= v;
        *this = f;
        return *this;
    }

    float16_t& operator *= (float16_t v) {
        float f = *this;
        f *= v;
        *this = f;
        return *this;
    }

    float16_t& operator /= (float16_t v) {
        float f = *this;
        f /= v;
        *this = f;
        return *this;
    }

    uint16_t val;
};

inline float16_t operator + (float16_t a, float16_t b) { a += b; return a; }

inline float16_t abs(float16_t v) { return fabs(v); }
inline float16_t acos(float16_t v) { return acosf(v); }
inline float16_t acosh(float16_t v) { return acoshf(v); }
inline float16_t pow(float16_t base, float16_t exponent) { return powf(base, exponent); }
inline bool isinf(float16_t v) { return std::isinf((float)v); }
inline bool isnan(float16_t v) { return std::isnan((float)v); }

#endif

#include <limits>
template<> class std::numeric_limits<float16_t> {
public:
    static constexpr bool is_specialized = true;
    static float16_t lowest() { return float16_t(-65504.0f); }
    static float16_t max() { return float16_t(65504.0f); }
    static float16_t min() { return float16_t(6.103515625e-5f); }
};
