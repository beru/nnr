#pragma once

#include <stdint.h>
#include "nnrconf.h"

struct bfloat16_t final
{
    bfloat16_t(){}
    bfloat16_t(float f) : val(float32_to_bfloat16(f)) {}
    operator float() const { return bfloat16_to_float32(val); }

    bfloat16_t& operator += (bfloat16_t v) {
        float f = *this;
        f += v;
        *this = f;
        return *this;
    }

    bfloat16_t& operator -= (bfloat16_t v) {
        float f = *this;
        f -= v;
        *this = f;
        return *this;
    }

    bfloat16_t& operator *= (bfloat16_t v) {
        float f = *this;
        f *= v;
        *this = f;
        return *this;
    }

    bfloat16_t& operator /= (bfloat16_t v) {
        float f = *this;
        f /= v;
        *this = f;
        return *this;
    }

    uint16_t val;
};

inline bfloat16_t operator + (bfloat16_t a, bfloat16_t b) { a += b; return a; }


inline bfloat16_t abs(bfloat16_t v) { return fabs(v); }
inline bfloat16_t acos(bfloat16_t v) { return acosf(v); }
inline bfloat16_t acosh(bfloat16_t v) { return acoshf(v); }
inline bool isinf(bfloat16_t v) { return std::isinf((float)v); }
inline bool isnan(bfloat16_t v) { return std::isnan((float)v); }

#include <limits>
template<> class std::numeric_limits<bfloat16_t> {
public:
    static constexpr bool is_specialized = true;
    static bfloat16_t lowest() { return bfloat16_t(-3.389531e+38f); }
    static bfloat16_t max() { return bfloat16_t(3.389531e+38f); }
    static bfloat16_t min() { return bfloat16_t(1.175494e-38f); }
};
