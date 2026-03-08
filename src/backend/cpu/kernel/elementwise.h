#pragma once
#ifdef NNR_ARCH_X64
#include "backend/x64/elementwise_x64.h"
#endif
#include <algorithm>
#include <cfloat>

namespace nnr {

inline void relu_inplace(float* data, int len) {
#ifdef NNR_ARCH_X64
    relu_inplace_x64(data, len);
#else
    for (int i = 0; i < len; ++i)
        if (data[i] < 0.0f) data[i] = 0.0f;
#endif
}

inline void clip_inplace(float* data, int len, float lo, float hi) {
#ifdef NNR_ARCH_X64
    clip_inplace_x64(data, len, lo, hi);
#else
    for (int i = 0; i < len; ++i)
        data[i] = std::max(lo, std::min(hi, data[i]));
#endif
}

inline void leaky_relu_inplace(float* data, int len, float alpha) {
#ifdef NNR_ARCH_X64
    leaky_relu_inplace_x64(data, len, alpha);
#else
    for (int i = 0; i < len; ++i)
        if (data[i] < 0.0f) data[i] *= alpha;
#endif
}

inline void sigmoid_inplace(float* data, int len) {
    for (int i = 0; i < len; ++i)
        data[i] = 1.0f / (1.0f + expf(-data[i]));
}

inline void add_elementwise(float* dst, const float* a, const float* b, int len) {
    for (int i = 0; i < len; ++i)
        dst[i] = a[i] + b[i];
}

inline void sub_elementwise(float* dst, const float* a, const float* b, int len) {
    for (int i = 0; i < len; ++i)
        dst[i] = a[i] - b[i];
}

inline void mul_elementwise(float* dst, const float* a, const float* b, int len) {
    for (int i = 0; i < len; ++i)
        dst[i] = a[i] * b[i];
}

inline void div_elementwise(float* dst, const float* a, const float* b, int len) {
    for (int i = 0; i < len; ++i)
        dst[i] = a[i] / b[i];
}

} // namespace nnr
