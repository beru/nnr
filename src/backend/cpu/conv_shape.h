#pragma once
// Shared Conv output-shape resolver.
//
// Extracted from the byte-for-byte identical shape phases of
// Conv::reshape() and QLinearConv::reshape() during the 2026-04-10
// bloat refactor. Pure function, no allocation, no side effects beyond
// the out-params.
//
// Enum convention (matches the private auto_pad_t in Conv.cpp and
// QLinearConv.cpp — both define the same 4 values in the same order):
//   0 = NOTSET, 1 = SAME_UPPER, 2 = SAME_LOWER, 3 = VALID
//
// Callers pass raw pointers so the helper is agnostic to the container
// type used for kernels/dilations/strides/pads (small_vector vs std::vector).

#include <cmath>
#include <cstring>

namespace nnr {

inline void compute_conv_output_shape(
    int auto_pad,
    const int* x_dims, int ndim,
    int w_out_channels,
    const int* kernels, int nk,
    const int* dilations,
    const int* strides,
    const int* pads,
    int* cpads,           // out: [nk*2]
    int* dims)            // out: [ndim]
{
    const size_t pad_bytes = sizeof(int) * (size_t)(nk * 2);
    switch (auto_pad) {
    case 0: // NOTSET
        memcpy(cpads, pads, pad_bytes);
        break;
    case 1: // SAME_UPPER
        for (int i = 0; i < nk; ++i) {
            int ek = (kernels[i] - 1) * dilations[i] + 1;
            int pad = (int)(ceilf(x_dims[i+2] / (float)strides[i]) - 1) * strides[i] + ek - x_dims[i+2];
            cpads[i] = pad / 2;
            cpads[i + nk] = pad - cpads[i];
        }
        break;
    case 2: // SAME_LOWER
        for (int i = 0; i < nk; ++i) {
            int ek = (kernels[i] - 1) * dilations[i] + 1;
            int pad = (int)(ceilf(x_dims[i+2] / (float)strides[i]) - 1) * strides[i] + ek - x_dims[i+2];
            cpads[i + nk] = pad / 2;
            cpads[i] = pad - cpads[i + nk];
        }
        break;
    case 3: // VALID
        memset(cpads, 0, pad_bytes);
        break;
    default:
        break;
    }

    dims[0] = x_dims[0];
    dims[1] = w_out_channels;
    for (int i = 0; i < ndim - 2; ++i) {
        int ek = (kernels[i] - 1) * dilations[i] + 1;
        switch (auto_pad) {
        case 0: // NOTSET
            dims[i+2] = (int)floorf((x_dims[i+2] + cpads[i] + cpads[i+nk] - ek) / (float)strides[i] + 1);
            break;
        case 1: // SAME_UPPER
        case 2: // SAME_LOWER
            dims[i+2] = (int)ceilf(x_dims[i+2] / (float)strides[i]);
            break;
        case 3: // VALID
            dims[i+2] = (int)ceilf((x_dims[i+2] - ek + 1) / (float)strides[i]);
            break;
        default:
            break;
        }
    }
}

} // namespace nnr
