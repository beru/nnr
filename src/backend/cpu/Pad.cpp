#include "nnr.h"
#include "util.h"
#include "thread_pool.h"

namespace nnr {

namespace {

struct Pad_operator : public operator_t {
    enum class pad_mode_t { constant, reflect, edge, wrap };
    pad_mode_t mode;
    // For opset < 11, pads from attributes
    small_vector<int, MAX_NDIM * 2> attr_pads;
    float attr_value;

    bool init() override {
        if (!(inputs.size() >= 1 && outputs.size() == 1)) {
            return false;
        }
        auto mode_str = attribute(attr_key_t::mode, "constant");
        if (mode_str == "reflect") {
            mode = pad_mode_t::reflect;
        } else if (mode_str == "edge") {
            mode = pad_mode_t::edge;
        } else if (mode_str == "wrap") {
            mode = pad_mode_t::wrap;
        } else {
            mode = pad_mode_t::constant;
        }

        // For opset < 11, pads and value come from attributes
        if (opset < 11) {
            int64_t* ints;
            int n = attribute(attr_key_t::pads, ints);
            if (n > 0) {
                attr_pads.resize(n);
                for (int i = 0; i < n; ++i) {
                    attr_pads[i] = static_cast<int>(ints[i]);
                }
            }
            attr_value = attribute(attr_key_t::value, 0.0f);
        }
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const int ndim = x->ndim;
        small_vector<int> dims(ndim);

        if (opset >= 11) {
            // pads from input tensor
            if (inputs.size() < 2 || !inputs[1] || inputs[1]->ndata == 0 || !inputs[1]->data) {
                return false;
            }
            const int64_t* ppads = (const int64_t*)inputs[1]->data;

            // axes input (opset >= 18)
            if (opset >= 18 && inputs.size() >= 4 && inputs[3] && inputs[3]->ndata > 0) {
                const int64_t* paxes = (const int64_t*)inputs[3]->data;
                int naxes = static_cast<int>(inputs[3]->ndata);
                for (int j = 0; j < ndim; ++j) {
                    dims[j] = x->dims[j];
                }
                for (int i = 0; i < naxes; ++i) {
                    int a = static_cast<int>(paxes[i]);
                    if (a < 0) a += ndim;
                    if (a < 0 || a >= ndim) return false;
                    dims[a] += static_cast<int>(ppads[i]) + static_cast<int>(ppads[i + naxes]);
                }
            } else {
                // pads format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
                if (static_cast<int>(inputs[1]->ndata) != ndim * 2) {
                    return false;
                }
                for (int i = 0; i < ndim; ++i) {
                    dims[i] = x->dims[i] + static_cast<int>(ppads[i]) + static_cast<int>(ppads[i + ndim]);
                }
            }
        } else {
            // opset < 11: pads from attributes
            if (static_cast<int>(attr_pads.size()) != ndim * 2) {
                return false;
            }
            for (int i = 0; i < ndim; ++i) {
                dims[i] = x->dims[i] + attr_pads[i] + attr_pads[i + ndim];
            }
        }

        return y->reshape(dims, x->type);
    }

    // Map a single index through the padding mode
    int map_index(int idx, int dim) const {
        if (idx >= 0 && idx < dim) return idx;
        switch (mode) {
        case pad_mode_t::edge:
            return std::clamp(idx, 0, dim - 1);
        case pad_mode_t::reflect:
            while (idx < 0 || idx >= dim) {
                if (idx < 0) idx = -idx;
                if (idx >= dim) idx = 2 * (dim - 1) - idx;
            }
            return idx;
        case pad_mode_t::wrap:
            idx = idx % dim;
            if (idx < 0) idx += dim;
            return idx;
        default:
            return -1;
        }
    }

    void get_pads(small_vector<int, MAX_NDIM * 2>& pads) const {
        int ndim = inputs[0]->ndim;
        pads.resize(ndim * 2);
        if (opset >= 11) {
            const int64_t* ppads = (const int64_t*)inputs[1]->data;
            if (opset >= 18 && inputs.size() >= 4 && inputs[3] && inputs[3]->ndata > 0) {
                const int64_t* paxes = (const int64_t*)inputs[3]->data;
                int naxes = static_cast<int>(inputs[3]->ndata);
                for (int i = 0; i < ndim * 2; ++i) pads[i] = 0;
                for (int i = 0; i < naxes; ++i) {
                    int a = static_cast<int>(paxes[i]);
                    if (a < 0) a += ndim;
                    pads[a] = static_cast<int>(ppads[i]);
                    pads[a + ndim] = static_cast<int>(ppads[i + naxes]);
                }
            } else {
                for (int i = 0; i < ndim * 2; ++i)
                    pads[i] = static_cast<int>(ppads[i]);
            }
        } else {
            for (int i = 0; i < ndim * 2; ++i)
                pads[i] = attr_pads[i];
        }
    }

    template <typename T>
    T get_cval() const {
        T cval{};
        if (mode == pad_mode_t::constant) {
            if (opset >= 11 && inputs.size() >= 3 && inputs[2] && inputs[2]->ndata > 0)
                cval = *(const T*)inputs[2]->data;
            else if (opset < 11)
                cval = T(attr_value);
        }
        return cval;
    }

    // Fast path for 4D tensors with zero padding on batch/channel dims.
    // Avoids per-element offset_to_indices/indices_to_offset entirely.
    template <typename T>
    bool exec_4d(const small_vector<int, MAX_NDIM * 2>& pads) {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;

        // All-zero pads: pure copy, single memcpy
        bool all_zero = true;
        for (int i = 0; i < (int)pads.size(); ++i)
            if (pads[i] != 0) { all_zero = false; break; }
        if (all_zero) {
            if (px != py)
                memcpy(py, px, (size_t)x->dims[0] * x->dims[1] * x->dims[2] * x->dims[3] * sizeof(T));
            return true;
        }

        // Only handle spatial-only padding (dims 0,1 unpadded) or
        // any constant-mode padding with threading
        const bool spatial_only = !(pads[0] || pads[1] || pads[4] || pads[5]);

        int N = x->dims[0], C = x->dims[1];
        int iH = x->dims[2], iW = x->dims[3];
        int oH = y->dims[2], oW = y->dims[3];
        int pH0 = pads[2], pW0 = pads[3];
        int NC = N * C;

        if (mode == pad_mode_t::constant && spatial_only) {
            T cval = get_cval<T>();
            nnr::for_static(0, NC, NC > 1, [&](int nc) {
                T* out = py + (size_t)nc * oH * oW;
                const T* inp = px + (size_t)nc * iH * iW;
                for (int h = 0; h < pH0; h++)
                    std::fill_n(out + h * oW, oW, cval);
                for (int h = 0; h < iH; h++) {
                    T* row = out + (pH0 + h) * oW;
                    std::fill_n(row, pW0, cval);
                    memcpy(row + pW0, inp + h * iW, iW * sizeof(T));
                    std::fill_n(row + pW0 + iW, oW - pW0 - iW, cval);
                }
                int bot_start = pH0 + iH;
                for (int h = bot_start; h < oH; h++)
                    std::fill_n(out + h * oW, oW, cval);
            });
        } else if (mode == pad_mode_t::constant && !spatial_only) {
            // General constant-mode padding (pads on any dims): threaded
            T cval = get_cval<T>();
            int oN = y->dims[0], oC = y->dims[1];
            int pN0 = pads[0], pC0 = pads[1];
            int pN1 = pads[4], pC1 = pads[5];
            int xN = x->dims[0], xC = x->dims[1];
            nnr::for_static(0, oN * oC, oN * oC > 1, [&](int oc) {
                int on = oc / oC, oc_ = oc % oC;
                T* out = py + (size_t)oc * oH * oW;
                int in_ = on - pN0, ic_ = oc_ - pC0;
                if (in_ < 0 || in_ >= xN || ic_ < 0 || ic_ >= xC) {
                    std::fill_n(out, oH * oW, cval);
                    return;
                }
                const T* inp = px + ((size_t)in_ * xC + ic_) * iH * iW;
                for (int h = 0; h < pH0; h++)
                    std::fill_n(out + h * oW, oW, cval);
                for (int h = 0; h < iH; h++) {
                    T* row = out + (pH0 + h) * oW;
                    std::fill_n(row, pW0, cval);
                    memcpy(row + pW0, inp + h * iW, iW * sizeof(T));
                    std::fill_n(row + pW0 + iW, oW - pW0 - iW, cval);
                }
                for (int h = pH0 + iH; h < oH; h++)
                    std::fill_n(out + h * oW, oW, cval);
            });
        } else if (!spatial_only) {
            return false;  // non-constant with N/C padding: fall through to generic
        } else {
            // reflect/edge/wrap: precompute index maps
            std::vector<int> row_map(oH), col_map(oW);
            for (int h = 0; h < oH; h++)
                row_map[h] = map_index(h - pH0, iH);
            for (int w = 0; w < oW; w++)
                col_map[w] = map_index(w - pW0, iW);

            bool interior_contiguous = (pW0 >= 0 && pW0 + iW <= oW);
            nnr::for_static(0, NC, NC > 4, [&](int nc) {
                T* out = py + (size_t)nc * oH * oW;
                const T* inp = px + (size_t)nc * iH * iW;
                for (int oh = 0; oh < oH; oh++) {
                    const T* src_row = inp + row_map[oh] * iW;
                    T* dst_row = out + oh * oW;
                    for (int w = 0; w < pW0; w++)
                        dst_row[w] = src_row[col_map[w]];
                    if (interior_contiguous)
                        memcpy(dst_row + pW0, src_row, iW * sizeof(T));
                    else
                        for (int w = pW0; w < pW0 + iW; w++)
                            dst_row[w] = src_row[col_map[w]];
                    for (int w = pW0 + iW; w < oW; w++)
                        dst_row[w] = src_row[col_map[w]];
                }
            });
        }
        return true;
    }

    scroll_info_t scroll_info() const override {
        if (inputs[0]->ndim != 4) return {};
        if (mode == pad_mode_t::wrap) return {};

        small_vector<int, MAX_NDIM * 2> p;
        get_pads(p);
        if ((int)p.size() < 8) return {};
        // Only support zero padding on N,C dims
        if (p[0] || p[1] || p[4] || p[5]) return {};

        return {
            .scrollable = true,
            .halo_top = p[2],
            .halo_bottom = p[6],
            .stride_h = 1,
        };
    }

    bool exec_strip(int out_row_start, int out_rows,
        int in_row_start, int in_rows) override
    {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return false;

        small_vector<int, MAX_NDIM * 2> p;
        get_pads(p);

        int N = x->dims[0], C = x->dims[1];
        int iH = ring_in.orig_H > 0 ? ring_in.orig_H : x->dims[2];
        int iW = x->dims[3];
        int oH = ring_out.orig_H > 0 ? ring_out.orig_H : y->dims[2];
        int oW = y->dims[3];
        int pH0 = p[2], pW0 = p[3];
        int NC = N * C;

        // All-zero pads: copy rows directly
        if (pH0 == 0 && pW0 == 0 && !(p[0] || p[1] || p[4] || p[5])
            && p[6] == 0 && p[7] == 0) {
            // Row strides for addressing (may be ring_H)
            int in_stride_H = x->dims[2];
            int out_stride_H = y->dims[2];
            const float* px_f = (const float*)x->data;
            float* py_f = (float*)y->data;
            int out_end = std::min(out_row_start + out_rows, oH);
            for (int nc = 0; nc < NC; nc++) {
                const float* inp = px_f + (size_t)nc * in_stride_H * iW;
                float* out = py_f + (size_t)nc * out_stride_H * oW;
                for (int oh = out_row_start; oh < out_end; oh++)
                    memcpy(out + oh * oW, inp + oh * iW, iW * sizeof(float));
            }
            return true;
        }

        // Row strides for addressing (may be ring_H)
        int in_stride_H = x->dims[2];
        int out_stride_H = y->dims[2];

        const float* px = (const float*)x->data;
        float* py = (float*)y->data;

        int out_end = std::min(out_row_start + out_rows, oH);

        if (mode == pad_mode_t::constant) {
            float cval = get_cval<float>();
            for (int nc = 0; nc < NC; nc++) {
                const float* inp = px + (size_t)nc * in_stride_H * iW;
                float* out = py + (size_t)nc * out_stride_H * oW;
                for (int oh = out_row_start; oh < out_end; oh++) {
                    float* dst = out + oh * oW;
                    int ih = oh - pH0;
                    if (ih < 0 || ih >= iH) {
                        std::fill_n(dst, oW, cval);
                    } else {
                        const float* src = inp + ih * iW;
                        std::fill_n(dst, pW0, cval);
                        memcpy(dst + pW0, src, iW * sizeof(float));
                        std::fill_n(dst + pW0 + iW, oW - pW0 - iW, cval);
                    }
                }
            }
        } else {
            // reflect/edge: inline index mapping (pad widths are small)
            for (int nc = 0; nc < NC; nc++) {
                const float* inp = px + (size_t)nc * in_stride_H * iW;
                float* out = py + (size_t)nc * out_stride_H * oW;
                for (int oh = out_row_start; oh < out_end; oh++) {
                    float* dst = out + oh * oW;
                    int ih = oh - pH0;
                    int mapped_h = map_index(ih, iH);
                    const float* src = inp + mapped_h * iW;
                    for (int w = 0; w < pW0; w++)
                        dst[w] = src[map_index(w - pW0, iW)];
                    memcpy(dst + pW0, src, iW * sizeof(float));
                    for (int w = pW0 + iW; w < oW; w++)
                        dst[w] = src[map_index(w - pW0, iW)];
                }
            }
        }
        return true;
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const int ndim = x->ndim;

        small_vector<int, MAX_NDIM * 2> pads;
        get_pads(pads);

        // Zero-pad fast path: all pads are zero → identity copy
        {
            bool all_zero = true;
            for (int d = 0; d < ndim * 2; d++)
                if (pads[d] != 0) { all_zero = false; break; }
            if (all_zero) {
                if (x->data != y->data)
                    memcpy(y->data, x->data, x->ndata * data_type_sizeof(x));
                return true;
            }
        }

        // Fast 4D path
        if (ndim == 4 && exec_4d<T>(pads))
            return true;

        T cval = get_cval<T>();
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;

        // Generic fallback: per-element index mapping
        small_vector<int> iy(ndim);
        for (size_t oi = 0, l = y->ndata; oi < l; ++oi) {
            y->offset_to_indices(static_cast<int>(oi), iy);

            bool out_of_bounds = false;
            small_vector<int> ix(ndim);
            for (int d = 0; d < ndim; ++d) {
                int idx = iy[d] - pads[d];
                int dim = x->dims[d];

                if (idx < 0 || idx >= dim) {
                    if (mode == pad_mode_t::constant) {
                        out_of_bounds = true;
                        break;
                    }
                    ix[d] = map_index(idx, dim);
                } else {
                    ix[d] = idx;
                }
            }

            if (out_of_bounds) {
                py[oi] = cval;
            } else {
                int offset = x->indices_to_offset(ix);
                py[oi] = px[offset];
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Pad_operator,
            opset_t<13, bool_t, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float16_t, float, double, bfloat16_t, std::complex<float>, std::complex<double>>,
            opset_t<11, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float16_t, float, double, std::complex<float>, std::complex<double>>,
            opset_t<2, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=static scroll=yes
operator_t* resolver_default_op_Pad(int opset, pool_t& pool) { return pool_new<Pad_operator>(pool); }

} // namespace nnr
