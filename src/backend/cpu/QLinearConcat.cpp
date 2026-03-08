#include "nnr.h"
#include "util.h"
#include "thread_pool.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#ifdef NNR_ARCH_X64
#include <immintrin.h>
#endif

namespace nnr {

namespace {

// QLinearConcat: quantized concatenation.
// ONNX contrib op format:
//   Inputs: (Y_scale, Y_zp, T1, T1_scale, T1_zp, T2, T2_scale, T2_zp, ...)
//   where each input triple is (tensor, scale, zero_point).
//   Y_scale and Y_zp are the output quantization parameters.
// Attribute: axis (default 0).
struct QLinearConcat_operator : public operator_t {
    int axis = 0;

    bool init() override {
        // Minimum: Y_scale, Y_zp, T1, T1_scale, T1_zp = 5 inputs
        if (inputs.size() < 5 || outputs.size() != 1) return false;
        // After Y_scale, Y_zp: triples of (tensor, scale, zp) → (inputs.size()-2) % 3 == 0
        if ((inputs.size() - 2) % 3 != 0) return false;
        axis = attribute(attr_key_t::axis, 0);
        return true;
    }

    int num_tensors() const { return (int)(inputs.size() - 2) / 3; }
    const tensor_t* tensor(int i) const { return inputs[2 + i * 3]; }
    float scale(int i) const { return *(float*)inputs[2 + i * 3 + 1]->data; }
    int32_t zero_point(int i) const {
        tensor_t* zp = inputs[2 + i * 3 + 2];
        if (!zp || zp->ndata == 0) return 0;
        if (zp->type == NNR_DATA_TYPE_UINT8) return (int32_t)*(uint8_t*)zp->data;
        if (zp->type == NNR_DATA_TYPE_INT8) return (int32_t)*(int8_t*)zp->data;
        return 0;
    }

    bool reshape() override {
        int ntens = num_tensors();
        if (ntens < 1) return false;
        const tensor_t* t0 = tensor(0);
        int ndim = t0->ndim;
        int a = axis < 0 ? axis + ndim : axis;
        if (a < 0 || a >= ndim) return false;

        small_vector<int> dims(ndim);
        for (int d = 0; d < ndim; d++) dims[d] = t0->dims[d];

        // Sum along concat axis
        for (int i = 1; i < ntens; i++) {
            const tensor_t* ti = tensor(i);
            if (ti->ndim != ndim) return false;
            for (int d = 0; d < ndim; d++) {
                if (d == a)
                    dims[d] += ti->dims[d];
                else if (ti->dims[d] != t0->dims[d])
                    return false;
            }
        }
        // NHWC support: channel concat (axis=1) on 4D int8/uint8 tensors.
        // In NHWC memory, channel is innermost, so per-pixel we copy xC
        // contiguous values from each input into a yC-strided output slot.
        if (ndim == 4 && a == 1 &&
            (t0->type == NNR_DATA_TYPE_UINT8 || t0->type == NNR_DATA_TYPE_INT8))
            layout_mask = LAYOUT_NCHW | LAYOUT_NHWC;
        else
            layout_mask = LAYOUT_NCHW;
        return outputs[0]->reshape(dims, t0->type);
    }

    // Requantize a contiguous chunk: y[i] = clamp(round(rs * (x[i] - x_zp) + y_zp))
    template <typename T>
    static void requantize_chunk(T* __restrict dst, const T* __restrict src,
                                 size_t count, float rs, int32_t x_zp, int32_t y_zp,
                                 int clamp_min, int clamp_max) {
        size_t i = 0;
#if defined(NNR_ARCH_X64)
        if (count >= 16) {
            __m512 vrs = _mm512_set1_ps(rs);
            __m512 vzp_x = _mm512_set1_ps((float)x_zp);
            __m512 vzp_y = _mm512_set1_ps((float)y_zp);
            __m512 vmin = _mm512_set1_ps((float)clamp_min);
            __m512 vmax = _mm512_set1_ps((float)clamp_max);
            for (; i + 16 <= count; i += 16) {
                __m128i raw = _mm_loadu_si128((const __m128i*)(src + i));
                __m512i xi;
                if constexpr (std::is_same_v<T, uint8_t>)
                    xi = _mm512_cvtepu8_epi32(raw);
                else
                    xi = _mm512_cvtepi8_epi32(raw);
                __m512 xf = _mm512_cvtepi32_ps(xi);
                // rs * (x - x_zp) + y_zp
                __m512 val = _mm512_fmadd_ps(vrs, _mm512_sub_ps(xf, vzp_x), vzp_y);
                val = _mm512_max_ps(val, vmin);
                val = _mm512_min_ps(val, vmax);
                __m512i ri = _mm512_cvtps_epi32(val); // round to nearest (default rounding)
                if constexpr (std::is_same_v<T, uint8_t>) {
                    __m128i packed = _mm512_cvtusepi32_epi8(ri);
                    _mm_storeu_si128((__m128i*)(dst + i), packed);
                } else {
                    __m128i packed = _mm512_cvtsepi32_epi8(ri);
                    _mm_storeu_si128((__m128i*)(dst + i), packed);
                }
            }
        }
#endif
        for (; i < count; i++) {
            float val = rs * (float)((int32_t)src[i] - x_zp) + (float)y_zp;
            dst[i] = (T)std::clamp((int32_t)std::nearbyint(val), clamp_min, clamp_max);
        }
    }

    template <typename T>
    bool exec_typed() {
        tensor_t* y = outputs[0];
        float y_scale = *(float*)inputs[0]->data;
        int32_t y_zp = 0;
        if (inputs[1]->ndata > 0) {
            if (inputs[1]->type == NNR_DATA_TYPE_UINT8) y_zp = *(uint8_t*)inputs[1]->data;
            else if (inputs[1]->type == NNR_DATA_TYPE_INT8) y_zp = *(int8_t*)inputs[1]->data;
        }

        int clamp_min, clamp_max;
        if constexpr (std::is_same_v<T, uint8_t>) { clamp_min = 0; clamp_max = 255; }
        else { clamp_min = -128; clamp_max = 127; }

        int ntens = num_tensors();
        int ndim = y->ndim;
        int a = axis < 0 ? axis + ndim : axis;

        // Compute outer_size (dims before axis) and inner_size (dims after axis)
        size_t outer_size = 1, inner_size = 1;
        for (int d = 0; d < a; d++) outer_size *= y->dims[d];
        for (int d = a + 1; d < ndim; d++) inner_size *= y->dims[d];

        // Build a work list: for each (outer, tensor), record src/dst/chunk/params
        struct work_item_t {
            const T* src;
            T* dst;
            size_t count;
            float rs;
            int32_t x_zp;
            bool needs_requant;
        };
        std::vector<work_item_t> items;
        items.reserve(outer_size * ntens);

        T* py = (T*)y->data;
        size_t y_offset = 0;
        for (size_t o = 0; o < outer_size; o++) {
            for (int t = 0; t < ntens; t++) {
                const tensor_t* ti = tensor(t);
                float t_scale = scale(t);
                int32_t t_zp = zero_point(t);
                size_t chunk = (size_t)ti->dims[a] * inner_size;
                const T* src = (const T*)ti->data + o * chunk;
                bool same = (t_scale == y_scale && t_zp == y_zp);
                items.push_back({src, py + y_offset, chunk,
                                 same ? 0.f : t_scale / y_scale, t_zp, !same});
                y_offset += chunk;
            }
        }

        int n = (int)items.size();
        bool par = y->ndata > 4096 && n > 1;
        nnr::for_static(0, n, par, [&](int i) {
            auto& w = items[i];
            if (!w.needs_requant) {
                memcpy(w.dst, w.src, w.count * sizeof(T));
            } else {
                requantize_chunk(w.dst, w.src, w.count, w.rs, w.x_zp, y_zp,
                                 clamp_min, clamp_max);
            }
        });
        return true;
    }

    template <typename T>
    bool exec_typed_nhwc() {
        tensor_t* y = outputs[0];
        float y_scale = *(float*)inputs[0]->data;
        int32_t y_zp = 0;
        if (inputs[1]->ndata > 0) {
            if (inputs[1]->type == NNR_DATA_TYPE_UINT8) y_zp = *(uint8_t*)inputs[1]->data;
            else if (inputs[1]->type == NNR_DATA_TYPE_INT8) y_zp = *(int8_t*)inputs[1]->data;
        }

        int clamp_min, clamp_max;
        if constexpr (std::is_same_v<T, uint8_t>) { clamp_min = 0; clamp_max = 255; }
        else { clamp_min = -128; clamp_max = 127; }

        int ntens = num_tensors();
        int yC = y->dims[1];
        size_t NHW = (size_t)y->dims[0] * y->dims[2] * y->dims[3];

        // Precompute per-tensor info: data ptr, channel count, requant params, dst offset.
        struct tinfo_t {
            const T* data;
            int C;
            int c_off;
            float rs;
            int32_t x_zp;
            bool needs_requant;
        };
        small_vector<tinfo_t, 16> ti(ntens);
        int c_off = 0;
        for (int t = 0; t < ntens; t++) {
            const tensor_t* tt = tensor(t);
            float ts = scale(t);
            int32_t tzp = zero_point(t);
            bool same = (ts == y_scale && tzp == y_zp);
            ti[t] = tinfo_t{
                (const T*)tt->data,
                tt->dims[1],
                c_off,
                same ? 0.f : ts / y_scale,
                tzp,
                !same};
            c_off += tt->dims[1];
        }

        T* py = (T*)y->data;
        bool par = NHW > 256 && (NHW * yC) > 4096;
        nnr::for_static(0, (int)NHW, par, [&](int p) {
            T* dst_pix = py + (size_t)p * (size_t)yC;
            for (int t = 0; t < ntens; t++) {
                const auto& w = ti[t];
                const T* src_pix = w.data + (size_t)p * (size_t)w.C;
                T* dst_seg = dst_pix + w.c_off;
                if (!w.needs_requant) {
                    memcpy(dst_seg, src_pix, (size_t)w.C * sizeof(T));
                } else {
                    requantize_chunk(dst_seg, src_pix, (size_t)w.C,
                                     w.rs, w.x_zp, y_zp, clamp_min, clamp_max);
                }
            }
        });
        y->format = memory_layout_t::NHWC;
        y->set_quant(y_scale, y_zp);
        return true;
    }

    bool all_inputs_nhwc() const {
        int ntens = num_tensors();
        for (int i = 0; i < ntens; i++)
            if (tensor(i)->format != memory_layout_t::NHWC) return false;
        return true;
    }

    bool exec() override {
        data_type_t type = tensor(0)->type;
        if ((layout_mask & LAYOUT_NHWC) && all_inputs_nhwc()) {
            if (type == NNR_DATA_TYPE_UINT8) return exec_typed_nhwc<uint8_t>();
            if (type == NNR_DATA_TYPE_INT8) return exec_typed_nhwc<int8_t>();
        }
        if (type == NNR_DATA_TYPE_UINT8) return exec_typed<uint8_t>();
        if (type == NNR_DATA_TYPE_INT8) return exec_typed<int8_t>();
        return false;
    }
};

} // namespace

// @nnr-meta-op mt=static layout=[NCHW,NHWC]
operator_t* resolver_default_op_QLinearConcat(int opset, pool_t& pool) {
    return pool_new<QLinearConcat_operator>(pool);
}

} // namespace nnr
