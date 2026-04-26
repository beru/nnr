#include "nnr.h"
#include "util.h"
#include "thread_pool.h"
#include <cmath>
#include <algorithm>

namespace nnr {

namespace {

// QLinearLeakyRelu (com.microsoft, opset 1 / appears in onnxruntime contrib):
//   Y = quantize(LeakyRelu(dequantize(X)))
// Inputs: X, X_scale, X_zero_point, Y_scale, Y_zero_point
// Attribute: alpha (default 0.01)
//
// The op is unary on a quantized 8-bit input, so the entire transform
// collapses to a 256-entry lookup table indexed by the raw byte. Building
// the table once at exec time turns the hot loop into a single byte
// load + table lookup + byte store per element.
struct QLinearLeakyRelu_operator : public operator_t {
    float alpha;

    bool init() override {
        if (inputs.size() != 5 || outputs.size() != 1) return false;
        layout_mask = LAYOUT_ALL;
        alpha = attribute(attr_key_t::alpha, 0.01f);
        return true;
    }

    bool reshape() override {
        return outputs[0]->reshape_identity(inputs[0], inputs[0]->type);
    }

    template <typename T>
    bool exec_typed() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const float x_scale = *(const float*)inputs[1]->data;
        const float y_scale = *(const float*)inputs[3]->data;

        int32_t x_zp = 0, y_zp = 0;
        if (inputs[2]->ndata > 0) x_zp = (int32_t)((const T*)inputs[2]->data)[0];
        if (inputs[4]->ndata > 0) y_zp = (int32_t)((const T*)inputs[4]->data)[0];

        const int clamp_min = std::is_same_v<T, uint8_t> ? 0    : -128;
        const int clamp_max = std::is_same_v<T, uint8_t> ? 255  :  127;

        // Build the 256-entry LUT. Index is the raw byte read from input;
        // for int8 we use signed reinterpretation so the table covers
        // -128..127 at indices 0..255. Operation order mirrors ORT
        // (dequantize → LeakyRelu → quantize) so half-point rounding
        // matches; a single FMA folding x_scale*alpha/y_scale up front
        // produced 1-LSB drift on negative-half-tie inputs.
        uint8_t lut[256];
        for (int i = 0; i < 256; ++i) {
            const int32_t raw = std::is_same_v<T, uint8_t> ? i : (int8_t)i;
            const float centered = (float)(raw - x_zp);
            const float dq = centered * x_scale;
            const float lr = (dq >= 0.0f) ? dq : (dq * alpha);
            const int32_t q = (int32_t)std::nearbyint(lr / y_scale) + y_zp;
            const int32_t c = std::clamp(q, clamp_min, clamp_max);
            lut[i] = (uint8_t)c;
        }

        const uint8_t* px = (const uint8_t*)x->data;
        uint8_t* py = (uint8_t*)y->data;
        const size_t n = y->ndata;

        constexpr size_t BLOCK = 65536;
        const int nblocks = (int)((n + BLOCK - 1) / BLOCK);
        const int nt = nnr::elementwise_threads(n, 1, 1, 1);

        auto chunk_fn = [&](size_t i0, size_t i1) {
            for (size_t i = i0; i < i1; ++i)
                py[i] = lut[px[i]];
        };
        if (nt <= 1) {
            chunk_fn(0, n);
        } else {
            nnr::for_dynamic(0, nblocks, nt, [&](int, int blk) {
                const size_t start = (size_t)blk * BLOCK;
                const size_t end = std::min(start + BLOCK, n);
                chunk_fn(start, end);
            });
        }
        return true;
    }

    bool exec() override {
        const data_type_t type = inputs[0]->type;
        if (type == NNR_DATA_TYPE_UINT8) return exec_typed<uint8_t>();
        if (type == NNR_DATA_TYPE_INT8)  return exec_typed<int8_t>();
        return false;
    }
};

} // namespace

// @nnr-meta-op mt=dynamic layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
operator_t* resolver_default_op_QLinearLeakyRelu(int /*opset*/, pool_t& pool) {
    return pool_new<QLinearLeakyRelu_operator>(pool);
}

} // namespace nnr
