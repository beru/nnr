#include "nnr.h"
#include "cpu_features.h"
#include "util.h"
#include <cmath>
#include <algorithm>
#ifdef NNR_ARCH_X64
#include "backend/x64/gemm_int8_avx512.h"
#endif

namespace nnr {

namespace {

struct QLinearMatMul_operator : public operator_t {
    int m = 0;
    int n = 0;
    int k = 0;

#ifdef NNR_ARCH_X64
    // Pre-packed B weights and column sums (computed once in reshape)
    std::vector<int8_t> packed_b;
    std::vector<int32_t> col_sums;
    bool b_prepacked = false;
    bool b_gemv_packed = false;  // true = GEMV-tiled packing [N/16][Kgroups][16]
#endif

    bool init() override {
        if (inputs.size() != 8 || outputs.size() != 1)
            return false;
        return true;
    }

    bool reshape() override {
        tensor_t* y = outputs[0];
        const tensor_t* a = inputs[0];
        const tensor_t* b = inputs[3];
        small_vector<int> adims;
        small_vector<int> bdims;

        if (a->ndim == 1) {
            adims = {1, a->dims[0]};
        } else {
            adims.assign(a->dims, a->dims + a->ndim);
        }
        if (b->ndim == 1) {
            bdims = {b->dims[0], 1};
        } else {
            bdims.assign(b->dims, b->dims + b->ndim);
        }
        const int ndim = max(adims.size(), bdims.size());
        small_vector<int> dims;
        dims.resize(ndim);
        if (adims.size() < 2 || bdims.size() < 2)
            return false;
        if (adims[adims.size() - 1] != bdims[bdims.size() - 2])
            return false;
        dims[ndim - 2] = adims[adims.size() - 2];
        dims[ndim - 1] = bdims[bdims.size() - 1];
        for (int i = 3; i <= ndim; ++i) {
            int alen = (adims.size() - i) < 0 ? 1 : adims[adims.size() - i];
            int blen = (bdims.size() - i) < 0 ? 1 : bdims[bdims.size() - i];
            if (alen != blen && alen > 1 && blen > 1)
                return false;
            dims[ndim - i] = max(alen, blen);
        }
        m = adims[adims.size() - 2];
        n = bdims[bdims.size() - 1];
        k = adims[adims.size() - 1];
        data_type_t out_type = inputs[7]->type;
        if (!y->reshape(dims, out_type))
            return false;

#ifdef NNR_ARCH_X64
        // Pre-pack B weights if B is a constant initializer with data available
        b_prepacked = false;
        b_gemv_packed = false;
        if (b->data && b->ndata > 0 && b->type == NNR_DATA_TYPE_INT8) {
            col_sums.resize(n);
            if (m <= 4) {
                // GEMV-tiled packing: [N/16][Kgroups][16] for sequential access
                packed_b.resize(int8::pack_b_int8_gemv_size(k, n));
                int8::pack_b_int8_gemv(packed_b.data(), col_sums.data(),
                    (const int8_t*)b->data, k, n);
                b_gemv_packed = true;
            } else {
                packed_b.resize(int8::pack_b_int8_nr48_size(k, n));
                int8::pack_b_int8_nr48_and_col_sums(packed_b.data(), col_sums.data(),
                    (const int8_t*)b->data, k, n);
            }
            b_prepacked = true;
        }
#endif
        return true;
    }

    int32_t get_zp(const tensor_t* t) {
        if (t->type == NNR_DATA_TYPE_UINT8) return (int32_t)*(const uint8_t*)t->data;
        return (int32_t)*(const int8_t*)t->data;
    }

    int32_t get_val(const tensor_t* t, int idx) {
        if (t->type == NNR_DATA_TYPE_UINT8) return (int32_t)((const uint8_t*)t->data)[idx];
        return (int32_t)((const int8_t*)t->data)[idx];
    }

    float get_scale(const tensor_t* t) {
        if (t->type == NNR_DATA_TYPE_FLOAT16) return (float)*(const float16_t*)t->data;
        return *(const float*)t->data;
    }

    bool exec() override {
        const tensor_t* a = inputs[0];
        const tensor_t* b = inputs[3];
        tensor_t* y = outputs[0];

        float a_scale = get_scale(inputs[1]);
        int32_t a_zp = get_zp(inputs[2]);
        float b_scale = get_scale(inputs[4]);
        int32_t b_zp = get_zp(inputs[5]);
        float y_scale = get_scale(inputs[6]);
        int32_t y_zp = get_zp(inputs[7]);
        // Combined scale: a_scale * b_scale / y_scale
        // When scales are float16, compute in float16 precision to match ONNX reference
        float combined_scale;
        if (inputs[1]->type == NNR_DATA_TYPE_FLOAT16) {
            float16_t f16_a = *(const float16_t*)inputs[1]->data;
            float16_t f16_b = *(const float16_t*)inputs[4]->data;
            float16_t f16_y = *(const float16_t*)inputs[6]->data;
            float16_t f16_combined = f16_a;
            f16_combined *= f16_b;
            f16_combined /= f16_y;
            combined_scale = (float)f16_combined;
        } else {
            combined_scale = a_scale * b_scale / y_scale;
        }

#ifdef NNR_ARCH_X64
        // VNNI fast path: uint8 A × int8 B
        if (has_avx512() && cpu_features().avx512vnni
            && a->type == NNR_DATA_TYPE_UINT8 && b->type == NNR_DATA_TYPE_INT8) {

            // Temporary buffers for row sums and int32 accumulator
            int n16 = (n + 15) & ~15;  // padded ldc for NR=48 tail safety
            int32_t* row_sums = (int32_t*)_aligned_malloc((size_t)m * sizeof(int32_t), 64);
            int32_t* c_buf = (int32_t*)_aligned_malloc((size_t)m * n16 * sizeof(int32_t), 64);

            // Per-batch B packing (only when B varies per batch or wasn't prepacked)
            int8_t* tmp_packed = nullptr;
            int32_t* tmp_col_sums = nullptr;
            if (!b_prepacked) {
                tmp_packed = (int8_t*)_aligned_malloc(int8::pack_b_int8_nr48_size(k, n), 64);
                tmp_col_sums = (int32_t*)_aligned_malloc((size_t)n16 * sizeof(int32_t), 64);
            }

            for (size_t i = 0, l = y->ndata; i < l; i += m * n) {
                int a_off = (int)(a->ndata <= (size_t)(m * k) ? 0 : (i / (m * n)) * m * k);
                const uint8_t* pa = (const uint8_t*)a->data + a_off;

                const int8_t* use_packed;
                const int32_t* use_col_sums;

                if (b_prepacked && b->ndata <= (size_t)(k * n)) {
                    // Single B matrix, prepacked in reshape
                    use_packed = packed_b.data();
                    use_col_sums = col_sums.data();
                } else {
                    // Batched or non-prepacked: pack per iteration
                    int b_off = (int)(b->ndata <= (size_t)(k * n) ? 0 : (i / (m * n)) * k * n);
                    const int8_t* pb = (const int8_t*)b->data + b_off;
                    int8::pack_b_int8_nr48_and_col_sums(tmp_packed, tmp_col_sums, pb, k, n);
                    use_packed = tmp_packed;
                    use_col_sums = tmp_col_sums;
                }

                int8::compute_row_sums(row_sums, pa, m, k);
                if (b_gemv_packed) {
                    int8::gemm_int8_gemv(m, n, k, pa, a_zp, use_packed, b_zp,
                        use_col_sums, row_sums, c_buf);
                } else {
                    int8::gemm_int8_nr48(m, n, k, pa, a_zp, use_packed, b_zp,
                        use_col_sums, row_sums, c_buf);
                }

                // Requantize: float val = sum * combined_scale + y_zp
                uint8_t* y_u8 = (uint8_t*)y->data + i;
                int8_t* y_s8 = (int8_t*)y->data + i;
                bool is_u8 = (y->type == NNR_DATA_TYPE_UINT8);
                for (int r = 0; r < m; ++r) {
                    for (int c = 0; c < n; ++c) {
                        float val = c_buf[r * n16 + c] * combined_scale + y_zp;
                        int32_t ival = (int32_t)std::nearbyint(val);
                        int u = r * n + c;
                        if (is_u8)
                            y_u8[u] = (uint8_t)std::clamp(ival, 0, 255);
                        else
                            y_s8[u] = (int8_t)std::clamp(ival, -128, 127);
                    }
                }
            }

            _aligned_free(row_sums);
            _aligned_free(c_buf);
            _aligned_free(tmp_packed);
            _aligned_free(tmp_col_sums);
            return true;
        }
#endif

        // Scalar fallback
        for (size_t i = 0, l = y->ndata; i < l; i += m * n) {
            int a_off = (int)(a->ndata <= (size_t)(m * k) ? 0 : (i / (m * n)) * m * k);
            int b_off = (int)(b->ndata <= (size_t)(k * n) ? 0 : (i / (m * n)) * k * n);

            for (int u = 0; u < m; ++u) {
                for (int v = 0; v < n; ++v) {
                    int32_t sum = 0;
                    for (int w = 0; w < k; ++w) {
                        int32_t av = get_val(a, a_off + u * k + w) - a_zp;
                        int32_t bv = get_val(b, b_off + w * n + v) - b_zp;
                        sum += av * bv;
                    }
                    float val = sum * combined_scale + y_zp;
                    int32_t ival = (int32_t)std::nearbyint(val);
                    if (y->type == NNR_DATA_TYPE_UINT8)
                        ((uint8_t*)y->data)[i + u * n + v] = (uint8_t)std::clamp(ival, 0, 255);
                    else
                        ((int8_t*)y->data)[i + u * n + v] = (int8_t)std::clamp(ival, -128, 127);
                }
            }
        }
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no prepack=yes
operator_t* resolver_default_op_QLinearMatMul(int opset, pool_t& pool) {
    return pool_new<QLinearMatMul_operator>(pool);
}

} // namespace nnr
