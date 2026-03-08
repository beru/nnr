#include "nnr.h"
#include "util.h"
#ifdef NNR_ARCH_X64
#include "backend/x64/gemm_int8_avx512.h"
#endif

namespace nnr {

namespace {

struct MatMulInteger_operator : public operator_t {
    int m = 0;
    int n = 0;
    int k = 0;

    bool init() override {
        if (inputs.size() < 2 || inputs.size() > 4 || outputs.size() != 1)
            return false;
        return true;
    }

    bool reshape() override {
        tensor_t* y = outputs[0];
        const tensor_t* a = inputs[0];
        const tensor_t* b = inputs[1];
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
        return y->reshape(dims, NNR_DATA_TYPE_INT32);
    }

    template <typename TA, typename TB>
    bool exec_typed() {
        tensor_t* y = outputs[0];
        const tensor_t* a = inputs[0];
        const tensor_t* b = inputs[1];
        int32_t* py = (int32_t*)y->data;

        int32_t azp = 0;
        int32_t bzp = 0;
        if (inputs.size() > 2 && inputs[2]->ndata > 0)
            azp = (int32_t)(*(const TA*)inputs[2]->data);
        if (inputs.size() > 3 && inputs[3]->ndata > 0)
            bzp = (int32_t)(*(const TB*)inputs[3]->data);

        for (size_t i = 0, l = y->ndata; i < l; i += m * n) {
            const TA* pa = (const TA*)a->broadcast_map_address(y, i);
            const TB* pb = (const TB*)b->broadcast_map_address(y, i);
            for (int u = 0; u < m; ++u) {
                for (int v = 0; v < n; ++v) {
                    int32_t sum = 0;
                    for (int w = 0; w < k; ++w) {
                        sum += ((int32_t)pa[u * k + w] - azp) * ((int32_t)pb[w * n + v] - bzp);
                    }
                    py[i + u * n + v] = sum;
                }
            }
        }
        return true;
    }

    bool exec() override {
        auto atype = inputs[0]->type;
        auto btype = inputs[1]->type;

#ifdef NNR_ARCH_X64
        // VNNI fast path: uint8 A × int8 B (most common ONNX quantized config)
        if (has_avx512() && cpu_features().avx512vnni
            && atype == NNR_DATA_TYPE_UINT8 && btype == NNR_DATA_TYPE_INT8) {
            tensor_t* y = outputs[0];
            const tensor_t* a = inputs[0];
            const tensor_t* b = inputs[1];
            int32_t azp = 0, bzp = 0;
            if (inputs.size() > 2 && inputs[2]->ndata > 0)
                azp = (int32_t)*(const uint8_t*)inputs[2]->data;
            if (inputs.size() > 3 && inputs[3]->ndata > 0)
                bzp = (int32_t)*(const int8_t*)inputs[3]->data;

            // Pack B into NR=48 VNNI format and precompute sums
            int n16 = (n + 15) & ~15;
            size_t psz = int8::pack_b_int8_nr48_size(k, n);
            int8_t* packed = (int8_t*)_aligned_malloc(psz, 64);
            int32_t* col_sums = (int32_t*)_aligned_malloc((size_t)n16 * sizeof(int32_t), 64);
            int32_t* row_sums = (int32_t*)_aligned_malloc((size_t)m * sizeof(int32_t), 64);
            int32_t* c_tmp = (int32_t*)_aligned_malloc((size_t)m * n16 * sizeof(int32_t), 64);

            for (size_t i = 0, l = y->ndata; i < l; i += m * n) {
                const uint8_t* pa = (const uint8_t*)a->broadcast_map_address(y, (int)i);
                const int8_t* pb = (const int8_t*)b->broadcast_map_address(y, (int)i);

                int8::pack_b_int8_nr48_and_col_sums(packed, col_sums, pb, k, n);
                int8::compute_row_sums(row_sums, pa, m, k);

                int8::gemm_int8_nr48(m, n, k, pa, azp, packed, bzp,
                    col_sums, row_sums, c_tmp, n16);

                // Copy from padded C to output (contiguous [m × n])
                int32_t* y_out = (int32_t*)y->data + i;
                for (int r = 0; r < m; r++)
                    memcpy(y_out + r * n, c_tmp + r * n16, (size_t)n * sizeof(int32_t));
            }

            _aligned_free(packed);
            _aligned_free(col_sums);
            _aligned_free(row_sums);
            _aligned_free(c_tmp);
            return true;
        }
#endif

        if (atype == NNR_DATA_TYPE_UINT8 && btype == NNR_DATA_TYPE_UINT8)
            return exec_typed<uint8_t, uint8_t>();
        if (atype == NNR_DATA_TYPE_UINT8 && btype == NNR_DATA_TYPE_INT8)
            return exec_typed<uint8_t, int8_t>();
        if (atype == NNR_DATA_TYPE_INT8 && btype == NNR_DATA_TYPE_UINT8)
            return exec_typed<int8_t, uint8_t>();
        if (atype == NNR_DATA_TYPE_INT8 && btype == NNR_DATA_TYPE_INT8)
            return exec_typed<int8_t, int8_t>();
        return false;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_MatMulInteger(int opset, pool_t& pool) {
    if (opset >= 10)
        return pool_new<MatMulInteger_operator>(pool);
    return nullptr;
}

} // namespace nnr
