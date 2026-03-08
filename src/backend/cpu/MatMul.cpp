#include "nnr.h"
#include "util.h"
#include "kernel/gemm.h"
#include "kernel/f16_convert.h"
#ifdef NNR_ARCH_X64
#include "backend/x64/gemm_bf16_avx512.h"
#endif

namespace nnr {

namespace {

struct MatMul_operator : public operator_t {
    int m = 0;
    int n = 0;
    int k = 0;
    small_vector<int> batch_dims;
    small_vector<int> a_batch_strides;
    small_vector<int> b_batch_strides;

    std::vector<float> b_packed;  // pre-packed B weights (float32)
    std::vector<uint16_t> b_packed_bf16;  // VNNI-packed BF16 weights
    int collapsed_m = 0;  // >0: batch collapsed into M dimension

    bool init() override {
        if (!is_inout_size(2, 1)) {
            return false;
        }
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
        }else {
            adims.assign(a->dims, a->dims + a->ndim);
        }
        if (b->ndim == 1) {
            bdims = {b->dims[0], 1};
        }else {
            bdims.assign(b->dims, b->dims + b->ndim);
        }
        const int ndim = max(adims.size(), bdims.size());
        small_vector<int> dims;
        dims.resize(ndim);
        if (adims.size() < 2 || bdims.size() < 2) return false;
        if (adims[adims.size() - 1] != bdims[bdims.size() - 2]) return false;
        dims[ndim - 2] = adims[adims.size() - 2];
        dims[ndim - 1] = bdims[bdims.size() - 1];
        for (int i = 3; i <= ndim; ++i) {
            int ai = (int)adims.size() - i;
            int bi = (int)bdims.size() - i;
            int alen = ai < 0 ? 1 : adims[ai];
            int blen = bi < 0 ? 1 : bdims[bi];
            if (alen != blen && alen > 1 && blen > 1) return false;
            dims[ndim - i] = max(alen, blen);
        }
        m = adims[adims.size() - 2];
        n = bdims[bdims.size() - 1];
        k = adims[adims.size() - 1];

        // Compute batch dimensions and broadcast strides.
        // For each batch dimension, compute the stride to advance through A and B.
        // stride=0 means broadcast (that tensor's batch dim is 1).
        int nbatch = ndim - 2;
        batch_dims.resize(nbatch);
        a_batch_strides.resize(nbatch);
        b_batch_strides.resize(nbatch);
        int a_stride = m * k;  // elements per batch slice of A
        int b_stride = k * n;  // elements per batch slice of B
        for (int i = nbatch - 1; i >= 0; --i) {
            batch_dims[i] = dims[i];
            // Map output batch dim i back to A's and B's dimension index
            int ai = i - (nbatch - ((int)adims.size() - 2));
            int bi = i - (nbatch - ((int)bdims.size() - 2));
            if (ai >= 0 && adims[ai] > 1) {
                a_batch_strides[i] = a_stride;
                a_stride *= adims[ai];
            }else {
                a_batch_strides[i] = 0;
            }
            if (bi >= 0 && bdims[bi] > 1) {
                b_batch_strides[i] = b_stride;
                b_stride *= bdims[bi];
            }else {
                b_batch_strides[i] = 0;
            }
        }

        // Pre-pack B if constant float32/float16 and uniform across batches
        b_packed.clear();
        const tensor_t* bp = inputs[1];
        if ((bp->type == NNR_DATA_TYPE_FLOAT32 || bp->type == NNR_DATA_TYPE_FLOAT16)
            && k > 0 && n > 0 && ctx && ctx->initializer_names.count(bp->name)) {
            bool b_uniform = true;
            for (int i = 0; i < nbatch; i++) {
                if (b_batch_strides[i] != 0) {
                    b_uniform = false;
                    break;
                }
            }
            if (b_uniform) {
                size_t psz = pack_b_size(k, n);
                if (psz > 0) {
                    size_t bn = (size_t)k * n;
                    float* tmp = nullptr;
                    const float* pb;
                    if (bp->type == NNR_DATA_TYPE_FLOAT16) {
                        tmp = (float*)_aligned_malloc(bn * sizeof(float), 64);
                        convert_f16_to_f32(tmp, (const float16_t*)bp->data, bn);
                        pb = tmp;
                    } else {
                        pb = (const float*)bp->data;
                    }
                    b_packed.resize(psz);
                    pack_b(b_packed.data(), pb, k, n);
                    _aligned_free(tmp);
                }
            }
        }

        // Pre-pack BF16 weights into VNNI format for VDPBF16PS.
        // Also converts fp32 constant weights to bf16 — halves weight memory
        // traffic for BW-bound thin-M GEMMs (e.g., embedding models).
        b_packed_bf16.clear();
#ifdef NNR_ARCH_X64
        if (has_avx512_bf16() && k > 0 && n > 0 && ctx
            && ctx->initializer_names.count(bp->name)) {
            bool b_uniform = true;
            for (int i = 0; i < nbatch; i++)
                if (b_batch_strides[i] != 0) { b_uniform = false; break; }
            if (b_uniform) {
                size_t bn = (size_t)k * n;
                size_t psz = bf16::pack_b_bf16_size(k, n);
                if (bp->type == NNR_DATA_TYPE_BFLOAT16) {
                    b_packed_bf16.resize(psz);
                    bf16::pack_b_bf16(b_packed_bf16.data(), (const uint16_t*)bp->data, k, n);
                } else if (bp->type == NNR_DATA_TYPE_FLOAT32) {
                    // Convert fp32 weights → bf16, then pack into VNNI format
                    std::vector<uint16_t> tmp_bf16(bn);
                    convert_f32_to_bf16((bfloat16_t*)tmp_bf16.data(),
                                        (const float*)bp->data, bn);
                    b_packed_bf16.resize(psz);
                    bf16::pack_b_bf16(b_packed_bf16.data(), tmp_bf16.data(), k, n);
                }
            }
        }
#endif

        // Check if batch can be collapsed into M dimension:
        // B must be uniform (broadcast) and A contiguous across all batch dims.
        collapsed_m = 0;
        if (!b_packed.empty() && nbatch > 0) {
            bool can_collapse = true;
            int64_t total_batches = 1;
            int expected_stride = m * k;
            for (int i = nbatch - 1; i >= 0; --i) {
                if (b_batch_strides[i] != 0 || a_batch_strides[i] != expected_stride) {
                    can_collapse = false;
                    break;
                }
                total_batches *= batch_dims[i];
                expected_stride *= batch_dims[i];
            }
            if (can_collapse && total_batches > 1)
                collapsed_m = (int)(total_batches * m);
        }

        // Per ONNX spec: if a is 1D, remove prepended 1; if b is 1D, remove appended 1
        small_vector<int> out_dims;
        if (a->ndim == 1 && b->ndim == 1) {
            // scalar output
        }else if (a->ndim == 1) {
            for (int i = 0; i < ndim; ++i) {
                if (i != ndim - 2) out_dims.push_back(dims[i]);
            }
        }else if (b->ndim == 1) {
            for (int i = 0; i < ndim - 1; ++i) {
                out_dims.push_back(dims[i]);
            }
        }else {
            out_dims.assign(dims.begin(), dims.end());
        }
        return y->reshape(out_dims, a->type);
    }

    template <typename T>
    bool exec() {
        tensor_t* y = outputs[0];
        const tensor_t* a = inputs[0];
        const tensor_t* b = inputs[1];
        const T* pa_base = (const T*)a->data;
        const T* pb_base = (const T*)b->data;
        T* py = (T*)y->data;

        int nbatch = batch_dims.size();
        small_vector<int> idx(nbatch);
        int out_offset = 0;
        do {
            int a_off = 0, b_off = 0;
            for (int i = 0; i < nbatch; ++i) {
                a_off += idx[i] * a_batch_strides[i];
                b_off += idx[i] * b_batch_strides[i];
            }
            const T* pa = pa_base + a_off;
            const T* pb = pb_base + b_off;
            T* py_slice = py + out_offset;

            if constexpr (std::is_same_v<T, float>) {
                if (!b_packed.empty()) {
                    dgemm_packed_b(m, n, k, pa, b_packed.data(), py_slice,
                        gemm_post_t(nullptr, 0, py_slice, out_offset, this));
                } else {
                    dgemm_generic(m, n, k, pa, pb, py_slice,
                        gemm_post_t(nullptr, 0, py_slice, out_offset, this));
                }
            } else {
                dgemm_generic(m, n, k, pa, pb, py_slice);
            }

            out_offset += m * n;
        } while (dim_next(idx, batch_dims));
        return true;
    }

    // FP16 I/O with FP32 compute: convert inputs, run float GEMM, convert output back.
    bool exec_f16_as_f32() {
        tensor_t* y = outputs[0];
        const tensor_t* a = inputs[0];
        const tensor_t* b = inputs[1];

        int64_t total_batches = 1;
        for (int i = 0; i < batch_dims.size(); ++i)
            total_batches *= batch_dims[i];

        // Workspace layout: [A_f32 | B_f32 (if not packed) | Y_f32]
        float* ws = (float*)ctx->workspace;
        size_t a_total = (size_t)a->ndata;
        size_t b_total = (size_t)b->ndata;
        size_t y_total = (size_t)y->ndata;

        float* a_f32 = ws;
        convert_f16_to_f32(a_f32, (const float16_t*)a->data, a_total);

        float* b_f32;
        float* y_f32;
        if (!b_packed.empty()) {
            b_f32 = nullptr;
            y_f32 = a_f32 + a_total;
        } else {
            b_f32 = a_f32 + a_total;
            convert_f16_to_f32(b_f32, (const float16_t*)b->data, b_total);
            y_f32 = b_f32 + b_total;
        }

        int nbatch = batch_dims.size();
        small_vector<int> idx(nbatch);
        int out_offset = 0;
        do {
            int a_off = 0, b_off = 0;
            for (int i = 0; i < nbatch; ++i) {
                a_off += idx[i] * a_batch_strides[i];
                b_off += idx[i] * b_batch_strides[i];
            }
            const float* pa = a_f32 + a_off;
            float* py_slice = y_f32 + out_offset;

            if (!b_packed.empty()) {
                dgemm_packed_b(m, n, k, pa, b_packed.data(), py_slice,
                    gemm_post_t(nullptr, 0, py_slice, out_offset, this));
            } else {
                const float* pb = b_f32 + b_off;
                dgemm_generic(m, n, k, pa, pb, py_slice,
                    gemm_post_t(nullptr, 0, py_slice, out_offset, this));
            }

            out_offset += m * n;
        } while (dim_next(idx, batch_dims));

        // Convert output back to FP16
        convert_f32_to_f16((float16_t*)y->data, y_f32, y_total);
        return true;
    }

    int64_t num_ops() const override {
        int64_t batches = 1;
        for (int i = 0; i < batch_dims.size(); ++i)
            batches *= batch_dims[i];
        return (int64_t)2 * batches * m * n * k;
    }

    // BF16 GEMM: native VDPBF16PS when available, else convert-to-FP32 fallback.
    bool exec_bf16() {
        tensor_t* y = outputs[0];
        const tensor_t* a = inputs[0];
        const tensor_t* b = inputs[1];
        size_t y_total = y->ndata;

        float* y_f32 = (float*)ctx->workspace;

#ifdef NNR_ARCH_X64
        if (has_avx512_bf16() && !b_packed_bf16.empty()) {
            // Native VDPBF16PS path with pre-packed B
            int nbatch = batch_dims.size();
            small_vector<int> idx(nbatch);
            int out_offset = 0;
            do {
                int a_off = 0;
                for (int i = 0; i < nbatch; ++i)
                    a_off += idx[i] * a_batch_strides[i];
                const uint16_t* pa = (const uint16_t*)a->data + a_off;
                float* py_slice = y_f32 + out_offset;

                gemm_post_t post;
                bf16::dgemm_bf16(m, n, k, pa, b_packed_bf16.data(), py_slice, post);

                out_offset += m * n;
            } while (dim_next(idx, batch_dims));

            convert_f32_to_bf16((bfloat16_t*)y->data, y_f32, y_total);
            return true;
        }
#endif
        // Fallback: convert BF16 → FP32, use float GEMM
        size_t a_total = a->ndata;
        size_t b_total = b->ndata;
        float* a_f32 = y_f32 + y_total;
        convert_bf16_to_f32(a_f32, (const bfloat16_t*)a->data, a_total);
        float* b_f32 = a_f32 + a_total;
        convert_bf16_to_f32(b_f32, (const bfloat16_t*)b->data, b_total);

        int nbatch = batch_dims.size();
        small_vector<int> idx(nbatch);
        int out_offset = 0;
        do {
            int a_off = 0, b_off = 0;
            for (int i = 0; i < nbatch; ++i) {
                a_off += idx[i] * a_batch_strides[i];
                b_off += idx[i] * b_batch_strides[i];
            }
            dgemm_generic(m, n, k, a_f32 + a_off, b_f32 + b_off, y_f32 + out_offset);
            out_offset += m * n;
        } while (dim_next(idx, batch_dims));

        convert_f32_to_bf16((bfloat16_t*)y->data, y_f32, y_total);
        return true;
    }

    size_t workspace_size() const override {
        if (inputs[0]->type == NNR_DATA_TYPE_FLOAT16) {
            size_t ws = inputs[0]->ndata + outputs[0]->ndata;
            if (b_packed.empty())
                ws += inputs[1]->ndata;
            return ws * sizeof(float);
        }
        if (inputs[0]->type == NNR_DATA_TYPE_BFLOAT16) {
            size_t ws = outputs[0]->ndata;  // Y_f32
#ifdef NNR_ARCH_X64
            if (!has_avx512_bf16() || b_packed_bf16.empty())
#endif
                ws += inputs[0]->ndata + inputs[1]->ndata;  // A_f32 + B_f32
            return ws * sizeof(float);
        }
#ifdef NNR_ARCH_X64
        // fp32 with bf16 packed weights: need workspace for A_bf16 conversion
        if (inputs[0]->type == NNR_DATA_TYPE_FLOAT32 && !b_packed_bf16.empty()) {
            // A_bf16[M×K] as uint16_t (2 bytes each)
            return inputs[0]->ndata * sizeof(uint16_t);
        }
#endif
        return 0;
    }

    // fp32 input with bf16-packed weights: convert A on-the-fly, use VDPBF16PS.
    // Halves weight memory traffic at ~0.4% precision cost (bf16 truncation).
    bool exec_f32_via_bf16() {
#ifdef NNR_ARCH_X64
        const tensor_t* a = inputs[0];
        tensor_t* y = outputs[0];

        int nbatch = batch_dims.size();
        int actual_m = (collapsed_m > 0) ? collapsed_m : m;

        // Convert fp32 A → bf16 into workspace
        uint16_t* a_bf16 = (uint16_t*)ctx->workspace;
        size_t a_count = (size_t)actual_m * k;
        convert_f32_to_bf16((bfloat16_t*)a_bf16, (const float*)a->data, a_count);

        float* py = (float*)y->data;
        gemm_post_t post;

        if (collapsed_m > 0) {
            bf16::dgemm_bf16(actual_m, n, k, a_bf16, b_packed_bf16.data(), py, post);
        } else {
            small_vector<int> idx(nbatch);
            int out_offset = 0;
            do {
                int a_off = 0;
                for (int i = 0; i < nbatch; ++i)
                    a_off += idx[i] * a_batch_strides[i];
                bf16::dgemm_bf16(m, n, k, a_bf16 + a_off,
                    b_packed_bf16.data(), py + out_offset, post);
                out_offset += m * n;
            } while (dim_next(idx, batch_dims));
        }

        return true;
#else
        return false;
#endif
    }

    bool exec() override {
        if (inputs[0]->type == NNR_DATA_TYPE_FLOAT16)
            return exec_f16_as_f32();
        if (inputs[0]->type == NNR_DATA_TYPE_BFLOAT16)
            return exec_bf16();
#ifdef NNR_ARCH_X64
        // fp32 with bf16-packed weights: use VDPBF16PS for ~2× BW reduction
        if (inputs[0]->type == NNR_DATA_TYPE_FLOAT32 && !b_packed_bf16.empty())
            return exec_f32_via_bf16();
#endif
        // Batch collapse: temporarily widen M and clear batch dims so exec<float>
        // runs a single large GEMM instead of per-batch dispatches.
        if (collapsed_m > 0 && inputs[0]->type == NNR_DATA_TYPE_FLOAT32) {
            int saved_m = m;
            m = collapsed_m;
            auto saved_batch = std::move(batch_dims);
            batch_dims.clear();
            bool ok = exec<float>();
            m = saved_m;
            batch_dims = std::move(saved_batch);
            return ok;
        }
        return typed_exec<MatMul_operator,
            opset_t<13, int32_t, int64_t, uint32_t, uint64_t, float, double>,
            opset_t<9, int32_t, int64_t, uint32_t, uint64_t, float, double>,
            opset_t<1, float, double>
        >(this, opset, inputs[0]->type);
    }

};

} // namespace {

// @nnr-meta-op mt=static workspace=yes prepack=yes fusion=post_op
operator_t* resolver_default_op_MatMul(int opset, pool_t& pool) { return pool_new<MatMul_operator>(pool); }

} // namespace nnr
