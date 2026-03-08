#include "nnr.h"
#include "util.h"
#include "thread_pool.h"
#ifdef NNR_ARCH_X64
#include "backend/x64/layout_x64.h"
#endif

namespace nnr {

namespace {

struct Transpose_operator : public operator_t {
    small_vector<int> perm;

    bool init() override {
        if (!is_inout_size(1, 1)) {
            return false;
        }
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        perm.resize(x->ndim);
        int64_t* ints;
        int nint = attribute(attr_key_t::perm, ints);
        if (nint == x->ndim) {
            for (int i = 0; i < x->ndim; ++i) {
                perm[i] = (int)ints[i];
            }
        }else {
            for (int i = 0; i < x->ndim; ++i) {
                perm[i] = x->ndim - i - 1;
            }
        }
        small_vector<int> dims(x->ndim);
        for (int i = 0; i < x->ndim; ++i) {
            dims[i] = x->dims[perm[i]];
        }
        if (!y->reshape(dims, x->type)) return false;
        if (x->is_quantized()) y->set_quant(x->quant_scale, x->quant_zero_point);
        return true;
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;
        int ndim = x->ndim;
        size_t l = y->ndata;

        // Path 1: identity permutation
        bool is_identity = true;
        for (int i = 0; i < ndim; ++i) if (perm[i] != i) { is_identity = false; break; }
        if (is_identity) {
            if constexpr (std::is_same_v<T, std::string>) {
                for (size_t i = 0; i < l; ++i) py[i] = px[i];
            } else {
                memcpy(py, px, l * sizeof(T));
            }
            return true;
        }

        // Path 2: single adjacent-axis swap — batch + tiled 2D transpose
        int sa = -1, sb = -1;
        for (int i = 0; i < ndim; ++i) {
            if (perm[i] != i) {
                if (sa < 0) sa = i;
                else if (sb < 0) sb = i;
                else { sa = -1; break; }
            }
        }
        if (sa >= 0 && sb == sa + 1 && perm[sa] == sb && perm[sb] == sa) {
            int batch = 1; for (int i = 0; i < sa; ++i) batch *= x->dims[i];
            int M = x->dims[sa], N = x->dims[sb];
            int tail = 1; for (int i = sb + 1; i < ndim; ++i) tail *= x->dims[i];
            if constexpr (!std::is_same_v<T, std::string>) {
#ifdef NNR_ARCH_X64
                // AVX-512 fast path: float, tail==1, M and N >= 16
                if constexpr (std::is_same_v<T, float>) {
                    if (tail == 1) {
                        for (int b = 0; b < batch; ++b)
                            transpose_2d_avx512(
                                (const float*)px + (size_t)b * M * N,
                                (float*)py + (size_t)b * N * M, M, N);
                        return true;
                    }
                }
#endif
                constexpr int TILE = 32;
                int total_work = batch * ((M + TILE - 1) / TILE) * ((N + TILE - 1) / TILE);
                int tile_cols = (N + TILE - 1) / TILE;
                int tiles_per_batch = ((M + TILE - 1) / TILE) * tile_cols;
                nnr::for_static(0, total_work, total_work > 4, [&](int work) {
                    int b = work / tiles_per_batch;
                    int tile = work % tiles_per_batch;
                    int ti = tile / tile_cols;
                    int tj = tile % tile_cols;
                    int i0 = ti * TILE, j0 = tj * TILE;
                    int imax = std::min(i0 + TILE, M);
                    int jmax = std::min(j0 + TILE, N);
                    const T* src = px + (size_t)b * M * N * tail;
                    T*       dst = py + (size_t)b * N * M * tail;
                    if (tail == 1) {
                        for (int i = i0; i < imax; ++i)
                            for (int j = j0; j < jmax; ++j)
                                dst[j * M + i] = src[i * N + j];
                    } else {
                        for (int i = i0; i < imax; ++i)
                            for (int j = j0; j < jmax; ++j)
                                memcpy(&dst[(j * M + i) * tail],
                                       &src[(i * N + j) * tail],
                                       tail * sizeof(T));
                    }
                });
            } else {
                for (int b = 0; b < batch; ++b)
                    for (int i = 0; i < M; ++i)
                        for (int j = 0; j < N; ++j)
                            for (int t = 0; t < tail; ++t)
                                py[(b*N*M + j*M + i) * tail + t] = px[(b*M*N + i*N + j) * tail + t];
            }
            return true;
        }

        // Path 3: collapse contiguous axis groups, reduce to adjacent swap
        // Example: perm=[0,2,3,1] has groups {0},{2,3},{1} → 3D swap(1,2)
        //          M=C, N=H*W, batch=N → reuse Path 2 tiled transpose
        if constexpr (!std::is_same_v<T, std::string>) {
            // Build collapsed groups: each group is a run of consecutive axes in perm
            struct group_t { int size; int src_axis; };
            small_vector<group_t> groups;
            groups.push_back({x->dims[perm[0]], perm[0]});
            for (int i = 1; i < ndim; ++i) {
                if (perm[i] == perm[i-1] + 1) {
                    groups.back().size *= x->dims[perm[i]];
                } else {
                    groups.push_back({x->dims[perm[i]], perm[i]});
                }
            }
            // Check if collapsed perm is an adjacent swap
            if (groups.size() == 3) {
                // Find which two groups are swapped
                // groups[0].src_axis < groups[1].src_axis and groups[1].src_axis > groups[2].src_axis
                // means swap at position 1,2
                int g0_first = groups[0].src_axis;
                int g1_first = groups[1].src_axis;
                int g2_first = groups[2].src_axis;
                if (g0_first < g1_first && g1_first > g2_first && g2_first > g0_first) {
                    // Collapsed to: [batch, M, N] → swap axes 1,2
                    // Source: [batch][cM][cN], Dest: [batch][cN][cM]
                    int batch = groups[0].size;
                    int cN = groups[1].size;
                    int cM = groups[2].size;
#ifdef NNR_ARCH_X64
                    if constexpr (std::is_same_v<T, float>) {
                        for (int b = 0; b < batch; ++b)
                            transpose_2d_avx512(
                                (const float*)px + (size_t)b * cM * cN,
                                (float*)py + (size_t)b * cN * cM, cM, cN);
                        return true;
                    }
#endif
                    constexpr int TILE = 32;
                    int total_work = batch * ((cM + TILE - 1) / TILE) * ((cN + TILE - 1) / TILE);
                    int tile_cols = (cN + TILE - 1) / TILE;
                    int tiles_per_batch = ((cM + TILE - 1) / TILE) * tile_cols;
                    nnr::for_static(0, total_work, total_work > 4, [&](int work) {
                        int b = work / tiles_per_batch;
                        int tile = work % tiles_per_batch;
                        int ti = tile / tile_cols;
                        int tj = tile % tile_cols;
                        int i0 = ti * TILE, j0 = tj * TILE;
                        int imax = std::min(i0 + TILE, cM);
                        int jmax = std::min(j0 + TILE, cN);
                        const T* src = px + (size_t)b * cM * cN;
                        T*       dst = py + (size_t)b * cN * cM;
                        for (int i = i0; i < imax; ++i)
                            for (int j = j0; j < jmax; ++j)
                                dst[j * cM + i] = src[i * cN + j];
                    });
                    return true;
                }
                if (g0_first < g2_first && g1_first < g0_first) {
                    // Collapsed to: [M, batch, N] → swap axes 0,1
                    // Actually: groups are [g1, g0, g2] in original order
                    // This means the permuted order puts a later group first
                    int cM = groups[0].size;
                    int batch = groups[1].size;
                    int cN = groups[2].size;
                    constexpr int TILE = 32;
                    int total_work = batch * ((cM + TILE - 1) / TILE) * ((cN + TILE - 1) / TILE);
                    int tile_cols = (cN + TILE - 1) / TILE;
                    int tiles_per_batch = ((cM + TILE - 1) / TILE) * tile_cols;
                    nnr::for_static(0, total_work, total_work > 4, [&](int work) {
                        int b = work / tiles_per_batch;
                        int tile = work % tiles_per_batch;
                        int ti = tile / tile_cols;
                        int tj = tile % tile_cols;
                        int i0 = ti * TILE, j0 = tj * TILE;
                        int imax = std::min(i0 + TILE, cM);
                        int jmax = std::min(j0 + TILE, cN);
                        // src layout: [M, batch, N] before perm
                        // dst layout: after perm
                        const T* src_base = px + (size_t)b * cN;
                        T*       dst_base = py + (size_t)b * cN;
                        for (int i = i0; i < imax; ++i)
                            for (int j = j0; j < jmax; ++j)
                                dst_base[j * cM * batch + i * batch] = src_base[i * batch * cN + j];
                    });
                    return true;
                }
            }
            // 2-group case: single swap of two contiguous blocks
            // Source layout: [cN, cM] (group1 before group0 in source order)
            // Dest layout: [cM, cN] (group0 before group1 in perm order)
            if (groups.size() == 2) {
                int cM = groups[0].size;
                int cN = groups[1].size;
#ifdef NNR_ARCH_X64
                if constexpr (std::is_same_v<T, float>) {
                    // Source is [cN rows, cM cols], transpose to [cM rows, cN cols]
                    transpose_2d_avx512((const float*)px, (float*)py, cN, cM);
                    return true;
                }
#endif
                constexpr int TILE = 32;
                int total_work = ((cM + TILE - 1) / TILE) * ((cN + TILE - 1) / TILE);
                int tile_cols = (cN + TILE - 1) / TILE;
                nnr::for_static(0, total_work, total_work > 4, [&](int work) {
                    int ti = work / tile_cols;
                    int tj = work % tile_cols;
                    int i0 = ti * TILE, j0 = tj * TILE;
                    int imax = std::min(i0 + TILE, cM);
                    int jmax = std::min(j0 + TILE, cN);
                    for (int i = i0; i < imax; ++i)
                        for (int j = j0; j < jmax; ++j)
                            py[i * cN + j] = px[j * cM + i];
                });
                return true;
            }
        }

        // Path 4: general fallback — permuted source strides with threading
        small_vector<int> xps(ndim);
        for (int i = 0; i < ndim; ++i)
            xps[i] = x->strides[perm[i]];
        // Thread over the outermost dimension
        int outer_dim = y->dims[0];
        size_t inner_count = l / outer_dim;
        nnr::for_static(0, outer_dim, outer_dim > 4, [&](int o) {
            small_vector<int> idx(ndim);
            std::fill(idx.begin(), idx.end(), 0);
            idx[0] = o;
            T* dst = py + (size_t)o * inner_count;
            for (size_t oy = 0; oy < inner_count; ++oy) {
                int ox = 0;
                for (int i = 0; i < ndim; ++i) ox += idx[i] * xps[i];
                dst[oy] = px[ox];
                // Increment idx for dims 1..ndim-1
                for (int d = ndim - 1; d >= 1; --d) {
                    if (++idx[d] < y->dims[d]) break;
                    idx[d] = 0;
                }
            }
        });
        return true;
    }

    bool exec() override {
        return typed_exec<Transpose_operator,
            opset_t<13, bool_t, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float16_t, float, double, bfloat16_t, std::complex<float>, std::complex<double>, std::string>,
            opset_t<1, bool_t, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float16_t, float, double, std::complex<float>, std::complex<double>, std::string>
        >(this, opset, inputs[0]->type);
    }

};

} // namespace {

// @nnr-meta-op mt=static
operator_t* resolver_default_op_Transpose(int opset, pool_t& pool)
{
    return pool_new<Transpose_operator>(pool);
}

} // namespace nnr
