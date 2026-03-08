#include <limits>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct variadic_broadcast_base : operator_t {
    bool init() override { return (inputs.size() >= 1) && (outputs.size() == 1); }
    bool reshape() override {
        tensor_t* y = outputs[0];
        if (!y->reshape_identity(inputs[0])) return false;
        for (size_t i = 1; i < inputs.size(); ++i)
            if (!y->reshape_multi_broadcast(y, inputs[i], y->type)) return false;
        return true;
    }
};

struct Min_op : variadic_broadcast_base {
    template <typename T>
    bool exec() {
        tensor_t* y = outputs[0];
        T* py = (T*)y->data;
        for (size_t i = 0, l = y->ndata; i < l; ++i) {
            T minv = std::numeric_limits<T>::max();
            for (size_t j = 0; j < inputs.size(); ++j) {
                const T* px = (const T*)inputs[j]->broadcast_map_address(y, i);
                if (*px < minv) minv = *px;
            }
            py[i] = minv;
        }
        return true;
    }
    bool exec() override {
        return typed_exec<Min_op,
            opset_t<13, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, bfloat16_t, float16_t, float, double>,
            opset_t<12, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float16_t, float, double>,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

struct Max_op : variadic_broadcast_base {
    template <typename T>
    bool exec() {
        tensor_t* y = outputs[0];
        T* py = (T*)y->data;
        for (size_t i = 0, l = y->ndata; i < l; ++i) {
            T maxv = std::numeric_limits<T>::lowest();
            for (size_t j = 0; j < inputs.size(); ++j) {
                const T* px = (const T*)inputs[j]->broadcast_map_address(y, i);
                if (*px > maxv) maxv = *px;
            }
            py[i] = maxv;
        }
        return true;
    }
    bool exec() override {
        return typed_exec<Max_op,
            opset_t<13, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, bfloat16_t, float16_t, float, double>,
            opset_t<12, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float16_t, float, double>,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

struct Mean_op : variadic_broadcast_base {
    template <typename T>
    bool exec() {
        tensor_t* y = outputs[0];
        T* py = (T*)y->data;
        T inv = T(1.0 / inputs.size());
        for (size_t i = 0, l = y->ndata; i < l; ++i) {
            T sum = 0;
            for (size_t j = 0; j < inputs.size(); ++j)
                sum += *(const T*)inputs[j]->broadcast_map_address(y, i);
            py[i] = sum * inv;
        }
        return true;
    }
    bool exec() override {
        return typed_exec<Mean_op,
            opset_t<13, bfloat16_t, float16_t, float, double>,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

struct Sum_op : variadic_broadcast_base {
    template <typename T>
    bool exec() {
        tensor_t* y = outputs[0];
        T* py = (T*)y->data;
        size_t l = y->ndata;
        // Fast path: all inputs same shape as output
        bool all_same = true;
        for (size_t j = 0; j < inputs.size(); j++) {
            if (inputs[j]->ndata != l || inputs[j]->ndim != y->ndim
                || memcmp(inputs[j]->dims, y->dims, y->ndim * sizeof(int)) != 0) {
                all_same = false;
                break;
            }
        }
        if (all_same) {
            memcpy(py, (const T*)inputs[0]->data, l * sizeof(T));
            for (size_t j = 1; j < inputs.size(); j++) {
                const T* px = (const T*)inputs[j]->data;
                for (size_t i = 0; i < l; i++)
                    py[i] += px[i];
            }
            return true;
        }
        for (size_t i = 0; i < l; ++i) {
            T sum = 0;
            for (size_t j = 0; j < inputs.size(); ++j)
                sum += *(const T*)inputs[j]->broadcast_map_address(y, i);
            py[i] = sum;
        }
        return true;
    }
    bool exec() override {
        return typed_exec<Sum_op,
            opset_t<13, bfloat16_t, float16_t, float, double>,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op op=Min mt=no
operator_t* resolver_default_op_Min(int opset, pool_t& pool) { return pool_new<Min_op>(pool); }
// @nnr-meta-op op=Max mt=no
operator_t* resolver_default_op_Max(int opset, pool_t& pool) { return pool_new<Max_op>(pool); }
// @nnr-meta-op op=Mean mt=no
operator_t* resolver_default_op_Mean(int opset, pool_t& pool) { return pool_new<Mean_op>(pool); }
// @nnr-meta-op op=Sum mt=no
operator_t* resolver_default_op_Sum(int opset, pool_t& pool) { return pool_new<Sum_op>(pool); }

} // namespace nnr
