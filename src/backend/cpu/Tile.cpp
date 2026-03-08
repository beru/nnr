#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Tile_operator : public operator_t {

    bool init() override {
        return is_inout_size(2, 1);
    }

    bool reshape() override {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        const tensor_t* r = inputs[1];
        const int64_t* pr = (const int64_t*)r->data;
        const int ndim = x->ndim;
        small_vector<int> dims(ndim);

        for (int i = 0; i < ndim; ++i) {
            dims[i] = x->dims[i] * pr[i];
        }
        return y->reshape(dims, x->type);
    }

    template <typename T>
    bool exec() {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        T* py = (T*)y->data;
        const T* px = (const T*)x->data;

        for (size_t i = 0, l = y->ndata; i < l; ++i) {
            px = (const T*)x->broadcast_map_address(y, i);
            py[i] = *px;
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Tile_operator,
            opset_t<13, bool_t, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float16_t, float, double, bfloat16_t, std::complex<float>, std::complex<double>, std::string>,
            opset_t<6, bool_t, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float16_t, float, double, std::complex<float>, std::complex<double>, std::string>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace {

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Tile(int opset, pool_t& pool)
{
    return pool_new<Tile_operator>(pool);
}

} // namespace nnr
