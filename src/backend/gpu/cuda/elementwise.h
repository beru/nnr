#pragma once
// Shared NVRTC source for elementwise pointwise kernels.
// One module compiled once per backend (keyed by kernel set), four CUfunctions
// inside it: add_f32, mul_f32, relu_f32, sigmoid_f32.

#if defined(NNR_USE_CUDA)

namespace nnr::gpu {

inline const char* elementwise_f32_source() {
    return R"CUDA(
extern "C" {

__global__ void add_f32(const float* __restrict__ a,
                        const float* __restrict__ b,
                        float* __restrict__ y,
                        unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

__global__ void mul_f32(const float* __restrict__ a,
                        const float* __restrict__ b,
                        float* __restrict__ y,
                        unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i];
}

__global__ void sub_f32(const float* __restrict__ a,
                        const float* __restrict__ b,
                        float* __restrict__ y,
                        unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] - b[i];
}

__global__ void div_f32(const float* __restrict__ a,
                        const float* __restrict__ b,
                        float* __restrict__ y,
                        unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] / b[i];
}

// int64 binary kernels (used by index manipulation ops in detection post-processing).
__global__ void add_i64(const long long* __restrict__ a,
                        const long long* __restrict__ b,
                        long long* __restrict__ y,
                        unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}
__global__ void mul_i64(const long long* __restrict__ a,
                        const long long* __restrict__ b,
                        long long* __restrict__ y,
                        unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i];
}
__global__ void sub_i64(const long long* __restrict__ a,
                        const long long* __restrict__ b,
                        long long* __restrict__ y,
                        unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] - b[i];
}
__global__ void add_i64_scalar_b(const long long* __restrict__ a,
                                 const long long* __restrict__ b,
                                 long long* __restrict__ y,
                                 unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    long long bv = b[0];
    if (i < n) y[i] = a[i] + bv;
}
__global__ void mul_i64_scalar_b(const long long* __restrict__ a,
                                 const long long* __restrict__ b,
                                 long long* __restrict__ y,
                                 unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    long long bv = b[0];
    if (i < n) y[i] = a[i] * bv;
}
__global__ void sub_i64_scalar_b(const long long* __restrict__ a,
                                 const long long* __restrict__ b,
                                 long long* __restrict__ y,
                                 unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    long long bv = b[0];
    if (i < n) y[i] = a[i] - bv;
}

__global__ void relu_f32(const float* __restrict__ x,
                         float* __restrict__ y,
                         unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v > 0.f ? v : 0.f;
    }
}

__global__ void sigmoid_f32(const float* __restrict__ x,
                            float* __restrict__ y,
                            unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = 1.f / (1.f + __expf(-v));
    }
}

__global__ void tanh_f32(const float* __restrict__ x,
                         float* __restrict__ y,
                         unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = tanhf(x[i]);
}

__global__ void abs_f32(const float* __restrict__ x,
                        float* __restrict__ y,
                        unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = fabsf(x[i]);
}

// GELU (tanh approximation, as used by ONNX Gelu opset 20 approximate="tanh"
// and by most transformer models). Faster than erf-based exact.
__global__ void gelu_f32(const float* __restrict__ x,
                         float* __restrict__ y,
                         unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        const float k = 0.7978845608028654f;   // sqrt(2/pi)
        float t = k * (v + 0.044715f * v * v * v);
        y[i] = 0.5f * v * (1.f + tanhf(t));
    }
}

// Activations with attributes (alpha/beta as kernel args).
__global__ void leaky_relu_f32(const float* __restrict__ x,
                               float* __restrict__ y,
                               unsigned long long n,
                               float alpha)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v > 0.f ? v : (alpha * v);
    }
}

__global__ void elu_f32(const float* __restrict__ x,
                        float* __restrict__ y,
                        unsigned long long n,
                        float alpha)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v > 0.f ? v : (alpha * (__expf(v) - 1.f));
    }
}

__global__ void celu_f32(const float* __restrict__ x,
                         float* __restrict__ y,
                         unsigned long long n,
                         float alpha)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = fmaxf(0.f, v) + fminf(0.f, alpha * (__expf(v / alpha) - 1.f));
    }
}
)CUDA"
R"CUDA(
__global__ void selu_f32(const float* __restrict__ x,
                         float* __restrict__ y,
                         unsigned long long n,
                         float alpha,
                         float gamma)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = gamma * (v > 0.f ? v : (alpha * __expf(v) - alpha));
    }
}

__global__ void hard_sigmoid_f32(const float* __restrict__ x,
                                 float* __restrict__ y,
                                 unsigned long long n,
                                 float alpha,
                                 float beta)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = alpha * x[i] + beta;
        y[i] = fminf(1.f, fmaxf(0.f, v));
    }
}

__global__ void hard_swish_f32(const float* __restrict__ x,
                               float* __restrict__ y,
                               unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float hs = fminf(1.f, fmaxf(0.f, v / 6.f + 0.5f));
        y[i] = v * hs;
    }
}

__global__ void clip_f32(const float* __restrict__ x,
                         float* __restrict__ y,
                         unsigned long long n,
                         float lo,
                         float hi)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = fminf(hi, fmaxf(lo, x[i]));
}

__global__ void softplus_f32(const float* __restrict__ x,
                             float* __restrict__ y,
                             unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    // log(1 + exp(v)) — stable form uses log1p and max
    if (i < n) {
        float v = x[i];
        y[i] = (v > 0.f) ? (v + log1pf(__expf(-v))) : log1pf(__expf(v));
    }
}

__global__ void softsign_f32(const float* __restrict__ x,
                             float* __restrict__ y,
                             unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; y[i] = v / (1.f + fabsf(v)); }
}

// Pure pointwise math.
__global__ void neg_f32(const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = -x[i];
}

__global__ void sqrt_f32(const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = sqrtf(x[i]);
}

__global__ void exp_f32(const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = __expf(x[i]);
}

__global__ void log_f32(const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = __logf(x[i]);
}

__global__ void erf_f32(const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = erff(x[i]);
}

__global__ void ceil_f32(const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = ceilf(x[i]);
}

__global__ void floor_f32(const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = floorf(x[i]);
}

__global__ void reciprocal_f32(const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.f / x[i];
}

// Trig
__global__ void sin_f32 (const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{ unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; if (i < n) y[i] = __sinf(x[i]); }
__global__ void cos_f32 (const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{ unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; if (i < n) y[i] = __cosf(x[i]); }
__global__ void tan_f32 (const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{ unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; if (i < n) y[i] = tanf(x[i]); }
__global__ void asin_f32(const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{ unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; if (i < n) y[i] = asinf(x[i]); }
__global__ void acos_f32(const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{ unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; if (i < n) y[i] = acosf(x[i]); }
__global__ void atan_f32(const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{ unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; if (i < n) y[i] = atanf(x[i]); }

// Hyperbolic
__global__ void sinh_f32 (const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{ unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; if (i < n) y[i] = sinhf(x[i]); }
__global__ void cosh_f32 (const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{ unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; if (i < n) y[i] = coshf(x[i]); }
__global__ void asinh_f32(const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{ unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; if (i < n) y[i] = asinhf(x[i]); }
__global__ void acosh_f32(const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{ unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; if (i < n) y[i] = acoshf(x[i]); }
__global__ void atanh_f32(const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{ unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; if (i < n) y[i] = atanhf(x[i]); }

// Rounding / Sign
__global__ void round_f32(const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{ unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; if (i < n) y[i] = nearbyintf(x[i]); }
__global__ void sign_f32 (const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; y[i] = (v > 0.f) - (v < 0.f); }
}

// Swish / Mish (not standard ONNX but present in NNR CPU backend)
__global__ void swish_f32(const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; y[i] = v / (1.f + __expf(-v)); }
}
__global__ void mish_f32(const float* __restrict__ x, float* __restrict__ y, unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; y[i] = v * tanhf(log1pf(__expf(v))); }
}

// PRelu: per-channel slope tensor (slope shape broadcasts to (C,) for NCHW).
// Handles scalar slope (slope_ndata == 1) and per-channel.
__global__ void prelu_f32_scalar(const float* __restrict__ x,
                                 const float* __restrict__ slope,
                                 float* __restrict__ y,
                                 unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v > 0.f ? v : (slope[0] * v);
    }
}

__global__ void prelu_f32_per_channel(const float* __restrict__ x,
                                      const float* __restrict__ slope,
                                      float* __restrict__ y,
                                      unsigned long long n,
                                      int C, int inner)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) {
        int c = (int)((i / (unsigned long long)inner) % (unsigned long long)C);
        float v = x[i];
        y[i] = v > 0.f ? v : (slope[c] * v);
    }
}

// Broadcast: scalar on B (b_ndata == 1).
__global__ void add_f32_scalar_b(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ y,
                                 unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[0];
}

__global__ void mul_f32_scalar_b(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ y,
                                 unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[0];
}

__global__ void sub_f32_scalar_b(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ y,
                                 unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] - b[0];
}

__global__ void div_f32_scalar_b(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ y,
                                 unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] / b[0];
}

// Broadcast: per-channel bias. A shape is (N, C, inner), B is (C,) or any shape
// with C total elements. Element index decomposes as (n*C + c)*inner + k.
// Caller passes n_total = N*C*inner, C, inner.
__global__ void add_f32_bias_nchw(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ y,
                                  unsigned long long n_total,
                                  int C, int inner)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n_total) {
        int c = (int)((i / (unsigned long long)inner) % (unsigned long long)C);
        y[i] = a[i] + b[c];
    }
}

__global__ void mul_f32_bias_nchw(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ y,
                                  unsigned long long n_total,
                                  int C, int inner)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n_total) {
        int c = (int)((i / (unsigned long long)inner) % (unsigned long long)C);
        y[i] = a[i] * b[c];
    }
}

} // extern "C"
)CUDA";
}

} // namespace nnr::gpu

#endif // NNR_USE_CUDA
