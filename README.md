# NNR

Lightweight neural network inference runtime. Loads ONNX and TFLite models and runs them on CPU (x64/ARM64) or NVIDIA GPU with no external dependencies.

Intended as a **readable baseline** for embedded developers porting inference to less mainstream CPUs — not a drop-in competitor to ORT/MNN/ncnn on flagship hardware. Kernels live in plain C++20 headers per ISA (`backend/x64/`, `backend/arm/`, `backend/gpu/cuda/`); adding a new backend is mostly writing one kernel file and wiring it into the dispatch switch. Zero runtime dependencies keeps the port surface small.

## Features

- **Formats**: ONNX (custom protobuf parser) and TFLite (hand-rolled FlatBuffer reader). 203 ONNX operators across opset 1–22.
- **x64 kernels**: AVX-512 and AVX2. Tiled GEMM, direct/im2col convolution, Winograd F(2×2,3×3) and F(4×4,3×3) in both NCHW and NCHWc layouts, depthwise/grouped convolution. Int8 via VNNI (VPDPBUSD) with QLinearConv/MatMul fast paths. JIT micro-kernels via Xbyak for specialized shapes.
- **ARM64 kernels**: NEON GEMM, Winograd, depthwise/pointwise/first-layer/last-layer convolutions. Fp16 compute, int8 dot-product. JIT via Xbyak_aarch64.
- **CUDA backend** (`-DNNR_USE_CUDA=ON`): 93 operators on NVIDIA GPUs. WMMA TensorCore GEMM (TF32 fp32, signed-int8 with shift-by-−128 unification), cp.async double-buffered tiles, fused epilogues, depthwise/grouped Conv, per-tensor device residency cache, CUDA Graph capture/replay (~100× lower launch overhead for small-kernel models). NVRTC runtime compilation — no offline `nvcc` step.
- **Graph optimizer**: Conv+BN fusion, post-op fusion (Relu/Sigmoid/Clip/HardSwish/Add), QDQ fusion for int8 paths, cost-based layout selection across NCHW/NHWC/NCHWc, constant folding, operator decomposition, transpose cancellation.
- **Runtime**: work-stealing thread pool (per-core run queues, adaptive spin + MONITORX/MWAITX on AMD), tensor pool with in-place optimization and lifetime analysis, zero heap allocation during inference, runtime CPU cache-topology detection (L1d/L2/L3-per-domain) feeding backend block-size and workspace budgets.
- **Distribution**: no external runtime dependencies (no protobuf, no flatbuffers, no BLAS, no ONNX Runtime, no cuBLAS, no cuDNN). Only Xbyak/Xbyak_aarch64 as header-only JIT assemblers via submodule, and enchantum as a header-only enum reflection library, and the CUDA Toolkit (runtime + NVRTC + driver API) when CUDA is enabled.

## Build

```bash
# Clone with submodules (third_party/xbyak, third_party/xbyak_aarch64)
git clone --recurse-submodules <repo-url>
cd nnr

# Configure + build (CPU only)
cmake -S . -B build
cmake --build build --config Release

# Configure + build with CUDA backend (requires CUDA Toolkit 12+)
cmake -S . -B build -DNNR_USE_CUDA=ON
cmake --build build --config Release
```

Requires CMake 3.16+ and a C++20 compiler (MSVC 2022, GCC 12+, Clang 15+). CUDA backend additionally requires the CUDA Toolkit (runtime headers + NVRTC); compiles for `sm_75` and up.

## Project layout

```
src/
  nnr.h, nnr.cpp           Core types (tensor_t, operator_t, context_t)
  graph_optimizer.cpp      Fusion, layout selection
  memory_planner.cpp       Tensor pool allocation
  thread_pool.h            Lock-free work-stealing thread pool
  cache_topology.h         Runtime L1d/L2/L3 detection
  cpu_features.h           ISA feature detection (CPUID)
  format/onnx/             ONNX loader (custom protobuf parser)
  format/tflite/           TFLite loader (hand-rolled FlatBuffer reader)
  backend/cpu/             Operator implementations (203 ops)
  backend/cpu/kernel/      ISA-dispatch wrappers (pool, gemm, etc.)
  backend/x64/             AVX-512/AVX2 kernels, JIT (Xbyak)
  backend/arm/             NEON kernels
  backend/gpu/             Device abstraction, per-tensor residency cache
  backend/gpu/cuda/        CUDA ops, NVRTC kernel cache, WMMA GEMM, graph replay
third_party/
  xbyak/                   x64 JIT assembler (submodule)
  xbyak_aarch64/           ARM64 JIT assembler (submodule)
  enchantum.hpp            Enum reflection library (single header)
```

## Acknowledgments

NNR originally stems from [libonnx](https://github.com/xboot/libonnx) — a
small, embeddable ONNX inference engine written in C. The "one TU per
operator, no external deps, readable from top to bottom" philosophy, the
overall shape of the operator registration / dispatch, and the TFLite /
ONNX loading layout all carry that DNA forward.

Many of the graph-optimization and layout-selection ideas were then
modelled directly on [ONNX Runtime](https://github.com/microsoft/onnxruntime):
the transpose-propagation / cancellation optimizer, the NCHWc blocked
layout and Winograd workspace strategy, QDQ fusion for int8 paths,
Conv post-op fusion, and much of the cost-based layout dispatch all
trace their lineage to ORT's design. NNR re-implements these from
scratch in a compact, dependency-free form aimed at embedded ports —
think of it as "the ORT playbook on top of the libonnx chassis,
condensed and portable."

Micro-kernel structure (GEMM blocking, tile sizes, packed-B formats)
draws additional inspiration from [MNN](https://github.com/alibaba/MNN),
[ncnn](https://github.com/Tencent/ncnn), and oneDNN.

## License

NNR is released under the [Boost Software License 1.0](LICENSE).

The bundled third-party code keeps its own licenses: `third_party/xbyak` is
BSD-3-Clause, `third_party/xbyak_aarch64` is Apache 2.0, and
`third_party/enchantum.hpp` is MIT. Redistributions must preserve all three.
