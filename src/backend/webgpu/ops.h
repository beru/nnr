// X-macro list of operators implemented by the WebGPU backend.
// Included by solve_operator.cpp to forward-declare resolvers and register
// them with the global registry under backend_t::WEBGPU.

// GEMM / MatMul
X(MatMul)
X(Gemm)
X(Conv)
X(ConvTranspose)

// Data movement
X(Transpose)
X(Concat)
X(Expand)
X(Slice)
X(Split)
X(Pad)
X(Gather)
X(GatherND)
X(ScatterND)
X(ScatterElements)
X(OneHot)
X(Tile)
X(Range)
X(Resize)

// Structural ops (fill / reshape-by-index)
X(ConstantOfShape)
X(DepthToSpace)
X(SpaceToDepth)

// Masks / prefix scans
X(Trilu)
X(CumSum)

// Elementwise with runtime params (not fit for the unary op_expr() macro)
X(Clip)

// Dtype conversion (supports f32/i32/u32 pairs; other dtypes → CPU fallback)
X(Cast)

// Ternary select (same-shape only, cond ∈ {u32, i32}; bool-cond/broadcast → CPU)
X(Where)

// Comparison ops (f32, f32) → u32 mask with NumPy/ONNX broadcasting.
// See comparison.cpp for the CPU-fallback cases.
X(Equal)
X(Greater)
X(Less)
X(GreaterOrEqual)
X(LessOrEqual)

// Logical ops on u32 masks (same file). Treat any non-zero as true,
// emit canonical 0/1.
X(And)
X(Or)
X(Xor)
X(Not)

// Bitwise ops on u32 — true bit manipulation, distinct from logical above.
X(BitwiseAnd)
X(BitwiseOr)
X(BitwiseXor)
X(BitwiseNot)
X(BitShift)

// Reductions (one naive thread-per-output-element kernel; see reduce.cpp)
X(ReduceSum)
X(ReduceMean)
X(ReduceMax)
X(ReduceMin)
X(ReduceProd)
X(ReduceSumSquare)
X(ReduceL1)
X(ReduceL2)
X(ReduceLogSum)
X(ReduceLogSumExp)

// Reduce-to-index (axis → single index). Output is i32 (we truncate i64).
X(ArgMax)
X(ArgMin)

// Top-K along an axis: emits K values (f32) + K indices (INT64 narrowed
// to i32 on GPU). See TopK.cpp for K bound (K_MAX=256).
X(TopK)

// Einsum (2-input, f32, no ellipsis, no repeated indices on one operand).
// Handles matmul variants, batched matmul, transpose-matmul, attention
// contractions. Parses equation at init + runtime-compiles one kernel
// per (output_ndim, n_contract) combination.
X(Einsum)

// 2D NCHW pooling (shared base; see pool_base.cpp)
X(MaxPool)
X(AveragePool)
X(GlobalMaxPool)
X(GlobalAveragePool)

// Parameterized activations
X(LeakyRelu)

// Normalization / activation over an axis
X(Softmax)
X(LayerNormalization)
X(BatchNormalization)
X(InstanceNormalization)

// View ops (zero-kernel — alias the GPU buffer)
X(Reshape)
X(Identity)
X(Flatten)
X(Squeeze)
X(Unsqueeze)
X(Dropout)

// Unary float32 elementwise (see unary_math.cpp)
X(Relu)
X(Sigmoid)
X(Tanh)
X(Abs)
X(Neg)
X(Exp)
X(Log)
X(Sqrt)
X(Ceil)
X(Floor)
X(Round)
X(Sin)
X(Cos)
X(Reciprocal)
X(Softplus)
X(Softsign)
X(Gelu)
X(HardSigmoid)
X(HardSwish)
X(Elu)
X(Mish)
X(Sign)
X(Sinh)
X(Cosh)
X(Tan)
X(Asin)
X(Acos)
X(Atan)
X(Asinh)
X(Acosh)
X(Atanh)
X(Erf)

// Binary float32 elementwise with NumPy/ONNX broadcast (see binary_math.cpp)
X(Add)
X(Sub)
X(Mul)
X(Div)
X(Pow)
X(Max)
X(Min)
X(PRelu)
