// X-macro list of same-shape f32×f32→f32 binary elementwise ops and their
// "pipe-first" WGSL patterns — `v` is the pipe input (output of the
// previous stage in the fused chain) and `$s` is a placeholder for the
// side input access, substituted to `S<k>[i]` per stage by the fusion
// pass.
//
// Consumed by:
//   - binary_math.cpp — fusable_binary_op_pattern() switch table
//   - graph_optimizer/fuse_webgpu_elementwise.cpp — via fusable_binary_op_pattern
//
// Note: the underlying ONNX op may accept either argument order, but fusion
// only handles the "pipe is inputs[0]" configuration. A producer whose
// inputs[1] is the pipe won't be fused (chain breaks).

BINARY(Add,   "v + ($s)")
BINARY(Sub,   "v - ($s)")
BINARY(Mul,   "v * ($s)")
BINARY(Div,   "v / ($s)")
BINARY(Pow,   "pow(v, ($s))")
BINARY(Max,   "max(v, ($s))")
BINARY(Min,   "min(v, ($s))")
// PRelu(x, slope): x if x>=0 else slope*x. `v` = x = pipe, `$s` = slope.
BINARY(PRelu, "select(($s) * v, v, v >= 0.0)")
