// Softmax along an arbitrary axis. Generalized form of the stride-1 last-axis
// case: for a rank-r input coerced to [outer, N, inner] (outer = prod of dims
// before axis, N = dims[axis], inner = prod of dims after axis), each
// (outer, inner) pair reduces N elements that sit `inner` apart in memory.
//
// Dispatch: one workgroup per (outer, inner) pair — grid (outer, inner, 1).
// Each workgroup reduces N elements across 256 threads in three passes:
//   1. rowwise max (for numerical stability)
//   2. exp(x - max) + running sum
//   3. divide by sum to normalize
// For N > WG each thread strides by WG; N < WG leaves some threads idle
// but the barriers still work because all threads execute the loops.
//
// Last-axis case (inner == 1) is the fast path of this shader — the stride
// multiply folds to `+ j`, matching the original 2D [outer, N] kernel.
// Strided case (inner > 1) is uncoalesced on the reduce axis; perf can be
// revisited (shared-memory tiling on stride) later — functional parity first.

struct Dims { outer : u32, N : u32, inner : u32, _a : u32 };

@group(0) @binding(0) var<storage, read>       X    : array<f32>;
@group(0) @binding(1) var<storage, read_write> Y    : array<f32>;
@group(0) @binding(2) var<uniform>             dims : Dims;

const WG : u32 = 256u;
var<workgroup> sh : array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let o = wid.x;
  let p = wid.y;
  if (o >= dims.outer || p >= dims.inner) { return; }
  let N      = dims.N;
  let stride = dims.inner;
  let base   = o * N * stride + p;

  // --- 1. rowwise max ---
  var m : f32 = -3.4e38;
  for (var j : u32 = lid.x; j < N; j = j + WG) {
    m = max(m, X[base + j * stride]);
  }
  sh[lid.x] = m;
  workgroupBarrier();
  var s : u32 = WG / 2u;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) { sh[lid.x] = max(sh[lid.x], sh[lid.x + s]); }
    workgroupBarrier();
    s = s / 2u;
  }
  let row_max = sh[0];
  workgroupBarrier();

  // --- 2. exp(x - max), store to Y, accumulate sum ---
  var sum : f32 = 0.0;
  for (var j : u32 = lid.x; j < N; j = j + WG) {
    let i = base + j * stride;
    let e = exp(X[i] - row_max);
    Y[i] = e;
    sum = sum + e;
  }
  sh[lid.x] = sum;
  workgroupBarrier();
  s = WG / 2u;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) { sh[lid.x] = sh[lid.x] + sh[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  let row_sum = sh[0];
  let inv = 1.0 / row_sum;
  workgroupBarrier();

  // --- 3. normalize ---
  for (var j : u32 = lid.x; j < N; j = j + WG) {
    let i = base + j * stride;
    Y[i] = Y[i] * inv;
  }
}
