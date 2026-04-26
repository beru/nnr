// LayerNormalization over the last axis.
// Input X shape [outer, N]; Scale & Bias shape [N].
// For each outer row: compute mean, variance, then
//   y[j] = ((x[j] - mean) / sqrt(var + eps)) * scale[j] + bias[j]
//
// Dispatch: outer workgroups, 256 threads each. Two-pass reduction using
// shared memory (sum, then sum-of-squared-deltas).

struct Dims { outer : u32, N : u32, _a : u32, _b : u32 };
struct Cfg  { eps : f32, _a : u32, _b : u32, _c : u32 };

@group(0) @binding(0) var<storage, read>       X     : array<f32>;
@group(0) @binding(1) var<storage, read>       Scale : array<f32>;
@group(0) @binding(2) var<storage, read>       Bias  : array<f32>;
@group(0) @binding(3) var<storage, read_write> Y     : array<f32>;
@group(0) @binding(4) var<uniform>             dims  : Dims;
@group(0) @binding(5) var<uniform>             cfg   : Cfg;

const WG : u32 = 256u;
var<workgroup> sh : array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let row = wid.x;
  if (row >= dims.outer) { return; }
  let N = dims.N;
  let off = row * N;
  let invN = 1.0 / f32(N);

  // --- pass 1: sum ---
  var s : f32 = 0.0;
  for (var j : u32 = lid.x; j < N; j = j + WG) {
    s = s + X[off + j];
  }
  sh[lid.x] = s;
  workgroupBarrier();
  var stride : u32 = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) { sh[lid.x] = sh[lid.x] + sh[lid.x + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let mean = sh[0] * invN;
  workgroupBarrier();

  // --- pass 2: sum of squared deviations ---
  var ss : f32 = 0.0;
  for (var j : u32 = lid.x; j < N; j = j + WG) {
    let d = X[off + j] - mean;
    ss = ss + d * d;
  }
  sh[lid.x] = ss;
  workgroupBarrier();
  stride = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) { sh[lid.x] = sh[lid.x] + sh[lid.x + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let variance = sh[0] * invN;
  let inv_std = 1.0 / sqrt(variance + cfg.eps);
  workgroupBarrier();

  // --- pass 3: write normalized output ---
  for (var j : u32 = lid.x; j < N; j = j + WG) {
    let normalized = (X[off + j] - mean) * inv_std;
    Y[off + j] = normalized * Scale[j] + Bias[j];
  }
}
