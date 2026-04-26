// BatchNormalization (inference). Input layout is NCHW — channel axis = 1.
// For each element at flat index i, channel = (i / per_channel) % C.
// Output: y = (x - mean[c]) / sqrt(var[c] + eps) * scale[c] + bias[c].

struct Params { total : u32, channels : u32, per_channel : u32, epsilon_bits : u32 };

@group(0) @binding(0) var<storage, read>       X     : array<f32>;
@group(0) @binding(1) var<storage, read>       Scale : array<f32>;
@group(0) @binding(2) var<storage, read>       Bias  : array<f32>;
@group(0) @binding(3) var<storage, read>       Mean  : array<f32>;
@group(0) @binding(4) var<storage, read>       Var   : array<f32>;
@group(0) @binding(5) var<storage, read_write> Y     : array<f32>;
@group(0) @binding(6) var<uniform>             p     : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.total) { return; }
  let c    = (i / p.per_channel) % p.channels;
  let eps  = bitcast<f32>(p.epsilon_bits);
  let inv  = 1.0 / sqrt(Var[c] + eps);
  Y[i] = (X[i] - Mean[c]) * inv * Scale[c] + Bias[c];
}
