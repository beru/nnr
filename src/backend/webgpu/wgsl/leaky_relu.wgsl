// LeakyRelu: y = x for x >= 0 else alpha * x. Alpha is supplied at runtime
// in the uniform so one shader covers every LeakyRelu with a different slope.

struct Params { n : u32, alpha_bits : u32, _a : u32, _b : u32 };

@group(0) @binding(0) var<storage, read>       X : array<f32>;
@group(0) @binding(1) var<storage, read_write> Y : array<f32>;
@group(0) @binding(2) var<uniform>             p : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let v     = X[i];
  let alpha = bitcast<f32>(p.alpha_bits);
  Y[i] = select(alpha * v, v, v >= 0.0);
}
