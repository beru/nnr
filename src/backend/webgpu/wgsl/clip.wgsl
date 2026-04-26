// Clip: elementwise clamp to [lo, hi]. Bounds are supplied at runtime via
// the uniform so one shader handles all (min, max) combinations including
// sentinel "unbounded" values the host picks (±3.4e38).

struct Params { n : u32, lo : f32, hi : f32, _pad : u32 };

@group(0) @binding(0) var<storage, read>       X : array<f32>;
@group(0) @binding(1) var<storage, read_write> Y : array<f32>;
@group(0) @binding(2) var<uniform>             p : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  Y[i] = clamp(X[i], p.lo, p.hi);
}
