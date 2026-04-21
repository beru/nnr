// InstanceNormalization: stats pass. One thread per (n, c) pair walks its
// H*W spatial slab and writes (mean, inv_std) into the stats buffer. The
// apply pass then reads those back per output element.

struct Params { NC : u32, HW : u32, epsilon_bits : u32, _pad : u32 };

@group(0) @binding(0) var<storage, read>       X     : array<f32>;
@group(0) @binding(1) var<storage, read_write> Stats : array<f32>;   // [NC, 2]: mean, inv_std
@group(0) @binding(2) var<uniform>             p     : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let nc = gid.x;
  if (nc >= p.NC) { return; }

  let base = nc * p.HW;
  var sum : f32 = 0.0;
  var ssq : f32 = 0.0;
  for (var i : u32 = 0u; i < p.HW; i = i + 1u) {
    let v = X[base + i];
    sum = sum + v;
    ssq = ssq + v * v;
  }
  let n    = f32(p.HW);
  let mean = sum / n;
  // Var = E[x^2] - mean^2. Guard against tiny negatives from catastrophic
  // cancellation by clamping to 0 before adding epsilon.
  let var_ = max(ssq / n - mean * mean, 0.0);
  let eps  = bitcast<f32>(p.epsilon_bits);
  let inv  = 1.0 / sqrt(var_ + eps);
  Stats[nc * 2u + 0u] = mean;
  Stats[nc * 2u + 1u] = inv;
}
