// InstanceNormalization: apply pass. One thread per output element. Reads
// (mean, inv_std) from the stats buffer at index (n * C + c), then writes
// y = (x - mean) * inv_std * scale[c] + bias[c].

struct Params { total : u32, C : u32, HW : u32, _pad : u32 };

@group(0) @binding(0) var<storage, read>       X     : array<f32>;
@group(0) @binding(1) var<storage, read>       Scale : array<f32>;
@group(0) @binding(2) var<storage, read>       Bias  : array<f32>;
@group(0) @binding(3) var<storage, read>       Stats : array<f32>;   // [NC, 2]
@group(0) @binding(4) var<storage, read_write> Y     : array<f32>;
@group(0) @binding(5) var<uniform>             p     : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.total) { return; }
  let nc   = i / p.HW;
  let c    = nc % p.C;
  let mean = Stats[nc * 2u + 0u];
  let inv  = Stats[nc * 2u + 1u];
  Y[i] = (X[i] - mean) * inv * Scale[c] + Bias[c];
}
