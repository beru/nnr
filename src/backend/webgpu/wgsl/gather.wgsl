// Gather along an axis. The output coord decomposes into three parts:
//   (prefix, idx_flat, suffix) corresponding to data.shape[:axis],
//   indices.shape, and data.shape[axis+1:]. The host uploads a u32 index
//   buffer with negative indices already normalized, so the kernel just
//   looks it up and remaps into the flat input coord.

struct Params {
  total          : u32,
  prefix_count   : u32,
  index_count    : u32,
  suffix_count   : u32,
  axis_dim       : u32,
  grid_stride_x  : u32,   // threads along x for 2D dispatch split
  _pad1          : u32,
  _pad2          : u32,
};

@group(0) @binding(0) var<storage, read>       X   : array<f32>;
@group(0) @binding(1) var<storage, read>       Idx : array<u32>;
@group(0) @binding(2) var<storage, read_write> Y   : array<f32>;
@group(0) @binding(3) var<uniform>             p   : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.y * p.grid_stride_x + gid.x;
  if (i >= p.total) { return; }

  let stride_prefix = p.index_count * p.suffix_count;
  let prefix   = i / stride_prefix;
  let rem1     = i - prefix * stride_prefix;
  let idx_flat = rem1 / p.suffix_count;
  let suffix   = rem1 % p.suffix_count;

  let idx = Idx[idx_flat];
  // Input flat = prefix * axis_dim * suffix + idx * suffix + suffix_off.
  let in_flat = prefix * p.axis_dim * p.suffix_count + idx * p.suffix_count + suffix;
  Y[i] = X[in_flat];
}
