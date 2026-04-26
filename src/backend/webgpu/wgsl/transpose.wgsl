// Generic N-D Transpose (N <= 8).
// out[o0, o1, ..., on-1] = in[o_perm[0], o_perm[1], ...]
// where out_dim[k] == in_dim[perm[k]].
//
// Dispatch: (N + 63) / 64 workgroups of 64 threads, one element per thread.
// Each thread unpacks its linear output index into N-D, maps through perm
// to an N-D input index, then re-flattens using input strides.

// All u32 arrays of length 8 are packed as 2 vec4<u32>s to force predictable
// 16-byte alignment (avoids WGSL storage-buffer struct layout surprises).
struct Meta {
  total          : u32,
  ndim           : u32,
  grid_stride_x  : u32,   // threads per row in 2D-split dispatch (= gx * 64)
  _b             : u32,
  in_strides_lo  : vec4<u32>,   // [0..3]
  in_strides_hi  : vec4<u32>,   // [4..7]
  out_dims_lo    : vec4<u32>,
  out_dims_hi    : vec4<u32>,
  perm_lo        : vec4<u32>,
  perm_hi        : vec4<u32>,
};

@group(0) @binding(0) var<storage, read>       X    : array<f32>;
@group(0) @binding(1) var<storage, read_write> Y    : array<f32>;
@group(0) @binding(2) var<storage, read>       md : Meta;

fn get_stride(i : u32) -> u32 {
  if (i < 4u) { return md.in_strides_lo[i]; }
  return md.in_strides_hi[i - 4u];
}
fn get_out_dim(i : u32) -> u32 {
  if (i < 4u) { return md.out_dims_lo[i]; }
  return md.out_dims_hi[i - 4u];
}
fn get_perm(i : u32) -> u32 {
  if (i < 4u) { return md.perm_lo[i]; }
  return md.perm_hi[i - 4u];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  // Host may dispatch 2D when the 1D workgroup count exceeds WebGPU's
  // 65535 per-dim cap. Reconstruct the flat thread id.
  let o = gid.y * md.grid_stride_x + gid.x;
  if (o >= md.total) { return; }

  var out_idx : array<u32, 8>;
  var tmp : u32 = o;
  for (var k : i32 = i32(md.ndim) - 1; k >= 0; k = k - 1) {
    let d = get_out_dim(u32(k));
    out_idx[u32(k)] = tmp % d;
    tmp = tmp / d;
  }

  var in_flat : u32 = 0u;
  for (var k : u32 = 0u; k < md.ndim; k = k + 1u) {
    in_flat = in_flat + out_idx[k] * get_stride(get_perm(k));
  }

  Y[o] = X[in_flat];
}
