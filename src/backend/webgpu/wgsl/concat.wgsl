// Per-input slice-copy used by Concat. Each Concat dispatch invokes this
// once per input tensor; the host writes a fresh `Meta` for each input
// describing where that input's slice lands inside the output.
//
// For each element i in [0, total):
//   coord = unflatten(i, in_dims)
//   dst   = sum(coord[k] * dst_strides[k]) + flat_offset
// where flat_offset = axis_offset * dst_strides[axis] is precomputed on the
// host, so the kernel has no per-axis branch.

struct Meta {
  total            : u32,
  ndim             : u32,
  flat_offset      : u32,
  grid_stride_x    : u32,   // threads along x for 2D dispatch split
  in_dims_lo       : vec4<u32>,
  in_dims_hi       : vec4<u32>,
  in_strides_lo    : vec4<u32>,
  in_strides_hi    : vec4<u32>,
  dst_strides_lo   : vec4<u32>,
  dst_strides_hi   : vec4<u32>,
};

@group(0) @binding(0) var<storage, read>       X  : array<f32>;
@group(0) @binding(1) var<storage, read_write> Y  : array<f32>;
@group(0) @binding(2) var<storage, read>       md : Meta;

fn get_in_dim(i : u32) -> u32 { if (i < 4u) { return md.in_dims_lo[i]; } return md.in_dims_hi[i - 4u]; }
fn get_in_stride(i : u32) -> u32 { if (i < 4u) { return md.in_strides_lo[i]; } return md.in_strides_hi[i - 4u]; }
fn get_dst_stride(i : u32) -> u32 { if (i < 4u) { return md.dst_strides_lo[i]; } return md.dst_strides_hi[i - 4u]; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.y * md.grid_stride_x + gid.x;
  if (i >= md.total) { return; }

  var dst : u32 = md.flat_offset;
  var tmp : u32 = i;
  for (var k : i32 = i32(md.ndim) - 1; k >= 0; k = k - 1) {
    let d = get_in_dim(u32(k));
    let c = tmp % d;
    tmp   = tmp / d;
    dst   = dst + c * get_dst_stride(u32(k));
  }

  Y[dst] = X[i];
}
