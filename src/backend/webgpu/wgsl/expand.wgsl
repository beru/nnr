// Expand: broadcast X to a target output shape. For each output index, the
// kernel unflattens through out_dims and maps to an input flat index via
// in_strides. An input axis that's size-1 (or missing) has stride 0, so the
// same input element feeds every output coord along that axis.

struct Meta {
  total         : u32,
  ndim          : u32,
  grid_stride_x : u32,   // threads along x for 2D dispatch split
  _b            : u32,
  out_dims_lo   : vec4<u32>,
  out_dims_hi   : vec4<u32>,
  in_strides_lo : vec4<u32>,
  in_strides_hi : vec4<u32>,
};

@group(0) @binding(0) var<storage, read>       X  : array<f32>;
@group(0) @binding(1) var<storage, read_write> Y  : array<f32>;
@group(0) @binding(2) var<storage, read>       md : Meta;

fn get_out_dim(i : u32) -> u32 { if (i < 4u) { return md.out_dims_lo[i]; } return md.out_dims_hi[i - 4u]; }
fn get_in_stride(i : u32) -> u32 { if (i < 4u) { return md.in_strides_lo[i]; } return md.in_strides_hi[i - 4u]; }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let o = gid.y * md.grid_stride_x + gid.x;
  if (o >= md.total) { return; }
  var x_flat : u32 = 0u;
  var tmp    : u32 = o;
  for (var k : i32 = i32(md.ndim) - 1; k >= 0; k = k - 1) {
    let d = get_out_dim(u32(k));
    let c = tmp % d;
    tmp   = tmp / d;
    x_flat = x_flat + c * get_in_stride(u32(k));
  }
  Y[o] = X[x_flat];
}
