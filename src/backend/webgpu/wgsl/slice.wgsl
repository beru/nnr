// Slice (positive steps only). The host precomputes:
//   base        = sum(start[k] * in_stride[k])
//   eff_strides = step[k] * in_stride[k]
// so the kernel has no per-axis arithmetic beyond the unflatten+dot-product.
// Input coord[k] for output coord o[k] is start[k] + o[k] * step[k]; under
// the precomputations this simplifies to base + sum(o[k] * eff_strides[k]).

struct Meta {
  total          : u32,
  ndim           : u32,
  base           : u32,
  grid_stride_x  : u32,   // threads along x for 2D dispatch split
  out_dims_lo    : vec4<u32>,
  out_dims_hi    : vec4<u32>,
  eff_strides_lo : vec4<u32>,
  eff_strides_hi : vec4<u32>,
};

@group(0) @binding(0) var<storage, read>       X  : array<f32>;
@group(0) @binding(1) var<storage, read_write> Y  : array<f32>;
@group(0) @binding(2) var<storage, read>       md : Meta;

fn get_out_dim(i : u32) -> u32 { if (i < 4u) { return md.out_dims_lo[i]; } return md.out_dims_hi[i - 4u]; }
fn get_eff_stride(i : u32) -> u32 { if (i < 4u) { return md.eff_strides_lo[i]; } return md.eff_strides_hi[i - 4u]; }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let o = gid.y * md.grid_stride_x + gid.x;
  if (o >= md.total) { return; }
  var x_flat : u32 = md.base;
  var tmp    : u32 = o;
  for (var k : i32 = i32(md.ndim) - 1; k >= 0; k = k - 1) {
    let d = get_out_dim(u32(k));
    let c = tmp % d;
    tmp   = tmp / d;
    x_flat = x_flat + c * get_eff_stride(u32(k));
  }
  Y[o] = X[x_flat];
}
