// Pad (constant mode only). For each output element, convert its coord to
// the input coord by subtracting pad_starts; if any axis's input coord is
// out of range, write pad_value. Otherwise, compute the input flat index
// via natural row-major strides and copy the input element.

struct Meta {
  total         : u32,
  ndim          : u32,
  pad_value     : u32,  // f32 bits
  grid_stride_x : u32,  // threads along x when 2D-splitting the dispatch
  out_dims_lo   : vec4<u32>,
  out_dims_hi   : vec4<u32>,
  in_dims_lo    : vec4<u32>,
  in_dims_hi    : vec4<u32>,
  in_strides_lo : vec4<u32>,
  in_strides_hi : vec4<u32>,
  pad_starts_lo : vec4<u32>,
  pad_starts_hi : vec4<u32>,
};

@group(0) @binding(0) var<storage, read>       X  : array<f32>;
@group(0) @binding(1) var<storage, read_write> Y  : array<f32>;
@group(0) @binding(2) var<storage, read>       md : Meta;

fn get_out_dim(i : u32)   -> u32 { if (i < 4u) { return md.out_dims_lo[i]; }   return md.out_dims_hi[i - 4u]; }
fn get_in_dim(i : u32)    -> u32 { if (i < 4u) { return md.in_dims_lo[i]; }    return md.in_dims_hi[i - 4u]; }
fn get_in_stride(i : u32) -> u32 { if (i < 4u) { return md.in_strides_lo[i]; } return md.in_strides_hi[i - 4u]; }
fn get_pad_start(i : u32) -> u32 { if (i < 4u) { return md.pad_starts_lo[i]; } return md.pad_starts_hi[i - 4u]; }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let o = gid.y * md.grid_stride_x + gid.x;
  if (o >= md.total) { return; }

  var x_flat : u32 = 0u;
  var tmp    : u32 = o;
  var oob    : bool = false;
  for (var k : i32 = i32(md.ndim) - 1; k >= 0; k = k - 1) {
    let d_out = get_out_dim(u32(k));
    let c_out = tmp % d_out;
    tmp       = tmp / d_out;
    let start = get_pad_start(u32(k));
    // c_in may underflow u32 → becomes a huge value; that still fails the
    // `>= d_in` check so we flag out-of-bounds either way.
    let c_in = c_out - start;
    let d_in = get_in_dim(u32(k));
    if (c_out < start || c_in >= d_in) { oob = true; }
    x_flat = x_flat + c_in * get_in_stride(u32(k));
  }

  if (oob) {
    Y[o] = bitcast<f32>(md.pad_value);
  } else {
    Y[o] = X[x_flat];
  }
}
