// Direct 2D NCHW convolution. One output element per thread. Inner loops
// iterate over this thread's group-local input channels and the kernel
// window, accumulating into a single f32. Pad regions are handled by
// range-checking the (ih, iw) indices rather than materializing padded
// input — keeps this op pointwise with zero extra memory.

struct Meta {
  total       : u32,
  N           : u32, C_out : u32, H_out : u32, W_out : u32,
  C_in        : u32, groups : u32,
  kH          : u32, kW    : u32,
  stride_h    : u32, stride_w : u32,
  pad_top     : u32, pad_left : u32,
  dilation_h  : u32, dilation_w : u32,
  H           : u32, W : u32,
  has_bias    : u32,
};

@group(0) @binding(0) var<storage, read>       X  : array<f32>;
@group(0) @binding(1) var<storage, read>       Wt : array<f32>;
@group(0) @binding(2) var<storage, read>       B  : array<f32>;
@group(0) @binding(3) var<storage, read_write> Y  : array<f32>;
@group(0) @binding(4) var<storage, read>       md : Meta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= md.total) { return; }

  // Unflatten i → (n, m, oh, ow).
  let ow = i % md.W_out;
  var tmp = i / md.W_out;
  let oh = tmp % md.H_out;
  tmp = tmp / md.H_out;
  let m = tmp % md.C_out;
  let n = tmp / md.C_out;

  let M_per_group    = md.C_out / md.groups;
  let C_in_per_group = md.C_in  / md.groups;
  let g = m / M_per_group;

  let x_batch_stride  = md.C_in * md.H * md.W;
  let x_chan_stride   = md.H * md.W;
  let w_outch_stride  = C_in_per_group * md.kH * md.kW;
  let w_inch_stride   = md.kH * md.kW;

  let base_x = n * x_batch_stride;
  let base_w = m * w_outch_stride;

  var acc : f32 = 0.0;
  for (var ic : u32 = 0u; ic < C_in_per_group; ic = ic + 1u) {
    let c = g * C_in_per_group + ic;
    let x_c_base = base_x + c * x_chan_stride;
    let w_c_base = base_w + ic * w_inch_stride;
    for (var kh : u32 = 0u; kh < md.kH; kh = kh + 1u) {
      let ih_i = i32(oh * md.stride_h) + i32(kh * md.dilation_h) - i32(md.pad_top);
      if (ih_i < 0 || ih_i >= i32(md.H)) { continue; }
      let ih = u32(ih_i);
      let x_h_base = x_c_base + ih * md.W;
      let w_h_base = w_c_base + kh * md.kW;
      for (var kw : u32 = 0u; kw < md.kW; kw = kw + 1u) {
        let iw_i = i32(ow * md.stride_w) + i32(kw * md.dilation_w) - i32(md.pad_left);
        if (iw_i < 0 || iw_i >= i32(md.W)) { continue; }
        let iw = u32(iw_i);
        acc = acc + X[x_h_base + iw] * Wt[w_h_base + kw];
      }
    }
  }
  if (md.has_bias != 0u) { acc = acc + B[m]; }
  Y[i] = acc;
}
