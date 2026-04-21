// Unified 1D / 2D / 3D NCHW ConvTranspose. One output element per thread,
// gather form. The host collapses rank-3 (1D) and rank-5 (3D) inputs to
// this rank-5 internal shape by setting unused dims to 1 in the meta.
//
// For each output (n, m, od, oh, ow) we sweep the kernel window (kd, kh, kw)
// and the group-local input channels. For a given (oh, kh), the input row
// that contributes is ih = (oh + pad_top - kh*dilation_h) / stride_h —
// only if the numerator is non-negative, divisible by stride_h, and the
// resulting ih is in [0, H). Same for depth and width.
//
// Weights are laid out ONNX-style for ConvTranspose:
// W[c_in, m_per_g, kD, kH, kW] (kD=1 for 2D, kD=kH=1 for 1D).

struct Meta {
  total       : u32,
  N           : u32, C_out : u32,
  D_out       : u32, H_out : u32, W_out : u32,
  C_in        : u32, groups : u32,
  kD          : u32, kH    : u32, kW : u32,
  stride_d    : u32, stride_h : u32, stride_w : u32,
  pad_front   : u32, pad_top  : u32, pad_left : u32,
  dilation_d  : u32, dilation_h : u32, dilation_w : u32,
  D           : u32, H : u32, W : u32,
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

  // Unflatten i → (n, m, od, oh, ow).
  let ow = i % md.W_out;
  var tmp = i / md.W_out;
  let oh = tmp % md.H_out;
  tmp = tmp / md.H_out;
  let od = tmp % md.D_out;
  tmp = tmp / md.D_out;
  let m = tmp % md.C_out;
  let n = tmp / md.C_out;

  let M_per_group    = md.C_out / md.groups;
  let C_in_per_group = md.C_in  / md.groups;
  let g = m / M_per_group;
  let m_in_g = m % M_per_group;

  let x_batch_stride = md.C_in * md.D * md.H * md.W;
  let x_chan_stride  = md.D * md.H * md.W;
  // W layout: [C_in, M_per_group, kD, kH, kW]
  let w_inch_stride  = M_per_group * md.kD * md.kH * md.kW;
  let w_outch_stride = md.kD * md.kH * md.kW;

  var acc : f32 = 0.0;
  let c_start = g * C_in_per_group;

  for (var kd : u32 = 0u; kd < md.kD; kd = kd + 1u) {
    let num_d = i32(od) + i32(md.pad_front) - i32(kd * md.dilation_d);
    if (num_d < 0) { continue; }
    let un_d = u32(num_d);
    if ((un_d % md.stride_d) != 0u) { continue; }
    let id = un_d / md.stride_d;
    if (id >= md.D) { continue; }

    for (var kh : u32 = 0u; kh < md.kH; kh = kh + 1u) {
      let num_h = i32(oh) + i32(md.pad_top) - i32(kh * md.dilation_h);
      if (num_h < 0) { continue; }
      let un_h = u32(num_h);
      if ((un_h % md.stride_h) != 0u) { continue; }
      let ih = un_h / md.stride_h;
      if (ih >= md.H) { continue; }

      for (var kw : u32 = 0u; kw < md.kW; kw = kw + 1u) {
        let num_w = i32(ow) + i32(md.pad_left) - i32(kw * md.dilation_w);
        if (num_w < 0) { continue; }
        let un_w = u32(num_w);
        if ((un_w % md.stride_w) != 0u) { continue; }
        let iw = un_w / md.stride_w;
        if (iw >= md.W) { continue; }

        let x_dhw_base = (id * md.H + ih) * md.W + iw;
        let w_dhw_base = (kd * md.kH + kh) * md.kW + kw;
        for (var ic : u32 = 0u; ic < C_in_per_group; ic = ic + 1u) {
          let c = c_start + ic;
          let x_flat = n * x_batch_stride + c * x_chan_stride + x_dhw_base;
          let w_flat = c * w_inch_stride + m_in_g * w_outch_stride + w_dhw_base;
          acc = acc + X[x_flat] * Wt[w_flat];
        }
      }
    }
  }
  if (md.has_bias != 0u) { acc = acc + B[m]; }
  Y[i] = acc;
}
