// 2D NCHW Conv — register-tiled variant with shared-memory weight cache.
//
// Dispatch grid: (ceil(W_out/8), ceil(H_out/8), N*C_out). Each workgroup
// computes an 8×8 tile of the (oh, ow) plane for a single (n, m) pair.
// Since all 64 threads need the same weight tile W[m, 0..C_in_per_group,
// 0..kH, 0..kW], they cooperatively load it into shared memory once,
// then reuse it across the whole tile. This cuts weight reads by ~64×
// (every weight read becomes a shared-memory access instead of global).
//
// Shared-memory budget: MAX_WEIGHTS = 4096 f32 = 16 KB, the WebGPU
// minimum guaranteed workgroup-storage size. Host code guards with
// `C_in_per_group * kH * kW ≤ MAX_WEIGHTS` and falls back to the naive
// kernel when that doesn't hold.
//
// Same Meta layout as conv.wgsl so the bind group layout is identical
// and the two pipelines share the uniforms buffer.

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

const WG_X : u32 = 8u;
const WG_Y : u32 = 8u;
const WG_SIZE : u32 = 64u;
const MAX_WEIGHTS : u32 = 4096u;

var<workgroup> sh_w : array<f32, 4096>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id)         wid : vec3<u32>,
        @builtin(local_invocation_id)  lid : vec3<u32>) {
  let nm = wid.z;
  let n  = nm / md.C_out;
  let m  = nm % md.C_out;

  let M_per_group    = md.C_out / md.groups;
  let C_in_per_group = md.C_in  / md.groups;
  let g              = m / M_per_group;

  let w_size = C_in_per_group * md.kH * md.kW;

  // Cooperatively load W[m, :, :, :] into shared memory. 64 threads share
  // the work; each thread handles `ceil(w_size / 64)` elements.
  let w_base = m * w_size;
  let tid = lid.y * WG_X + lid.x;
  for (var i : u32 = tid; i < w_size; i = i + WG_SIZE) {
    sh_w[i] = Wt[w_base + i];
  }
  workgroupBarrier();

  let oh = wid.y * WG_Y + lid.y;
  let ow = wid.x * WG_X + lid.x;
  // Out-of-range threads still needed to finish the cooperative load above;
  // they just skip the compute and output write here.
  if (oh >= md.H_out || ow >= md.W_out) { return; }

  let c_start         = g * C_in_per_group;
  let x_batch_stride  = md.C_in * md.H * md.W;
  let x_chan_stride   = md.H * md.W;
  let base_x          = n * x_batch_stride;

  var acc : f32 = 0.0;
  for (var ic : u32 = 0u; ic < C_in_per_group; ic = ic + 1u) {
    let c = c_start + ic;
    let x_c_base = base_x + c * x_chan_stride;
    let w_c_base = ic * md.kH * md.kW;   // index into sh_w
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
        acc = acc + X[x_h_base + iw] * sh_w[w_h_base + kw];
      }
    }
  }
  if (md.has_bias != 0u) { acc = acc + B[m]; }
  let y_flat = ((n * md.C_out + m) * md.H_out + oh) * md.W_out + ow;
  Y[y_flat] = acc;
}
