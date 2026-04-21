// SpaceToDepth 4D NCHW. Inverse of DepthToSpace (DCR mode).
//
// Forward relation (equivalent form, for each output):
//   output[n, oc, oh, ow] = input[n, c, oh*bs + bh, ow*bs + bw]
//   with  oc = bh * bs * C + bw * C + c  (DCR layout)
// Inverting oc: c = oc % C; bw = (oc / C) % bs; bh = (oc / C) / bs.

struct Cfg {
  total : u32,
  N     : u32,
  C_in  : u32,
  H     : u32,
  W     : u32,
  C_out : u32,
  bs    : u32,
  _pad  : u32,
};

@group(0) @binding(0) var<storage, read>       X   : array<f32>;
@group(0) @binding(1) var<storage, read_write> Y   : array<f32>;
@group(0) @binding(2) var<uniform>             cfg : Cfg;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= cfg.total) { return; }

  let oH = cfg.H / cfg.bs;
  let oW = cfg.W / cfg.bs;
  let ow = i % oW;
  var t  = i / oW;
  let oh = t % oH;
  t = t / oH;
  let oc = t % cfg.C_out;
  let n  = t / cfg.C_out;

  let c  = oc % cfg.C_in;
  let bw = (oc / cfg.C_in) % cfg.bs;
  let bh = (oc / cfg.C_in) / cfg.bs;
  let ih = oh * cfg.bs + bh;
  let iw = ow * cfg.bs + bw;
  let src = ((n * cfg.C_in + c) * cfg.H + ih) * cfg.W + iw;
  Y[i] = X[src];
}
