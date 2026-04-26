// DepthToSpace 4D NCHW. For each output element (n, oc, oh, ow), look up
// the source input channel depending on the mode (DCR or CRD). Zero-copy
// gather, same thread-per-output pattern as our other data-movement ops.

struct Cfg {
  total     : u32,
  N         : u32,
  C_in      : u32,
  H         : u32,
  W         : u32,
  C_out     : u32,
  bs        : u32,
  is_crd    : u32,   // 0 = DCR (default), 1 = CRD
};

@group(0) @binding(0) var<storage, read>       X   : array<f32>;
@group(0) @binding(1) var<storage, read_write> Y   : array<f32>;
@group(0) @binding(2) var<uniform>             cfg : Cfg;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= cfg.total) { return; }

  // Unflatten i → (n, oc, oh, ow) with oH = H*bs, oW = W*bs.
  let oH = cfg.H * cfg.bs;
  let oW = cfg.W * cfg.bs;
  let ow = i % oW;
  var t  = i / oW;
  let oh = t % oH;
  t = t / oH;
  let oc = t % cfg.C_out;
  let n  = t / cfg.C_out;

  let h  = oh / cfg.bs;
  let bh = oh % cfg.bs;
  let w  = ow / cfg.bs;
  let bw = ow % cfg.bs;

  var ic : u32;
  if (cfg.is_crd != 0u) {
    // CRD: ic = oc * bs^2 + bh * bs + bw
    ic = oc * cfg.bs * cfg.bs + bh * cfg.bs + bw;
  } else {
    // DCR (default): ic = bh * bs * C_out + bw * C_out + oc
    ic = bh * cfg.bs * cfg.C_out + bw * cfg.C_out + oc;
  }
  let src = ((n * cfg.C_in + ic) * cfg.H + h) * cfg.W + w;
  Y[i] = X[src];
}
