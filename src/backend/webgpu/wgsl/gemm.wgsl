// ONNX Gemm: Y = alpha * op(A) * op(B) + beta * C
// where op is optional transpose. A: [M, K] or [K, M] per transA.
//       B: [K, N] or [N, K] per transB. C is optional bias, broadcastable
// to [M, N]. We support C shapes: {} (absent), [N], [M, N].
// (When C is [M, 1] or [1, N] for broader broadcast, falls through to CPU.)

struct Cfg {
  M : u32, N : u32, K : u32, transA : u32,
  transB : u32, bias_kind : u32,   // 0=none, 1=per-col [N], 2=full [M,N]
  _a : u32, _b : u32,
  alpha : f32, beta : f32, _c : u32, _d : u32,
};

@group(0) @binding(0) var<storage, read>       A : array<f32>;
@group(0) @binding(1) var<storage, read>       B : array<f32>;
@group(0) @binding(2) var<storage, read>       C : array<f32>;  // bias (zero-size if kind=0)
@group(0) @binding(3) var<storage, read_write> Y : array<f32>;
@group(0) @binding(4) var<uniform>             cfg : Cfg;

const TILE : u32 = 16u;
var<workgroup> tA : array<array<f32, 16>, 16>;
var<workgroup> tB : array<array<f32, 16>, 16>;

fn load_A(r : u32, c : u32) -> f32 {
  if (r >= cfg.M || c >= cfg.K) { return 0.0; }
  if (cfg.transA == 0u) { return A[r * cfg.K + c]; }
  return A[c * cfg.M + r];
}

fn load_B(r : u32, c : u32) -> f32 {
  if (r >= cfg.K || c >= cfg.N) { return 0.0; }
  if (cfg.transB == 0u) { return B[r * cfg.N + c]; }
  return B[c * cfg.K + r];
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_id)  lid : vec3<u32>) {
  let row = gid.y;
  let col = gid.x;

  var acc : f32 = 0.0;
  let tiles = (cfg.K + TILE - 1u) / TILE;
  for (var t : u32 = 0u; t < tiles; t = t + 1u) {
    tA[lid.y][lid.x] = load_A(row, t * TILE + lid.x);
    tB[lid.y][lid.x] = load_B(t * TILE + lid.y, col);
    workgroupBarrier();
    for (var k : u32 = 0u; k < TILE; k = k + 1u) {
      acc = acc + tA[lid.y][k] * tB[k][lid.x];
    }
    workgroupBarrier();
  }

  if (row < cfg.M && col < cfg.N) {
    var v = cfg.alpha * acc;
    if (cfg.bias_kind == 1u) {
      v = v + cfg.beta * C[col];
    } else if (cfg.bias_kind == 2u) {
      v = v + cfg.beta * C[row * cfg.N + col];
    }
    Y[row * cfg.N + col] = v;
  }
}
