// float32 matmul C = A * B (row-major).
//
// Two modes, both served by this one shader:
//   - 2D / collapsed-left: A [M, K], B [K, N], C [M, N]. The M dimension may
//     be the flattened product of A's leading batch dims — the host side
//     computes M' = prod(batch) * M when b->ndim == 2 and dispatches batch=1.
//   - per-batch: A [batch, M, K], B [batch, M', K, N], C [batch, M, N].
//     Same-shape batch dims on both inputs (no broadcast). Host dispatches
//     with Z = batch_total and the shader offsets each buffer by batch_idx.
//
// Tiled with 16x16 workgroup + shared memory. Each workgroup covers the C
// tile it writes for its batch slice; each invocation computes one output.

struct Dims { M : u32, N : u32, K : u32, batch : u32 };

@group(0) @binding(0) var<storage, read>       A    : array<f32>;
@group(0) @binding(1) var<storage, read>       B    : array<f32>;
@group(0) @binding(2) var<storage, read_write> C    : array<f32>;
@group(0) @binding(3) var<uniform>             dims : Dims;

const TILE : u32 = 16u;
var<workgroup> tA : array<array<f32, 16>, 16>;
var<workgroup> tB : array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_id)  lid : vec3<u32>) {
  let batch_idx = gid.z;
  let row = gid.y;
  let col = gid.x;
  let M = dims.M;
  let N = dims.N;
  let K = dims.K;

  let a_base = batch_idx * M * K;
  let b_base = batch_idx * K * N;
  let c_base = batch_idx * M * N;

  var acc : f32 = 0.0;
  let tiles = (K + TILE - 1u) / TILE;
  for (var t : u32 = 0u; t < tiles; t = t + 1u) {
    let a_col = t * TILE + lid.x;
    let b_row = t * TILE + lid.y;
    if (row < M && a_col < K) {
      tA[lid.y][lid.x] = A[a_base + row * K + a_col];
    } else {
      tA[lid.y][lid.x] = 0.0;
    }
    if (b_row < K && col < N) {
      tB[lid.y][lid.x] = B[b_base + b_row * N + col];
    } else {
      tB[lid.y][lid.x] = 0.0;
    }
    workgroupBarrier();

    for (var k : u32 = 0u; k < TILE; k = k + 1u) {
      acc = acc + tA[lid.y][k] * tB[k][lid.x];
    }
    workgroupBarrier();
  }

  if (row < M && col < N) {
    C[c_base + row * N + col] = acc;
  }
}
