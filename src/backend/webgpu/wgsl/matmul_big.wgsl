// Large-tile register-tiled matmul. 16×16 workgroup, each thread computes
// a 4×4 sub-tile of outputs → 64×64 output tile per workgroup.
//
// Shared memory per workgroup: A-tile 64×16 + B-tile 16×64 = 2048 f32
// = 8 KB (within the 16 KB WebGPU minimum guarantee).
//
// Compared to matmul_tiled.wgsl (32×32 tile, 2×2 per thread), this kernel
// amortizes shared-memory reads 16× per K-step instead of 4×, with the
// same number of global-memory reads. Wins on large matmuls where the
// K dimension is the bottleneck (Transformer FFN, LLM attention heads
// with D ≥ 128).
//
// Dispatched as (ceil(N/64), ceil(M/64), batch). Matches the batch
// convention of the simple / 32-tile kernels.

struct Dims { M : u32, N : u32, K : u32, batch : u32 };

@group(0) @binding(0) var<storage, read>       A    : array<f32>;
@group(0) @binding(1) var<storage, read>       B    : array<f32>;
@group(0) @binding(2) var<storage, read_write> C    : array<f32>;
@group(0) @binding(3) var<uniform>             dims : Dims;

const WG_X : u32 = 16u;
const WG_Y : u32 = 16u;
const TM : u32 = 4u;
const TN : u32 = 4u;
const TILE_M : u32 = 64u;    // WG_Y * TM
const TILE_N : u32 = 64u;    // WG_X * TN
const TILE_K : u32 = 16u;

var<workgroup> tA : array<array<f32, 16>, 64>;   // [TILE_M][TILE_K]
var<workgroup> tB : array<array<f32, 64>, 16>;   // [TILE_K][TILE_N]

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(workgroup_id)         wid : vec3<u32>,
        @builtin(local_invocation_id)  lid : vec3<u32>) {
  let batch_idx = gid.z;
  let M = dims.M;
  let N = dims.N;
  let K = dims.K;

  let a_base = batch_idx * M * K;
  let b_base = batch_idx * K * N;
  let c_base = batch_idx * M * N;

  // 4×4 register accumulator per thread. Flattened to 16 f32's; the
  // shader treats them positionally as `acc[i*4 + j]` for i,j in [0,4).
  var acc : array<f32, 16>;
  for (var u : u32 = 0u; u < 16u; u = u + 1u) { acc[u] = 0.0; }

  let tid = lid.y * WG_X + lid.x;   // 0..255
  let row_base = wid.y * TILE_M + lid.y * TM;
  let col_base = wid.x * TILE_N + lid.x * TN;

  let n_tiles = (K + TILE_K - 1u) / TILE_K;
  let a_tile_row = wid.y * TILE_M;
  let b_tile_col = wid.x * TILE_N;

  for (var t : u32 = 0u; t < n_tiles; t = t + 1u) {
    // Load A tile: 64×16 = 1024 f32. 256 threads × 4 loads each.
    let k_base = t * TILE_K;
    for (var s : u32 = 0u; s < 4u; s = s + 1u) {
      let ia = s * 256u + tid;        // 0..1023
      let ra = ia / TILE_K;           // 0..63
      let ca = ia % TILE_K;           // 0..15
      let a_r = a_tile_row + ra;
      let a_c = k_base + ca;
      if (a_r < M && a_c < K) {
        tA[ra][ca] = A[a_base + a_r * K + a_c];
      } else {
        tA[ra][ca] = 0.0;
      }

      let ib = s * 256u + tid;        // 0..1023
      let rb = ib / TILE_N;           // 0..15
      let cb = ib % TILE_N;           // 0..63
      let b_r = k_base + rb;
      let b_c = b_tile_col + cb;
      if (b_r < K && b_c < N) {
        tB[rb][cb] = B[b_base + b_r * N + b_c];
      } else {
        tB[rb][cb] = 0.0;
      }
    }
    workgroupBarrier();

    // Inner K loop — each thread fetches its 4 rows of A and 4 cols of B
    // from shared memory, then does 16 FMAs.
    let lrow = lid.y * TM;
    let lcol = lid.x * TN;
    for (var k : u32 = 0u; k < TILE_K; k = k + 1u) {
      let a0 = tA[lrow + 0u][k];
      let a1 = tA[lrow + 1u][k];
      let a2 = tA[lrow + 2u][k];
      let a3 = tA[lrow + 3u][k];
      let b0 = tB[k][lcol + 0u];
      let b1 = tB[k][lcol + 1u];
      let b2 = tB[k][lcol + 2u];
      let b3 = tB[k][lcol + 3u];
      acc[ 0] = acc[ 0] + a0 * b0; acc[ 1] = acc[ 1] + a0 * b1;
      acc[ 2] = acc[ 2] + a0 * b2; acc[ 3] = acc[ 3] + a0 * b3;
      acc[ 4] = acc[ 4] + a1 * b0; acc[ 5] = acc[ 5] + a1 * b1;
      acc[ 6] = acc[ 6] + a1 * b2; acc[ 7] = acc[ 7] + a1 * b3;
      acc[ 8] = acc[ 8] + a2 * b0; acc[ 9] = acc[ 9] + a2 * b1;
      acc[10] = acc[10] + a2 * b2; acc[11] = acc[11] + a2 * b3;
      acc[12] = acc[12] + a3 * b0; acc[13] = acc[13] + a3 * b1;
      acc[14] = acc[14] + a3 * b2; acc[15] = acc[15] + a3 * b3;
    }
    workgroupBarrier();
  }

  // Write back 4×4 sub-tile with bounds checks for edge workgroups.
  for (var i : u32 = 0u; i < TM; i = i + 1u) {
    let r = row_base + i;
    if (r >= M) { continue; }
    for (var j : u32 = 0u; j < TN; j = j + 1u) {
      let c = col_base + j;
      if (c >= N) { continue; }
      C[c_base + r * N + c] = acc[i * 4u + j];
    }
  }
}
