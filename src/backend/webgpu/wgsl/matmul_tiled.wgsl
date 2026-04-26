// Register-tiled matmul. Same semantics as `matmul.wgsl` but each thread
// computes a 2x2 sub-tile of outputs instead of 1. Workgroup is still 16x16
// threads, so the per-workgroup output tile grows to 32x32.
//
// This amortizes shared-memory reads: per K-step each thread now does 4
// FMAs with only 2 A-loads + 2 B-loads from shared memory, vs 1 FMA per
// 1 load each in the simple kernel. Wins roughly scale with the K dim.
//
// Dispatched as (ceil(N/32), ceil(M/32), batch). Batch offset via gid.z
// (same convention as `matmul`).

struct Dims { M : u32, N : u32, K : u32, batch : u32 };

@group(0) @binding(0) var<storage, read>       A    : array<f32>;
@group(0) @binding(1) var<storage, read>       B    : array<f32>;
@group(0) @binding(2) var<storage, read_write> C    : array<f32>;
@group(0) @binding(3) var<uniform>             dims : Dims;

const WG_X : u32 = 16u;
const WG_Y : u32 = 16u;
const TM : u32 = 2u;
const TN : u32 = 2u;
const TILE_M : u32 = 32u;   // WG_Y * TM
const TILE_N : u32 = 32u;   // WG_X * TN
const TILE_K : u32 = 16u;

// 32*16 = 512 floats per tile; 2 tiles = 1024 × 4B = 4KB shared memory.
var<workgroup> tA : array<array<f32, 16>, 32>;   // [TILE_M][TILE_K]
var<workgroup> tB : array<array<f32, 32>, 16>;   // [TILE_K][TILE_N]

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

  let row_base = wid.y * TILE_M + lid.y * TM;
  let col_base = wid.x * TILE_N + lid.x * TN;

  // 2×2 register accumulator per thread.
  var acc00 : f32 = 0.0;
  var acc01 : f32 = 0.0;
  var acc10 : f32 = 0.0;
  var acc11 : f32 = 0.0;

  let tid = lid.y * WG_X + lid.x;  // 0..255
  let n_tiles = (K + TILE_K - 1u) / TILE_K;

  for (var t : u32 = 0u; t < n_tiles; t = t + 1u) {
    // Cooperatively load A (32x16 = 512 floats) and B (16x32 = 512 floats)
    // into shared memory. 256 threads × 2 loads each = 512 per buffer.
    //
    // A[row = ra, col = ca] where ra ∈ [0, TILE_M), ca ∈ [0, TILE_K).
    // Linear slot i ∈ [0, 512); ra = i / TILE_K; ca = i % TILE_K.
    let a_tile_row = wid.y * TILE_M;
    let b_tile_col = wid.x * TILE_N;
    let k_base     = t * TILE_K;

    for (var s : u32 = 0u; s < 2u; s = s + 1u) {
      let ia = s * 256u + tid;                 // 0..511
      let ra = ia / TILE_K;                    // 0..31
      let ca = ia % TILE_K;                    // 0..15
      let a_r = a_tile_row + ra;
      let a_c = k_base + ca;
      if (a_r < M && a_c < K) {
        tA[ra][ca] = A[a_base + a_r * K + a_c];
      } else {
        tA[ra][ca] = 0.0;
      }

      let ib = s * 256u + tid;                 // 0..511
      let rb = ib / TILE_N;                    // 0..15
      let cb = ib % TILE_N;                    // 0..31
      let b_r = k_base + rb;
      let b_c = b_tile_col + cb;
      if (b_r < K && b_c < N) {
        tB[rb][cb] = B[b_base + b_r * N + b_c];
      } else {
        tB[rb][cb] = 0.0;
      }
    }
    workgroupBarrier();

    // Inner K-loop: each thread owns 2 rows of A and 2 cols of B.
    let local_row0 = lid.y * TM;
    let local_row1 = local_row0 + 1u;
    let local_col0 = lid.x * TN;
    let local_col1 = local_col0 + 1u;

    for (var k : u32 = 0u; k < TILE_K; k = k + 1u) {
      let a0 = tA[local_row0][k];
      let a1 = tA[local_row1][k];
      let b0 = tB[k][local_col0];
      let b1 = tB[k][local_col1];
      acc00 = acc00 + a0 * b0;
      acc01 = acc01 + a0 * b1;
      acc10 = acc10 + a1 * b0;
      acc11 = acc11 + a1 * b1;
    }
    workgroupBarrier();
  }

  let r0 = row_base;
  let r1 = row_base + 1u;
  let c0 = col_base;
  let c1 = col_base + 1u;

  if (r0 < M && c0 < N) { C[c_base + r0 * N + c0] = acc00; }
  if (r0 < M && c1 < N) { C[c_base + r0 * N + c1] = acc01; }
  if (r1 < M && c0 < N) { C[c_base + r1 * N + c0] = acc10; }
  if (r1 < M && c1 < N) { C[c_base + r1 * N + c1] = acc11; }
}
