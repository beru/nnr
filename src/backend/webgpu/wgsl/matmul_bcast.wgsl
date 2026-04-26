// Batched matmul with broadcast across batch dims.
//   A[a_b..., M, K] @ B[b_b..., K, N] -> C[out_b..., M, N]
// where out_b is the NumPy-broadcast of a_b and b_b.
//
// Dispatched as (ceil(N/16), ceil(M/16), batch_total). For each z slot the
// shader unflattens batch_idx into out_coords[nb], then dots them with
// per-side stride tables from the meta buffer to recover the actual A/B
// batch offsets. Broadcast size-1 axes contribute stride 0 (the same
// slice replicates across that output axis). Up to 8 batch axes.

struct Meta {
  M          : u32,
  N          : u32,
  K          : u32,
  nbatch     : u32,
  out_dims   : array<vec4<u32>, 2>,  // up to 8 axes of the broadcast batch shape
  a_strides  : array<vec4<u32>, 2>,  // per-axis element stride in A; 0 = broadcast
  b_strides  : array<vec4<u32>, 2>,  // per-axis element stride in B; 0 = broadcast
};

@group(0) @binding(0) var<storage, read>       A  : array<f32>;
@group(0) @binding(1) var<storage, read>       B  : array<f32>;
@group(0) @binding(2) var<storage, read_write> C  : array<f32>;
@group(0) @binding(3) var<uniform>             md : Meta;

const TILE : u32 = 16u;
var<workgroup> tA : array<array<f32, 16>, 16>;
var<workgroup> tB : array<array<f32, 16>, 16>;

fn out_dim(i : u32)   -> u32 { if (i < 4u) { return md.out_dims[0][i]; }  return md.out_dims[1][i - 4u];  }
fn a_stride(i : u32)  -> u32 { if (i < 4u) { return md.a_strides[0][i]; } return md.a_strides[1][i - 4u]; }
fn b_stride(i : u32)  -> u32 { if (i < 4u) { return md.b_strides[0][i]; } return md.b_strides[1][i - 4u]; }

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_id)  lid : vec3<u32>) {
  let batch_idx = gid.z;
  let row = gid.y;
  let col = gid.x;
  let M = md.M;
  let N = md.N;
  let K = md.K;

  // Unflatten batch_idx against out_dims. Walk axes from the last to the
  // first; extract coord[d] = rem % out_dim[d]; rem /= out_dim[d].
  var rem : u32 = batch_idx;
  var a_off : u32 = 0u;
  var b_off : u32 = 0u;
  for (var d : u32 = md.nbatch; d > 0u; d = d - 1u) {
    let k = d - 1u;
    let w = out_dim(k);
    let c = rem % w;
    rem = rem / w;
    a_off = a_off + c * a_stride(k);
    b_off = b_off + c * b_stride(k);
  }
  let c_base = batch_idx * M * N;

  var acc : f32 = 0.0;
  let tiles = (K + TILE - 1u) / TILE;
  for (var t : u32 = 0u; t < tiles; t = t + 1u) {
    let a_col = t * TILE + lid.x;
    let b_row = t * TILE + lid.y;
    if (row < M && a_col < K) {
      tA[lid.y][lid.x] = A[a_off + row * K + a_col];
    } else {
      tA[lid.y][lid.x] = 0.0;
    }
    if (b_row < K && col < N) {
      tB[lid.y][lid.x] = B[b_off + b_row * N + col];
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
