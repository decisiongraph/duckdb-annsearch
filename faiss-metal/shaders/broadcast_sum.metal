#include <metal_stdlib>
using namespace metal;

/// Add precomputed norms to distance matrix for L2 distance computation.
///
/// L2(q, v) = ||q||^2 - 2*q·v + ||v||^2
///
/// Input dist matrix has -2*q·v (from GEMM), shape (nq x nv).
/// query_norms: ||q_i||^2 for each query (length nq)
/// vec_norms: ||v_j||^2 for each database vector (length nv)
///
/// Output: dist[i][j] += query_norms[i] + vec_norms[j]
///
/// Dispatch as 2D grid: (nv, nq)
kernel void broadcast_sum_l2(
    device float* dist [[buffer(0)]],
    device const float* query_norms [[buffer(1)]],
    device const float* vec_norms [[buffer(2)]],
    constant uint& nv [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {

    uint col = gid.x; // vector index
    uint row = gid.y; // query index

    dist[row * nv + col] += query_norms[row] + vec_norms[col];
}
