#include <metal_stdlib>
using namespace metal;

/// Half-precision L2 norm: reads float32 input, computes in FP16 for 2x throughput.
/// Uses SIMD reduction. Accumulates in float32 for precision on large dimensions.
/// Available on all Apple Silicon, but Apple8+ (M2+) has 2x FP16 ALU throughput.
///
/// Dispatch: threadgroups = nRows, threads_per_threadgroup = 32

kernel void l2_norm_f16(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {

    // Accumulate in float32 for precision, but load and multiply in half
    float sum = 0.0f;
    for (uint i = lane; i < dim; i += 32) {
        half val = half(input[row * dim + i]);
        sum += float(val * val);
    }

    sum = simd_sum(sum);

    if (lane == 0) {
        output[row] = sum;
    }
}

/// Large-dim variant with multiple SIMD groups per row.
/// Dispatch: threadgroups = nRows, threads_per_threadgroup = min(1024, ceil(dim/32)*32)

kernel void l2_norm_f16_large(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {

    threadgroup float partial_sums[32];

    float sum = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        half val = half(input[row * dim + i]);
        sum += float(val * val);
    }

    sum = simd_sum(sum);

    if (simd_lane == 0) {
        partial_sums[simd_id] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_id == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            output[row] = val;
        }
    }
}

/// Fused L2 distance for small batches: compute ||q - v||^2 directly.
/// More accurate than decomposed (norms + GEMM) for small nv.
/// Each threadgroup handles one (query, vector) pair.
///
/// Dispatch: grid = (nv, nq), threads_per_threadgroup = 32
kernel void l2_distance_direct_f16(
    device const float* queries [[buffer(0)]],   // (nq x d)
    device const float* vectors [[buffer(1)]],   // (nv x d)
    device float* distances [[buffer(2)]],       // (nq x nv)
    constant uint& d [[buffer(3)]],
    constant uint& nv [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {

    uint vec_idx = gid.x;
    uint query_idx = gid.y;

    float sum = 0.0f;
    for (uint i = lane; i < d; i += 32) {
        half diff = half(queries[query_idx * d + i]) - half(vectors[vec_idx * d + i]);
        sum += float(diff * diff);
    }

    sum = simd_sum(sum);

    if (lane == 0) {
        distances[query_idx * nv + vec_idx] = sum;
    }
}

/// Direct L2 with FP16-stored vectors. Queries remain float.
kernel void l2_distance_direct_f16storage(
    device const float* queries [[buffer(0)]],
    device const half* vectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& d [[buffer(3)]],
    constant uint& nv [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {

    uint vec_idx = gid.x;
    uint query_idx = gid.y;

    float sum = 0.0f;
    for (uint i = lane; i < d; i += 32) {
        half diff = half(queries[query_idx * d + i]) - vectors[vec_idx * d + i];
        sum += float(diff * diff);
    }

    sum = simd_sum(sum);

    if (lane == 0) {
        distances[query_idx * nv + vec_idx] = sum;
    }
}
