#include <metal_stdlib>
using namespace metal;

/// Compute ||v||^2 for each row of an (nRows x dim) matrix.
/// Output: float[nRows] with squared L2 norms.
/// Dispatch: threadgroups = nRows, threads_per_threadgroup = 32 (one SIMD group per row)
kernel void l2_norm(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {

    float sum = 0.0f;
    for (uint i = lane; i < dim; i += 32) {
        float val = input[row * dim + i];
        sum += val * val;
    }

    // Reduce across SIMD group (32 lanes)
    sum = simd_sum(sum);

    if (lane == 0) {
        output[row] = sum;
    }
}

/// Variant for large dimensions: uses multiple SIMD groups per row.
/// Dispatch: threadgroups = nRows, threads_per_threadgroup = ceil(dim/32)*32 (capped at 1024)
kernel void l2_norm_large(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {

    threadgroup float partial_sums[32]; // max 32 SIMD groups per threadgroup

    float sum = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = input[row * dim + i];
        sum += val * val;
    }

    // Reduce within SIMD group
    sum = simd_sum(sum);

    // Write SIMD group result
    if (simd_lane == 0) {
        partial_sums[simd_id] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by first SIMD group
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_id == 0) {
        float val = (simd_lane < num_simd_groups) ? partial_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            output[row] = val;
        }
    }
}
