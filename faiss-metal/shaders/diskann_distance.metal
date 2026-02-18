#include <metal_stdlib>
#include "DiskannDistanceParams.h"
using namespace metal;

/// Maximum dimension supported for threadgroup query preload.
/// 1536 floats = 6KB threadgroup memory — well within 32KB budget.
constant constexpr uint MAX_PRELOAD_DIM = 1536;

/// Batch L2 squared distance: one query vs N candidates.
/// Each threadgroup (32 threads = 1 simdgroup) computes distance for one candidate.
/// Query vector is loaded into threadgroup memory once, reducing device memory reads.
///
/// Dispatch: grid = (N, 1, 1), threadgroup = (32, 1, 1)

kernel void diskann_batch_l2(
    device const float* query [[buffer(0)]],       // (dim,)
    device const float* candidates [[buffer(1)]],  // (n * dim,) contiguous
    device float* out_distances [[buffer(2)]],     // (n,)
    constant DiskannDistParams& params [[buffer(3)]],
    uint candidate_idx [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {

    if (candidate_idx >= params.n) return;

    const uint dim = params.dim;

    // Preload query into threadgroup memory (shared across all candidates in this group)
    threadgroup float shared_query[MAX_PRELOAD_DIM];
    if (dim <= MAX_PRELOAD_DIM) {
        for (uint j = lane; j < dim; j += 32) {
            shared_query[j] = query[j];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    device const float* cand = candidates + candidate_idx * dim;

    // All 32 lanes split dimensions: consecutive lanes read consecutive addresses
    float partial = 0.0f;
    if (dim <= MAX_PRELOAD_DIM) {
        for (uint j = lane; j < dim; j += 32) {
            float diff = shared_query[j] - cand[j];
            partial += diff * diff;
        }
    } else {
        // Fallback for very high dimensions: read query from device memory
        for (uint j = lane; j < dim; j += 32) {
            float diff = query[j] - cand[j];
            partial += diff * diff;
        }
    }

    float dist = simd_sum(partial);

    if (lane == 0) {
        out_distances[candidate_idx] = dist;
    }
}

/// Multi-query batch L2 squared distance: Q queries, each with their own candidates.
/// query_map[i] gives the query index for candidate i.
/// Dispatch: grid = (total_n, 1, 1), threadgroup = (32, 1, 1)

kernel void diskann_multi_batch_l2(
    device const float* queries [[buffer(0)]],     // (nq * dim,) all query vectors
    device const float* candidates [[buffer(1)]],  // (total_n * dim,) contiguous
    device const uint* query_map [[buffer(2)]],    // (total_n,) query index per candidate
    device float* out_distances [[buffer(3)]],     // (total_n,)
    constant DiskannDistParams& params [[buffer(4)]],
    uint candidate_idx [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {

    if (candidate_idx >= params.n) return;

    const uint dim = params.dim;
    const uint qi = query_map[candidate_idx];
    device const float* q = queries + qi * dim;
    device const float* cand = candidates + candidate_idx * dim;

    // Preload query into threadgroup memory
    threadgroup float shared_query[MAX_PRELOAD_DIM];
    if (dim <= MAX_PRELOAD_DIM) {
        for (uint j = lane; j < dim; j += 32) {
            shared_query[j] = q[j];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    float partial = 0.0f;
    if (dim <= MAX_PRELOAD_DIM) {
        for (uint j = lane; j < dim; j += 32) {
            float diff = shared_query[j] - cand[j];
            partial += diff * diff;
        }
    } else {
        for (uint j = lane; j < dim; j += 32) {
            float diff = q[j] - cand[j];
            partial += diff * diff;
        }
    }

    float dist = simd_sum(partial);
    if (lane == 0) {
        out_distances[candidate_idx] = dist;
    }
}

/// Multi-query batch inner product distance.
/// Returns negated dot product (lower = more similar).

kernel void diskann_multi_batch_ip(
    device const float* queries [[buffer(0)]],
    device const float* candidates [[buffer(1)]],
    device const uint* query_map [[buffer(2)]],
    device float* out_distances [[buffer(3)]],
    constant DiskannDistParams& params [[buffer(4)]],
    uint candidate_idx [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {

    if (candidate_idx >= params.n) return;

    const uint dim = params.dim;
    const uint qi = query_map[candidate_idx];
    device const float* q = queries + qi * dim;
    device const float* cand = candidates + candidate_idx * dim;

    threadgroup float shared_query[MAX_PRELOAD_DIM];
    if (dim <= MAX_PRELOAD_DIM) {
        for (uint j = lane; j < dim; j += 32) {
            shared_query[j] = q[j];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    float partial = 0.0f;
    if (dim <= MAX_PRELOAD_DIM) {
        for (uint j = lane; j < dim; j += 32) {
            partial += shared_query[j] * cand[j];
        }
    } else {
        for (uint j = lane; j < dim; j += 32) {
            partial += q[j] * cand[j];
        }
    }

    float dot = simd_sum(partial);
    if (lane == 0) {
        out_distances[candidate_idx] = -dot;
    }
}

/// Batch inner product distance: one query vs N candidates.
/// Returns negated dot product (lower = more similar).

kernel void diskann_batch_ip(
    device const float* query [[buffer(0)]],
    device const float* candidates [[buffer(1)]],
    device float* out_distances [[buffer(2)]],
    constant DiskannDistParams& params [[buffer(3)]],
    uint candidate_idx [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {

    if (candidate_idx >= params.n) return;

    const uint dim = params.dim;

    // Preload query into threadgroup memory
    threadgroup float shared_query[MAX_PRELOAD_DIM];
    if (dim <= MAX_PRELOAD_DIM) {
        for (uint j = lane; j < dim; j += 32) {
            shared_query[j] = query[j];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    device const float* cand = candidates + candidate_idx * dim;

    float partial = 0.0f;
    if (dim <= MAX_PRELOAD_DIM) {
        for (uint j = lane; j < dim; j += 32) {
            partial += shared_query[j] * cand[j];
        }
    } else {
        for (uint j = lane; j < dim; j += 32) {
            partial += query[j] * cand[j];
        }
    }

    float dot = simd_sum(partial);

    if (lane == 0) {
        out_distances[candidate_idx] = -dot;
    }
}
