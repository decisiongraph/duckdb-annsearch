#include <metal_stdlib>
#include "GemmParams.h"
using namespace metal;

/// Fused L2 distance + top-k selection.
/// Computes L2 distances and maintains running top-k in one pass,
/// eliminating the nq*nv intermediate distance matrix.
///
/// Memory access: All 32 lanes in a simdgroup cooperatively compute
/// the dot product for ONE vector at a time, each lane handling
/// consecutive dimensions (perfectly coalesced device reads).
/// Prior design had each lane processing a separate vector with
/// stride = d * elem_size, causing < 5% bandwidth utilization.
///
/// Dispatch: grid = (nq, 1), threadgroup = (32, 4) = 128 threads
/// Requires: k <= 32

kernel void fused_l2_topk_min(
    device const float* queries [[buffer(0)]],    // (nq x d)
    device const float* vectors [[buffer(1)]],    // (nv x d), float32
    device const float* query_norms [[buffer(2)]], // (nq,)
    device const float* vec_norms [[buffer(3)]],  // (nv,)
    device float* out_dist [[buffer(4)]],         // (nq x k)
    device int32_t* out_idx [[buffer(5)]],        // (nq x k)
    constant FusedTopkParams& params [[buffer(6)]],
    threadgroup float* shared_query [[threadgroup(0)]],
    threadgroup float* partial_dist [[threadgroup(1)]],
    threadgroup int32_t* partial_idx [[threadgroup(2)]],
    uint q_idx [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {

    if (q_idx >= params.nq) return;

    const uint nv = params.nv;
    const uint dim = params.d;
    const uint k = params.k;

    // Cooperatively load query into shared memory
    device const float* q = queries + q_idx * dim;
    const uint thread_id = simd_group * 32 + lane;
    for (uint i = thread_id; i < dim; i += 128) {
        shared_query[i] = q[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float qnorm = query_norms[q_idx];

    // Each simdgroup scans a non-overlapping range of vectors
    uint chunk_size = (nv + 3) / 4;
    uint chunk_start = simd_group * chunk_size;
    uint chunk_end = min(chunk_start + chunk_size, nv);

    // Warp-select state (lanes 0..k-1 hold sorted top-k)
    float my_dist = INFINITY;
    int32_t my_idx = -1;

    // Process ONE vector at a time. All 32 lanes split the dot product
    // across dimensions → consecutive lanes read consecutive addresses (coalesced).
    for (uint v = chunk_start; v < chunk_end; v++) {
        float partial = 0.0f;
        for (uint j = lane; j < dim; j += 32) {
            partial += shared_query[j] * vectors[v * dim + j];
        }
        float dot = simd_sum(partial);
        float dist = qnorm + vec_norms[v] - 2.0f * dot;

        // Warp select insertion (all lanes see same dist and v → correct)
        float threshold = simd_broadcast(my_dist, k - 1);
        if (dist < threshold) {
            uint pos = 0;
            for (uint j = 0; j < k; j++) {
                if (dist >= simd_broadcast(my_dist, j)) pos++;
            }
            if (pos < k) {
                float old_dist = my_dist;
                int32_t old_idx = my_idx;
                if (lane == pos) {
                    my_dist = dist;
                    my_idx = (int32_t)v;
                } else if (lane > pos && lane < k) {
                    my_dist = simd_shuffle(old_dist, lane - 1);
                    my_idx = simd_shuffle(old_idx, lane - 1);
                }
            }
        }
    }

    // Write partial top-k to shared memory for merge
    partial_dist[simd_group * 32 + lane] = my_dist;
    partial_idx[simd_group * 32 + lane] = my_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Simdgroup 0 merges all 4 partial top-k lists
    if (simd_group == 0) {
        my_dist = INFINITY;
        my_idx = -1;

        // Feed 4*k values through warp_select, one at a time
        for (uint sg = 0; sg < 4; sg++) {
            for (uint l = 0; l < k; l++) {
                float d = partial_dist[sg * 32 + l];
                int32_t idx = partial_idx[sg * 32 + l];
                if (d == INFINITY) continue;

                float threshold = simd_broadcast(my_dist, k - 1);
                if (d < threshold) {
                    uint pos = 0;
                    for (uint j = 0; j < k; j++) {
                        if (d >= simd_broadcast(my_dist, j)) pos++;
                    }
                    if (pos < k) {
                        float old_dist = my_dist;
                        int32_t old_idx = my_idx;
                        if (lane == pos) {
                            my_dist = d;
                            my_idx = idx;
                        } else if (lane > pos && lane < k) {
                            my_dist = simd_shuffle(old_dist, lane - 1);
                            my_idx = simd_shuffle(old_idx, lane - 1);
                        }
                    }
                }
            }
        }

        if (lane < k) {
            out_dist[q_idx * k + lane] = my_dist;
            out_idx[q_idx * k + lane] = my_idx;
        }
    }
}

/// Inner product variant: find k largest dot products
kernel void fused_ip_topk_max(
    device const float* queries [[buffer(0)]],
    device const float* vectors [[buffer(1)]],
    device float* out_dist [[buffer(4)]],
    device int32_t* out_idx [[buffer(5)]],
    constant FusedTopkParams& params [[buffer(6)]],
    threadgroup float* shared_query [[threadgroup(0)]],
    threadgroup float* partial_dist [[threadgroup(1)]],
    threadgroup int32_t* partial_idx [[threadgroup(2)]],
    uint q_idx [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {

    if (q_idx >= params.nq) return;

    const uint nv = params.nv;
    const uint dim = params.d;
    const uint k = params.k;

    device const float* q = queries + q_idx * dim;
    const uint thread_id = simd_group * 32 + lane;
    for (uint i = thread_id; i < dim; i += 128) {
        shared_query[i] = q[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint chunk_size = (nv + 3) / 4;
    uint chunk_start = simd_group * chunk_size;
    uint chunk_end = min(chunk_start + chunk_size, nv);

    float my_dist = -INFINITY;
    int32_t my_idx = -1;

    for (uint v = chunk_start; v < chunk_end; v++) {
        float partial = 0.0f;
        for (uint j = lane; j < dim; j += 32) {
            partial += shared_query[j] * vectors[v * dim + j];
        }
        float dot = simd_sum(partial);

        float threshold = simd_broadcast(my_dist, k - 1);
        if (dot > threshold) {
            uint pos = 0;
            for (uint j = 0; j < k; j++) {
                if (dot <= simd_broadcast(my_dist, j)) pos++;
            }
            if (pos < k) {
                float old_dist = my_dist;
                int32_t old_idx = my_idx;
                if (lane == pos) {
                    my_dist = dot;
                    my_idx = (int32_t)v;
                } else if (lane > pos && lane < k) {
                    my_dist = simd_shuffle(old_dist, lane - 1);
                    my_idx = simd_shuffle(old_idx, lane - 1);
                }
            }
        }
    }

    partial_dist[simd_group * 32 + lane] = my_dist;
    partial_idx[simd_group * 32 + lane] = my_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        my_dist = -INFINITY;
        my_idx = -1;

        for (uint sg = 0; sg < 4; sg++) {
            for (uint l = 0; l < k; l++) {
                float d = partial_dist[sg * 32 + l];
                int32_t idx = partial_idx[sg * 32 + l];
                if (d == -INFINITY) continue;

                float threshold = simd_broadcast(my_dist, k - 1);
                if (d > threshold) {
                    uint pos = 0;
                    for (uint j = 0; j < k; j++) {
                        if (d <= simd_broadcast(my_dist, j)) pos++;
                    }
                    if (pos < k) {
                        float old_dist = my_dist;
                        int32_t old_idx = my_idx;
                        if (lane == pos) {
                            my_dist = d;
                            my_idx = idx;
                        } else if (lane > pos && lane < k) {
                            my_dist = simd_shuffle(old_dist, lane - 1);
                            my_idx = simd_shuffle(old_idx, lane - 1);
                        }
                    }
                }
            }
        }

        if (lane < k) {
            out_dist[q_idx * k + lane] = my_dist;
            out_idx[q_idx * k + lane] = my_idx;
        }
    }
}

// ============================================================================
// FP16 storage variants: vectors stored as half
// ============================================================================

kernel void fused_l2_topk_min_f16storage(
    device const float* queries [[buffer(0)]],
    device const half* vectors [[buffer(1)]],
    device const float* query_norms [[buffer(2)]],
    device const float* vec_norms [[buffer(3)]],
    device float* out_dist [[buffer(4)]],
    device int32_t* out_idx [[buffer(5)]],
    constant FusedTopkParams& params [[buffer(6)]],
    threadgroup float* shared_query [[threadgroup(0)]],
    threadgroup float* partial_dist [[threadgroup(1)]],
    threadgroup int32_t* partial_idx [[threadgroup(2)]],
    uint q_idx [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {

    if (q_idx >= params.nq) return;

    const uint nv = params.nv;
    const uint dim = params.d;
    const uint k = params.k;

    device const float* q = queries + q_idx * dim;
    const uint thread_id = simd_group * 32 + lane;
    for (uint i = thread_id; i < dim; i += 128) {
        shared_query[i] = q[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float qnorm = query_norms[q_idx];

    uint chunk_size = (nv + 3) / 4;
    uint chunk_start = simd_group * chunk_size;
    uint chunk_end = min(chunk_start + chunk_size, nv);

    float my_dist = INFINITY;
    int32_t my_idx = -1;

    for (uint v = chunk_start; v < chunk_end; v++) {
        float partial = 0.0f;
        for (uint j = lane; j < dim; j += 32) {
            partial += float(half(shared_query[j]) * vectors[v * dim + j]);
        }
        float dot = simd_sum(partial);
        float dist = qnorm + vec_norms[v] - 2.0f * dot;

        float threshold = simd_broadcast(my_dist, k - 1);
        if (dist < threshold) {
            uint pos = 0;
            for (uint j = 0; j < k; j++) {
                if (dist >= simd_broadcast(my_dist, j)) pos++;
            }
            if (pos < k) {
                float old_dist = my_dist;
                int32_t old_idx = my_idx;
                if (lane == pos) {
                    my_dist = dist;
                    my_idx = (int32_t)v;
                } else if (lane > pos && lane < k) {
                    my_dist = simd_shuffle(old_dist, lane - 1);
                    my_idx = simd_shuffle(old_idx, lane - 1);
                }
            }
        }
    }

    partial_dist[simd_group * 32 + lane] = my_dist;
    partial_idx[simd_group * 32 + lane] = my_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        my_dist = INFINITY;
        my_idx = -1;

        for (uint sg = 0; sg < 4; sg++) {
            for (uint l = 0; l < k; l++) {
                float d = partial_dist[sg * 32 + l];
                int32_t idx = partial_idx[sg * 32 + l];
                if (d == INFINITY) continue;

                float threshold = simd_broadcast(my_dist, k - 1);
                if (d < threshold) {
                    uint pos = 0;
                    for (uint j = 0; j < k; j++) {
                        if (d >= simd_broadcast(my_dist, j)) pos++;
                    }
                    if (pos < k) {
                        float old_dist = my_dist;
                        int32_t old_idx = my_idx;
                        if (lane == pos) {
                            my_dist = d;
                            my_idx = idx;
                        } else if (lane > pos && lane < k) {
                            my_dist = simd_shuffle(old_dist, lane - 1);
                            my_idx = simd_shuffle(old_idx, lane - 1);
                        }
                    }
                }
            }
        }

        if (lane < k) {
            out_dist[q_idx * k + lane] = my_dist;
            out_idx[q_idx * k + lane] = my_idx;
        }
    }
}

kernel void fused_ip_topk_max_f16storage(
    device const float* queries [[buffer(0)]],
    device const half* vectors [[buffer(1)]],
    device float* out_dist [[buffer(4)]],
    device int32_t* out_idx [[buffer(5)]],
    constant FusedTopkParams& params [[buffer(6)]],
    threadgroup float* shared_query [[threadgroup(0)]],
    threadgroup float* partial_dist [[threadgroup(1)]],
    threadgroup int32_t* partial_idx [[threadgroup(2)]],
    uint q_idx [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {

    if (q_idx >= params.nq) return;

    const uint nv = params.nv;
    const uint dim = params.d;
    const uint k = params.k;

    device const float* q = queries + q_idx * dim;
    const uint thread_id = simd_group * 32 + lane;
    for (uint i = thread_id; i < dim; i += 128) {
        shared_query[i] = q[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint chunk_size = (nv + 3) / 4;
    uint chunk_start = simd_group * chunk_size;
    uint chunk_end = min(chunk_start + chunk_size, nv);

    float my_dist = -INFINITY;
    int32_t my_idx = -1;

    for (uint v = chunk_start; v < chunk_end; v++) {
        float partial = 0.0f;
        for (uint j = lane; j < dim; j += 32) {
            partial += float(half(shared_query[j]) * vectors[v * dim + j]);
        }
        float dot = simd_sum(partial);

        float threshold = simd_broadcast(my_dist, k - 1);
        if (dot > threshold) {
            uint pos = 0;
            for (uint j = 0; j < k; j++) {
                if (dot <= simd_broadcast(my_dist, j)) pos++;
            }
            if (pos < k) {
                float old_dist = my_dist;
                int32_t old_idx = my_idx;
                if (lane == pos) {
                    my_dist = dot;
                    my_idx = (int32_t)v;
                } else if (lane > pos && lane < k) {
                    my_dist = simd_shuffle(old_dist, lane - 1);
                    my_idx = simd_shuffle(old_idx, lane - 1);
                }
            }
        }
    }

    partial_dist[simd_group * 32 + lane] = my_dist;
    partial_idx[simd_group * 32 + lane] = my_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        my_dist = -INFINITY;
        my_idx = -1;

        for (uint sg = 0; sg < 4; sg++) {
            for (uint l = 0; l < k; l++) {
                float d = partial_dist[sg * 32 + l];
                int32_t idx = partial_idx[sg * 32 + l];
                if (d == -INFINITY) continue;

                float threshold = simd_broadcast(my_dist, k - 1);
                if (d > threshold) {
                    uint pos = 0;
                    for (uint j = 0; j < k; j++) {
                        if (d <= simd_broadcast(my_dist, j)) pos++;
                    }
                    if (pos < k) {
                        float old_dist = my_dist;
                        int32_t old_idx = my_idx;
                        if (lane == pos) {
                            my_dist = d;
                            my_idx = idx;
                        } else if (lane > pos && lane < k) {
                            my_dist = simd_shuffle(old_dist, lane - 1);
                            my_idx = simd_shuffle(old_idx, lane - 1);
                        }
                    }
                }
            }
        }

        if (lane < k) {
            out_dist[q_idx * k + lane] = my_dist;
            out_idx[q_idx * k + lane] = my_idx;
        }
    }
}
