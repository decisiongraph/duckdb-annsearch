#include <metal_stdlib>
using namespace metal;

/// SIMD-group level top-k selection for small k (k <= 32).
/// Each SIMD group processes one query row.
/// Each lane maintains one slot in a sorted top-k buffer.
///
/// Dispatch: threadgroups = nq, threads_per_threadgroup = 32
/// Output is sorted (lane 0 = best, lane k-1 = worst of top-k).

kernel void warp_select_min(
    device const float* distances [[buffer(0)]],   // (nq x nv)
    device float* out_distances [[buffer(1)]],      // (nq x k)
    device int32_t* out_indices [[buffer(2)]],      // (nq x k)
    constant uint& nv [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {

    // Each lane holds one slot: lane 0 = best (smallest), lane k-1 = k-th best
    float my_dist = INFINITY;
    int32_t my_idx = -1;

    device const float* row_data = distances + row * nv;

    for (uint base = 0; base < nv; base += 32) {
        // All 32 lanes read one candidate each (coalesced)
        uint cand_idx = base + lane;
        float d = (cand_idx < nv) ? row_data[cand_idx] : INFINITY;

        // Quick check: skip batch if no candidate beats threshold
        float threshold = simd_broadcast(my_dist, k - 1);
        bool dominated = simd_all(d >= threshold);
        if (dominated) continue;

        // Process each lane's candidate one by one via simd_broadcast
        for (uint l = 0; l < 32 && (base + l) < nv; l++) {
            float cand_d = simd_broadcast(d, l);
            float th = simd_broadcast(my_dist, k - 1);
            if (cand_d < th) {
                // Find insertion point: count how many current top-k values are better
                uint pos = 0;
                for (uint j = 0; j < k; j++) {
                    if (cand_d >= simd_broadcast(my_dist, j)) pos++;
                }

                if (pos < k) {
                    float old_dist = my_dist;
                    int32_t old_idx = my_idx;

                    if (lane == pos) {
                        my_dist = cand_d;
                        my_idx = (int32_t)(base + l);
                    } else if (lane > pos && lane < k) {
                        my_dist = simd_shuffle(old_dist, lane - 1);
                        my_idx = simd_shuffle(old_idx, lane - 1);
                    }
                }
            }
        }
    }

    // Write sorted results
    if (lane < k) {
        out_distances[row * k + lane] = my_dist;
        out_indices[row * k + lane] = my_idx;
    }
}

/// Max variant for inner product (find k largest)
kernel void warp_select_max(
    device const float* distances [[buffer(0)]],
    device float* out_distances [[buffer(1)]],
    device int32_t* out_indices [[buffer(2)]],
    constant uint& nv [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {

    float my_dist = -INFINITY;
    int32_t my_idx = -1;

    device const float* row_data = distances + row * nv;

    for (uint base = 0; base < nv; base += 32) {
        // All 32 lanes read one candidate each (coalesced)
        uint cand_idx = base + lane;
        float d = (cand_idx < nv) ? row_data[cand_idx] : -INFINITY;

        // Quick check: skip batch if no candidate beats threshold
        float threshold = simd_broadcast(my_dist, k - 1);
        bool dominated = simd_all(d <= threshold);
        if (dominated) continue;

        // Process each lane's candidate one by one via simd_broadcast
        for (uint l = 0; l < 32 && (base + l) < nv; l++) {
            float cand_d = simd_broadcast(d, l);
            float th = simd_broadcast(my_dist, k - 1);
            if (cand_d > th) {
                uint pos = 0;
                for (uint j = 0; j < k; j++) {
                    if (cand_d <= simd_broadcast(my_dist, j)) pos++;
                }

                if (pos < k) {
                    float old_dist = my_dist;
                    int32_t old_idx = my_idx;

                    if (lane == pos) {
                        my_dist = cand_d;
                        my_idx = (int32_t)(base + l);
                    } else if (lane > pos && lane < k) {
                        my_dist = simd_shuffle(old_dist, lane - 1);
                        my_idx = simd_shuffle(old_idx, lane - 1);
                    }
                }
            }
        }
    }

    if (lane < k) {
        out_distances[row * k + lane] = my_dist;
        out_indices[row * k + lane] = my_idx;
    }
}
