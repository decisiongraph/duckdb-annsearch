#include <metal_stdlib>
using namespace metal;

/// Block-level top-k selection for k > 32.
/// Each threadgroup processes one query row.
///
/// Phase 1: Each thread scans a stripe, keeps LOCAL_K best in a sorted buffer.
/// Phase 2: Bitonic merge sort across all threads in shared memory.
/// Phase 3: Cooperative parallel output of top-k results.
///
/// Dispatch: threadgroups = nq, threads_per_threadgroup = 256 (or 512 on M3+)
/// k must be <= LOCAL_K * threads_per_threadgroup

#define LOCAL_K 8

// Insertion sort a small buffer (ascending for min)
inline void sort_local_asc(thread float* dist, thread int32_t* idx, uint count) {
    for (uint i = 1; i < count; i++) {
        float d = dist[i];
        int32_t ix = idx[i];
        uint j = i;
        while (j > 0 && dist[j - 1] > d) {
            dist[j] = dist[j - 1];
            idx[j] = idx[j - 1];
            j--;
        }
        dist[j] = d;
        idx[j] = ix;
    }
}

inline void sort_local_desc(thread float* dist, thread int32_t* idx, uint count) {
    for (uint i = 1; i < count; i++) {
        float d = dist[i];
        int32_t ix = idx[i];
        uint j = i;
        while (j > 0 && dist[j - 1] < d) {
            dist[j] = dist[j - 1];
            idx[j] = idx[j - 1];
            j--;
        }
        dist[j] = d;
        idx[j] = ix;
    }
}

/// Compare-and-swap helper for bitonic sort (shared memory).
/// Swaps elements at positions i and j if they are out of order
/// according to the ascending flag.
inline void bitonic_cas(
    threadgroup float* sd,
    threadgroup int32_t* si,
    uint i, uint j, bool ascending)
{
    float di = sd[i];
    float dj = sd[j];
    if ((ascending && di > dj) || (!ascending && di < dj)) {
        sd[i] = dj;
        sd[j] = di;
        int32_t ti = si[i];
        si[i] = si[j];
        si[j] = ti;
    }
}

kernel void block_select_min(
    device const float* distances [[buffer(0)]],   // (nq x nv)
    device float* out_distances [[buffer(1)]],      // (nq x k)
    device int32_t* out_indices [[buffer(2)]],      // (nq x k)
    constant uint& nv [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    threadgroup float* shared_dist [[threadgroup(0)]],
    threadgroup int32_t* shared_idx [[threadgroup(1)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {

    float local_dist[LOCAL_K];
    int32_t local_idx[LOCAL_K];
    uint local_count = 0;
    float local_max = INFINITY;

    device const float* row_data = distances + row * nv;

    // Phase 1: Each thread scans its stripe, keeping LOCAL_K best
    for (uint i = tid; i < nv; i += tg_size) {
        float d = row_data[i];
        if (local_count < LOCAL_K) {
            local_dist[local_count] = d;
            local_idx[local_count] = (int32_t)i;
            local_count++;
            if (local_count == LOCAL_K) {
                local_max = local_dist[0];
                for (uint j = 1; j < LOCAL_K; j++)
                    local_max = max(local_max, local_dist[j]);
            }
        } else if (d < local_max) {
            uint worst = 0;
            for (uint j = 1; j < LOCAL_K; j++) {
                if (local_dist[j] > local_dist[worst]) worst = j;
            }
            local_dist[worst] = d;
            local_idx[worst] = (int32_t)i;
            local_max = local_dist[0];
            for (uint j = 1; j < LOCAL_K; j++)
                local_max = max(local_max, local_dist[j]);
        }
    }

    sort_local_asc(local_dist, local_idx, local_count);

    uint base = tid * LOCAL_K;
    for (uint i = 0; i < local_count; i++) {
        shared_dist[base + i] = local_dist[i];
        shared_idx[base + i] = local_idx[i];
    }
    for (uint i = local_count; i < LOCAL_K; i++) {
        shared_dist[base + i] = INFINITY;
        shared_idx[base + i] = -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Bitonic merge sort across all threads in shared memory.
    // After Phase 1, each thread's LOCAL_K block is sorted ascending.
    // Reverse odd-numbered blocks to create alternating asc/desc pattern.
    uint N = tg_size * LOCAL_K;
    if (tid & 1) {
        for (uint i = 0; i < LOCAL_K / 2; i++) {
            uint a = base + i;
            uint b = base + LOCAL_K - 1 - i;
            float td_val = shared_dist[a]; shared_dist[a] = shared_dist[b]; shared_dist[b] = td_val;
            int32_t ti_val = shared_idx[a]; shared_idx[a] = shared_idx[b]; shared_idx[b] = ti_val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic merge network: ascending overall (min-k)
    for (uint block_size = LOCAL_K * 2; block_size <= N; block_size *= 2) {
        for (uint step = block_size / 2; step >= 1; step /= 2) {
            for (uint loc = 0; loc < LOCAL_K; loc++) {
                uint i = base + loc;
                uint j = i ^ step;
                if (j > i && j < N) {
                    // ascending = true when (i & block_size) == 0 → overall ascending sort
                    bool ascending = ((i & block_size) == 0);
                    bitonic_cas(shared_dist, shared_idx, i, j, ascending);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Phase 3: Cooperative parallel output
    device float* out_d = out_distances + row * k;
    device int32_t* out_i = out_indices + row * k;
    for (uint i = tid; i < k; i += tg_size) {
        out_d[i] = shared_dist[i];
        out_i[i] = shared_idx[i];
    }
}

kernel void block_select_max(
    device const float* distances [[buffer(0)]],
    device float* out_distances [[buffer(1)]],
    device int32_t* out_indices [[buffer(2)]],
    constant uint& nv [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    threadgroup float* shared_dist [[threadgroup(0)]],
    threadgroup int32_t* shared_idx [[threadgroup(1)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {

    float local_dist[LOCAL_K];
    int32_t local_idx[LOCAL_K];
    uint local_count = 0;
    float local_min = -INFINITY;

    device const float* row_data = distances + row * nv;

    for (uint i = tid; i < nv; i += tg_size) {
        float d = row_data[i];
        if (local_count < LOCAL_K) {
            local_dist[local_count] = d;
            local_idx[local_count] = (int32_t)i;
            local_count++;
            if (local_count == LOCAL_K) {
                local_min = local_dist[0];
                for (uint j = 1; j < LOCAL_K; j++)
                    local_min = min(local_min, local_dist[j]);
            }
        } else if (d > local_min) {
            uint worst = 0;
            for (uint j = 1; j < LOCAL_K; j++) {
                if (local_dist[j] < local_dist[worst]) worst = j;
            }
            local_dist[worst] = d;
            local_idx[worst] = (int32_t)i;
            local_min = local_dist[0];
            for (uint j = 1; j < LOCAL_K; j++)
                local_min = min(local_min, local_dist[j]);
        }
    }

    sort_local_desc(local_dist, local_idx, local_count);

    uint base = tid * LOCAL_K;
    for (uint i = 0; i < local_count; i++) {
        shared_dist[base + i] = local_dist[i];
        shared_idx[base + i] = local_idx[i];
    }
    for (uint i = local_count; i < LOCAL_K; i++) {
        shared_dist[base + i] = -INFINITY;
        shared_idx[base + i] = -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Bitonic merge sort — descending overall (max-k)
    uint N = tg_size * LOCAL_K;
    if (tid & 1) {
        for (uint i = 0; i < LOCAL_K / 2; i++) {
            uint a = base + i;
            uint b = base + LOCAL_K - 1 - i;
            float td_val = shared_dist[a]; shared_dist[a] = shared_dist[b]; shared_dist[b] = td_val;
            int32_t ti_val = shared_idx[a]; shared_idx[a] = shared_idx[b]; shared_idx[b] = ti_val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint block_size = LOCAL_K * 2; block_size <= N; block_size *= 2) {
        for (uint step = block_size / 2; step >= 1; step /= 2) {
            for (uint loc = 0; loc < LOCAL_K; loc++) {
                uint i = base + loc;
                uint j = i ^ step;
                if (j > i && j < N) {
                    // ascending = true when (i & block_size) != 0 → overall descending sort
                    bool ascending = ((i & block_size) != 0);
                    bitonic_cas(shared_dist, shared_idx, i, j, ascending);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Phase 3: Cooperative parallel output
    device float* out_d = out_distances + row * k;
    device int32_t* out_i = out_indices + row * k;
    for (uint i = tid; i < k; i += tg_size) {
        out_d[i] = shared_dist[i];
        out_i[i] = shared_idx[i];
    }
}
