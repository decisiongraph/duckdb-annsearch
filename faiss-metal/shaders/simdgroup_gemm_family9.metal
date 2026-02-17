#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "GemmParams.h"
using namespace metal;

/// Apple Family 9 (M3/M4) optimized GEMM variants.
///
/// Key optimization: B (vectors) loaded directly from device memory.
/// On Family 9 GPUs, threadgroup and device memory share the same cache
/// hierarchy, so direct device reads avoid threadgroup staging overhead
/// and free threadgroup memory for better occupancy.
///
/// A (queries) still uses threadgroup for float→half conversion.
/// FP16 storage only (B already half, no conversion needed).

// Tile sizes for 32x32 direct-read variant (same dispatch as standard GEMM)
constant uint BM = 32;
constant uint BN = 32;
constant uint BK = 32;
constant uint SIMD_TILE = 8;

// ============================================================================
// 32x32 tile, direct B reads (FP16 storage)
// ============================================================================

/// Direct-read GEMM for M3/M4 with FP16 storage.
kernel void simdgroup_gemm_direct_f16storage(
    device const float* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {

    const uint m_start = gid.y * BM;
    const uint n_start = gid.x * BN;

    if (m_start >= params.M || n_start >= params.N) return;

    // Only A needs threadgroup staging (float→half conversion)
    threadgroup half shared_a[2][BM][BK + 4] __attribute__((aligned(16)));

    const uint simd_m = (simd_group / 2) * 16;
    const uint simd_n = (simd_group % 2) * 16;

    simdgroup_float8x8 c_frag[2][2];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 2; j++)
            c_frag[i][j] = simdgroup_float8x8(0.0f);

    const uint num_k_tiles = (params.K + BK - 1) / BK;
    const uint thread_id = simd_group * 32 + simd_lane;
    const uint total_threads = 128;
    uint buf = 0;

    {
        for (uint i = thread_id; i < BM * BK; i += total_threads) {
            uint r = i / BK, c = i % BK;
            uint row = m_start + r, col = c;
            shared_a[0][r][c] = (row < params.M && col < params.K)
                ? half(A[row * params.K + col]) : half(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_k = 0; tile_k < num_k_tiles; tile_k++) {
        uint next_buf = 1 - buf;
        uint k_offset = tile_k * BK;

        if (tile_k + 1 < num_k_tiles) {
            uint k_next = (tile_k + 1) * BK;
            for (uint i = thread_id; i < BM * BK; i += total_threads) {
                uint r = i / BK, c = i % BK;
                uint row = m_start + r, col = k_next + c;
                shared_a[next_buf][r][c] = (row < params.M && col < params.K)
                    ? half(A[row * params.K + col]) : half(0.0h);
            }
        }

        for (uint k = 0; k < BK; k += SIMD_TILE) {
            simdgroup_half8x8 a_frag[2];
            for (uint i = 0; i < 2; i++)
                simdgroup_load(a_frag[i], &shared_a[buf][simd_m + i * 8][k], BK + 4);

            for (uint i = 0; i < 2; i++) {
                for (uint j = 0; j < 2; j++) {
                    simdgroup_half8x8 bt;
                    // Load B directly from device (Family 9 cache handles this)
                    uint b_row = n_start + simd_n + j * 8;
                    uint b_col = k_offset + k;
                    if (b_row < params.N && b_col < params.K) {
                        simdgroup_load(bt, B + b_row * params.K + b_col, params.K, true);
                    } else {
                        bt = simdgroup_half8x8(0.0h);
                    }
                    simdgroup_multiply_accumulate(c_frag[i][j], a_frag[i], bt, c_frag[i][j]);
                }
            }
        }

        buf = next_buf;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float store_buf[BM][BN + 4] __attribute__((aligned(16)));

    for (uint i = 0; i < 2; i++) {
        for (uint j = 0; j < 2; j++) {
            simdgroup_store(c_frag[i][j],
                            &store_buf[simd_m + i * 8][simd_n + j * 8],
                            BN + 4);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_id; idx < BM * BN; idx += total_threads) {
        uint r = idx / BN;
        uint c = idx % BN;
        uint out_row = m_start + r;
        uint out_col = n_start + c;
        if (out_row < params.M && out_col < params.N) {
            uint oidx = out_row * params.N + out_col;
            float val = store_buf[r][c] * params.alpha;
            if (params.beta != 0.0f) {
                val += params.beta * C[oidx];
            }
            C[oidx] = val;
        }
    }
}

/// Direct-read fused L2 GEMM for M3/M4 with FP16 storage.
kernel void simdgroup_gemm_l2_fused_direct_f16storage(
    device const float* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GemmL2Params& params [[buffer(3)]],
    device const float* row_norms [[buffer(4)]],
    device const float* col_norms [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {

    const uint m_start = gid.y * BM;
    const uint n_start = gid.x * BN;

    if (m_start >= params.M || n_start >= params.N) return;

    threadgroup half shared_a[2][BM][BK + 4] __attribute__((aligned(16)));

    const uint simd_m = (simd_group / 2) * 16;
    const uint simd_n = (simd_group % 2) * 16;

    simdgroup_float8x8 c_frag[2][2];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 2; j++)
            c_frag[i][j] = simdgroup_float8x8(0.0f);

    const uint num_k_tiles = (params.K + BK - 1) / BK;
    const uint thread_id = simd_group * 32 + simd_lane;
    const uint total_threads = 128;
    uint buf = 0;

    {
        for (uint i = thread_id; i < BM * BK; i += total_threads) {
            uint r = i / BK, c = i % BK;
            uint row = m_start + r, col = c;
            shared_a[0][r][c] = (row < params.M && col < params.K)
                ? half(A[row * params.K + col]) : half(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_k = 0; tile_k < num_k_tiles; tile_k++) {
        uint next_buf = 1 - buf;
        uint k_offset = tile_k * BK;

        if (tile_k + 1 < num_k_tiles) {
            uint k_next = (tile_k + 1) * BK;
            for (uint i = thread_id; i < BM * BK; i += total_threads) {
                uint r = i / BK, c = i % BK;
                uint row = m_start + r, col = k_next + c;
                shared_a[next_buf][r][c] = (row < params.M && col < params.K)
                    ? half(A[row * params.K + col]) : half(0.0h);
            }
        }

        for (uint k = 0; k < BK; k += SIMD_TILE) {
            simdgroup_half8x8 a_frag[2];
            for (uint i = 0; i < 2; i++)
                simdgroup_load(a_frag[i], &shared_a[buf][simd_m + i * 8][k], BK + 4);

            for (uint i = 0; i < 2; i++) {
                for (uint j = 0; j < 2; j++) {
                    simdgroup_half8x8 bt;
                    uint b_row = n_start + simd_n + j * 8;
                    uint b_col = k_offset + k;
                    if (b_row < params.N && b_col < params.K) {
                        simdgroup_load(bt, B + b_row * params.K + b_col, params.K, true);
                    } else {
                        bt = simdgroup_half8x8(0.0h);
                    }
                    simdgroup_multiply_accumulate(c_frag[i][j], a_frag[i], bt, c_frag[i][j]);
                }
            }
        }

        buf = next_buf;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float store_buf[BM][BN + 4] __attribute__((aligned(16)));

    for (uint i = 0; i < 2; i++) {
        for (uint j = 0; j < 2; j++) {
            simdgroup_store(c_frag[i][j],
                            &store_buf[simd_m + i * 8][simd_n + j * 8],
                            BN + 4);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_id; idx < BM * BN; idx += total_threads) {
        uint r = idx / BN;
        uint c = idx % BN;
        uint out_row = m_start + r;
        uint out_col = n_start + c;
        if (out_row < params.M && out_col < params.N) {
            uint oidx = out_row * params.N + out_col;
            float val = store_buf[r][c] * params.alpha
                      + row_norms[out_row] + col_norms[out_col];
            C[oidx] = val;
        }
    }
}

// ============================================================================
// 64x64 tile variants for M3/M4.
// 8 simdgroups (256 threads), each computes 16x16 in a 4x2 grid within 64x64.
// 4x fewer threadgroups dispatched, better data reuse per tile.
// Uses direct B reads (no threadgroup for B).
//
// Threadgroup memory:
//   shared_a[2][64][36] half = 9216 bytes (A double-buffered)
//   store_buf[64][68] float = 17408 bytes
//   Total = 26624 bytes < 32KB
//
// Dispatch: grid = (ceil(N/64), ceil(M/64)), threadgroup = (32, 8) = 256 threads
// ============================================================================

constant uint BM_LARGE = 64;
constant uint BN_LARGE = 64;
constant uint BK_LARGE = 32;

/// 64x64 tile GEMM for M3/M4 with FP16 storage and direct B reads.
kernel void simdgroup_gemm_large_direct_f16storage(
    device const float* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {

    const uint m_start = gid.y * BM_LARGE;
    const uint n_start = gid.x * BN_LARGE;

    if (m_start >= params.M || n_start >= params.N) return;

    // A in threadgroup (float→half), B from device directly
    threadgroup half shared_a[2][BM_LARGE][BK_LARGE + 4] __attribute__((aligned(16)));

    // 8 simdgroups in 4x2 grid: 4 rows x 2 cols of 16x16 blocks
    const uint simd_m = (simd_group / 2) * 16;
    const uint simd_n = (simd_group % 2) * 16;

    // Each simdgroup: 2x2 grid of 8x8 fragments
    simdgroup_float8x8 c_frag[2][2];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 2; j++)
            c_frag[i][j] = simdgroup_float8x8(0.0f);

    const uint num_k_tiles = (params.K + BK_LARGE - 1) / BK_LARGE;
    const uint thread_id = simd_group * 32 + simd_lane;
    const uint total_threads = 256;
    uint buf = 0;

    // Preload first A tile
    {
        for (uint i = thread_id; i < BM_LARGE * BK_LARGE; i += total_threads) {
            uint r = i / BK_LARGE, c = i % BK_LARGE;
            uint row = m_start + r, col = c;
            shared_a[0][r][c] = (row < params.M && col < params.K)
                ? half(A[row * params.K + col]) : half(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_k = 0; tile_k < num_k_tiles; tile_k++) {
        uint next_buf = 1 - buf;
        uint k_offset = tile_k * BK_LARGE;

        // Prefetch next A tile
        if (tile_k + 1 < num_k_tiles) {
            uint k_next = (tile_k + 1) * BK_LARGE;
            for (uint i = thread_id; i < BM_LARGE * BK_LARGE; i += total_threads) {
                uint r = i / BK_LARGE, c = i % BK_LARGE;
                uint row = m_start + r, col = k_next + c;
                shared_a[next_buf][r][c] = (row < params.M && col < params.K)
                    ? half(A[row * params.K + col]) : half(0.0h);
            }
        }

        for (uint k = 0; k < BK_LARGE; k += SIMD_TILE) {
            simdgroup_half8x8 a_frag[2];
            for (uint i = 0; i < 2; i++)
                simdgroup_load(a_frag[i], &shared_a[buf][simd_m + i * 8][k], BK_LARGE + 4);

            for (uint i = 0; i < 2; i++) {
                for (uint j = 0; j < 2; j++) {
                    simdgroup_half8x8 bt;
                    uint b_row = n_start + simd_n + j * 8;
                    uint b_col = k_offset + k;
                    if (b_row < params.N && b_col < params.K) {
                        simdgroup_load(bt, B + b_row * params.K + b_col, params.K, true);
                    } else {
                        bt = simdgroup_half8x8(0.0h);
                    }
                    simdgroup_multiply_accumulate(c_frag[i][j], a_frag[i], bt, c_frag[i][j]);
                }
            }
        }

        buf = next_buf;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store: reuse shared_a memory for store_buf (compute is done)
    threadgroup float store_buf[BM_LARGE][BN_LARGE + 4] __attribute__((aligned(16)));

    for (uint i = 0; i < 2; i++) {
        for (uint j = 0; j < 2; j++) {
            simdgroup_store(c_frag[i][j],
                            &store_buf[simd_m + i * 8][simd_n + j * 8],
                            BN_LARGE + 4);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_id; idx < BM_LARGE * BN_LARGE; idx += total_threads) {
        uint r = idx / BN_LARGE;
        uint c = idx % BN_LARGE;
        uint out_row = m_start + r;
        uint out_col = n_start + c;
        if (out_row < params.M && out_col < params.N) {
            uint oidx = out_row * params.N + out_col;
            float val = store_buf[r][c] * params.alpha;
            if (params.beta != 0.0f) {
                val += params.beta * C[oidx];
            }
            C[oidx] = val;
        }
    }
}

/// 64x64 tile fused L2 GEMM for M3/M4 with FP16 storage and direct B reads.
kernel void simdgroup_gemm_l2_fused_large_direct_f16storage(
    device const float* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GemmL2Params& params [[buffer(3)]],
    device const float* row_norms [[buffer(4)]],
    device const float* col_norms [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {

    const uint m_start = gid.y * BM_LARGE;
    const uint n_start = gid.x * BN_LARGE;

    if (m_start >= params.M || n_start >= params.N) return;

    threadgroup half shared_a[2][BM_LARGE][BK_LARGE + 4] __attribute__((aligned(16)));

    const uint simd_m = (simd_group / 2) * 16;
    const uint simd_n = (simd_group % 2) * 16;

    simdgroup_float8x8 c_frag[2][2];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 2; j++)
            c_frag[i][j] = simdgroup_float8x8(0.0f);

    const uint num_k_tiles = (params.K + BK_LARGE - 1) / BK_LARGE;
    const uint thread_id = simd_group * 32 + simd_lane;
    const uint total_threads = 256;
    uint buf = 0;

    {
        for (uint i = thread_id; i < BM_LARGE * BK_LARGE; i += total_threads) {
            uint r = i / BK_LARGE, c = i % BK_LARGE;
            uint row = m_start + r, col = c;
            shared_a[0][r][c] = (row < params.M && col < params.K)
                ? half(A[row * params.K + col]) : half(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_k = 0; tile_k < num_k_tiles; tile_k++) {
        uint next_buf = 1 - buf;
        uint k_offset = tile_k * BK_LARGE;

        if (tile_k + 1 < num_k_tiles) {
            uint k_next = (tile_k + 1) * BK_LARGE;
            for (uint i = thread_id; i < BM_LARGE * BK_LARGE; i += total_threads) {
                uint r = i / BK_LARGE, c = i % BK_LARGE;
                uint row = m_start + r, col = k_next + c;
                shared_a[next_buf][r][c] = (row < params.M && col < params.K)
                    ? half(A[row * params.K + col]) : half(0.0h);
            }
        }

        for (uint k = 0; k < BK_LARGE; k += SIMD_TILE) {
            simdgroup_half8x8 a_frag[2];
            for (uint i = 0; i < 2; i++)
                simdgroup_load(a_frag[i], &shared_a[buf][simd_m + i * 8][k], BK_LARGE + 4);

            for (uint i = 0; i < 2; i++) {
                for (uint j = 0; j < 2; j++) {
                    simdgroup_half8x8 bt;
                    uint b_row = n_start + simd_n + j * 8;
                    uint b_col = k_offset + k;
                    if (b_row < params.N && b_col < params.K) {
                        simdgroup_load(bt, B + b_row * params.K + b_col, params.K, true);
                    } else {
                        bt = simdgroup_half8x8(0.0h);
                    }
                    simdgroup_multiply_accumulate(c_frag[i][j], a_frag[i], bt, c_frag[i][j]);
                }
            }
        }

        buf = next_buf;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float store_buf[BM_LARGE][BN_LARGE + 4] __attribute__((aligned(16)));

    for (uint i = 0; i < 2; i++) {
        for (uint j = 0; j < 2; j++) {
            simdgroup_store(c_frag[i][j],
                            &store_buf[simd_m + i * 8][simd_n + j * 8],
                            BN_LARGE + 4);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_id; idx < BM_LARGE * BN_LARGE; idx += total_threads) {
        uint r = idx / BN_LARGE;
        uint c = idx % BN_LARGE;
        uint out_row = m_start + r;
        uint out_col = n_start + c;
        if (out_row < params.M && out_col < params.N) {
            uint oidx = out_row * params.N + out_col;
            float val = store_buf[r][c] * params.alpha
                      + row_norms[out_row] + col_norms[out_col];
            C[oidx] = val;
        }
    }
}
