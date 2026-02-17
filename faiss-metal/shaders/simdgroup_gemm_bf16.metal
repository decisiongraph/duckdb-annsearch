#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "GemmParams.h"
using namespace metal;

/// BFloat16 storage GEMM variants.
/// Vectors stored as bfloat16 (8-bit exponent, same dynamic range as float32).
/// Same 2x bandwidth savings as FP16 but better numerical stability for
/// outlier-heavy embeddings.
///
/// Strategy: load bfloat16 from device, convert to half for simdgroup_matrix
/// multiply (bfloat→half loses 3 mantissa bits but gains 3 exponent bits vs
/// keeping in bfloat — the multiply precision is dominated by the accumulator
/// which is float32 anyway).

constant uint BM = 32;
constant uint BN = 32;
constant uint BK = 32;
constant uint SIMD_TILE = 8;

/// GEMM with BF16-stored B matrix. A is float, converted to half on load.
/// B is bfloat16, converted to half in threadgroup.
kernel void simdgroup_gemm_bf16storage(
    device const float* A [[buffer(0)]],
    device const bfloat* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {

    const uint m_start = gid.y * BM;
    const uint n_start = gid.x * BN;

    if (m_start >= params.M || n_start >= params.N) return;

    threadgroup half shared_a[2][BM][BK + 4] __attribute__((aligned(16)));
    threadgroup half shared_b[2][BN][BK + 4] __attribute__((aligned(16)));

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
        for (uint i = thread_id; i < BN * BK; i += total_threads) {
            uint r = i / BK, c = i % BK;
            uint row = n_start + r, col = c;
            shared_b[0][r][c] = (row < params.N && col < params.K)
                ? half(B[row * params.K + col]) : half(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_k = 0; tile_k < num_k_tiles; tile_k++) {
        uint next_buf = 1 - buf;

        if (tile_k + 1 < num_k_tiles) {
            uint k_next = (tile_k + 1) * BK;
            for (uint i = thread_id; i < BM * BK; i += total_threads) {
                uint r = i / BK, c = i % BK;
                uint row = m_start + r, col = k_next + c;
                shared_a[next_buf][r][c] = (row < params.M && col < params.K)
                    ? half(A[row * params.K + col]) : half(0.0h);
            }
            for (uint i = thread_id; i < BN * BK; i += total_threads) {
                uint r = i / BK, c = i % BK;
                uint row = n_start + r, col = k_next + c;
                shared_b[next_buf][r][c] = (row < params.N && col < params.K)
                    ? half(B[row * params.K + col]) : half(0.0h);
            }
        }

        for (uint k = 0; k < BK; k += SIMD_TILE) {
            simdgroup_half8x8 a_frag[2];
            for (uint i = 0; i < 2; i++)
                simdgroup_load(a_frag[i], &shared_a[buf][simd_m + i * 8][k], BK + 4);

            for (uint i = 0; i < 2; i++) {
                for (uint j = 0; j < 2; j++) {
                    simdgroup_half8x8 bt;
                    simdgroup_load(bt, &shared_b[buf][simd_n + j * 8][k], BK + 4, true);
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

/// Fused L2 distance GEMM with BF16-stored vectors.
kernel void simdgroup_gemm_l2_fused_bf16storage(
    device const float* A [[buffer(0)]],
    device const bfloat* B [[buffer(1)]],
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
    threadgroup half shared_b[2][BN][BK + 4] __attribute__((aligned(16)));

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
        for (uint i = thread_id; i < BN * BK; i += total_threads) {
            uint r = i / BK, c = i % BK;
            uint row = n_start + r, col = c;
            shared_b[0][r][c] = (row < params.N && col < params.K)
                ? half(B[row * params.K + col]) : half(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_k = 0; tile_k < num_k_tiles; tile_k++) {
        uint next_buf = 1 - buf;

        if (tile_k + 1 < num_k_tiles) {
            uint k_next = (tile_k + 1) * BK;
            for (uint i = thread_id; i < BM * BK; i += total_threads) {
                uint r = i / BK, c = i % BK;
                uint row = m_start + r, col = k_next + c;
                shared_a[next_buf][r][c] = (row < params.M && col < params.K)
                    ? half(A[row * params.K + col]) : half(0.0h);
            }
            for (uint i = thread_id; i < BN * BK; i += total_threads) {
                uint r = i / BK, c = i % BK;
                uint row = n_start + r, col = k_next + c;
                shared_b[next_buf][r][c] = (row < params.N && col < params.K)
                    ? half(B[row * params.K + col]) : half(0.0h);
            }
        }

        for (uint k = 0; k < BK; k += SIMD_TILE) {
            simdgroup_half8x8 a_frag[2];
            for (uint i = 0; i < 2; i++)
                simdgroup_load(a_frag[i], &shared_a[buf][simd_m + i * 8][k], BK + 4);

            for (uint i = 0; i < 2; i++) {
                for (uint j = 0; j < 2; j++) {
                    simdgroup_half8x8 bt;
                    simdgroup_load(bt, &shared_b[buf][simd_n + j * 8][k], BK + 4, true);
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
