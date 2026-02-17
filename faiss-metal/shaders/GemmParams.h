#pragma once

/// Shared between Metal shaders and C++ host code.
/// Metal defines __METAL_VERSION__; C++ does not.
#ifdef __METAL_VERSION__
using uint32 = uint;
#else
#include <cstdint>
using uint32 = uint32_t;
#endif

struct GemmParams {
	uint32 M; // rows of A / rows of C
	uint32 N; // rows of B / cols of C  (B is transposed)
	uint32 K; // cols of A / cols of B
	float alpha;
	float beta;
};

/// Extended params for fused GEMM + L2 norm broadcast.
/// C[i][j] = alpha * (A * B^T)[i][j] + row_norms[i] + col_norms[j]
struct GemmL2Params {
	uint32 M;
	uint32 N;
	uint32 K;
	float alpha; // -2.0 for L2
};

/// Params for fused distance + top-k kernel.
/// Eliminates the nq*nv intermediate distance matrix for small nq.
struct FusedTopkParams {
	uint32 nq; // number of queries
	uint32 nv; // number of vectors
	uint32 d;  // dimension
	uint32 k;  // top-k
};
