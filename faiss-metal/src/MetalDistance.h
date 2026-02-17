#pragma once

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <faiss/MetricType.h>
#include <cstddef>
#include <memory>

namespace faiss_metal {

class MetalResources;
class MetalL2Norm;

/// Distance computation on Metal.
///
/// Path selection based on device generation:
///   M1 (Apple7): MPS GEMM + custom norm/broadcast shaders
///   M2+ (Apple8+): simdgroup_matrix GEMM (avoids MPS overhead, 2x FP16 throughput)
///   M3+ (Apple9+): same as M2 but with dynamic threadgroup memory caching
///
/// L2: dist = ||q||^2 + ||v||^2 - 2*q·v
/// IP: dist = Q * V^T
class MetalDistance {
public:
	MetalDistance(MetalResources *resources);
	~MetalDistance();

	/// Force MPS GEMM path even on M2+ hardware. For testing both paths.
	void setForceMPS(bool force) {
		forceMPS_ = force;
	}

	/// Tell distance kernels that the vectors buffer contains half (FP16) data.
	void setVectorsFloat16(bool f16) {
		vectorsFloat16_ = f16;
	}

	/// Tell distance kernels that the vectors buffer contains bfloat16 data.
	void setVectorsBFloat16(bool bf16) {
		vectorsBFloat16_ = bf16;
	}

	/// Encode distance computation into an existing command buffer.
	/// For L2, also needs a scratch buffer for query norms (queryNormsBuf).
	/// Caller is responsible for committing.
	void encode(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> queries, id<MTLBuffer> vectors, id<MTLBuffer> vecNorms,
	            id<MTLBuffer> queryNormsBuf, id<MTLBuffer> distOutput, size_t nq, size_t nv, size_t d,
	            faiss::MetricType metric);

	/// Encode fused distance + top-k into a command buffer.
	/// Eliminates the nq*nv intermediate distance matrix.
	/// Returns true if the fused path was used, false if caller should
	/// fall back to separate encode() + select().
	bool encodeFused(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> queries, id<MTLBuffer> vectors, id<MTLBuffer> vecNorms,
	                 id<MTLBuffer> queryNormsBuf, id<MTLBuffer> outDistances, id<MTLBuffer> outIndices, size_t nq,
	                 size_t nv, size_t d, size_t k, faiss::MetricType metric);

	/// Convenience: create command buffer, encode, commit, wait.
	void compute(id<MTLBuffer> queries, id<MTLBuffer> vectors, id<MTLBuffer> vecNorms, id<MTLBuffer> distOutput,
	             size_t nq, size_t nv, size_t d, faiss::MetricType metric, id<MTLCommandQueue> queue);

private:
	MetalResources *resources_;
	bool useSimdGroupGemm_;               // M2+: use custom GEMM instead of MPS
	bool forceMPS_ = false;               // override: force MPS path for testing
	bool vectorsFloat16_ = false;         // vectors stored as half
	bool vectorsBFloat16_ = false;        // vectors stored as bfloat16
	std::unique_ptr<MetalL2Norm> l2norm_; // reused across calls

	id<MTLComputePipelineState> broadcastSumPipeline_;
	id<MTLComputePipelineState> simdgroupGemmPipeline_;
	id<MTLComputePipelineState> simdgroupGemmL2FusedPipeline_;
	id<MTLComputePipelineState> directL2Pipeline_;

	// FP16 storage variants (vectors stored as half)
	id<MTLComputePipelineState> simdgroupGemmF16StoragePipeline_;
	id<MTLComputePipelineState> simdgroupGemmL2FusedF16StoragePipeline_;
	id<MTLComputePipelineState> directL2F16StoragePipeline_;

	// BFloat16 storage pipelines (M2+/macOS 14+)
	id<MTLComputePipelineState> bf16GemmPipeline_;
	id<MTLComputePipelineState> bf16GemmL2FusedPipeline_;

	// Family 9 (M3/M4) direct device read pipelines (FP16 storage only)
	bool useFamily9Direct_ = false;
	bool useFamily9Large_ = false;
	id<MTLComputePipelineState> directGemmF16StoragePipeline_;
	id<MTLComputePipelineState> directGemmL2FusedF16StoragePipeline_;
	id<MTLComputePipelineState> largeDirectGemmF16StoragePipeline_;
	id<MTLComputePipelineState> largeDirectGemmL2FusedF16StoragePipeline_;

	// Fused distance + top-k pipelines (eliminates intermediate buffer)
	id<MTLComputePipelineState> fusedL2TopkMinPipeline_;
	id<MTLComputePipelineState> fusedIPTopkMaxPipeline_;
	id<MTLComputePipelineState> fusedL2TopkMinF16StoragePipeline_;
	id<MTLComputePipelineState> fusedIPTopkMaxF16StoragePipeline_;

	void encodeMPS(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> queries, id<MTLBuffer> vectors, id<MTLBuffer> vecNorms,
	               id<MTLBuffer> queryNormsBuf, id<MTLBuffer> distOutput, size_t nq, size_t nv, size_t d,
	               faiss::MetricType metric);

	void encodeSimdGroup(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> queries, id<MTLBuffer> vectors,
	                     id<MTLBuffer> vecNorms, id<MTLBuffer> queryNormsBuf, id<MTLBuffer> distOutput, size_t nq,
	                     size_t nv, size_t d, faiss::MetricType metric);

	void encodeFusedImpl(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> queries, id<MTLBuffer> vectors,
	                     id<MTLBuffer> vecNorms, id<MTLBuffer> queryNormsBuf, id<MTLBuffer> outDistances,
	                     id<MTLBuffer> outIndices, size_t nq, size_t nv, size_t d, size_t k, faiss::MetricType metric);
};

} // namespace faiss_metal
