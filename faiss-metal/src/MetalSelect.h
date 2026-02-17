#pragma once

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <faiss/MetricType.h>
#include <cstddef>

namespace faiss_metal {

class MetalResources;

/// Top-k selection from a distance matrix.
/// Routes to different implementations based on k:
///   k <= 32: warp_select (SIMD-group level)
///   k > 32: block_select (threadgroup level, bitonic merge)
class MetalSelect {
public:
	MetalSelect(MetalResources *resources);

	/// Encode top-k selection into an existing command buffer.
	void encode(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> distances, id<MTLBuffer> outDistances,
	            id<MTLBuffer> outIndices, size_t nq, size_t nv, size_t k, faiss::MetricType metric);

	/// Convenience: create command buffer, encode, commit, wait.
	void select(id<MTLBuffer> distances, id<MTLBuffer> outDistances, id<MTLBuffer> outIndices, size_t nq, size_t nv,
	            size_t k, faiss::MetricType metric, id<MTLCommandQueue> queue);

private:
	MetalResources *resources_;
	size_t blockSelectThreads_; // 256 default, 512 on M3+ (dynamic threadgroup mem)

	id<MTLComputePipelineState> blockSelectMinPipeline_;
	id<MTLComputePipelineState> blockSelectMaxPipeline_;
	id<MTLComputePipelineState> warpSelectMinPipeline_;
	id<MTLComputePipelineState> warpSelectMaxPipeline_;
};

} // namespace faiss_metal
