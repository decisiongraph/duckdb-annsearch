#import "MetalSelect.h"
#import <faiss-metal/MetalResources.h>
#import <faiss-metal/MetalDeviceCapabilities.h>
#include <faiss/impl/FaissAssert.h>
#include <algorithm>

namespace faiss_metal {

MetalSelect::MetalSelect(MetalResources *resources) : resources_(resources) {
    id<MTLLibrary> lib = resources->getMetalLibrary();
    id<MTLDevice> device = resources->getDevice();
    NSError *error = nil;

    const auto &caps = resources->getCapabilities();
    blockSelectThreads_ = caps.hasDynamicThreadgroupMemory ? 512 : 256;

    auto makePipeline = [&](NSString *name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        FAISS_THROW_IF_NOT_FMT(fn, "Metal function '%s' not found", [name UTF8String]);
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
        FAISS_THROW_IF_NOT_FMT(pso, "Failed to create pipeline '%s'", [name UTF8String]);
        return pso;
    };

    blockSelectMinPipeline_ = makePipeline(@"block_select_min");
    blockSelectMaxPipeline_ = makePipeline(@"block_select_max");
    warpSelectMinPipeline_ = makePipeline(@"warp_select_min");
    warpSelectMaxPipeline_ = makePipeline(@"warp_select_max");
}

void MetalSelect::encode(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> distances, id<MTLBuffer> outDistances,
                         id<MTLBuffer> outIndices, size_t nq, size_t nv, size_t k, faiss::MetricType metric) {

    if (nq == 0 || nv == 0 || k == 0)
        return;

    bool wantMin = (metric == faiss::METRIC_L2);
    uint32_t nv32 = (uint32_t)nv;
    uint32_t k32 = (uint32_t)k;

    // All k values use GPU-native selection — no MPS, no CPU fallback.
    // k <= 32: warp_select (SIMD-group, 1 thread per top-k slot)
    // k > 32: block_select (threadgroup, local sort + parallel merge)
    id<MTLComputePipelineState> pipeline;
    size_t blockThreads;

    if (k <= 32) {
        pipeline = wantMin ? warpSelectMinPipeline_ : warpSelectMaxPipeline_;
        blockThreads = 32;
    } else {
        pipeline = wantMin ? blockSelectMinPipeline_ : blockSelectMaxPipeline_;
        blockThreads = blockSelectThreads_;
        constexpr size_t LOCAL_K_LIMIT = 8; // must match shader #define LOCAL_K
        size_t maxK = blockThreads * LOCAL_K_LIMIT;
        FAISS_THROW_IF_NOT_FMT(k <= maxK, "k (%zu) exceeds Metal block_select limit (%zu = %zu threads * %zu local_k)",
                               k, maxK, blockThreads, LOCAL_K_LIMIT);
    }

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:distances offset:0 atIndex:0];
    [encoder setBuffer:outDistances offset:0 atIndex:1];
    [encoder setBuffer:outIndices offset:0 atIndex:2];
    [encoder setBytes:&nv32 length:sizeof(nv32) atIndex:3];
    [encoder setBytes:&k32 length:sizeof(k32) atIndex:4];
    if (k > 32) {
        constexpr size_t LOCAL_K = 8; // must match shader #define
        size_t tgMemBytes = blockThreads * LOCAL_K * sizeof(float);
        [encoder setThreadgroupMemoryLength:tgMemBytes atIndex:0];
        [encoder setThreadgroupMemoryLength:tgMemBytes atIndex:1];
    }
    [encoder dispatchThreadgroups:MTLSizeMake(nq, 1, 1) threadsPerThreadgroup:MTLSizeMake(blockThreads, 1, 1)];
    [encoder endEncoding];
}

void MetalSelect::select(id<MTLBuffer> distances, id<MTLBuffer> outDistances, id<MTLBuffer> outIndices, size_t nq,
                         size_t nv, size_t k, faiss::MetricType metric, id<MTLCommandQueue> queue) {

    if (nq == 0 || nv == 0 || k == 0)
        return;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    encode(cmdBuf, distances, outDistances, outIndices, nq, nv, k, metric);
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

} // namespace faiss_metal
