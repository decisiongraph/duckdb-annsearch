#import "MetalDistance.h"
#import "MetalL2Norm.h"
#import <faiss-metal/MetalResources.h>
#import <faiss-metal/MetalDeviceCapabilities.h>
#include <faiss/impl/FaissAssert.h>
#include <algorithm>
#include "GemmParams.h"

namespace faiss_metal {

MetalDistance::MetalDistance(MetalResources *resources)
    : resources_(resources), l2norm_(std::make_unique<MetalL2Norm>(resources)) {
    id<MTLLibrary> lib = resources->getMetalLibrary();
    id<MTLDevice> device = resources->getDevice();
    NSError *error = nil;

    const auto &caps = resources->getCapabilities();
    useSimdGroupGemm_ = caps.hasFastFP16;

    id<MTLFunction> bsFn = [lib newFunctionWithName:@"broadcast_sum_l2"];
    FAISS_THROW_IF_NOT_MSG(bsFn, "Metal function 'broadcast_sum_l2' not found");
    broadcastSumPipeline_ = [device newComputePipelineStateWithFunction:bsFn error:&error];
    FAISS_THROW_IF_NOT_MSG(broadcastSumPipeline_, "Failed to create broadcast_sum pipeline");

    if (useSimdGroupGemm_) {
        id<MTLFunction> gemmFn = [lib newFunctionWithName:@"simdgroup_gemm_f32_via_f16"];
        FAISS_THROW_IF_NOT_MSG(gemmFn, "Metal function 'simdgroup_gemm_f32_via_f16' not found");
        simdgroupGemmPipeline_ = [device newComputePipelineStateWithFunction:gemmFn error:&error];
        FAISS_THROW_IF_NOT_MSG(simdgroupGemmPipeline_, "Failed to create simdgroup_gemm pipeline");

        id<MTLFunction> fusedFn = [lib newFunctionWithName:@"simdgroup_gemm_l2_fused"];
        FAISS_THROW_IF_NOT_MSG(fusedFn, "Metal function 'simdgroup_gemm_l2_fused' not found");
        simdgroupGemmL2FusedPipeline_ = [device newComputePipelineStateWithFunction:fusedFn error:&error];
        FAISS_THROW_IF_NOT_MSG(simdgroupGemmL2FusedPipeline_, "Failed to create fused L2 GEMM pipeline");

        id<MTLFunction> directFn = [lib newFunctionWithName:@"l2_distance_direct_f16"];
        FAISS_THROW_IF_NOT_MSG(directFn, "Metal function 'l2_distance_direct_f16' not found");
        directL2Pipeline_ = [device newComputePipelineStateWithFunction:directFn error:&error];
        FAISS_THROW_IF_NOT_MSG(directL2Pipeline_, "Failed to create direct L2 pipeline");

        // FP16 storage variants (vectors stored as half)
        id<MTLFunction> gemmF16Fn = [lib newFunctionWithName:@"simdgroup_gemm_f16storage"];
        FAISS_THROW_IF_NOT_MSG(gemmF16Fn, "Metal function 'simdgroup_gemm_f16storage' not found");
        simdgroupGemmF16StoragePipeline_ = [device newComputePipelineStateWithFunction:gemmF16Fn error:&error];
        FAISS_THROW_IF_NOT_MSG(simdgroupGemmF16StoragePipeline_, "Failed to create f16storage GEMM pipeline");

        id<MTLFunction> fusedF16Fn = [lib newFunctionWithName:@"simdgroup_gemm_l2_fused_f16storage"];
        FAISS_THROW_IF_NOT_MSG(fusedF16Fn, "Metal function 'simdgroup_gemm_l2_fused_f16storage' not found");
        simdgroupGemmL2FusedF16StoragePipeline_ = [device newComputePipelineStateWithFunction:fusedF16Fn error:&error];
        FAISS_THROW_IF_NOT_MSG(simdgroupGemmL2FusedF16StoragePipeline_,
                               "Failed to create f16storage fused L2 pipeline");

        id<MTLFunction> directF16Fn = [lib newFunctionWithName:@"l2_distance_direct_f16storage"];
        FAISS_THROW_IF_NOT_MSG(directF16Fn, "Metal function 'l2_distance_direct_f16storage' not found");
        directL2F16StoragePipeline_ = [device newComputePipelineStateWithFunction:directF16Fn error:&error];
        FAISS_THROW_IF_NOT_MSG(directL2F16StoragePipeline_, "Failed to create f16storage direct L2 pipeline");
    }

    // BFloat16 storage pipelines (M2+/macOS 14+)
    if (caps.hasBFloat16) {
        id<MTLFunction> bf16GemmFn = [lib newFunctionWithName:@"simdgroup_gemm_bf16storage"];
        FAISS_THROW_IF_NOT_MSG(bf16GemmFn, "Metal function 'simdgroup_gemm_bf16storage' not found");
        bf16GemmPipeline_ = [device newComputePipelineStateWithFunction:bf16GemmFn error:&error];
        FAISS_THROW_IF_NOT_MSG(bf16GemmPipeline_, "Failed to create bf16 GEMM pipeline");

        id<MTLFunction> bf16L2Fn = [lib newFunctionWithName:@"simdgroup_gemm_l2_fused_bf16storage"];
        FAISS_THROW_IF_NOT_MSG(bf16L2Fn, "Metal function 'simdgroup_gemm_l2_fused_bf16storage' not found");
        bf16GemmL2FusedPipeline_ = [device newComputePipelineStateWithFunction:bf16L2Fn error:&error];
        FAISS_THROW_IF_NOT_MSG(bf16GemmL2FusedPipeline_, "Failed to create bf16 fused L2 pipeline");
    }

    // Family 9 (M3/M4) direct device read pipelines
    useFamily9Direct_ = caps.hasDynamicThreadgroupMemory;
    useFamily9Large_ = caps.hasDynamicThreadgroupMemory;
    if (useFamily9Direct_) {
        auto makePso = [&](NSString *name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            FAISS_THROW_IF_NOT_FMT(fn, "Metal function '%s' not found", [name UTF8String]);
            id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
            FAISS_THROW_IF_NOT_FMT(pso, "Failed to create pipeline '%s'", [name UTF8String]);
            return pso;
        };

        directGemmF16StoragePipeline_ = makePso(@"simdgroup_gemm_direct_f16storage");
        directGemmL2FusedF16StoragePipeline_ = makePso(@"simdgroup_gemm_l2_fused_direct_f16storage");
        largeDirectGemmF16StoragePipeline_ = makePso(@"simdgroup_gemm_large_direct_f16storage");
        largeDirectGemmL2FusedF16StoragePipeline_ = makePso(@"simdgroup_gemm_l2_fused_large_direct_f16storage");
    }

    // Fused distance + top-k pipelines (all chips, gated by nq/k at dispatch)
    auto makePipeline = [&](NSString *name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        FAISS_THROW_IF_NOT_FMT(fn, "Metal function '%s' not found", [name UTF8String]);
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
        FAISS_THROW_IF_NOT_FMT(pso, "Failed to create pipeline '%s'", [name UTF8String]);
        return pso;
    };

    fusedL2TopkMinPipeline_ = makePipeline(@"fused_l2_topk_min");
    fusedIPTopkMaxPipeline_ = makePipeline(@"fused_ip_topk_max");
    fusedL2TopkMinF16StoragePipeline_ = makePipeline(@"fused_l2_topk_min_f16storage");
    fusedIPTopkMaxF16StoragePipeline_ = makePipeline(@"fused_ip_topk_max_f16storage");
}

MetalDistance::~MetalDistance() = default;

void MetalDistance::encode(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> queries, id<MTLBuffer> vectors,
                           id<MTLBuffer> vecNorms, id<MTLBuffer> queryNormsBuf, id<MTLBuffer> distOutput, size_t nq,
                           size_t nv, size_t d, faiss::MetricType metric) {

    if (nq == 0 || nv == 0)
        return;

    if (useSimdGroupGemm_ && !forceMPS_) {
        encodeSimdGroup(cmdBuf, queries, vectors, vecNorms, queryNormsBuf, distOutput, nq, nv, d, metric);
    } else {
        encodeMPS(cmdBuf, queries, vectors, vecNorms, queryNormsBuf, distOutput, nq, nv, d, metric);
    }
}

void MetalDistance::compute(id<MTLBuffer> queries, id<MTLBuffer> vectors, id<MTLBuffer> vecNorms,
                            id<MTLBuffer> distOutput, size_t nq, size_t nv, size_t d, faiss::MetricType metric,
                            id<MTLCommandQueue> queue) {

    if (nq == 0 || nv == 0)
        return;

    id<MTLDevice> device = resources_->getDevice();
    id<MTLBuffer> queryNormsBuf = nil;
    if (metric == faiss::METRIC_L2) {
        queryNormsBuf = [device newBufferWithLength:nq * sizeof(float) options:MTLResourceStorageModeShared];
    }

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    encode(cmdBuf, queries, vectors, vecNorms, queryNormsBuf, distOutput, nq, nv, d, metric);
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

// --- M1 path: MPS GEMM ---

void MetalDistance::encodeMPS(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> queries, id<MTLBuffer> vectors,
                              id<MTLBuffer> vecNorms, id<MTLBuffer> queryNormsBuf, id<MTLBuffer> distOutput, size_t nq,
                              size_t nv, size_t d, faiss::MetricType metric) {

    MPSMatrixDescriptor *queryDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:nq
                                                                           columns:d
                                                                          rowBytes:d * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
    MPSDataType vecType = vectorsFloat16_ ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
    size_t vecElemSize = vectorsFloat16_ ? sizeof(uint16_t) : sizeof(float);
    MPSMatrixDescriptor *vecDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:nv
                                                                         columns:d
                                                                        rowBytes:d * vecElemSize
                                                                        dataType:vecType];
    MPSMatrixDescriptor *outDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:nq
                                                                         columns:nv
                                                                        rowBytes:nv * sizeof(float)
                                                                        dataType:MPSDataTypeFloat32];

    MPSMatrix *queryMat = [[MPSMatrix alloc] initWithBuffer:queries descriptor:queryDesc];
    MPSMatrix *vecMat = [[MPSMatrix alloc] initWithBuffer:vectors descriptor:vecDesc];
    MPSMatrix *outMat = [[MPSMatrix alloc] initWithBuffer:distOutput descriptor:outDesc];

    if (metric == faiss::METRIC_INNER_PRODUCT) {
        MPSMatrixMultiplication *gemm = [[MPSMatrixMultiplication alloc] initWithDevice:resources_->getDevice()
                                                                          transposeLeft:false
                                                                         transposeRight:true
                                                                             resultRows:nq
                                                                          resultColumns:nv
                                                                        interiorColumns:d
                                                                                  alpha:1.0
                                                                                   beta:0.0];
        [gemm encodeToCommandBuffer:cmdBuf leftMatrix:queryMat rightMatrix:vecMat resultMatrix:outMat];
    } else {
        // L2: norms + (-2*Q*V^T) + broadcast_sum — all in one command buffer
        l2norm_->encode(cmdBuf, queries, queryNormsBuf, nq, d);

        MPSMatrixMultiplication *gemm = [[MPSMatrixMultiplication alloc] initWithDevice:resources_->getDevice()
                                                                          transposeLeft:false
                                                                         transposeRight:true
                                                                             resultRows:nq
                                                                          resultColumns:nv
                                                                        interiorColumns:d
                                                                                  alpha:-2.0
                                                                                   beta:0.0];
        [gemm encodeToCommandBuffer:cmdBuf leftMatrix:queryMat rightMatrix:vecMat resultMatrix:outMat];

        // Encode broadcast_sum into same command buffer
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        uint32_t nv32 = (uint32_t)nv;
        [enc setComputePipelineState:broadcastSumPipeline_];
        [enc setBuffer:distOutput offset:0 atIndex:0];
        [enc setBuffer:queryNormsBuf offset:0 atIndex:1];
        [enc setBuffer:vecNorms offset:0 atIndex:2];
        [enc setBytes:&nv32 length:sizeof(nv32) atIndex:3];
        MTLSize gridSize = MTLSizeMake(nv, nq, 1);
        MTLSize groupSize = MTLSizeMake(std::min((size_t)256, nv), std::min((size_t)4, nq), 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [enc endEncoding];
    }
}

// --- M2+ path: simdgroup_matrix GEMM ---

void MetalDistance::encodeSimdGroup(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> queries, id<MTLBuffer> vectors,
                                    id<MTLBuffer> vecNorms, id<MTLBuffer> queryNormsBuf, id<MTLBuffer> distOutput,
                                    size_t nq, size_t nv, size_t d, faiss::MetricType metric) {

    // For small nv with L2, use fused direct distance
    if (metric == faiss::METRIC_L2 && nv <= 256) {
        uint32_t d32 = (uint32_t)d;
        uint32_t nv32 = (uint32_t)nv;
        auto pso = vectorsFloat16_ ? directL2F16StoragePipeline_ : directL2Pipeline_;
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:queries offset:0 atIndex:0];
        [enc setBuffer:vectors offset:0 atIndex:1];
        [enc setBuffer:distOutput offset:0 atIndex:2];
        [enc setBytes:&d32 length:sizeof(d32) atIndex:3];
        [enc setBytes:&nv32 length:sizeof(nv32) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(nv, nq, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        return;
    }

    // Select tile size and pipeline variant based on device generation.
    // Family 9 (M3/M4) with FP16 storage: use direct device reads.
    // Large tiles (64x64) for Family 9 when problem is large enough to benefit.
    bool useLargeTile = useFamily9Large_ && vectorsFloat16_ && nv >= 256 && nq >= 64;
    bool useDirectRead = useFamily9Direct_ && vectorsFloat16_ && !useLargeTile;

    size_t tileM = useLargeTile ? 64 : 32;
    size_t tileN = useLargeTile ? 64 : 32;
    size_t gridX = (nv + tileN - 1) / tileN;
    size_t gridY = (nq + tileM - 1) / tileM;
    size_t threadsY = useLargeTile ? 8 : 4; // 256 or 128 threads

    if (metric == faiss::METRIC_L2) {
        l2norm_->encode(cmdBuf, queries, queryNormsBuf, nq, d);

        GemmL2Params params{(uint32_t)nq, (uint32_t)nv, (uint32_t)d, -2.0f};

        id<MTLComputePipelineState> pso;
        if (useLargeTile) {
            pso = largeDirectGemmL2FusedF16StoragePipeline_;
        } else if (useDirectRead) {
            pso = directGemmL2FusedF16StoragePipeline_;
        } else if (vectorsBFloat16_) {
            pso = bf16GemmL2FusedPipeline_;
        } else {
            pso = vectorsFloat16_ ? simdgroupGemmL2FusedF16StoragePipeline_ : simdgroupGemmL2FusedPipeline_;
        }

        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:queries offset:0 atIndex:0];
        [enc setBuffer:vectors offset:0 atIndex:1];
        [enc setBuffer:distOutput offset:0 atIndex:2];
        [enc setBytes:&params length:sizeof(params) atIndex:3];
        [enc setBuffer:queryNormsBuf offset:0 atIndex:4];
        [enc setBuffer:vecNorms offset:0 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(gridX, gridY, 1) threadsPerThreadgroup:MTLSizeMake(32, threadsY, 1)];
        [enc endEncoding];
    } else {
        GemmParams params{(uint32_t)nq, (uint32_t)nv, (uint32_t)d, 1.0f, 0.0f};

        id<MTLComputePipelineState> pso;
        if (useLargeTile) {
            pso = largeDirectGemmF16StoragePipeline_;
        } else if (useDirectRead) {
            pso = directGemmF16StoragePipeline_;
        } else if (vectorsBFloat16_) {
            pso = bf16GemmPipeline_;
        } else {
            pso = vectorsFloat16_ ? simdgroupGemmF16StoragePipeline_ : simdgroupGemmPipeline_;
        }

        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:queries offset:0 atIndex:0];
        [enc setBuffer:vectors offset:0 atIndex:1];
        [enc setBuffer:distOutput offset:0 atIndex:2];
        [enc setBytes:&params length:sizeof(params) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(gridX, gridY, 1) threadsPerThreadgroup:MTLSizeMake(32, threadsY, 1)];
        [enc endEncoding];
    }
}

// --- Fused distance + top-k path ---

bool MetalDistance::encodeFused(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> queries, id<MTLBuffer> vectors,
                                id<MTLBuffer> vecNorms, id<MTLBuffer> queryNormsBuf, id<MTLBuffer> outDistances,
                                id<MTLBuffer> outIndices, size_t nq, size_t nv, size_t d, size_t k,
                                faiss::MetricType metric) {

    // Fused path: eliminates the nq*nv intermediate distance buffer by computing
    // distances on-the-fly with warp_select. Only beneficial when that buffer
    // would be large (>32MB), since the fused kernel scans vectors sequentially
    // and is slower per-element than the tiled GEMM path.
    // Requires k <= 32 (warp_select constraint) and nq <= 4 (no query batching).
    if (nq == 0 || nv == 0 || k == 0 || k > 32 || nq > 4) {
        return false;
    }

    // Only fuse when intermediate buffer would exceed ~32MB.
    // Below that, GEMM + separate select is faster (better compute efficiency).
    if (nq * nv < 8'000'000) {
        return false;
    }

    if (metric == faiss::METRIC_L2) {
        l2norm_->encode(cmdBuf, queries, queryNormsBuf, nq, d);
    }

    encodeFusedImpl(cmdBuf, queries, vectors, vecNorms, queryNormsBuf, outDistances, outIndices, nq, nv, d, k, metric);
    return true;
}

void MetalDistance::encodeFusedImpl(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> queries, id<MTLBuffer> vectors,
                                    id<MTLBuffer> vecNorms, id<MTLBuffer> queryNormsBuf, id<MTLBuffer> outDistances,
                                    id<MTLBuffer> outIndices, size_t nq, size_t nv, size_t d, size_t k,
                                    faiss::MetricType metric) {

    FusedTopkParams params{(uint32_t)nq, (uint32_t)nv, (uint32_t)d, (uint32_t)k};

    id<MTLComputePipelineState> pso;
    if (metric == faiss::METRIC_L2) {
        pso = vectorsFloat16_ ? fusedL2TopkMinF16StoragePipeline_ : fusedL2TopkMinPipeline_;
    } else {
        pso = vectorsFloat16_ ? fusedIPTopkMaxF16StoragePipeline_ : fusedIPTopkMaxPipeline_;
    }

    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:queries offset:0 atIndex:0];
    [enc setBuffer:vectors offset:0 atIndex:1];
    if (metric == faiss::METRIC_L2) {
        [enc setBuffer:queryNormsBuf offset:0 atIndex:2];
        [enc setBuffer:vecNorms offset:0 atIndex:3];
    }
    [enc setBuffer:outDistances offset:0 atIndex:4];
    [enc setBuffer:outIndices offset:0 atIndex:5];
    [enc setBytes:&params length:sizeof(params) atIndex:6];

    // Threadgroup memory: shared_query, partial_dist, partial_idx
    [enc setThreadgroupMemoryLength:d * sizeof(float) atIndex:0];
    [enc setThreadgroupMemoryLength:4 * 32 * sizeof(float) atIndex:1];
    [enc setThreadgroupMemoryLength:4 * 32 * sizeof(int32_t) atIndex:2];

    [enc dispatchThreadgroups:MTLSizeMake(nq, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    [enc endEncoding];
}

} // namespace faiss_metal
