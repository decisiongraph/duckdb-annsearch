#import "MetalL2Norm.h"
#import <faiss-metal/MetalResources.h>
#import <faiss-metal/MetalDeviceCapabilities.h>
#include <faiss/impl/FaissAssert.h>
#include <algorithm>

namespace faiss_metal {

MetalL2Norm::MetalL2Norm(MetalResources *resources) {
    id<MTLLibrary> lib = resources->getMetalLibrary();
    id<MTLDevice> device = resources->getDevice();
    NSError *error = nil;

    const auto &caps = resources->getCapabilities();
    useFP16_ = caps.hasFastFP16; // M2+ has 2x FP16 throughput

    // Always build FP32 pipelines (baseline)
    id<MTLFunction> fn = [lib newFunctionWithName:@"l2_norm"];
    FAISS_THROW_IF_NOT_MSG(fn, "Metal function 'l2_norm' not found");
    pipeline_ = [device newComputePipelineStateWithFunction:fn error:&error];
    FAISS_THROW_IF_NOT_MSG(pipeline_, "Failed to create l2_norm pipeline");

    id<MTLFunction> fnLarge = [lib newFunctionWithName:@"l2_norm_large"];
    FAISS_THROW_IF_NOT_MSG(fnLarge, "Metal function 'l2_norm_large' not found");
    pipelineLarge_ = [device newComputePipelineStateWithFunction:fnLarge error:&error];
    FAISS_THROW_IF_NOT_MSG(pipelineLarge_, "Failed to create l2_norm_large pipeline");

    // Build FP16 pipelines if available
    if (useFP16_) {
        id<MTLFunction> fnF16 = [lib newFunctionWithName:@"l2_norm_f16"];
        FAISS_THROW_IF_NOT_MSG(fnF16, "Metal function 'l2_norm_f16' not found");
        pipelineF16_ = [device newComputePipelineStateWithFunction:fnF16 error:&error];
        FAISS_THROW_IF_NOT_MSG(pipelineF16_, "Failed to create l2_norm_f16 pipeline");

        id<MTLFunction> fnF16Large = [lib newFunctionWithName:@"l2_norm_f16_large"];
        FAISS_THROW_IF_NOT_MSG(fnF16Large, "Metal function 'l2_norm_f16_large' not found");
        pipelineF16Large_ = [device newComputePipelineStateWithFunction:fnF16Large error:&error];
        FAISS_THROW_IF_NOT_MSG(pipelineF16Large_, "Failed to create l2_norm_f16_large pipeline");
    }
}

void MetalL2Norm::encode(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> input, id<MTLBuffer> output, size_t n, size_t d) {

    if (n == 0)
        return;

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    uint32_t dim = (uint32_t)d;

    if (d <= 1024) {
        auto pso = useFP16_ ? pipelineF16_ : pipeline_;
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:output offset:0 atIndex:1];
        [encoder setBytes:&dim length:sizeof(dim) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    } else {
        size_t threadsPerGroup = std::min((size_t)1024, ((d + 31) / 32) * 32);
        auto pso = useFP16_ ? pipelineF16Large_ : pipelineLarge_;
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:output offset:0 atIndex:1];
        [encoder setBytes:&dim length:sizeof(dim) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
    }

    [encoder endEncoding];
}

void MetalL2Norm::compute(id<MTLBuffer> input, id<MTLBuffer> output, size_t n, size_t d, id<MTLCommandQueue> queue) {

    if (n == 0)
        return;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    encode(cmdBuf, input, output, n, d);
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

} // namespace faiss_metal
