#import <faiss-metal/MetalIndexFlat.h>
#import <faiss-metal/MetalResources.h>
#import <faiss-metal/StandardMetalResources.h>
#import <faiss/IndexFlat.h>
#import "MetalDistance.h"
#import "MetalL2Norm.h"
#import "MetalSelect.h"
#import "MetalTensor.h"
#include <faiss/impl/FaissAssert.h>
#include <dispatch/dispatch.h>
#include <cstring>

namespace faiss_metal {

// --- MetalSearchToken ---

struct MetalSearchToken::Impl {
    id<MTLCommandBuffer> cmdBuf;
    id<MTLBuffer> outDistBuf;
    id<MTLBuffer> outIdxBuf;
    // Also hold refs to per-call buffers so they stay alive until wait()
    id<MTLBuffer> queryBuf;
    id<MTLBuffer> distBuf;      // may be nil (fused path)
    id<MTLBuffer> queryNormBuf; // may be nil (IP metric)

    float *distances;
    faiss::idx_t *labels;
    faiss::idx_t n;
    faiss::idx_t k;
    faiss::idx_t effective_k;
    faiss::MetricType metric_type;
    bool waited = false;
};

MetalSearchToken::MetalSearchToken(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
}

MetalSearchToken::MetalSearchToken(MetalSearchToken &&) noexcept = default;
MetalSearchToken &MetalSearchToken::operator=(MetalSearchToken &&) noexcept = default;

MetalSearchToken::~MetalSearchToken() {
    // If caller never called wait(), block now to ensure GPU is done
    // before releasing buffers.
    if (impl_ && !impl_->waited) {
        [impl_->cmdBuf waitUntilCompleted];
    }
}

void MetalSearchToken::wait() {
    FAISS_THROW_IF_NOT_MSG(impl_, "MetalSearchToken: moved-from token");
    if (impl_->waited)
        return;

    [impl_->cmdBuf waitUntilCompleted];
    impl_->waited = true;

    // Copy results back (same logic as sync search)
    float *outDistPtr = (float *)[impl_->outDistBuf contents];
    int32_t *outIdxPtr = (int32_t *)[impl_->outIdxBuf contents];

    float sentinel_dist = (impl_->metric_type == faiss::METRIC_L2) ? INFINITY : -INFINITY;

    for (faiss::idx_t i = 0; i < impl_->n; i++) {
        for (faiss::idx_t j = 0; j < impl_->effective_k; j++) {
            impl_->distances[i * impl_->k + j] = outDistPtr[i * impl_->effective_k + j];
            impl_->labels[i * impl_->k + j] = (faiss::idx_t)outIdxPtr[i * impl_->effective_k + j];
        }
        for (faiss::idx_t j = impl_->effective_k; j < impl_->k; j++) {
            impl_->distances[i * impl_->k + j] = sentinel_dist;
            impl_->labels[i * impl_->k + j] = -1;
        }
    }
}

bool MetalSearchToken::isReady() const {
    FAISS_THROW_IF_NOT_MSG(impl_, "MetalSearchToken: moved-from token");
    if (impl_->waited)
        return true;
    return [impl_->cmdBuf status] == MTLCommandBufferStatusCompleted;
}

// --- MetalIndexFlat ---

struct MetalIndexFlat::Impl {
    std::shared_ptr<MetalResources> resources;
    StorageType storageType = StorageType::Float32;
    bool useFloat16Storage = false;
    bool useBFloat16Storage = false;

    // FP32 storage (default)
    MetalTensor<float, 2> vectors; // (ntotal x d) stored vectors

    // FP16 storage (when useFloat16Storage == true)
    id<MTLBuffer> vectorsF16Buf = nil;
    size_t vectorsF16Capacity = 0; // capacity in number of vectors

    MetalTensor<float, 1> norms; // (ntotal) precomputed ||v||^2, always FP32
    size_t capacity = 0;         // allocated capacity in vectors (FP32 mode)

    std::unique_ptr<MetalDistance> distance;
    std::unique_ptr<MetalL2Norm> l2norm;
    std::unique_ptr<MetalSelect> selector;

    // Scratch buffers reused across search() calls to avoid per-call allocation
    id<MTLBuffer> scratchDistBuf = nil;
    size_t scratchDistBytes = 0;
    id<MTLBuffer> scratchOutDistBuf = nil;
    id<MTLBuffer> scratchOutIdxBuf = nil;
    size_t scratchTopkElems = 0;
    id<MTLBuffer> scratchQueryNormBuf = nil;
    size_t scratchQueryNormElems = 0;

    id<MTLBuffer> getScratchDist(id<MTLDevice> device, size_t bytes) {
        if (bytes > scratchDistBytes) {
            scratchDistBytes = bytes;
            // Private: GPU-only intermediate (GEMM out → Select in), no CPU access needed
            scratchDistBuf = [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        }
        return scratchDistBuf;
    }

    void getScratchTopk(id<MTLDevice> device, size_t elems, id<MTLBuffer> &outDist, id<MTLBuffer> &outIdx) {
        if (elems > scratchTopkElems) {
            scratchTopkElems = elems;
            scratchOutDistBuf = [device newBufferWithLength:elems * sizeof(float) options:MTLResourceStorageModeShared];
            scratchOutIdxBuf = [device newBufferWithLength:elems * sizeof(int32_t)
                                                   options:MTLResourceStorageModeShared];
        }
        outDist = scratchOutDistBuf;
        outIdx = scratchOutIdxBuf;
    }

    id<MTLBuffer> getScratchQueryNorms(id<MTLDevice> device, size_t nq) {
        if (nq > scratchQueryNormElems) {
            scratchQueryNormElems = nq;
            // Private: GPU-only intermediate (l2_norm out → GEMM in), no CPU access needed
            scratchQueryNormBuf = [device newBufferWithLength:nq * sizeof(float) options:MTLResourceStorageModePrivate];
        }
        return scratchQueryNormBuf;
    }

    id<MTLBuffer> getVectorsBuf() const {
        return (useFloat16Storage || useBFloat16Storage) ? vectorsF16Buf : vectors.buffer();
    }

    Impl(std::shared_ptr<MetalResources> res, StorageType storage)
        : resources(std::move(res)), storageType(storage), useFloat16Storage(storage == StorageType::Float16),
          useBFloat16Storage(storage == StorageType::BFloat16),
          distance(std::make_unique<MetalDistance>(resources.get())),
          l2norm(std::make_unique<MetalL2Norm>(resources.get())),
          selector(std::make_unique<MetalSelect>(resources.get())) {
        distance->setVectorsFloat16(useFloat16Storage);
        distance->setVectorsBFloat16(useBFloat16Storage);
    }
};

MetalIndexFlat::MetalIndexFlat(std::shared_ptr<MetalResources> resources, int d, faiss::MetricType metric,
                               bool useFloat16Storage)
    : faiss::Index(d, metric),
      impl_(std::make_unique<Impl>(std::move(resources),
                                   useFloat16Storage ? StorageType::Float16 : StorageType::Float32)) {
    is_trained = true;
}

MetalIndexFlat::MetalIndexFlat(std::shared_ptr<MetalResources> resources, int d, faiss::MetricType metric,
                               StorageType storage)
    : faiss::Index(d, metric), impl_(std::make_unique<Impl>(std::move(resources), storage)) {
    is_trained = true;
}

MetalIndexFlat::~MetalIndexFlat() = default;

void MetalIndexFlat::add(faiss::idx_t n, const float *x) {
    if (n == 0)
        return;

    id<MTLDevice> device = impl_->resources->getDevice();
    id<MTLCommandQueue> queue = impl_->resources->getDefaultCommandQueue();
    size_t newTotal = ntotal + n;

    if (impl_->useFloat16Storage || impl_->useBFloat16Storage) {
        // --- Reduced-precision storage path (FP16 or BF16) ---
        size_t f16Cap = impl_->vectorsF16Capacity;
        if (newTotal > f16Cap) {
            size_t newCap = std::max(newTotal, f16Cap * 2);
            newCap = std::max(newCap, (size_t)1024);

            id<MTLBuffer> newBuf = [device newBufferWithLength:newCap * d * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
            MetalTensor<float, 1> newNorms(device, {newCap});

            if (ntotal > 0) {
                memcpy([newBuf contents], [impl_->vectorsF16Buf contents], ntotal * d * sizeof(uint16_t));
                memcpy(newNorms.data(), impl_->norms.data(), ntotal * sizeof(float));
            }

            impl_->vectorsF16Buf = newBuf;
            impl_->vectorsF16Capacity = newCap;
            impl_->norms = std::move(newNorms);

            // Register new buffers for proactive GPU residency
            if (auto *std_res = dynamic_cast<StandardMetalResources *>(impl_->resources.get())) {
                std_res->registerForResidency(impl_->vectorsF16Buf);
                std_res->registerForResidency(impl_->norms.buffer());
            }
        }

        // Convert float32 → reduced precision and store
        // Parallelize with GCD for large adds (>100K elements)
        size_t count = (size_t)n * d;
        if (impl_->useBFloat16Storage) {
            // BFloat16: truncate float32 mantissa (keep top 7 bits + 8-bit exponent)
            uint16_t *dst = (uint16_t *)[impl_->vectorsF16Buf contents] + ntotal * d;
            const uint32_t *src = (const uint32_t *)x;
            if (count > 100000) {
                dispatch_apply(count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
                  uint32_t bits = src[i];
                  uint32_t lsb = (bits >> 16) & 1;
                  uint32_t rounding = 0x7FFF + lsb;
                  dst[i] = (uint16_t)((bits + rounding) >> 16);
                });
            } else {
                for (size_t i = 0; i < count; i++) {
                    uint32_t bits = src[i];
                    uint32_t lsb = (bits >> 16) & 1;
                    uint32_t rounding = 0x7FFF + lsb;
                    dst[i] = (uint16_t)((bits + rounding) >> 16);
                }
            }
        } else {
            __fp16 *dst = (__fp16 *)[impl_->vectorsF16Buf contents] + ntotal * d;
            if (count > 100000) {
                dispatch_apply(count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
                  dst[i] = (__fp16)x[i];
                });
            } else {
                for (size_t i = 0; i < count; i++) {
                    dst[i] = (__fp16)x[i];
                }
            }
        }

        // Compute norms from float32 input (more accurate than from half)
        if (metric_type == faiss::METRIC_L2) {
            id<MTLBuffer> newVecBuf = [device newBufferWithBytes:x
                                                          length:n * d * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
            id<MTLBuffer> tmpNorms = [device newBufferWithLength:n * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
            impl_->l2norm->compute(newVecBuf, tmpNorms, n, d, queue);
            memcpy(impl_->norms.data() + ntotal, [tmpNorms contents], n * sizeof(float));
        }
    } else {
        // --- FP32 storage path (original) ---
        if (newTotal > impl_->capacity) {
            size_t newCap = std::max(newTotal, impl_->capacity * 2);
            newCap = std::max(newCap, (size_t)1024);

            MetalTensor<float, 2> newVectors(device, {newCap, (size_t)d});
            MetalTensor<float, 1> newNorms(device, {newCap});

            if (ntotal > 0) {
                memcpy(newVectors.data(), impl_->vectors.data(), ntotal * d * sizeof(float));
                memcpy(newNorms.data(), impl_->norms.data(), ntotal * sizeof(float));
            }

            impl_->vectors = std::move(newVectors);
            impl_->norms = std::move(newNorms);
            impl_->capacity = newCap;

            // Register new buffers for proactive GPU residency
            if (auto *std_res = dynamic_cast<StandardMetalResources *>(impl_->resources.get())) {
                std_res->registerForResidency(impl_->vectors.buffer());
                std_res->registerForResidency(impl_->norms.buffer());
            }
        }

        memcpy(impl_->vectors.data() + ntotal * d, x, n * d * sizeof(float));

        if (metric_type == faiss::METRIC_L2) {
            id<MTLBuffer> newVecBuf = [device newBufferWithBytes:(impl_->vectors.data() + ntotal * d)
                                                          length:n * d * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
            id<MTLBuffer> tmpNorms = [device newBufferWithLength:n * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
            impl_->l2norm->compute(newVecBuf, tmpNorms, n, d, queue);
            memcpy(impl_->norms.data() + ntotal, [tmpNorms contents], n * sizeof(float));
        }
    }

    ntotal = newTotal;
}

void MetalIndexFlat::search(faiss::idx_t n, const float *x, faiss::idx_t k, float *distances, faiss::idx_t *labels,
                            const faiss::SearchParameters *params) const {

    FAISS_THROW_IF_NOT_MSG(k > 0, "k must be > 0");

    // Metal shaders use int32_t indices — cap at 2^31-1 vectors
    FAISS_THROW_IF_NOT_MSG(ntotal <= (faiss::idx_t)INT32_MAX,
                           "MetalIndexFlat supports at most 2^31-1 vectors (int32 indices)");

    if (n == 0 || ntotal == 0) {
        for (faiss::idx_t i = 0; i < n * k; i++) {
            distances[i] = (metric_type == faiss::METRIC_L2) ? INFINITY : -INFINITY;
            labels[i] = -1;
        }
        return;
    }

    // Clamp k to ntotal (matches FAISS CPU behavior)
    faiss::idx_t effective_k = std::min(k, (faiss::idx_t)ntotal);

    id<MTLDevice> device = impl_->resources->getDevice();
    id<MTLCommandQueue> queue = impl_->resources->getDefaultCommandQueue();

    id<MTLBuffer> queryBuf = [device newBufferWithBytes:x
                                                 length:n * d * sizeof(float)
                                                options:MTLResourceStorageModeShared];

    id<MTLBuffer> vecBuf = impl_->getVectorsBuf();
    id<MTLBuffer> normBuf = impl_->norms.buffer();

    id<MTLBuffer> outDistBuf, outIdxBuf;
    impl_->getScratchTopk(device, n * effective_k, outDistBuf, outIdxBuf);

    id<MTLBuffer> queryNormsBuf = nil;
    if (metric_type == faiss::METRIC_L2) {
        queryNormsBuf = impl_->getScratchQueryNorms(device, n);
    }

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];

    // Try fused distance+topk path (avoids nq*nv intermediate buffer).
    // Falls back to separate GEMM + select for large nq or k > 32.
    bool usedFused = impl_->distance->encodeFused(cmdBuf, queryBuf, vecBuf, normBuf, queryNormsBuf, outDistBuf,
                                                  outIdxBuf, n, ntotal, d, effective_k, metric_type);

    if (!usedFused) {
        // Standard path: full distance matrix + separate top-k selection
        size_t distBytes = n * ntotal * sizeof(float);
        id<MTLBuffer> distBuf = impl_->getScratchDist(device, distBytes);

        impl_->distance->encode(cmdBuf, queryBuf, vecBuf, normBuf, queryNormsBuf, distBuf, n, ntotal, d, metric_type);
        impl_->selector->encode(cmdBuf, distBuf, outDistBuf, outIdxBuf, n, ntotal, effective_k, metric_type);
    }

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    // Copy results back
    float *outDistPtr = (float *)[outDistBuf contents];
    int32_t *outIdxPtr = (int32_t *)[outIdxBuf contents];

    float sentinel_dist = (metric_type == faiss::METRIC_L2) ? INFINITY : -INFINITY;

    for (faiss::idx_t i = 0; i < n; i++) {
        // Copy effective_k results
        for (faiss::idx_t j = 0; j < effective_k; j++) {
            distances[i * k + j] = outDistPtr[i * effective_k + j];
            labels[i * k + j] = (faiss::idx_t)outIdxPtr[i * effective_k + j];
        }
        // Fill remaining slots if k > ntotal
        for (faiss::idx_t j = effective_k; j < k; j++) {
            distances[i * k + j] = sentinel_dist;
            labels[i * k + j] = -1;
        }
    }
}

MetalSearchToken MetalIndexFlat::searchAsync(faiss::idx_t n, const float *x, faiss::idx_t k, float *distances,
                                             faiss::idx_t *labels, const faiss::SearchParameters *params) const {

    FAISS_THROW_IF_NOT_MSG(k > 0, "k must be > 0");
    FAISS_THROW_IF_NOT_MSG(ntotal <= (faiss::idx_t)INT32_MAX,
                           "MetalIndexFlat supports at most 2^31-1 vectors (int32 indices)");

    auto tokenImpl = std::make_unique<MetalSearchToken::Impl>();
    tokenImpl->distances = distances;
    tokenImpl->labels = labels;
    tokenImpl->n = n;
    tokenImpl->k = k;
    tokenImpl->metric_type = metric_type;

    if (n == 0 || ntotal == 0) {
        // No GPU work — fill sentinel results immediately
        float sentinel = (metric_type == faiss::METRIC_L2) ? INFINITY : -INFINITY;
        for (faiss::idx_t i = 0; i < n * k; i++) {
            distances[i] = sentinel;
            labels[i] = -1;
        }
        tokenImpl->effective_k = 0;
        tokenImpl->waited = true;
        return MetalSearchToken(std::move(tokenImpl));
    }

    faiss::idx_t effective_k = std::min(k, (faiss::idx_t)ntotal);
    tokenImpl->effective_k = effective_k;

    id<MTLDevice> device = impl_->resources->getDevice();
    id<MTLCommandQueue> queue = impl_->resources->getDefaultCommandQueue();

    // Per-call buffers (not shared scratch)
    id<MTLBuffer> queryBuf = [device newBufferWithBytes:x
                                                 length:n * d * sizeof(float)
                                                options:MTLResourceStorageModeShared];
    tokenImpl->queryBuf = queryBuf;

    id<MTLBuffer> vecBuf = impl_->getVectorsBuf();
    id<MTLBuffer> normBuf = impl_->norms.buffer();

    id<MTLBuffer> outDistBuf = [device newBufferWithLength:n * effective_k * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> outIdxBuf = [device newBufferWithLength:n * effective_k * sizeof(int32_t)
                                                  options:MTLResourceStorageModeShared];
    tokenImpl->outDistBuf = outDistBuf;
    tokenImpl->outIdxBuf = outIdxBuf;

    id<MTLBuffer> queryNormsBuf = nil;
    if (metric_type == faiss::METRIC_L2) {
        queryNormsBuf = [device newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModePrivate];
        tokenImpl->queryNormBuf = queryNormsBuf;
    }

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];

    bool usedFused = impl_->distance->encodeFused(cmdBuf, queryBuf, vecBuf, normBuf, queryNormsBuf, outDistBuf,
                                                  outIdxBuf, n, ntotal, d, effective_k, metric_type);

    if (!usedFused) {
        size_t distBytes = n * ntotal * sizeof(float);
        id<MTLBuffer> distBuf = [device newBufferWithLength:distBytes options:MTLResourceStorageModePrivate];
        tokenImpl->distBuf = distBuf;

        impl_->distance->encode(cmdBuf, queryBuf, vecBuf, normBuf, queryNormsBuf, distBuf, n, ntotal, d, metric_type);
        impl_->selector->encode(cmdBuf, distBuf, outDistBuf, outIdxBuf, n, ntotal, effective_k, metric_type);
    }

    [cmdBuf commit];
    tokenImpl->cmdBuf = cmdBuf;

    return MetalSearchToken(std::move(tokenImpl));
}

void MetalIndexFlat::reset() {
    ntotal = 0;
    impl_->capacity = 0;
    impl_->vectors = MetalTensor<float, 2>();
    impl_->vectorsF16Buf = nil;
    impl_->vectorsF16Capacity = 0;
    impl_->norms = MetalTensor<float, 1>();
    impl_->scratchDistBuf = nil;
    impl_->scratchDistBytes = 0;
    impl_->scratchOutDistBuf = nil;
    impl_->scratchOutIdxBuf = nil;
    impl_->scratchTopkElems = 0;
    impl_->scratchQueryNormBuf = nil;
    impl_->scratchQueryNormElems = 0;
}

void MetalIndexFlat::reconstruct(faiss::idx_t key, float *recons) const {
    FAISS_THROW_IF_NOT_MSG(key >= 0 && key < ntotal, "reconstruct: key out of range");
    if (impl_->useBFloat16Storage) {
        // BFloat16 → float32: shift left by 16 bits
        const uint16_t *src = (const uint16_t *)[impl_->vectorsF16Buf contents] + key * d;
        uint32_t *dst = (uint32_t *)recons;
        for (int i = 0; i < d; i++) {
            dst[i] = (uint32_t)src[i] << 16;
        }
    } else if (impl_->useFloat16Storage) {
        const __fp16 *src = (const __fp16 *)[impl_->vectorsF16Buf contents] + key * d;
        for (int i = 0; i < d; i++) {
            recons[i] = (float)src[i];
        }
    } else {
        memcpy(recons, impl_->vectors.data() + key * d, d * sizeof(float));
    }
}

const float *MetalIndexFlat::getVectorsData() const {
    if (impl_->useFloat16Storage || impl_->useBFloat16Storage)
        return nullptr;
    return impl_->vectors.data();
}

bool MetalIndexFlat::isFloat16Storage() const {
    return impl_->useFloat16Storage;
}

bool MetalIndexFlat::isBFloat16Storage() const {
    return impl_->useBFloat16Storage;
}

StorageType MetalIndexFlat::getStorageType() const {
    return impl_->storageType;
}

void MetalIndexFlat::setForceMPS(bool force) {
    impl_->distance->setForceMPS(force);
}

// --- Conversion helpers ---

std::unique_ptr<MetalIndexFlat> index_cpu_to_metal(std::shared_ptr<MetalResources> resources,
                                                   const faiss::IndexFlat *cpu_index) {

    auto metal_index = std::make_unique<MetalIndexFlat>(resources, cpu_index->d, cpu_index->metric_type);

    if (cpu_index->ntotal > 0) {
        const float *data = cpu_index->get_xb();
        metal_index->add(cpu_index->ntotal, data);
    }

    return metal_index;
}

std::unique_ptr<faiss::IndexFlat> index_metal_to_cpu(const MetalIndexFlat *metal_index) {

    auto cpu_index = std::make_unique<faiss::IndexFlat>(metal_index->d, metal_index->metric_type);

    if (metal_index->ntotal > 0) {
        if (metal_index->isFloat16Storage() || metal_index->isBFloat16Storage()) {
            // Reconstruct each vector from reduced precision → FP32
            std::vector<float> tmp(metal_index->d);
            for (faiss::idx_t i = 0; i < metal_index->ntotal; i++) {
                metal_index->reconstruct(i, tmp.data());
                cpu_index->add(1, tmp.data());
            }
        } else {
            cpu_index->add(metal_index->ntotal, metal_index->getVectorsData());
        }
    }

    return cpu_index;
}

} // namespace faiss_metal
