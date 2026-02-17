#import <Foundation/Foundation.h>
#import <faiss-metal/StandardMetalResources.h>
#import <faiss-metal/MetalDeviceCapabilities.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// assert() is disabled by -DNDEBUG; use FATAL_CHECK for real runtime checks
#define FATAL_CHECK(cond, msg)                                                                                         \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            printf("FATAL: %s\n", msg);                                                                                \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

// Internal headers for direct testing
#include "../src/MetalL2Norm.h"
#include "../src/MetalDistance.h"

using namespace faiss_metal;

static void test_l2_norm() {
    printf("test_l2_norm... ");

    auto res = std::make_shared<StandardMetalResources>();
    MetalL2Norm norm(res.get());

    const size_t n = 100;
    const size_t d = 128;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Generate random vectors
    std::vector<float> data(n * d);
    for (auto &v : data)
        v = dist(rng);

    // Compute on Metal
    id<MTLDevice> device = res->getDevice();
    id<MTLBuffer> inputBuf = [device newBufferWithBytes:data.data()
                                                 length:n * d * sizeof(float)
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuf = [device newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];

    norm.compute(inputBuf, outputBuf, n, d, res->getDefaultCommandQueue());

    float *metalNorms = (float *)[outputBuf contents];

    // Compute reference on CPU
    for (size_t i = 0; i < n; i++) {
        float cpuNorm = 0.0f;
        for (size_t j = 0; j < d; j++) {
            float v = data[i * d + j];
            cpuNorm += v * v;
        }
        float diff = std::abs(metalNorms[i] - cpuNorm);
        FATAL_CHECK(diff < 1e-3f, "L2 norm mismatch");
    }

    printf("PASS\n");
}

static void test_l2_norm_large_dim() {
    printf("test_l2_norm_large_dim... ");

    auto res = std::make_shared<StandardMetalResources>();
    MetalL2Norm norm(res.get());

    const size_t n = 10;
    const size_t d = 1536; // large dimension

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> data(n * d);
    for (auto &v : data)
        v = dist(rng);

    id<MTLDevice> device = res->getDevice();
    id<MTLBuffer> inputBuf = [device newBufferWithBytes:data.data()
                                                 length:n * d * sizeof(float)
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuf = [device newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];

    norm.compute(inputBuf, outputBuf, n, d, res->getDefaultCommandQueue());

    float *metalNorms = (float *)[outputBuf contents];

    for (size_t i = 0; i < n; i++) {
        float cpuNorm = 0.0f;
        for (size_t j = 0; j < d; j++) {
            float v = data[i * d + j];
            cpuNorm += v * v;
        }
        float relDiff = std::abs(metalNorms[i] - cpuNorm) / std::max(cpuNorm, 1e-6f);
        FATAL_CHECK(relDiff < 1e-4f, "L2 norm (large dim) mismatch");
    }

    printf("PASS\n");
}

static void test_l2_distance() {
    printf("test_l2_distance... ");

    auto res = std::make_shared<StandardMetalResources>();
    MetalDistance distCalc(res.get());
    MetalL2Norm normCalc(res.get());

    const size_t nq = 5;
    const size_t nv = 50;
    const size_t d = 64;

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> queries(nq * d);
    std::vector<float> vectors(nv * d);
    for (auto &v : queries)
        v = dist(rng);
    for (auto &v : vectors)
        v = dist(rng);

    id<MTLDevice> device = res->getDevice();
    id<MTLCommandQueue> queue = res->getDefaultCommandQueue();

    id<MTLBuffer> queryBuf = [device newBufferWithBytes:queries.data()
                                                 length:nq * d * sizeof(float)
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> vecBuf = [device newBufferWithBytes:vectors.data()
                                               length:nv * d * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> normBuf = [device newBufferWithLength:nv * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> distBuf = [device newBufferWithLength:nq * nv * sizeof(float) options:MTLResourceStorageModeShared];

    // Precompute vector norms
    normCalc.compute(vecBuf, normBuf, nv, d, queue);

    // Compute L2 distances
    distCalc.compute(queryBuf, vecBuf, normBuf, distBuf, nq, nv, d, faiss::METRIC_L2, queue);

    float *metalDist = (float *)[distBuf contents];

    // Reference CPU computation
    for (size_t i = 0; i < nq; i++) {
        for (size_t j = 0; j < nv; j++) {
            float cpuDist = 0.0f;
            for (size_t k = 0; k < d; k++) {
                float diff = queries[i * d + k] - vectors[j * d + k];
                cpuDist += diff * diff;
            }
            float absDiff = std::abs(metalDist[i * nv + j] - cpuDist);
            float relDiff = absDiff / std::max(cpuDist, 1e-6f);
            // FP16 GEMM path (M2+) has ~1e-2 relative error; FP32 MPS path ~1e-4
            FATAL_CHECK(relDiff < 5e-2f, "L2 distance mismatch");
        }
    }

    printf("PASS\n");
}

static void test_ip_distance() {
    printf("test_ip_distance... ");

    auto res = std::make_shared<StandardMetalResources>();
    MetalDistance distCalc(res.get());

    const size_t nq = 5;
    const size_t nv = 50;
    const size_t d = 64;

    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> queries(nq * d);
    std::vector<float> vectors(nv * d);
    for (auto &v : queries)
        v = dist(rng);
    for (auto &v : vectors)
        v = dist(rng);

    id<MTLDevice> device = res->getDevice();

    id<MTLBuffer> queryBuf = [device newBufferWithBytes:queries.data()
                                                 length:nq * d * sizeof(float)
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> vecBuf = [device newBufferWithBytes:vectors.data()
                                               length:nv * d * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> distBuf = [device newBufferWithLength:nq * nv * sizeof(float) options:MTLResourceStorageModeShared];

    distCalc.compute(queryBuf, vecBuf, nil, distBuf, nq, nv, d, faiss::METRIC_INNER_PRODUCT,
                     res->getDefaultCommandQueue());

    float *metalDist = (float *)[distBuf contents];

    for (size_t i = 0; i < nq; i++) {
        for (size_t j = 0; j < nv; j++) {
            float cpuDist = 0.0f;
            for (size_t k = 0; k < d; k++) {
                cpuDist += queries[i * d + k] * vectors[j * d + k];
            }
            float absDiff = std::abs(metalDist[i * nv + j] - cpuDist);
            float relDiff = absDiff / std::max(std::abs(cpuDist), 1e-6f);
            // FP16 GEMM path (M2+) has ~1e-2 relative error; FP32 MPS path ~1e-4
            FATAL_CHECK(relDiff < 5e-2f, "IP distance mismatch");
        }
    }

    // Verify top-1 correctness: best IP result must match CPU
    for (size_t i = 0; i < nq; i++) {
        size_t metalBest = 0, cpuBest = 0;
        float metalBestD = metalDist[i * nv];
        float cpuBestD = 0.0f;
        for (size_t k = 0; k < d; k++)
            cpuBestD += queries[i * d + k] * vectors[0];
        for (size_t j = 1; j < nv; j++) {
            if (metalDist[i * nv + j] > metalBestD) {
                metalBestD = metalDist[i * nv + j];
                metalBest = j;
            }
            float cpuD = 0.0f;
            for (size_t k = 0; k < d; k++)
                cpuD += queries[i * d + k] * vectors[j * d + k];
            if (cpuD > cpuBestD) {
                cpuBestD = cpuD;
                cpuBest = j;
            }
        }
        FATAL_CHECK(metalBest == cpuBest, "IP top-1 label mismatch");
    }

    printf("PASS\n");
}

/// Test forced MPS path on M2+ hardware (exercises both code paths)
static void test_forced_mps_l2() {
    printf("test_forced_mps_l2... ");

    auto res = std::make_shared<StandardMetalResources>();
    MetalDistance distCalc(res.get());
    MetalDistance distCalcMPS(res.get());
    distCalcMPS.setForceMPS(true);
    MetalL2Norm normCalc(res.get());

    const size_t nq = 5;
    const size_t nv = 50;
    const size_t d = 64;

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> queries(nq * d);
    std::vector<float> vectors(nv * d);
    for (auto &v : queries)
        v = dist(rng);
    for (auto &v : vectors)
        v = dist(rng);

    id<MTLDevice> device = res->getDevice();
    id<MTLCommandQueue> queue = res->getDefaultCommandQueue();

    id<MTLBuffer> queryBuf = [device newBufferWithBytes:queries.data()
                                                 length:nq * d * sizeof(float)
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> vecBuf = [device newBufferWithBytes:vectors.data()
                                               length:nv * d * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> normBuf = [device newBufferWithLength:nv * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> distBuf1 = [device newBufferWithLength:nq * nv * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> distBuf2 = [device newBufferWithLength:nq * nv * sizeof(float) options:MTLResourceStorageModeShared];

    normCalc.compute(vecBuf, normBuf, nv, d, queue);

    // Default path (simdgroup on M2+, MPS on M1)
    distCalc.compute(queryBuf, vecBuf, normBuf, distBuf1, nq, nv, d, faiss::METRIC_L2, queue);
    // Forced MPS path
    distCalcMPS.compute(queryBuf, vecBuf, normBuf, distBuf2, nq, nv, d, faiss::METRIC_L2, queue);

    float *d1 = (float *)[distBuf1 contents];
    float *d2 = (float *)[distBuf2 contents];

    // Both paths should agree within tolerance
    for (size_t i = 0; i < nq * nv; i++) {
        float relDiff = std::abs(d1[i] - d2[i]) / std::max(std::abs(d1[i]), 1e-6f);
        FATAL_CHECK(relDiff < 5e-2f, "Default vs forced-MPS L2 mismatch");
    }

    printf("PASS\n");
}

static void test_forced_mps_ip() {
    printf("test_forced_mps_ip... ");

    auto res = std::make_shared<StandardMetalResources>();
    MetalDistance distCalc(res.get());
    MetalDistance distCalcMPS(res.get());
    distCalcMPS.setForceMPS(true);

    const size_t nq = 5;
    const size_t nv = 50;
    const size_t d = 64;

    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> queries(nq * d);
    std::vector<float> vectors(nv * d);
    for (auto &v : queries)
        v = dist(rng);
    for (auto &v : vectors)
        v = dist(rng);

    id<MTLDevice> device = res->getDevice();
    id<MTLCommandQueue> queue = res->getDefaultCommandQueue();

    id<MTLBuffer> queryBuf = [device newBufferWithBytes:queries.data()
                                                 length:nq * d * sizeof(float)
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> vecBuf = [device newBufferWithBytes:vectors.data()
                                               length:nv * d * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> distBuf1 = [device newBufferWithLength:nq * nv * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> distBuf2 = [device newBufferWithLength:nq * nv * sizeof(float) options:MTLResourceStorageModeShared];

    distCalc.compute(queryBuf, vecBuf, nil, distBuf1, nq, nv, d, faiss::METRIC_INNER_PRODUCT, queue);
    distCalcMPS.compute(queryBuf, vecBuf, nil, distBuf2, nq, nv, d, faiss::METRIC_INNER_PRODUCT, queue);

    float *d1 = (float *)[distBuf1 contents];
    float *d2 = (float *)[distBuf2 contents];

    for (size_t i = 0; i < nq * nv; i++) {
        float relDiff = std::abs(d1[i] - d2[i]) / std::max(std::abs(d1[i]), 1e-6f);
        FATAL_CHECK(relDiff < 5e-2f, "Default vs forced-MPS IP mismatch");
    }

    printf("PASS\n");
}

int main() {
    @autoreleasepool {
        auto res = std::make_shared<StandardMetalResources>();
        const auto &caps = res->getCapabilities();
        printf("=== Metal Distance Tests ===\n");
        printf("%s\n", faiss_metal::describeCapabilities(caps).c_str());

        test_l2_norm();
        test_l2_norm_large_dim();
        test_l2_distance();
        test_ip_distance();
        test_forced_mps_l2();
        test_forced_mps_ip();
        printf("All distance tests passed!\n");
    }
    return 0;
}
