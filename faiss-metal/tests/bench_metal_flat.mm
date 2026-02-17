#import <Foundation/Foundation.h>
#import <faiss-metal/MetalIndexFlat.h>
#import <faiss-metal/StandardMetalResources.h>
#import <faiss-metal/MetalDeviceCapabilities.h>
#import <faiss/IndexFlat.h>

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

static std::shared_ptr<faiss_metal::StandardMetalResources> g_res;

static void bench(const char *label, size_t nv, size_t nq, size_t d, size_t k, int warmup_iters = 3,
                  int bench_iters = 10) {

    printf("--- %s: nv=%zu nq=%zu d=%zu k=%zu ---\n", label, nv, nq, d, k);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vectors(nv * d);
    std::vector<float> queries(nq * d);
    for (auto &v : vectors)
        v = dist(rng);
    for (auto &v : queries)
        v = dist(rng);

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);

    // --- CPU FAISS ---
    {
        faiss::IndexFlatL2 cpu_index(d);
        cpu_index.add(nv, vectors.data());

        for (int i = 0; i < warmup_iters; i++)
            cpu_index.search(nq, queries.data(), k, distances.data(), labels.data());

        auto start = Clock::now();
        for (int i = 0; i < bench_iters; i++)
            cpu_index.search(nq, queries.data(), k, distances.data(), labels.data());
        auto end = Clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_ms = total_ms / bench_iters;
        double qps = (nq * bench_iters) / (total_ms / 1000.0);
        printf("  CPU:   %.2f ms/search  (%.0f QPS)\n", avg_ms, qps);
    }

    // --- Metal FP32 ---
    {
        faiss_metal::MetalIndexFlat metal_index(g_res, d, faiss::METRIC_L2);
        metal_index.add(nv, vectors.data());

        for (int i = 0; i < warmup_iters; i++)
            metal_index.search(nq, queries.data(), k, distances.data(), labels.data());

        auto start = Clock::now();
        for (int i = 0; i < bench_iters; i++)
            metal_index.search(nq, queries.data(), k, distances.data(), labels.data());
        auto end = Clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_ms = total_ms / bench_iters;
        double qps = (nq * bench_iters) / (total_ms / 1000.0);
        printf("  Metal:     %.2f ms/search  (%.0f QPS)\n", avg_ms, qps);
    }

    // --- Metal FP16 storage ---
    {
        faiss_metal::MetalIndexFlat metal_f16(g_res, d, faiss::METRIC_L2,
                                              /*useFloat16Storage=*/true);
        metal_f16.add(nv, vectors.data());

        for (int i = 0; i < warmup_iters; i++)
            metal_f16.search(nq, queries.data(), k, distances.data(), labels.data());

        auto start = Clock::now();
        for (int i = 0; i < bench_iters; i++)
            metal_f16.search(nq, queries.data(), k, distances.data(), labels.data());
        auto end = Clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_ms = total_ms / bench_iters;
        double qps = (nq * bench_iters) / (total_ms / 1000.0);
        printf("  Metal F16: %.2f ms/search  (%.0f QPS)\n", avg_ms, qps);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    @autoreleasepool {
        g_res = std::make_shared<faiss_metal::StandardMetalResources>();
        const auto &caps = g_res->getCapabilities();

        printf("=== faiss-metal Benchmark ===\n");
        printf("%s", faiss_metal::describeCapabilities(caps).c_str());

        // Show which code paths will be used
        printf("\nActive code paths:\n");
        printf("  GEMM: %s\n", caps.hasFastFP16 ? "simdgroup_matrix (FP16)" : "MPS (FP32)");
        printf("  L2 norm: %s\n", caps.hasFastFP16 ? "FP16 fast path" : "FP32");
        printf("  Threadgroup: %s\n", caps.hasDynamicThreadgroupMemory ? "dynamic caching (M3+)" : "standard");
        printf("\n");

        // Small dataset, low dimensions
        bench("small-32d", 10000, 100, 32, 10);
        bench("small-128d", 10000, 100, 128, 10);

        // Medium dataset
        bench("medium-128d", 100000, 100, 128, 10);
        bench("medium-768d", 100000, 10, 768, 10);

        // Large dimensions (embedding models)
        bench("large-1536d", 10000, 10, 1536, 10);

        // High k
        bench("highk-128d", 10000, 10, 128, 100);

        // Single query (latency test)
        bench("latency-128d", 100000, 1, 128, 10, 10, 100);

        printf("Benchmark complete.\n");
    }
    return 0;
}
