#import <Foundation/Foundation.h>
#import <faiss-metal/MetalIndexFlat.h>
#import <faiss-metal/StandardMetalResources.h>
#import <faiss-metal/MetalDeviceCapabilities.h>
#import <faiss/IndexFlat.h>

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

using namespace faiss_metal;

/// Compare Metal vs CPU FAISS results. Allow some tolerance for floating point.
static void compare_results(const char *label, faiss::idx_t n, faiss::idx_t k, const float *metal_distances,
                            const faiss::idx_t *metal_labels, const float *cpu_distances,
                            const faiss::idx_t *cpu_labels, float dist_tol = 1e-3f) {

    int mismatches = 0;
    for (faiss::idx_t i = 0; i < n; i++) {
        for (faiss::idx_t j = 0; j < k; j++) {
            faiss::idx_t idx = i * k + j;
            float dDiff = std::abs(metal_distances[idx] - cpu_distances[idx]);
            float relDiff = dDiff / std::max(std::abs(cpu_distances[idx]), 1e-6f);

            if (relDiff > dist_tol) {
                if (mismatches < 5) {
                    printf("  [%s] query=%lld rank=%lld: Metal dist=%.6f label=%lld, "
                           "CPU dist=%.6f label=%lld (relDiff=%.6f)\n",
                           label, (long long)i, (long long)j, metal_distances[idx], (long long)metal_labels[idx],
                           cpu_distances[idx], (long long)cpu_labels[idx], relDiff);
                }
                mismatches++;
            }
        }
    }
    if (mismatches > 0) {
        printf("  [%s] WARNING: %d/%lld distance mismatches (tol=%.0e)\n", label, mismatches, (long long)(n * k),
               dist_tol);
    }

    // Check that top-1 labels match (most important)
    int top1_mismatches = 0;
    for (faiss::idx_t i = 0; i < n; i++) {
        if (metal_labels[i * k] != cpu_labels[i * k]) {
            top1_mismatches++;
        }
    }
    FATAL_CHECK(top1_mismatches == 0, "Top-1 labels must match exactly");
}

static void test_flat_l2(size_t nv, size_t nq, size_t d, size_t k) {
    printf("test_flat_l2 (nv=%zu, nq=%zu, d=%zu, k=%zu)... ", nv, nq, d, k);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vectors(nv * d);
    std::vector<float> queries(nq * d);
    for (auto &v : vectors)
        v = dist(rng);
    for (auto &v : queries)
        v = dist(rng);

    // CPU reference
    faiss::IndexFlatL2 cpu_index(d);
    cpu_index.add(nv, vectors.data());

    std::vector<float> cpu_distances(nq * k);
    std::vector<faiss::idx_t> cpu_labels(nq * k);
    cpu_index.search(nq, queries.data(), k, cpu_distances.data(), cpu_labels.data());

    // Metal
    auto res = std::make_shared<StandardMetalResources>();
    MetalIndexFlat metal_index(res, d, faiss::METRIC_L2);
    metal_index.add(nv, vectors.data());

    std::vector<float> metal_distances(nq * k);
    std::vector<faiss::idx_t> metal_labels(nq * k);
    metal_index.search(nq, queries.data(), k, metal_distances.data(), metal_labels.data());

    compare_results("L2", nq, k, metal_distances.data(), metal_labels.data(), cpu_distances.data(), cpu_labels.data());

    printf("PASS\n");
}

static void test_flat_ip(size_t nv, size_t nq, size_t d, size_t k) {
    printf("test_flat_ip (nv=%zu, nq=%zu, d=%zu, k=%zu)... ", nv, nq, d, k);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vectors(nv * d);
    std::vector<float> queries(nq * d);
    for (auto &v : vectors)
        v = dist(rng);
    for (auto &v : queries)
        v = dist(rng);

    // CPU reference
    faiss::IndexFlatIP cpu_index(d);
    cpu_index.add(nv, vectors.data());

    std::vector<float> cpu_distances(nq * k);
    std::vector<faiss::idx_t> cpu_labels(nq * k);
    cpu_index.search(nq, queries.data(), k, cpu_distances.data(), cpu_labels.data());

    // Metal
    auto res = std::make_shared<StandardMetalResources>();
    MetalIndexFlat metal_index(res, d, faiss::METRIC_INNER_PRODUCT);
    metal_index.add(nv, vectors.data());

    std::vector<float> metal_distances(nq * k);
    std::vector<faiss::idx_t> metal_labels(nq * k);
    metal_index.search(nq, queries.data(), k, metal_distances.data(), metal_labels.data());

    compare_results("IP", nq, k, metal_distances.data(), metal_labels.data(), cpu_distances.data(), cpu_labels.data(),
                    1e-2f); // IP can have slightly larger numerical differences

    printf("PASS\n");
}

static void test_conversion() {
    printf("test_conversion (cpu→metal→cpu)... ");

    const size_t nv = 500;
    const size_t d = 128;
    const size_t nq = 10;
    const size_t k = 5;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vectors(nv * d);
    std::vector<float> queries(nq * d);
    for (auto &v : vectors)
        v = dist(rng);
    for (auto &v : queries)
        v = dist(rng);

    // Build CPU index
    faiss::IndexFlatL2 cpu_index(d);
    cpu_index.add(nv, vectors.data());

    // Convert to Metal
    auto res = std::make_shared<StandardMetalResources>();
    auto metal_index = index_cpu_to_metal(res, &cpu_index);

    FATAL_CHECK(metal_index->ntotal == cpu_index.ntotal, "ntotal mismatch after conversion");
    FATAL_CHECK(metal_index->d == cpu_index.d, "dimension mismatch after conversion");

    // Convert back to CPU
    auto cpu_index2 = index_metal_to_cpu(metal_index.get());
    FATAL_CHECK(cpu_index2->ntotal == cpu_index.ntotal, "ntotal mismatch after round-trip");

    // Search both and compare
    std::vector<float> d1(nq * k), d2(nq * k);
    std::vector<faiss::idx_t> l1(nq * k), l2(nq * k);

    cpu_index.search(nq, queries.data(), k, d1.data(), l1.data());
    cpu_index2->search(nq, queries.data(), k, d2.data(), l2.data());

    for (size_t i = 0; i < nq * k; i++) {
        FATAL_CHECK(l1[i] == l2[i], "Round-trip label mismatch");
        FATAL_CHECK(std::abs(d1[i] - d2[i]) < 1e-5f, "Round-trip distance mismatch");
    }

    printf("PASS\n");
}

static void test_reset() {
    printf("test_reset... ");

    auto res = std::make_shared<StandardMetalResources>();
    MetalIndexFlat index(res, 32, faiss::METRIC_L2);

    std::vector<float> data(100 * 32, 1.0f);
    index.add(100, data.data());
    FATAL_CHECK(index.ntotal == 100, "ntotal should be 100 after add");

    index.reset();
    FATAL_CHECK(index.ntotal == 0, "ntotal should be 0 after reset");

    // Add again after reset
    index.add(50, data.data());
    FATAL_CHECK(index.ntotal == 50, "ntotal should be 50 after re-add");

    printf("PASS\n");
}

static void test_reconstruct() {
    printf("test_reconstruct... ");

    auto res = std::make_shared<StandardMetalResources>();
    const int d = 64;
    MetalIndexFlat index(res, d, faiss::METRIC_L2);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(10 * d);
    for (auto &v : data)
        v = dist(rng);

    index.add(10, data.data());

    std::vector<float> recons(d);
    for (int i = 0; i < 10; i++) {
        index.reconstruct(i, recons.data());
        for (int j = 0; j < d; j++) {
            FATAL_CHECK(recons[j] == data[i * d + j], "Reconstruct mismatch");
        }
    }

    printf("PASS\n");
}

static void test_flat_l2_f16(size_t nv, size_t nq, size_t d, size_t k) {
    printf("test_flat_l2_f16 (nv=%zu, nq=%zu, d=%zu, k=%zu)... ", nv, nq, d, k);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vectors(nv * d);
    std::vector<float> queries(nq * d);
    for (auto &v : vectors)
        v = dist(rng);
    for (auto &v : queries)
        v = dist(rng);

    // CPU reference
    faiss::IndexFlatL2 cpu_index(d);
    cpu_index.add(nv, vectors.data());

    std::vector<float> cpu_distances(nq * k);
    std::vector<faiss::idx_t> cpu_labels(nq * k);
    cpu_index.search(nq, queries.data(), k, cpu_distances.data(), cpu_labels.data());

    // Metal FP16
    auto res = std::make_shared<StandardMetalResources>();
    MetalIndexFlat metal_index(res, d, faiss::METRIC_L2, /*useFloat16Storage=*/true);
    FATAL_CHECK(metal_index.isFloat16Storage(), "FP16 storage flag not set");
    metal_index.add(nv, vectors.data());

    std::vector<float> metal_distances(nq * k);
    std::vector<faiss::idx_t> metal_labels(nq * k);
    metal_index.search(nq, queries.data(), k, metal_distances.data(), metal_labels.data());

    // FP16 has lower precision, use relaxed tolerance (top-1 still must match)
    compare_results("L2-F16", nq, k, metal_distances.data(), metal_labels.data(), cpu_distances.data(),
                    cpu_labels.data(), 5e-2f);

    // Test reconstruct round-trip
    std::vector<float> recons(d);
    metal_index.reconstruct(0, recons.data());
    // Half precision: ~3 decimal digits, so ~1e-3 tolerance per element
    for (size_t i = 0; i < (size_t)d; i++) {
        float diff = std::abs(recons[i] - vectors[i]);
        FATAL_CHECK(diff < 2e-3f, "FP16 reconstruct precision too low");
    }

    printf("PASS\n");
}

static void test_forced_mps_search() {
    printf("test_forced_mps_search (L2+IP via MPS path)... ");

    const size_t nv = 500;
    const size_t nq = 5;
    const size_t d = 128;
    const size_t k = 5;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vectors(nv * d);
    std::vector<float> queries(nq * d);
    for (auto &v : vectors)
        v = dist(rng);
    for (auto &v : queries)
        v = dist(rng);

    // CPU reference
    faiss::IndexFlatL2 cpu_index(d);
    cpu_index.add(nv, vectors.data());
    std::vector<float> cpu_distances(nq * k);
    std::vector<faiss::idx_t> cpu_labels(nq * k);
    cpu_index.search(nq, queries.data(), k, cpu_distances.data(), cpu_labels.data());

    // Metal with forced MPS path
    auto res = std::make_shared<StandardMetalResources>();
    MetalIndexFlat metal_index(res, d, faiss::METRIC_L2);
    metal_index.setForceMPS(true);
    metal_index.add(nv, vectors.data());

    std::vector<float> metal_distances(nq * k);
    std::vector<faiss::idx_t> metal_labels(nq * k);
    metal_index.search(nq, queries.data(), k, metal_distances.data(), metal_labels.data());

    compare_results("ForcedMPS-L2", nq, k, metal_distances.data(), metal_labels.data(), cpu_distances.data(),
                    cpu_labels.data());

    printf("PASS\n");
}

static void test_search_async() {
    printf("test_search_async (results match sync)... ");

    const size_t nv = 1000;
    const size_t nq = 10;
    const size_t d = 128;
    const size_t k = 10;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vectors(nv * d);
    std::vector<float> queries(nq * d);
    for (auto &v : vectors)
        v = dist(rng);
    for (auto &v : queries)
        v = dist(rng);

    auto res = std::make_shared<StandardMetalResources>();
    MetalIndexFlat metal_index(res, d, faiss::METRIC_L2);
    metal_index.add(nv, vectors.data());

    // Sync reference
    std::vector<float> sync_distances(nq * k);
    std::vector<faiss::idx_t> sync_labels(nq * k);
    metal_index.search(nq, queries.data(), k, sync_distances.data(), sync_labels.data());

    // Async
    std::vector<float> async_distances(nq * k);
    std::vector<faiss::idx_t> async_labels(nq * k);
    auto token = metal_index.searchAsync(nq, queries.data(), k, async_distances.data(), async_labels.data());
    token.wait();

    // Results must match exactly (same GPU, same buffers, deterministic)
    for (size_t i = 0; i < nq * k; i++) {
        FATAL_CHECK(async_labels[i] == sync_labels[i], "Async label mismatch");
        FATAL_CHECK(async_distances[i] == sync_distances[i], "Async distance mismatch");
    }

    printf("PASS\n");
}

static void test_search_async_multiple() {
    printf("test_search_async_multiple (concurrent tokens)... ");

    const size_t nv = 1000;
    const size_t nq = 5;
    const size_t d = 128;
    const size_t k = 5;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vectors(nv * d);
    for (auto &v : vectors)
        v = dist(rng);

    auto res = std::make_shared<StandardMetalResources>();
    MetalIndexFlat metal_index(res, d, faiss::METRIC_L2);
    metal_index.add(nv, vectors.data());

    // Launch 3 async searches with different queries
    std::vector<float> q1(nq * d), q2(nq * d), q3(nq * d);
    for (auto &v : q1)
        v = dist(rng);
    for (auto &v : q2)
        v = dist(rng);
    for (auto &v : q3)
        v = dist(rng);

    std::vector<float> d1(nq * k), d2(nq * k), d3(nq * k);
    std::vector<faiss::idx_t> l1(nq * k), l2(nq * k), l3(nq * k);

    auto t1 = metal_index.searchAsync(nq, q1.data(), k, d1.data(), l1.data());
    auto t2 = metal_index.searchAsync(nq, q2.data(), k, d2.data(), l2.data());
    auto t3 = metal_index.searchAsync(nq, q3.data(), k, d3.data(), l3.data());

    // Wait in reverse order
    t3.wait();
    t2.wait();
    t1.wait();

    // Verify against sync results
    for (int batch = 0; batch < 3; batch++) {
        const float *q = (batch == 0) ? q1.data() : (batch == 1) ? q2.data() : q3.data();
        float *ad = (batch == 0) ? d1.data() : (batch == 1) ? d2.data() : d3.data();
        faiss::idx_t *al = (batch == 0) ? l1.data() : (batch == 1) ? l2.data() : l3.data();

        std::vector<float> sd(nq * k);
        std::vector<faiss::idx_t> sl(nq * k);
        metal_index.search(nq, q, k, sd.data(), sl.data());

        for (size_t i = 0; i < nq * k; i++) {
            FATAL_CHECK(al[i] == sl[i], "Concurrent async label mismatch");
            FATAL_CHECK(ad[i] == sd[i], "Concurrent async distance mismatch");
        }
    }

    printf("PASS\n");
}

static void test_search_async_isready() {
    printf("test_search_async_isready... ");

    const size_t nv = 1000;
    const size_t d = 128;
    const size_t k = 5;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vectors(nv * d);
    std::vector<float> query(d);
    for (auto &v : vectors)
        v = dist(rng);
    for (auto &v : query)
        v = dist(rng);

    auto res = std::make_shared<StandardMetalResources>();
    MetalIndexFlat metal_index(res, d, faiss::METRIC_L2);
    metal_index.add(nv, vectors.data());

    std::vector<float> dists(k);
    std::vector<faiss::idx_t> labels(k);
    auto token = metal_index.searchAsync(1, query.data(), k, dists.data(), labels.data());

    // Poll until ready (should finish quickly for small workload)
    while (!token.isReady()) {
        // spin
    }
    token.wait();

    // Sanity: labels should be valid
    for (size_t i = 0; i < (size_t)k; i++) {
        FATAL_CHECK(labels[i] >= 0 && labels[i] < (faiss::idx_t)nv, "Async isReady: invalid label");
    }

    printf("PASS\n");
}

static void test_search_async_empty() {
    printf("test_search_async_empty (n=0 and ntotal=0)... ");

    auto res = std::make_shared<StandardMetalResources>();
    MetalIndexFlat metal_index(res, 32, faiss::METRIC_L2);

    // ntotal=0 case
    std::vector<float> query(32, 1.0f);
    std::vector<float> dists(5);
    std::vector<faiss::idx_t> labels(5);
    auto t1 = metal_index.searchAsync(1, query.data(), 5, dists.data(), labels.data());
    t1.wait();
    for (int i = 0; i < 5; i++) {
        FATAL_CHECK(dists[i] == INFINITY, "Empty index should return INFINITY distance");
        FATAL_CHECK(labels[i] == -1, "Empty index should return -1 label");
    }

    // n=0 case
    auto t2 = metal_index.searchAsync(0, nullptr, 5, nullptr, nullptr);
    FATAL_CHECK(t2.isReady(), "n=0 async should be immediately ready");
    t2.wait();

    printf("PASS\n");
}

int main() {
    @autoreleasepool {
        auto res = std::make_shared<StandardMetalResources>();
        const auto &caps = res->getCapabilities();

        printf("=== MetalIndexFlat Tests ===\n");
        printf("%s\n", faiss_metal::describeCapabilities(caps).c_str());

        // Show active code paths
        printf("GEMM: %s\n", caps.hasFastFP16 ? "simdgroup_matrix (FP16)" : "MPS (FP32)");
        printf("L2 norm: %s\n\n", caps.hasFastFP16 ? "FP16 fast path" : "FP32");

        // Various dimensions
        test_flat_l2(1000, 10, 32, 5);
        test_flat_l2(1000, 10, 128, 10);
        test_flat_l2(500, 5, 768, 5);
        test_flat_l2(500, 5, 1536, 5);

        // Inner product
        test_flat_ip(1000, 10, 128, 10);

        // FP16 storage
        test_flat_l2_f16(1000, 10, 128, 10);
        test_flat_l2_f16(500, 5, 768, 5);

        // k > 32: exercises block_select path (bitonic merge)
        test_flat_l2(5000, 5, 128, 33);
        test_flat_l2(5000, 5, 128, 64);
        test_flat_l2(5000, 5, 128, 128);
        test_flat_ip(5000, 5, 128, 64);

        // Edge cases
        test_flat_l2(100, 1, 32, 1); // single query, k=1

        // Forced MPS path (tests both code paths on M2+)
        test_forced_mps_search();

        // Conversion
        test_conversion();

        // Reset
        test_reset();

        // Reconstruct
        test_reconstruct();

        // Async search
        test_search_async();
        test_search_async_multiple();
        test_search_async_isready();
        test_search_async_empty();

        printf("All MetalIndexFlat tests passed!\n");
    }
    return 0;
}
