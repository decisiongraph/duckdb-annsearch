// FAISS headers before ObjC (nil macro conflict with InvertedLists.h)
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

#import <Foundation/Foundation.h>
#import <faiss-metal/MetalIndexFlat.h>
#import <faiss-metal/MetalIndexIVFFlat.h>
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

using namespace faiss_metal;

static void test_ivfflat_l2_basic() {
    printf("test_ivfflat_l2_basic... ");

    const size_t nv = 2000;
    const size_t d = 64;
    const size_t nlist = 16;
    const size_t nprobe = 4;
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

    // CPU reference
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat cpu_index(&quantizer, d, nlist, faiss::METRIC_L2);
    cpu_index.own_fields = false;
    cpu_index.train(nv, vectors.data());
    cpu_index.add(nv, vectors.data());
    cpu_index.nprobe = nprobe;

    // Metal via conversion
    auto res = std::make_shared<StandardMetalResources>();
    auto metal_index = index_cpu_to_metal_ivf(res, &cpu_index);

    if (metal_index->getNlist() != nlist || metal_index->getNprobe() != nprobe ||
        metal_index->ntotal != (faiss::idx_t)nv || !metal_index->is_trained) {
        printf("FATAL: conversion metadata mismatch\n");
        abort();
    }

    // CPU reference search (same nprobe)
    std::vector<float> cpu_distances(nq * k);
    std::vector<faiss::idx_t> cpu_labels(nq * k);
    cpu_index.search(nq, queries.data(), k, cpu_distances.data(), cpu_labels.data());

    std::vector<float> metal_distances(nq * k);
    std::vector<faiss::idx_t> metal_labels(nq * k);
    metal_index->search(nq, queries.data(), k, metal_distances.data(), metal_labels.data());

    // Top-1 labels must match CPU exactly
    int top1_mismatches = 0;
    for (size_t qi = 0; qi < nq; qi++) {
        if (metal_labels[qi * k] != cpu_labels[qi * k]) {
            if (top1_mismatches < 5) {
                printf("  L2 IVF top-1 mismatch: query=%zu metal=%lld cpu=%lld\n", qi, (long long)metal_labels[qi * k],
                       (long long)cpu_labels[qi * k]);
            }
            top1_mismatches++;
        }
    }
    FATAL_CHECK(top1_mismatches == 0, "IVFFlat L2 top-1 labels must match CPU");

    // Distances within tolerance
    int dist_mismatches = 0;
    for (size_t i = 0; i < nq * k; i++) {
        float relDiff = std::abs(metal_distances[i] - cpu_distances[i]) / std::max(std::abs(cpu_distances[i]), 1e-6f);
        if (relDiff > 5e-2f)
            dist_mismatches++;
    }
    if (dist_mismatches > 0) {
        printf("  WARNING: %d/%zu IVFFlat L2 distance mismatches (tol=5e-2)\n", dist_mismatches, nq * k);
    }

    printf("PASS\n");
}

static void test_ivfflat_ip() {
    printf("test_ivfflat_ip... ");

    const size_t nv = 1000;
    const size_t d = 128;
    const size_t nlist = 8;
    const size_t nprobe = 2;
    const size_t nq = 5;
    const size_t k = 10;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vectors(nv * d);
    std::vector<float> queries(nq * d);
    for (auto &v : vectors)
        v = dist(rng);
    for (auto &v : queries)
        v = dist(rng);

    // CPU reference
    faiss::IndexFlatIP quantizer(d);
    faiss::IndexIVFFlat cpu_index(&quantizer, d, nlist, faiss::METRIC_INNER_PRODUCT);
    cpu_index.own_fields = false;
    cpu_index.train(nv, vectors.data());
    cpu_index.add(nv, vectors.data());
    cpu_index.nprobe = nprobe;

    std::vector<float> cpu_distances(nq * k);
    std::vector<faiss::idx_t> cpu_labels(nq * k);
    cpu_index.search(nq, queries.data(), k, cpu_distances.data(), cpu_labels.data());

    // Metal
    auto res = std::make_shared<StandardMetalResources>();
    auto metal_index = index_cpu_to_metal_ivf(res, &cpu_index);

    std::vector<float> metal_distances(nq * k);
    std::vector<faiss::idx_t> metal_labels(nq * k);
    metal_index->search(nq, queries.data(), k, metal_distances.data(), metal_labels.data());

    // Top-1 labels must match CPU exactly
    int top1_mismatches = 0;
    for (size_t qi = 0; qi < nq; qi++) {
        if (metal_labels[qi * k] != cpu_labels[qi * k]) {
            if (top1_mismatches < 5) {
                printf("  IP IVF top-1 mismatch: query=%zu metal=%lld cpu=%lld\n", qi, (long long)metal_labels[qi * k],
                       (long long)cpu_labels[qi * k]);
            }
            top1_mismatches++;
        }
    }
    FATAL_CHECK(top1_mismatches == 0, "IVFFlat IP top-1 labels must match CPU");

    // Distances within tolerance
    int dist_mismatches = 0;
    for (size_t i = 0; i < nq * k; i++) {
        float relDiff = std::abs(metal_distances[i] - cpu_distances[i]) / std::max(std::abs(cpu_distances[i]), 1e-6f);
        if (relDiff > 5e-2f)
            dist_mismatches++;
    }
    if (dist_mismatches > 0) {
        printf("  WARNING: %d/%zu IVFFlat IP distance mismatches (tol=5e-2)\n", dist_mismatches, nq * k);
    }

    printf("PASS\n");
}

static void test_ivfflat_conversion_roundtrip() {
    printf("test_ivfflat_conversion_roundtrip... ");

    const size_t nv = 500;
    const size_t d = 32;
    const size_t nlist = 4;
    const size_t nq = 5;
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
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat cpu_index(&quantizer, d, nlist, faiss::METRIC_L2);
    cpu_index.own_fields = false;
    cpu_index.train(nv, vectors.data());
    cpu_index.add(nv, vectors.data());
    cpu_index.nprobe = 2;

    std::vector<float> cpu_d1(nq * k);
    std::vector<faiss::idx_t> cpu_l1(nq * k);
    cpu_index.search(nq, queries.data(), k, cpu_d1.data(), cpu_l1.data());

    // CPU → Metal → CPU
    auto res = std::make_shared<StandardMetalResources>();
    auto metal_index = index_cpu_to_metal_ivf(res, &cpu_index);
    auto cpu_index2 = index_metal_to_cpu_ivf(metal_index.get());

    if (cpu_index2->ntotal != cpu_index.ntotal || cpu_index2->nlist != cpu_index.nlist ||
        cpu_index2->nprobe != cpu_index.nprobe) {
        printf("FATAL: round-trip metadata mismatch\n");
        abort();
    }

    std::vector<float> cpu_d2(nq * k);
    std::vector<faiss::idx_t> cpu_l2(nq * k);
    cpu_index2->search(nq, queries.data(), k, cpu_d2.data(), cpu_l2.data());

    for (size_t i = 0; i < nq * k; i++) {
        if (cpu_l1[i] != cpu_l2[i]) {
            printf("FATAL: round-trip label mismatch at %zu\n", i);
            abort();
        }
        if (std::abs(cpu_d1[i] - cpu_d2[i]) > 1e-5f) {
            printf("FATAL: round-trip distance mismatch at %zu: %.6f vs %.6f\n", i, cpu_d1[i], cpu_d2[i]);
            abort();
        }
    }

    printf("PASS\n");
}

static void test_ivfflat_train_and_add() {
    printf("test_ivfflat_train_and_add... ");

    const size_t nv = 1000;
    const size_t d = 32;
    const size_t nlist = 8;
    const size_t k = 5;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vectors(nv * d);
    for (auto &v : vectors)
        v = dist(rng);

    auto res = std::make_shared<StandardMetalResources>();
    MetalIndexIVFFlat metal_index(res, d, nlist, faiss::METRIC_L2);

    if (metal_index.is_trained || metal_index.ntotal != 0) {
        printf("FATAL: bad initial state\n");
        abort();
    }

    // Train
    metal_index.train(nv, vectors.data());
    if (!metal_index.is_trained) {
        printf("FATAL: not trained\n");
        abort();
    }

    // Add
    metal_index.add(nv, vectors.data());
    if (metal_index.ntotal != (faiss::idx_t)nv) {
        printf("FATAL: ntotal mismatch\n");
        abort();
    }

    // Search
    std::vector<float> query(d);
    for (auto &v : query)
        v = dist(rng);

    std::vector<float> distances(k);
    std::vector<faiss::idx_t> labels(k);
    metal_index.setNprobe(2);
    metal_index.search(1, query.data(), k, distances.data(), labels.data());

    // Verify search completes and returns some labels
    bool any_valid = false;
    for (size_t i = 0; i < k; i++) {
        if (labels[i] >= 0)
            any_valid = true;
    }
    if (!any_valid) {
        printf("FATAL: search returned no valid labels\n");
        abort();
    }

    printf("PASS\n");
}

static void test_ivfflat_reset() {
    printf("test_ivfflat_reset... ");

    const size_t d = 32;
    const size_t nlist = 4;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vectors(200 * d);
    for (auto &v : vectors)
        v = dist(rng);

    auto res = std::make_shared<StandardMetalResources>();
    MetalIndexIVFFlat metal_index(res, d, nlist, faiss::METRIC_L2);
    metal_index.train(200, vectors.data());
    metal_index.add(200, vectors.data());
    if (metal_index.ntotal != 200) {
        printf("FATAL: ntotal\n");
        abort();
    }

    metal_index.reset();
    if (metal_index.ntotal != 0) {
        printf("FATAL: reset\n");
        abort();
    }

    // Re-add
    metal_index.add(100, vectors.data());
    if (metal_index.ntotal != 100) {
        printf("FATAL: re-add\n");
        abort();
    }

    printf("PASS\n");
}

static void test_ivfflat_empty_search() {
    printf("test_ivfflat_empty_search... ");

    auto res = std::make_shared<StandardMetalResources>();
    MetalIndexIVFFlat metal_index(res, 32, 4, faiss::METRIC_L2);

    // Train with dummy data (required before search)
    std::vector<float> dummy(100 * 32, 0.0f);
    metal_index.train(100, dummy.data());

    // Search with 0 vectors added
    std::vector<float> query(32, 1.0f);
    std::vector<float> distances(5);
    std::vector<faiss::idx_t> labels(5);
    metal_index.search(1, query.data(), 5, distances.data(), labels.data());

    for (int i = 0; i < 5; i++) {
        if (distances[i] != INFINITY || labels[i] != -1) {
            printf("FATAL: empty search should return sentinels\n");
            abort();
        }
    }

    printf("PASS\n");
}

int main() {
    @autoreleasepool {
        auto res = std::make_shared<StandardMetalResources>();
        const auto &caps = res->getCapabilities();

        printf("=== MetalIndexIVFFlat Tests ===\n");
        printf("%s\n\n", faiss_metal::describeCapabilities(caps).c_str());

        test_ivfflat_l2_basic();
        test_ivfflat_ip();
        test_ivfflat_conversion_roundtrip();
        test_ivfflat_train_and_add();
        test_ivfflat_reset();
        test_ivfflat_empty_search();

        printf("\nAll MetalIndexIVFFlat tests passed!\n");
    }
    return 0;
}
