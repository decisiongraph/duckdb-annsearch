/// Standalone Metal vs CPU distance computation benchmark.
/// Compiles directly against Metal.framework, loads the built metallib.
/// Usage: clang++ -O2 -std=c++17 -framework Metal -framework Foundation bench.mm -o bench && ./bench

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

// CPU L2 squared distance
static float cpu_l2(const float *a, const float *b, int dim) {
    float sum = 0;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

// CPU batch L2 distance
static void cpu_batch_l2(const float *query, const float *candidates, int n, int dim, float *out) {
    for (int i = 0; i < n; i++) {
        out[i] = cpu_l2(query, candidates + i * dim, dim);
    }
}

struct MetalState {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> l2_pipeline;
    bool ok = false;

    bool init(const char *metallib_path) {
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device) {
                fprintf(stderr, "No Metal device\n");
                return false;
            }

            queue = [device newCommandQueue];

            NSString *path = [NSString stringWithUTF8String:metallib_path];
            NSError *error = nil;
            NSURL *url = [NSURL fileURLWithPath:path];
            id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
            if (!library) {
                fprintf(stderr, "Failed to load metallib: %s\n", [[error localizedDescription] UTF8String]);
                return false;
            }

            id<MTLFunction> fn = [library newFunctionWithName:@"diskann_batch_l2"];
            if (!fn) {
                fprintf(stderr, "Shader not found\n");
                return false;
            }

            l2_pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
            if (!l2_pipeline) {
                fprintf(stderr, "Pipeline failed\n");
                return false;
            }

            printf("Metal: %s\n", [[device name] UTF8String]);
            ok = true;
        }
        return true;
    }

    void batch_l2(const float *query, const float *candidates, int n, int dim, float *out) {
        @autoreleasepool {
            size_t q_size = dim * sizeof(float);
            size_t c_size = (size_t)n * dim * sizeof(float);
            size_t o_size = n * sizeof(float);

            id<MTLBuffer> q_buf = [device newBufferWithBytes:query length:q_size options:MTLResourceStorageModeShared];
            id<MTLBuffer> c_buf = [device newBufferWithBytes:candidates
                                                      length:c_size
                                                     options:MTLResourceStorageModeShared];
            id<MTLBuffer> o_buf = [device newBufferWithLength:o_size options:MTLResourceStorageModeShared];
            struct {
                uint32_t n;
                uint32_t dim;
            } params = {(uint32_t)n, (uint32_t)dim};
            id<MTLBuffer> p_buf = [device newBufferWithBytes:&params
                                                      length:sizeof(params)
                                                     options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:l2_pipeline];
            [enc setBuffer:q_buf offset:0 atIndex:0];
            [enc setBuffer:c_buf offset:0 atIndex:1];
            [enc setBuffer:o_buf offset:0 atIndex:2];
            [enc setBuffer:p_buf offset:0 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            memcpy(out, [o_buf contents], o_size);
        }
    }
};

static void fill_random(float *data, int count) {
    for (int i = 0; i < count; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
}

static void bench(MetalState &metal, int n, int dim, int iters) {
    printf("\n--- n=%d, dim=%d, iters=%d ---\n", n, dim, iters);

    std::vector<float> query(dim);
    std::vector<float> candidates(n * dim);
    std::vector<float> cpu_out(n);
    std::vector<float> gpu_out(n);

    fill_random(query.data(), dim);
    fill_random(candidates.data(), n * dim);

    // Warm up
    cpu_batch_l2(query.data(), candidates.data(), n, dim, cpu_out.data());
    if (metal.ok) {
        metal.batch_l2(query.data(), candidates.data(), n, dim, gpu_out.data());
    }

    // CPU benchmark
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) {
        cpu_batch_l2(query.data(), candidates.data(), n, dim, cpu_out.data());
    }
    auto t1 = Clock::now();
    double cpu_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

    // GPU benchmark
    double gpu_us = 0;
    if (metal.ok) {
        t0 = Clock::now();
        for (int i = 0; i < iters; i++) {
            metal.batch_l2(query.data(), candidates.data(), n, dim, gpu_out.data());
        }
        t1 = Clock::now();
        gpu_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    }

    // Verify correctness
    float max_err = 0;
    if (metal.ok) {
        for (int i = 0; i < n; i++) {
            float err = fabs(cpu_out[i] - gpu_out[i]);
            if (err > max_err)
                max_err = err;
        }
    }

    printf("  CPU: %8.1f us\n", cpu_us);
    if (metal.ok) {
        printf("  GPU: %8.1f us\n", gpu_us);
        printf("  Speedup: %.2fx\n", cpu_us / gpu_us);
        printf("  Max error: %.6e\n", max_err);
    }
}

int main(int argc, char **argv) {
    const char *metallib = "build/release/faiss_metal.metallib";
    if (argc > 1)
        metallib = argv[1];

    srand(42);

    MetalState metal;
    metal.init(metallib);

    // DiskANN-realistic: max_degree neighbors per iteration
    // MIN_GPU_WORK = 8192, so n*dim >= 8192 for GPU path

    // Below threshold (CPU only in DiskANN)
    bench(metal, 32, 128, 1000); // 4096 < 8192
    bench(metal, 64, 64, 1000);  // 4096 < 8192

    // At/above threshold (GPU used in DiskANN)
    bench(metal, 64, 128, 1000);  // 8192 = 8192
    bench(metal, 64, 256, 500);   // 16384
    bench(metal, 64, 768, 200);   // 49152
    bench(metal, 128, 256, 500);  // 32768
    bench(metal, 128, 768, 200);  // 98304
    bench(metal, 128, 1536, 100); // 196608

    return 0;
}
