#ifdef FAISS_METAL_ENABLED

#include "include/metal_diskann_bridge.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstring>
#include <mach/mach.h>

namespace {

/// Number of buffer sets in the ring (matches Apple's MaxFramesInFlight pattern).
static constexpr int RING_SIZE = 3;

/// A reusable set of Metal buffers for one dispatch.
struct BufferSet {
    id<MTLBuffer> query = nil;
    id<MTLBuffer> candidates = nil;
    id<MTLBuffer> distances = nil;
    id<MTLBuffer> params = nil;
    id<MTLBuffer> query_map = nil; // multi-query: per-candidate query index
    size_t query_capacity = 0;
    size_t candidates_capacity = 0;
    size_t distances_capacity = 0;
    size_t query_map_capacity = 0;
};

/// Round up to next power of 2 (minimum 4096 = page size).
static size_t next_pow2(size_t n) {
    if (n <= 4096)
        return 4096;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

struct DiskannMetalState {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> l2_pipeline = nil;
    id<MTLComputePipelineState> ip_pipeline = nil;
    id<MTLComputePipelineState> multi_l2_pipeline = nil;
    id<MTLComputePipelineState> multi_ip_pipeline = nil;
    bool initialized = false;

    // Ring buffer pool: 3 pre-allocated buffer sets
    BufferSet ring[RING_SIZE];
    int ring_idx = 0;

    bool init() {
        if (initialized)
            return true;

        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device)
                return false;

            queue = [device newCommandQueue];
            if (!queue)
                return false;

            // Load metallib (same path used by faiss-metal)
            NSString *path = @FAISS_METAL_METALLIB_PATH;
            NSError *error = nil;
            NSURL *url = [NSURL fileURLWithPath:path];
            id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
            if (!library)
                return false;

            id<MTLFunction> l2_fn = [library newFunctionWithName:@"diskann_batch_l2"];
            id<MTLFunction> ip_fn = [library newFunctionWithName:@"diskann_batch_ip"];
            id<MTLFunction> multi_l2_fn = [library newFunctionWithName:@"diskann_multi_batch_l2"];
            id<MTLFunction> multi_ip_fn = [library newFunctionWithName:@"diskann_multi_batch_ip"];
            if (!l2_fn || !ip_fn || !multi_l2_fn || !multi_ip_fn)
                return false;

            l2_pipeline = [device newComputePipelineStateWithFunction:l2_fn error:&error];
            if (!l2_pipeline)
                return false;

            ip_pipeline = [device newComputePipelineStateWithFunction:ip_fn error:&error];
            if (!ip_pipeline)
                return false;

            multi_l2_pipeline = [device newComputePipelineStateWithFunction:multi_l2_fn error:&error];
            if (!multi_l2_pipeline)
                return false;

            multi_ip_pipeline = [device newComputePipelineStateWithFunction:multi_ip_fn error:&error];
            if (!multi_ip_pipeline)
                return false;

            initialized = true;
        }
        return true;
    }

    /// Get or grow a buffer in the current ring slot.
    /// Returns nil on allocation failure.
    id<MTLBuffer> ensure_buffer(id<MTLBuffer> &buf, size_t &capacity, size_t needed) {
        if (buf && capacity >= needed) {
            return buf;
        }
        size_t alloc_size = next_pow2(needed);
        buf = [device newBufferWithLength:alloc_size options:MTLResourceStorageModeShared];
        capacity = buf ? alloc_size : 0;
        return buf;
    }

    /// Get next ring buffer set, growing buffers as needed.
    /// Returns pointer to BufferSet, or nullptr on allocation failure.
    BufferSet *next_buffers(size_t query_size, size_t candidates_size, size_t distances_size) {
        auto &bs = ring[ring_idx];
        ring_idx = (ring_idx + 1) % RING_SIZE;

        if (!ensure_buffer(bs.query, bs.query_capacity, query_size))
            return nullptr;
        if (!ensure_buffer(bs.candidates, bs.candidates_capacity, candidates_size))
            return nullptr;
        if (!ensure_buffer(bs.distances, bs.distances_capacity, distances_size))
            return nullptr;

        // Params buffer is tiny (8 bytes), allocate once
        if (!bs.params) {
            bs.params = [device newBufferWithLength:sizeof(uint32_t) * 2 options:MTLResourceStorageModeShared];
            if (!bs.params)
                return nullptr;
        }

        return &bs;
    }

    static DiskannMetalState &instance() {
        static DiskannMetalState state;
        return state;
    }
};

} // anonymous namespace

extern "C" int diskann_metal_available(void) {
    auto &state = DiskannMetalState::instance();
    if (!state.initialized) {
        state.init();
    }
    return state.initialized ? 1 : 0;
}

extern "C" int diskann_metal_batch_distances(const float *query, const float *candidates, int n, int dim, int metric,
                                             float *out_distances) {
    if (n <= 0 || dim <= 0 || !query || !candidates || !out_distances) {
        return -1;
    }

    auto &state = DiskannMetalState::instance();
    if (!state.initialized && !state.init()) {
        return -1;
    }

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = (metric == 0) ? state.l2_pipeline : state.ip_pipeline;

        size_t query_size = (size_t)dim * sizeof(float);
        size_t candidates_size = (size_t)n * dim * sizeof(float);
        size_t distances_size = (size_t)n * sizeof(float);

        // Reuse pre-allocated buffers from ring pool
        auto *bs = state.next_buffers(query_size, candidates_size, distances_size);
        if (!bs) {
            return -1;
        }

        // Copy query into shared buffer
        memcpy([bs->query contents], query, query_size);

        // Zero-copy input: wrap caller's candidates buffer if page-aligned
        id<MTLBuffer> cand_buf = nil;
        bool zero_copy_cands = false;

        if (((uintptr_t)candidates % vm_page_size) == 0 && (candidates_size % vm_page_size) == 0 &&
            candidates_size >= vm_page_size) {
            cand_buf = [state.device newBufferWithBytesNoCopy:(void *)candidates
                                                       length:candidates_size
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
            zero_copy_cands = (cand_buf != nil);
        }

        if (!zero_copy_cands) {
            memcpy([bs->candidates contents], candidates, candidates_size);
            cand_buf = bs->candidates;
        }

        // Write params
        struct {
            uint32_t n;
            uint32_t dim;
        } params = {(uint32_t)n, (uint32_t)dim};
        memcpy([bs->params contents], &params, sizeof(params));

        // Zero-copy output: wrap caller's buffer directly if page-aligned
        id<MTLBuffer> out_buf = nil;
        bool zero_copy_output = false;

        if (((uintptr_t)out_distances % vm_page_size) == 0 && (distances_size % vm_page_size) == 0) {
            // Caller's buffer is page-aligned and page-multiple size — zero-copy
            out_buf = [state.device newBufferWithBytesNoCopy:out_distances
                                                      length:distances_size
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
            zero_copy_output = (out_buf != nil);
        }

        if (!zero_copy_output) {
            out_buf = bs->distances;
        }

        id<MTLCommandBuffer> cmd_buf = [state.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bs->query offset:0 atIndex:0];
        [encoder setBuffer:cand_buf offset:0 atIndex:1];
        [encoder setBuffer:out_buf offset:0 atIndex:2];
        [encoder setBuffer:bs->params offset:0 atIndex:3];

        // One threadgroup per candidate, 32 threads per threadgroup (1 simdgroup)
        MTLSize grid_size = MTLSizeMake(n, 1, 1);
        MTLSize group_size = MTLSizeMake(32, 1, 1);
        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:group_size];

        [encoder endEncoding];
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        if (cmd_buf.status == MTLCommandBufferStatusError) {
            return -1;
        }

        // Only memcpy if we didn't use zero-copy output
        if (!zero_copy_output) {
            memcpy(out_distances, [out_buf contents], distances_size);
        }
    }

    return 0;
}

extern "C" int diskann_metal_multi_batch_distances(const float *queries, const float *candidates,
                                                   const unsigned int *query_map, int total_n, int nq, int dim,
                                                   int metric, float *out_distances) {
    if (total_n <= 0 || nq <= 0 || dim <= 0 || !queries || !candidates || !query_map || !out_distances) {
        return -1;
    }

    auto &state = DiskannMetalState::instance();
    if (!state.initialized && !state.init()) {
        return -1;
    }

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = (metric == 0) ? state.multi_l2_pipeline : state.multi_ip_pipeline;

        size_t queries_size = (size_t)nq * dim * sizeof(float);
        size_t candidates_size = (size_t)total_n * dim * sizeof(float);
        size_t query_map_size = (size_t)total_n * sizeof(uint32_t);
        size_t distances_size = (size_t)total_n * sizeof(float);

        // Reuse ring pool for queries + candidates + distances + query_map
        auto *bs = state.next_buffers(queries_size, candidates_size, distances_size);
        if (!bs) {
            return -1;
        }

        // Ensure query_map buffer is large enough
        if (!state.ensure_buffer(bs->query_map, bs->query_map_capacity, query_map_size)) {
            return -1;
        }

        memcpy([bs->query contents], queries, queries_size);
        memcpy([bs->candidates contents], candidates, candidates_size);
        memcpy([bs->query_map contents], query_map, query_map_size);

        // Write params
        struct {
            uint32_t n;
            uint32_t dim;
        } params = {(uint32_t)total_n, (uint32_t)dim};
        memcpy([bs->params contents], &params, sizeof(params));

        id<MTLCommandBuffer> cmd_buf = [state.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bs->query offset:0 atIndex:0];      // queries
        [encoder setBuffer:bs->candidates offset:0 atIndex:1]; // candidates
        [encoder setBuffer:bs->query_map offset:0 atIndex:2];  // query_map
        [encoder setBuffer:bs->distances offset:0 atIndex:3];  // out_distances
        [encoder setBuffer:bs->params offset:0 atIndex:4];     // params

        MTLSize grid_size = MTLSizeMake(total_n, 1, 1);
        MTLSize group_size = MTLSizeMake(32, 1, 1);
        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:group_size];

        [encoder endEncoding];
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        if (cmd_buf.status == MTLCommandBufferStatusError) {
            return -1;
        }

        memcpy(out_distances, [bs->distances contents], distances_size);
    }

    return 0;
}

#endif // FAISS_METAL_ENABLED
