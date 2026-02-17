#import <faiss-metal/StandardMetalResources.h>
#import <faiss-metal/MetalDeviceCapabilities.h>
#import "MetalContext.h"
#import <Foundation/Foundation.h>
#include <stdexcept>
#include <string>

namespace faiss_metal {

struct StandardMetalResources::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    MetalDeviceCapabilities capabilities;
    id<NSObject> residencySet; // id<MTLResidencySet>, typed as NSObject for availability

    Impl() {
        auto &ctx = MetalContext::instance();
        device = ctx.device();
        queue = [device newCommandQueue];
        if (!queue) {
            throw std::runtime_error("faiss_metal: Failed to create command queue");
        }

        // Load metallib from compiled path
        NSString *path = @FAISS_METAL_METALLIB_PATH;
        library = ctx.loadLibrary(path);

        // Query device capabilities
        capabilities = queryDeviceCapabilities(device);

        // Create residency set for proactive GPU page residency.
        // Buffers registered here are pre-paged to avoid page fault
        // latency on first GPU access.
        if (@available(macOS 15.0, *)) {
            MTLResidencySetDescriptor *rsDesc = [[MTLResidencySetDescriptor alloc] init];
            rsDesc.initialCapacity = 8;
            NSError *rsError = nil;
            residencySet = [device newResidencySetWithDescriptor:rsDesc error:&rsError];
            if (residencySet) {
                [(id<MTLResidencySet>)residencySet requestResidency];
            }
        }
    }
};

StandardMetalResources::StandardMetalResources() : impl_(std::make_unique<Impl>()) {
}

StandardMetalResources::~StandardMetalResources() = default;

const MetalDeviceCapabilities &StandardMetalResources::getCapabilities() const {
    return impl_->capabilities;
}

id<MTLDevice> StandardMetalResources::getDevice() {
    return impl_->device;
}

id<MTLCommandQueue> StandardMetalResources::getDefaultCommandQueue() {
    return impl_->queue;
}

id<MTLLibrary> StandardMetalResources::getMetalLibrary() {
    return impl_->library;
}

id<MTLBuffer> StandardMetalResources::allocBuffer(size_t size) {
    id<MTLBuffer> buf = [impl_->device newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (!buf) {
        throw std::runtime_error("faiss_metal: Failed to allocate buffer of size " + std::to_string(size));
    }
    return buf;
}

void StandardMetalResources::deallocBuffer(id<MTLBuffer> buf) {
    // ARC handles deallocation; explicit release not needed
    (void)buf;
}

void StandardMetalResources::registerForResidency(id<MTLBuffer> buffer) {
    if (@available(macOS 15.0, *)) {
        id<MTLResidencySet> rs = (id<MTLResidencySet>)impl_->residencySet;
        if (rs && buffer) {
            [rs addAllocation:buffer];
            [rs commit];
        }
    }
}

} // namespace faiss_metal
