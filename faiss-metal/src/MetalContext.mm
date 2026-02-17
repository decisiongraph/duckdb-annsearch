#import "MetalContext.h"
#import <Foundation/Foundation.h>
#include <stdexcept>
#include <string>

namespace faiss_metal {

MetalContext &MetalContext::instance() {
    static MetalContext ctx;
    return ctx;
}

MetalContext::MetalContext() {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
        throw std::runtime_error("faiss_metal: No Metal device found");
    }
}

MetalContext::~MetalContext() = default;

id<MTLLibrary> MetalContext::loadLibrary(NSString *path) {
    NSError *error = nil;
    NSURL *url = [NSURL fileURLWithPath:path];
    id<MTLLibrary> lib = [device_ newLibraryWithURL:url error:&error];
    if (!lib) {
        NSString *desc = [error localizedDescription];
        throw std::runtime_error(std::string("faiss_metal: Failed to load metallib: ") + [desc UTF8String]);
    }
    return lib;
}

} // namespace faiss_metal
