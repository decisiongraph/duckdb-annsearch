#pragma once

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace faiss_metal {

/// Singleton providing the default Metal device and shader library.
class MetalContext {
  public:
    static MetalContext &instance();

    id<MTLDevice> device() const {
        return device_;
    }

    /// Load the compiled metallib from the given path
    id<MTLLibrary> loadLibrary(NSString *path);

  private:
    MetalContext();
    ~MetalContext();
    MetalContext(const MetalContext &) = delete;
    MetalContext &operator=(const MetalContext &) = delete;

    id<MTLDevice> device_;
};

} // namespace faiss_metal
