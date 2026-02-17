#pragma once

#include "MetalResources.h"

namespace faiss_metal {

/// Default MetalResources implementation.
/// Manages a single MTLDevice + MTLCommandQueue + metallib.
class StandardMetalResources : public MetalResources {
public:
	StandardMetalResources();
	~StandardMetalResources() override;

	const MetalDeviceCapabilities &getCapabilities() const override;

#ifdef __OBJC__
	id<MTLDevice> getDevice() override;
	id<MTLCommandQueue> getDefaultCommandQueue() override;
	id<MTLLibrary> getMetalLibrary() override;

	id<MTLBuffer> allocBuffer(size_t size) override;
	void deallocBuffer(id<MTLBuffer> buf) override;

	/// Register a buffer for proactive GPU residency.
	/// Pre-pages GPU memory to avoid page fault latency on first access.
	void registerForResidency(id<MTLBuffer> buffer);
#endif

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

} // namespace faiss_metal
