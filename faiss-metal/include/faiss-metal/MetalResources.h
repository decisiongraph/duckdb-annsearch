#pragma once

#include "MetalDeviceCapabilities.h"
#include <cstddef>
#include <cstdlib>
#include <memory>

#ifdef __OBJC__
#import <Metal/Metal.h>
#else
// Opaque forward declarations for C++ headers
typedef void *id;
#endif

namespace faiss_metal {

/// Allocate page-aligned memory suitable for zero-copy MTLBuffer wrapping
/// via newBufferWithBytesNoCopy. Returns nullptr on failure.
/// Free with faiss_metal::free_aligned().
inline void *alloc_aligned(size_t bytes) {
	void *ptr = nullptr;
	// Round up to page size for Metal's newBufferWithBytesNoCopy requirement
	size_t pageSize = (size_t)getpagesize();
	size_t aligned = (bytes + pageSize - 1) & ~(pageSize - 1);
	if (posix_memalign(&ptr, pageSize, aligned) != 0) {
		return nullptr;
	}
	return ptr;
}

/// Free memory allocated with alloc_aligned().
inline void free_aligned(void *ptr) {
	free(ptr);
}

/// Abstract interface for Metal GPU resource management.
/// NOT a subclass of faiss::gpu::GpuResources -- Metal types are fundamentally different.
class MetalResources {
public:
	virtual ~MetalResources() = default;

	/// Device capabilities (available from both C++ and ObjC++)
	virtual const MetalDeviceCapabilities &getCapabilities() const = 0;

#ifdef __OBJC__
	virtual id<MTLDevice> getDevice() = 0;
	virtual id<MTLCommandQueue> getDefaultCommandQueue() = 0;
	virtual id<MTLLibrary> getMetalLibrary() = 0;

	/// Allocate a shared-mode buffer (CPU+GPU accessible, zero-copy on Apple Silicon)
	virtual id<MTLBuffer> allocBuffer(size_t size) = 0;
	virtual void deallocBuffer(id<MTLBuffer> buf) = 0;
#endif
};

} // namespace faiss_metal
