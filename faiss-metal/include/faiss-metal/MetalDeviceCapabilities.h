#pragma once

#include <cstdint>
#include <string>

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

namespace faiss_metal {

/// Apple Silicon generation detected via Metal GPU family.
enum class AppleSiliconGen : uint8_t {
	Unknown = 0,
	M1 = 1, // Apple7 family (A14, M1)
	M2 = 2, // Apple8 family (A15/A16, M2)
	M3 = 3, // Apple9 family (A17 Pro, M3)
	M4 = 4, // Apple9+ family (M4) -- detected via device name heuristic
};

/// Capability flags for Metal GPU, determined at runtime.
/// Use this to select optimal code paths per generation.
struct MetalDeviceCapabilities {
	/// Detected Apple Silicon generation
	AppleSiliconGen generation = AppleSiliconGen::Unknown;

	/// Device name string (e.g. "Apple M2 Max")
	std::string deviceName;

	// --- Capability flags ---

	/// simdgroup_matrix_multiply_accumulate (Apple7+ but Apple8+ for reliable FP16)
	/// Enables custom GEMM shaders that bypass MPS overhead.
	bool hasSimdGroupMatrix = false;

	/// Fast FP16 ALU (all Apple Silicon, but Apple8+ has dedicated FP16 pipelines
	/// with 2x throughput vs FP32)
	bool hasFastFP16 = false;

	/// Apple9+ dynamic shader core memory: threadgroup/tile memory is cached,
	/// dynamically allocated from shared on-chip pool. Higher effective occupancy.
	bool hasDynamicThreadgroupMemory = false;

	/// BFloat16 support in Metal shaders (Apple8+/macOS Sonoma+)
	bool hasBFloat16 = false;

	/// Max threadgroup memory size in bytes
	uint32_t maxThreadgroupMemoryBytes = 0;

	/// Max threads per threadgroup
	uint32_t maxThreadsPerThreadgroup = 0;

	/// SIMD width (always 32 on Apple Silicon, but query to be safe)
	uint32_t simdWidth = 32;

	/// Whether this is a Pro/Max/Ultra variant with more GPU cores
	bool isProMaxUltra = false;

	/// Number of GPU cores (0 if unknown)
	uint32_t gpuCoreCount = 0;
};

#ifdef __OBJC__
/// Query capabilities from a Metal device.
MetalDeviceCapabilities queryDeviceCapabilities(id<MTLDevice> device);
#endif

/// Human-readable description of capabilities.
std::string describeCapabilities(const MetalDeviceCapabilities &caps);

} // namespace faiss_metal
