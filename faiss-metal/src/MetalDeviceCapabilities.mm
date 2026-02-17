#import <faiss-metal/MetalDeviceCapabilities.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <string>

namespace faiss_metal {

MetalDeviceCapabilities queryDeviceCapabilities(id<MTLDevice> device) {
    MetalDeviceCapabilities caps;

    // Device name
    NSString *name = [device name];
    caps.deviceName = name ? [name UTF8String] : "unknown";

    // Detect generation via GPU family (cumulative: Apple9 implies Apple8 implies Apple7)
    bool apple7 = [device supportsFamily:MTLGPUFamilyApple7];
    bool apple8 = [device supportsFamily:MTLGPUFamilyApple8];
    bool apple9 = [device supportsFamily:MTLGPUFamilyApple9];

    // Apple9 covers both M3 and M4. Distinguish via device name.
    if (apple9) {
        NSString *upper = [name uppercaseString];
        if ([upper containsString:@"M4"]) {
            caps.generation = AppleSiliconGen::M4;
        } else {
            caps.generation = AppleSiliconGen::M3;
        }
    } else if (apple8) {
        caps.generation = AppleSiliconGen::M2;
    } else if (apple7) {
        caps.generation = AppleSiliconGen::M1;
    }

    // Pro/Max/Ultra detection from device name
    {
        NSString *upper = [name uppercaseString];
        caps.isProMaxUltra =
            [upper containsString:@"PRO"] || [upper containsString:@"MAX"] || [upper containsString:@"ULTRA"];
    }

    // --- Capability flags ---

    // simdgroup_matrix: available on Apple7+ but Apple8+ has robust FP16 path
    caps.hasSimdGroupMatrix = apple7;

    // Fast FP16: Apple8+ has dedicated FP16 pipelines with 2x throughput
    // Apple7 (M1) also has good FP16 but Apple8+ is notably better
    caps.hasFastFP16 = apple8;

    // Dynamic threadgroup memory (cached on-chip): Apple9+
    caps.hasDynamicThreadgroupMemory = apple9;

    // BFloat16 in Metal shaders: Apple8+ on macOS 14+
    if (apple8) {
        // Check if runtime supports bfloat (macOS Sonoma+)
        if (@available(macOS 14.0, *)) {
            caps.hasBFloat16 = true;
        }
    }

    // Max threadgroup memory
    // Apple7: 32KB, Apple8: 32KB, Apple9: 32KB base but dynamic caching
    // helps effective utilization
    caps.maxThreadgroupMemoryBytes = (uint32_t)[device maxThreadgroupMemoryLength];

    // Max threads per threadgroup
    caps.maxThreadsPerThreadgroup = (uint32_t)[device maxThreadsPerThreadgroup].width;

    // SIMD width: query from a dummy pipeline or assume 32
    // All Apple Silicon GPUs use SIMD width 32
    caps.simdWidth = 32;

    // GPU core count: not directly exposed by Metal API.
    // We can estimate from device name for known chips.
    // This is best-effort; 0 means unknown.
    {
        NSString *upper = [name uppercaseString];
        if ([upper containsString:@"ULTRA"]) {
            // Ultra variants: M1 Ultra 48/64, M2 Ultra 60/76, M3 Ultra 60/76, M4 Ultra 64/80
            caps.gpuCoreCount = 64; // approximate
        } else if ([upper containsString:@"MAX"]) {
            // Max variants: M1 Max 24/32, M2 Max 30/38, M3 Max 30/40, M4 Max 32/40
            caps.gpuCoreCount = 32; // approximate
        } else if ([upper containsString:@"PRO"]) {
            // Pro variants: M1 Pro 14/16, M2 Pro 16/19, M3 Pro 14/18, M4 Pro 16/20
            caps.gpuCoreCount = 16; // approximate
        } else {
            // Base variants: M1 7/8, M2 8/10, M3 8/10, M4 10
            caps.gpuCoreCount = 10; // approximate
        }
    }

    return caps;
}

std::string describeCapabilities(const MetalDeviceCapabilities &caps) {
    std::string gen;
    switch (caps.generation) {
    case AppleSiliconGen::M1:
        gen = "M1";
        break;
    case AppleSiliconGen::M2:
        gen = "M2";
        break;
    case AppleSiliconGen::M3:
        gen = "M3";
        break;
    case AppleSiliconGen::M4:
        gen = "M4";
        break;
    default:
        gen = "Unknown";
        break;
    }

    std::string result;
    result += "Device: " + caps.deviceName + "\n";
    result += "Generation: " + gen;
    if (caps.isProMaxUltra)
        result += " (Pro/Max/Ultra)";
    result += "\n";
    result += "GPU cores: ~" + std::to_string(caps.gpuCoreCount) + "\n";
    result += "SIMD width: " + std::to_string(caps.simdWidth) + "\n";
    result += "Max threadgroup memory: " + std::to_string(caps.maxThreadgroupMemoryBytes / 1024) + " KB\n";
    result += "Max threads/threadgroup: " + std::to_string(caps.maxThreadsPerThreadgroup) + "\n";
    result += "Capabilities:\n";
    result += "  simdgroup_matrix: " + std::string(caps.hasSimdGroupMatrix ? "yes" : "no") + "\n";
    result += "  fast FP16: " + std::string(caps.hasFastFP16 ? "yes" : "no") + "\n";
    result += "  dynamic threadgroup mem: " + std::string(caps.hasDynamicThreadgroupMemory ? "yes" : "no") + "\n";
    result += "  bfloat16: " + std::string(caps.hasBFloat16 ? "yes" : "no") + "\n";

    return result;
}

} // namespace faiss_metal
