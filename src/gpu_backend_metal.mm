#ifdef FAISS_AVAILABLE

#include "gpu_backend.hpp"

#ifdef FAISS_METAL_ENABLED

#include "faiss_wrapper.hpp"

#include <faiss-metal/MetalIndexFlat.h>
#include <faiss-metal/MetalIndexIVFFlat.h>
#include <faiss-metal/StandardMetalResources.h>

namespace duckdb {

class MetalGpuBackend : public GpuBackend {
  public:
    MetalGpuBackend() {
        try {
            resources_ = std::make_shared<faiss_metal::StandardMetalResources>();
            available_ = true;
        } catch (...) {
            available_ = false;
        }
    }

    bool IsAvailable() const override {
        return available_;
    }

    std::string DeviceInfo() const override {
        if (!available_) {
            return "Metal: not available";
        }
        auto &caps = resources_->getCapabilities();
        return "Metal GPU (" + caps.deviceName + ")";
    }

    std::string BackendName() const override {
        return "metal";
    }

    std::unique_ptr<faiss::Index> CpuToGpu(faiss::Index *cpu_index) override {
        if (!available_) {
            throw std::runtime_error("Metal GPU backend not available");
        }

        // Try IndexIVFFlat first (it also contains an IndexFlat quantizer)
        auto *ivfflat = dynamic_cast<faiss::IndexIVFFlat *>(cpu_index);
        if (ivfflat) {
            return faiss_metal::index_cpu_to_metal_ivf(resources_, ivfflat);
        }

        auto *flat = dynamic_cast<faiss::IndexFlat *>(cpu_index);
        if (flat) {
            return faiss_metal::index_cpu_to_metal(resources_, flat);
        }

        throw std::runtime_error("Metal GPU supports IndexFlat and IndexIVFFlat. "
                                 "Got an unsupported index type.");
    }

    std::unique_ptr<faiss::Index> GpuToCpu(faiss::Index *gpu_index) override {
        auto *metal_ivf = dynamic_cast<faiss_metal::MetalIndexIVFFlat *>(gpu_index);
        if (metal_ivf) {
            return faiss_metal::index_metal_to_cpu_ivf(metal_ivf);
        }

        auto *metal_flat = dynamic_cast<faiss_metal::MetalIndexFlat *>(gpu_index);
        if (metal_flat) {
            return faiss_metal::index_metal_to_cpu(metal_flat);
        }

        throw std::runtime_error("Index is not a Metal index -- cannot convert to CPU");
    }

  private:
    std::shared_ptr<faiss_metal::MetalResources> resources_;
    bool available_ = false;
};

GpuBackend &GetGpuBackend() {
    static MetalGpuBackend instance;
    return instance;
}

} // namespace duckdb

#endif // FAISS_METAL_ENABLED

#endif // FAISS_AVAILABLE
