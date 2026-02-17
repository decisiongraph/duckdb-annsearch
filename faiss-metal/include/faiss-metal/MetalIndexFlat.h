#pragma once

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/MetricType.h>
#include <memory>

namespace faiss_metal {

class MetalResources;
class MetalIndexFlat;

/// Move-only token returned by MetalIndexFlat::searchAsync().
/// Owns all per-call Metal buffers so multiple async searches can be in-flight.
class MetalSearchToken {
public:
	MetalSearchToken(MetalSearchToken &&) noexcept;
	MetalSearchToken &operator=(MetalSearchToken &&) noexcept;
	~MetalSearchToken();

	// Non-copyable
	MetalSearchToken(const MetalSearchToken &) = delete;
	MetalSearchToken &operator=(const MetalSearchToken &) = delete;

	/// Block until GPU finishes, copy results to caller's output arrays.
	void wait();

	/// Non-blocking: true if GPU has finished.
	bool isReady() const;

private:
	friend class MetalIndexFlat;
	struct Impl;
	std::unique_ptr<Impl> impl_;
	explicit MetalSearchToken(std::unique_ptr<Impl> impl);
};

/// Vector storage precision for MetalIndexFlat.
enum class StorageType : uint8_t {
	Float32 = 0,  // Full precision (default)
	Float16 = 1,  // FP16: 2x bandwidth savings, 5-bit exponent
	BFloat16 = 2, // BF16: 2x bandwidth savings, 8-bit exponent (wider range)
};

/// Flat (brute-force) index on Metal GPU.
/// Subclasses faiss::Index directly -- all data lives in MTLBuffers (unified memory).
class MetalIndexFlat : public faiss::Index {
public:
	/// @param resources  Metal resource manager (device, queue, shaders)
	/// @param d          Vector dimension
	/// @param metric     METRIC_L2 or METRIC_INNER_PRODUCT
	/// @param useFloat16Storage  Store vectors as FP16 (halves memory bandwidth,
	///                           slight precision loss). Queries remain FP32.
	MetalIndexFlat(std::shared_ptr<MetalResources> resources, int d, faiss::MetricType metric = faiss::METRIC_L2,
	               bool useFloat16Storage = false);

	/// Extended constructor with explicit storage type.
	MetalIndexFlat(std::shared_ptr<MetalResources> resources, int d, faiss::MetricType metric, StorageType storage);

	~MetalIndexFlat() override;

	// --- faiss::Index interface ---
	void add(faiss::idx_t n, const float *x) override;
	void search(faiss::idx_t n, const float *x, faiss::idx_t k, float *distances, faiss::idx_t *labels,
	            const faiss::SearchParameters *params = nullptr) const override;
	void reset() override;
	void reconstruct(faiss::idx_t key, float *recons) const override;

	/// Asynchronous search: encodes + commits GPU work, returns immediately.
	/// The returned token owns all per-call buffers. Call token.wait() to
	/// block until complete and copy results into distances/labels.
	MetalSearchToken searchAsync(faiss::idx_t n, const float *x, faiss::idx_t k, float *distances, faiss::idx_t *labels,
	                             const faiss::SearchParameters *params = nullptr) const;

	/// Direct access to stored vectors (CPU pointer into unified memory).
	/// Returns nullptr if using FP16 storage -- use reconstruct() instead.
	const float *getVectorsData() const;

	/// True if vectors are stored as FP16.
	bool isFloat16Storage() const;

	/// True if vectors are stored as BFloat16.
	bool isBFloat16Storage() const;

	/// Get the storage type.
	StorageType getStorageType() const;

	/// Force MPS GEMM path even on M2+ hardware. For testing both paths.
	void setForceMPS(bool force);

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

/// Convert CPU IndexFlat → MetalIndexFlat (copies vectors into MTLBuffer)
std::unique_ptr<MetalIndexFlat> index_cpu_to_metal(std::shared_ptr<MetalResources> resources,
                                                   const faiss::IndexFlat *cpu_index);

/// Convert MetalIndexFlat → CPU IndexFlat
std::unique_ptr<faiss::IndexFlat> index_metal_to_cpu(const MetalIndexFlat *metal_index);

} // namespace faiss_metal
