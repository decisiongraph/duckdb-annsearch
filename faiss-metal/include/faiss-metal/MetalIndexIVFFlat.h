#pragma once

#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <memory>

namespace faiss {
class IndexIVFFlat;
}

namespace faiss_metal {

class MetalResources;

/// IVFFlat (Inverted File Index with flat storage) on Metal GPU.
///
/// Uses CPU for coarse quantizer search (finding probe cells), then Metal GPU
/// for batch distance computation and top-k selection within selected cells.
/// This gives sub-linear search: only vectors in probed cells are scanned.
///
/// Typical flow:
///   1. Create on CPU via faiss::IndexIVFFlat, train, add vectors
///   2. Convert to Metal via index_cpu_to_metal_ivf()
///   3. Search on GPU (coarse quantizer runs on CPU, fine search on GPU)
class MetalIndexIVFFlat : public faiss::Index {
public:
	/// @param resources  Metal resource manager (device, queue, shaders)
	/// @param d          Vector dimension
	/// @param nlist      Number of inverted lists (clusters)
	/// @param metric     METRIC_L2 or METRIC_INNER_PRODUCT
	MetalIndexIVFFlat(std::shared_ptr<MetalResources> resources, int d, size_t nlist,
	                  faiss::MetricType metric = faiss::METRIC_L2);

	~MetalIndexIVFFlat() override;

	// --- faiss::Index interface ---

	/// Train the coarse quantizer via CPU k-means.
	void train(faiss::idx_t n, const float *x) override;

	/// Add vectors: assigns each to nearest centroid, stores in inverted list.
	void add(faiss::idx_t n, const float *x) override;

	/// Search: CPU coarse quantizer + GPU fine search within probed cells.
	void search(faiss::idx_t n, const float *x, faiss::idx_t k, float *distances, faiss::idx_t *labels,
	            const faiss::SearchParameters *params = nullptr) const override;

	void reset() override;

	// --- IVF-specific ---

	/// Number of inverted lists (clusters).
	size_t getNlist() const;

	/// Number of lists to probe during search (default: 1).
	size_t getNprobe() const;

	/// Set number of lists to probe. Higher = more accurate but slower.
	void setNprobe(size_t nprobe);

private:
	friend std::unique_ptr<MetalIndexIVFFlat> index_cpu_to_metal_ivf(std::shared_ptr<MetalResources> resources,
	                                                                 const faiss::IndexIVFFlat *cpu_index);
	friend std::unique_ptr<faiss::IndexIVFFlat> index_metal_to_cpu_ivf(const MetalIndexIVFFlat *metal_index);

	struct Impl;
	std::unique_ptr<Impl> impl_;
};

/// Convert CPU IndexIVFFlat -> MetalIndexIVFFlat.
/// Copies centroids, inverted lists, and precomputes vector norms.
std::unique_ptr<MetalIndexIVFFlat> index_cpu_to_metal_ivf(std::shared_ptr<MetalResources> resources,
                                                          const faiss::IndexIVFFlat *cpu_index);

/// Convert MetalIndexIVFFlat -> CPU IndexIVFFlat.
std::unique_ptr<faiss::IndexIVFFlat> index_metal_to_cpu_ivf(const MetalIndexIVFFlat *metal_index);

} // namespace faiss_metal
