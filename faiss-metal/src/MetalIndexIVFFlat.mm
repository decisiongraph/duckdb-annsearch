// FAISS headers must be included BEFORE ObjC headers because macOS defines
// `nil` as `nullptr`, and FAISS InvertedLists.h uses `nil` as a parameter name.
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/FaissAssert.h>

#import <faiss-metal/MetalIndexIVFFlat.h>
#import <faiss-metal/MetalResources.h>
#import <faiss-metal/StandardMetalResources.h>
#import "MetalDistance.h"
#import "MetalL2Norm.h"
#import "MetalSelect.h"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

namespace faiss_metal {

// --- MetalIndexIVFFlat ---

struct MetalIndexIVFFlat::Impl {
    std::shared_ptr<MetalResources> resources;
    size_t nlist_val;
    size_t nprobe_val = 1;

    // Coarse quantizer: uses FAISS IndexFlat to guarantee identical cell
    // selection as CPU IndexIVFFlat. Stores centroids (nlist * d floats).
    std::unique_ptr<faiss::IndexFlat> quantizer;

    // Inverted lists
    struct InvertedList {
        std::vector<faiss::idx_t> ids;
        std::vector<float> vectors; // ids.size() * d floats
        std::vector<float> norms;   // ||v||^2 per vector (for L2)
    };
    std::vector<InvertedList> invlists;

    // Metal compute objects
    std::unique_ptr<MetalDistance> distance;
    std::unique_ptr<MetalL2Norm> l2norm;
    std::unique_ptr<MetalSelect> selector;

    Impl(std::shared_ptr<MetalResources> res, size_t nlist, int d, faiss::MetricType metric)
        : resources(std::move(res)), nlist_val(nlist), quantizer(std::make_unique<faiss::IndexFlat>(d, metric)),
          invlists(nlist), distance(std::make_unique<MetalDistance>(resources.get())),
          l2norm(std::make_unique<MetalL2Norm>(resources.get())),
          selector(std::make_unique<MetalSelect>(resources.get())) {
    }

    // Assign a single vector to its nearest centroid
    faiss::idx_t assign_one(const float *vec) const {
        float dist;
        faiss::idx_t label;
        quantizer->search(1, vec, 1, &dist, &label);
        return label;
    }

    // Find nprobe nearest centroids for a single vector
    void coarse_search(const float *vec, size_t nprobe, std::vector<faiss::idx_t> &out_cells,
                       std::vector<float> &out_dists) const {
        size_t np = std::min(nprobe, nlist_val);
        out_cells.resize(np);
        out_dists.resize(np);
        quantizer->search(1, vec, np, out_dists.data(), out_cells.data());
    }

    // Compute L2 norm of a single vector
    static float compute_norm(const float *vec, int dim) {
        float norm = 0;
        for (int i = 0; i < dim; i++) {
            norm += vec[i] * vec[i];
        }
        return norm;
    }
};

MetalIndexIVFFlat::MetalIndexIVFFlat(std::shared_ptr<MetalResources> resources, int d, size_t nlist,
                                     faiss::MetricType metric)
    : faiss::Index(d, metric), impl_(std::make_unique<Impl>(std::move(resources), nlist, d, metric)) {
    is_trained = false;
}

MetalIndexIVFFlat::~MetalIndexIVFFlat() = default;

void MetalIndexIVFFlat::train(faiss::idx_t n, const float *x) {
    FAISS_THROW_IF_NOT_MSG(n > 0, "train: need at least 1 vector");

    // Use FAISS CPU k-means via a temporary IndexIVFFlat
    faiss::IndexFlat tmp_quantizer(d, metric_type);
    faiss::IndexIVFFlat cpu_ivf(&tmp_quantizer, d, impl_->nlist_val, metric_type);
    cpu_ivf.own_fields = false; // tmp_quantizer is on stack
    cpu_ivf.train(n, x);

    // Copy trained centroids into our persistent quantizer
    impl_->quantizer->reset();
    impl_->quantizer->add(tmp_quantizer.ntotal, tmp_quantizer.get_xb());

    is_trained = true;
}

void MetalIndexIVFFlat::add(faiss::idx_t n, const float *x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "add: index must be trained first");

    for (faiss::idx_t i = 0; i < n; i++) {
        const float *vec = x + i * d;

        // Assign to nearest centroid via FAISS quantizer
        size_t cell = (size_t)impl_->assign_one(vec);

        auto &list = impl_->invlists[cell];
        list.ids.push_back(ntotal);
        list.vectors.insert(list.vectors.end(), vec, vec + d);
        if (metric_type == faiss::METRIC_L2) {
            list.norms.push_back(Impl::compute_norm(vec, d));
        }

        ntotal++;
    }
}

void MetalIndexIVFFlat::search(faiss::idx_t n, const float *x, faiss::idx_t k, float *distances, faiss::idx_t *labels,
                               const faiss::SearchParameters *params) const {

    FAISS_THROW_IF_NOT_MSG(is_trained, "search: index must be trained first");
    FAISS_THROW_IF_NOT_MSG(k > 0, "k must be > 0");

    float sentinel_dist = (metric_type == faiss::METRIC_L2) ? INFINITY : -INFINITY;

    if (n == 0 || ntotal == 0) {
        for (faiss::idx_t i = 0; i < n * k; i++) {
            distances[i] = sentinel_dist;
            labels[i] = -1;
        }
        return;
    }

    id<MTLDevice> device = impl_->resources->getDevice();
    id<MTLCommandQueue> queue = impl_->resources->getDefaultCommandQueue();

    // Reusable buffers across queries
    std::vector<faiss::idx_t> probe_cells;
    std::vector<float> probe_dists;
    std::vector<float> cand_vecs;
    std::vector<float> cand_norms;
    std::vector<faiss::idx_t> cand_ids;

    // Process each query independently so each probes only its own cells.
    for (faiss::idx_t qi = 0; qi < n; qi++) {
        const float *q = x + qi * d;
        float *out_dist = distances + qi * k;
        faiss::idx_t *out_labels = labels + qi * k;

        // --- Step 1: CPU coarse search via FAISS quantizer ---
        impl_->coarse_search(q, impl_->nprobe_val, probe_cells, probe_dists);

        // --- Step 2: Gather candidates from probed cells ---
        size_t total_candidates = 0;
        for (faiss::idx_t c : probe_cells) {
            if (c < 0 || (size_t)c >= impl_->nlist_val)
                continue;
            total_candidates += impl_->invlists[c].ids.size();
        }

        if (total_candidates == 0) {
            for (faiss::idx_t j = 0; j < k; j++) {
                out_dist[j] = sentinel_dist;
                out_labels[j] = -1;
            }
            continue;
        }

        cand_vecs.resize(total_candidates * d);
        cand_ids.resize(total_candidates);
        if (metric_type == faiss::METRIC_L2) {
            cand_norms.resize(total_candidates);
        }

        size_t offset = 0;
        for (faiss::idx_t c : probe_cells) {
            if (c < 0 || (size_t)c >= impl_->nlist_val)
                continue;
            auto &list = impl_->invlists[c];
            size_t cnt = list.ids.size();
            if (cnt == 0)
                continue;

            memcpy(cand_vecs.data() + offset * d, list.vectors.data(), cnt * d * sizeof(float));
            memcpy(cand_ids.data() + offset, list.ids.data(), cnt * sizeof(faiss::idx_t));
            if (metric_type == faiss::METRIC_L2) {
                memcpy(cand_norms.data() + offset, list.norms.data(), cnt * sizeof(float));
            }
            offset += cnt;
        }

        // --- Step 3: GPU distance + top-k ---
        id<MTLBuffer> queryBuf = [device newBufferWithBytes:q
                                                     length:d * sizeof(float)
                                                    options:MTLResourceStorageModeShared];

        id<MTLBuffer> candBuf = [device newBufferWithBytes:cand_vecs.data()
                                                    length:total_candidates * d * sizeof(float)
                                                   options:MTLResourceStorageModeShared];

        id<MTLBuffer> normBuf = nil;
        if (metric_type == faiss::METRIC_L2) {
            normBuf = [device newBufferWithBytes:cand_norms.data()
                                          length:total_candidates * sizeof(float)
                                         options:MTLResourceStorageModeShared];
        }

        faiss::idx_t effective_k = std::min(k, (faiss::idx_t)total_candidates);

        id<MTLBuffer> outDistBuf = [device newBufferWithLength:effective_k * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> outIdxBuf = [device newBufferWithLength:effective_k * sizeof(int32_t)
                                                      options:MTLResourceStorageModeShared];

        id<MTLBuffer> queryNormBuf = nil;
        if (metric_type == faiss::METRIC_L2) {
            queryNormBuf = [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModePrivate];
        }

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];

        bool usedFused = impl_->distance->encodeFused(cmdBuf, queryBuf, candBuf, normBuf, queryNormBuf, outDistBuf,
                                                      outIdxBuf, 1, total_candidates, d, effective_k, metric_type);

        if (!usedFused) {
            size_t distBytes = total_candidates * sizeof(float);
            id<MTLBuffer> distBuf = [device newBufferWithLength:distBytes options:MTLResourceStorageModePrivate];

            impl_->distance->encode(cmdBuf, queryBuf, candBuf, normBuf, queryNormBuf, distBuf, 1, total_candidates, d,
                                    metric_type);
            impl_->selector->encode(cmdBuf, distBuf, outDistBuf, outIdxBuf, 1, total_candidates, effective_k,
                                    metric_type);
        }

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        // --- Step 4: Copy results and remap indices ---
        float *outDistPtr = (float *)[outDistBuf contents];
        int32_t *outIdxPtr = (int32_t *)[outIdxBuf contents];

        for (faiss::idx_t j = 0; j < effective_k; j++) {
            out_dist[j] = outDistPtr[j];
            int32_t cand_idx = outIdxPtr[j];
            out_labels[j] = (cand_idx >= 0 && cand_idx < (int32_t)total_candidates) ? cand_ids[cand_idx] : -1;
        }
        for (faiss::idx_t j = effective_k; j < k; j++) {
            out_dist[j] = sentinel_dist;
            out_labels[j] = -1;
        }
    }
}

void MetalIndexIVFFlat::reset() {
    ntotal = 0;
    for (auto &list : impl_->invlists) {
        list.ids.clear();
        list.vectors.clear();
        list.norms.clear();
    }
}

size_t MetalIndexIVFFlat::getNlist() const {
    return impl_->nlist_val;
}

size_t MetalIndexIVFFlat::getNprobe() const {
    return impl_->nprobe_val;
}

void MetalIndexIVFFlat::setNprobe(size_t nprobe) {
    FAISS_THROW_IF_NOT_MSG(nprobe > 0, "nprobe must be > 0");
    FAISS_THROW_IF_NOT_MSG(nprobe <= impl_->nlist_val, "nprobe cannot exceed nlist");
    impl_->nprobe_val = nprobe;
}

// --- Conversion helpers ---

std::unique_ptr<MetalIndexIVFFlat> index_cpu_to_metal_ivf(std::shared_ptr<MetalResources> resources,
                                                          const faiss::IndexIVFFlat *cpu_index) {

    FAISS_THROW_IF_NOT_MSG(cpu_index->is_trained, "CPU IndexIVFFlat must be trained");

    auto metal = std::make_unique<MetalIndexIVFFlat>(resources, cpu_index->d, cpu_index->nlist, cpu_index->metric_type);

    // Copy centroids from CPU quantizer into our FAISS IndexFlat quantizer
    auto *quantizer = dynamic_cast<const faiss::IndexFlat *>(cpu_index->quantizer);
    FAISS_THROW_IF_NOT_MSG(quantizer, "IndexIVFFlat quantizer must be IndexFlat");

    metal->impl_->quantizer->reset();
    metal->impl_->quantizer->add(quantizer->ntotal, quantizer->get_xb());
    metal->is_trained = true;

    // Copy inverted lists
    const faiss::InvertedLists *invlists = cpu_index->invlists;
    for (size_t c = 0; c < cpu_index->nlist; c++) {
        size_t list_size = invlists->list_size(c);
        if (list_size == 0)
            continue;

        const faiss::idx_t *ids = invlists->get_ids(c);
        const uint8_t *codes = invlists->get_codes(c);
        const float *vectors = (const float *)codes;

        auto &list = metal->impl_->invlists[c];
        list.ids.assign(ids, ids + list_size);
        list.vectors.assign(vectors, vectors + list_size * cpu_index->d);

        // Precompute norms for L2
        if (cpu_index->metric_type == faiss::METRIC_L2) {
            list.norms.resize(list_size);
            for (size_t i = 0; i < list_size; i++) {
                list.norms[i] = MetalIndexIVFFlat::Impl::compute_norm(vectors + i * cpu_index->d, cpu_index->d);
            }
        }
    }

    metal->ntotal = cpu_index->ntotal;
    metal->impl_->nprobe_val = cpu_index->nprobe;

    return metal;
}

std::unique_ptr<faiss::IndexIVFFlat> index_metal_to_cpu_ivf(const MetalIndexIVFFlat *metal_index) {

    FAISS_THROW_IF_NOT_MSG(metal_index->is_trained, "Metal index must be trained");

    int dim = metal_index->d;
    size_t nlist = metal_index->getNlist();

    // Create quantizer with centroids from our FAISS quantizer
    auto *quantizer = new faiss::IndexFlat(dim, metal_index->metric_type);
    quantizer->add(metal_index->impl_->quantizer->ntotal, metal_index->impl_->quantizer->get_xb());

    auto cpu_index = std::make_unique<faiss::IndexIVFFlat>(quantizer, dim, nlist, metal_index->metric_type);
    cpu_index->own_fields = true; // takes ownership of quantizer
    cpu_index->is_trained = true;
    cpu_index->nprobe = metal_index->getNprobe();

    // Copy inverted lists
    for (size_t c = 0; c < nlist; c++) {
        auto &list = metal_index->impl_->invlists[c];
        if (list.ids.empty())
            continue;

        size_t list_size = list.ids.size();
        cpu_index->invlists->add_entries(c, list_size, list.ids.data(), (const uint8_t *)list.vectors.data());
    }

    cpu_index->ntotal = metal_index->ntotal;
    return cpu_index;
}

} // namespace faiss_metal
