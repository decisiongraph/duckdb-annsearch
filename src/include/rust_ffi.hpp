#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace duckdb {

// ========================================
// Detached handle API (for BoundIndex)
// ========================================

// Opaque handle to a Rust InMemoryIndex
typedef void *DiskannHandle;

// Create a detached index (not in global registry). Returns handle, throws on error.
DiskannHandle DiskannCreateDetached(int32_t dimension, const std::string &metric, int32_t max_degree,
                                    int32_t build_complexity, float alpha);

// Free a detached index handle.
void DiskannFreeDetached(DiskannHandle handle);

// Add vector to detached index. Returns assigned label.
int64_t DiskannDetachedAdd(DiskannHandle handle, const float *vector, int32_t dimension);

// Search detached index. Returns number of results.
int32_t DiskannDetachedSearch(DiskannHandle handle, const float *query, int32_t dimension, int32_t k,
                              int32_t search_complexity, int64_t *out_labels, float *out_distances);

// Get vector count.
int64_t DiskannDetachedCount(DiskannHandle handle);

// Multi-query batch search on detached index. Returns 0 on success, -1 on error.
// Uses GPU-accelerated lock-step BFS when Metal is available and batch is large enough.
int32_t DiskannDetachedSearchBatch(DiskannHandle handle, const float *query_matrix, int32_t nq, int32_t dimension,
                                   int32_t k, int32_t search_complexity, int64_t *out_labels, float *out_distances,
                                   int32_t *out_counts);

// Serialize detached index to bytes. Caller must free with DiskannFreeSerializedBytes.
struct DiskannSerializedData {
	uint8_t *data;
	size_t len;
};
DiskannSerializedData DiskannDetachedSerialize(DiskannHandle handle);

// Deserialize bytes into a detached index. Returns handle.
DiskannHandle DiskannDetachedDeserialize(const uint8_t *data, size_t len, float alpha);

// Free serialized bytes.
void DiskannFreeSerializedBytes(DiskannSerializedData bytes);

// ========================================
// Compact / vacuum
// ========================================

struct DiskannCompactResult {
	DiskannHandle new_handle; // New index (caller owns). Null on error.
	uint32_t *label_map;      // [old0, new0, old1, new1, ...] pairs
	size_t map_len;           // Number of pairs
};

// Compact a detached index by rebuilding without deleted labels.
// Returns new handle + label map. Old handle is NOT freed.
DiskannCompactResult DiskannDetachedCompact(DiskannHandle handle, const uint32_t *deleted_labels, size_t num_deleted);

// Free the label map from DiskannDetachedCompact.
void DiskannFreeLabelMap(uint32_t *map, size_t map_len);

// ========================================
// Vector accessor (for MergeIndexes)
// ========================================

// Get a vector by label. Returns dimension copied, or 0 if not found.
int32_t DiskannDetachedGetVector(DiskannHandle handle, uint32_t label, float *out_vec, int32_t capacity);

// ========================================
// Streaming build API
// ========================================

struct DiskannStreamingBuildResult {
	int32_t num_vectors;
	int32_t dimension;
	int32_t sample_size;
};

// Run two-pass streaming build from binary vectors file to .diskann index file.
DiskannStreamingBuildResult DiskannStreamingBuild(const std::string &input_path, const std::string &output_path,
                                                  const std::string &metric, int32_t max_degree,
                                                  int32_t build_complexity, float alpha, int32_t sample_size);

// SQ8 Quantization
void DiskannDetachedQuantizeSQ8(DiskannHandle handle);
bool DiskannDetachedIsQuantized(DiskannHandle handle);

// ========================================
// Batch search (multi-query, GPU-accelerated for DiskIndex)
// ========================================

struct DiskannBatchSearchResult {
	std::vector<int64_t> labels;  // nq * k (row-major, -1 for unfilled)
	std::vector<float> distances; // nq * k (row-major, MAX for unfilled)
	std::vector<int32_t> counts;  // nq — actual result count per query
};

// Multi-query batch search via global registry index.
// Uses GPU-accelerated lock-step search for DiskIndex, sequential for InMemoryIndex.
DiskannBatchSearchResult DiskannBatchSearch(const std::string &name, const float *query_matrix, int32_t nq,
                                            int32_t dimension, int32_t k, int32_t search_complexity);

} // namespace duckdb
