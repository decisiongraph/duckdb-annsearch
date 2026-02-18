// Rust DiskANN FFI wrapper for DuckDB extension (detached handle API only)

#include "rust_ffi.hpp"
#include <limits>
#include <stdexcept>
#include <string>

// ========================================
// Rust FFI declarations
// ========================================
extern "C" {

// Streaming build FFI
int32_t diskann_streaming_build_buf(const char *input_path, const char *output_path, const char *metric,
                                    int32_t max_degree, int32_t build_complexity, float alpha, int32_t sample_size,
                                    int32_t *out_num_vectors, int32_t *out_dimension, int32_t *out_sample_size,
                                    char *err_buf, int32_t err_buf_len);

// Detached handle FFI
struct DiskannBytes {
	uint8_t *data;
	size_t len;
};

void *diskann_create_detached(int32_t dimension, const char *metric, int32_t max_degree, int32_t build_complexity,
                              float alpha, char *err_buf, int32_t err_buf_len);

void diskann_free_detached(void *handle);

int64_t diskann_detached_add(void *handle, const float *vector_ptr, int32_t dimension, char *err_buf,
                             int32_t err_buf_len);

int32_t diskann_detached_search(void *handle, const float *query_ptr, int32_t dimension, int32_t k,
                                int32_t search_complexity, int64_t *out_labels, float *out_distances, char *err_buf,
                                int32_t err_buf_len);

int64_t diskann_detached_count(void *handle);

DiskannBytes diskann_detached_serialize(void *handle, char *err_buf, int32_t err_buf_len);

void *diskann_detached_deserialize(const uint8_t *data, size_t len, float alpha, char *err_buf, int32_t err_buf_len);

void diskann_free_bytes(DiskannBytes bytes);

// Compact / vacuum
struct DiskannCompactResultFFI {
	void *new_handle;
	uint32_t *label_map;
	size_t map_len;
};

DiskannCompactResultFFI diskann_detached_compact(void *handle, const uint32_t *deleted_labels, size_t num_deleted,
                                                 char *err_buf, int32_t err_buf_len);

void diskann_free_label_map(uint32_t *map, size_t map_len);

// Vector accessor
int32_t diskann_detached_get_vector(void *handle, uint32_t label, float *out_vec, int32_t out_capacity);

// SQ8 Quantization
int32_t diskann_detached_quantize_sq8(void *handle);
int32_t diskann_detached_is_quantized(void *handle);

// Detached batch search (multi-query, GPU-accelerated)
int32_t diskann_detached_search_batch(void *handle, const float *query_matrix, int32_t nq, int32_t dimension, int32_t k,
                                      int32_t search_complexity, int64_t *out_labels, float *out_distances,
                                      int32_t *out_counts, char *err_buf, int32_t err_buf_len);

// Batch search (multi-query, global registry)
int32_t diskann_batch_search_buf(const char *name, const float *query_matrix, int32_t nq, int32_t dimension, int32_t k,
                                 int32_t search_complexity, int64_t *out_labels, float *out_distances,
                                 int32_t *out_counts, char *err_buf, int32_t err_buf_len);
}

namespace duckdb {

constexpr int ERR_BUF_LEN = 512;

// ========================================
// Streaming build wrapper
// ========================================

DiskannStreamingBuildResult DiskannStreamingBuild(const std::string &input_path, const std::string &output_path,
                                                  const std::string &metric, int32_t max_degree,
                                                  int32_t build_complexity, float alpha, int32_t sample_size) {
	char err_buf[ERR_BUF_LEN] = {0};
	int32_t out_num_vectors = 0;
	int32_t out_dimension = 0;
	int32_t out_sample_size = 0;
	int32_t rc = diskann_streaming_build_buf(input_path.c_str(), output_path.c_str(), metric.c_str(), max_degree,
	                                         build_complexity, alpha, sample_size, &out_num_vectors, &out_dimension,
	                                         &out_sample_size, err_buf, ERR_BUF_LEN);
	if (rc != 0) {
		throw std::runtime_error("DiskANN streaming build: " + std::string(err_buf));
	}
	return {out_num_vectors, out_dimension, out_sample_size};
}

// ========================================
// Detached handle wrappers
// ========================================

DiskannHandle DiskannCreateDetached(int32_t dimension, const std::string &metric, int32_t max_degree,
                                    int32_t build_complexity, float alpha) {
	char err_buf[ERR_BUF_LEN] = {0};
	auto handle =
	    diskann_create_detached(dimension, metric.c_str(), max_degree, build_complexity, alpha, err_buf, ERR_BUF_LEN);
	if (!handle) {
		throw std::runtime_error("DiskANN create detached: " + std::string(err_buf));
	}
	return handle;
}

void DiskannFreeDetached(DiskannHandle handle) {
	diskann_free_detached(handle);
}

int64_t DiskannDetachedAdd(DiskannHandle handle, const float *vector, int32_t dimension) {
	char err_buf[ERR_BUF_LEN] = {0};
	int64_t label = diskann_detached_add(handle, vector, dimension, err_buf, ERR_BUF_LEN);
	if (label < 0) {
		throw std::runtime_error("DiskANN detached add: " + std::string(err_buf));
	}
	return label;
}

int32_t DiskannDetachedSearch(DiskannHandle handle, const float *query, int32_t dimension, int32_t k,
                              int32_t search_complexity, int64_t *out_labels, float *out_distances) {
	char err_buf[ERR_BUF_LEN] = {0};
	int32_t n = diskann_detached_search(handle, query, dimension, k, search_complexity, out_labels, out_distances,
	                                    err_buf, ERR_BUF_LEN);
	if (n < 0) {
		throw std::runtime_error("DiskANN detached search: " + std::string(err_buf));
	}
	return n;
}

int64_t DiskannDetachedCount(DiskannHandle handle) {
	return diskann_detached_count(handle);
}

DiskannSerializedData DiskannDetachedSerialize(DiskannHandle handle) {
	char err_buf[ERR_BUF_LEN] = {0};
	auto result = diskann_detached_serialize(handle, err_buf, ERR_BUF_LEN);
	if (!result.data) {
		throw std::runtime_error("DiskANN serialize: " + std::string(err_buf));
	}
	return {result.data, result.len};
}

DiskannHandle DiskannDetachedDeserialize(const uint8_t *data, size_t len, float alpha) {
	char err_buf[ERR_BUF_LEN] = {0};
	auto handle = diskann_detached_deserialize(data, len, alpha, err_buf, ERR_BUF_LEN);
	if (!handle) {
		throw std::runtime_error("DiskANN deserialize: " + std::string(err_buf));
	}
	return handle;
}

void DiskannFreeSerializedBytes(DiskannSerializedData bytes) {
	DiskannBytes raw_bytes;
	raw_bytes.data = bytes.data;
	raw_bytes.len = bytes.len;
	diskann_free_bytes(raw_bytes);
}

// ========================================
// Compact / vacuum wrappers
// ========================================

DiskannCompactResult DiskannDetachedCompact(DiskannHandle handle, const uint32_t *deleted_labels, size_t num_deleted) {
	char err_buf[ERR_BUF_LEN] = {0};
	auto result = diskann_detached_compact(handle, deleted_labels, num_deleted, err_buf, ERR_BUF_LEN);
	if (!result.new_handle) {
		throw std::runtime_error("DiskANN compact: " + std::string(err_buf));
	}
	return {result.new_handle, result.label_map, result.map_len};
}

void DiskannFreeLabelMap(uint32_t *map, size_t map_len) {
	diskann_free_label_map(map, map_len);
}

// ========================================
// Vector accessor wrapper
// ========================================

int32_t DiskannDetachedGetVector(DiskannHandle handle, uint32_t label, float *out_vec, int32_t capacity) {
	return diskann_detached_get_vector(handle, label, out_vec, capacity);
}

// ========================================
// SQ8 Quantization wrappers
// ========================================

void DiskannDetachedQuantizeSQ8(DiskannHandle handle) {
	diskann_detached_quantize_sq8(handle);
}

bool DiskannDetachedIsQuantized(DiskannHandle handle) {
	return diskann_detached_is_quantized(handle) != 0;
}

// ========================================
// Detached batch search wrapper
// ========================================

int32_t DiskannDetachedSearchBatch(DiskannHandle handle, const float *query_matrix, int32_t nq, int32_t dimension,
                                   int32_t k, int32_t search_complexity, int64_t *out_labels, float *out_distances,
                                   int32_t *out_counts) {
	char err_buf[ERR_BUF_LEN] = {0};
	int32_t rc = diskann_detached_search_batch(handle, query_matrix, nq, dimension, k, search_complexity, out_labels,
	                                           out_distances, out_counts, err_buf, ERR_BUF_LEN);
	if (rc != 0) {
		throw std::runtime_error("DiskANN detached batch search: " + std::string(err_buf));
	}
	return 0;
}

// ========================================
// Global registry batch search wrapper
// ========================================

DiskannBatchSearchResult DiskannBatchSearch(const std::string &name, const float *query_matrix, int32_t nq,
                                            int32_t dimension, int32_t k, int32_t search_complexity) {
	char err_buf[ERR_BUF_LEN] = {0};
	DiskannBatchSearchResult result;
	result.labels.resize(static_cast<size_t>(nq) * k, -1);
	result.distances.resize(static_cast<size_t>(nq) * k, std::numeric_limits<float>::max());
	result.counts.resize(nq, 0);

	int32_t rc =
	    diskann_batch_search_buf(name.c_str(), query_matrix, nq, dimension, k, search_complexity, result.labels.data(),
	                             result.distances.data(), result.counts.data(), err_buf, ERR_BUF_LEN);
	if (rc != 0) {
		throw std::runtime_error("DiskANN batch search: " + std::string(err_buf));
	}
	return result;
}

} // namespace duckdb
