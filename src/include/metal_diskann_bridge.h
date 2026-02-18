#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/// Returns 1 if Metal DiskANN acceleration is available, 0 otherwise.
int diskann_metal_available(void);

/// Compute batch distances: query (dim,) vs candidates (n*dim contiguous floats).
/// metric: 0=L2, 1=IP
/// Results written to out_distances (n floats).
/// Returns 0 on success, -1 on error.
int diskann_metal_batch_distances(const float *query, const float *candidates, int n, int dim, int metric,
                                  float *out_distances);

/// Multi-query batch distances: nq queries, total_n candidates with query_map.
/// queries: (nq * dim) contiguous floats — all query vectors.
/// candidates: (total_n * dim) contiguous floats — all candidate vectors.
/// query_map: (total_n) uint32s — query_map[i] = which query candidate i belongs to.
/// Returns 0 on success, -1 on error.
int diskann_metal_multi_batch_distances(const float *queries, const float *candidates, const unsigned int *query_map,
                                        int total_n, int nq, int dim, int metric, float *out_distances);

#ifdef __cplusplus
}
#endif
