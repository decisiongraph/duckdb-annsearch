#pragma once

#include "duckdb.hpp"

namespace duckdb {

class AnnExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override;
	std::string Name() override;
	std::string Version() const override;
};

// DiskANN function registration (always available)
void RegisterDiskannIndexScanFunction(ExtensionLoader &loader);
void RegisterDiskannStreamingBuildFunction(ExtensionLoader &loader);

// Convenience search (works with both DISKANN and FAISS indexes)
void RegisterAnnSearchFunction(ExtensionLoader &loader);

// Unified listing
void RegisterAnnListFunction(ExtensionLoader &loader);

// Optimizer: ORDER BY array_distance(...) LIMIT k → ANN index scan
void RegisterAnnOptimizer(DatabaseInstance &db);

#ifdef FAISS_AVAILABLE
// FAISS BoundIndex scan function
void RegisterFaissIndexScanFunction(ExtensionLoader &loader);

// FAISS GPU info
void RegisterFaissGpuFunctions(ExtensionLoader &loader);
#endif

} // namespace duckdb
