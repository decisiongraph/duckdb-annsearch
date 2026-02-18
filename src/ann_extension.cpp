#define DUCKDB_EXTENSION_MAIN

#include "ann_extension.hpp"
#include "diskann_index.hpp"
#include "duckdb.hpp"
#include "duckdb/execution/index/index_type.hpp"
#include "duckdb/execution/index/index_type_set.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/database.hpp"

#ifdef FAISS_AVAILABLE
#include "faiss_index.hpp"
#endif

namespace duckdb {

static void LoadInternal(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();

	// ========================================
	// Register DISKANN index type
	// ========================================
	IndexType diskann_type;
	diskann_type.name = DiskannIndex::TYPE_NAME;
	diskann_type.create_instance = DiskannIndex::Create;
	diskann_type.create_plan = DiskannIndex::CreatePlan;
	db.config.GetIndexTypes().RegisterIndexType(diskann_type);

	// DiskANN functions
	RegisterDiskannIndexScanFunction(loader);
	RegisterDiskannStreamingBuildFunction(loader);

#ifdef FAISS_AVAILABLE
	// ========================================
	// Register FAISS index type
	// ========================================
	IndexType faiss_type;
	faiss_type.name = FaissIndex::TYPE_NAME;
	faiss_type.create_instance = FaissIndex::Create;
	faiss_type.create_plan = FaissIndex::CreatePlan;
	db.config.GetIndexTypes().RegisterIndexType(faiss_type);

	// FAISS BoundIndex search function
	RegisterFaissIndexScanFunction(loader);

	// FAISS GPU info
	RegisterFaissGpuFunctions(loader);
#endif

	// Convenience ann_search function (always available)
	RegisterAnnSearchFunction(loader);

	// Unified listing (always available)
	RegisterAnnListFunction(loader);

	// Extension settings
	auto &config = DBConfig::GetConfig(db);
	config.AddExtensionOption("ann_overfetch_multiplier",
	                          "Multiplier for ANN index scan overfetch when filters are present (default 3)",
	                          LogicalType::BIGINT, Value::BIGINT(3));

	// Optimizer: ORDER BY array_distance(...) LIMIT k → ANN index scan
	RegisterAnnOptimizer(db);
}

void AnnExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string AnnExtension::Name() {
	return "ann";
}

std::string AnnExtension::Version() const {
#ifdef EXT_VERSION_ANN
	return EXT_VERSION_ANN;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(ann, loader) {
	duckdb::LoadInternal(loader);
}
}
