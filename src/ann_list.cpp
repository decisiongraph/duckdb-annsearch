#include "ann_extension.hpp"
#include "diskann_index.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/index_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/storage/data_table.hpp"

#ifdef FAISS_AVAILABLE
#include "faiss_index.hpp"
#endif

namespace duckdb {

// ========================================
// ann_list()
// Lists all ANN indexes (DISKANN, FAISS) from the DuckDB catalog.
// ========================================

struct AnnListEntry {
	string name;
	string engine;
	string table_name;
};

struct AnnListState : public GlobalTableFunctionState {
	vector<AnnListEntry> entries;
	idx_t position = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> AnnListBind(ClientContext &context, TableFunctionBindInput &input,
                                            vector<LogicalType> &return_types, vector<string> &names) {
	return_types.push_back(LogicalType::VARCHAR);
	return_types.push_back(LogicalType::VARCHAR);
	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("name");
	names.push_back("engine");
	names.push_back("table_name");
	return make_uniq<TableFunctionData>();
}

static unique_ptr<GlobalTableFunctionState> AnnListInit(ClientContext &context, TableFunctionInitInput &input) {
	auto state = make_uniq<AnnListState>();

	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		schema.get().Scan(context, CatalogType::INDEX_ENTRY, [&](CatalogEntry &entry) {
			auto &index_entry = entry.Cast<IndexCatalogEntry>();
			auto &idx_type = index_entry.index_type;
			if (idx_type == "DISKANN" || idx_type == "FAISS") {
				AnnListEntry e;
				e.name = index_entry.name;
				e.engine = idx_type;
				e.table_name = index_entry.GetTableName();
				state->entries.push_back(std::move(e));
			}
		});
	}

	return std::move(state);
}

static void AnnListScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<AnnListState>();

	if (state.position >= state.entries.size()) {
		output.SetCardinality(0);
		return;
	}

	idx_t chunk_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.entries.size() - state.position);

	for (idx_t i = 0; i < chunk_size; i++) {
		auto &entry = state.entries[state.position + i];
		output.SetValue(0, i, Value(entry.name));
		output.SetValue(1, i, Value(entry.engine));
		output.SetValue(2, i, Value(entry.table_name));
	}

	state.position += chunk_size;
	output.SetCardinality(chunk_size);
}

void RegisterAnnListFunction(ExtensionLoader &loader) {
	TableFunction func("ann_list", {}, AnnListScan, AnnListBind, AnnListInit);
	loader.RegisterFunction(func);

	// ========================================
	// ann_index_info()
	// Detailed index diagnostics with stats
	// ========================================

	auto info_bind = [](ClientContext &context, TableFunctionBindInput &input, vector<LogicalType> &return_types,
	                    vector<string> &names) -> unique_ptr<FunctionData> {
		return_types.push_back(LogicalType::VARCHAR);
		names.push_back("name");
		return_types.push_back(LogicalType::VARCHAR);
		names.push_back("engine");
		return_types.push_back(LogicalType::VARCHAR);
		names.push_back("table_name");
		return_types.push_back(LogicalType::BIGINT);
		names.push_back("num_vectors");
		return_types.push_back(LogicalType::BIGINT);
		names.push_back("num_deleted");
		return_types.push_back(LogicalType::BIGINT);
		names.push_back("memory_bytes");
		return_types.push_back(LogicalType::BOOLEAN);
		names.push_back("quantized");
		return make_uniq<TableFunctionData>();
	};

	struct InfoEntry {
		string name;
		string engine;
		string table_name;
		int64_t num_vectors = 0;
		int64_t num_deleted = 0;
		int64_t memory_bytes = 0;
		bool quantized = false;
	};

	struct InfoState : public GlobalTableFunctionState {
		vector<InfoEntry> entries;
		idx_t position = 0;
		idx_t MaxThreads() const override {
			return 1;
		}
	};

	auto info_init = [](ClientContext &context, TableFunctionInitInput &input) -> unique_ptr<GlobalTableFunctionState> {
		auto state = make_uniq<InfoState>();

		auto schemas = Catalog::GetAllSchemas(context);
		for (auto &schema : schemas) {
			schema.get().Scan(context, CatalogType::INDEX_ENTRY, [&](CatalogEntry &entry) {
				auto &index_entry = entry.Cast<IndexCatalogEntry>();
				auto &idx_type = index_entry.index_type;
				if (idx_type != "DISKANN" && idx_type != "FAISS") {
					return;
				}

				InfoEntry e;
				e.name = index_entry.name;
				e.engine = idx_type;
				e.table_name = index_entry.GetTableName();

				// Try to get the table and its BoundIndex for stats
				auto table_entry =
				    Catalog::GetEntry<TableCatalogEntry>(context, index_entry.catalog.GetName(), schema.get().name,
				                                         e.table_name, OnEntryNotFound::RETURN_NULL);
				if (table_entry && table_entry->IsDuckTable()) {
					auto &duck_table = table_entry->Cast<DuckTableEntry>();
					auto &storage = duck_table.GetStorage();
					auto &table_info = *storage.GetDataTableInfo();
					auto &indexes = table_info.GetIndexes();

					if (idx_type == "DISKANN") {
						indexes.Bind(context, table_info, DiskannIndex::TYPE_NAME);
					}
#ifdef FAISS_AVAILABLE
					if (idx_type == "FAISS") {
						indexes.Bind(context, table_info, FaissIndex::TYPE_NAME);
					}
#endif

					auto idx_ptr = indexes.Find(e.name);
					if (idx_ptr) {
						if (idx_type == "DISKANN") {
							auto &diskann = idx_ptr->Cast<DiskannIndex>();
							e.num_vectors = static_cast<int64_t>(diskann.GetVectorCount());
							e.num_deleted = static_cast<int64_t>(diskann.GetDeletedCount());
							auto &bound = static_cast<BoundIndex &>(diskann);
							e.memory_bytes = static_cast<int64_t>(bound.GetInMemorySize());
							e.quantized = diskann.IsQuantized();
						}
#ifdef FAISS_AVAILABLE
						else if (idx_type == "FAISS") {
							auto &faiss = idx_ptr->Cast<FaissIndex>();
							e.num_vectors = static_cast<int64_t>(faiss.GetVectorCount());
							e.num_deleted = static_cast<int64_t>(faiss.GetDeletedCount());
							auto &bound = static_cast<BoundIndex &>(faiss);
							e.memory_bytes = static_cast<int64_t>(bound.GetInMemorySize());
						}
#endif
					}
				}

				state->entries.push_back(std::move(e));
			});
		}

		return std::move(state);
	};

	auto info_scan = [](ClientContext &context, TableFunctionInput &data, DataChunk &output) {
		auto &state = data.global_state->Cast<InfoState>();
		if (state.position >= state.entries.size()) {
			output.SetCardinality(0);
			return;
		}
		idx_t chunk_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.entries.size() - state.position);
		for (idx_t i = 0; i < chunk_size; i++) {
			auto &e = state.entries[state.position + i];
			output.SetValue(0, i, Value(e.name));
			output.SetValue(1, i, Value(e.engine));
			output.SetValue(2, i, Value(e.table_name));
			output.SetValue(3, i, Value::BIGINT(e.num_vectors));
			output.SetValue(4, i, Value::BIGINT(e.num_deleted));
			output.SetValue(5, i, Value::BIGINT(e.memory_bytes));
			output.SetValue(6, i, Value::BOOLEAN(e.quantized));
		}
		state.position += chunk_size;
		output.SetCardinality(chunk_size);
	};

	TableFunction info_func("ann_index_info", {}, info_scan, info_bind, info_init);
	loader.RegisterFunction(info_func);
}

} // namespace duckdb
