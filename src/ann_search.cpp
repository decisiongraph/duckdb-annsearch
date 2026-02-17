#include "diskann_index.hpp"
#include "annsearch_extension.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"

#ifdef FAISS_AVAILABLE
#include "faiss_index.hpp"
#endif

namespace duckdb {

// ========================================
// ann_search(table, index, query, k)
// ========================================
// Convenience function: performs index scan + row fetch in one call.
// Returns all table columns + a distance column, ordered by distance.

struct AnnSearchBindData : public TableFunctionData {
	string table_name;
	string index_name;
	vector<float> query;
	int32_t k;
	int32_t search_complexity = 0;
	int32_t oversample = 1;

	// Resolved at bind time
	vector<string> column_names;
	vector<LogicalType> column_types;
	vector<StorageIndex> storage_ids;
};

struct AnnSearchState : public GlobalTableFunctionState {
	vector<pair<row_t, float>> results;
	idx_t offset = 0;
	bool fetched = false;

	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> AnnSearchBind(ClientContext &context, TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<AnnSearchBindData>();
	bind_data->table_name = input.inputs[0].GetValue<string>();
	bind_data->index_name = input.inputs[1].GetValue<string>();

	auto &query_list = ListValue::GetChildren(input.inputs[2]);
	for (auto &v : query_list) {
		bind_data->query.push_back(v.GetValue<float>());
	}
	bind_data->k = input.inputs[3].GetValue<int32_t>();

	for (auto &kv : input.named_parameters) {
		if (kv.first == "search_complexity") {
			bind_data->search_complexity = kv.second.GetValue<int32_t>();
		} else if (kv.first == "oversample") {
			bind_data->oversample = MaxValue<int32_t>(1, kv.second.GetValue<int32_t>());
		}
	}

	// Look up the table to get its columns
	auto &catalog = Catalog::GetCatalog(context, "");
	auto &duck_table =
	    catalog.GetEntry<TableCatalogEntry>(context, DEFAULT_SCHEMA, bind_data->table_name).Cast<DuckTableEntry>();
	auto &columns = duck_table.GetColumns();

	// Add all physical columns from the table
	for (auto &col : columns.Physical()) {
		bind_data->column_names.push_back(col.Name());
		bind_data->column_types.push_back(col.Type());
		bind_data->storage_ids.emplace_back(columns.LogicalToPhysical(col.Logical()).index);

		names.push_back(col.Name());
		return_types.push_back(col.Type());
	}

	// Add distance column
	names.push_back("_distance");
	return_types.push_back(LogicalType::FLOAT);

	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> AnnSearchInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<AnnSearchState>();
}

static void AnnSearchScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind = data.bind_data->Cast<AnnSearchBindData>();
	auto &state = data.global_state->Cast<AnnSearchState>();

	if (!state.fetched) {
		state.fetched = true;

		// Find the table and index
		auto &catalog = Catalog::GetCatalog(context, "");
		auto &duck_table =
		    catalog.GetEntry<TableCatalogEntry>(context, DEFAULT_SCHEMA, bind.table_name).Cast<DuckTableEntry>();
		auto &storage = duck_table.GetStorage();
		auto &indexes = storage.GetDataTableInfo()->GetIndexes();

		auto fetch_k = bind.k * bind.oversample;

		bool found = false;
		indexes.Scan([&](Index &idx) {
			if (idx.GetIndexName() != bind.index_name) {
				return false;
			}
			auto &bound = idx.Cast<BoundIndex>();

			auto *diskann = dynamic_cast<DiskannIndex *>(&bound);
			if (diskann) {
				state.results = diskann->Search(bind.query.data(), static_cast<int32_t>(bind.query.size()), fetch_k,
				                                bind.search_complexity);
				found = true;
				return true;
			}

#ifdef FAISS_AVAILABLE
			auto *faiss = dynamic_cast<FaissIndex *>(&bound);
			if (faiss) {
				state.results = faiss->Search(bind.query.data(), static_cast<int32_t>(bind.query.size()), fetch_k);
				found = true;
				return true;
			}
#endif

			return false;
		});

		if (!found) {
			throw InvalidInputException("ANN index '%s' not found on table '%s'", bind.index_name, bind.table_name);
		}
	}

	if (state.offset >= state.results.size()) {
		output.SetCardinality(0);
		return;
	}

	// Fetch rows from the table by row_id
	auto &catalog = Catalog::GetCatalog(context, "");
	auto &duck_table =
	    catalog.GetEntry<TableCatalogEntry>(context, DEFAULT_SCHEMA, bind.table_name).Cast<DuckTableEntry>();
	auto &storage = duck_table.GetStorage();
	auto &transaction = DuckTransaction::Get(context, storage.db);

	auto num_table_cols = bind.column_types.size();
	auto distance_col_idx = num_table_cols;

	auto batch_size = MinValue<idx_t>(state.results.size() - state.offset, STANDARD_VECTOR_SIZE);

	// Build row_id vector
	Vector row_ids_vec(LogicalType::ROW_TYPE, batch_size);
	auto row_ids_data = FlatVector::GetData<row_t>(row_ids_vec);
	for (idx_t i = 0; i < batch_size; i++) {
		row_ids_data[i] = state.results[state.offset + i].first;
	}

	// Fetch rows from table
	DataChunk fetch_chunk;
	fetch_chunk.Initialize(context, bind.column_types);
	ColumnFetchState fetch_state;
	storage.Fetch(transaction, fetch_chunk, bind.storage_ids, row_ids_vec, batch_size, fetch_state);

	// Copy fetched columns to output
	for (idx_t col = 0; col < num_table_cols; col++) {
		for (idx_t i = 0; i < batch_size; i++) {
			output.data[col].SetValue(i, fetch_chunk.data[col].GetValue(i));
		}
	}

	// Set distance column
	for (idx_t i = 0; i < batch_size; i++) {
		output.data[distance_col_idx].SetValue(i, Value::FLOAT(state.results[state.offset + i].second));
	}

	state.offset += batch_size;
	output.SetCardinality(batch_size);
}

// ========================================
// ann_search_batch(table, index, queries, k)
// ========================================
// Batch search: accepts LIST of query vectors, returns query_idx + table columns + distance.

struct AnnSearchBatchBindData : public TableFunctionData {
	string table_name;
	string index_name;
	vector<vector<float>> queries;
	int32_t k;
	int32_t search_complexity = 0;

	vector<string> column_names;
	vector<LogicalType> column_types;
	vector<StorageIndex> storage_ids;
};

struct AnnSearchBatchState : public GlobalTableFunctionState {
	// Flattened results: (query_idx, row_id, distance)
	struct BatchResult {
		int32_t query_idx;
		row_t row_id;
		float distance;
	};
	vector<BatchResult> results;
	idx_t offset = 0;
	bool fetched = false;

	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> AnnSearchBatchBind(ClientContext &context, TableFunctionBindInput &input,
                                                   vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<AnnSearchBatchBindData>();
	bind_data->table_name = input.inputs[0].GetValue<string>();
	bind_data->index_name = input.inputs[1].GetValue<string>();

	// Parse LIST of LIST of FLOAT → vector<vector<float>>
	auto &outer_list = ListValue::GetChildren(input.inputs[2]);
	for (auto &inner : outer_list) {
		auto &inner_list = ListValue::GetChildren(inner);
		vector<float> query;
		for (auto &v : inner_list) {
			query.push_back(v.GetValue<float>());
		}
		bind_data->queries.push_back(std::move(query));
	}
	bind_data->k = input.inputs[3].GetValue<int32_t>();

	for (auto &kv : input.named_parameters) {
		if (kv.first == "search_complexity") {
			bind_data->search_complexity = kv.second.GetValue<int32_t>();
		}
	}

	// query_idx column first
	names.push_back("query_idx");
	return_types.push_back(LogicalType::INTEGER);

	// Look up the table to get its columns
	auto &catalog = Catalog::GetCatalog(context, "");
	auto &duck_table =
	    catalog.GetEntry<TableCatalogEntry>(context, DEFAULT_SCHEMA, bind_data->table_name).Cast<DuckTableEntry>();
	auto &columns = duck_table.GetColumns();

	for (auto &col : columns.Physical()) {
		bind_data->column_names.push_back(col.Name());
		bind_data->column_types.push_back(col.Type());
		bind_data->storage_ids.emplace_back(columns.LogicalToPhysical(col.Logical()).index);
		names.push_back(col.Name());
		return_types.push_back(col.Type());
	}

	names.push_back("_distance");
	return_types.push_back(LogicalType::FLOAT);

	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> AnnSearchBatchInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<AnnSearchBatchState>();
}

static void AnnSearchBatchScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind = data.bind_data->Cast<AnnSearchBatchBindData>();
	auto &state = data.global_state->Cast<AnnSearchBatchState>();

	if (!state.fetched) {
		state.fetched = true;

		auto &catalog = Catalog::GetCatalog(context, "");
		auto &duck_table =
		    catalog.GetEntry<TableCatalogEntry>(context, DEFAULT_SCHEMA, bind.table_name).Cast<DuckTableEntry>();
		auto &storage = duck_table.GetStorage();
		auto &indexes = storage.GetDataTableInfo()->GetIndexes();

		for (int32_t qi = 0; qi < static_cast<int32_t>(bind.queries.size()); qi++) {
			auto &query = bind.queries[qi];
			vector<pair<row_t, float>> results;

			indexes.Scan([&](Index &idx) {
				if (idx.GetIndexName() != bind.index_name) {
					return false;
				}
				auto &bound = idx.Cast<BoundIndex>();

				auto *diskann = dynamic_cast<DiskannIndex *>(&bound);
				if (diskann) {
					results = diskann->Search(query.data(), static_cast<int32_t>(query.size()), bind.k,
					                          bind.search_complexity);
					return true;
				}

#ifdef FAISS_AVAILABLE
				auto *faiss = dynamic_cast<FaissIndex *>(&bound);
				if (faiss) {
					results = faiss->Search(query.data(), static_cast<int32_t>(query.size()), bind.k);
					return true;
				}
#endif
				return false;
			});

			for (auto &[row_id, dist] : results) {
				state.results.push_back({qi, row_id, dist});
			}
		}
	}

	if (state.offset >= state.results.size()) {
		output.SetCardinality(0);
		return;
	}

	auto &catalog = Catalog::GetCatalog(context, "");
	auto &duck_table =
	    catalog.GetEntry<TableCatalogEntry>(context, DEFAULT_SCHEMA, bind.table_name).Cast<DuckTableEntry>();
	auto &storage = duck_table.GetStorage();
	auto &transaction = DuckTransaction::Get(context, storage.db);

	auto num_table_cols = bind.column_types.size();
	auto batch_size = MinValue<idx_t>(state.results.size() - state.offset, STANDARD_VECTOR_SIZE);

	// Build row_id vector
	Vector row_ids_vec(LogicalType::ROW_TYPE, batch_size);
	auto row_ids_data = FlatVector::GetData<row_t>(row_ids_vec);
	for (idx_t i = 0; i < batch_size; i++) {
		row_ids_data[i] = state.results[state.offset + i].row_id;
	}

	// Fetch rows from table
	DataChunk fetch_chunk;
	fetch_chunk.Initialize(context, bind.column_types);
	ColumnFetchState fetch_state;
	storage.Fetch(transaction, fetch_chunk, bind.storage_ids, row_ids_vec, batch_size, fetch_state);

	// Column 0: query_idx
	for (idx_t i = 0; i < batch_size; i++) {
		output.data[0].SetValue(i, Value::INTEGER(state.results[state.offset + i].query_idx));
	}

	// Columns 1..N: table columns
	for (idx_t col = 0; col < num_table_cols; col++) {
		for (idx_t i = 0; i < batch_size; i++) {
			output.data[col + 1].SetValue(i, fetch_chunk.data[col].GetValue(i));
		}
	}

	// Last column: distance
	auto dist_col = num_table_cols + 1;
	for (idx_t i = 0; i < batch_size; i++) {
		output.data[dist_col].SetValue(i, Value::FLOAT(state.results[state.offset + i].distance));
	}

	state.offset += batch_size;
	output.SetCardinality(batch_size);
}

void RegisterAnnSearchFunction(ExtensionLoader &loader) {
	// Single-query search
	TableFunction func(
	    "ann_search",
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::LIST(LogicalType::FLOAT), LogicalType::INTEGER},
	    AnnSearchScan, AnnSearchBind, AnnSearchInit);
	func.named_parameters["search_complexity"] = LogicalType::INTEGER;
	func.named_parameters["oversample"] = LogicalType::INTEGER;
	loader.RegisterFunction(func);

	// Batch search: LIST of LIST of FLOAT
	TableFunction batch_func("ann_search_batch",
	                         {LogicalType::VARCHAR, LogicalType::VARCHAR,
	                          LogicalType::LIST(LogicalType::LIST(LogicalType::FLOAT)), LogicalType::INTEGER},
	                         AnnSearchBatchScan, AnnSearchBatchBind, AnnSearchBatchInit);
	batch_func.named_parameters["search_complexity"] = LogicalType::INTEGER;
	loader.RegisterFunction(batch_func);
}

} // namespace duckdb
