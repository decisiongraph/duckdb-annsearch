#include "diskann_index.hpp"
#include "rust_ffi.hpp"
#include "ann_extension.hpp"
#include "metal_diskann_bridge.h"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

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
		auto &table_info = *storage.GetDataTableInfo();
		auto &indexes = table_info.GetIndexes();

		// Bind unbound indexes (needed after database reopen)
		indexes.Bind(context, table_info, DiskannIndex::TYPE_NAME);
#ifdef FAISS_AVAILABLE
		indexes.Bind(context, table_info, FaissIndex::TYPE_NAME);
#endif

		auto fetch_k = bind.k * bind.oversample;

		bool found = false;
		auto idx_ptr = indexes.Find(bind.index_name);
		if (idx_ptr) {
			auto *diskann = dynamic_cast<DiskannIndex *>(idx_ptr.get());
			if (diskann) {
				state.results = diskann->Search(bind.query.data(), static_cast<int32_t>(bind.query.size()), fetch_k,
				                                bind.search_complexity);
				found = true;
			}

#ifdef FAISS_AVAILABLE
			if (!found) {
				auto *faiss = dynamic_cast<FaissIndex *>(idx_ptr.get());
				if (faiss) {
					state.results = faiss->Search(bind.query.data(), static_cast<int32_t>(bind.query.size()), fetch_k);
					found = true;
				}
			}
#endif
		}

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
		auto &table_info = *storage.GetDataTableInfo();
		auto &indexes = table_info.GetIndexes();

		// Bind unbound indexes (needed after database reopen)
		indexes.Bind(context, table_info, DiskannIndex::TYPE_NAME);
#ifdef FAISS_AVAILABLE
		indexes.Bind(context, table_info, FaissIndex::TYPE_NAME);
#endif

		auto idx_ptr = indexes.Find(bind.index_name);
		bool found = false;

		// Try DiskannIndex batch search
		if (idx_ptr) {
			auto *diskann = dynamic_cast<DiskannIndex *>(idx_ptr.get());
			if (diskann) {
				auto batch_results = diskann->SearchBatch(bind.queries, bind.k, bind.search_complexity);
				for (int32_t qi = 0; qi < static_cast<int32_t>(batch_results.size()); qi++) {
					for (auto &pair : batch_results[qi]) {
						state.results.push_back({qi, pair.first, pair.second});
					}
				}
				found = true;
			}
		}

#ifdef FAISS_AVAILABLE
		// FAISS fallback: per-query
		if (!found && idx_ptr) {
			auto *faiss = dynamic_cast<FaissIndex *>(idx_ptr.get());
			if (faiss) {
				for (int32_t qi = 0; qi < static_cast<int32_t>(bind.queries.size()); qi++) {
					auto &query = bind.queries[qi];
					auto results = faiss->Search(query.data(), static_cast<int32_t>(query.size()), bind.k);
					for (auto &pair : results) {
						state.results.push_back({qi, pair.first, pair.second});
					}
				}
				found = true;
			}
		}
#endif

		if (!found) {
			throw InvalidInputException("ANN index '%s' not found on table '%s'", bind.index_name, bind.table_name);
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

// ========================================
// ann_search_table(TABLE queries, table, index, k)
// ========================================
// Streaming batch search: accepts TABLE input (subqueries, CTEs, generate_series, etc.).
// Uses DuckDB's in_out_function pattern for true streaming.
// Returns: input columns + base table columns + _distance.

struct AnnSearchTableBindData : public FunctionData {
	string table_name;
	string index_name;
	int32_t k;
	int32_t search_complexity = 0;
	idx_t vector_col_idx; // which input column has the query vector

	// Base table columns
	vector<string> base_column_names;
	vector<LogicalType> base_column_types;
	vector<StorageIndex> base_storage_ids;

	// Input table columns
	vector<LogicalType> input_types;
	vector<string> input_names;

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<AnnSearchTableBindData>();
		copy->table_name = table_name;
		copy->index_name = index_name;
		copy->k = k;
		copy->search_complexity = search_complexity;
		copy->vector_col_idx = vector_col_idx;
		copy->base_column_names = base_column_names;
		copy->base_column_types = base_column_types;
		copy->base_storage_ids = base_storage_ids;
		copy->input_types = input_types;
		copy->input_names = input_names;
		return std::move(copy);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<AnnSearchTableBindData>();
		return table_name == other.table_name && index_name == other.index_name && k == other.k &&
		       vector_col_idx == other.vector_col_idx;
	}
};

struct AnnSearchTableGlobalState : public GlobalTableFunctionState {
	idx_t MaxThreads() const override {
		return 1;
	}
};

struct AnnSearchTableLocalState : public LocalTableFunctionState {
	struct ResultTriple {
		idx_t input_row;
		row_t base_row_id;
		float distance;
	};
	vector<ResultTriple> results;
	idx_t emit_offset = 0;
	bool processed = false;
};

static unique_ptr<FunctionData> AnnSearchTableBind(ClientContext &context, TableFunctionBindInput &input,
                                                   vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<AnnSearchTableBindData>();

	// Scalar args: inputs[0] is empty (TABLE placeholder), inputs[1..3] are scalars
	bind_data->table_name = input.inputs[1].GetValue<string>();
	bind_data->index_name = input.inputs[2].GetValue<string>();
	bind_data->k = input.inputs[3].GetValue<int32_t>();

	for (auto &kv : input.named_parameters) {
		if (kv.first == "search_complexity") {
			bind_data->search_complexity = kv.second.GetValue<int32_t>();
		}
	}

	// Save input table schema
	bind_data->input_types = input.input_table_types;
	bind_data->input_names = input.input_table_names;

	// Find the vector column: first LIST or ARRAY with numeric child (FLOAT, DOUBLE, DECIMAL, etc.)
	bind_data->vector_col_idx = DConstants::INVALID_INDEX;
	for (idx_t i = 0; i < input.input_table_types.size(); i++) {
		auto &type = input.input_table_types[i];
		if (type.id() == LogicalTypeId::LIST) {
			auto child = ListType::GetChildType(type).id();
			if (child == LogicalTypeId::FLOAT || child == LogicalTypeId::DOUBLE || child == LogicalTypeId::DECIMAL ||
			    child == LogicalTypeId::INTEGER || child == LogicalTypeId::BIGINT || child == LogicalTypeId::SMALLINT ||
			    child == LogicalTypeId::TINYINT) {
				bind_data->vector_col_idx = i;
				break;
			}
		}
		if (type.id() == LogicalTypeId::ARRAY) {
			auto child = ArrayType::GetChildType(type).id();
			if (child == LogicalTypeId::FLOAT || child == LogicalTypeId::DOUBLE || child == LogicalTypeId::DECIMAL ||
			    child == LogicalTypeId::INTEGER || child == LogicalTypeId::BIGINT || child == LogicalTypeId::SMALLINT ||
			    child == LogicalTypeId::TINYINT) {
				bind_data->vector_col_idx = i;
				break;
			}
		}
	}
	if (bind_data->vector_col_idx == DConstants::INVALID_INDEX) {
		throw BinderException("ann_search_table: input table must have a numeric LIST or ARRAY column for queries");
	}

	// Output: input columns + base table columns + _distance
	for (idx_t i = 0; i < input.input_table_types.size(); i++) {
		names.push_back(input.input_table_names[i]);
		return_types.push_back(input.input_table_types[i]);
	}

	// Collect input column names for dedup
	unordered_set<string> used_names;
	for (auto &n : names) {
		used_names.insert(n);
	}

	// Look up base table columns — prefix with table name if name conflicts
	auto &catalog = Catalog::GetCatalog(context, "");
	auto &duck_table =
	    catalog.GetEntry<TableCatalogEntry>(context, DEFAULT_SCHEMA, bind_data->table_name).Cast<DuckTableEntry>();
	auto &columns = duck_table.GetColumns();

	for (auto &col : columns.Physical()) {
		bind_data->base_column_names.push_back(col.Name());
		bind_data->base_column_types.push_back(col.Type());
		bind_data->base_storage_ids.emplace_back(columns.LogicalToPhysical(col.Logical()).index);

		auto col_name = col.Name();
		if (used_names.count(col_name)) {
			col_name = bind_data->table_name + "_" + col_name;
		}
		used_names.insert(col_name);
		names.push_back(col_name);
		return_types.push_back(col.Type());
	}

	names.push_back("_distance");
	return_types.push_back(LogicalType::FLOAT);

	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> AnnSearchTableGlobalInit(ClientContext &context,
                                                                     TableFunctionInitInput &input) {
	return make_uniq<AnnSearchTableGlobalState>();
}

static unique_ptr<LocalTableFunctionState> AnnSearchTableLocalInit(ExecutionContext &context,
                                                                   TableFunctionInitInput &input,
                                                                   GlobalTableFunctionState *global_state) {
	return make_uniq<AnnSearchTableLocalState>();
}

/// Extract float vectors from a DataChunk column (LIST or ARRAY of FLOAT).
static vector<vector<float>> ExtractVectors(DataChunk &input, idx_t col_idx) {
	vector<vector<float>> result;
	auto count = input.size();
	result.reserve(count);

	for (idx_t i = 0; i < count; i++) {
		auto val = input.data[col_idx].GetValue(i);
		if (val.IsNull()) {
			result.emplace_back();
			continue;
		}
		auto &children = ListValue::GetChildren(val);
		vector<float> vec;
		vec.reserve(children.size());
		for (auto &v : children) {
			vec.push_back(v.GetValue<float>());
		}
		result.push_back(std::move(vec));
	}
	return result;
}

static OperatorResultType AnnSearchTableInOut(ExecutionContext &context, TableFunctionInput &data, DataChunk &input,
                                              DataChunk &output) {
	auto &bind = data.bind_data->Cast<AnnSearchTableBindData>();
	auto &lstate = data.local_state->Cast<AnnSearchTableLocalState>();
	auto &client = context.client;

	// Process input chunk if not yet done
	if (!lstate.processed) {
		lstate.results.clear();
		lstate.emit_offset = 0;

		auto queries = ExtractVectors(input, bind.vector_col_idx);

		// Find the DiskANN index
		auto &catalog = Catalog::GetCatalog(client, "");
		auto &duck_table =
		    catalog.GetEntry<TableCatalogEntry>(client, DEFAULT_SCHEMA, bind.table_name).Cast<DuckTableEntry>();
		auto &storage = duck_table.GetStorage();
		auto &table_info = *storage.GetDataTableInfo();
		auto &indexes = table_info.GetIndexes();

		indexes.Bind(client, table_info, DiskannIndex::TYPE_NAME);
#ifdef FAISS_AVAILABLE
		indexes.Bind(client, table_info, FaissIndex::TYPE_NAME);
#endif

		auto idx_ptr = indexes.Find(bind.index_name);
		if (!idx_ptr) {
			throw InvalidInputException("ANN index '%s' not found on table '%s'", bind.index_name, bind.table_name);
		}

		auto *diskann = dynamic_cast<DiskannIndex *>(idx_ptr.get());
		if (diskann) {
			auto batch_results = diskann->SearchBatch(queries, bind.k, bind.search_complexity);
			for (idx_t qi = 0; qi < batch_results.size(); qi++) {
				for (auto &pair : batch_results[qi]) {
					lstate.results.push_back({qi, pair.first, pair.second});
				}
			}
		}
#ifdef FAISS_AVAILABLE
		else {
			auto *faiss = dynamic_cast<FaissIndex *>(idx_ptr.get());
			if (faiss) {
				for (idx_t qi = 0; qi < queries.size(); qi++) {
					auto results = faiss->Search(queries[qi].data(), static_cast<int32_t>(queries[qi].size()), bind.k);
					for (auto &pair : results) {
						lstate.results.push_back({qi, pair.first, pair.second});
					}
				}
			}
		}
#endif

		lstate.processed = true;
	}

	// Emit results
	auto remaining = lstate.results.size() - lstate.emit_offset;
	if (remaining == 0) {
		output.SetCardinality(0);
		lstate.processed = false;
		return OperatorResultType::NEED_MORE_INPUT;
	}

	auto batch_size = MinValue<idx_t>(remaining, STANDARD_VECTOR_SIZE);
	auto n_input_cols = bind.input_types.size();
	auto n_base_cols = bind.base_column_types.size();

	// Copy input columns (replicated per result row)
	for (idx_t col = 0; col < n_input_cols; col++) {
		for (idx_t i = 0; i < batch_size; i++) {
			auto input_row = lstate.results[lstate.emit_offset + i].input_row;
			output.data[col].SetValue(i, input.data[col].GetValue(input_row));
		}
	}

	// Fetch base table rows
	auto &catalog = Catalog::GetCatalog(client, "");
	auto &duck_table =
	    catalog.GetEntry<TableCatalogEntry>(client, DEFAULT_SCHEMA, bind.table_name).Cast<DuckTableEntry>();
	auto &storage = duck_table.GetStorage();
	auto &transaction = DuckTransaction::Get(client, storage.db);

	Vector row_ids_vec(LogicalType::ROW_TYPE, batch_size);
	auto row_ids_data = FlatVector::GetData<row_t>(row_ids_vec);
	for (idx_t i = 0; i < batch_size; i++) {
		row_ids_data[i] = lstate.results[lstate.emit_offset + i].base_row_id;
	}

	DataChunk fetch_chunk;
	fetch_chunk.Initialize(client, bind.base_column_types);
	ColumnFetchState fetch_state;
	storage.Fetch(transaction, fetch_chunk, bind.base_storage_ids, row_ids_vec, batch_size, fetch_state);

	for (idx_t col = 0; col < n_base_cols; col++) {
		for (idx_t i = 0; i < batch_size; i++) {
			output.data[n_input_cols + col].SetValue(i, fetch_chunk.data[col].GetValue(i));
		}
	}

	// Distance column (last)
	auto dist_col = n_input_cols + n_base_cols;
	for (idx_t i = 0; i < batch_size; i++) {
		output.data[dist_col].SetValue(i, Value::FLOAT(lstate.results[lstate.emit_offset + i].distance));
	}

	output.SetCardinality(batch_size);
	lstate.emit_offset += batch_size;

	if (lstate.emit_offset >= lstate.results.size()) {
		lstate.processed = false;
		return OperatorResultType::NEED_MORE_INPUT;
	}
	return OperatorResultType::HAVE_MORE_OUTPUT;
}

static OperatorFinalizeResultType AnnSearchTableFinal(ExecutionContext &context, TableFunctionInput &data,
                                                      DataChunk &output) {
	output.SetCardinality(0);
	return OperatorFinalizeResultType::FINISHED;
}

// ========================================
// Distance computation helpers (GPU + CPU)
// ========================================

/// GPU threshold for one-shot batch distance (no iterative overhead).
/// At 768-dim: fires at ~64 candidates (49152/768=64).
static constexpr size_t MIN_GPU_WORK_ONESHOT = 49152;

/// CPU L2/IP distance computation fallback.
static void ComputeDistancesCPU(const float *query, const float *candidates, idx_t n, idx_t dim, int metric,
                                float *out) {
	for (idx_t i = 0; i < n; i++) {
		const float *cand = candidates + i * dim;
		float sum = 0;
		if (metric == 0) { // L2
			for (idx_t j = 0; j < dim; j++) {
				float diff = query[j] - cand[j];
				sum += diff * diff;
			}
		} else { // IP — negate so lower is better
			for (idx_t j = 0; j < dim; j++) {
				sum += query[j] * cand[j];
			}
			sum = -sum;
		}
		out[i] = sum;
	}
}

/// Compute distances: Metal GPU if batch large enough, else CPU.
static void ComputeDistances(const float *query, const float *candidates, idx_t n, idx_t dim, int metric, float *out) {
	if (static_cast<size_t>(n) * dim >= MIN_GPU_WORK_ONESHOT) {
		int rc =
		    diskann_metal_batch_distances(query, candidates, static_cast<int>(n), static_cast<int>(dim), metric, out);
		if (rc == 0) {
			return;
		}
	}
	ComputeDistancesCPU(query, candidates, n, dim, metric, out);
}

// ========================================
// vector_distances(query, TABLE candidates)
// ========================================
// Compute distances between a query vector and candidate vectors from TABLE input.
// Uses Metal GPU when batch is large enough, CPU SIMD otherwise.
// Returns: all input columns + _distance.

struct VectorDistancesBindData : public FunctionData {
	vector<float> query;
	int metric = 0; // 0=L2, 1=IP

	idx_t vector_col_idx;
	vector<LogicalType> input_types;
	vector<string> input_names;

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<VectorDistancesBindData>();
		copy->query = query;
		copy->metric = metric;
		copy->vector_col_idx = vector_col_idx;
		copy->input_types = input_types;
		copy->input_names = input_names;
		return std::move(copy);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<VectorDistancesBindData>();
		return query == other.query && metric == other.metric && vector_col_idx == other.vector_col_idx;
	}
};

struct VectorDistancesGlobalState : public GlobalTableFunctionState {
	idx_t MaxThreads() const override {
		return 1;
	}
};

struct VectorDistancesLocalState : public LocalTableFunctionState {};

static unique_ptr<FunctionData> VectorDistancesBind(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<VectorDistancesBindData>();

	// inputs[0] is empty (TABLE placeholder), inputs[1] is the query vector
	auto &query_list = ListValue::GetChildren(input.inputs[1]);
	for (auto &v : query_list) {
		bind_data->query.push_back(v.GetValue<float>());
	}

	for (auto &kv : input.named_parameters) {
		if (kv.first == "metric") {
			auto val = kv.second.ToString();
			if (val == "IP" || val == "ip" || val == "inner_product") {
				bind_data->metric = 1;
			}
		}
	}

	bind_data->input_types = input.input_table_types;
	bind_data->input_names = input.input_table_names;

	// Auto-detect vector column (first LIST or ARRAY with numeric child)
	bind_data->vector_col_idx = DConstants::INVALID_INDEX;
	for (idx_t i = 0; i < input.input_table_types.size(); i++) {
		auto &type = input.input_table_types[i];
		if (type.id() == LogicalTypeId::LIST) {
			auto child = ListType::GetChildType(type).id();
			if (child == LogicalTypeId::FLOAT || child == LogicalTypeId::DOUBLE || child == LogicalTypeId::DECIMAL ||
			    child == LogicalTypeId::INTEGER || child == LogicalTypeId::BIGINT || child == LogicalTypeId::SMALLINT ||
			    child == LogicalTypeId::TINYINT) {
				bind_data->vector_col_idx = i;
				break;
			}
		}
		if (type.id() == LogicalTypeId::ARRAY) {
			auto child = ArrayType::GetChildType(type).id();
			if (child == LogicalTypeId::FLOAT || child == LogicalTypeId::DOUBLE || child == LogicalTypeId::DECIMAL ||
			    child == LogicalTypeId::INTEGER || child == LogicalTypeId::BIGINT || child == LogicalTypeId::SMALLINT ||
			    child == LogicalTypeId::TINYINT) {
				bind_data->vector_col_idx = i;
				break;
			}
		}
	}
	if (bind_data->vector_col_idx == DConstants::INVALID_INDEX) {
		throw BinderException("vector_distances: input table must have a numeric LIST or ARRAY column");
	}

	// Output: all input columns + _distance
	for (idx_t i = 0; i < input.input_table_types.size(); i++) {
		names.push_back(input.input_table_names[i]);
		return_types.push_back(input.input_table_types[i]);
	}
	names.push_back("_distance");
	return_types.push_back(LogicalType::FLOAT);

	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> VectorDistancesGlobalInit(ClientContext &context,
                                                                      TableFunctionInitInput &input) {
	return make_uniq<VectorDistancesGlobalState>();
}

static unique_ptr<LocalTableFunctionState> VectorDistancesLocalInit(ExecutionContext &context,
                                                                    TableFunctionInitInput &input,
                                                                    GlobalTableFunctionState *global_state) {
	return make_uniq<VectorDistancesLocalState>();
}

static OperatorResultType VectorDistancesInOut(ExecutionContext &context, TableFunctionInput &data, DataChunk &input,
                                               DataChunk &output) {
	auto &bind = data.bind_data->Cast<VectorDistancesBindData>();
	auto count = input.size();

	if (count == 0) {
		output.SetCardinality(0);
		return OperatorResultType::NEED_MORE_INPUT;
	}

	auto dim = bind.query.size();
	auto n_input_cols = bind.input_types.size();

	// Extract vectors from input chunk
	auto vectors = ExtractVectors(input, bind.vector_col_idx);

	// Gather into contiguous buffer for GPU/CPU batch computation
	vector<float> flat(count * dim, 0.0f);
	for (idx_t i = 0; i < count; i++) {
		if (vectors[i].size() == dim) {
			memcpy(flat.data() + i * dim, vectors[i].data(), dim * sizeof(float));
		}
	}

	// Compute distances (GPU if large enough, else CPU)
	vector<float> distances(count);
	ComputeDistances(bind.query.data(), flat.data(), count, dim, bind.metric, distances.data());

	// Copy all input columns to output
	for (idx_t col = 0; col < n_input_cols; col++) {
		for (idx_t i = 0; i < count; i++) {
			output.data[col].SetValue(i, input.data[col].GetValue(i));
		}
	}

	// Append distance column
	for (idx_t i = 0; i < count; i++) {
		output.data[n_input_cols].SetValue(i, Value::FLOAT(distances[i]));
	}

	output.SetCardinality(count);
	return OperatorResultType::NEED_MORE_INPUT;
}

static OperatorFinalizeResultType VectorDistancesFinal(ExecutionContext &context, TableFunctionInput &data,
                                                       DataChunk &output) {
	output.SetCardinality(0);
	return OperatorFinalizeResultType::FINISHED;
}

// ========================================
// hybrid_search(table, index, vector_col, id_col, query_vec, query_text)
// ========================================
// Combines BM25 full-text search + DiskANN vector search + RRF fusion.
// Returns: all table columns + _rrf_score + _bm25_rank + _vector_rank.
//
// Requires DuckDB's FTS extension to be loaded and an FTS index created on the table.
// FTS schema is derived as fts_main_<table_name>.

struct HybridSearchBindData : public TableFunctionData {
	string table_name;
	string diskann_index_name;
	string vector_column;
	string id_column;
	vector<float> query_vector;
	string query_text;

	int32_t k = 20;
	float bm25_weight = 0.3f;
	float vector_weight = 0.7f;
	int32_t bm25_candidates = 50;
	int32_t vector_candidates = 50;
	int32_t search_complexity = 0;

	// Resolved at bind time
	vector<string> column_names;
	vector<LogicalType> column_types;
	vector<StorageIndex> storage_ids;
};

struct HybridSearchState : public GlobalTableFunctionState {
	struct HybridResult {
		row_t row_id;
		float rrf_score;
		int32_t bm25_rank;   // 0 = not in BM25 results
		int32_t vector_rank; // 0 = not in vector results
	};
	vector<HybridResult> results;
	idx_t offset = 0;
	bool fetched = false;

	idx_t MaxThreads() const override {
		return 1;
	}
};

/// Escape single quotes in SQL string literals.
static string EscapeSQLString(const string &input) {
	string result;
	result.reserve(input.size());
	for (char c : input) {
		if (c == '\'') {
			result += "''";
		} else {
			result += c;
		}
	}
	return result;
}

/// Quote a SQL identifier (table/column name).
static string QuoteIdentifier(const string &name) {
	return "\"" + name + "\"";
}

static unique_ptr<FunctionData> HybridSearchBind(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<HybridSearchBindData>();
	bind_data->table_name = input.inputs[0].GetValue<string>();
	bind_data->diskann_index_name = input.inputs[1].GetValue<string>();
	bind_data->vector_column = input.inputs[2].GetValue<string>();
	bind_data->id_column = input.inputs[3].GetValue<string>();

	auto &query_list = ListValue::GetChildren(input.inputs[4]);
	for (auto &v : query_list) {
		bind_data->query_vector.push_back(v.GetValue<float>());
	}
	bind_data->query_text = input.inputs[5].GetValue<string>();

	for (auto &kv : input.named_parameters) {
		if (kv.first == "k") {
			bind_data->k = kv.second.GetValue<int32_t>();
		} else if (kv.first == "bm25_weight") {
			bind_data->bm25_weight = kv.second.GetValue<float>();
		} else if (kv.first == "vector_weight") {
			bind_data->vector_weight = kv.second.GetValue<float>();
		} else if (kv.first == "bm25_candidates") {
			bind_data->bm25_candidates = kv.second.GetValue<int32_t>();
		} else if (kv.first == "vector_candidates") {
			bind_data->vector_candidates = kv.second.GetValue<int32_t>();
		} else if (kv.first == "search_complexity") {
			bind_data->search_complexity = kv.second.GetValue<int32_t>();
		}
	}

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

	// Add score/rank columns
	names.push_back("_rrf_score");
	return_types.push_back(LogicalType::FLOAT);
	names.push_back("_bm25_rank");
	return_types.push_back(LogicalType::INTEGER);
	names.push_back("_vector_rank");
	return_types.push_back(LogicalType::INTEGER);

	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> HybridSearchInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<HybridSearchState>();
}

static void HybridSearchScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind = data.bind_data->Cast<HybridSearchBindData>();
	auto &state = data.global_state->Cast<HybridSearchState>();

	if (!state.fetched) {
		state.fetched = true;

		// RRF constant
		constexpr float RRF_K = 60.0f;

		// Maps row_id -> (bm25_rank, vector_rank)
		unordered_map<row_t, pair<int32_t, int32_t>> rank_map;

		// ---- Step 1: BM25 search via DuckDB FTS ----
		// Use a separate Connection to avoid deadlocking the ClientContext mutex.
		{
			auto fts_schema = "fts_main_" + bind.table_name;
			auto escaped_query = EscapeSQLString(bind.query_text);
			auto sql = "SELECT rowid, " + QuoteIdentifier(fts_schema) + ".match_bm25(" +
			           QuoteIdentifier(bind.id_column) + ", '" + escaped_query + "') AS __score FROM " +
			           QuoteIdentifier(bind.table_name) + " WHERE __score IS NOT NULL ORDER BY __score DESC LIMIT " +
			           to_string(bind.bm25_candidates);

			Connection fts_conn(*context.db);
			auto result = fts_conn.Query(sql);
			if (!result->HasError()) {
				int32_t rank = 1;
				while (true) {
					auto chunk = result->Fetch();
					if (!chunk || chunk->size() == 0) {
						break;
					}
					for (idx_t i = 0; i < chunk->size(); i++) {
						auto rowid = chunk->data[0].GetValue(i).GetValue<row_t>();
						rank_map[rowid].first = rank++;
					}
				}
			}
			// If FTS fails (extension not loaded or index not created), proceed with vector-only
		}

		// ---- Step 2: DiskANN vector search ----
		{
			auto &catalog = Catalog::GetCatalog(context, "");
			auto &duck_table =
			    catalog.GetEntry<TableCatalogEntry>(context, DEFAULT_SCHEMA, bind.table_name).Cast<DuckTableEntry>();
			auto &storage = duck_table.GetStorage();
			auto &table_info = *storage.GetDataTableInfo();
			auto &indexes = table_info.GetIndexes();

			indexes.Bind(context, table_info, DiskannIndex::TYPE_NAME);

			auto idx_ptr = indexes.Find(bind.diskann_index_name);
			if (!idx_ptr) {
				throw InvalidInputException("ANN index '%s' not found on table '%s'", bind.diskann_index_name,
				                            bind.table_name);
			}

			auto *diskann = dynamic_cast<DiskannIndex *>(idx_ptr.get());
			if (!diskann) {
				throw InvalidInputException("Index '%s' is not a DiskANN index", bind.diskann_index_name);
			}

			auto vector_results =
			    diskann->Search(bind.query_vector.data(), static_cast<int32_t>(bind.query_vector.size()),
			                    bind.vector_candidates, bind.search_complexity);

			int32_t rank = 1;
			for (auto &pair : vector_results) {
				rank_map[pair.first].second = rank++;
			}
		}

		// ---- Step 3: RRF fusion ----
		for (auto &entry : rank_map) {
			auto row_id = entry.first;
			auto bm25_rank = entry.second.first;
			auto vector_rank = entry.second.second;

			float score = 0;
			if (bm25_rank > 0) {
				score += bind.bm25_weight * (1.0f / (RRF_K + static_cast<float>(bm25_rank)));
			}
			if (vector_rank > 0) {
				score += bind.vector_weight * (1.0f / (RRF_K + static_cast<float>(vector_rank)));
			}

			state.results.push_back({row_id, score, bm25_rank, vector_rank});
		}

		// Sort by RRF score descending, take top k
		std::sort(state.results.begin(), state.results.end(),
		          [](const HybridSearchState::HybridResult &a, const HybridSearchState::HybridResult &b) {
			          return a.rrf_score > b.rrf_score;
		          });
		if (static_cast<int32_t>(state.results.size()) > bind.k) {
			state.results.resize(bind.k);
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

	// Copy table columns
	for (idx_t col = 0; col < num_table_cols; col++) {
		for (idx_t i = 0; i < batch_size; i++) {
			output.data[col].SetValue(i, fetch_chunk.data[col].GetValue(i));
		}
	}

	// Score/rank columns
	for (idx_t i = 0; i < batch_size; i++) {
		auto &r = state.results[state.offset + i];
		output.data[num_table_cols].SetValue(i, Value::FLOAT(r.rrf_score));
		output.data[num_table_cols + 1].SetValue(i, Value::INTEGER(r.bm25_rank));
		output.data[num_table_cols + 2].SetValue(i, Value::INTEGER(r.vector_rank));
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

	// Table-input streaming batch search: accepts subqueries, CTEs, generate_series, etc.
	// Usage: SELECT * FROM ann_search_table((SELECT vec FROM queries), 'base_table', 'idx', 10)
	TableFunction table_func("ann_search_table",
	                         {LogicalType::TABLE, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER},
	                         nullptr, AnnSearchTableBind, AnnSearchTableGlobalInit, AnnSearchTableLocalInit);
	table_func.in_out_function = AnnSearchTableInOut;
	table_func.in_out_function_final = AnnSearchTableFinal;
	table_func.named_parameters["search_complexity"] = LogicalType::INTEGER;
	loader.RegisterFunction(table_func);

	// vector_distances: compute distances between query and TABLE of candidate vectors.
	// Uses Metal GPU when batch is large enough, CPU SIMD otherwise.
	// Usage: SELECT * FROM vector_distances((SELECT id, embedding FROM chunks WHERE ...), [0.1, ...]::FLOAT[768])
	TableFunction vd_func("vector_distances", {LogicalType::TABLE, LogicalType::LIST(LogicalType::FLOAT)}, nullptr,
	                      VectorDistancesBind, VectorDistancesGlobalInit, VectorDistancesLocalInit);
	vd_func.in_out_function = VectorDistancesInOut;
	vd_func.in_out_function_final = VectorDistancesFinal;
	vd_func.named_parameters["metric"] = LogicalType::VARCHAR;
	loader.RegisterFunction(vd_func);

	// hybrid_search: BM25 + DiskANN vector search + RRF fusion in one call.
	// Requires DuckDB FTS extension loaded + FTS index created.
	// Usage: SELECT * FROM hybrid_search('chunks', 'chunks_idx', 'embedding', 'id',
	//                                    [0.1, ...]::FLOAT[768], 'search query', k := 20)
	TableFunction hs_func("hybrid_search",
	                      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                       LogicalType::LIST(LogicalType::FLOAT), LogicalType::VARCHAR},
	                      HybridSearchScan, HybridSearchBind, HybridSearchInit);
	hs_func.named_parameters["k"] = LogicalType::INTEGER;
	hs_func.named_parameters["bm25_weight"] = LogicalType::FLOAT;
	hs_func.named_parameters["vector_weight"] = LogicalType::FLOAT;
	hs_func.named_parameters["bm25_candidates"] = LogicalType::INTEGER;
	hs_func.named_parameters["vector_candidates"] = LogicalType::INTEGER;
	hs_func.named_parameters["search_complexity"] = LogicalType::INTEGER;
	loader.RegisterFunction(hs_func);
}

} // namespace duckdb
