#include "ann_extension.hpp"
#include "diskann_index.hpp"
#include "rust_ffi.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/storage/data_table.hpp"

namespace duckdb {

// ========================================
// diskann_index_scan(table, index, query, k)
// Searches a BoundIndex (CREATE INDEX ... USING DISKANN)
// Returns: (row_id BIGINT, distance FLOAT)
// ========================================

struct DiskannIndexScanBindData : public TableFunctionData {
	string table_name;
	string index_name;
	vector<float> query;
	int32_t k;
	int32_t search_complexity;
};

struct DiskannIndexScanState : public GlobalTableFunctionState {
	vector<row_t> row_ids;
	vector<float> distances;
	idx_t position = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> DiskannIndexScanBind(ClientContext &context, TableFunctionBindInput &input,
                                                     vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<DiskannIndexScanBindData>();
	bind_data->table_name = input.inputs[0].GetValue<string>();
	bind_data->index_name = input.inputs[1].GetValue<string>();

	auto list_val = input.inputs[2];
	auto &children = ListValue::GetChildren(list_val);
	for (auto &child : children) {
		bind_data->query.push_back(child.GetValue<float>());
	}

	bind_data->k = input.inputs[3].GetValue<int32_t>();
	bind_data->search_complexity = 0;

	for (auto &kv : input.named_parameters) {
		if (kv.first == "search_complexity") {
			bind_data->search_complexity = kv.second.GetValue<int32_t>();
		}
	}

	return_types.push_back(LogicalType::BIGINT);
	return_types.push_back(LogicalType::FLOAT);
	names.push_back("row_id");
	names.push_back("distance");
	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> DiskannIndexScanInit(ClientContext &context,
                                                                 TableFunctionInitInput &input) {
	auto state = make_uniq<DiskannIndexScanState>();
	auto &bind = input.bind_data->Cast<DiskannIndexScanBindData>();

	// Look up table and index from catalog
	auto &catalog = Catalog::GetCatalog(context, "");
	auto &table_entry = catalog.GetEntry<TableCatalogEntry>(context, DEFAULT_SCHEMA, bind.table_name);
	auto &duck_table = table_entry.Cast<DuckTableEntry>();
	auto &storage = duck_table.GetStorage();
	auto &table_info = *storage.GetDataTableInfo();
	auto &indexes = table_info.GetIndexes();

	// Bind unbound DISKANN indexes (needed after database reopen)
	indexes.Bind(context, table_info, DiskannIndex::TYPE_NAME);

	auto index_ptr = indexes.Find(bind.index_name);
	if (!index_ptr) {
		throw InvalidInputException("Index '%s' not found on table '%s'", bind.index_name, bind.table_name);
	}

	auto &diskann_idx = index_ptr->Cast<DiskannIndex>();

	auto results =
	    diskann_idx.Search(bind.query.data(), static_cast<int32_t>(bind.query.size()), bind.k, bind.search_complexity);

	for (auto &result : results) {
		state->row_ids.push_back(result.first);
		state->distances.push_back(result.second);
	}

	return std::move(state);
}

static void DiskannIndexScanScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<DiskannIndexScanState>();

	if (state.position >= state.row_ids.size()) {
		output.SetCardinality(0);
		return;
	}

	idx_t chunk_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.row_ids.size() - state.position);

	auto rowid_data = FlatVector::GetData<int64_t>(output.data[0]);
	auto dist_data = FlatVector::GetData<float>(output.data[1]);

	for (idx_t i = 0; i < chunk_size; i++) {
		rowid_data[i] = state.row_ids[state.position + i];
		dist_data[i] = state.distances[state.position + i];
	}

	state.position += chunk_size;
	output.SetCardinality(chunk_size);
}

void RegisterDiskannIndexScanFunction(ExtensionLoader &loader) {
	TableFunction func(
	    "diskann_index_scan",
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::LIST(LogicalType::FLOAT), LogicalType::INTEGER},
	    DiskannIndexScanScan, DiskannIndexScanBind, DiskannIndexScanInit);
	func.named_parameters["search_complexity"] = LogicalType::INTEGER;
	loader.RegisterFunction(func);
}

// ========================================
// diskann_streaming_build(input_path, output_path)
// Two-pass streaming index build from binary vectors file.
// Returns: (num_vectors INTEGER, dimension INTEGER, sample_size INTEGER)
// ========================================

struct StreamingBuildBindData : public TableFunctionData {
	string input_path;
	string output_path;
	string metric = "l2";
	int32_t max_degree = 64;
	int32_t build_complexity = 128;
	float alpha = 1.2f;
	int32_t sample_size = 0;
};

struct StreamingBuildState : public GlobalTableFunctionState {
	bool done = false;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> StreamingBuildBind(ClientContext &context, TableFunctionBindInput &input,
                                                   vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<StreamingBuildBindData>();
	bind_data->input_path = input.inputs[0].GetValue<string>();
	bind_data->output_path = input.inputs[1].GetValue<string>();

	for (auto &kv : input.named_parameters) {
		if (kv.first == "metric") {
			bind_data->metric = kv.second.GetValue<string>();
		} else if (kv.first == "max_degree") {
			bind_data->max_degree = kv.second.GetValue<int32_t>();
		} else if (kv.first == "build_complexity") {
			bind_data->build_complexity = kv.second.GetValue<int32_t>();
		} else if (kv.first == "alpha") {
			bind_data->alpha = kv.second.GetValue<float>();
		} else if (kv.first == "sample_size") {
			bind_data->sample_size = kv.second.GetValue<int32_t>();
		}
	}

	return_types.push_back(LogicalType::INTEGER);
	return_types.push_back(LogicalType::INTEGER);
	return_types.push_back(LogicalType::INTEGER);
	names.push_back("num_vectors");
	names.push_back("dimension");
	names.push_back("sample_size");
	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> StreamingBuildInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<StreamingBuildState>();
}

static void StreamingBuildScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind = data.bind_data->Cast<StreamingBuildBindData>();
	auto &state = data.global_state->Cast<StreamingBuildState>();

	if (state.done) {
		output.SetCardinality(0);
		return;
	}
	state.done = true;

	auto result = DiskannStreamingBuild(bind.input_path, bind.output_path, bind.metric, bind.max_degree,
	                                    bind.build_complexity, bind.alpha, bind.sample_size);

	output.data[0].SetValue(0, Value::INTEGER(result.num_vectors));
	output.data[1].SetValue(0, Value::INTEGER(result.dimension));
	output.data[2].SetValue(0, Value::INTEGER(result.sample_size));
	output.SetCardinality(1);
}

void RegisterDiskannStreamingBuildFunction(ExtensionLoader &loader) {
	TableFunction func("diskann_streaming_build", {LogicalType::VARCHAR, LogicalType::VARCHAR}, StreamingBuildScan,
	                   StreamingBuildBind, StreamingBuildInit);
	func.named_parameters["metric"] = LogicalType::VARCHAR;
	func.named_parameters["max_degree"] = LogicalType::INTEGER;
	func.named_parameters["build_complexity"] = LogicalType::INTEGER;
	func.named_parameters["alpha"] = LogicalType::FLOAT;
	func.named_parameters["sample_size"] = LogicalType::INTEGER;
	loader.RegisterFunction(func);
}

} // namespace duckdb
