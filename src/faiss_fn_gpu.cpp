#ifdef FAISS_AVAILABLE

#include "ann_extension.hpp"
#include "gpu_backend.hpp"
#include "duckdb/function/table_function.hpp"

namespace duckdb {

// ========================================
// faiss_gpu_info()
// ========================================

struct FaissGpuInfoState : public GlobalTableFunctionState {
	bool done = false;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> FaissGpuInfoBind(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<string> &names) {
	names.push_back("available");
	return_types.push_back(LogicalType::BOOLEAN);
	names.push_back("device");
	return_types.push_back(LogicalType::VARCHAR);
	return make_uniq<TableFunctionData>();
}

static unique_ptr<GlobalTableFunctionState> FaissGpuInfoInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<FaissGpuInfoState>();
}

static void FaissGpuInfoScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<FaissGpuInfoState>();

	if (state.done) {
		output.SetCardinality(0);
		return;
	}
	state.done = true;

	auto &gpu = GetGpuBackend();
	output.SetCardinality(1);
	output.data[0].SetValue(0, Value::BOOLEAN(gpu.IsAvailable()));
	output.data[1].SetValue(0, Value(gpu.DeviceInfo()));
}

// ========================================
// Registration
// ========================================

void RegisterFaissGpuFunctions(ExtensionLoader &loader) {
	TableFunctionSet set("faiss_gpu_info");
	set.AddFunction(TableFunction({}, FaissGpuInfoScan, FaissGpuInfoBind, FaissGpuInfoInit));
	loader.RegisterFunction(set);
}

} // namespace duckdb

#endif // FAISS_AVAILABLE
