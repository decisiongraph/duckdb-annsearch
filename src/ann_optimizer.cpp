// ANN index scan optimizer: rewrites ORDER BY array_distance(...) LIMIT k
// to use DISKANN/FAISS index scan instead of full table scan.

#include "annsearch_extension.hpp"
#include "diskann_index.hpp"

#ifdef FAISS_AVAILABLE
#include "faiss_index.hpp"
#endif

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"

namespace duckdb {

// ========================================
// AnnIndexScan: replacement table function
// ========================================

struct AnnIndexScanBindData : public TableFunctionData {
	DuckTableEntry *table_entry = nullptr;
	string index_name;
	bool is_diskann = true;

	unsafe_unique_array<float> query_vector;
	idx_t vector_size = 0;
	idx_t limit = 100;
	int32_t search_complexity = 0;

	// Column mapping for DataTable::Fetch()
	vector<StorageIndex> storage_ids;
};

struct AnnIndexScanGlobalState : public GlobalTableFunctionState {
	vector<pair<row_t, float>> results;
	idx_t offset = 0;

	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> AnnIndexScanBind(ClientContext &, TableFunctionBindInput &, vector<LogicalType> &,
                                                 vector<string> &) {
	throw InternalException("AnnIndexScan bind should not be called directly — set by optimizer");
}

static unique_ptr<GlobalTableFunctionState> AnnIndexScanInit(ClientContext &context, TableFunctionInitInput &input) {
	auto state = make_uniq<AnnIndexScanGlobalState>();
	auto &bind_data = input.bind_data->Cast<AnnIndexScanBindData>();

	auto &storage = bind_data.table_entry->GetStorage();
	auto &table_info = *storage.GetDataTableInfo();
	auto &indexes = table_info.GetIndexes();

	if (bind_data.is_diskann) {
		indexes.Bind(context, table_info, DiskannIndex::TYPE_NAME);
		auto idx_ptr = indexes.Find(bind_data.index_name);
		if (idx_ptr) {
			auto &diskann_idx = idx_ptr->Cast<DiskannIndex>();
			state->results =
			    diskann_idx.Search(bind_data.query_vector.get(), static_cast<int32_t>(bind_data.vector_size),
			                       static_cast<int32_t>(bind_data.limit), bind_data.search_complexity);
		}
	}
#ifdef FAISS_AVAILABLE
	else {
		indexes.Bind(context, table_info, FaissIndex::TYPE_NAME);
		auto idx_ptr = indexes.Find(bind_data.index_name);
		if (idx_ptr) {
			auto &faiss_idx = idx_ptr->Cast<FaissIndex>();
			state->results = faiss_idx.Search(bind_data.query_vector.get(), static_cast<int32_t>(bind_data.vector_size),
			                                  static_cast<int32_t>(bind_data.limit));
		}
	}
#endif

	return std::move(state);
}

static void AnnIndexScanScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<AnnIndexScanBindData>();
	auto &state = data.global_state->Cast<AnnIndexScanGlobalState>();

	auto remaining = state.results.size() - state.offset;
	if (remaining == 0) {
		output.SetCardinality(0);
		return;
	}

	auto batch_size = MinValue<idx_t>(remaining, STANDARD_VECTOR_SIZE);

	// Build row_id vector
	Vector row_ids_vec(LogicalType::ROW_TYPE, batch_size);
	auto row_ids_data = FlatVector::GetData<row_t>(row_ids_vec);
	for (idx_t i = 0; i < batch_size; i++) {
		row_ids_data[i] = state.results[state.offset + i].first;
	}

	// Fetch rows from table storage
	auto &storage = bind_data.table_entry->GetStorage();
	auto &transaction = DuckTransaction::Get(context, storage.db);
	ColumnFetchState fetch_state;
	storage.Fetch(transaction, output, bind_data.storage_ids, row_ids_vec, batch_size, fetch_state);

	state.offset += batch_size;
	output.SetCardinality(batch_size);
}

static unique_ptr<NodeStatistics> AnnIndexScanCardinality(ClientContext &, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<AnnIndexScanBindData>();
	return make_uniq<NodeStatistics>(bind_data.limit);
}

static TableFunction GetAnnIndexScanFunction() {
	TableFunction func("_ann_index_scan_internal", {}, AnnIndexScanScan, AnnIndexScanBind, AnnIndexScanInit);
	func.cardinality = AnnIndexScanCardinality;
	func.projection_pushdown = false;
	return func;
}

// ========================================
// Optimizer: detect ORDER BY array_distance(...) and rewrite
// ========================================

// Find a LogicalGet (seq_scan) by walking child nodes
static LogicalGet *FindSeqScan(LogicalOperator &op, idx_t target_table_index) {
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = op.Cast<LogicalGet>();
		if (get.table_index == target_table_index && get.function.name == "seq_scan") {
			return &get;
		}
	}
	for (auto &child : op.children) {
		auto *result = FindSeqScan(*child, target_table_index);
		if (result) {
			return result;
		}
	}
	return nullptr;
}

// Check if a FILTER exists between the projection and the seq_scan
static bool HasFilterBetween(LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_FILTER) {
		return true;
	}
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		return false;
	}
	for (auto &child : op.children) {
		if (HasFilterBetween(*child)) {
			return true;
		}
	}
	return false;
}

// Extract query vector from a constant expression (ARRAY or LIST of FLOAT)
static bool ExtractQueryVector(const BoundConstantExpression &const_expr, vector<float> &out) {
	auto &val = const_expr.value;
	if (val.type().id() == LogicalTypeId::ARRAY) {
		auto &children = ArrayValue::GetChildren(val);
		for (auto &child : children) {
			out.push_back(child.GetValue<float>());
		}
		return !out.empty();
	}
	if (val.type().id() == LogicalTypeId::LIST) {
		auto &children = ListValue::GetChildren(val);
		for (auto &child : children) {
			out.push_back(child.GetValue<float>());
		}
		return !out.empty();
	}
	return false;
}

// Find an ANN index on a table that covers a specific column
struct FoundIndex {
	string name;
	bool is_diskann = true;
};

static bool FindAnnIndex(ClientContext &context, DuckTableEntry &duck_table, column_t physical_col,
                         FoundIndex &result) {
	auto &storage = duck_table.GetStorage();
	auto &table_info = *storage.GetDataTableInfo();
	auto &indexes = table_info.GetIndexes();
	bool found = false;

	indexes.Scan([&](Index &idx) {
		auto type = idx.GetIndexType();
		if (type != "DISKANN" && type != "FAISS") {
			return false;
		}
		auto &col_ids = idx.GetColumnIds();
		for (auto &cid : col_ids) {
			if (cid == physical_col) {
				result.name = idx.GetIndexName();
				result.is_diskann = (type == "DISKANN");
				found = true;
				return true;
			}
		}
		return false;
	});

	return found;
}

// Try to optimize an ORDER BY node by rewriting to ANN index scan
static bool TryOptimizeOrderBy(ClientContext &context, unique_ptr<LogicalOperator> &op, idx_t limit_val) {
	auto &order_by = op->Cast<LogicalOrder>();

	// Must have exactly one ASC order
	if (order_by.orders.size() != 1) {
		return false;
	}
	auto &order = order_by.orders[0];
	if (order.type != OrderType::ASCENDING) {
		return false;
	}

	// ORDER BY expression must be a column ref (pointing into a projection)
	if (order.expression->type != ExpressionType::BOUND_COLUMN_REF) {
		return false;
	}
	auto &col_ref = order.expression->Cast<BoundColumnRefExpression>();

	// Child must be a projection
	if (order_by.children.size() != 1 || order_by.children[0]->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		return false;
	}
	auto &projection = order_by.children[0]->Cast<LogicalProjection>();

	// Resolve the ORDER BY column to the projection expression
	auto proj_idx = col_ref.binding.column_index;
	if (proj_idx >= projection.expressions.size()) {
		return false;
	}
	auto &proj_expr = projection.expressions[proj_idx];

	// Must be a function call (array_distance, etc.)
	if (proj_expr->type != ExpressionType::BOUND_FUNCTION) {
		return false;
	}
	auto &func_expr = proj_expr->Cast<BoundFunctionExpression>();

	// Check function name — support multiple distance functions
	auto &fn_name = func_expr.function.name;
	if (fn_name != "array_distance" && fn_name != "list_distance" && fn_name != "array_inner_product" &&
	    fn_name != "list_inner_product" && fn_name != "array_cosine_similarity" &&
	    fn_name != "list_cosine_similarity") {
		return false;
	}
	if (func_expr.children.size() != 2) {
		return false;
	}

	// Identify column ref and constant among the two arguments
	Expression *col_child = nullptr;
	Expression *const_child = nullptr;
	for (auto &child : func_expr.children) {
		if (child->type == ExpressionType::BOUND_COLUMN_REF) {
			col_child = child.get();
		} else if (child->type == ExpressionType::VALUE_CONSTANT) {
			const_child = child.get();
		}
	}
	if (!col_child || !const_child) {
		return false;
	}

	// Get the table column binding
	auto &indexed_col_ref = col_child->Cast<BoundColumnRefExpression>();
	auto table_index = indexed_col_ref.binding.table_index;
	auto column_index = indexed_col_ref.binding.column_index;

	// Find the seq_scan LogicalGet for this table
	auto *target_get = FindSeqScan(projection, table_index);
	if (!target_get) {
		return false;
	}

	// Bail if column_ids includes ROW_ID
	auto &col_ids = target_get->GetColumnIds();
	for (auto &cid : col_ids) {
		if (cid.IsRowIdColumn()) {
			return false;
		}
	}

	// Map column_index to physical column ID
	if (column_index >= col_ids.size()) {
		return false;
	}
	auto physical_col = col_ids[column_index].GetPrimaryIndex();

	// Get the table entry
	auto table_ptr = target_get->GetTable();
	if (!table_ptr || !table_ptr->IsDuckTable()) {
		return false;
	}
	auto &duck_table = table_ptr->Cast<DuckTableEntry>();

	// Find an ANN index on this column
	FoundIndex found_idx;
	if (!FindAnnIndex(context, duck_table, physical_col, found_idx)) {
		return false;
	}

	// Extract the query vector
	vector<float> query_vector;
	if (!ExtractQueryVector(const_child->Cast<BoundConstantExpression>(), query_vector)) {
		return false;
	}

	// Cost estimation: skip ANN index if limit is too large relative to table size
	auto estimated_cardinality = target_get->EstimateCardinality(context);
	if (limit_val > 0 && estimated_cardinality > 0 && limit_val > estimated_cardinality / 10) {
		return false; // Full scan likely faster when fetching >10% of table
	}

	// Set limit: use the provided limit_val, or default to a reasonable number
	idx_t k = (limit_val > 0) ? limit_val : 100;

	// If there's a FILTER in the plan, overfetch to compensate for filtered-out rows
	if (HasFilterBetween(projection)) {
		k = MaxValue<idx_t>(k * 3, k + 100);
	}

	// Build the replacement bind data
	auto bind_data = make_uniq<AnnIndexScanBindData>();
	bind_data->table_entry = &duck_table;
	bind_data->index_name = found_idx.name;
	bind_data->is_diskann = found_idx.is_diskann;
	bind_data->vector_size = query_vector.size();
	bind_data->limit = k;
	bind_data->search_complexity = 0;

	bind_data->query_vector = make_unsafe_uniq_array<float>(query_vector.size());
	memcpy(bind_data->query_vector.get(), query_vector.data(), query_vector.size() * sizeof(float));

	for (auto &cid : col_ids) {
		bind_data->storage_ids.emplace_back(StorageIndex(cid.GetPrimaryIndex()));
	}

	// Replace the seq_scan function with our ANN index scan
	auto engine = found_idx.is_diskann ? "DISKANN" : "FAISS";
	target_get->function = GetAnnIndexScanFunction();
	target_get->bind_data = std::move(bind_data);
	target_get->has_estimated_cardinality = true;
	target_get->estimated_cardinality = k;

	// EXPLAIN visibility: show optimizer rewrite details
	target_get->extra_info.file_filters =
	    StringUtil::Format("ANN_INDEX_SCAN (index: %s, k: %llu, engine: %s)", found_idx.name, k, engine);

	// Remove the ORDER BY node — results from index scan are already sorted
	op = std::move(order_by.children[0]);

	return true;
}

// Recursively walk the plan tree looking for optimization opportunities
static void OptimizeRecursive(ClientContext &context, unique_ptr<LogicalOperator> &op) {
	// Process children first (bottom-up)
	for (auto &child : op->children) {
		OptimizeRecursive(context, child);
	}

	// Check for LIMIT → ORDER BY pattern
	if (op->type == LogicalOperatorType::LOGICAL_LIMIT) {
		auto &limit = op->Cast<LogicalLimit>();
		if (!limit.children.empty() && limit.children[0]->type == LogicalOperatorType::LOGICAL_ORDER_BY) {
			idx_t limit_val = 0;
			if (limit.limit_val.Type() == LimitNodeType::CONSTANT_VALUE) {
				limit_val = limit.limit_val.GetConstantValue();
			}
			if (TryOptimizeOrderBy(context, limit.children[0], limit_val)) {
				// ORDER BY was removed — LIMIT now directly wraps the projection
				return;
			}
		}
	}

	// Check for standalone ORDER BY (no LIMIT above — use default k)
	if (op->type == LogicalOperatorType::LOGICAL_ORDER_BY) {
		TryOptimizeOrderBy(context, op, 0);
	}
}

static void AnnOptimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	OptimizeRecursive(input.context, plan);
}

// ========================================
// Registration
// ========================================

void RegisterAnnOptimizer(DatabaseInstance &db) {
	OptimizerExtension ext;
	ext.optimize_function = AnnOptimize;
	db.config.optimizer_extensions.push_back(std::move(ext));
}

} // namespace duckdb
