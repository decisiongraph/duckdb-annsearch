#include "diskann_index.hpp"
#include "linked_block_storage.hpp"

#include "duckdb/catalog/catalog_entry/duck_index_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/exception/transaction_exception.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_create_index.hpp"
#include "duckdb/storage/partial_block_manager.hpp"
#include "duckdb/storage/table_io_manager.hpp"

namespace duckdb {

// ========================================
// DiskannIndex: Constructor / Destructor
// ========================================

DiskannIndex::DiskannIndex(const string &name, IndexConstraintType constraint_type, const vector<column_t> &column_ids,
                           TableIOManager &table_io_manager, const vector<unique_ptr<Expression>> &unbound_expressions,
                           AttachedDatabase &db, const case_insensitive_map_t<Value> &options,
                           const IndexStorageInfo &info)
    : BoundIndex(name, TYPE_NAME, constraint_type, column_ids, table_io_manager, unbound_expressions, db) {

	if (constraint_type != IndexConstraintType::NONE) {
		throw NotImplementedException("DISKANN indexes do not support unique/primary key constraints");
	}

	// Parse options
	auto params = DiskannParams::Parse(options);
	metric_ = params.metric;
	max_degree_ = params.max_degree;
	build_complexity_ = params.build_complexity;
	alpha_ = params.alpha;
	quantize_sq8_ = params.quantize_sq8;

	// Detect dimension from the expression type
	if (!unbound_expressions.empty()) {
		auto &type = unbound_expressions[0]->return_type;
		if (type.id() == LogicalTypeId::ARRAY) {
			dimension_ = static_cast<int32_t>(ArrayType::GetSize(type));
		}
	}

	// Initialize block allocator for persistence
	auto &block_manager = table_io_manager.GetIndexBlockManager();
	block_allocator_ = make_uniq<FixedSizeAllocator>(LinkedBlock::BLOCK_SIZE, block_manager);

	// If loading from storage, deserialize
	if (info.IsValid()) {
		LoadFromStorage(info);
	}
}

DiskannIndex::~DiskannIndex() {
	if (rust_handle_) {
		DiskannFreeDetached(rust_handle_);
		rust_handle_ = nullptr;
	}
}

// ========================================
// Static factories
// ========================================

unique_ptr<BoundIndex> DiskannIndex::Create(CreateIndexInput &input) {
	return make_uniq<DiskannIndex>(input.name, input.constraint_type, input.column_ids, input.table_io_manager,
	                               input.unbound_expressions, input.db, input.options, input.storage_info);
}

PhysicalOperator &DiskannIndex::CreatePlan(PlanIndexInput &input) {
	auto &op = input.op;
	auto &planner = input.planner;

	// Validate: single FLOAT[N] column
	if (op.unbound_expressions.size() != 1) {
		throw InvalidInputException("DISKANN index requires exactly one column");
	}
	auto &type = op.unbound_expressions[0]->return_type;
	if (type.id() != LogicalTypeId::ARRAY || ArrayType::GetChildType(type).id() != LogicalTypeId::FLOAT) {
		throw InvalidInputException("DISKANN index column must be FLOAT[N] (fixed-size array)");
	}

	// PROJECTION on indexed column + row_id
	vector<LogicalType> new_column_types;
	vector<unique_ptr<Expression>> select_list;
	for (idx_t i = 0; i < op.expressions.size(); i++) {
		new_column_types.push_back(op.expressions[i]->return_type);
		select_list.push_back(std::move(op.expressions[i]));
	}
	new_column_types.emplace_back(LogicalType::ROW_TYPE);
	select_list.push_back(make_uniq<BoundReferenceExpression>(LogicalType::ROW_TYPE, op.info->scan_types.size() - 1));

	auto &proj = planner.Make<PhysicalProjection>(new_column_types, std::move(select_list), op.estimated_cardinality);
	proj.children.push_back(input.table_scan);

	// CREATE INDEX operator
	auto &create_idx = planner.Make<PhysicalCreateDiskannIndex>(
	    op, op.table, op.info->column_ids, std::move(op.info), std::move(op.unbound_expressions),
	    op.estimated_cardinality, std::move(op.alter_table_info));
	create_idx.children.push_back(proj);
	return create_idx;
}

// ========================================
// PhysicalCreateDiskannIndex
// ========================================

PhysicalCreateDiskannIndex::PhysicalCreateDiskannIndex(PhysicalPlan &physical_plan, LogicalOperator &op,
                                                       TableCatalogEntry &table_p, const vector<column_t> &column_ids,
                                                       unique_ptr<CreateIndexInfo> info_p,
                                                       vector<unique_ptr<Expression>> unbound_expressions_p,
                                                       idx_t estimated_cardinality,
                                                       unique_ptr<AlterTableInfo> alter_table_info_p)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::CREATE_INDEX, op.types, estimated_cardinality),
      table(table_p.Cast<DuckTableEntry>()), info(std::move(info_p)),
      unbound_expressions(std::move(unbound_expressions_p)), alter_table_info(std::move(alter_table_info_p)) {

	for (auto &column_id : column_ids) {
		storage_ids.push_back(table.GetColumns().LogicalToPhysical(LogicalIndex(column_id)).index);
	}
}

// Sink state
class CreateDiskannGlobalSinkState : public GlobalSinkState {
public:
	vector<float> all_vectors;
	vector<row_t> all_rowids;
	int32_t dimension = 0;
	DiskannParams params;
};

class CreateDiskannLocalSinkState : public LocalSinkState {};

unique_ptr<GlobalSinkState> PhysicalCreateDiskannIndex::GetGlobalSinkState(ClientContext &context) const {
	auto state = make_uniq<CreateDiskannGlobalSinkState>();

	// Detect dimension
	auto &type = unbound_expressions[0]->return_type;
	state->dimension = static_cast<int32_t>(ArrayType::GetSize(type));

	state->params = DiskannParams::Parse(info->options);

	// Pre-reserve based on estimated cardinality to avoid realloc+copy cycles
	if (estimated_cardinality > 0) {
		state->all_vectors.reserve(estimated_cardinality * state->dimension);
		state->all_rowids.reserve(estimated_cardinality);
	}

	return std::move(state);
}

unique_ptr<LocalSinkState> PhysicalCreateDiskannIndex::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<CreateDiskannLocalSinkState>();
}

SinkResultType PhysicalCreateDiskannIndex::Sink(ExecutionContext &context, DataChunk &chunk,
                                                OperatorSinkInput &input) const {
	auto &state = input.global_state.Cast<CreateDiskannGlobalSinkState>();

	// chunk layout: [indexed_columns...][row_id]
	auto col_count = chunk.ColumnCount();
	D_ASSERT(col_count >= 2); // at least one data column + row_id

	auto &vec_col = chunk.data[0];
	auto &rowid_col = chunk.data[col_count - 1]; // row_id is always last

	auto count = chunk.size();
	if (count == 0) {
		return SinkResultType::NEED_MORE_INPUT;
	}

	// Get array data
	auto &array_child = ArrayVector::GetEntry(vec_col);
	auto array_size = ArrayType::GetSize(vec_col.GetType());
	auto child_data = FlatVector::GetData<float>(array_child);

	// Use UnifiedVectorFormat for row IDs (may be SEQUENCE_VECTOR, not flat)
	UnifiedVectorFormat rowid_format;
	rowid_col.ToUnifiedFormat(count, rowid_format);
	auto rowid_data = reinterpret_cast<row_t *>(rowid_format.data);

	for (idx_t i = 0; i < count; i++) {
		auto row_idx = rowid_format.sel->get_index(i);
		auto row_id = rowid_data[row_idx];
		const float *vec_ptr = child_data + i * array_size;

		state.all_vectors.insert(state.all_vectors.end(), vec_ptr, vec_ptr + array_size);
		state.all_rowids.push_back(row_id);
	}

	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalCreateDiskannIndex::Combine(ExecutionContext &context,
                                                          OperatorSinkCombineInput &input) const {
	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalCreateDiskannIndex::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                      OperatorSinkFinalizeInput &input) const {
	auto &state = input.global_state.Cast<CreateDiskannGlobalSinkState>();

	auto &storage = table.GetStorage();
	if (!storage.IsMainTable()) {
		throw TransactionException(
		    "Transaction conflict: cannot add an index to a table that has been altered or dropped");
	}

	auto options = state.params.ToOptions();

	auto index = make_uniq<DiskannIndex>(info->index_name, info->constraint_type, storage_ids,
	                                     TableIOManager::Get(storage), unbound_expressions, storage.db, options);

	// Create Rust index and add all vectors
	auto rust_handle = DiskannCreateDetached(state.dimension, state.params.metric, state.params.max_degree,
	                                         state.params.build_complexity, state.params.alpha);

	idx_t n_vectors = state.all_rowids.size();
	vector<row_t> label_to_rowid(n_vectors);
	unordered_map<row_t, uint32_t> rowid_to_label;
	rowid_to_label.reserve(n_vectors);

	for (idx_t i = 0; i < n_vectors; i++) {
		const float *vec_ptr = state.all_vectors.data() + i * state.dimension;
		auto label = DiskannDetachedAdd(rust_handle, vec_ptr, state.dimension);
		auto label_u32 = static_cast<uint32_t>(label);
		label_to_rowid[i] = state.all_rowids[i];
		rowid_to_label[state.all_rowids[i]] = label_u32;
	}

	// Transfer built state
	index->rust_handle_ = rust_handle;
	index->dimension_ = state.dimension;
	index->metric_ = state.params.metric;
	index->max_degree_ = state.params.max_degree;
	index->build_complexity_ = state.params.build_complexity;
	index->alpha_ = state.params.alpha;
	index->label_to_rowid_ = std::move(label_to_rowid);
	index->rowid_to_label_ = std::move(rowid_to_label);
	index->is_dirty_ = true;

	// Apply SQ8 quantization if requested
	if (state.params.quantize_sq8 && rust_handle) {
		DiskannDetachedQuantizeSQ8(index->rust_handle_);
		index->quantize_sq8_ = true;
	}

	// Call through BoundIndex reference to avoid name hiding from our overrides
	BoundIndex &bi = *index;
	bi.Vacuum();
	D_ASSERT(!bi.VerifyAndToString(true).empty());
	bi.VerifyAllocations();

	auto &schema = table.schema;
	info->column_ids = storage_ids;

	if (!alter_table_info) {
		auto entry = schema.GetEntry(schema.GetCatalogTransaction(context), CatalogType::INDEX_ENTRY, info->index_name);
		if (entry) {
			if (info->on_conflict != OnCreateConflict::IGNORE_ON_CONFLICT) {
				throw CatalogException("Index with name \"%s\" already exists!", info->index_name);
			}
			return SinkFinalizeType::READY;
		}

		auto index_entry = schema.CreateIndex(schema.GetCatalogTransaction(context), *info, table).get();
		D_ASSERT(index_entry);
		auto &idx_entry = index_entry->Cast<DuckIndexEntry>();
		idx_entry.initial_index_size = bi.GetInMemorySize();
	} else {
		auto &indexes = storage.GetDataTableInfo()->GetIndexes();
		indexes.Scan([&](Index &idx) {
			if (idx.GetIndexName() == info->index_name) {
				throw CatalogException("an index with that name already exists for this table: %s", info->index_name);
			}
			return false;
		});

		auto &catalog = Catalog::GetCatalog(context, info->catalog);
		catalog.Alter(context, *alter_table_info);
	}

	storage.AddIndex(std::move(index));
	return SinkFinalizeType::READY;
}

SourceResultType PhysicalCreateDiskannIndex::GetData(ExecutionContext &context, DataChunk &chunk,
                                                     OperatorSourceInput &input) const {
	return SourceResultType::FINISHED;
}

// ========================================
// DiskannIndex: Append / Insert / Delete
// ========================================

ErrorData DiskannIndex::Append(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) {
	auto count = entries.size();
	if (count == 0) {
		return ErrorData {};
	}

	// Execute expressions to extract indexed column from full table data
	DataChunk expr_chunk;
	expr_chunk.Initialize(Allocator::DefaultAllocator(), logical_types);
	ExecuteExpressions(entries, expr_chunk);

	if (!rust_handle_) {
		rust_handle_ = DiskannCreateDetached(dimension_, metric_, max_degree_, build_complexity_, alpha_);
	}

	auto &vec_col = expr_chunk.data[0];
	auto &array_child = ArrayVector::GetEntry(vec_col);
	auto array_size = ArrayType::GetSize(vec_col.GetType());
	auto child_data = FlatVector::GetData<float>(array_child);

	UnifiedVectorFormat rowid_format;
	row_identifiers.ToUnifiedFormat(count, rowid_format);
	auto rowid_data = reinterpret_cast<row_t *>(rowid_format.data);

	for (idx_t i = 0; i < count; i++) {
		auto row_idx = rowid_format.sel->get_index(i);
		auto row_id = rowid_data[row_idx];
		const float *vec_ptr = child_data + i * array_size;

		auto label = DiskannDetachedAdd(rust_handle_, vec_ptr, dimension_);
		auto label_u32 = static_cast<uint32_t>(label);

		if (label_u32 >= label_to_rowid_.size()) {
			label_to_rowid_.resize(label_u32 + 1, -1);
		}
		label_to_rowid_[label_u32] = row_id;
		rowid_to_label_[row_id] = label_u32;
	}

	is_dirty_ = true;
	return ErrorData {};
}

ErrorData DiskannIndex::Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) {
	return Append(lock, data, row_ids);
}

void DiskannIndex::Delete(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) {
	auto count = entries.size();
	if (count == 0) {
		return;
	}

	UnifiedVectorFormat rowid_format;
	row_identifiers.ToUnifiedFormat(count, rowid_format);
	auto rowid_data = reinterpret_cast<row_t *>(rowid_format.data);

	for (idx_t i = 0; i < count; i++) {
		auto row_idx = rowid_format.sel->get_index(i);
		auto row_id = rowid_data[row_idx];

		auto it = rowid_to_label_.find(row_id);
		if (it != rowid_to_label_.end()) {
			deleted_labels_.insert(it->second);
			rowid_to_label_.erase(it);
		}
	}

	is_dirty_ = true;
}

void DiskannIndex::CommitDrop(IndexLock &lock) {
	if (rust_handle_) {
		DiskannFreeDetached(rust_handle_);
		rust_handle_ = nullptr;
	}
	label_to_rowid_.clear();
	rowid_to_label_.clear();
	deleted_labels_.clear();

	if (root_block_ptr_.Get() != 0) {
		block_allocator_->Reset();
		root_block_ptr_ = IndexPointer();
	}
}

// ========================================
// Serialization
// ========================================

static constexpr uint32_t DISKANN_STORAGE_VERSION = 1;

void DiskannIndex::PersistToDisk() {
	if (!is_dirty_ || !rust_handle_) {
		return;
	}

	if (root_block_ptr_.Get() == 0) {
		root_block_ptr_ = block_allocator_->New();
	}

	auto serialized = DiskannDetachedSerialize(rust_handle_);

	uint64_t diskann_len = serialized.len;
	uint64_t num_mappings = label_to_rowid_.size();
	uint64_t num_tombstones = deleted_labels_.size();

	LinkedBlockWriter writer(*block_allocator_, root_block_ptr_);
	writer.Reset();

	// Write version header
	writer.Write(reinterpret_cast<const uint8_t *>(&DISKANN_STORAGE_VERSION), sizeof(uint32_t));

	writer.Write(reinterpret_cast<const uint8_t *>(&diskann_len), sizeof(uint64_t));
	writer.Write(serialized.data, serialized.len);

	writer.Write(reinterpret_cast<const uint8_t *>(&num_mappings), sizeof(uint64_t));
	if (num_mappings > 0) {
		writer.Write(reinterpret_cast<const uint8_t *>(label_to_rowid_.data()), num_mappings * sizeof(row_t));
	}

	writer.Write(reinterpret_cast<const uint8_t *>(&num_tombstones), sizeof(uint64_t));
	if (num_tombstones > 0) {
		vector<uint32_t> tombstone_vec(deleted_labels_.begin(), deleted_labels_.end());
		writer.Write(reinterpret_cast<const uint8_t *>(tombstone_vec.data()), num_tombstones * sizeof(uint32_t));
	}

	writer.Write(reinterpret_cast<const uint8_t *>(&dimension_), sizeof(int32_t));
	writer.Write(reinterpret_cast<const uint8_t *>(&max_degree_), sizeof(int32_t));
	writer.Write(reinterpret_cast<const uint8_t *>(&build_complexity_), sizeof(int32_t));
	uint32_t metric_len = static_cast<uint32_t>(metric_.size());
	writer.Write(reinterpret_cast<const uint8_t *>(&metric_len), sizeof(uint32_t));
	writer.Write(reinterpret_cast<const uint8_t *>(metric_.data()), metric_len);
	uint32_t alpha_bits;
	memcpy(&alpha_bits, &alpha_, sizeof(float));
	writer.Write(reinterpret_cast<const uint8_t *>(&alpha_bits), sizeof(uint32_t));

	DiskannFreeSerializedBytes(serialized);
	is_dirty_ = false;
}

void DiskannIndex::LoadFromStorage(const IndexStorageInfo &info) {
	if (!info.IsValid() || info.allocator_infos.empty()) {
		return;
	}

	root_block_ptr_.Set(info.root);
	block_allocator_->Init(info.allocator_infos[0]);

	LinkedBlockReader reader(*block_allocator_, root_block_ptr_);

	// Read and validate version header
	uint32_t version = 0;
	reader.Read(reinterpret_cast<uint8_t *>(&version), sizeof(uint32_t));
	if (version != DISKANN_STORAGE_VERSION) {
		throw IOException("DiskANN index storage version mismatch: found %u, expected %u. "
		                  "Drop and recreate the index.",
		                  version, DISKANN_STORAGE_VERSION);
	}

	uint64_t diskann_len = 0;
	reader.Read(reinterpret_cast<uint8_t *>(&diskann_len), sizeof(uint64_t));

	vector<uint8_t> diskann_data(diskann_len);
	reader.Read(diskann_data.data(), diskann_len);

	uint64_t num_mappings = 0;
	reader.Read(reinterpret_cast<uint8_t *>(&num_mappings), sizeof(uint64_t));
	label_to_rowid_.resize(num_mappings);
	if (num_mappings > 0) {
		reader.Read(reinterpret_cast<uint8_t *>(label_to_rowid_.data()), num_mappings * sizeof(row_t));
	}

	uint64_t num_tombstones = 0;
	reader.Read(reinterpret_cast<uint8_t *>(&num_tombstones), sizeof(uint64_t));
	if (num_tombstones > 0) {
		vector<uint32_t> tombstones(num_tombstones);
		reader.Read(reinterpret_cast<uint8_t *>(tombstones.data()), num_tombstones * sizeof(uint32_t));
		deleted_labels_.insert(tombstones.begin(), tombstones.end());
	}

	reader.Read(reinterpret_cast<uint8_t *>(&dimension_), sizeof(int32_t));
	reader.Read(reinterpret_cast<uint8_t *>(&max_degree_), sizeof(int32_t));
	reader.Read(reinterpret_cast<uint8_t *>(&build_complexity_), sizeof(int32_t));
	uint32_t metric_len = 0;
	reader.Read(reinterpret_cast<uint8_t *>(&metric_len), sizeof(uint32_t));
	vector<char> metric_buf(metric_len);
	reader.Read(reinterpret_cast<uint8_t *>(metric_buf.data()), metric_len);
	metric_.assign(metric_buf.data(), metric_len);
	uint32_t alpha_bits = 0;
	reader.Read(reinterpret_cast<uint8_t *>(&alpha_bits), sizeof(uint32_t));
	memcpy(&alpha_, &alpha_bits, sizeof(float));

	rowid_to_label_.reserve(num_mappings);
	for (size_t i = 0; i < label_to_rowid_.size(); i++) {
		if (deleted_labels_.count(static_cast<uint32_t>(i)) == 0) {
			rowid_to_label_[label_to_rowid_[i]] = static_cast<uint32_t>(i);
		}
	}

	rust_handle_ = DiskannDetachedDeserialize(diskann_data.data(), diskann_data.size(), alpha_);
	is_dirty_ = false;
}

IndexStorageInfo DiskannIndex::SerializeToDisk(QueryContext context, const case_insensitive_map_t<Value> &options) {
	PersistToDisk();

	IndexStorageInfo info;
	info.name = name;
	info.root = root_block_ptr_.Get();

	auto &block_manager = table_io_manager.GetIndexBlockManager();
	PartialBlockManager partial_block_manager(context, block_manager, PartialBlockType::FULL_CHECKPOINT);
	block_allocator_->SerializeBuffers(partial_block_manager);
	partial_block_manager.FlushPartialBlocks();
	info.allocator_infos.push_back(block_allocator_->GetInfo());

	return info;
}

IndexStorageInfo DiskannIndex::SerializeToWAL(const case_insensitive_map_t<Value> &options) {
	PersistToDisk();

	IndexStorageInfo info;
	info.name = name;
	info.root = root_block_ptr_.Get();
	info.buffers.push_back(block_allocator_->InitSerializationToWAL());
	info.allocator_infos.push_back(block_allocator_->GetInfo());

	return info;
}

// ========================================
// Search
// ========================================

vector<pair<row_t, float>> DiskannIndex::Search(const float *query, int32_t dimension, int32_t k,
                                                int32_t search_complexity) {
	if (!rust_handle_ || dimension != dimension_) {
		return {};
	}

	int64_t request_k64 = static_cast<int64_t>(k) + static_cast<int64_t>(deleted_labels_.size());
	int64_t total_count = static_cast<int64_t>(DiskannDetachedCount(rust_handle_));
	request_k64 = MinValue<int64_t>(request_k64, total_count);
	int32_t request_k = static_cast<int32_t>(MinValue<int64_t>(request_k64, static_cast<int64_t>(INT32_MAX)));
	if (request_k <= 0) {
		return {};
	}

	// Thread-local scratch buffers — allocated once per thread, reused across queries
	thread_local vector<int64_t> tl_labels;
	thread_local vector<float> tl_distances;
	tl_labels.resize(request_k);
	tl_distances.resize(request_k);

	auto n = DiskannDetachedSearch(rust_handle_, query, dimension, request_k, search_complexity, tl_labels.data(),
	                               tl_distances.data());

	// Shrink thread-local buffers if a previous large request inflated them
	if (tl_labels.capacity() > 4096 && request_k < 1024) {
		tl_labels.shrink_to_fit();
		tl_distances.shrink_to_fit();
	}

	vector<pair<row_t, float>> results;
	results.reserve(k);

	for (int32_t i = 0; i < n && static_cast<int32_t>(results.size()) < k; i++) {
		auto label = static_cast<uint32_t>(tl_labels[i]);
		if (deleted_labels_.count(label) > 0) {
			continue;
		}
		if (label < label_to_rowid_.size()) {
			results.emplace_back(label_to_rowid_[label], tl_distances[i]);
		}
	}

	return results;
}

vector<vector<pair<row_t, float>>> DiskannIndex::SearchBatch(const vector<vector<float>> &queries, int32_t k,
                                                             int32_t search_complexity) {
	auto nq = static_cast<int32_t>(queries.size());
	vector<vector<pair<row_t, float>>> all_results(nq);

	if (!rust_handle_ || nq == 0) {
		return all_results;
	}

	// Flatten queries into contiguous buffer
	vector<float> flat_queries;
	flat_queries.reserve(static_cast<size_t>(nq) * dimension_);
	for (auto &q : queries) {
		flat_queries.insert(flat_queries.end(), q.begin(), q.end());
	}

	// Allocate output buffers
	auto total = static_cast<size_t>(nq) * k;
	vector<int64_t> flat_labels(total, -1);
	vector<float> flat_distances(total, std::numeric_limits<float>::max());
	vector<int32_t> counts(nq, 0);

	// Single batch FFI call — GPU-accelerated lock-step BFS
	DiskannDetachedSearchBatch(rust_handle_, flat_queries.data(), nq, dimension_, k, search_complexity,
	                           flat_labels.data(), flat_distances.data(), counts.data());

	// Reconstruct per-query results with row_id mapping
	for (int32_t qi = 0; qi < nq; qi++) {
		auto n = counts[qi];
		auto base = static_cast<size_t>(qi) * k;
		all_results[qi].reserve(n);
		for (int32_t i = 0; i < n; i++) {
			auto label = static_cast<uint32_t>(flat_labels[base + i]);
			if (label < label_to_rowid_.size()) {
				all_results[qi].emplace_back(label_to_rowid_[label], flat_distances[base + i]);
			}
		}
	}

	return all_results;
}

// ========================================
// Utility methods
// ========================================

idx_t DiskannIndex::GetInMemorySize(IndexLock &state) {
	idx_t size = sizeof(DiskannIndex);
	size += label_to_rowid_.size() * sizeof(row_t);
	size += rowid_to_label_.size() * (sizeof(row_t) + sizeof(uint32_t));
	if (rust_handle_) {
		auto count = DiskannDetachedCount(rust_handle_);
		size += count * dimension_ * sizeof(float);
		size += count * max_degree_ * sizeof(uint32_t);
	}
	return size;
}

bool DiskannIndex::MergeIndexes(IndexLock &state, BoundIndex &other_index) {
	auto &other = other_index.Cast<DiskannIndex>();
	if (!other.rust_handle_ || !rust_handle_) {
		is_dirty_ = true;
		return true;
	}

	auto other_count = DiskannDetachedCount(other.rust_handle_);
	vector<float> vec_buf(dimension_);

	for (int64_t label = 0; label < other_count; label++) {
		auto l = static_cast<uint32_t>(label);

		// Skip deleted labels in other index
		if (other.deleted_labels_.count(l) > 0) {
			continue;
		}

		// Get the vector from other index
		auto dim = DiskannDetachedGetVector(other.rust_handle_, l, vec_buf.data(), dimension_);
		if (dim <= 0) {
			continue;
		}

		// Get the row_id from other's mapping
		if (l >= other.label_to_rowid_.size()) {
			continue;
		}
		auto row_id = other.label_to_rowid_[l];

		// Add to this index
		auto new_label = DiskannDetachedAdd(rust_handle_, vec_buf.data(), dimension_);
		auto new_l = static_cast<uint32_t>(new_label);

		// Update mappings
		if (new_l >= label_to_rowid_.size()) {
			label_to_rowid_.resize(new_l + 1, -1);
		}
		label_to_rowid_[new_l] = row_id;
		rowid_to_label_[row_id] = new_l;
	}

	is_dirty_ = true;
	return true;
}

void DiskannIndex::Vacuum(IndexLock &state) {
	if (deleted_labels_.empty() || !rust_handle_) {
		return;
	}

	// Convert deleted_labels_ set to a vector for FFI
	vector<uint32_t> deleted_vec(deleted_labels_.begin(), deleted_labels_.end());

	// Compact in Rust: rebuild index without deleted labels
	auto result = DiskannDetachedCompact(rust_handle_, deleted_vec.data(), deleted_vec.size());

	// Swap handles
	DiskannFreeDetached(rust_handle_);
	rust_handle_ = result.new_handle;

	// Rebuild row_id mappings using the label map
	// old label_to_rowid_ has the row_id for each old label
	// We need: new_label -> old row_id
	vector<row_t> old_label_to_rowid = std::move(label_to_rowid_);

	auto new_count = DiskannDetachedCount(rust_handle_);
	label_to_rowid_.clear();
	label_to_rowid_.resize(new_count, -1);
	rowid_to_label_.clear();

	for (size_t i = 0; i < result.map_len; i++) {
		auto old_label = result.label_map[i * 2];
		auto new_label = result.label_map[i * 2 + 1];
		if (old_label < old_label_to_rowid.size()) {
			auto row_id = old_label_to_rowid[old_label];
			if (new_label < static_cast<uint32_t>(label_to_rowid_.size())) {
				label_to_rowid_[new_label] = row_id;
				rowid_to_label_[row_id] = new_label;
			}
		}
	}

	DiskannFreeLabelMap(result.label_map, result.map_len);
	deleted_labels_.clear();
	is_dirty_ = true;
}

string DiskannIndex::VerifyAndToString(IndexLock &state, const bool only_verify) {
	auto count = rust_handle_ ? DiskannDetachedCount(rust_handle_) : 0;
	return StringUtil::Format("DiskANN Index %s (dim=%d, vectors=%lld, metric=%s)", name, dimension_, count, metric_);
}

void DiskannIndex::VerifyAllocations(IndexLock &state) {
}

void DiskannIndex::VerifyBuffers(IndexLock &l) {
}

string DiskannIndex::GetConstraintViolationMessage(VerifyExistenceType verify_type, idx_t failed_index,
                                                   DataChunk &input) {
	return "DISKANN indexes do not support constraints";
}

} // namespace duckdb
