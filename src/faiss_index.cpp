#ifdef FAISS_AVAILABLE

#include "faiss_index.hpp"
#include "gpu_backend.hpp"
#include "linked_block_storage.hpp"

#include "duckdb/catalog/catalog_entry/duck_index_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/exception/transaction_exception.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_create_index.hpp"
#include "duckdb/storage/partial_block_manager.hpp"
#include "duckdb/storage/table_io_manager.hpp"

#include "faiss_wrapper.hpp"

namespace duckdb {

// ========================================
// Helper: parse FAISS metric
// ========================================

static faiss::MetricType ParseFaissMetric(const string &metric) {
	if (metric == "IP" || metric == "ip" || metric == "inner_product") {
		return faiss::METRIC_INNER_PRODUCT;
	}
	return faiss::METRIC_L2;
}

// ========================================
// Helper: create a FAISS index from parameters
// ========================================

static std::unique_ptr<faiss::Index> MakeFaissIndex(int32_t dimension, const string &metric, const string &index_type,
                                                    const string &description, int32_t hnsw_m, int32_t ivf_nlist) {
	auto faiss_metric = ParseFaissMetric(metric);

	if (!description.empty()) {
		return std::unique_ptr<faiss::Index>(faiss::index_factory(dimension, description.c_str(), faiss_metric));
	}

	if (index_type == "HNSW" || index_type == "hnsw") {
		return make_faiss_unique<faiss::IndexHNSWFlat>(dimension, hnsw_m, faiss_metric);
	}

	if (index_type == "IVFFlat" || index_type == "ivfflat") {
		auto quantizer = new faiss::IndexFlat(dimension, faiss_metric);
		auto idx = make_faiss_unique<faiss::IndexIVFFlat>(quantizer, dimension, ivf_nlist, faiss_metric);
		idx->own_fields = true;
		return idx;
	}

	// Default: Flat
	return make_faiss_unique<faiss::IndexFlat>(dimension, faiss_metric);
}

// ========================================
// FaissIndex: Constructor / Destructor
// ========================================

FaissIndex::FaissIndex(const string &name, IndexConstraintType constraint_type, const vector<column_t> &column_ids,
                       TableIOManager &table_io_manager, const vector<unique_ptr<Expression>> &unbound_expressions,
                       AttachedDatabase &db, const case_insensitive_map_t<Value> &options, const IndexStorageInfo &info)
    : BoundIndex(name, TYPE_NAME, constraint_type, column_ids, table_io_manager, unbound_expressions, db) {

	if (constraint_type != IndexConstraintType::NONE) {
		throw NotImplementedException("FAISS indexes do not support unique/primary key constraints");
	}

	// Parse options
	for (auto &kv : options) {
		if (kv.first == "metric") {
			metric_ = kv.second.ToString();
		} else if (kv.first == "type") {
			index_type_ = kv.second.ToString();
		} else if (kv.first == "hnsw_m") {
			hnsw_m_ = kv.second.GetValue<int32_t>();
		} else if (kv.first == "ivf_nlist") {
			ivf_nlist_ = kv.second.GetValue<int32_t>();
		} else if (kv.first == "nprobe") {
			nprobe_ = MaxValue<int32_t>(1, kv.second.GetValue<int32_t>());
		} else if (kv.first == "train_sample") {
			train_sample_ = kv.second.GetValue<int64_t>();
		} else if (kv.first == "description") {
			description_ = kv.second.ToString();
		} else if (kv.first == "gpu") {
			gpu_ = BooleanValue::Get(kv.second.DefaultCastAs(LogicalType::BOOLEAN));
		}
	}

	// Detect dimension
	if (!unbound_expressions.empty()) {
		auto &type = unbound_expressions[0]->return_type;
		if (type.id() == LogicalTypeId::ARRAY) {
			dimension_ = static_cast<int32_t>(ArrayType::GetSize(type));
		}
	}

	// Initialize block allocator
	auto &block_manager = table_io_manager.GetIndexBlockManager();
	block_allocator_ = make_uniq<FixedSizeAllocator>(LinkedBlock::BLOCK_SIZE, block_manager);

	// If loading from storage, deserialize
	if (info.IsValid()) {
		LoadFromStorage(info);
	}
}

FaissIndex::~FaissIndex() {
	gpu_index_.reset();
}

void FaissIndex::EnsureGpuIndex() {
	if (!gpu_ || !faiss_index_ || gpu_index_) {
		return;
	}
	auto &backend = GetGpuBackend();
	if (!backend.IsAvailable()) {
		return;
	}
	gpu_index_ = backend.CpuToGpu(faiss_index_.get());
}

void FaissIndex::InvalidateGpuIndex() {
	gpu_index_.reset();
}

// ========================================
// Static factories
// ========================================

unique_ptr<BoundIndex> FaissIndex::Create(CreateIndexInput &input) {
	return make_uniq<FaissIndex>(input.name, input.constraint_type, input.column_ids, input.table_io_manager,
	                             input.unbound_expressions, input.db, input.options, input.storage_info);
}

PhysicalOperator &FaissIndex::CreatePlan(PlanIndexInput &input) {
	auto &op = input.op;
	auto &planner = input.planner;

	if (op.unbound_expressions.size() != 1) {
		throw InvalidInputException("FAISS index requires exactly one column");
	}
	auto &type = op.unbound_expressions[0]->return_type;
	if (type.id() != LogicalTypeId::ARRAY || ArrayType::GetChildType(type).id() != LogicalTypeId::FLOAT) {
		throw InvalidInputException("FAISS index column must be FLOAT[N] (fixed-size array)");
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

	auto &create_idx = planner.Make<PhysicalCreateFaissIndex>(op, op.table, op.info->column_ids, std::move(op.info),
	                                                          std::move(op.unbound_expressions),
	                                                          op.estimated_cardinality, std::move(op.alter_table_info));
	create_idx.children.push_back(proj);
	return create_idx;
}

// ========================================
// PhysicalCreateFaissIndex
// ========================================

PhysicalCreateFaissIndex::PhysicalCreateFaissIndex(PhysicalPlan &physical_plan, LogicalOperator &op,
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

// Sink state: collect all vectors, then build index in Finalize (needed for IVF training)
class CreateFaissGlobalSinkState : public GlobalSinkState {
public:
	vector<float> all_vectors;
	vector<row_t> all_rowids;
	int32_t dimension = 0;
	string metric = "L2";
	string index_type = "Flat";
	int32_t hnsw_m = 32;
	int32_t ivf_nlist = 100;
	int32_t nprobe = 1;
	int64_t train_sample = 0;
	string description;
	bool gpu = false;
};

class CreateFaissLocalSinkState : public LocalSinkState {};

unique_ptr<GlobalSinkState> PhysicalCreateFaissIndex::GetGlobalSinkState(ClientContext &context) const {
	auto state = make_uniq<CreateFaissGlobalSinkState>();

	auto &type = unbound_expressions[0]->return_type;
	state->dimension = static_cast<int32_t>(ArrayType::GetSize(type));

	for (auto &kv : info->options) {
		if (kv.first == "metric") {
			state->metric = kv.second.ToString();
		} else if (kv.first == "type") {
			state->index_type = kv.second.ToString();
		} else if (kv.first == "hnsw_m") {
			state->hnsw_m = kv.second.GetValue<int32_t>();
		} else if (kv.first == "ivf_nlist") {
			state->ivf_nlist = kv.second.GetValue<int32_t>();
		} else if (kv.first == "nprobe") {
			state->nprobe = MaxValue<int32_t>(1, kv.second.GetValue<int32_t>());
		} else if (kv.first == "train_sample") {
			state->train_sample = kv.second.GetValue<int64_t>();
		} else if (kv.first == "description") {
			state->description = kv.second.ToString();
		} else if (kv.first == "gpu") {
			state->gpu = BooleanValue::Get(kv.second.DefaultCastAs(LogicalType::BOOLEAN));
		}
	}

	if (state->index_type.empty()) {
		state->index_type = "Flat";
	}

	return std::move(state);
}

unique_ptr<LocalSinkState> PhysicalCreateFaissIndex::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<CreateFaissLocalSinkState>();
}

SinkResultType PhysicalCreateFaissIndex::Sink(ExecutionContext &context, DataChunk &chunk,
                                              OperatorSinkInput &input) const {
	auto &state = input.global_state.Cast<CreateFaissGlobalSinkState>();

	auto col_count = chunk.ColumnCount();
	D_ASSERT(col_count >= 2);

	auto &vec_col = chunk.data[0];
	auto &rowid_col = chunk.data[col_count - 1];

	auto count = chunk.size();
	if (count == 0) {
		return SinkResultType::NEED_MORE_INPUT;
	}

	auto &array_child = ArrayVector::GetEntry(vec_col);
	auto array_size = ArrayType::GetSize(vec_col.GetType());
	auto child_data = FlatVector::GetData<float>(array_child);

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

SinkCombineResultType PhysicalCreateFaissIndex::Combine(ExecutionContext &context,
                                                        OperatorSinkCombineInput &input) const {
	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalCreateFaissIndex::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                    OperatorSinkFinalizeInput &input) const {
	auto &state = input.global_state.Cast<CreateFaissGlobalSinkState>();

	auto &storage = table.GetStorage();
	if (!storage.IsMainTable()) {
		throw TransactionException(
		    "Transaction conflict: cannot add an index to a table that has been altered or dropped");
	}

	// Create FAISS index
	auto faiss_idx = MakeFaissIndex(state.dimension, state.metric, state.index_type, state.description, state.hnsw_m,
	                                state.ivf_nlist);

	// Train if needed (IVF indexes)
	idx_t n_vectors = state.all_rowids.size();
	if (n_vectors > 0 && !faiss_idx->is_trained) {
		if (state.train_sample > 0 && state.train_sample < static_cast<int64_t>(n_vectors)) {
			// Use a subset of vectors for training
			auto sample_n = static_cast<idx_t>(state.train_sample);
			vector<float> sample(sample_n * state.dimension);
			// Deterministic stride-based sampling
			double stride = static_cast<double>(n_vectors) / sample_n;
			for (idx_t i = 0; i < sample_n; i++) {
				auto src_idx = static_cast<idx_t>(i * stride);
				memcpy(sample.data() + i * state.dimension, state.all_vectors.data() + src_idx * state.dimension,
				       state.dimension * sizeof(float));
			}
			faiss_idx->train(static_cast<faiss::idx_t>(sample_n), sample.data());
		} else {
			faiss_idx->train(static_cast<faiss::idx_t>(n_vectors), state.all_vectors.data());
		}
	}

	// Set nprobe for IVF indexes
	if (state.nprobe > 1) {
		auto *ivf = dynamic_cast<faiss::IndexIVFFlat *>(faiss_idx.get());
		if (ivf) {
			ivf->nprobe = static_cast<size_t>(state.nprobe);
		}
	}

	// Add all vectors
	if (n_vectors > 0) {
		faiss_idx->add(static_cast<faiss::idx_t>(n_vectors), state.all_vectors.data());
	}

	// Build row ID mapping
	vector<row_t> label_to_rowid(n_vectors);
	unordered_map<row_t, int64_t> rowid_to_label;
	for (idx_t i = 0; i < n_vectors; i++) {
		label_to_rowid[i] = state.all_rowids[i];
		rowid_to_label[state.all_rowids[i]] = static_cast<int64_t>(i);
	}

	// Build options map
	case_insensitive_map_t<Value> options;
	options["metric"] = Value(state.metric);
	options["type"] = Value(state.index_type);
	options["hnsw_m"] = Value::INTEGER(state.hnsw_m);
	options["ivf_nlist"] = Value::INTEGER(state.ivf_nlist);
	if (!state.description.empty()) {
		options["description"] = Value(state.description);
	}
	if (state.gpu) {
		options["gpu"] = Value::BOOLEAN(true);
	}

	auto index = make_uniq<FaissIndex>(info->index_name, info->constraint_type, storage_ids,
	                                   TableIOManager::Get(storage), unbound_expressions, storage.db, options);

	// Transfer built state
	index->faiss_index_ = std::move(faiss_idx);
	index->dimension_ = state.dimension;
	index->metric_ = state.metric;
	index->index_type_ = state.index_type;
	index->hnsw_m_ = state.hnsw_m;
	index->ivf_nlist_ = state.ivf_nlist;
	index->nprobe_ = state.nprobe;
	index->train_sample_ = state.train_sample;
	index->description_ = state.description;
	index->gpu_ = state.gpu;
	index->label_to_rowid_ = std::move(label_to_rowid);
	index->rowid_to_label_ = std::move(rowid_to_label);
	index->is_dirty_ = true;

	// Upload to GPU if requested
	index->EnsureGpuIndex();

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

SourceResultType PhysicalCreateFaissIndex::GetData(ExecutionContext &context, DataChunk &chunk,
                                                   OperatorSourceInput &input) const {
	return SourceResultType::FINISHED;
}

// ========================================
// FaissIndex: Append / Insert / Delete
// ========================================

ErrorData FaissIndex::Append(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) {
	auto count = entries.size();
	if (count == 0) {
		return ErrorData {};
	}

	DataChunk expr_chunk;
	expr_chunk.Initialize(Allocator::DefaultAllocator(), logical_types);
	ExecuteExpressions(entries, expr_chunk);

	if (!faiss_index_) {
		faiss_index_ = MakeFaissIndex(dimension_, metric_, index_type_, description_, hnsw_m_, ivf_nlist_);
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

		auto label = faiss_index_->ntotal;
		faiss_index_->add(1, vec_ptr);

		if (label >= static_cast<int64_t>(label_to_rowid_.size())) {
			label_to_rowid_.resize(label + 1, -1);
		}
		label_to_rowid_[label] = row_id;
		rowid_to_label_[row_id] = label;
	}

	InvalidateGpuIndex();
	is_dirty_ = true;
	return ErrorData {};
}

ErrorData FaissIndex::Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) {
	return Append(lock, data, row_ids);
}

void FaissIndex::Delete(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) {
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

void FaissIndex::CommitDrop(IndexLock &lock) {
	gpu_index_.reset();
	faiss_index_.reset();
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

void FaissIndex::PersistToDisk() {
	if (!is_dirty_ || !faiss_index_) {
		return;
	}

	if (root_block_ptr_.Get() == 0) {
		root_block_ptr_ = block_allocator_->New();
	}

	// Serialize FAISS index to bytes
	faiss::VectorIOWriter faiss_writer;
	faiss::write_index(faiss_index_.get(), &faiss_writer);

	uint64_t faiss_len = faiss_writer.data.size();
	uint64_t num_mappings = label_to_rowid_.size();
	uint64_t num_tombstones = deleted_labels_.size();

	LinkedBlockWriter writer(*block_allocator_, root_block_ptr_);
	writer.Reset();

	// Write FAISS serialized data
	writer.Write(reinterpret_cast<const uint8_t *>(&faiss_len), sizeof(uint64_t));
	writer.Write(faiss_writer.data.data(), faiss_len);

	// Write row ID mapping
	writer.Write(reinterpret_cast<const uint8_t *>(&num_mappings), sizeof(uint64_t));
	if (num_mappings > 0) {
		writer.Write(reinterpret_cast<const uint8_t *>(label_to_rowid_.data()), num_mappings * sizeof(row_t));
	}

	// Write tombstones
	writer.Write(reinterpret_cast<const uint8_t *>(&num_tombstones), sizeof(uint64_t));
	if (num_tombstones > 0) {
		vector<int64_t> tombstone_vec(deleted_labels_.begin(), deleted_labels_.end());
		writer.Write(reinterpret_cast<const uint8_t *>(tombstone_vec.data()), num_tombstones * sizeof(int64_t));
	}

	// Write index parameters
	writer.Write(reinterpret_cast<const uint8_t *>(&dimension_), sizeof(int32_t));
	uint32_t metric_len = static_cast<uint32_t>(metric_.size());
	writer.Write(reinterpret_cast<const uint8_t *>(&metric_len), sizeof(uint32_t));
	writer.Write(reinterpret_cast<const uint8_t *>(metric_.data()), metric_len);
	uint32_t type_len = static_cast<uint32_t>(index_type_.size());
	writer.Write(reinterpret_cast<const uint8_t *>(&type_len), sizeof(uint32_t));
	writer.Write(reinterpret_cast<const uint8_t *>(index_type_.data()), type_len);
	writer.Write(reinterpret_cast<const uint8_t *>(&hnsw_m_), sizeof(int32_t));
	writer.Write(reinterpret_cast<const uint8_t *>(&ivf_nlist_), sizeof(int32_t));
	writer.Write(reinterpret_cast<const uint8_t *>(&nprobe_), sizeof(int32_t));
	writer.Write(reinterpret_cast<const uint8_t *>(&train_sample_), sizeof(int64_t));
	uint32_t desc_len = static_cast<uint32_t>(description_.size());
	writer.Write(reinterpret_cast<const uint8_t *>(&desc_len), sizeof(uint32_t));
	if (desc_len > 0) {
		writer.Write(reinterpret_cast<const uint8_t *>(description_.data()), desc_len);
	}
	uint8_t gpu_byte = gpu_ ? 1 : 0;
	writer.Write(&gpu_byte, sizeof(uint8_t));

	is_dirty_ = false;
}

void FaissIndex::LoadFromStorage(const IndexStorageInfo &info) {
	if (!info.IsValid() || info.allocator_infos.empty()) {
		return;
	}

	root_block_ptr_.Set(info.root);
	block_allocator_->Init(info.allocator_infos[0]);

	LinkedBlockReader reader(*block_allocator_, root_block_ptr_);

	// Read FAISS serialized data
	uint64_t faiss_len = 0;
	reader.Read(reinterpret_cast<uint8_t *>(&faiss_len), sizeof(uint64_t));

	vector<uint8_t> faiss_data(faiss_len);
	reader.Read(faiss_data.data(), faiss_len);

	// Read row ID mapping
	uint64_t num_mappings = 0;
	reader.Read(reinterpret_cast<uint8_t *>(&num_mappings), sizeof(uint64_t));
	label_to_rowid_.resize(num_mappings);
	if (num_mappings > 0) {
		reader.Read(reinterpret_cast<uint8_t *>(label_to_rowid_.data()), num_mappings * sizeof(row_t));
	}

	// Read tombstones
	uint64_t num_tombstones = 0;
	reader.Read(reinterpret_cast<uint8_t *>(&num_tombstones), sizeof(uint64_t));
	if (num_tombstones > 0) {
		vector<int64_t> tombstones(num_tombstones);
		reader.Read(reinterpret_cast<uint8_t *>(tombstones.data()), num_tombstones * sizeof(int64_t));
		deleted_labels_.insert(tombstones.begin(), tombstones.end());
	}

	// Read index parameters
	reader.Read(reinterpret_cast<uint8_t *>(&dimension_), sizeof(int32_t));
	uint32_t metric_len = 0;
	reader.Read(reinterpret_cast<uint8_t *>(&metric_len), sizeof(uint32_t));
	vector<char> metric_buf(metric_len);
	reader.Read(reinterpret_cast<uint8_t *>(metric_buf.data()), metric_len);
	metric_.assign(metric_buf.data(), metric_len);

	uint32_t type_len = 0;
	reader.Read(reinterpret_cast<uint8_t *>(&type_len), sizeof(uint32_t));
	vector<char> type_buf(type_len);
	reader.Read(reinterpret_cast<uint8_t *>(type_buf.data()), type_len);
	index_type_.assign(type_buf.data(), type_len);

	reader.Read(reinterpret_cast<uint8_t *>(&hnsw_m_), sizeof(int32_t));
	reader.Read(reinterpret_cast<uint8_t *>(&ivf_nlist_), sizeof(int32_t));
	reader.Read(reinterpret_cast<uint8_t *>(&nprobe_), sizeof(int32_t));
	reader.Read(reinterpret_cast<uint8_t *>(&train_sample_), sizeof(int64_t));

	uint32_t desc_len = 0;
	reader.Read(reinterpret_cast<uint8_t *>(&desc_len), sizeof(uint32_t));
	if (desc_len > 0) {
		vector<char> desc_buf(desc_len);
		reader.Read(reinterpret_cast<uint8_t *>(desc_buf.data()), desc_len);
		description_.assign(desc_buf.data(), desc_len);
	}
	uint8_t gpu_byte = 0;
	reader.Read(&gpu_byte, sizeof(uint8_t));
	gpu_ = (gpu_byte != 0);

	// Rebuild rowid_to_label_ from label_to_rowid_
	for (size_t i = 0; i < label_to_rowid_.size(); i++) {
		if (deleted_labels_.count(static_cast<int64_t>(i)) == 0) {
			rowid_to_label_[label_to_rowid_[i]] = static_cast<int64_t>(i);
		}
	}

	// Deserialize FAISS index from bytes
	faiss::VectorIOReader faiss_reader;
	faiss_reader.data = std::move(faiss_data);
	faiss_index_.reset(faiss::read_index(&faiss_reader));

	// Upload to GPU if the index was created with gpu=true
	EnsureGpuIndex();

	is_dirty_ = false;
}

IndexStorageInfo FaissIndex::SerializeToDisk(QueryContext context, const case_insensitive_map_t<Value> &options) {
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

IndexStorageInfo FaissIndex::SerializeToWAL(const case_insensitive_map_t<Value> &options) {
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

vector<pair<row_t, float>> FaissIndex::Search(const float *query, int32_t dimension, int32_t k) {
	if (!faiss_index_ || dimension != dimension_) {
		return {};
	}

	int32_t request_k = k + static_cast<int32_t>(deleted_labels_.size());
	request_k = MinValue<int32_t>(request_k, static_cast<int32_t>(faiss_index_->ntotal));
	if (request_k <= 0) {
		return {};
	}

	// Set nprobe for IVF indexes before searching
	if (nprobe_ > 1) {
		auto *ivf = dynamic_cast<faiss::IndexIVFFlat *>(faiss_index_.get());
		if (ivf) {
			ivf->nprobe = static_cast<size_t>(nprobe_);
		}
	}

	// Use GPU index for search if available
	EnsureGpuIndex();
	auto *search_index = gpu_index_ ? gpu_index_.get() : faiss_index_.get();

	// Thread-local scratch buffers — allocated once per thread, reused across queries
	thread_local vector<faiss::idx_t> tl_labels;
	thread_local vector<float> tl_distances;
	tl_labels.resize(request_k);
	tl_distances.resize(request_k);

	search_index->search(1, query, request_k, tl_distances.data(), tl_labels.data());

	vector<pair<row_t, float>> results;
	results.reserve(k);

	for (int32_t i = 0; i < request_k && static_cast<int32_t>(results.size()) < k; i++) {
		auto label = tl_labels[i];
		if (label < 0) {
			continue; // FAISS returns -1 for unfilled slots
		}
		if (deleted_labels_.count(label) > 0) {
			continue;
		}
		if (label < static_cast<int64_t>(label_to_rowid_.size())) {
			results.emplace_back(label_to_rowid_[label], tl_distances[i]);
		}
	}

	return results;
}

// ========================================
// Utility methods
// ========================================

idx_t FaissIndex::GetInMemorySize(IndexLock &state) {
	idx_t size = sizeof(FaissIndex);
	size += label_to_rowid_.size() * sizeof(row_t);
	size += rowid_to_label_.size() * (sizeof(row_t) + sizeof(int64_t));
	if (faiss_index_) {
		// Estimate: vectors + overhead
		size += faiss_index_->ntotal * dimension_ * sizeof(float);
	}
	if (gpu_index_) {
		// GPU copy uses roughly the same amount of memory
		size += faiss_index_ ? faiss_index_->ntotal * dimension_ * sizeof(float) : 0;
	}
	return size;
}

bool FaissIndex::MergeIndexes(IndexLock &state, BoundIndex &other_index) {
	is_dirty_ = true;
	return true;
}

void FaissIndex::Vacuum(IndexLock &state) {
}

string FaissIndex::VerifyAndToString(IndexLock &state, const bool only_verify) {
	auto count = faiss_index_ ? faiss_index_->ntotal : 0;
	return StringUtil::Format("FAISS Index %s (type=%s, dim=%d, vectors=%lld, metric=%s)", name, index_type_,
	                          dimension_, count, metric_);
}

void FaissIndex::VerifyAllocations(IndexLock &state) {
}

void FaissIndex::VerifyBuffers(IndexLock &l) {
}

string FaissIndex::GetConstraintViolationMessage(VerifyExistenceType verify_type, idx_t failed_index,
                                                 DataChunk &input) {
	return "FAISS indexes do not support constraints";
}

// ========================================
// faiss_index_scan table function
// ========================================

struct FaissIndexScanBindData : public TableFunctionData {
	string table_name;
	string index_name;
	vector<float> query;
	int32_t k;
};

struct FaissIndexScanState : public GlobalTableFunctionState {
	vector<pair<row_t, float>> results;
	idx_t offset = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> FaissIndexScanBind(ClientContext &context, TableFunctionBindInput &input,
                                                   vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<FaissIndexScanBindData>();
	bind_data->table_name = input.inputs[0].GetValue<string>();
	bind_data->index_name = input.inputs[1].GetValue<string>();

	auto &query_list = ListValue::GetChildren(input.inputs[2]);
	for (auto &v : query_list) {
		bind_data->query.push_back(v.GetValue<float>());
	}
	bind_data->k = input.inputs[3].GetValue<int32_t>();

	names.push_back("row_id");
	return_types.push_back(LogicalType::BIGINT);
	names.push_back("distance");
	return_types.push_back(LogicalType::FLOAT);

	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> FaissIndexScanInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<FaissIndexScanState>();
}

static void FaissIndexScanScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind = data.bind_data->Cast<FaissIndexScanBindData>();
	auto &state = data.global_state->Cast<FaissIndexScanState>();

	if (state.offset == 0 && state.results.empty()) {
		// Find the index
		auto &catalog = Catalog::GetCatalog(context, "");
		auto &duck_table =
		    catalog.GetEntry<TableCatalogEntry>(context, DEFAULT_SCHEMA, bind.table_name).Cast<DuckTableEntry>();
		auto &storage = duck_table.GetStorage();
		auto &table_info = *storage.GetDataTableInfo();
		auto &indexes = table_info.GetIndexes();

		// Bind unbound FAISS indexes (needed after database reopen)
		indexes.Bind(context, table_info, FaissIndex::TYPE_NAME);

		FaissIndex *faiss_idx = nullptr;
		auto idx_ptr = indexes.Find(bind.index_name);
		if (idx_ptr) {
			faiss_idx = dynamic_cast<FaissIndex *>(idx_ptr.get());
		}

		if (!faiss_idx) {
			throw InvalidInputException("FAISS index '%s' not found on table '%s'", bind.index_name, bind.table_name);
		}

		state.results = faiss_idx->Search(bind.query.data(), static_cast<int32_t>(bind.query.size()), bind.k);
	}

	idx_t count = 0;
	while (state.offset < state.results.size() && count < STANDARD_VECTOR_SIZE) {
		auto &result = state.results[state.offset];
		output.data[0].SetValue(count, Value::BIGINT(result.first));
		output.data[1].SetValue(count, Value::FLOAT(result.second));
		count++;
		state.offset++;
	}

	output.SetCardinality(count);
}

void RegisterFaissIndexScanFunction(ExtensionLoader &loader) {
	TableFunction func(
	    "faiss_index_scan",
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::LIST(LogicalType::FLOAT), LogicalType::INTEGER},
	    FaissIndexScanScan, FaissIndexScanBind, FaissIndexScanInit);
	loader.RegisterFunction(func);
}

} // namespace duckdb

#endif // FAISS_AVAILABLE
