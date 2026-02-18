use anyhow::{anyhow, Result};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::cell::RefCell;
use std::io::{BufWriter, Cursor};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};

use diskann::graph::{
    DiskANNIndex, SearchParams,
    config::{Builder, MaxDegree, PruneKind},
    search_output_buffer::IdDistance,
};
use diskann_vector::distance::Metric as DiskANNMetric;

use crate::disk_provider::DiskProvider;
use crate::file_format;
use crate::provider::{DefaultContext, FullPrecisionStrategy, Provider};
use crate::runtime;

// Bounds-checked byte readers for safe deserialization of untrusted data.

fn read_u32(data: &[u8], offset: usize) -> Result<u32> {
    data.get(offset..offset + 4)
        .and_then(|s| s.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| anyhow!("truncated data at offset {}", offset))
}

fn read_u64_le(data: &[u8], offset: usize) -> Result<u64> {
    data.get(offset..offset + 8)
        .and_then(|s| s.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or_else(|| anyhow!("truncated data at offset {}", offset))
}

fn read_f32(data: &[u8], offset: usize) -> Result<f32> {
    data.get(offset..offset + 4)
        .and_then(|s| s.try_into().ok())
        .map(f32::from_le_bytes)
        .ok_or_else(|| anyhow!("truncated data at offset {}", offset))
}

/// Read a contiguous block of f32 values from a byte slice.
fn read_f32_vec(data: &[u8], offset: usize, count: usize) -> Result<Vec<f32>> {
    let byte_len = count.checked_mul(4).ok_or_else(|| anyhow!("overflow computing f32 vec size"))?;
    let end = offset.checked_add(byte_len).ok_or_else(|| anyhow!("overflow computing f32 vec end"))?;
    if end > data.len() {
        return Err(anyhow!("truncated data: need {} bytes at offset {}, have {}", byte_len, offset, data.len()));
    }
    Ok(data[offset..end]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Global index registry.
static INDEXES: LazyLock<DashMap<String, Arc<ManagedIndex>>> = LazyLock::new(DashMap::new);

/// Unified index: either in-memory (read-write) or disk-backed (read-only).
pub enum ManagedIndex {
    InMemory(InMemoryIndex),
    Disk(DiskIndex),
}

impl ManagedIndex {
    pub fn name(&self) -> &str {
        match self {
            ManagedIndex::InMemory(idx) => &idx.name,
            ManagedIndex::Disk(idx) => &idx.name,
        }
    }

    pub fn dimension(&self) -> usize {
        match self {
            ManagedIndex::InMemory(idx) => idx.dimension,
            ManagedIndex::Disk(idx) => idx.provider.dimension(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ManagedIndex::InMemory(idx) => idx.provider.len(),
            ManagedIndex::Disk(idx) => idx.provider.len(),
        }
    }

    pub fn metric(&self) -> Metric {
        match self {
            ManagedIndex::InMemory(idx) => idx.metric,
            ManagedIndex::Disk(idx) => idx.provider.metric(),
        }
    }

    pub fn max_degree(&self) -> u32 {
        match self {
            ManagedIndex::InMemory(idx) => idx.max_degree,
            ManagedIndex::Disk(idx) => idx.provider.max_degree() as u32,
        }
    }

    pub fn build_complexity(&self) -> u32 {
        match self {
            ManagedIndex::InMemory(idx) => idx.build_complexity,
            ManagedIndex::Disk(idx) => idx.build_complexity,
        }
    }

    pub fn alpha(&self) -> f32 {
        match self {
            ManagedIndex::InMemory(idx) => idx.alpha,
            ManagedIndex::Disk(_) => 0.0,
        }
    }

    pub fn is_read_only(&self) -> bool {
        matches!(self, ManagedIndex::Disk(_))
    }

    pub fn add(&self, vector: &[f32]) -> Result<u64> {
        match self {
            ManagedIndex::InMemory(idx) => idx.add(vector),
            ManagedIndex::Disk(_) => Err(anyhow!("Cannot add to read-only disk index")),
        }
    }

    pub fn search(&self, query: &[f32], k: usize, search_complexity: u32) -> Result<Vec<(u64, f32)>> {
        match self {
            ManagedIndex::InMemory(idx) => idx.search(query, k, search_complexity),
            ManagedIndex::Disk(idx) => idx.search(query, k, search_complexity),
        }
    }

    /// Batch search: multiple queries. DiskIndex uses lock-step GPU batching;
    /// InMemoryIndex falls back to sequential per-query search.
    pub fn search_batch(
        &self,
        queries: &[&[f32]],
        k: usize,
        search_complexity: u32,
    ) -> Result<Vec<Vec<(u64, f32)>>> {
        match self {
            ManagedIndex::InMemory(idx) => idx.search_batch(queries, k, search_complexity),
            ManagedIndex::Disk(idx) => idx.search_batch(queries, k, search_complexity),
        }
    }
}

/// In-memory vector store backed by DiskANN graph-based ANN search.
pub struct InMemoryIndex {
    pub name: String,
    pub dimension: usize,
    pub metric: Metric,
    pub max_degree: u32,
    pub build_complexity: u32,
    pub alpha: f32,
    provider: Provider,
    index: RwLock<Option<DiskANNIndex<Provider>>>,
    next_label: AtomicU64,
}

/// Disk-backed read-only index loaded from .diskann file.
pub struct DiskIndex {
    pub name: String,
    pub path: String,
    pub build_complexity: u32,
    provider: DiskProvider,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    L2,
    InnerProduct,
}

impl std::fmt::Display for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Metric::L2 => write!(f, "L2"),
            Metric::InnerProduct => write!(f, "IP"),
        }
    }
}

impl Metric {
    pub fn to_diskann(self) -> DiskANNMetric {
        match self {
            Metric::L2 => DiskANNMetric::L2,
            Metric::InnerProduct => DiskANNMetric::InnerProduct,
        }
    }
}

/// Reusable search scratch space to reduce per-search allocations.
/// Buffers grow as needed but are never shrunk.
struct SearchContext {
    ids: Vec<u32>,
    distances: Vec<f32>,
}

impl SearchContext {
    fn new() -> Self {
        Self {
            ids: Vec::new(),
            distances: Vec::new(),
        }
    }

    fn ensure_capacity(&mut self, k: usize) {
        if self.ids.len() < k {
            self.ids.resize(k, 0);
            self.distances.resize(k, 0.0);
        }
    }

    fn split_slices(&mut self, k: usize) -> (&mut [u32], &mut [f32]) {
        (&mut self.ids[..k], &mut self.distances[..k])
    }
}

thread_local! {
    static SEARCH_CTX: RefCell<SearchContext> = RefCell::new(SearchContext::new());
}

impl InMemoryIndex {
    /// Create a detached (unregistered) index for streaming build.
    pub fn new_detached(
        dimension: usize,
        metric: Metric,
        max_degree: u32,
        build_complexity: u32,
        alpha: f32,
    ) -> Self {
        Self::new(String::new(), dimension, metric, max_degree, build_complexity, alpha)
    }

    fn new(
        name: String,
        dimension: usize,
        metric: Metric,
        max_degree: u32,
        build_complexity: u32,
        alpha: f32,
    ) -> Self {
        let diskann_metric = metric.to_diskann();
        let provider = Provider::new(dimension, max_degree as usize, diskann_metric);

        Self {
            name,
            dimension,
            metric,
            max_degree,
            build_complexity,
            alpha,
            provider,
            index: RwLock::new(None),
            next_label: AtomicU64::new(0),
        }
    }

    /// Add a single vector. Returns the assigned label.
    pub fn add(&self, vector: &[f32]) -> Result<u64> {
        if vector.len() != self.dimension {
            return Err(anyhow!(
                "Expected dimension {}, got {}",
                self.dimension,
                vector.len()
            ));
        }

        let label = self.next_label.fetch_add(1, Ordering::Relaxed) as u32;

        // Fast path: index already initialized
        {
            let idx_guard = self.index.read();
            if let Some(index) = idx_guard.as_ref() {
                let strategy = FullPrecisionStrategy::new();
                let ctx = DefaultContext;
                runtime::block_on(index.insert(strategy, &ctx, &label, vector))
                    .map_err(|e| anyhow!("DiskANN insert error: {}", e))?;
                return Ok(label as u64);
            }
        }

        // Slow path: first vector, need write lock
        let mut idx_guard = self.index.write();
        if let Some(index) = idx_guard.as_ref() {
            let strategy = FullPrecisionStrategy::new();
            let ctx = DefaultContext;
            runtime::block_on(index.insert(strategy, &ctx, &label, vector))
                .map_err(|e| anyhow!("DiskANN insert error: {}", e))?;
        } else {
            self.provider.insert_start_point(label, vector.to_vec());

            let prune_kind = PruneKind::from_metric(self.metric.to_diskann());
            let mut builder = Builder::new(
                self.max_degree as usize,
                MaxDegree::default_slack(),
                self.build_complexity as usize,
                prune_kind,
            );
            builder.alpha(self.alpha);
            let config = builder
                .build()
                .map_err(|e| anyhow!("DiskANN config error: {}", e))?;

            let index = DiskANNIndex::new(config, self.provider.clone(), None);
            *idx_guard = Some(index);
        }

        Ok(label as u64)
    }

    pub fn search(&self, query: &[f32], k: usize, search_complexity: u32) -> Result<Vec<(u64, f32)>> {
        if query.len() != self.dimension {
            return Err(anyhow!(
                "Query dimension {} doesn't match index dimension {}",
                query.len(),
                self.dimension
            ));
        }

        let n = self.provider.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let k = k.min(n);

        if n == 1 {
            let dist = self.single_vector_distance(query);
            return Ok(vec![(0, dist)]);
        }

        let idx_guard = self.index.read();
        let index = idx_guard
            .as_ref()
            .ok_or_else(|| anyhow!("Index not initialized"))?;

        let strategy = FullPrecisionStrategy::new();
        let ctx = DefaultContext;

        let base_l = if search_complexity > 0 {
            search_complexity as usize
        } else {
            self.build_complexity as usize
        };
        let l_search = k.max(base_l);
        let params = SearchParams::new(k, l_search, None)
            .map_err(|e| anyhow!("SearchParams error: {}", e))?;

        // Use thread-local scratch buffers to avoid per-search allocations
        SEARCH_CTX.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.ensure_capacity(k);

            // Zero out reused buffers
            scratch.ids[..k].fill(0);
            scratch.distances[..k].fill(0.0);

            let result_count = {
                let (id_slice, dist_slice) = scratch.split_slices(k);
                let mut buffer = IdDistance::new(id_slice, dist_slice);

                let stats =
                    runtime::block_on(index.search(&strategy, &ctx, query, &params, &mut buffer))
                        .map_err(|e| anyhow!("DiskANN search error: {}", e))?;

                stats.result_count as usize
            };

            let results: Vec<(u64, f32)> = scratch.ids[..result_count]
                .iter()
                .zip(scratch.distances[..result_count].iter())
                .map(|(id, dist)| (*id as u64, *dist))
                .collect();

            Ok(results)
        })
    }

    /// Multi-query batch search with GPU acceleration.
    ///
    /// Uses Provider's lock-step BFS with Metal GPU batch distance when enough
    /// work accumulates. Falls back to sequential per-query search via the
    /// DiskANN crate for very small batches (nq < 2).
    pub fn search_batch(
        &self,
        queries: &[&[f32]],
        k: usize,
        search_complexity: u32,
    ) -> Result<Vec<Vec<(u64, f32)>>> {
        for (i, q) in queries.iter().enumerate() {
            if q.len() != self.dimension {
                return Err(anyhow!(
                    "Query {} dimension {} doesn't match index dimension {}",
                    i, q.len(), self.dimension
                ));
            }
        }

        let base_l = if search_complexity > 0 {
            search_complexity as usize
        } else {
            self.build_complexity as usize
        };
        let l_search = k.max(base_l);

        Ok(self.provider.search_batch(queries, k, l_search, self.metric))
    }

    /// Search writing results directly into caller-provided buffers.
    /// Returns number of results written. Avoids intermediate Vec allocation.
    pub fn search_into(
        &self,
        query: &[f32],
        k: usize,
        search_complexity: u32,
        out_labels: &mut [i64],
        out_distances: &mut [f32],
    ) -> Result<usize> {
        if query.len() != self.dimension {
            return Err(anyhow!(
                "Query dimension {} doesn't match index dimension {}",
                query.len(),
                self.dimension
            ));
        }

        let n = self.provider.len();
        if n == 0 {
            return Ok(0);
        }

        let k = k.min(n);

        if n == 1 {
            let dist = self.single_vector_distance(query);
            if !out_labels.is_empty() && !out_distances.is_empty() {
                out_labels[0] = 0;
                out_distances[0] = dist;
                return Ok(1);
            }
            return Ok(0);
        }

        let idx_guard = self.index.read();
        let index = idx_guard
            .as_ref()
            .ok_or_else(|| anyhow!("Index not initialized"))?;

        let strategy = FullPrecisionStrategy::new();
        let ctx = DefaultContext;

        let base_l = if search_complexity > 0 {
            search_complexity as usize
        } else {
            self.build_complexity as usize
        };
        let l_search = k.max(base_l);
        let params = SearchParams::new(k, l_search, None)
            .map_err(|e| anyhow!("SearchParams error: {}", e))?;

        SEARCH_CTX.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.ensure_capacity(k);
            scratch.ids[..k].fill(0);
            scratch.distances[..k].fill(0.0);

            let result_count = {
                let (id_slice, dist_slice) = scratch.split_slices(k);
                let mut buffer = IdDistance::new(id_slice, dist_slice);
                let stats =
                    runtime::block_on(index.search(&strategy, &ctx, query, &params, &mut buffer))
                        .map_err(|e| anyhow!("DiskANN search error: {}", e))?;
                stats.result_count as usize
            };

            let cap = result_count.min(out_labels.len()).min(out_distances.len());
            for i in 0..cap {
                out_labels[i] = scratch.ids[i] as i64;
                out_distances[i] = scratch.distances[i];
            }
            Ok(cap)
        })
    }

    /// Get adjacency lists for all vectors 0..count, each padded/truncated to max_deg.
    pub fn get_all_adjacency(&self, count: usize, max_deg: usize) -> Vec<Vec<u32>> {
        let mut result = Vec::with_capacity(count);
        for id in 0..count as u32 {
            let neighbors = self.provider.get_neighbors(id);
            let mut adj = neighbors.unwrap_or_default();
            adj.truncate(max_deg);
            result.push(adj);
        }
        result
    }

    /// Get entry point IDs from the provider.
    pub fn get_entry_points(&self) -> Vec<u32> {
        self.provider.get_entry_points()
    }

    /// Serialize the index to bytes (reuses the .diskann binary format).
    /// If SQ8 is active, appends quantization data after the standard format.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>> {
        let mut cursor = Cursor::new(Vec::new());
        file_format::write_index(&mut cursor, &self.provider, self.metric, self.build_complexity)
            .map_err(|e| anyhow!("Serialization error: {}", e))?;

        // Append SQ8 data if quantized
        if let Some(params) = self.provider.get_sq8_params() {
            if let Some(qdata) = self.provider.get_quantized_data() {
                use std::io::Write;
                let marker = b"SQ8\0";
                cursor.write_all(marker)?;
                let dim = self.dimension as u32;
                cursor.write_all(&dim.to_le_bytes())?;
                let qlen = qdata.len() as u64;
                cursor.write_all(&qlen.to_le_bytes())?;
                // Write min + scale params
                for v in &params.min {
                    cursor.write_all(&v.to_le_bytes())?;
                }
                for v in &params.scale {
                    cursor.write_all(&v.to_le_bytes())?;
                }
                // Write quantized data
                cursor.write_all(&qdata)?;
            }
        }

        Ok(cursor.into_inner())
    }

    /// Reconstruct an InMemoryIndex from serialized bytes.
    pub fn from_bytes(data: &[u8], alpha: f32) -> Result<Self> {
        if data.len() < file_format::HEADER_SIZE {
            return Err(anyhow!("Data too small for header ({} bytes)", data.len()));
        }
        if &data[..4] != file_format::MAGIC {
            return Err(anyhow!("Invalid magic bytes"));
        }
        let version = read_u32(data, 4)?;
        if version != file_format::VERSION {
            return Err(anyhow!("Unsupported version {}", version));
        }

        let num_vectors = read_u32(data, 8)?;
        let dimension = read_u32(data, 12)? as usize;
        let max_degree = read_u32(data, 16)?;
        let num_entry_points = read_u32(data, 20)?;
        let metric_byte = data[24];
        let build_complexity = read_u32(data, 28)?;

        if dimension == 0 {
            return Err(anyhow!("Invalid dimension 0"));
        }

        let metric = match metric_byte {
            1 => Metric::InnerProduct,
            _ => Metric::L2,
        };

        // Read entry points (bounds-checked per element)
        let ep_offset = file_format::HEADER_SIZE;
        let mut entry_points = Vec::with_capacity(num_entry_points as usize);
        for i in 0..num_entry_points as usize {
            let off = ep_offset + i * 4;
            entry_points.push(read_u32(data, off)?);
        }

        // Read flat vectors
        let vec_offset = ep_offset + num_entry_points as usize * 4;
        let total_floats = num_vectors as usize * dimension;
        let flat_vectors = read_f32_vec(data, vec_offset, total_floats)?;

        // Read adjacency lists
        let vec_size = total_floats * 4;
        let adj_offset = vec_offset + vec_size;
        let deg = max_degree as usize;
        let adj_size = num_vectors as usize * deg * 4;
        if data.len() < adj_offset + adj_size {
            return Err(anyhow!("Data too small for adjacency"));
        }
        let mut adjacency_lists = Vec::with_capacity(num_vectors as usize);
        for i in 0..num_vectors as usize {
            let row_offset = adj_offset + i * deg * 4;
            let mut neighbors = Vec::new();
            for j in 0..deg {
                let val = read_u32(data, row_offset + j * 4)?;
                if val == u32::MAX {
                    break;
                }
                neighbors.push(val);
            }
            adjacency_lists.push(neighbors);
        }

        // Build provider from raw data
        let diskann_metric = metric.to_diskann();
        let provider = Provider::bulk_load(
            dimension,
            max_degree as usize,
            diskann_metric,
            flat_vectors,
            adjacency_lists,
            entry_points,
            num_vectors,
        );

        // Rebuild DiskANN index on the pre-populated provider
        let prune_kind = PruneKind::from_metric(diskann_metric);
        let mut builder = Builder::new(
            max_degree as usize,
            MaxDegree::default_slack(),
            build_complexity as usize,
            prune_kind,
        );
        builder.alpha(alpha);
        let config = builder
            .build()
            .map_err(|e| anyhow!("DiskANN config error: {}", e))?;

        let index = DiskANNIndex::new(config, provider.clone(), None);

        // Check for SQ8 data appended after the standard format
        let standard_end = adj_offset + adj_size;
        if data.len() > standard_end + 4 && &data[standard_end..standard_end + 4] == b"SQ8\0" {
            let sq_offset = standard_end + 4;
            let sq_dim = read_u32(data, sq_offset)? as usize;
            let qlen = read_u64_le(data, sq_offset + 4)? as usize;
            let params_offset = sq_offset + 12;

            // Validate SQ8 section fits in data
            let sq_params_size = sq_dim * 8; // min + scale, each sq_dim * 4 bytes
            let sq_total = params_offset + sq_params_size + qlen;
            if sq_total > data.len() {
                return Err(anyhow!(
                    "SQ8 section truncated: need {} bytes, have {}",
                    sq_total,
                    data.len()
                ));
            }

            // Read min + scale
            let mut mins = Vec::with_capacity(sq_dim);
            let mut scales = Vec::with_capacity(sq_dim);
            for d in 0..sq_dim {
                mins.push(read_f32(data, params_offset + d * 4)?);
            }
            for d in 0..sq_dim {
                scales.push(read_f32(data, params_offset + sq_dim * 4 + d * 4)?);
            }

            let qdata_offset = params_offset + sq_params_size;
            let qdata = data[qdata_offset..qdata_offset + qlen].to_vec();

            provider.load_sq8(
                qdata,
                crate::provider::SQ8Params {
                    min: mins,
                    scale: scales,
                },
            );
        }

        Ok(Self {
            name: String::new(),
            dimension,
            metric,
            max_degree,
            build_complexity,
            alpha,
            provider,
            index: RwLock::new(Some(index)),
            next_label: AtomicU64::new(num_vectors as u64),
        })
    }

    /// Compact the index by rebuilding from non-deleted labels.
    /// Returns a new InMemoryIndex with contiguous labels 0..n and
    /// a mapping from old_label -> new_label for each kept vector.
    /// The caller provides the set of deleted labels to exclude.
    pub fn compact(&self, deleted_labels: &std::collections::HashSet<u32>) -> Result<(Self, Vec<(u32, u32)>)> {
        let count = self.provider.len();
        let mut vectors: Vec<Vec<f32>> = Vec::new();
        let mut label_map: Vec<(u32, u32)> = Vec::new(); // (old_label, new_label)

        for old_label in 0..count as u32 {
            if deleted_labels.contains(&old_label) {
                continue;
            }
            if let Some(vec) = self.provider.get_vector(old_label) {
                let new_label = vectors.len() as u32;
                label_map.push((old_label, new_label));
                vectors.push(vec);
            }
        }

        let new_index = Self::new_detached(
            self.dimension,
            self.metric,
            self.max_degree,
            self.build_complexity,
            self.alpha,
        );

        for vec in &vectors {
            new_index.add(vec)?;
        }

        Ok((new_index, label_map))
    }

    /// Get a copy of a vector by label.
    pub fn get_vector(&self, label: u32) -> Option<Vec<f32>> {
        self.provider.get_vector(label)
    }

    /// Get current vector count.
    pub fn len(&self) -> usize {
        self.provider.len()
    }

    /// Apply SQ8 scalar quantization to existing vectors.
    /// Reduces memory by 4x for vector storage.
    pub fn quantize_sq8(&self) {
        self.provider.quantize_sq8();
    }

    /// Check if SQ8 quantization is active.
    pub fn is_quantized(&self) -> bool {
        self.provider.is_quantized()
    }

    fn single_vector_distance(&self, query: &[f32]) -> f32 {
        let term = self.provider.get_vector(0);
        match term {
            Some(v) => crate::distance::compute_distance(self.metric, query, &v),
            None => f32::MAX,
        }
    }
}

impl DiskIndex {
    pub fn search(&self, query: &[f32], k: usize, search_complexity: u32) -> Result<Vec<(u64, f32)>> {
        let dim = self.provider.dimension();
        if query.len() != dim {
            return Err(anyhow!(
                "Query dimension {} doesn't match index dimension {}",
                query.len(),
                dim
            ));
        }

        let base_l = if search_complexity > 0 {
            search_complexity as usize
        } else {
            self.build_complexity as usize
        };
        let l_search = k.max(base_l);

        Ok(self.provider.search(query, k, l_search))
    }

    pub fn search_batch(
        &self,
        queries: &[&[f32]],
        k: usize,
        search_complexity: u32,
    ) -> Result<Vec<Vec<(u64, f32)>>> {
        let dim = self.provider.dimension();
        for (i, q) in queries.iter().enumerate() {
            if q.len() != dim {
                return Err(anyhow!(
                    "Query {} dimension {} doesn't match index dimension {}",
                    i,
                    q.len(),
                    dim
                ));
            }
        }

        let base_l = if search_complexity > 0 {
            search_complexity as usize
        } else {
            self.build_complexity as usize
        };
        let l_search = k.max(base_l);

        Ok(self.provider.search_batch(queries, k, l_search))
    }

    pub fn dimension(&self) -> usize {
        self.provider.dimension()
    }
}

// ========================================
// Public API for managing indexes
// ========================================

pub fn create_index(
    name: &str,
    dimension: usize,
    metric: Metric,
    max_degree: u32,
    build_complexity: u32,
    alpha: f32,
) -> Result<()> {
    if INDEXES.contains_key(name) {
        return Err(anyhow!("Index '{}' already exists", name));
    }
    let index = InMemoryIndex::new(
        name.to_string(),
        dimension,
        metric,
        max_degree,
        build_complexity,
        alpha,
    );
    INDEXES.insert(name.to_string(), Arc::new(ManagedIndex::InMemory(index)));
    Ok(())
}

pub fn destroy_index(name: &str) -> Result<()> {
    INDEXES
        .remove(name)
        .ok_or_else(|| anyhow!("Index '{}' not found", name))?;
    Ok(())
}

pub fn get_index(name: &str) -> Result<Arc<ManagedIndex>> {
    INDEXES
        .get(name)
        .map(|r| r.value().clone())
        .ok_or_else(|| anyhow!("Index '{}' not found", name))
}

/// Save an in-memory index to a .diskann file.
pub fn save_index(name: &str, path: &str) -> Result<()> {
    let idx = get_index(name)?;
    match idx.as_ref() {
        ManagedIndex::InMemory(mem) => {
            let file = std::fs::File::create(path)
                .map_err(|e| anyhow!("Failed to create file '{}': {}", path, e))?;
            let mut writer = BufWriter::new(file);
            file_format::write_index(&mut writer, &mem.provider, mem.metric, mem.build_complexity)
                .map_err(|e| anyhow!("Failed to write index: {}", e))?;
            Ok(())
        }
        ManagedIndex::Disk(_) => Err(anyhow!("Cannot save a disk-backed index (already on disk)")),
    }
}

/// Load a .diskann file as a read-only disk-backed index.
pub fn load_index(name: &str, path: &str, build_complexity: u32) -> Result<()> {
    if INDEXES.contains_key(name) {
        return Err(anyhow!("Index '{}' already exists", name));
    }

    let provider = DiskProvider::open(Path::new(path))
        .map_err(|e| anyhow!("Failed to open '{}': {}", path, e))?;

    let bc = if build_complexity > 0 {
        build_complexity
    } else {
        provider.build_complexity()
    };

    let disk_index = DiskIndex {
        name: name.to_string(),
        path: path.to_string(),
        build_complexity: bc,
        provider,
    };

    INDEXES.insert(
        name.to_string(),
        Arc::new(ManagedIndex::Disk(disk_index)),
    );
    Ok(())
}

#[derive(Debug)]
pub struct IndexInfo {
    pub name: String,
    pub dimension: usize,
    pub count: usize,
    pub metric: Metric,
    pub max_degree: u32,
    pub build_complexity: u32,
    pub alpha: f32,
    pub read_only: bool,
}

pub fn list_indexes() -> Vec<IndexInfo> {
    INDEXES
        .iter()
        .map(|entry| {
            let idx = entry.value();
            IndexInfo {
                name: idx.name().to_string(),
                dimension: idx.dimension(),
                count: idx.len(),
                metric: idx.metric(),
                max_degree: idx.max_degree(),
                build_complexity: idx.build_complexity(),
                alpha: idx.alpha(),
                read_only: idx.is_read_only(),
            }
        })
        .collect()
}

pub fn get_info(name: &str) -> Result<IndexInfo> {
    let idx = get_index(name)?;
    Ok(IndexInfo {
        name: idx.name().to_string(),
        dimension: idx.dimension(),
        count: idx.len(),
        metric: idx.metric(),
        max_degree: idx.max_degree(),
        build_complexity: idx.build_complexity(),
        alpha: idx.alpha(),
        read_only: idx.is_read_only(),
    })
}
