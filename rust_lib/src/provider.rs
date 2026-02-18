//! In-memory DataProvider implementation for DiskANN.
//!
//! Uses flat contiguous vector storage for cache-friendly memory layout.
//! Adjacency lists are stored in a DashMap for concurrent insert safety.

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use dashmap::DashMap;
use diskann::{
    ANNError, ANNResult,
    error::Infallible,
    graph::{AdjacencyList, glue},
    provider,
    utils::VectorRepr,
};
use diskann_vector::distance::Metric;
use parking_lot::RwLock;

// ==================
// Storage
// ==================

/// SQ8 quantization parameters: per-dimension min and scale.
/// Dequantize: val = (quantized / 255.0) * scale + min
#[derive(Debug, Clone)]
pub struct SQ8Params {
    pub min: Vec<f32>,
    pub scale: Vec<f32>,
}

/// Optional quantized storage alongside full precision.
#[derive(Debug)]
struct QuantizedStorage {
    data: Vec<u8>, // [id * dim .. (id+1) * dim] as u8
    params: SQ8Params,
}

#[derive(Debug)]
struct Inner {
    /// Flat contiguous vector storage: [id*dim .. (id+1)*dim]
    vectors: RwLock<Vec<f32>>,
    /// Per-node adjacency lists (concurrent-safe for graph build)
    adjacency: DashMap<u32, AdjacencyList<u32>>,
    count: AtomicU32,
    start_point_ids: RwLock<Vec<u32>>,
    #[allow(dead_code)]
    max_degree: usize,
    dimension: usize,
    metric: Metric,
    /// Optional SQ8 quantized storage (set after bulk build)
    quantized: RwLock<Option<QuantizedStorage>>,
}

/// Newtype wrapper for the in-memory provider, allowing trait impls.
#[derive(Debug, Clone)]
pub struct Provider(Arc<Inner>);

impl Provider {
    pub fn new(dimension: usize, max_degree: usize, metric: Metric) -> Self {
        Self(Arc::new(Inner {
            vectors: RwLock::new(Vec::new()),
            adjacency: DashMap::new(),
            count: AtomicU32::new(0),
            start_point_ids: RwLock::new(Vec::new()),
            max_degree,
            dimension,
            metric,
            quantized: RwLock::new(None),
        }))
    }

    /// Reconstruct a Provider from pre-existing data (for deserialization).
    pub fn bulk_load(
        dimension: usize,
        max_degree: usize,
        metric: Metric,
        flat_vectors: Vec<f32>,
        adjacency_lists: Vec<Vec<u32>>,
        entry_points: Vec<u32>,
        count: u32,
    ) -> Self {
        let inner = Arc::new(Inner {
            vectors: RwLock::new(flat_vectors),
            adjacency: DashMap::new(),
            count: AtomicU32::new(count),
            start_point_ids: RwLock::new(entry_points),
            max_degree,
            dimension,
            metric,
            quantized: RwLock::new(None),
        });

        for (id, neighbors) in adjacency_lists.into_iter().enumerate() {
            let mut adj = AdjacencyList::new();
            adj.extend_from_slice(&neighbors);
            inner.adjacency.insert(id as u32, adj);
        }

        Self(inner)
    }

    pub fn len(&self) -> usize {
        self.0.count.load(Ordering::Relaxed) as usize
    }

    /// Insert as a start point. Called for the very first vector.
    pub fn insert_start_point(&self, id: u32, vector: Vec<f32>) {
        {
            let mut vecs = self.0.vectors.write();
            let offset = id as usize * self.0.dimension;
            if vecs.len() < offset + self.0.dimension {
                vecs.resize(offset + self.0.dimension, 0.0);
            }
            vecs[offset..offset + self.0.dimension].copy_from_slice(&vector);
        }
        self.0.adjacency.insert(id, AdjacencyList::new());
        self.0.count.fetch_max(id + 1, Ordering::Relaxed);
        self.0.start_point_ids.write().push(id);
    }

    /// Get a copy of the vector data for the given id.
    /// If SQ8 is active and the vector is in quantized range, dequantizes.
    pub fn get_vector(&self, id: u32) -> Option<Vec<f32>> {
        let dim = self.0.dimension;
        let offset = id as usize * dim;

        // Try full precision first
        {
            let vecs = self.0.vectors.read();
            if offset + dim <= vecs.len() {
                return Some(vecs[offset..offset + dim].to_vec());
            }
        }

        // Fall back to quantized storage
        let q_guard = self.0.quantized.read();
        if let Some(q) = q_guard.as_ref() {
            if offset + dim <= q.data.len() {
                let mut result = vec![0.0f32; dim];
                for d in 0..dim {
                    result[d] =
                        (q.data[offset + d] as f32 / 255.0) * q.params.scale[d] + q.params.min[d];
                }
                return Some(result);
            }
        }
        None
    }

    /// Get a copy of the neighbor list for the given id.
    pub fn get_neighbors(&self, id: u32) -> Option<Vec<u32>> {
        self.0.adjacency.get(&id).map(|adj| adj.to_vec())
    }

    /// Quantize all existing vectors to SQ8 (u8 per dimension).
    /// Computes per-dimension min/max, scales to [0, 255].
    /// Full precision vectors are kept for new inserts; quantized data
    /// is used for search (dequantized on the fly in get_element).
    pub fn quantize_sq8(&self) {
        let vecs = self.0.vectors.read();
        let count = self.0.count.load(Ordering::Relaxed) as usize;
        let dim = self.0.dimension;
        if count == 0 || dim == 0 {
            return;
        }

        // Compute per-dimension min/max
        let mut mins = vec![f32::MAX; dim];
        let mut maxs = vec![f32::MIN; dim];
        for i in 0..count {
            let offset = i * dim;
            for d in 0..dim {
                let val = vecs[offset + d];
                if val < mins[d] {
                    mins[d] = val;
                }
                if val > maxs[d] {
                    maxs[d] = val;
                }
            }
        }

        // Compute scale per dimension
        let mut scale = vec![0.0f32; dim];
        for d in 0..dim {
            let range = maxs[d] - mins[d];
            scale[d] = if range > 0.0 { range } else { 1.0 };
        }

        // Quantize to u8
        let mut quantized = vec![0u8; count * dim];
        for i in 0..count {
            let offset = i * dim;
            for d in 0..dim {
                let normalized = (vecs[offset + d] - mins[d]) / scale[d];
                quantized[offset + d] = (normalized * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }

        let params = SQ8Params {
            min: mins,
            scale,
        };
        *self.0.quantized.write() = Some(QuantizedStorage {
            data: quantized,
            params,
        });
    }

    /// Whether SQ8 quantization is active.
    pub fn is_quantized(&self) -> bool {
        self.0.quantized.read().is_some()
    }

    /// Get SQ8 params (for serialization).
    pub fn get_sq8_params(&self) -> Option<SQ8Params> {
        self.0.quantized.read().as_ref().map(|q| q.params.clone())
    }

    /// Get raw quantized data (for serialization).
    pub fn get_quantized_data(&self) -> Option<Vec<u8>> {
        self.0.quantized.read().as_ref().map(|q| q.data.clone())
    }

    /// Load quantized data from deserialization.
    pub fn load_sq8(&self, data: Vec<u8>, params: SQ8Params) {
        *self.0.quantized.write() = Some(QuantizedStorage { data, params });
    }

    /// Memory used by vector storage (full precision + quantized).
    pub fn vector_memory_bytes(&self) -> usize {
        let vecs = self.0.vectors.read();
        let mut size = vecs.len() * std::mem::size_of::<f32>();
        if let Some(q) = self.0.quantized.read().as_ref() {
            size += q.data.len();
            size += q.params.min.len() * std::mem::size_of::<f32>() * 2;
        }
        size
    }

    /// Lock-step multi-query batch search with GPU acceleration.
    ///
    /// Holds the vectors read lock once for the entire search. Aggregates
    /// neighbor distance work across all queries and dispatches to Metal GPU
    /// when total work exceeds MIN_GPU_WORK. Falls back to CPU SIMD otherwise.
    pub fn search_batch(
        &self,
        queries: &[&[f32]],
        k: usize,
        l_search: usize,
        metric: crate::index_manager::Metric,
    ) -> Vec<Vec<(u64, f32)>> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let nq = queries.len();
        if nq == 0 || self.len() == 0 || k == 0 {
            return vec![Vec::new(); nq];
        }

        // Single query: skip lock-step overhead
        if nq == 1 {
            return vec![self.search_single(queries[0], k, l_search, metric)];
        }

        let k = k.min(self.len());
        let l = l_search.max(k);
        let dim = self.0.dimension;
        let metric_code: u8 = match metric {
            crate::index_manager::Metric::L2 => 0,
            crate::index_manager::Metric::InnerProduct => 1,
        };

        // Per-query state
        struct QState {
            visited: hashbrown::HashSet<u32>,
            candidates: BinaryHeap<Reverse<(FloatOrd, u32)>>,
            result: Vec<(f32, u32)>,
            active: bool,
        }

        let mut states: Vec<QState> = (0..nq)
            .map(|_| QState {
                visited: hashbrown::HashSet::with_capacity(l * 2),
                candidates: BinaryHeap::new(),
                result: Vec::new(),
                active: true,
            })
            .collect();

        // Hold vectors read lock for entire search
        let vecs = self.0.vectors.read();
        let n_vecs = self.0.count.load(std::sync::atomic::Ordering::Relaxed);
        let entry_points = self.0.start_point_ids.read().clone();

        // Helper: get vector slice from locked vecs
        let get_vec = |id: u32| -> Option<&[f32]> {
            let offset = id as usize * dim;
            let end = offset + dim;
            if end <= vecs.len() {
                Some(&vecs[offset..end])
            } else {
                None
            }
        };

        // Seed entry points
        for (qi, state) in states.iter_mut().enumerate() {
            for &ep in &entry_points {
                if state.visited.insert(ep) {
                    if let Some(vec) = get_vec(ep) {
                        let dist = crate::distance::compute_distance(metric, queries[qi], vec);
                        state.candidates.push(Reverse((FloatOrd(dist), ep)));
                        state.result.push((dist, ep));
                    }
                }
            }
            state.result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Flat queries buffer for GPU
        let queries_flat: Vec<f32> = queries.iter().flat_map(|q| q.iter().copied()).collect();

        // Scratch buffers
        let max_per_iter = nq * self.0.max_degree;
        let mut all_neighbor_ids: Vec<u32> = Vec::with_capacity(max_per_iter);
        let mut all_query_map: Vec<u32> = Vec::with_capacity(max_per_iter);
        let mut batch_buf: Vec<f32> = Vec::with_capacity(max_per_iter * dim);
        let mut batch_dist: Vec<f32> = Vec::with_capacity(max_per_iter);

        loop {
            let active_count = states.iter().filter(|s| s.active).count();
            if active_count == 0 {
                break;
            }

            all_neighbor_ids.clear();
            all_query_map.clear();

            for (qi, state) in states.iter_mut().enumerate() {
                if !state.active {
                    continue;
                }

                match state.candidates.pop() {
                    None => {
                        state.active = false;
                        continue;
                    }
                    Some(Reverse((FloatOrd(c_dist), c_id))) => {
                        if state.result.len() >= l && c_dist > state.result[l - 1].0 {
                            state.active = false;
                            continue;
                        }

                        if let Some(adj) = self.0.adjacency.get(&c_id) {
                            let neighbors: &[u32] = &*adj;
                            for &neighbor in neighbors {
                                if neighbor >= n_vecs {
                                    continue;
                                }
                                if !state.visited.insert(neighbor) {
                                    continue;
                                }
                                if get_vec(neighbor).is_none() {
                                    continue;
                                }
                                all_neighbor_ids.push(neighbor);
                                all_query_map.push(qi as u32);
                            }
                        }
                    }
                }
            }

            if all_neighbor_ids.is_empty() {
                continue;
            }

            let total_n = all_neighbor_ids.len();

            // GPU path
            let use_gpu = total_n * dim >= crate::metal_ffi::MIN_GPU_WORK;
            if use_gpu {
                batch_buf.clear();
                batch_buf.reserve(total_n * dim);
                for &id in &all_neighbor_ids {
                    if let Some(v) = get_vec(id) {
                        batch_buf.extend_from_slice(v);
                    }
                }
                batch_dist.resize(total_n, 0.0);

                let gpu_ok = crate::metal_ffi::metal_multi_batch_distances(
                    &queries_flat,
                    &batch_buf,
                    &all_query_map,
                    total_n,
                    nq,
                    dim,
                    metric_code,
                    &mut batch_dist,
                );

                if gpu_ok {
                    for i in 0..total_n {
                        let qi = all_query_map[i] as usize;
                        let state = &mut states[qi];
                        Self::insert_result_batch(&mut state.result, &mut state.candidates, l, batch_dist[i], all_neighbor_ids[i]);
                    }
                    continue;
                }
            }

            // CPU fallback
            for i in 0..total_n {
                let qi = all_query_map[i] as usize;
                let neighbor = all_neighbor_ids[i];
                if let Some(vec) = get_vec(neighbor) {
                    let dist = crate::distance::compute_distance(metric, queries[qi], vec);
                    let state = &mut states[qi];
                    Self::insert_result_batch(&mut state.result, &mut state.candidates, l, dist, neighbor);
                }
            }
        }

        states
            .into_iter()
            .map(|state| {
                state
                    .result
                    .into_iter()
                    .take(k)
                    .map(|(dist, id)| (id as u64, dist))
                    .collect()
            })
            .collect()
    }

    /// Single-query search (used by search_batch for nq=1 and by InMemoryIndex).
    fn search_single(
        &self,
        query: &[f32],
        k: usize,
        l_search: usize,
        metric: crate::index_manager::Metric,
    ) -> Vec<(u64, f32)> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let n = self.len();
        if n == 0 || k == 0 {
            return Vec::new();
        }

        let k = k.min(n);
        let l = l_search.max(k);
        let dim = self.0.dimension;

        let vecs = self.0.vectors.read();
        let n_vecs = self.0.count.load(std::sync::atomic::Ordering::Relaxed);
        let entry_points = self.0.start_point_ids.read().clone();

        let get_vec = |id: u32| -> Option<&[f32]> {
            let offset = id as usize * dim;
            let end = offset + dim;
            if end <= vecs.len() {
                Some(&vecs[offset..end])
            } else {
                None
            }
        };

        let mut visited = hashbrown::HashSet::with_capacity(l * 2);
        let mut candidates: BinaryHeap<Reverse<(FloatOrd, u32)>> = BinaryHeap::new();
        let mut result: Vec<(f32, u32)> = Vec::new();

        for &ep in &entry_points {
            if visited.insert(ep) {
                if let Some(vec) = get_vec(ep) {
                    let dist = crate::distance::compute_distance(metric, query, vec);
                    candidates.push(Reverse((FloatOrd(dist), ep)));
                    result.push((dist, ep));
                }
            }
        }
        result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        while let Some(Reverse((FloatOrd(c_dist), c_id))) = candidates.pop() {
            if result.len() >= l && c_dist > result[l - 1].0 {
                break;
            }

            if let Some(adj) = self.0.adjacency.get(&c_id) {
                let neighbors: &[u32] = &*adj;
                for &neighbor in neighbors {
                    if neighbor >= n_vecs {
                        continue;
                    }
                    if !visited.insert(neighbor) {
                        continue;
                    }
                    if let Some(vec) = get_vec(neighbor) {
                        let dist = crate::distance::compute_distance(metric, query, vec);
                        Self::insert_result_batch(&mut result, &mut candidates, l, dist, neighbor);
                    }
                }
            }
        }

        result
            .into_iter()
            .take(k)
            .map(|(dist, id)| (id as u64, dist))
            .collect()
    }

    #[inline]
    fn insert_result_batch(
        result: &mut Vec<(f32, u32)>,
        candidates: &mut std::collections::BinaryHeap<std::cmp::Reverse<(FloatOrd, u32)>>,
        l: usize,
        dist: f32,
        neighbor: u32,
    ) {
        if result.len() < l || dist < result[result.len() - 1].0 {
            let pos = result
                .binary_search_by(|probe| {
                    probe.0.partial_cmp(&dist).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or_else(|e| e);
            result.insert(pos, (dist, neighbor));
            if result.len() > l {
                result.truncate(l);
            }
            candidates.push(std::cmp::Reverse((FloatOrd(dist), neighbor)));
        }
    }

    pub fn dim(&self) -> usize {
        self.0.dimension
    }

    pub fn metric(&self) -> Metric {
        self.0.metric
    }

    pub fn max_degree(&self) -> usize {
        self.0.max_degree
    }

    /// Write flat vectors to a writer (for serialization).
    pub fn write_vectors_to(&self, w: &mut dyn Write) -> std::io::Result<()> {
        let vecs = self.0.vectors.read();
        let count = self.0.count.load(Ordering::Relaxed) as usize;
        let total = count * self.0.dimension;
        let data = &vecs[..total];
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, total * 4) };
        w.write_all(bytes)
    }

    /// Write fixed-width padded adjacency to a writer.
    /// Each node gets exactly `max_degree` u32 slots, unused padded with u32::MAX.
    pub fn write_adjacency_to(&self, w: &mut dyn Write, max_degree: usize) -> std::io::Result<()> {
        let count = self.0.count.load(Ordering::Relaxed) as usize;
        let sentinel = u32::MAX;
        let mut row = vec![sentinel; max_degree];
        for id in 0..count as u32 {
            // Fill row with sentinel
            row.fill(sentinel);
            if let Some(adj) = self.0.adjacency.get(&id) {
                let neighbors: &[u32] = &*adj;
                let n = neighbors.len().min(max_degree);
                row[..n].copy_from_slice(&neighbors[..n]);
            }
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(row.as_ptr() as *const u8, max_degree * 4)
            };
            w.write_all(bytes)?;
        }
        Ok(())
    }

    /// Expose start point IDs for serialization.
    pub fn get_entry_points(&self) -> Vec<u32> {
        self.0.start_point_ids.read().clone()
    }
}

// ==================
// Context
// ==================

#[derive(Debug, Clone)]
pub struct DefaultContext;

impl provider::ExecutionContext for DefaultContext {
    fn wrap_spawn<F, T>(&self, f: F) -> impl std::future::Future<Output = T> + Send + 'static
    where
        F: std::future::Future<Output = T> + Send + 'static,
    {
        f
    }
}

// ==================
// Error type
// ==================

#[derive(Debug, Clone, Copy)]
pub struct ProviderError(pub u32);

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid id {}", self.0)
    }
}

impl std::error::Error for ProviderError {}

impl From<ProviderError> for ANNError {
    #[track_caller]
    fn from(err: ProviderError) -> Self {
        ANNError::opaque(err)
    }
}

diskann::always_escalate!(ProviderError);

// ==================
// DataProvider
// ==================

impl provider::DataProvider for Provider {
    type Context = DefaultContext;
    type InternalId = u32;
    type ExternalId = u32;
    type Error = ProviderError;

    fn to_internal_id(&self, _context: &DefaultContext, gid: &u32) -> Result<u32, ProviderError> {
        Ok(*gid)
    }

    fn to_external_id(&self, _context: &DefaultContext, id: u32) -> Result<u32, ProviderError> {
        Ok(id)
    }
}

// ==================
// Delete
// ==================

impl provider::Delete for Provider {
    async fn delete(
        &self,
        _context: &Self::Context,
        _gid: &Self::ExternalId,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    async fn release(
        &self,
        _context: &Self::Context,
        _id: Self::InternalId,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    async fn status_by_internal_id(
        &self,
        _context: &DefaultContext,
        id: u32,
    ) -> Result<provider::ElementStatus, Self::Error> {
        if (id as usize) < self.len() {
            Ok(provider::ElementStatus::Valid)
        } else {
            Err(ProviderError(id))
        }
    }

    fn status_by_external_id(
        &self,
        context: &DefaultContext,
        gid: &u32,
    ) -> impl std::future::Future<Output = Result<provider::ElementStatus, Self::Error>> + Send
    {
        self.status_by_internal_id(context, *gid)
    }
}

// ==================
// SetElement
// ==================

impl provider::SetElement<[f32]> for Provider {
    type SetError = ANNError;
    type Guard = provider::NoopGuard<u32>;

    async fn set_element(
        &self,
        _context: &DefaultContext,
        id: &u32,
        element: &[f32],
    ) -> Result<Self::Guard, Self::SetError> {
        {
            let mut vecs = self.0.vectors.write();
            let offset = *id as usize * self.0.dimension;
            if vecs.len() < offset + self.0.dimension {
                vecs.resize(offset + self.0.dimension, 0.0);
            }
            vecs[offset..offset + self.0.dimension].copy_from_slice(element);
        }
        self.0.adjacency.insert(*id, AdjacencyList::new());
        self.0.count.fetch_max(*id + 1, Ordering::Relaxed);
        Ok(provider::NoopGuard::new(*id))
    }
}

// ==================
// DefaultAccessor
// ==================

impl provider::DefaultAccessor for Provider {
    type Accessor<'a> = NeighborHandle<'a>;

    fn default_accessor(&self) -> Self::Accessor<'_> {
        NeighborHandle { inner: &self.0 }
    }
}

// ==================
// NeighborHandle (for NeighborAccessor / NeighborAccessorMut)
// ==================

#[derive(Debug, Clone, Copy)]
pub struct NeighborHandle<'a> {
    inner: &'a Inner,
}

impl provider::HasId for NeighborHandle<'_> {
    type Id = u32;
}

impl provider::NeighborAccessor for NeighborHandle<'_> {
    async fn get_neighbors(
        self,
        id: Self::Id,
        neighbors: &mut AdjacencyList<Self::Id>,
    ) -> ANNResult<Self> {
        match self.inner.adjacency.get(&id) {
            Some(adj) => {
                neighbors.overwrite_trusted(&adj);
                Ok(self)
            }
            None => Err(ANNError::opaque(ProviderError(id))),
        }
    }
}

impl provider::NeighborAccessorMut for NeighborHandle<'_> {
    async fn set_neighbors(self, id: Self::Id, neighbors: &[Self::Id]) -> ANNResult<Self> {
        match self.inner.adjacency.get_mut(&id) {
            Some(mut adj) => {
                adj.clear();
                adj.extend_from_slice(neighbors);
                Ok(self)
            }
            None => Err(ANNError::opaque(ProviderError(id))),
        }
    }

    async fn append_vector(self, id: Self::Id, neighbors: &[Self::Id]) -> ANNResult<Self> {
        match self.inner.adjacency.get_mut(&id) {
            Some(mut adj) => {
                adj.extend_from_slice(neighbors);
                Ok(self)
            }
            None => Err(ANNError::opaque(ProviderError(id))),
        }
    }
}

// ==================
// Accessor (data accessor with buffer)
// ==================

#[derive(Debug)]
pub struct ProviderAccessor<'a> {
    inner: &'a Inner,
    buffer: Box<[f32]>,
}

impl<'a> ProviderAccessor<'a> {
    fn new(inner: &'a Inner) -> Self {
        let buffer = vec![0.0f32; inner.dimension].into_boxed_slice();
        Self { inner, buffer }
    }
}

impl provider::HasId for ProviderAccessor<'_> {
    type Id = u32;
}

impl provider::Accessor for ProviderAccessor<'_> {
    type Extended = Box<[f32]>;
    type Element<'e>
        = &'e [f32]
    where
        Self: 'e;
    type ElementRef<'e> = &'e [f32];
    type GetError = ProviderError;

    async fn get_element(&mut self, id: u32) -> Result<&[f32], ProviderError> {
        let dim = self.inner.dimension;
        let offset = id as usize * dim;

        // Try full precision first
        {
            let vecs = self.inner.vectors.read();
            if offset + dim <= vecs.len() {
                self.buffer.copy_from_slice(&vecs[offset..offset + dim]);
                return Ok(&*self.buffer);
            }
        }

        // Fall back to quantized storage (dequantize into buffer)
        {
            let q_guard = self.inner.quantized.read();
            if let Some(q) = q_guard.as_ref() {
                if offset + dim <= q.data.len() {
                    for d in 0..dim {
                        self.buffer[d] = (q.data[offset + d] as f32 / 255.0) * q.params.scale[d]
                            + q.params.min[d];
                    }
                    return Ok(&*self.buffer);
                }
            }
        }

        Err(ProviderError(id))
    }
}

impl<'a> provider::DelegateNeighbor<'a> for ProviderAccessor<'_> {
    type Delegate = NeighborHandle<'a>;
    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        NeighborHandle { inner: self.inner }
    }
}

impl provider::BuildQueryComputer<[f32]> for ProviderAccessor<'_> {
    type QueryComputerError = Infallible;
    type QueryComputer = <f32 as VectorRepr>::QueryDistance;

    fn build_query_computer(
        &self,
        from: &[f32],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        Ok(f32::query_distance(from, self.inner.metric))
    }
}

impl provider::BuildDistanceComputer for ProviderAccessor<'_> {
    type DistanceComputerError = Infallible;
    type DistanceComputer = <f32 as VectorRepr>::Distance;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        Ok(f32::distance(self.inner.metric, Some(self.inner.dimension)))
    }
}

// ==================
// Glue traits
// ==================

impl glue::SearchExt for ProviderAccessor<'_> {
    fn starting_points(&self) -> impl std::future::Future<Output = ANNResult<Vec<u32>>> + Send {
        let ids = self.inner.start_point_ids.read().clone();
        futures_util::future::ok(ids)
    }
}

impl glue::ExpandBeam<[f32]> for ProviderAccessor<'_> {}
impl glue::FillSet for ProviderAccessor<'_> {}

impl<'a> glue::AsElement<&'a [f32]> for ProviderAccessor<'a> {
    type Error = Infallible;
    fn as_element(
        &mut self,
        vector: &'a [f32],
        _id: Self::Id,
    ) -> impl std::future::Future<Output = Result<Self::Element<'_>, Self::Error>> + Send {
        std::future::ready(Ok(vector))
    }
}

// ==================
// Strategy
// ==================

#[derive(Debug, Default, Clone, Copy)]
pub struct FullPrecisionStrategy;

impl FullPrecisionStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl glue::SearchStrategy<Provider, [f32]> for FullPrecisionStrategy {
    type QueryComputer = <f32 as VectorRepr>::QueryDistance;
    type PostProcessor = glue::CopyIds;
    type SearchAccessorError = Infallible;
    type SearchAccessor<'a> = ProviderAccessor<'a>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a Provider,
        _context: &'a DefaultContext,
    ) -> Result<ProviderAccessor<'a>, Infallible> {
        Ok(ProviderAccessor::new(&provider.0))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

impl glue::PruneStrategy<Provider> for FullPrecisionStrategy {
    type DistanceComputer = <f32 as VectorRepr>::Distance;
    type PruneAccessor<'a> = ProviderAccessor<'a>;
    type PruneAccessorError = Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a Provider,
        _context: &'a DefaultContext,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(ProviderAccessor::new(&provider.0))
    }
}

impl glue::InsertStrategy<Provider, [f32]> for FullPrecisionStrategy {
    type PruneStrategy = Self;

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }

    fn insert_search_accessor<'a>(
        &'a self,
        provider: &'a Provider,
        _context: &'a DefaultContext,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(ProviderAccessor::new(&provider.0))
    }
}

// ==================
// FloatOrd (for BinaryHeap)
// ==================

#[derive(Debug, Clone, Copy, PartialEq)]
struct FloatOrd(f32);

impl Eq for FloatOrd {}

impl PartialOrd for FloatOrd {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FloatOrd {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}
