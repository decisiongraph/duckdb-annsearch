//! Read-only mmap-backed DiskANN index with standalone greedy best-first search.

use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::io;
use std::path::Path;

use memmap2::Mmap;

use crate::file_format::{FileHeader, HEADER_SIZE, MAGIC, VERSION};
use crate::index_manager::Metric;

/// Page-aligned Vec<f32> for zero-copy Metal buffer wrapping.
/// Uses `std::alloc` with page alignment (typically 16384 on Apple Silicon).
struct PageAlignedVec {
    ptr: *mut f32,
    len: usize,
    capacity: usize, // in f32 elements
}

impl PageAlignedVec {
    fn new() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
            capacity: 0,
        }
    }

    fn page_size() -> usize {
        // Apple Silicon page size is 16384; fall back to 4096
        #[cfg(target_os = "macos")]
        {
            16384
        }
        #[cfg(not(target_os = "macos"))]
        {
            4096
        }
    }

    /// Ensure capacity for at least `needed` f32 elements.
    /// Rounds up to page boundary for Metal zero-copy compatibility.
    fn reserve(&mut self, needed: usize) {
        if self.capacity >= needed {
            return;
        }
        // Free old allocation
        if !self.ptr.is_null() {
            unsafe {
                let layout = std::alloc::Layout::from_size_align_unchecked(
                    self.capacity * 4,
                    Self::page_size(),
                );
                std::alloc::dealloc(self.ptr as *mut u8, layout);
            }
        }
        // Round up to page boundary
        let page = Self::page_size();
        let byte_size = needed * 4;
        let aligned_size = (byte_size + page - 1) & !(page - 1);
        let new_cap = aligned_size / 4;
        unsafe {
            let layout = std::alloc::Layout::from_size_align_unchecked(aligned_size, page);
            let p = std::alloc::alloc(layout) as *mut f32;
            if p.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            self.ptr = p;
        }
        self.capacity = new_cap;
    }

    fn clear(&mut self) {
        self.len = 0;
    }

    fn extend_from_slice(&mut self, data: &[f32]) {
        let new_len = self.len + data.len();
        if new_len > self.capacity {
            self.reserve(new_len);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(self.len), data.len());
        }
        self.len = new_len;
    }

    fn as_slice(&self) -> &[f32] {
        if self.ptr.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    fn resize(&mut self, new_len: usize, val: f32) {
        if new_len > self.capacity {
            self.reserve(new_len);
        }
        if new_len > self.len {
            // Fill new elements
            for i in self.len..new_len {
                unsafe { *self.ptr.add(i) = val; }
            }
        }
        self.len = new_len;
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        if self.ptr.is_null() {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }
}

impl Drop for PageAlignedVec {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let layout = std::alloc::Layout::from_size_align_unchecked(
                    self.capacity * 4,
                    Self::page_size(),
                );
                std::alloc::dealloc(self.ptr as *mut u8, layout);
            }
        }
    }
}

// SAFETY: PageAlignedVec owns its allocation, only used within RefCell<SearchContext>
unsafe impl Send for PageAlignedVec {}

/// Bounds-checked u32 read from a byte slice, returning io::Error on truncation.
fn read_u32_io(data: &[u8], offset: usize) -> io::Result<u32> {
    data.get(offset..offset + 4)
        .and_then(|s| s.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("truncated at offset {}", offset)))
}

/// Reusable search buffers to avoid per-query allocations.
struct SearchContext {
    visited: hashbrown::HashSet<u32>,
    candidates: BinaryHeap<std::cmp::Reverse<(FloatOrd, u32)>>,
    result: Vec<(f32, u32)>,
    /// Scratch: unvisited neighbor IDs for current candidate expansion.
    batch_ids: Vec<u32>,
    /// Scratch: contiguous candidate vectors for GPU batch distance (page-aligned for Metal zero-copy).
    batch_buf: PageAlignedVec,
    /// Scratch: output distances from GPU batch distance (page-aligned for Metal zero-copy).
    batch_dist: PageAlignedVec,
}

impl SearchContext {
    fn new() -> Self {
        Self {
            visited: hashbrown::HashSet::new(),
            candidates: BinaryHeap::new(),
            result: Vec::new(),
            batch_ids: Vec::new(),
            batch_buf: PageAlignedVec::new(),
            batch_dist: PageAlignedVec::new(),
        }
    }

    fn clear(&mut self) {
        self.visited.clear();
        self.candidates.clear();
        self.result.clear();
        self.batch_ids.clear();
        // batch_buf and batch_dist are cleared per-iteration, not here
    }
}

thread_local! {
    static SEARCH_CTX: RefCell<SearchContext> = RefCell::new(SearchContext::new());
}

/// Read-only mmap-backed DiskANN index.
///
/// Uses offset-based access into the mmap slice — no raw pointers stored.
/// Mmap is read-only so this is safe to share across threads.
pub struct DiskProvider {
    mmap: Mmap,
    header: FileHeader,
    entry_point_ids: Vec<u32>,
    vectors_offset: usize,
    adjacency_offset: usize,
    /// Pre-computed end of vectors segment for bounds checking.
    vectors_end: usize,
    /// Pre-computed end of adjacency segment for bounds checking.
    adjacency_end: usize,
    metric: Metric,
}

impl DiskProvider {
    /// Open and validate a .diskann file.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
        }

        // Validate magic
        if &mmap[..4] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid magic bytes",
            ));
        }

        let version = read_u32_io(&mmap, 4)?;
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported version {} (expected {})", version, VERSION),
            ));
        }

        let num_vectors = read_u32_io(&mmap, 8)?;
        let dimension = read_u32_io(&mmap, 12)?;
        let max_degree = read_u32_io(&mmap, 16)?;
        let num_entry_points = read_u32_io(&mmap, 20)?;
        let metric_byte = mmap[24];
        // bytes 25..28: padding
        let build_complexity = read_u32_io(&mmap, 28)?;

        let header = FileHeader {
            num_vectors,
            dimension,
            max_degree,
            num_entry_points,
            metric: metric_byte,
            build_complexity,
        };

        // Validate file size
        let expected = header.total_file_size();
        if mmap.len() < expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "file too small: expected {} bytes, got {}",
                    expected,
                    mmap.len()
                ),
            ));
        }

        // Read entry point IDs
        let ep_offset = header.entry_points_offset();
        let mut entry_point_ids = Vec::with_capacity(num_entry_points as usize);
        for i in 0..num_entry_points as usize {
            let off = ep_offset + i * 4;
            entry_point_ids.push(read_u32_io(&mmap, off)?);
        }

        let metric = header.metric_enum();
        let vectors_offset = header.vectors_offset();
        let adjacency_offset = header.adjacency_offset();
        let vectors_end = vectors_offset + header.vectors_size();
        let adjacency_end = adjacency_offset + header.adjacency_size();

        Ok(Self {
            mmap,
            header,
            entry_point_ids,
            vectors_offset,
            adjacency_offset,
            vectors_end,
            adjacency_end,
            metric,
        })
    }

    pub fn dimension(&self) -> usize {
        self.header.dimension as usize
    }

    pub fn len(&self) -> usize {
        self.header.num_vectors as usize
    }

    pub fn max_degree(&self) -> usize {
        self.header.max_degree as usize
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn build_complexity(&self) -> u32 {
        self.header.build_complexity
    }

    /// Zero-copy vector read from mmap via byte-offset slicing.
    /// Returns empty slice if id is out of bounds.
    fn get_vector(&self, id: u32) -> &[f32] {
        let dim = self.header.dimension as usize;
        let byte_offset = self.vectors_offset + id as usize * dim * 4;
        let byte_end = byte_offset + dim * 4;
        if byte_end > self.vectors_end || byte_end > self.mmap.len() {
            return &[];
        }
        let bytes = &self.mmap[byte_offset..byte_end];
        // SAFETY: mmap is aligned to page boundary (always >= 4-byte aligned),
        // vectors_offset is a multiple of 4, and dim*4 is a multiple of 4.
        // Bounds verified above.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, dim) }
    }

    /// Read neighbors from mmap, trimming u32::MAX sentinels.
    /// Returns empty slice if id is out of bounds.
    fn get_neighbors(&self, id: u32) -> &[u32] {
        let deg = self.header.max_degree as usize;
        let byte_offset = self.adjacency_offset + id as usize * deg * 4;
        let byte_end = byte_offset + deg * 4;
        if byte_end > self.adjacency_end || byte_end > self.mmap.len() {
            return &[];
        }
        let bytes = &self.mmap[byte_offset..byte_end];
        // SAFETY: same alignment reasoning as get_vector. Bounds verified above.
        let raw = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, deg) };
        // Find first sentinel
        let len = raw.iter().position(|&x| x == u32::MAX).unwrap_or(deg);
        &raw[..len]
    }

    /// Greedy best-first search on the mmap'd graph.
    ///
    /// When Metal GPU is available and the batch is large enough, neighbor
    /// distance computations are dispatched to the GPU. Otherwise falls
    /// back to CPU SIMD distance.
    pub fn search(&self, query: &[f32], k: usize, l_search: usize) -> Vec<(u64, f32)> {
        let n = self.len();
        if n == 0 || k == 0 {
            return Vec::new();
        }

        let k = k.min(n);
        let l = l_search.max(k);
        let dim = self.dimension();
        let metric_code: u8 = match self.metric {
            Metric::L2 => 0,
            Metric::InnerProduct => 1,
        };

        SEARCH_CTX.with(|cell| {
            let mut ctx = cell.borrow_mut();
            ctx.clear();

            // Destructure into separate borrows to satisfy borrow checker
            let SearchContext {
                ref mut visited,
                ref mut candidates,
                ref mut result,
                ref mut batch_ids,
                ref mut batch_buf,
                ref mut batch_dist,
            } = *ctx;

            // Reserve capacity if needed (grows once, reused across queries)
            let needed = l * 2;
            let cap = visited.capacity();
            if cap < needed {
                visited.reserve(needed - cap);
            }

            // Seed with entry points
            for &ep in &self.entry_point_ids {
                if visited.insert(ep) {
                    let vec = self.get_vector(ep);
                    if vec.is_empty() {
                        continue;
                    }
                    let dist = self.compute_distance(query, vec);
                    candidates.push(std::cmp::Reverse((FloatOrd(dist), ep)));
                    result.push((dist, ep));
                }
            }

            result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            while let Some(std::cmp::Reverse((FloatOrd(c_dist), c_id))) = candidates.pop() {
                if result.len() >= l && c_dist > result[l - 1].0 {
                    break;
                }

                // Collect unvisited neighbors for this candidate
                batch_ids.clear();
                for &neighbor in self.get_neighbors(c_id) {
                    if neighbor >= self.header.num_vectors {
                        continue;
                    }
                    if !visited.insert(neighbor) {
                        continue;
                    }
                    let vec = self.get_vector(neighbor);
                    if vec.is_empty() {
                        continue;
                    }
                    batch_ids.push(neighbor);
                }

                if batch_ids.is_empty() {
                    continue;
                }

                let batch_n = batch_ids.len();

                // Try Metal GPU batch distance when worthwhile
                let use_gpu = batch_n * dim >= crate::metal_ffi::MIN_GPU_WORK;
                if use_gpu {
                    // Gather vectors into page-aligned contiguous buffer
                    batch_buf.clear();
                    batch_buf.reserve(batch_n * dim);
                    for i in 0..batch_n {
                        batch_buf.extend_from_slice(self.get_vector(batch_ids[i]));
                    }
                    batch_dist.resize(batch_n, 0.0);

                    let gpu_ok = crate::metal_ffi::metal_batch_distances(
                        query,
                        batch_buf.as_slice(),
                        batch_n,
                        dim,
                        metric_code,
                        batch_dist.as_mut_slice(),
                    );

                    if gpu_ok {
                        let dist_slice = batch_dist.as_slice();
                        for i in 0..batch_n {
                            let dist = dist_slice[i];
                            let neighbor = batch_ids[i];
                            Self::insert_result(result, candidates, l, dist, neighbor);
                        }
                        continue;
                    }
                }

                // CPU fallback: compute distances individually
                for i in 0..batch_n {
                    let neighbor = batch_ids[i];
                    let vec = self.get_vector(neighbor);
                    let dist = self.compute_distance(query, vec);
                    Self::insert_result(result, candidates, l, dist, neighbor);
                }
            }

            result
                .iter()
                .take(k)
                .map(|&(dist, id)| (id as u64, dist))
                .collect()
        })
    }

    /// Multi-query batch search on the mmap'd graph.
    ///
    /// Runs lock-step BFS across all queries, aggregating neighbor distance
    /// work into single Metal GPU dispatches. Converged queries are removed
    /// from the active set each iteration. Falls back to CPU SIMD when Metal
    /// is unavailable or total work is below threshold.
    pub fn search_batch(
        &self,
        queries: &[&[f32]],
        k: usize,
        l_search: usize,
    ) -> Vec<Vec<(u64, f32)>> {
        let nq = queries.len();
        if nq == 0 || self.len() == 0 || k == 0 {
            return vec![Vec::new(); nq];
        }

        // For single query, delegate to existing single-query path
        if nq == 1 {
            return vec![self.search(queries[0], k, l_search)];
        }

        let k = k.min(self.len());
        let l = l_search.max(k);
        let dim = self.dimension();
        let metric_code: u8 = match self.metric {
            Metric::L2 => 0,
            Metric::InnerProduct => 1,
        };

        // Per-query search state (not thread-local — owned by this call)
        struct QState {
            visited: hashbrown::HashSet<u32>,
            candidates: BinaryHeap<std::cmp::Reverse<(FloatOrd, u32)>>,
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

        // Seed entry points for all queries
        for (qi, state) in states.iter_mut().enumerate() {
            for &ep in &self.entry_point_ids {
                if state.visited.insert(ep) {
                    let vec = self.get_vector(ep);
                    if vec.is_empty() {
                        continue;
                    }
                    let dist = self.compute_distance(queries[qi], vec);
                    state.candidates.push(std::cmp::Reverse((FloatOrd(dist), ep)));
                    state.result.push((dist, ep));
                }
            }
            state
                .result
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Build flat queries buffer for GPU (nq * dim, built once)
        let queries_flat: Vec<f32> = queries.iter().flat_map(|q| q.iter().copied()).collect();

        // Shared scratch buffers (reused across iterations)
        let max_neighbors_per_iter = nq * self.max_degree();
        let mut all_neighbor_ids: Vec<u32> = Vec::with_capacity(max_neighbors_per_iter);
        let mut all_query_map: Vec<u32> = Vec::with_capacity(max_neighbors_per_iter);
        let mut batch_buf: Vec<f32> = Vec::with_capacity(max_neighbors_per_iter * dim);
        let mut batch_dist: Vec<f32> = Vec::with_capacity(max_neighbors_per_iter);

        loop {
            let active_count = states.iter().filter(|s| s.active).count();
            if active_count == 0 {
                break;
            }

            // Phase 1: Pop best candidate from each active query, collect unvisited neighbors
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
                    Some(std::cmp::Reverse((FloatOrd(c_dist), c_id))) => {
                        if state.result.len() >= l && c_dist > state.result[l - 1].0 {
                            state.active = false;
                            continue;
                        }

                        for &neighbor in self.get_neighbors(c_id) {
                            if neighbor >= self.header.num_vectors {
                                continue;
                            }
                            if !state.visited.insert(neighbor) {
                                continue;
                            }
                            let vec = self.get_vector(neighbor);
                            if vec.is_empty() {
                                continue;
                            }
                            all_neighbor_ids.push(neighbor);
                            all_query_map.push(qi as u32);
                        }
                    }
                }
            }

            if all_neighbor_ids.is_empty() {
                continue;
            }

            let total_n = all_neighbor_ids.len();

            // Phase 2: Compute distances — GPU if enough work, else CPU
            let use_gpu = total_n * dim >= crate::metal_ffi::MIN_GPU_WORK;

            if use_gpu {
                // Gather all candidate vectors
                batch_buf.clear();
                batch_buf.reserve(total_n * dim);
                for &id in &all_neighbor_ids {
                    batch_buf.extend_from_slice(self.get_vector(id));
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
                        Self::insert_result(
                            &mut state.result,
                            &mut state.candidates,
                            l,
                            batch_dist[i],
                            all_neighbor_ids[i],
                        );
                    }
                    continue;
                }
                // GPU failed — fall through to CPU
            }

            // CPU fallback: compute distances per-neighbor
            for i in 0..total_n {
                let qi = all_query_map[i] as usize;
                let neighbor = all_neighbor_ids[i];
                let vec = self.get_vector(neighbor);
                let dist = self.compute_distance(queries[qi], vec);
                let state = &mut states[qi];
                Self::insert_result(&mut state.result, &mut state.candidates, l, dist, neighbor);
            }
        }

        // Collect results
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

    /// Insert a distance result into the sorted result list and candidate heap.
    #[inline]
    fn insert_result(
        result: &mut Vec<(f32, u32)>,
        candidates: &mut BinaryHeap<std::cmp::Reverse<(FloatOrd, u32)>>,
        l: usize,
        dist: f32,
        neighbor: u32,
    ) {
        if result.len() < l || dist < result[result.len() - 1].0 {
            let pos = result
                .binary_search_by(|probe| {
                    probe
                        .0
                        .partial_cmp(&dist)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or_else(|e| e);
            result.insert(pos, (dist, neighbor));
            if result.len() > l {
                result.truncate(l);
            }
            candidates.push(std::cmp::Reverse((FloatOrd(dist), neighbor)));
        }
    }

    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        crate::distance::compute_distance(self.metric, a, b)
    }
}

/// Wrapper for f32 that implements Ord (for BinaryHeap).
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
