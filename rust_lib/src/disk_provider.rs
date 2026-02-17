//! Read-only mmap-backed DiskANN index with standalone greedy best-first search.

use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::io;
use std::path::Path;

use memmap2::Mmap;

use crate::file_format::{FileHeader, HEADER_SIZE, MAGIC, VERSION};
use crate::index_manager::Metric;

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
}

impl SearchContext {
    fn new() -> Self {
        Self {
            visited: hashbrown::HashSet::new(),
            candidates: BinaryHeap::new(),
            result: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.visited.clear();
        self.candidates.clear();
        self.result.clear();
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
    pub fn search(&self, query: &[f32], k: usize, l_search: usize) -> Vec<(u64, f32)> {
        let n = self.len();
        if n == 0 || k == 0 {
            return Vec::new();
        }

        let k = k.min(n);
        let l = l_search.max(k);

        SEARCH_CTX.with(|cell| {
            let mut ctx = cell.borrow_mut();
            ctx.clear();

            // Reserve capacity if needed (grows once, reused across queries)
            let needed = l * 2;
            let cap = ctx.visited.capacity();
            if cap < needed {
                ctx.visited.reserve(needed - cap);
            }

            // Seed with entry points
            for &ep in &self.entry_point_ids {
                if ctx.visited.insert(ep) {
                    let vec = self.get_vector(ep);
                    if vec.is_empty() {
                        continue;
                    }
                    let dist = self.compute_distance(query, vec);
                    ctx.candidates.push(std::cmp::Reverse((FloatOrd(dist), ep)));
                    ctx.result.push((dist, ep));
                }
            }

            ctx.result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            while let Some(std::cmp::Reverse((FloatOrd(c_dist), c_id))) = ctx.candidates.pop() {
                if ctx.result.len() >= l && c_dist > ctx.result[l - 1].0 {
                    break;
                }

                for &neighbor in self.get_neighbors(c_id) {
                    if neighbor >= self.header.num_vectors {
                        continue;
                    }
                    if !ctx.visited.insert(neighbor) {
                        continue;
                    }

                    let vec = self.get_vector(neighbor);
                    if vec.is_empty() {
                        continue;
                    }
                    let dist = self.compute_distance(query, vec);

                    if ctx.result.len() < l || dist < ctx.result[ctx.result.len() - 1].0 {
                        let pos = ctx
                            .result
                            .binary_search_by(|probe| {
                                probe
                                    .0
                                    .partial_cmp(&dist)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .unwrap_or_else(|e| e);
                        ctx.result.insert(pos, (dist, neighbor));
                        if ctx.result.len() > l {
                            ctx.result.truncate(l);
                        }

                        ctx.candidates
                            .push(std::cmp::Reverse((FloatOrd(dist), neighbor)));
                    }
                }
            }

            ctx.result
                .iter()
                .take(k)
                .map(|&(dist, id)| (id as u64, dist))
                .collect()
        })
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
