//! FFI bridge to Metal GPU batch distance computation.
//!
//! The actual Metal implementation lives in C++/ObjC++ (metal_diskann_bridge.mm).
//! On non-macOS platforms, stub functions return "not available".
//! Symbols are resolved at link time when the Rust static lib is linked
//! with the C++ extension.

use std::sync::atomic::{AtomicI32, Ordering};

extern "C" {
    fn diskann_metal_available() -> i32;
    fn diskann_metal_batch_distances(
        query: *const f32,
        candidates: *const f32,
        n: i32,
        dim: i32,
        metric: i32,
        out_distances: *mut f32,
    ) -> i32;
    fn diskann_metal_multi_batch_distances(
        queries: *const f32,
        candidates: *const f32,
        query_map: *const u32,
        total_n: i32,
        nq: i32,
        dim: i32,
        metric: i32,
        out_distances: *mut f32,
    ) -> i32;
}

/// Cached Metal availability: -1=unchecked, 0=unavailable, 1=available
static METAL_STATUS: AtomicI32 = AtomicI32::new(-1);

/// Minimum n*dim product to justify GPU dispatch over CPU SIMD.
/// With buffer pool reuse + threadgroup query preload, dispatch overhead is
/// ~100-200us (down from ~450us with per-call buffer allocation).
/// CPU NEON SIMD processes ~1 float-op/ns. Break-even ~100-200K elements.
/// Per-iteration DiskANN search (64-128 neighbors) still rarely reaches this;
/// multi-query batching (Phase 1) aggregates enough work to trigger GPU.
pub const MIN_GPU_WORK: usize = 131072;

/// Lower threshold for one-shot batch distance (no iterative overhead).
/// Used by vector_distances() where a single GPU dispatch computes all distances.
/// At 768-dim: fires at ~64 candidates (49152/768=64).
pub const MIN_GPU_WORK_ONESHOT: usize = 49152;

/// Check if Metal GPU acceleration is available (cached after first call).
pub fn is_metal_available() -> bool {
    let status = METAL_STATUS.load(Ordering::Relaxed);
    if status >= 0 {
        return status == 1;
    }
    let avail = unsafe { diskann_metal_available() };
    METAL_STATUS.store(avail, Ordering::Relaxed);
    avail == 1
}

/// Compute batch distances using Metal GPU (single query vs N candidates).
///
/// `candidates` must be `n * dim` contiguous floats.
/// `metric`: 0=L2, 1=InnerProduct.
/// `out` must have length >= n.
///
/// Returns true on success. Returns false if Metal is unavailable,
/// the batch is too small, or the GPU dispatch fails.
pub fn metal_batch_distances(
    query: &[f32],
    candidates: &[f32],
    n: usize,
    dim: usize,
    metric: u8,
    out: &mut [f32],
) -> bool {
    if n == 0 || dim == 0 {
        return true; // nothing to compute
    }
    if n * dim < MIN_GPU_WORK || !is_metal_available() {
        return false;
    }
    debug_assert_eq!(candidates.len(), n * dim);
    debug_assert!(out.len() >= n);

    let ret = unsafe {
        diskann_metal_batch_distances(
            query.as_ptr(),
            candidates.as_ptr(),
            n as i32,
            dim as i32,
            metric as i32,
            out.as_mut_ptr(),
        )
    };
    ret == 0
}

/// Multi-query batch distances using Metal GPU.
///
/// `queries`: nq * dim contiguous floats (all query vectors).
/// `candidates`: total_n * dim contiguous floats (all candidate vectors).
/// `query_map`: total_n u32s — query_map[i] = query index for candidate i.
/// `metric`: 0=L2, 1=InnerProduct.
/// `out`: total_n floats output.
///
/// Returns true on success. Returns false if Metal unavailable or dispatch fails.
/// Does NOT check MIN_GPU_WORK — caller should check before calling.
pub fn metal_multi_batch_distances(
    queries: &[f32],
    candidates: &[f32],
    query_map: &[u32],
    total_n: usize,
    nq: usize,
    dim: usize,
    metric: u8,
    out: &mut [f32],
) -> bool {
    if total_n == 0 || nq == 0 || dim == 0 {
        return true;
    }
    if !is_metal_available() {
        return false;
    }
    debug_assert_eq!(queries.len(), nq * dim);
    debug_assert_eq!(candidates.len(), total_n * dim);
    debug_assert_eq!(query_map.len(), total_n);
    debug_assert!(out.len() >= total_n);

    let ret = unsafe {
        diskann_metal_multi_batch_distances(
            queries.as_ptr(),
            candidates.as_ptr(),
            query_map.as_ptr(),
            total_n as i32,
            nq as i32,
            dim as i32,
            metric as i32,
            out.as_mut_ptr(),
        )
    };
    ret == 0
}
