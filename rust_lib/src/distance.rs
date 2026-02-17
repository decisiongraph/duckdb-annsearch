//! SIMD-accelerated distance functions using diskann-vector.
//!
//! Replaces scalar l2/inner-product loops with platform-optimized kernels
//! (AVX2/AVX-512 on x86_64, auto-vectorized on aarch64).

use crate::index_manager::Metric;
use diskann_vector::distance::implementations::{InnerProduct, SquaredL2};
use diskann_vector::PureDistanceFunction;

/// Compute distance between two vectors using the given metric.
///
/// For L2: returns squared Euclidean distance (sum of squared differences).
/// For InnerProduct: returns negated dot product (-dot(a,b)), so lower = more similar.
#[inline]
pub fn compute_distance(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        Metric::L2 => {
            <SquaredL2 as PureDistanceFunction<&[f32], &[f32], f32>>::evaluate(a, b)
        }
        Metric::InnerProduct => {
            // PureDistanceFunction returns raw dot product; negate for distance semantics
            -<InnerProduct as PureDistanceFunction<&[f32], &[f32], f32>>::evaluate(a, b)
        }
    }
}
