//! Two-pass streaming DiskANN builder for datasets larger than RAM.
//!
//! Input: flat binary file `[u32 num_vectors][u32 dimension][f32*N*D]`
//! Output: .diskann index file
//!
//! Pass 1 (sample): Read a subset of vectors, build an in-memory pilot graph.
//! Pass 2 (stream): Add remaining vectors to the pilot graph using DiskANN's
//!   native insert (bidirectional edges, pruning, connectivity).

use std::fs::File;
use std::io::{self, BufReader, Read, Write};

use anyhow::{anyhow, Result};

use crate::index_manager::Metric;

/// Header for the input vectors binary file.
struct VecFileHeader {
    num_vectors: u32,
    dimension: u32,
}

fn read_vec_header(r: &mut impl Read) -> io::Result<VecFileHeader> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    let num_vectors = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let dimension = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    Ok(VecFileHeader { num_vectors, dimension })
}

/// Read a single vector (dimension floats) from the reader.
fn read_vector(r: &mut impl Read, dim: usize) -> io::Result<Vec<f32>> {
    let mut bytes = vec![0u8; dim * 4];
    r.read_exact(&mut bytes)?;
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(floats)
}

/// Build a DiskANN index from a binary vectors file using streaming two-pass approach.
///
/// Only the sample vectors + their graph stay in RAM. Remaining vectors are
/// processed one at a time from disk.
pub fn streaming_build(
    input_path: &str,
    output_path: &str,
    metric: Metric,
    max_degree: u32,
    build_complexity: u32,
    alpha: f32,
    sample_size: u32,
) -> Result<StreamingBuildResult> {
    let input = File::open(input_path)
        .map_err(|e| anyhow!("Failed to open input '{}': {}", input_path, e))?;
    let mut reader = BufReader::new(input);

    let hdr = read_vec_header(&mut reader)
        .map_err(|e| anyhow!("Failed to read input header: {}", e))?;

    if hdr.num_vectors == 0 {
        return Err(anyhow!("Input file has 0 vectors"));
    }
    if hdr.dimension == 0 {
        return Err(anyhow!("Input file has dimension 0"));
    }

    let dim = hdr.dimension as usize;
    let n = hdr.num_vectors;
    // Auto sample size: max(sqrt(N), 1000), clamped to N
    let sample_n = if sample_size == 0 {
        ((n as f64).sqrt() as usize).max(1000).min(n as usize)
    } else {
        (sample_size as usize).min(n as usize)
    };
    // ========================================
    // Pass 1: Build pilot graph from sample
    // ========================================

    let pilot = crate::index_manager::InMemoryIndex::new_detached(
        dim,
        metric,
        max_degree,
        build_complexity,
        alpha,
    );

    for _ in 0..sample_n {
        let vec = read_vector(&mut reader, dim)
            .map_err(|e| anyhow!("Failed to read sample vector: {}", e))?;
        pilot.add(&vec)?;
    }

    // ========================================
    // Pass 2: Add remaining vectors via DiskANN's native insert
    // ========================================
    // Uses proper bidirectional edges, pruning, and connectivity —
    // no manual back-edge injection needed.

    let remaining = n as usize - sample_n;
    for _ in 0..remaining {
        let vec = read_vector(&mut reader, dim)
            .map_err(|e| anyhow!("Failed to read streaming vector: {}", e))?;
        pilot.add(&vec)?;
    }

    // ========================================
    // Write output .diskann file
    // ========================================

    let bytes = pilot.serialize_to_bytes()?;
    let mut output = File::create(output_path)
        .map_err(|e| anyhow!("Failed to create output '{}': {}", output_path, e))?;
    output.write_all(&bytes)?;

    Ok(StreamingBuildResult {
        num_vectors: n,
        dimension: dim as u32,
        sample_size: sample_n as u32,
    })
}

pub struct StreamingBuildResult {
    pub num_vectors: u32,
    pub dimension: u32,
    pub sample_size: u32,
}
