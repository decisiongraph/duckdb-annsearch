# faiss-metal

Metal GPU backend for [FAISS](https://github.com/facebookresearch/faiss) vector similarity search on Apple Silicon.

FAISS only supports NVIDIA CUDA for GPU acceleration. faiss-metal fills this gap for macOS by implementing brute-force index operations on Apple Silicon GPUs via Metal compute shaders. It subclasses `faiss::Index` directly so it works as a drop-in replacement.

**Key design advantage:** Apple Silicon's unified memory means zero CPU-GPU transfer overhead. Vectors stored in MTLBuffers are accessible by both CPU and GPU without copies.

## Features

- `MetalIndexFlat` with L2 and inner product metrics
- **FP16 vector storage** -- halves memory bandwidth, +22-34% throughput on bandwidth-bound workloads
- Automatic code path selection per Apple Silicon generation (M1/M2/M3/M4)
- Fused L2 GEMM kernel (norms + GEMM in one dispatch on M2+)
- GPU-native top-k selection for all k values (no CPU fallback)
- CPU-to-Metal and Metal-to-CPU index conversion
- Scratch buffer reuse with Private storage mode for GPU-only intermediaries
- Page-aligned allocation helper for zero-copy buffer wrapping

## Hardware support

| Generation | Metal Family | GEMM path | Block select threads |
|:---|:---|:---|:---|
| M1 | Apple7 | MPS MatrixMultiplication (FP32) | 256 |
| M2 | Apple8 | simdgroup_matrix GEMM (FP16, 2x throughput) | 256 |
| M3 | Apple9 | simdgroup_matrix GEMM (FP16) + dynamic cache | 512 |
| M4 | Apple9+ | simdgroup_matrix GEMM (FP16) + dynamic cache | 512 |

Top-k routing: `k <= 32` uses warp_select (SIMD-group), `k > 32` uses block_select (threadgroup with parallel merge). Maximum k = threads * 8 (2048 on M1/M2, 4096 on M3+).

## Usage

```cpp
#include <faiss-metal/faiss_metal.h>

auto res = std::make_shared<faiss_metal::StandardMetalResources>();
faiss_metal::MetalIndexFlat index(res, 128, faiss::METRIC_L2);

index.add(n, vectors);

std::vector<float> distances(nq * k);
std::vector<faiss::idx_t> labels(nq * k);
index.search(nq, queries, k, distances.data(), labels.data());
```

### FP16 vector storage

Store vectors as half-precision to halve memory bandwidth. Queries remain FP32. Slight precision loss (~3 decimal digits) but top-1 results still match FP32.

```cpp
// Enable FP16 storage (4th argument)
faiss_metal::MetalIndexFlat index(res, 128, faiss::METRIC_L2, /*useFloat16Storage=*/true);

index.add(n, vectors);           // float* input, converted to half internally
index.search(nq, queries, k, distances.data(), labels.data());  // queries stay FP32

// Reconstruct converts half → float
std::vector<float> recons(d);
index.reconstruct(0, recons.data());

// Check storage mode
index.isFloat16Storage();  // true
index.getVectorsData();    // nullptr (use reconstruct instead)
```

### Convert from CPU FAISS index

```cpp
faiss::IndexFlatL2 cpu_index(128);
cpu_index.add(n, vectors);

auto res = std::make_shared<faiss_metal::StandardMetalResources>();
auto metal_index = faiss_metal::index_cpu_to_metal(res, &cpu_index);
metal_index->search(nq, queries, k, distances, labels);

// Convert back (handles both FP32 and FP16 storage)
auto cpu_index2 = faiss_metal::index_metal_to_cpu(metal_index.get());
```

### Zero-copy buffer wrapping

For maximum performance, allocate query/vector data with page-aligned memory to hit the zero-copy `newBufferWithBytesNoCopy` fast path in `MetalTensor`:

```cpp
#include <faiss-metal/MetalResources.h>

// Page-aligned allocation (16KB on Apple Silicon)
float* data = (float*)faiss_metal::alloc_aligned(n * d * sizeof(float));
// ... fill data ...
// When wrapping with MetalTensor, isNoCopy() == true (no memcpy)

faiss_metal::free_aligned(data);
```

Standard `malloc`/`new`/`std::vector` allocations are rarely page-aligned and will fall back to allocate+copy.

## Prerequisites

- **macOS 14+** on Apple Silicon (M1/M2/M3/M4)
- **Xcode** (full install, not just Command Line Tools) -- required for Metal compiler and MPS framework
- **Metal Toolchain** -- must be downloaded separately:
  ```bash
  xcodebuild -downloadComponent MetalToolchain
  ```
- **CMake 3.24+**
- **FAISS** -- installed as a library (e.g. via Homebrew, nix, or from source)
- **OpenMP** -- required by FAISS (e.g. `libomp` from Homebrew or nix)

### Using devenv (recommended)

The project includes a [devenv](https://devenv.sh) configuration that provides cmake, faiss, and openmp via nix. The Metal compiler and frameworks come from your Xcode install.

```bash
devenv shell
build          # configure & build
test           # build & run tests
bench          # build & run benchmarks
```

> **Note:** The devenv scripts unset `SDKROOT` and use `/usr/bin/clang` because nix's apple-sdk lacks Metal Toolchain and MPS headers. The system compiler + real macOS SDK is required for Metal development.

### Manual build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Run tests

```bash
cmake -B build -DFAISS_METAL_BUILD_TESTS=ON
cmake --build build -j
cd build && ctest --output-on-failure
```

### Run benchmarks

```bash
./build/bench_metal_flat
```

Benchmarks compare Metal (FP32 and FP16 storage) vs CPU FAISS across dataset sizes (10K-100K), dimensions (32-1536), and k values. See [BENCHMARK.md](../BENCHMARK.md) for detailed results and analysis.

## Architecture

### GPU pipeline

A single `search()` call encodes all work into one `MTLCommandBuffer`:

```
MTLCommandBuffer
  ├─ [1] L2 norm: compute ||q||^2 for queries
  ├─ [2] Distance: GEMM + fused L2 norms (or MPS GEMM + broadcast_sum on M1)
  └─ [3] Top-k: warp_select (k<=32) or block_select (k>32)
commit + waitUntilCompleted
```

No CPU round-trips between stages. GPU executes all dispatches back-to-back.

### Storage modes

Intermediate buffers (`scratchDistBuf`, `scratchQueryNormBuf`) use `MTLResourceStorageModePrivate` -- GPU-only, no CPU cache snooping. Final output buffers use `StorageModeShared` for CPU readback.

### FP16 storage pipeline

When `useFloat16Storage=true`:
- `add()` converts float32 input to float16, stores in half-precision MTLBuffer
- L2 norms computed from the float32 input (before conversion) for accuracy
- GEMM shaders read vectors as `device const half*` directly -- no float-to-half conversion in the shader, halving global memory bandwidth
- MPS path uses `MPSDataTypeFloat16` matrix descriptors
- `reconstruct()` converts half back to float

## Project structure

```
include/faiss-metal/           Public C++ headers
  MetalIndexFlat.h             Flat index (subclasses faiss::Index)
  MetalResources.h             Abstract resource manager + alloc_aligned()
  StandardMetalResources.h     Default resource manager
  MetalDeviceCapabilities.h    Runtime Apple Silicon detection
  faiss_metal.h                Convenience header (includes all)

src/                           Implementation (ObjC++)
  MetalDistance.h/.mm          L2/IP distance via MPS or simdgroup GEMM
  MetalL2Norm.h/.mm            L2 norm computation
  MetalSelect.h/.mm            Top-k selection dispatch
  MetalTensor.h                Tensor wrapping MTLBuffer
  MetalContext.h/.mm           Metal device/library init
  MetalIndexFlat.mm            Index add/search/reset
  StandardMetalResources.mm    Default resources

shaders/                       Metal compute shaders
  simdgroup_gemm.metal         Custom FP16 GEMM (M2+), fused L2, F16 storage variants
  l2_norm.metal                FP32 L2 norm
  l2_norm_f16.metal            FP16 L2 norm + direct L2 distance + F16 storage variant
  broadcast_sum.metal          L2 distance norm broadcast (M1 path)
  warp_select.metal            SIMD-group top-k (k <= 32)
  block_select.metal           Threadgroup top-k with parallel merge (k > 32)
  GemmParams.h                 Shared struct (Metal + C++)

tests/
  test_metal_flat.mm           Index correctness vs CPU FAISS (FP32 + FP16)
  test_metal_distance.mm       Distance kernel accuracy
  bench_metal_flat.mm          Performance benchmarks (FP32 + FP16)
```

## Numerical precision

The FP16 GEMM path (M2+) trades some precision for 2x throughput. Distances have ~1% relative error vs FP32, but top-1 results match exactly. The MPS path (M1 or forced) uses FP32 throughout.

FP16 vector storage adds additional quantization at storage time (~3 decimal digits of precision per element). Top-1 search results still match FP32 in practice for typical embedding vectors.

## Limitations

- macOS only (Apple Silicon required)
- Flat index only (IVF planned)
- Maximum 2^31-1 vectors (Metal shaders use int32 indices)
- Maximum k = 2048 (M1/M2) or 4096 (M3+)
- FP16 storage: `getVectorsData()` returns nullptr; use `reconstruct()` for individual vectors
- Single-query latency higher than CPU due to Metal dispatch overhead (~200-500us)
- No Python bindings yet

## License

MIT
