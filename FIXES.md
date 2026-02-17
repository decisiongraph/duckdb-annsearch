# Build Fixes for FAISS + Metal GPU Support

Fixes needed to compile the extension with FAISS 1.13.2 and faiss-metal on macOS (Apple Silicon).

## CMakeLists.txt

### 1. OpenMP must be found before FAISS

FAISS's cmake targets reference `OpenMP::OpenMP_CXX` in their interface link libraries. If `find_package(OpenMP)` hasn't run in the parent scope, cmake fails during generation with "target was not found".

```cmake
# Before find_package(faiss)
find_package(OpenMP QUIET)
```

### 2. OpenMP linked to extension targets

FAISS headers (`<faiss/impl/HNSW.h>`) include `<omp.h>`. Without linking `OpenMP::OpenMP_CXX`, the compiler can't find `omp.h`.

```cmake
if(OpenMP_CXX_FOUND)
    target_link_libraries(${EXTENSION_NAME} OpenMP::OpenMP_CXX)
endif()
```

### 3. faiss-metal subdirectory build fixes

When built via `add_subdirectory()`, faiss-metal's `CMAKE_SOURCE_DIR` resolves to the DuckDB root (not faiss-metal's own directory). Fixed by:

- Changing `CMAKE_SOURCE_DIR` to `CMAKE_CURRENT_SOURCE_DIR` in faiss-metal's CMakeLists.txt (committed to faiss-metal repo)
- Disabling faiss-metal tests: `set(FAISS_METAL_BUILD_TESTS OFF CACHE BOOL "" FORCE)`
- Fixing include dirs for install export with `$<BUILD_INTERFACE:...>` generator expression
- Adding `faiss_metal` to the install export set alongside the extension target

## src/faiss_index.cpp

### 4. DuckDB intercepts `std::make_unique`

DuckDB redefines `std::make_unique` to a `static_assert("Use make_uniq instead!")`. Since we can't use `make_uniq` for FAISS types (they're external), use explicit `std::unique_ptr<T>(new T(...))` instead.

```cpp
// Bad: triggers static_assert
std::make_unique<faiss::IndexFlat>(dim, metric);

// Good: bypasses DuckDB's interception
std::unique_ptr<faiss::Index>(new faiss::IndexFlat(dim, metric));
```

### 5. FAISS VectorIOReader API changed in 1.13.2

`VectorIOReader` no longer has a `size` member or pointer-based `data`. It owns a `std::vector<uint8_t> data` and tracks read position internally via `rp`.

```cpp
// Old (broken)
faiss_reader.data = faiss_data.data();
faiss_reader.size = faiss_data.size();

// New (1.13.2)
faiss_reader.data = std::move(faiss_data);
```

### 6. `gpu` option parsed as string, not bool

DuckDB `WITH (gpu=false)` passes option values as strings. `GetValue<bool>()` on the string `"false"` throws. Use explicit cast:

```cpp
gpu_ = BooleanValue::Get(kv.second.DefaultCastAs(LogicalType::BOOLEAN));
```

## src/gpu_backend_metal.mm

### 7. Obj-C++ `nil` keyword conflicts with FAISS headers

FAISS's `InvertedLists.h` uses `nil` as a C++ parameter name (`int nil`). In Objective-C++, `nil` is a keyword. Fix by temporarily undefining it around FAISS includes:

```objc
#pragma push_macro("nil")
#undef nil
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#pragma pop_macro("nil")
```

### 8. MetalDeviceCapabilities API mismatch

The `metalFamily` member was replaced with `generation` enum and `deviceName` string:

```cpp
// Old
return "Metal GPU (family=" + std::to_string(caps.metalFamily) + ")";

// New
return "Metal GPU (" + caps.deviceName + ")";
```

## src/include/faiss_index.hpp

### 9. Missing ExtensionLoader include

`RegisterFaissIndexScanFunction` takes `ExtensionLoader&` but the header didn't include it:

```cpp
#include "duckdb/main/extension/extension_loader.hpp"
```

## Build Command

```sh
# Inside devenv shell (provides faiss, openmp, rust, etc.)
devenv shell -- make release GEN=ninja
```

If not using devenv, ensure `faiss`, `OpenMP`, and `faiss-metal` (sibling directory) are findable by cmake.
