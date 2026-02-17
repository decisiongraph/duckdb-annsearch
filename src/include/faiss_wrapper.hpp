#pragma once

#ifdef FAISS_AVAILABLE

// DuckDB redefines make_unique to explicitely fail compilation.
// We need a helper to create unique_ptrs for FAISS objects.
#include <memory>

// Handle Objective-C++ conflict with 'nil' macro in FAISS headers
#ifdef __OBJC__
#pragma push_macro("nil")
#undef nil
#endif

// Include widely used FAISS headers
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/MetricType.h>
#include <faiss/impl/io.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>

#ifdef __OBJC__
#pragma pop_macro("nil")
#endif

namespace duckdb {

// Helper to bypass DuckDB's make_unique restriction for FAISS types
template <typename T, typename... Args>
std::unique_ptr<T> make_faiss_unique(Args &&...args) {
	return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

} // namespace duckdb

#endif // FAISS_AVAILABLE
