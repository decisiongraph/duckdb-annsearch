#pragma once

/// Shared between Metal shaders and C++ host code.
#ifdef __METAL_VERSION__
using uint32 = uint;
#else
#include <cstdint>
using uint32 = uint32_t;
#endif

struct DiskannDistParams {
	uint32 n;   // number of candidate vectors
	uint32 dim; // vector dimension
};
