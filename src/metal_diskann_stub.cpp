// Stub implementations for diskann_metal_* functions on non-Metal platforms.
// When FAISS_METAL_ENABLED is defined, the real implementations in
// metal_diskann_bridge.mm take precedence.

#ifndef FAISS_METAL_ENABLED

#include "include/metal_diskann_bridge.h"

extern "C" int diskann_metal_available(void) {
	return 0;
}

extern "C" int diskann_metal_batch_distances(const float * /*query*/, const float * /*candidates*/, int /*n*/,
                                             int /*dim*/, int /*metric*/, float * /*out_distances*/
) {
	return -1;
}

extern "C" int diskann_metal_multi_batch_distances(const float * /*queries*/, const float * /*candidates*/,
                                                   const unsigned int * /*query_map*/, int /*total_n*/, int /*nq*/,
                                                   int /*dim*/, int /*metric*/, float * /*out_distances*/
) {
	return -1;
}

#endif // !FAISS_METAL_ENABLED
