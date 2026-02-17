#pragma once

#import <Metal/Metal.h>
#include <faiss/impl/FaissAssert.h>
#include <array>
#include <cstddef>

namespace faiss_metal {

/// Lightweight tensor wrapping an MTLBuffer.
/// Uses StorageModeShared for zero-copy CPU+GPU access on Apple Silicon.
template <typename T, int Dim> class MetalTensor {
  public:
    MetalTensor() : buffer_(nil), numElements_(0), isNoCopy_(false) {
        sizes_.fill(0);
        strides_.fill(0);
    }

    /// Allocate a new buffer on the given device
    MetalTensor(id<MTLDevice> device, std::array<size_t, Dim> sizes) : sizes_(sizes), isNoCopy_(false) {
        computeStrides();
        numElements_ = computeNumElements();
        size_t bytes = numElements_ * sizeof(T);
        buffer_ = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        FAISS_THROW_IF_NOT_MSG(buffer_, "Failed to allocate MTLBuffer");
    }

    /// Wrap existing CPU pointer with zero-copy MTLBuffer (newBufferWithBytesNoCopy).
    /// If the pointer is page-aligned, uses true zero-copy (isNoCopy() == true).
    /// Otherwise falls back to allocate + copy (isNoCopy() == false).
    MetalTensor(id<MTLDevice> device, T *data, std::array<size_t, Dim> sizes) : sizes_(sizes), isNoCopy_(false) {
        computeStrides();
        numElements_ = computeNumElements();
        size_t bytes = numElements_ * sizeof(T);

        // Page-align check: newBufferWithBytesNoCopy requires page-aligned pointer
        // If not aligned, fall back to copy
        if (reinterpret_cast<uintptr_t>(data) % getpagesize() == 0) {
            buffer_ = [device newBufferWithBytesNoCopy:data
                                                length:bytes
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
            if (buffer_)
                isNoCopy_ = true;
        }
        if (!buffer_) {
            // Fall back: allocate and copy
            buffer_ = [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
        }
        FAISS_THROW_IF_NOT_MSG(buffer_, "Failed to create MTLBuffer from pointer");
    }

    /// CPU-accessible data pointer (also GPU-accessible via unified memory)
    T *data() {
        return buffer_ ? static_cast<T *>([buffer_ contents]) : nullptr;
    }
    const T *data() const {
        return buffer_ ? static_cast<const T *>([buffer_ contents]) : nullptr;
    }

    id<MTLBuffer> buffer() const {
        return buffer_;
    }
    size_t size(int dim) const {
        return sizes_[dim];
    }
    size_t stride(int dim) const {
        return strides_[dim];
    }
    size_t numElements() const {
        return numElements_;
    }
    size_t sizeInBytes() const {
        return numElements_ * sizeof(T);
    }

    bool empty() const {
        return numElements_ == 0;
    }

    /// True if buffer wraps external pointer via NoCopy (caller owns memory).
    bool isNoCopy() const {
        return isNoCopy_;
    }

    /// Resize (reallocates if needed, does NOT preserve data)
    void resize(id<MTLDevice> device, std::array<size_t, Dim> newSizes) {
        sizes_ = newSizes;
        computeStrides();
        size_t newNumElements = computeNumElements();
        if (newNumElements != numElements_ || !buffer_) {
            numElements_ = newNumElements;
            isNoCopy_ = false;
            size_t bytes = numElements_ * sizeof(T);
            buffer_ = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            FAISS_THROW_IF_NOT_MSG(buffer_, "Failed to resize MTLBuffer");
        }
    }

  private:
    void computeStrides() {
        if constexpr (Dim > 0) {
            strides_[Dim - 1] = 1;
            for (int i = Dim - 2; i >= 0; --i) {
                strides_[i] = strides_[i + 1] * sizes_[i + 1];
            }
        }
    }

    size_t computeNumElements() const {
        size_t n = 1;
        for (int i = 0; i < Dim; ++i) {
            n *= sizes_[i];
        }
        return n;
    }

    id<MTLBuffer> buffer_;
    std::array<size_t, Dim> sizes_;
    std::array<size_t, Dim> strides_;
    size_t numElements_;
    bool isNoCopy_;
};

} // namespace faiss_metal
