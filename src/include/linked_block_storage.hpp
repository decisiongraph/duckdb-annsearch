#pragma once

#include "duckdb/execution/index/fixed_size_allocator.hpp"
#include "duckdb/execution/index/index_pointer.hpp"

namespace duckdb {

struct LinkedBlock {
	static constexpr idx_t BLOCK_SIZE = Storage::DEFAULT_BLOCK_SIZE - sizeof(validity_t);
	static constexpr idx_t BLOCK_DATA_SIZE = BLOCK_SIZE - sizeof(IndexPointer);

	IndexPointer next_block;
	char data[BLOCK_DATA_SIZE] = {0};
};

class LinkedBlockWriter {
public:
	LinkedBlockWriter(FixedSizeAllocator &allocator, IndexPointer root)
	    : allocator_(allocator), root_(root), current_(root), pos_(0) {
	}

	void Reset() {
		current_ = root_;
		pos_ = 0;
	}

	idx_t Write(const uint8_t *buffer, idx_t length) {
		idx_t written = 0;
		while (written < length) {
			auto block = allocator_.Get<LinkedBlock>(current_, true);
			auto to_write = MinValue<idx_t>(length - written, LinkedBlock::BLOCK_DATA_SIZE - pos_);
			memcpy(block->data + pos_, buffer + written, to_write);
			written += to_write;
			pos_ += to_write;
			if (pos_ == LinkedBlock::BLOCK_DATA_SIZE) {
				pos_ = 0;
				if (block->next_block.Get() == 0) {
					block->next_block = allocator_.New();
				}
				current_ = block->next_block;
			}
		}
		return written;
	}

private:
	FixedSizeAllocator &allocator_;
	IndexPointer root_;
	IndexPointer current_;
	idx_t pos_;
};

class LinkedBlockReader {
public:
	LinkedBlockReader(FixedSizeAllocator &allocator, IndexPointer root)
	    : allocator_(allocator), current_(root), pos_(0), exhausted_(false) {
	}

	idx_t Read(uint8_t *buffer, idx_t length) {
		idx_t total_read = 0;
		while (total_read < length && !exhausted_) {
			auto block = allocator_.Get<LinkedBlock>(current_, false);
			auto to_read = MinValue<idx_t>(length - total_read, LinkedBlock::BLOCK_DATA_SIZE - pos_);
			memcpy(buffer + total_read, block->data + pos_, to_read);
			total_read += to_read;
			pos_ += to_read;
			if (pos_ == LinkedBlock::BLOCK_DATA_SIZE) {
				pos_ = 0;
				if (block->next_block.Get() == 0) {
					exhausted_ = true;
				} else {
					current_ = block->next_block;
				}
			}
		}
		return total_read;
	}

private:
	FixedSizeAllocator &allocator_;
	IndexPointer current_;
	idx_t pos_;
	bool exhausted_;
};

} // namespace duckdb
