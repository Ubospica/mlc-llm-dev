/*!
 * Copyright (c) 2023 by Contributors
 * \file support/dynamic_bitset.h
 * \brief The header for utilities used in grammar-guided generation.
 */
#ifndef MLC_LLM_SUPPORT_DYNAMIC_BITSET_H_
#define MLC_LLM_SUPPORT_DYNAMIC_BITSET_H_

#include <tvm/runtime/logging.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace mlc {
namespace llm {

/*! \brief A bitset with runtime specified length. It manages memory internally or the memory
 * provided externally with enough size. */
class DynamicBitset {
 public:
  static int CalculateBufferSize(int element_size) { return (element_size + 31) / 32; }

  DynamicBitset() : size_(0), buffer_size_(0), data_(nullptr), is_internal_(true) {}

  DynamicBitset(int size, uint32_t* data = nullptr)
      : size_(size), buffer_size_(CalculateBufferSize(size)) {
    if (data == nullptr) {
      internal_buffer_.resize(buffer_size_, 0);
      data_ = internal_buffer_.data();
      is_internal_ = true;
    } else {
      data_ = data;
      is_internal_ = false;
    }
  }

  DynamicBitset& operator=(const DynamicBitset& other) {
    DCHECK(is_internal_ || size_ >= other.size_) << "Expanding bitset size is not allowed when the "
                                                    "memory of the bitset is externally managed";
    size_ = other.size_;
    buffer_size_ = other.buffer_size_;
    if (is_internal_) {
      internal_buffer_.reserve(buffer_size_);
      data_ = internal_buffer_.data();
    }
    if (data_ != other.data_) {
      std::memcpy(data_, other.data_, buffer_size_ * sizeof(uint32_t));
    }
    return *this;
  }

  DynamicBitset& operator=(DynamicBitset&& other) {
    size_ = other.size_;
    buffer_size_ = other.buffer_size_;
    is_internal_ = other.is_internal_;
    if (is_internal_) {
      internal_buffer_ = std::move(other.internal_buffer_);
      data_ = internal_buffer_.data();
    } else {
      data_ = other.data_;
    }
    return *this;
  }

  bool operator[](int index) const {
    DCHECK(data_ && index >= 0 && index < size_);
    return (data_[index / 32] >> (index % 32)) & 1;
  }

  int Size() const { return size_; }

  void Set() {
    DCHECK(data_);
    std::memset(data_, 0xFF, buffer_size_ * sizeof(uint32_t));
  }

  void Set(int index, bool value = true) {
    DCHECK(data_ && index >= 0 && index < size_);
    if (value) {
      data_[index / 32] |= 1 << (index % 32);
    } else {
      data_[index / 32] &= ~(1 << (index % 32));
    }
  }

  void Reset() {
    DCHECK(data_);
    std::memset(data_, 0, buffer_size_ * sizeof(uint32_t));
  }

  void Reset(int index) { Set(index, false); }

  DynamicBitset& operator|=(const DynamicBitset& other) {
    DCHECK(buffer_size_ <= other.buffer_size_);
    for (int i = 0; i < buffer_size_; ++i) {
      data_[i] |= other.data_[i];
    }
    return *this;
  }

 private:
  int size_;
  int buffer_size_;
  uint32_t* data_;
  std::vector<uint32_t> internal_buffer_;
  bool is_internal_;
};

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SUPPORT_DYNAMIC_BITSET_H_
