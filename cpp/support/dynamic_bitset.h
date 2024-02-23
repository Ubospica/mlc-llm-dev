/*!
 * Copyright (c) 2023 by Contributors
 * \file dynamic_bitset.h
 * \brief The header for the dynamic bitset container.
 */
#ifndef MLC_LLM_SUPPORT_DYNAMIC_BITSET_H_
#define MLC_LLM_SUPPORT_DYNAMIC_BITSET_H_

#include <cstdint>
#include <vector>

namespace mlc {
namespace llm {

class BitsetManager {
 public:
  BitsetManager(uint32_t* data, int buffer_size) : data_(data), buffer_size_(buffer_size) {}

  static int GetBitsetSize(int size) { return (size + 31) / 32; }

  bool operator[](int index) const {
    DCHECK(index >= 0 && index / 32 < buffer_size_);
    return (data_[index / 32] >> (index % 32)) & 1;
  }

  void Set(int index, bool value) {
    DCHECK(index >= 0 && index / 32 < buffer_size_);
    if (value) {
      data_[index / 32] |= 1 << (index % 32);
    } else {
      data_[index / 32] &= ~(1 << (index % 32));
    }
  }

  void Reset(int size, bool value) {
    DCHECK(buffer_size_ >= GetBitsetSize(size));
    std::memset(data_, value ? 0xFF : 0, GetBitsetSize(size) * sizeof(uint32_t));
  }

 private:
  uint32_t* const data_;
  const int buffer_size_;
};

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SUPPORT_DYNAMIC_BITSET_H_
