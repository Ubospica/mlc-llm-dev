/*!
 * Copyright (c) 2023 by Contributors
 * \file progress_bar.h
 * \brief A simple progress bar in C++.
 */
#ifndef MLC_LLM_SUPPORT_DYNAMIC_BITSET_H_
#define MLC_LLM_SUPPORT_DYNAMIC_BITSET_H_

#include <cstdint>
#include <vector>

namespace mlc {
namespace llm {

class DynamicBitSet {
 public:
  explicit DynamicBitSet(int size = 0)
      : size_(size), internal_size_((size + 31) / 32), data_(internal_size_, 0) {}

  bool operator[](int index) const { return (data_[index / 32] >> (index % 32)) & 1; }

  template <bool value = false>
  void Reset(int size = -1) {
    if (size != -1) {
      size_ = size;
      internal_size_ = (size + 31) / 32;
    }
    if constexpr (value) {
      data_.assign(internal_size_, 0xFFFFFFFF);
    } else {
      data_.assign(internal_size_, 0);
    }
  }

  void Set(int index, bool value = true) {
    DCHECK(index < size_);
    if (value) {
      data_[index / 32] |= 1 << (index % 32);
    } else {
      data_[index / 32] &= ~(1 << (index % 32));
    }
  }

  template <bool value = true>
  void SetConst(int index) {
    DCHECK(index < size_);
    if constexpr (value) {
      data_[index / 32] |= 1 << (index % 32);
    } else {
      data_[index / 32] &= ~(1 << (index % 32));
    }
  }

  int Size() const { return size_; }

  int InternalSize() const { return internal_size_; }

  //   class Iterator {
  //    public:
  //     Iterator(const DynamicBitSet* bitset, int index) : bitset_(bitset), index_(index) {}

  //     bool operator!=(const Iterator& other) const {
  //       return bitset_ != other.bitset_ || index_ != other.index_;
  //     }

  //     Iterator& operator++() {
  //       DCHECK(index_ + 1 < bitset_->Size());
  //       index_++;
  //       return *this;
  //     }

  //     int operator*() const { return (*bitset_)[index_]; }

  //    private:
  //     const DynamicBitSet* bitset_;
  //     int index_;
  //   };

  //   Iterator begin() const { return Iterator(this, 0); }
  //   Iterator end() const { return Iterator(this, size_); }

 private:
  int size_;
  int internal_size_;
  std::vector<int32_t> data_;
};

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SUPPORT_DYNAMIC_BITSET_H_