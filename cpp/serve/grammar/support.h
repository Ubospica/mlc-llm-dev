/*!
 * Copyright (c) 2023 by Contributors
 * \file dynamic_bitset.h
 * \brief The header for the dynamic bitset container.
 */
#ifndef MLC_LLM_SERVE_GRAMMAR_SUPPORT_H_
#define MLC_LLM_SERVE_GRAMMAR_SUPPORT_H_

#include <tvm/runtime/logging.h>

#include <cstdint>
#include <cstring>

namespace mlc {
namespace llm {
namespace serve {

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

/*!
 * \brief Unionize the target set with the source set, and store the result in the target
 * set in O(n) time complexity. Suppose that both sets are sorted.
 */
void UnionizeWith(std::vector<int32_t>* target, const std::vector<int32_t>& source) {
  static std::vector<int32_t> result;
  result.clear();
  result.reserve(target->size() + source.size());
  auto it1 = target->begin();
  auto it2 = source.begin();
  while (it1 != target->end() && it2 != source.end()) {
    if (*it1 < *it2) {
      result.push_back(*it1);
      ++it1;
    } else if (*it1 > *it2) {
      result.push_back(*it2);
      ++it2;
    } else {
      result.push_back(*it1);
      ++it1;
      ++it2;
    }
  }
  while (it1 != target->end()) {
    result.push_back(*it1);
    ++it1;
  }
  while (it2 != source.end()) {
    result.push_back(*it2);
    ++it2;
  }
  target->swap(result);
}

/*!
 * \brief Intersect the target set with the source set, and store the result in the target
 * set in O(n) time complexity. Suppose that both sets are sorted.
 * \note When the target is {-1}, it represents the universal set. The result will be the source
 * set.
 */
void IntersectWith(std::vector<int32_t>* target, const std::vector<int32_t>& source) {
  static std::vector<int32_t> result;

  if (!target->empty() && target->at(0) == -1) {
    *target = source;
    return;
  }

  result.clear();
  result.reserve(std::min(target->size(), source.size()));
  auto it1 = target->begin();
  auto it2 = source.begin();
  while (it1 != target->end() && it2 != source.end()) {
    if (*it1 < *it2) {
      ++it1;
    } else if (*it1 > *it2) {
      ++it2;
    } else {
      result.push_back(*it1);
      ++it1;
      ++it2;
    }
  }
  target->swap(result);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_SUPPORT_H_
