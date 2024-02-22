/*!
 * Copyright (c) 2023 by Contributors
 * \file set_operation.h
 * \brief The header for the support of set operations.
 */
#ifndef MLC_LLM_SUPPORT_SET_OPERATION_H_
#define MLC_LLM_SUPPORT_SET_OPERATION_H_

#include <cstdint>
#include <vector>

namespace mlc {
namespace llm {

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

/*!
 * \brief Find the set difference target\source, and store the result in the target set in O(n)
 * time complexity. Suppose that both sets are sorted.
 */
void DifferenceWith(std::vector<int32_t>* target, const std::vector<int32_t>& source) {
  static std::vector<int32_t> result;
  result.clear();
  result.reserve(target->size());
  auto it1 = target->begin();
  auto it2 = source.begin();
  while (it1 != target->end() && it2 != source.end()) {
    if (*it1 < *it2) {
      result.push_back(*it1);
      ++it1;
    } else if (*it1 > *it2) {
      ++it2;
    } else {
      ++it1;
      ++it2;
    }
  }
  while (it1 != target->end()) {
    result.push_back(*it1);
    ++it1;
  }
  target->swap(result);
}

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SUPPORT_SET_OPERATION_H_
