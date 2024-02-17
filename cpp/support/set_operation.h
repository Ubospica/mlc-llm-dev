/*!
 * Copyright (c) 2023 by Contributors
 * \file progress_bar.h
 * \brief A simple progress bar in C++.
 */
#ifndef MLC_LLM_SUPPORT_SET_OPERATION_H_
#define MLC_LLM_SUPPORT_SET_OPERATION_H_

#include <cstdint>
#include <vector>

namespace mlc {
namespace llm {

void UnionizeWith(std::vector<int32_t>* target, const std::vector<int32_t>& source) {
  // target and source are sorted, and avoid memory allocation in this function
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

void IntersectWith(std::vector<int32_t>* target, const std::vector<int32_t>& source) {
  // target and source are sorted, and avoid memory allocation in this function
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

void DifferenceWith(std::vector<int32_t>* target, const std::vector<int32_t>& source) {
  // target and source are sorted, and avoid memory allocation in this function
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
