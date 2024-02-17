// /*!
//  * Copyright (c) 2023 by Contributors
//  * \file progress_bar.h
//  * \brief A simple progress bar in C++.
//  */
// #ifndef MLC_LLM_SUPPORT_CSR_STORAGE_H_
// #define MLC_LLM_SUPPORT_CSR_STORAGE_H_

// #include <tvm/runtime/logging.h>

// #include <cstdint>
// #include <vector>

// namespace mlc {
// namespace llm {

// template <typename T>
// struct PointerVector {
//   const T* data;
//   int32_t len;

//   Vector(const T* data, int32_t len) : data(data), len(len) {}
//   Vector(const std::vector<T>& data) : data(data.data()), len(data.size()) {}

//   const T* begin() const { return data; }
//   const T* end() const { return data + len; }
//   const int32_t size() const { return len; }
//   /*! \brief Get the i-th element of the data array. */
//   const T& operator[](int i) const {
//     DCHECK(i >= 0 && i < static_cast<int32_t>(len)) << "Index " << i << " is out of bound";
//     return data[i];
//   }
// };

// template <typename T>
// class CSRStorage {
//  public:
//   int32_t Insert(const PointerVector& element) {
//     indptr_.push_back(data_.size());
//     data_.insert(element.len);
//     data_.insert(data_.end(), element.begin(), element.end());
//     return indptr_.size() - 1;
//   }

//   PointerVector operator[](int32_t index) const {
//     DCHECK(index < data_.size());
//     const T* ptr = data_.data() + index;
//     return PointerVector(ptr + 1, *ptr);
//   }

//   int32_t Size() const { return indptr_.size(); }

//  private:
//   std::vector<T> data_;
//   std::vector<int32_t> indptr_;
// };

// }  // namespace llm
// }  // namespace mlc

// #endif  // MLC_LLM_SUPPORT_CSR_STORAGE_H_
