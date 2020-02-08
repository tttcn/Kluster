#ifndef KLUSTER_MATRIX_H_
#define KLUSTER_MATRIX_H_

/*
    matrix是处理的数据结构，数据存储在CPU上
*/

#include "config.h"

#include <cstdio>
#include <cstring>

#include "cuda_wrapper.h"

namespace Kluster {
struct Memory {
  size_t memory_size_;
  void *raw_ptr_;
  bool is_occupied_;
};

template <typename ElementType> class Matrix {
public:
  ~Matrix() {
    // 回收订阅的UM内存
    CudaFree(managed_ptr_);
  }

  Matrix(size_t row_num, size_t col_num, DeviceType device_type) {
    row_num_ = row_num;
    col_num_ = col_num;
    element_size_ = sizeof(ElementType);
    device_type_ = device_type;
    data_size_ = row_num_ * col_num_ * element_size_;
    managed_ptr_ = nullptr;
    // 直接订阅大量的UM内存，此时并不会传入
    switch (device_type_) {
    case CUDA:
      if (CudaMallocManaged((void **)(&managed_ptr_), data_size_) != NO_ERROR) {
        // ~Matrix();
      }
      break;
    case CPU:
      break;
    }
  }

  Matrix *Slice(Uint32 row_begin, Uint32 row_end, Uint32 col_begin,
                Uint32 col_end) {
    // check params
    if (row_end > row_num_ || col_end > col_num_) {
      return nullptr;
    }

    size_t slice_row_num = row_end - row_begin;
    size_t slice_col_num = col_end - col_begin;
    size_t element_size = element_size_;
    // cached_slice_ = new Matrix(row_num, col_num, elemen    //t_type_,
    // device_type_);
    Matrix<ElementType> *new_mat_ptr =
        new Matrix<ElementType>(slice_row_num, slice_col_num, device_type_);
    for (auto row_id = row_begin; row_id < row_end; ++row_id) {
      ElementType *buf_ptr = managed_ptr_ + (row_id * col_num_ + col_begin);
      ElementType *raw_ptr =
          new_mat_ptr->Get() + slice_col_num * (row_id - row_begin);
      memcpy(raw_ptr, buf_ptr, slice_col_num * element_size);
    }
    // 预取
    // CudaPrefechAsync();
    return new_mat_ptr;
  }

  void Show() {
    for (auto i = 0; i < row_num_; ++i) {
      for (auto j = 0; j < col_num_; ++j) {
        auto idx = i * col_num_ + j;
        printf("%f,", managed_ptr_[idx]);
      }
      printf("\n");
    }
  }

  // get pointer
  ElementType *Get() { return managed_ptr_; }
  const ElementType *Get() const { return managed_ptr_; }

  ElementType &operator[](Uint32 idx) { return managed_ptr_[idx]; }

  // private:
  size_t row_num_;
  size_t col_num_;
  size_t element_size_;
  DeviceType device_type_;
  size_t data_size_;
  ElementType *managed_ptr_;
  // Matrix *cached_slice_;

  // Memory buf_list_;
};
}

#endif // KLUSTER_MATRIX_H_