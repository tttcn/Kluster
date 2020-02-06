#ifndef KLUSTER_MATRIX_H_
#define KLUSTER_MATRIX_H_

/*
    matrix是处理的数据结构，数据存储在CPU上
*/

#include "config.h"

#include <vector>

struct Memory {
  size_t memory_size_;
  void *raw_pointer_;
  bool is_occupied_;
};

template <typename T> class Matrix {
public:
  // flag 控制不要在初始化的同时拷贝数据？
  Matrix(void *origin_pointer, size_t row_num, size_t col_num,
         ElementType element_type, DeviceType device_type);
  Matrix(size_t row_num, size_t col_num, ElementType element_type,
         DeviceType device_type);
  ~Matrix();
  void Init(size_t row_num, size_t col_num, ElementType element_type,
            DeviceType device_type);
  T Get();
  Matrix Slice();

  // private:
  size_t row_num_;
  size_t col_num_;
  ElementType element_type_;
  DeviceType device_type_;
  size_t data_size_;
  T *managed_pointer;
  // Matrix *cached_slice_;

  // Memory buffer_list_;
};

#endif // KLUSTER_MATRIX_H_