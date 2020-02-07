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
  Matrix(const void *origin_pointer, size_t row_num, size_t col_num,
         DeviceType device_type);
  Matrix(size_t row_num, size_t col_num, DeviceType device_type);
  ~Matrix();
  void Init(size_t row_num, size_t col_num, DeviceType device_type);
  T *Get();
  const T *Get() const;
  Matrix Slice(Uint32 row_begin, Uint32 row_end, Uint32 col_begin,
               Uint32 col_end);
  T &operator[](Uint32 idx);

  // private:
  size_t row_num_;
  size_t col_num_;
  size_t element_size_;
  DeviceType device_type_;
  size_t data_size_;
  T *managed_pointer;
  // Matrix *cached_slice_;

  // Memory buffer_list_;
};

Matrix::Matrix(const void *origin_pointer, size_t row_num, size_t col_num,
               DeviceType device_type) {
  Init(size_t row_num, size_t col_num, DeviceType device_type);
  memcpy(managed_pointer_, origin_pointer, data_size_);
}

Matrix::Matrix(size_t row_num, size_t col_num, DeviceType device_type) {
  Init(size_t row_num, size_t col_num, DeviceType device_type);
}

Matrix::~Matrix() {
  // 回收订阅的UM内存
  CudaFree(managed_pointer_);
}

void Init(size_t row_num, size_t col_num, DeviceType device_type) {
  row_num_ = row_num;
  col_num_ = col_num;
  element_size_ = sizeof(T);
  device_type_ = device_type;
  data_size_ = row_num_ * col_num_ * element_size_;
  managed_pointer_ = nullptr;
  // 直接订阅大量的UM内存，此时并不会传入
  switch (device_type_) {
  case:
    CUDA if (CudaManagedMalloc(&managed_pointer_, data_size_) != NO_ERROR) {
      ~Matrix();
    }
    break;
  case:
    CPU
  }
}

Matrix<T> Slice(Uint32 row_begin, Uint32 row_end, Uint32 col_begin,
                Uint32 col_end) {
  size_t slice_row_num = row_end - row_begin;
  size_t slice_col_num = col_end - col_begin;
  size_t element_size = element_size_;
  // cached_slice_ = new Matrix(row_num, col_num, element_type_, device_type_);
  Matrix<T> new_matrix(row_num, col_num, device_type_);
  for (int row_id = row_begin, row_id < row_end; ++row_id) {
    T *buffer_pointer = managed_pointer + (row_id * col_num_ + col_begin);
    memcpy(new_matrix.Get(), buffer_pointer, slice_col_num * element_size);
  }
  // 预取
  // CudaPrefechAsync();
  return new_matrix;
}

T *Matrix::Get() { return managed_pointer; }
const T *Get() const { return managed_pointer; }

T &operator[](Uint32 idx){return managed_pointer[idx]}

#endif // KLUSTER_MATRIX_H_