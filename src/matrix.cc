#include "matrix.h"

#include <memory>

#include "cuda_wrapper.h"

Matrix::Matrix(void *origin_pointer, size_t row_num, size_t col_num,
               ElementType element_type, DeviceType device_type) {
  Init(size_t row_num, size_t col_num, ElementType element_type,
       DeviceType device_type);
  memcpy(managed_pointer_, origin_pointer, data_size_);
}

Matrix::Matrix(size_t row_num, size_t col_num, ElementType element_type,
               DeviceType device_type) {
  Init(size_t row_num, size_t col_num, ElementType element_type,
       DeviceType device_type);
}

Matrix::~Matrix() {
  // 回收订阅的UM内存
  CudaFree(managed_pointer_);
}

void Init(size_t row_num, size_t col_num, DeviceType device_type) {
  row_num_ = row_num;
  col_num_ = col_num;
  // element_type_ = element_type;
  device_type_ = device_type;
  data_size_ = row_num_ * col_num_ * sizeof(ElementType);
  managed_pointer_ = NULL;
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

Matrix Slice(size_t row_begin, size_t row_end, size_t col_begin,
             size_t col_end) {
  size_t slice_row_num = row_end - row_begin;
  size_t slice_col_num = col_end - col_begin;
  size_t element_size = kElementSize[element_type_];
  // cached_slice_ = new Matrix(row_num, col_num, element_type_, device_type_);
  Matrix new_matrix(row_num, col_num, element_type_, device_type_);
  for (int row_id = row_begin, row_id < row_end; ++row_id) {
    void *buffer_pointer =
        managed_pointer + (row_id * col_num_ + col_begin) * element_size;
    memcpy(new_matrix.Get(), buffer_pointer, slice_col_num * element_size);
  }
  // 预取
  // CudaPrefechAsync();
  return new_matrix;
}

void *Matrix::Get() { return managed_pointer; }