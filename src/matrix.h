#ifndef MATRIX_H_
#define MATRIX_H_

/*
    matrix是处理的数据结构，数据存储在CPU上
*/

#include <vector>

#include "src/config.h"
#include "src/cuda_wrapper.h"

struct Memory{
    size_t memory_size_;
    void * raw_pointer_;
    bool is_occupied_;
};

class Matrix
{
private:
    size_t row_num_;
    size_t col_num_;
    ElementType element_type_;
    DeviceType device_type_;
    size_t data_size_;
    void * managed_pointer;

    // Memory buffer_list_;
public:
    Matrix(void * origin_pointer, size_t row_num, size_t col_num, ElementType element_type,DeviceType device_type);
    Matrix(size_t row_num, size_t col_num, ElementType element_type,DeviceType device_type);
    ~Matrix();
    void Init(size_t row_num, size_t col_num, ElementType element_type,DeviceType device_type);
    void * Get();
    Matrix Slice();
};

#endif // MATRIX_H_