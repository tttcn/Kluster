#ifndef KLUSTER_CUDA_GEMM_H_
#define KLUSTER_CUDA_GEMM_H_

#include "src/matrix.h"

namespace Kluster {
void GemmBlas(const float *lhs_d, const float *rhs_d, float *result_d,
              int l_col, int r_row, int cross_dim);

template <typename T>
void Gemm(const Matrix<T> &lhs, const Matrix<T> &rhs, Matrix<T> &result) {
  // check parameters
  // check type
  bool type_check_passed = true;
  // check shape
  bool shape_check_passed = true;
  // if (lhs.col_num_ != rhs.row_num_ || result.row_num_ != lhs.row_num_ ||
  //     result.col_num_ != rhs.col_num_) {
  //   shape_check_passed = false;
  // }
  bool check_passed = type_check_passed && shape_check_passed;
  // check_passed = CheckParameters(lhs_d, rhs_d);

  if (check_passed) {
    printf("undefined gemm\n");
  }

  return;
}
}

#endif // KLUSTER_CUDA_GEMM_H_