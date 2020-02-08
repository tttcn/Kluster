#ifndef KLUSTER_CUDA_GEMM_H_
#define KLUSTER_CUDA_GEMM_H_

#include "src/matrix.h"

namespace Kluster {
void GemmBlas(const float *lhs_d, const float *rhs_d, float *result_d,
              int l_col, int r_row, int cross_dim);

template <typename T>
void Gemm(const Matrix<T> &lhs, const Matrix<T> &rhs, Matrix<T> &result);
}

#endif // KLUSTER_CUDA_GEMM_H_