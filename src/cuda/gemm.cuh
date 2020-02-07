#include "src/matrix.h"

void GemmBlas(const float *lhs_d, const float *rhs_d, float *result_d,
              int l_col, int r_row, int cross_dim);

void Gemm(const Matrix &lhs, const Matrix &rhs, Matrix &result);