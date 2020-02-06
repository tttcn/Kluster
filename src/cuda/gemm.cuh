void GemmBlas(const float *lhs_d, const float *rhs_d, float *result_d,
              int l_col, int r_row, int cross_dim);

void Gemm(const Matrix & lhs_d, const Matrix & rhs_d, Matrix & result_d);