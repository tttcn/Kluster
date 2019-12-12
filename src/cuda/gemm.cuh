#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusparse.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>

void GemmBlas(const float *lhs_d, const float *rhs_d, float *result_d,
              int l_col, int r_row, int cross_dim);