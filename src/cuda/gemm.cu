/*
*   使用cuBLAS库所实现的距离计算方法，封装了INT8和FLOAT32两种类型的计算接口，INT8部分使用SIMD指令。
*   cuBLAS实现的gemm效率极高，经过了汇编指令优化，几乎可以完全发挥出gpu的全部算力。
*   ---------------------------------------------------------------------------------------------------------------
*   编译说明：
*   使用cuBLAS，需要在nvcc编译时加上"-lcublas"选项
*   使用cuSparse，需要在nvcc编译时加上"-lsparse"选项
*   使用OpenMP，需要在gcc编译时加上"-fopenmp"；或者使用nvcc编译，使用"-Xcompiler
* -fopenmp"将"-fopenmp"参数直接传递给gcc。
*/
#include "gemm.cuh"

#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

#include "src/matrix.h"

// len/num/size
// rhs(r_row,cross_dim)/lhs(cross_dim,l_col)
// 都是按行顺序存储的，与cublas中按列存储的方式不同
void GemmBlas(const float *lhs_d, const float *rhs_d, float *result_d,
              int l_col, int r_row, int cross_dim) {
  // cublas init
  cublasHandle_t handle;
  cublasCreate(&handle);

  float a = 1.f;
  float b = 0.f;
  // C=a*opt(A)*opt(B)+b*C
  // result = -2*trans(rhs)*lhs + mod_mat
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, r_row, l_col, cross_dim, //
              &a, rhs_d, cross_dim,                                      // A
              lhs_d, cross_dim,                                          // B
              &b, result_d, r_row);
  cudaDeviceSynchronize();
  // clear cublas handle
  cublasDestroy(handle);
  return;
}

// gemm(lhs,rhs,result) = result = -2*trans(rhs)*lhs
//
template <typename T>
void Gemm(const Matrix<T> &lhs, const Matrix<T> &rhs, Matrix<T> &result) {
  // check parameters
  // check type
  bool type_check_passed = true;
  // ElementType element_type = lhs.element_type_;
  // if (lhs.element_type_ != rhs.element_type_ ||
  //     lhs.element_type_ != result.element_type_ ||
  //     rhs.element_type_ != result.element_type_) {
  //   type_check_passed = false;
  // }
  // check shape
  bool shape_check_passed = true;
  if (lhs.col_num_ != rhs.row_num_ || result.row_num_ != lhs.row_num_ ||
          result.col_num_ != rhs.col_num_) {
    shape_check_passed = false;
  }
  bool check_passed = type_check_passed && shape_check_passed;
  // check_passed = CheckParameters(lhs_d, rhs_d);

  if (check_passed) {
    // prefetch
    // float *lhs_ptr = lhs.Prefetch();
    // float *rhs_ptr = rhs.Prefetch();

    // cublas init
    // cublasHandle_t handle;
    // cublasCreate(&handle);

    // switch (0) {
    // case:
    //   INT8 // SIMD指令,对计算能力要求在x.y以上。
    //       break;
    // case:
    //   FLOAT32
    //   float a = 1.f;
    //   float b = 0.f;
    //   float *lhs_ptr = lhs.Get();
    //   float *rhs_ptr = rhs.Get();
    //   float *result_ptr = result.Get();
    //   int r_row = rhs.row_num_;
    //   int l_col = lhs.col_num_;
    //   int cross_dim = lhs.row_num_;
    //   // C=a*opt(A)*opt(B)+b*C
    //   // result = trans(rhs)*lhs + mod_mat
    //   cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, r_row, l_col, cross_dim, //
    //               &a, rhs_ptr, cross_dim, // A
    //               lhs_ptr, cross_dim,     // B
    //               &b, result_ptr, r_row);
    //   break;
    // }

    // cudaDeviceSynchronize();
    // // clear cublas handle
    // cublasDestroy(handle);
  }

  return;
}

void Gemm(const Matrix<Float32> &lhs, const Matrix<Float32> &rhs,
          Matrix<Float32> &result) {
  // check parameters
  // check type
  bool type_check_passed = true;
  // check shape
  bool shape_check_passed = true;
  if (lhs.col_num_ != rhs.row_num_ || result.row_num_ != lhs.row_num_ ||
          result.col_num_ != rhs.col_num_) {
    shape_check_passed = false;
  }
  bool check_passed = type_check_passed && shape_check_passed;
  // check_passed = CheckParameters(lhs_d, rhs_d);

  if (check_passed) {
    // prefetch
    // float *lhs_ptr = lhs.Prefetch();
    // float *rhs_ptr = rhs.Prefetch();

    // cublas init
    cublasHandle_t handle;
    cublasCreate(&handle);

    float a = 1.f;
    float b = 0.f;
    auto lhs_ptr = lhs.Get();
    auto rhs_ptr = rhs.Get();
    auto result_ptr = result.Get();
    int r_row = rhs.row_num_;
    int l_col = lhs.col_num_;
    int cross_dim = lhs.row_num_;
    // C=a*opt(A)*opt(B)+b*C
    // result = trans(rhs)*lhs + mod_mat
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, r_row, l_col, cross_dim, //
                &a, rhs_ptr, cross_dim,                                    // A
                lhs_ptr, cross_dim,                                        // B
                &b, result_ptr, r_row);
    cudaDeviceSynchronize();
    // clear cublas handle
    cublasDestroy(handle);
  }

  return;
}
