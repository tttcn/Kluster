/*
*   使用cuBLAS库所实现的距离计算方法，封装了INT8和FLOAT32两种类型的计算接口，INT8部分使用SIMD指令。
*   cuBLAS实现的gemm效率极高，经过了汇编指令优化，几乎可以完全发挥出gpu的全部算力，唯一可能不够好的地方可能是没有做strassen算法优化。
*   --------------------------------------------------------------------------------------------------------------- 
*   编译说明：
*   使用cuBLAS，需要在nvcc编译时加上"-lcublas"选项
*   使用cuSparse，需要在nvcc编译时加上"-lsparse"选项
*   使用OpenMP，需要在gcc编译时加上"-fopenmp"；或者使用nvcc编译，使用"-Xcompiler -fopenmp"将"-fopenmp"参数直接传递给gcc。
*/
#include "gemm.cuh"

// len/num/size
// rhs(r_row,cross_dim)/lhs(cross_dim,l_col) 都是按行顺序存储的，与cublas中按列存储的方式不同
void GemmBlas(const float *lhs_d, const float *rhs_d, float *result_d,
              int l_col, int r_row, int cross_dim)
{
    // cublas init
    cublasHandle_t handle;
    cublasCreate(&handle);

    float a = 1.f;
    float b = 0.f;
    // C=a*opt(A)*opt(B)+b*C
    // result = -2*trans(rhs)*lhs + mod_mat
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                l_col, r_row, cross_dim, //
                &a,
                rhs_d, cross_dim, //A
                lhs_d, cross_dim, //B
                &b,
                result_d, r_row);
    cudaDeviceSynchronize();
    // clear cublas handle
    cublasDestroy(handle);
    return;
}
