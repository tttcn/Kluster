/*
*   topk方法。
*   暂定使用reduce实现。
*   --------------------------------------------------------------------------------------------------------------- 
*   依赖说明：
*   使用cuBLAS，需要在nvcc编译时加上"-lcublas"选项
*   使用cuSparse，需要在nvcc编译时加上"-lsparse"选项
*   使用OpenMP，需要在gcc编译时加上"-fopenmp"；或者使用nvcc编译，使用"-Xcompiler -fopenmp"将"-fopenmp"参数直接传递给gcc。
*/
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusparse.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>

