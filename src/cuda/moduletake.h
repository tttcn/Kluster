#ifndef KLUSTER_CUDA_MODULETAKE_H_
#define KLUSTER_CUDA_MODULETAKE_H_

#include <cuda_runtime.h>

#include "src/config.h"
#include "src/matrix.h"

namespace Kluster {
template <typename T> void SetModule(const Matrix<T> &data, Matrix<T> &module);

template <typename T>
Uint32 ModuleTake(const Matrix<T> &product, const Matrix<T> &module_base,
                  const Matrix<T> &module_query, Matrix<Coo> &output,
                  float threshold);

__global__ void ModuleTakeKernel(const float *product_d,
                                 const float *module_base_d,
                                 const float *module_query_d,
                                 unsigned int *take_num_d, Coo *output_d,
                                 int base_len, int query_len, float threshold);

__global__ void SetModuleKernel(const float *data_d, float *module_d,
                                int data_num, int data_dim);

template <typename T> void SetModule(const Matrix<T> &data, Matrix<T> &module){ printf("undefined\n");}
template <typename T>
Uint32 ModuleTake(const Matrix<T> &product, const Matrix<T> &module_base,
                  const Matrix<T> &module_query, Matrix<Coo> &output,
                  float threshold){printf("undefined\n");}
}

#endif // KLUSTER_CUDA_MODULETAKE_H_