
#include "cuda_wrapper.h"

#include <cuda_runtime.h>
#include <stdio.h>

#include "debug_tool.h"

namespace Kluster {
ErrorType CudaCheck(cudaError_t error_code) {
  ErrorType execute_state = NO_ERROR;
  if (error_code != cudaSuccess) {
    execute_state = CUDA_ERROR;
    DEBUG("Cuda error name: %s\n", cudaGetErrorName(error_code));
    DEBUG("Cuda error description: %s\n", cudaGetErrorString(error_code));
  }
  return execute_state;
}

// ErrorType CudaMemPrefetchAsync(void ** pointer_address, size_t buffer_size)
// {
//     return CudaCheck(cudaMemPrefetchAsync(pointer_address, buffer_size));
// }

ErrorType CudaMallocManaged(void **pointer_address, size_t buffer_size) {
  return CudaCheck(cudaMallocManaged(pointer_address, buffer_size));
}

ErrorType CudaSetDevice(int device_index) {
  return CudaCheck(cudaSetDevice(device_index));
}

ErrorType CudaFree(void *pointer_managed) {
  return CudaCheck(cudaFree(pointer_managed));
}

// ErrorType CudaMemPrefetchAsync(void * pointer_managed, size_t buffer_size)
// {
//     return
//     CudaCheck(cudaMemPrefetchAsync(pointer_managed,buffer_size,0,NULL));
// }
}