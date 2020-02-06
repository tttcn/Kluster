#ifndef KLUSTER_CUDA_WRAPPER_H_
#define KLUSTER_CUDA_WRAPPER_H_

#include <cuda_runtime.h>

#include "config.h"

ErrorType CudaCheck(cudaError_t error_code);

ErrorType CudaMallocManaged(void ** pointer_address, size_t buffer_size);
ErrorType CudaSetDevice(int device_index);
ErrorType CudaFree(void * pointer_managed);

#endif  // KLUSTER_CUDA_WRAPPER_H_