#ifndef CUDA_WRAPPER_H_
#define CUDA_WRAPPER_H_

#include <cuda_runtime.h>

#include "config.h"

ErrorType CudaMallocManaged(void * pointer_managed, size_t buffer_size);
ErrorType CudaSetDevice(int device_index);
ErrorType CudaFree(void * pointer_managed);

#endif  // CUDA_WRAPPER_H_