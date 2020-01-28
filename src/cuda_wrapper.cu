
#include "cuda_wrapper.h"

ErrorType CudaMallocManaged(void * pointer_managed, size_t buffer_size)
{   
    cudaMallocManaged((void **)&pointer_managed, buffer_size);
    return;
}

ErrorType CudaSetDevice(int device_index){
    cudaSetDevice(device_index);
    return ;
}

ErrorType CudaFree(void * pointer_managed){
    cudaFree(pointer_managed);
    return;
}