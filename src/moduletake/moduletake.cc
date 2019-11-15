#include "moduletake.h"
#include <cuda_runtime.h>

//都是d的原因是这些指针的量都属于临时生成的，没有必要复制回ram
void ModuleTake(const float *product_d,
                const float *module_base_d,
                const float *module_query_d,
                int base_num,
                int query_num,
                float threshold,
                coo *output_h)
{
    dim3 grid_dim(base_num, query_num / 512, 1);
    dim3 block_dim(512, 1, 1);

    //

    // printf("batch %d,%d: condtake init:%d.\n",lid,rid,*condtake_num_d);
    ModuleTakeKernel<<<grid_dim, block_dim>>>(product_d,
                                              module_base_d,
                                              module_query_d,
                                              base_num, query_num,
                                              output_d);
    cudaDeviceSynchronize();
    

    cuda
    return;
}