/*
*   在条件取出前加上向量的模方，减少GPU的IO时间（有部分）。
*   其实模方这部分读取并不是太耗时间，如果和condtake分开也只要读N+N次，但是还有一个矩阵结果N*N的读取；写入也需要N*N次，还是有不少。
*   如果合并做，相当于读取上省了一个N*N，写入上省了一个N*N。
*   --------------------------------------------------------------------------------------------------------------- 
*   依赖说明：
*   使用cuda_runtime.h
*/

#include "moduletake.cuh"

void SetModule(const float *data_d,
               float *module_d,
               int data_num,
               int data_dim)
{
    dim3 grid_dim(data_num / 512, 1, 1);
    dim3 block_dim(512, 1, 1);
    SetModuleKernel<<<grid_dim, block_dim>>>(data_d, module_d, data_num, data_dim);
    return;
}

void ModuleTake(const float *product_d,
                const float *module_base_d,
                const float *module_query_d,
                unsigned int *take_num_d,
                coo *output_d,
                int base_len,
                int query_len,
                float threshold)
{
    dim3 grid_dim(base_len / 512, 1, 1);
    dim3 block_dim(512, 1, 1);
    ModuleTakeKernel<<<grid_dim, block_dim>>>(product_d, module_base_d, module_query_d, take_num_d, output_d, base_len, query_len, threshold);
    return;
}

// 复杂度O(nd)，1M数据400ms左右，只要执行一次，优化意义不大
// 优化的思路是按列扫描，提高一个线程束的读取效率
// 用atomic的原因是因为进入这一分支的线程并不多。
__global__ void ModuleTakeKernel(const float *product_d,
                                 const float *module_base_d,
                                 const float *module_query_d,
                                 unsigned int *take_num_d,
                                 coo *output_d,
                                 int base_len,
                                 int query_len,
                                 float threshold)
{
    int query_id = (blockIdx.x * blockDim.x + threadIdx.x);
    for (int base_id = 0; base_id < base_len; ++base_id)
    {
        // 这个优化感觉没有必要做，未测试
        // if (base_id == query_id)
        //     continue;
        float distance = -2.0f * product_d[base_id * query_len + query_id] + module_base_d[base_id] + module_query_d[query_id];
        if (distance < threshold)
        {
            int output_id = atomicAdd(take_num_d, 1);
            output_d[output_id].base_id = base_id;
            output_d[output_id].query_id = query_id;
            output_d[output_id].distance = distance;
        }
    }
    return;
}

__global__ void SetModuleKernel(const float *data_d,
                                float *module_d,
                                int data_num,
                                int data_dim)
{
    int data_id = (blockIdx.x * blockDim.x + threadIdx.x);
    float module_square = 0.0f;
    for (int element_id = 0; element_id < data_dim; ++element_id)
    {
        float element = data_d[data_id * data_dim + element_id];
        module_square += element * element; // 关于这里是用内置power还是直接乘，没有测试过。
    }
    module_d[data_id] = module_square;
    return;
}