#include "cuda_runtime.h"
#include "stdio.h"
struct coo
{
    int base_id;
    int query_id;
    float distance;
    float for_align;
};

__global__ void Kernel(unsigned int * ptr){
    atomicAdd(ptr, 1);
}

int main()
{
    size_t batch_len = 8192; 
    size_t data_num = 900000;
    size_t data_dim = 34;

    // 检查device状况
    int device = 0;
    cudaSetDevice(device);
    printf("device checked\n");


    // 预先分配空间
    float *data_m, *result_m, *module_m;
    coo *output_m;
    unsigned int *take_num_m;
    
    size_t data_size = data_num * data_dim * sizeof(float);

    // _m means managed
    cudaMallocManaged((void **)&data_m, data_size); // 原始数据
    // memcpy(data_m, node_data, data_size);
    // cudaMemPrefetchAsync(data_d,data_size,0,NULL);

    cudaMallocManaged((void **)&module_m, data_num * sizeof(float)); // 模方数据
    // cudaMemPrefetchAsync(module_d,data_num*sizeof(float),0,NULL);

    cudaMallocManaged((void **)&result_m, batch_len * batch_len * sizeof(float)); // 距离结果数据
    // cudaMemPrefetchAsync(result_d,batch_num*batch_num*sizeof(float),0,NULL);

    cudaMallocManaged((void **)&output_m, batch_len * batch_len * sizeof(coo)); // 输出coo数据
    // cudaMemPrefetchAsync(output_d,batch_num*batch_num*sizeof(coo),0,NULL);

    cudaMallocManaged((void **)&take_num_m, sizeof(int)); // 取边数目
    // cudaMemPrefetchAsync(take_num_m,sizeof(int),0,NULL);

    Kernel<<<223,14>>>(take_num_m);    

    printf("pre-allocation done.\n");
    


    // 回收空间
    cudaFree(data_m);
    cudaFree(result_m);
    cudaFree(module_m);
    cudaFree(output_m);
    cudaFree(take_num_m);

    return 0;
}


