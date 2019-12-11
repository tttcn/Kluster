#include "api.h"

void DistanceLinker(const void *node_data,
                    void *edge_data,
                    int data_num,
                    int data_dim,
                    float threshold)
{
    // 检查device状况
    cudaSetDevice(device);

    // 预先分配空间
    float *data_m, *result_m, *module_m;
    coo *output_m;
    unsigned int *condtake_num_m;
    unsigned long long int data_size = data_num * data_dim * sizeof(float);

    cudaMallocManaged((void **)&data_m, data_size); // 原始数据
    memcpy(data_m, node_data, data_size);
    // cudaMemPrefetchAsync(data_d,data_size,0,NULL);

    cudaMallocManaged((void **)&module_m, data_num * sizeof(float)); // 模方数据
    // cudaMemPrefetchAsync(module_d,data_num*sizeof(float),0,NULL);

    cudaMallocManaged((void **)&result_m, batch_num * batch_num * sizeof(float)); // 距离结果数据
    // cudaMemPrefetchAsync(result_d,batch_num*batch_num*sizeof(float),0,NULL);

    cudaMallocManaged((void **)&output_m, batch_num * batch_num * sizeof(coo)); // 输出coo数据
    // cudaMemPrefetchAsync(output_d,batch_num*batch_num*sizeof(coo),0,NULL);

    cudaMallocManaged((void **)&condtake_num_m, sizeof(int)); // 取边数目
    // cudaMemPrefetchAsync(condtake_num_d,sizeof(int),0,NULL);

    printf("pre-allocation done.\n");

    // 分块计算
    // 先计算module array
    SetModule(data_d, module_d, data_num, data_dim);

    // seg_num代表了需要分段计算的段数，每一个seg之间都要依次计算距离。
    int batch_len = 8192;   // 经验参数
    int seg_num = data_len / batch_len;
    printf("batch len is %d\n", batch_num);

    // 限制最多得到1024条边
    int output_size = 1024 * sizeof(coo);   

    // 循环计算各个seg
    unsigned long long int total_condtake_num;
    for (int i = 0; i < seg_num; ++i)
    {
        for (int j = i; j < seg_num; ++j)
        {
            if (i < j)
                break;
            float *lhs_m = data_m + lid * batch_len * data_dim;
            float *rhs_m = data_m + rid * batch_len * data_dim;
            float *module_base= module_m + lid * batch_len;
            float *module_query= module_m + rid * batch_len;
            GemmBlas(lhs_m,rhs_m, result_m,batch_len,batch_len, data_dim);
            ModuleTake(result_m,module_base,module_query,output_m,batch_len,batch_len,threshold);
            // 写回结果
            memcpy(edge_data +, output_m, output_size);
        }
    }
    printf("condtake num:%u\n", *condtake_num_d);
    printf("total condtake num:%lld\n", total_condtake_num);

    // 回收空间
    cudaFree(data_m);
    cudaFree(result_m);
    cudaFree(module_m);
    cudaFree(output_m);
    cudaFree(condtake_num_m);

    return;
}

void Knn(const void *node_data,
         const void *edge_data,
         float threshold, int neighest_k)
{
    return;
}
