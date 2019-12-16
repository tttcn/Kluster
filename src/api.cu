#include "api.cuh"

int DistanceLinker(const void *node_data,
                   void *edge_data,
                   size_t data_num,
                   size_t data_dim,
                   float threshold,
                   size_t batch_len)
{
    DEBUG("data num is %lu\n", data_num);
    DEBUG("data dim is %lu\n", data_dim);
    DEBUG("threshold is %f\n", threshold);

    // 检查device状况
    int device = 0;
    cudaSetDevice(device);
    DEBUG("device checked\n");

    // 预先分配空间
    float *data_m, *result_m, *module_m;
    coo *output_m;
    unsigned int *take_num_m;
    size_t data_size = data_num * data_dim * sizeof(float);
    DEBUG("data size is %lu\n", data_size);

    // _m means managed
    cudaMallocManaged((void **)&data_m, data_size); // 原始数据
    memcpy(data_m, node_data, data_size);
    // cudaMemPrefetchAsync(data_m,data_size,0,NULL);

    cudaMallocManaged((void **)&module_m, data_num * sizeof(float)); // 模方数据
    // cudaMemPrefetchAsync(module_m,data_num*sizeof(float),0,NULL);

    cudaMallocManaged((void **)&result_m, batch_len * batch_len * sizeof(float)); // 距离结果数据
    // cudaMemPrefetchAsync(result_m,batch_num*batch_num*sizeof(float),0,NULL);

    cudaMallocManaged((void **)&output_m, batch_len * batch_len * sizeof(coo)); // 输出coo数据
    // cudaMemPrefetchAsync(output_m,batch_num*batch_num*sizeof(coo),0,NULL);

    cudaMallocManaged((void **)&take_num_m, sizeof(int)); // 取边数目
    // cudaMemPrefetchAsync(take_num_m,sizeof(int),0,NULL);

    DEBUG("managed vram allocated.\n");

    // 分块计算
    // 先计算module array
    SetModule(data_m, module_m, data_num, data_dim);
    DEBUG("module set.\n");

    // seg_num代表了需要分段计算的段数，每一个seg之间都要依次计算距离。
    int seg_num = data_num / batch_len;

#ifdef DEBUG_
    seg_num = 3;
#endif

    DEBUG("batch len is %lu\n", batch_len);
    DEBUG("segment number is %d\n", seg_num);

    // 限制最多得到1024条边
    // int output_size = 100 * sizeof(coo);

    // 循环计算各个seg
    int cal_flag = 0;
    int total_take_num = 0;
    for (int lid = 0; cal_flag == 0 && lid < seg_num; ++lid)
    {
        for (int rid = lid; cal_flag == 0 && rid < seg_num; ++rid)
        {
            *take_num_m = 0;
            float *lhs_m = data_m + lid * batch_len * data_dim;
            float *rhs_m = data_m + rid * batch_len * data_dim;
            float *module_base = module_m + lid * batch_len;
            float *module_query = module_m + rid * batch_len;
            GemmBlas(lhs_m, rhs_m, result_m, batch_len, batch_len, data_dim);
#ifdef DEBUG_
            gemmtest(data_m, result_m, batch_len, lid, rid);
#endif
            ModuleTake(result_m, module_base, module_query, take_num_m, output_m, batch_len, batch_len, threshold);
#ifdef DEBUG_
            taketest(data_m, result_m, output_m, batch_len, *take_num_m, lid, rid);
#endif
            // 写回结果
            // memcpy((coo *)(edge_data) + *take_num_m, output_m + *take_num_m, *take_num_m * sizeof(coo));
            for (int edge_id = 0; cal_flag == 0 && edge_id < *take_num_m; ++edge_id)
            {
                int base_id = output_m[edge_id].base_id + lid * batch_len;
                int query_id = output_m[edge_id].query_id + rid * batch_len;
                float distance = output_m[edge_id].distance;
                if (base_id < query_id)
                {
                    ((coo *)(edge_data) + total_take_num)->base_id = base_id;
                    ((coo *)(edge_data) + total_take_num)->query_id = query_id;
                    ((coo *)(edge_data) + total_take_num)->distance = distance;
                    ++total_take_num;
                    if (total_take_num == MAX_EDGE_NUM)
                    {
                        total_take_num = -1;
                        cal_flag = 1;
                    }
#ifdef DEBUG_
                    if (distance < PRECISION_LIMIT)
                    {
                        float cpu_distance = l2_float32(data_m, base_id, query_id, data_dim);
                        float diff = cpu_distance - distance;
                        if (diff > PRECISION_LIMIT * 0.01)
                        {
                            printf("gpu:%f |cpu:%f\n", distance, cpu_distance);
                        }
                    }
#endif
                }
            }
        }
    }
    DEBUG("total take num (edge num) is %d\n", total_take_num);

    // 回收空间
    cudaFree(data_m);
    cudaFree(result_m);
    cudaFree(module_m);
    cudaFree(output_m);
    cudaFree(take_num_m);

    return total_take_num;
}

void Knn(const void *node_data,
         const void *edge_data,
         float threshold, int neighest_k)
{
    return;
}