#include "api.cuh"
int count;

float l2_float32(const float* data, int data_i, int data_j, int data_dim){
    float sum=0.0;
    for (int dim=0;dim<data_dim;++dim){
        float rest=data[data_i*data_dim+dim]-data[data_j*data_dim+dim];
        sum+=rest*rest;
    }
    // printf("dis[%d,%d]=%f\n",data_i,data_j,sum);
    return sum;
}
float dot_float32(const float* data, int data_i, int data_j, int data_dim){
    float sum=0.0;
    for (int dim=0;dim<data_dim;++dim){
        sum+=data[data_i*data_dim+dim]*data[data_j*data_dim+dim];
    }
    // printf("dis[%d,%d]=%f\n",data_i,data_j,sum);
    return sum;
}

void gemmtest(const float *data, float* result,int batch_len,int x,int y){
    for (int i=0;i<batch_len;++i){
        for (int j=0;j<batch_len;++j){
            float diff = fabs(result[i*batch_len+j]-dot_float32(data,x*batch_len+i,y*batch_len+j,20));
            if (diff>0.01){
                count++;
                if (count<100)
                printf("error:[%d,%d]%f-%f\n",i,j,result[i*batch_len+j],dot_float32(data,x+i,y+j,20));
                // return;
            }
        }
    }
return;
}

void taketest(const float *data, float* result, coo* output, int batch_len,int edge_num,int lid,int rid){
    for (int edge_id=0;edge_id<edge_num;++edge_id){
        int base_id = output[edge_id].base_id+lid*batch_len;
        int query_id = output[edge_id].query_id+rid*batch_len;
        float distance = output[edge_id].distance;
            float diff = fabs(distance-l2_float32(data,base_id,query_id,20));
            if (diff>0.01){
                count++;
                if (count<200)
                printf("error:[%d,%d]%f-%f\n",base_id,query_id,distance,l2_float32(data,base_id,query_id,20));
                // return;
            }
    }
return;
}


int DistanceLinker(const void *node_data,
                    void *edge_data,
                    size_t data_num,
                    size_t data_dim,
                    float threshold,
                    size_t batch_len)
{
    printf("%lu\n%lu\n%f\n", data_num, data_dim, threshold);

    // 检查device状况
    int device = 0;
    cudaSetDevice(device);
    printf("device checked\n");

    // 预先分配空间
    float *data_m, *result_m, *module_m;
    coo *output_m;
    unsigned int *take_num_m;
    size_t data_size = data_num * data_dim * sizeof(float);
    printf("%ld\n",data_size);

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

    printf("allocated.\n");

    // // 读取测试
    // for (int line_id = 0; line_id < 10; ++line_id)
    // {
    //     for (int element_id = 0; element_id < data_dim; ++element_id)
    //     {
    //         printf("%f,", data_m[line_id * data_dim + element_id]);
    //     }
    //     printf("\n");
    // }
    // printf("ready to run\n");

    // 分块计算
    // 先计算module array
    SetModule(data_m, module_m, data_num, data_dim);
    printf("module set.\n");

    // seg_num代表了需要分段计算的段数，每一个seg之间都要依次计算距离。
    int seg_num = data_num / batch_len;
    // seg_num = 1;
    printf("batch len is %lu\n", batch_len);
    printf("segment number is %d\n",seg_num);

    // 限制最多得到1024条边
    // int output_size = 100 * sizeof(coo);

    // 循环计算各个seg
    int cal_flag = 0;
    int total_take_num=0;
    for (int lid = 0;cal_flag == 0 && lid < seg_num; ++lid)
    {
        for (int rid = lid; cal_flag == 0 && rid < seg_num; ++rid)
        {
            *take_num_m = 0;
            // if (rid < lid)
            //     break;
            float *lhs_m = data_m + lid * batch_len * data_dim;
            float *rhs_m = data_m + rid * batch_len * data_dim;
            float *module_base = module_m + lid * batch_len;
            float *module_query = module_m + rid * batch_len;
            GemmBlas(lhs_m, rhs_m, result_m, batch_len, batch_len, data_dim);
            // gemmtest(data_m,result_m,batch_len,lid,rid);
            ModuleTake(result_m, module_base, module_query, take_num_m, output_m, batch_len, batch_len, threshold);
            // taketest(data_m,result_m,output_m,batch_len,*take_num_m,lid,rid);
            // 写回结果
            // memcpy((coo *)(edge_data) + *take_num_m, output_m + *take_num_m, *take_num_m * sizeof(coo));
            for (int edge_id=0;cal_flag == 0 && edge_id<*take_num_m;++edge_id){
                int base_id = output_m[edge_id].base_id+lid*batch_len;
                int query_id = output_m[edge_id].query_id+rid*batch_len;
                float distance = output_m[edge_id].distance;
                // distance = l2_float32(data_m,base_id,query_id,data_dim);
                if (base_id<query_id){
                    if (distance<PRECISION_LIMIT*-10.0f) {
                        float diff = l2_float32(data_m,base_id,query_id,data_dim) - distance;
                        if (distance > 0 && diff > PRECISION_LIMIT*-1.0f) {
                        printf("gpu:%f |cpu:%f\n",distance,l2_float32(data_m,base_id,query_id,data_dim));
                        ++count;
                        //     return 0;
                        }
                        
                    }
                    else{
                        ((coo *)(edge_data)+total_take_num)->base_id=base_id;
                        ((coo *)(edge_data)+total_take_num)->query_id=query_id;  
                        ((coo *)(edge_data)+total_take_num)->distance=distance;
                        ++total_take_num; 
                    }  
                }
                if (total_take_num==MAX_EDGE_NUM){
                    total_take_num = -1;
                    cal_flag = 1;
                }      
            }
        }
    }
    // printf("take num:%u\n", *take_num_m);
    printf("total take num:%d\n", total_take_num);
    printf("%d\n",count);

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


