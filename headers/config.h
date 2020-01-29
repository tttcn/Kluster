#ifndef KLUSTER_CONFIG_H_
#define KLUSTER_CONFIG_H_

// consts
#define PRECISION_LIMIT 0.01f   // 可能存在的绝对浮点误差大小
#define MAX_EDGE_NUM 0x1<<28    // 256M
#define BATCH_LEN 16384

struct coo
{
    int base_id;
    int query_id;
    float distance;
    float for_align;
};

struct Node
{
    int data;
    int rank;
    int parent;
};

const unsigned long long int GB=1024*1024*1024;
const unsigned long long int MB = 1024 * 1024;
const int Num = 1024 * 1024;
const int Dim = 512;
const int Batch = 1024 * 16;
const int Grid_dim_x = Num;
const int Block_dim_x = Dim / 2;
const int Grid_dim_y = Num / Block_dim_x;
const int Batch_TH = Batch * Batch;

// types
typedef int DeviceIndex;

// enumerates
enum DeviceType {CPU,NUMA,CUDA};
enum ElementType {FLOAT64,FLOAT32,FLOAT16,INT32,INT16,INT8};
enum ErrorType {NO_ERROR,FUNCTION_ERROR,CUDA_ERROR,USAGE_ERROR,INITIALIZATION_ERROR};


#endif // KLUSTER_CONFIG_H_