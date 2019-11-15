/*
*   测试程序
*   测试1：对各种随机形状的输入计算起coo输出。
*   测试2：正确性检验，使用cpu计算相同结果测试cuda代码的正确性。
*   --------------------------------------------------------------------------------------------------------------- 
*   依赖说明：
*   使用OpenMP，需要在gcc编译时加上"-fopenmp"；或者使用nvcc编译，使用"-Xcompiler -fopenmp"将"-fopenmp"参数直接传递给gcc。
*/

// #include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>

// 内部
#include "operator.h"

// const unsigned long long int GB=1024*1024*1024;
const unsigned long long int MB=1024*1024;
// const float TH = 100.0;
const int Num=1024*1024;
const int Dim=512;
const int Batch=1024*32;
const int Grid_dim_x=Num;
const int Block_dim_x=Dim/2;
const int Grid_dim_y=Num/Block_dim_x;
const int Len=64;


#define NUM 1024
#define DIM 512
#define BLOCK_BATCH 8


__global__
void IP(const float* data_d, float *result_d, int data_num, int data_dim){
    unsigned long long int i=blockIdx.x*data_dim;
    unsigned long long int j=(blockIdx.y*blockDim.x+threadIdx.x)*data_dim;
    if (i<j)   return;
    float sum=0.0;
    for (int dim=0;dim<data_dim;dim+=4){
        sum+=(data_d[i+dim]*data_d[j+dim]
            +data_d[i+dim+1]*data_d[j+dim+1]
            +data_d[i+dim+2]*data_d[j+dim+2]
            +data_d[i+dim+3]*data_d[j+dim+3]);
    }
    result_d[blockIdx.x*data_num+(blockIdx.y*blockDim.x+threadIdx.x)]=sum;
    return;
}

__global__
void IPReduce(const float* data_d, float *result_d, int data_num, int data_dim){
    unsigned long long int i=blockIdx.x*data_dim;
    unsigned long long int j=(blockIdx.y*blockDim.x+threadIdx.x)*data_dim;
    if (i<j)   return;
    float sum=0.0;
    for (int dim=0;dim<data_dim;dim+=4){
        sum+=(data_d[i+dim]*data_d[j+dim]
            +data_d[i+dim+1]*data_d[j+dim+1]
            +data_d[i+dim+2]*data_d[j+dim+2]
            +data_d[i+dim+3]*data_d[j+dim+3]);
    }
    result_d[blockIdx.x*data_num+(blockIdx.y*blockDim.x+threadIdx.x)]=sum;
    return;
}

__global__
void L2(const float* data_d, float *result_d, int *count_d, int data_num, int data_dim){
    // 分warp进行分支优化
    unsigned long long int i=blockIdx.x*data_dim;
    unsigned long long int j=(blockIdx.y*blockDim.x+threadIdx.x)*data_dim;
    // __shared__ float diff[512];
    // __shared__ float result_s[256];
    if (i<j)   return;
    float sum=0.0;
    for (int dim=0;dim<data_dim;++dim){
        if (sum>4000){
            return;
        }
        float diff=data_d[i+dim]-data_d[j+dim];
        sum+=diff*diff;
    }
    // __syncthreads();
    // result_s[threadIdx.x]=sum;
    result_d[blockIdx.x*data_num+blockIdx.y*blockDim.x+threadIdx.x]=sum;
    // count_d[blockIdx.y*blockDim.x+threadIdx.x]++;
    return;
}

// 这个函数是暴力展开一个8x8的方阵运算
__device__ __inline__
void d_rank8x8(double *C, double *A, double* B){
    double a[8],b;
    a[0]=A[0*16];
    a[1]=A[1*16];
    a[2]=A[2*16];
    a[3]=A[3*16];
    a[4]=A[4*16];
    a[5]=A[5*16];
    a[6]=A[6*16];
    a[7]=A[7*16];
    #pragma unroll
    for (int i=0;i<8;++i){
        b=B[i*16];
        C[i*8+0]+=a[0]*b;
        C[i*8+1]+=a[1]*b;
        C[i*8+2]+=a[2]*b;
        C[i*8+3]+=a[3]*b;
        C[i*8+4]+=a[4]*b;
        C[i*8+5]+=a[5]*b;
        C[i*8+6]+=a[6]*b;
        C[i*8+7]+=a[7]*b;
    }
    return;
}

__global__ 
void cuk_dgemm_unroll( double* d_C, const double* d_A, const double* __restrict__ d_B, int n, int lda, int ldb, int ldc )
{
    __shared__ double smem[2048];
    double p0, p1,p2, p3, q0, q1, q2, q3, c[64]={0.f};
    int bx, by,i, x, y, u, v, k;
    bx=blockIdx.x;
    by=blockIdx.y;
    i=threadIdx.x;
    x=i&31;
    y=i>>5;
    u=i&15;
    v=i>>4;
    d_C+=((by<<7)+(i>>7))*ldc+(i&127);
    d_A+=y*lda+(bx<<7)+x;
    d_B+=((by<<7)+u)*ldb+v;
    double*St=&smem[(y<<7)+x];
    double*At=&smem[u];
    double*Bt=&smem[1024+v];
    p0=d_A[0*32];
    p1=d_A[1*32];
    p2=d_A[2*32];
    p3=d_A[3*32];
    q0=d_B[0*32*ldb];
    q1=d_B[1*32*ldb];
    q2=d_B[2*32*ldb];
    q3=d_B[3*32*ldb];
    for( k=n-8;k>=0; k-=8 )
    {
        *(St+0*32)=p0;
        *(St+1*32)=p1;
        *(St+2*32)=p2;
        *(St+3*32)=p3;
        *(St+4*32)=q0;
        *(St+5*32)=q1;
        *(St+6*32)=q2;
        *(St+7*32)=q3;                        
        __syncthreads();                    
        if(y<k){
            d_A+=lda<<3;d_B+=8;
            p0=d_A[0*32];
            p1=d_A[1*32];
            p2=d_A[2*32];
            p3=d_A[3*32];
            q0=d_B[0*32*ldb];
            q1=d_B[1*32*ldb];
            q2=d_B[2*32*ldb];
            q3=d_B[3*32*ldb];
        }
        #pragma unroll
        for( int i=0;i<8; ++i ){
            d_rank8x8(c, &At[i*128], &Bt[i*128] );
        }__syncthreads();
    }
    #pragma unroll
    for( int i=0;i<64; i+=8 ){
        d_C[0*16]=c[i+0];
        d_C[1*16]=c[i+1];
        d_C[2*16]=c[i+2];
        d_C[3*16]=c[i+3];
        d_C[4*16]=c[i+4];
        d_C[5*16]=c[i+5];
        d_C[6*16]=c[i+6];
        d_C[7*16]=c[i+7];
        d_C+=ldc<<4;
    }
}

__global__
void L2_128_128_8(const float* data_d, float *result_d, int data_num, int data_dim){
    // 分warp进行分支优化
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;
    int k=blockIdx.z*blockDim.z+threadIdx.z;
    // __shared__ float diff[512];
    // __shared__ float result_s[256];
    // if (i<j)   return;
    float sum=result_d[i*data_num+j];
    float diff=data_d[i*data_dim+k]-data_d[j*data_dim+k];
    if (sum>4000){
        return;
    }
    sum+=diff*diff;
    result_d[i*data_num+j]=sum;
    __syncthreads();
    // result_s[threadIdx.x]=sum;
    // count_d[blockIdx.y*blockDim.x+threadIdx.x]++;
    return;
}

__global__
void L2Reduce(const float* data_d, float *result_d, int *count_d, int data_num, int data_dim){
    unsigned long long int i=blockIdx.x*data_dim;
    unsigned long long int j=blockIdx.y*data_dim;
    int tid=threadIdx.x;
    __shared__ float res[DIM];
    if (i<j)   return;
    res[tid*2]=data_d[i+tid*2]-data_d[j+tid*2];
    res[tid*2+1]=data_d[i+tid*2+1]-data_d[j+tid*2+1];
    // res[tid*2+2]=data_d[i+tid*2+2]-data_d[j+tid*2+2];
    // res[tid*2+3]=data_d[i+tid*2+3]-data_d[j+tid*2+3];
    // res[tid*2+4]=data_d[i+tid*2+4]-data_d[j+tid*2+4];
    // res[tid*2+5]=data_d[i+tid*2+5]-data_d[j+tid*2+5];
    // res[tid*2+6]=data_d[i+tid*2+6]-data_d[j+tid*2+6];
    // res[tid*2+7]=data_d[i+tid*2+7]-data_d[j+tid*2+7];
    res[tid*2]*=res[tid*2];
    res[tid*2+1]*=res[tid*2+1];
    // res[tid*2+2]*=res[tid*2+2];
    // res[tid*2+3]*=res[tid*2+3];
    // res[tid*2+4]*=res[tid*2+4];
    // res[tid*2+5]*=res[tid*2+5];
    // res[tid*2+6]*=res[tid*2+6];
    // res[tid*2+7]*=res[tid*2+7];

    __syncthreads();
    for (int stride=blockDim.x;stride>0;stride>>=1){
        if (tid<stride){
            res[tid]+=res[tid+stride];
        }
        // if (res[tid]>4000){
        //     return;
        // }
        __syncthreads();
    }
    if (tid==0){
        result_d[blockIdx.x*data_num+blockIdx.y]=res[0];
    }
    return;
}

__global__
void L2Opt(const float* data_d, float *result_d, int *count_d, int data_num, int data_dim){
    unsigned long long int data_i=blockIdx.x*DIM*BLOCK_BATCH;
    unsigned long long int data_j=blockIdx.y*DIM*BLOCK_BATCH;
    int tid=threadIdx.x;
    __shared__ float res[DIM];
    __shared__ float col[BLOCK_BATCH][DIM];
    __shared__ float rol[BLOCK_BATCH][DIM];
    int gid = tid>>5;
    int sid = tid&0x1F;
    #pragma unroll 16
    for (int i=0;i<16;++i){
        int dim=sid*16+i;
        col[gid][dim]=data_d[data_i*gid+dim];
        rol[gid][dim]=data_d[data_j*gid+dim];
    }
    __syncthreads();
    for (int i=0;i<BLOCK_BATCH;++i){
        for (int j=0;j<BLOCK_BATCH;++j){

        }
    }
    

    __syncthreads();
    for (int stride=blockDim.x;stride>0;stride>>=1){
        if (tid<stride){
            res[tid]+=res[tid+stride];
        }
        // if (res[tid]>4000){
        //     return;
        // }
        __syncthreads();
    }
    if (tid==0){
        result_d[blockIdx.x*data_num+blockIdx.y]=res[0];
    }
    return;
}

__device__ __inline__
void module(){}

__global__
void SetModuleVectorL(const float* lhs_d, float *lm, int row, int dim){
    int rid=(blockIdx.x*blockDim.x+threadIdx.x);
    float tlm=0.f;
    for (int i=0;i<dim;++i){
        float pos=lhs_d[rid*dim+i];
        tlm+=pos*pos;
    }
    lm[rid]=tlm;
    // #pragma unroll 
    // for (int i=0;i<row;++i){
    //     result_d[i*col+cid]=lm[i]+rm[cid];
    // }
    return;
}

__global__
void SetModuleVectorR(const float *rhs_d, float * rm, int col, int dim){
    int cid=(blockIdx.x*blockDim.x+threadIdx.x);
    float trm=0.f;
    for (int i=0;i<dim;++i){
        float pos=rhs_d[cid*dim+i];
        trm+=pos*pos;
    }
    rm[cid]=trm;
    // #pragma unroll 
    // for (int i=0;i<row;++i){
    //     result_d[i*col+cid]=lm[i]+rm[cid];
    // }
    return;
}

__global__
void SetModule(const float* lm, const float *rm, float *result_d, int row, int col, int dim){
    int cid=(blockIdx.x*blockDim.x+threadIdx.x);
    #pragma unroll 
    for (int i=0;i<row;++i){
        result_d[i*col+cid]=lm[i]+rm[cid];
    }
    return;
}

__global__
void CondTake(float *result_d, int row, int col){
    int cid=(blockIdx.x*blockDim.x+threadIdx.x);
    #pragma unroll 
    for (int i=0;i<row;++i){
        if (result_d[i*col+cid]>1000){
            result_d[i*col+cid]=0;
        }
    }
    return;
}

__global__
void IP_line_test(const float* device_data, int data_num, int data_dim,unsigned long long int* device_count,float *device_dis_array,int line_idx){
    unsigned long long int i=blockIdx.x*data_dim;
    unsigned long long int j=(blockIdx.y*blockDim.x+threadIdx.x)*data_dim;
    float sum=0.0;
    for (int dim=0;dim<data_dim;++dim){
        float rest=(device_data[i+dim]-device_data[j+dim]);
        sum+=rest*rest;
    }
    // device_dis_array[blockIdx.x]=sum;
    // if (blockIdx.x==line_idx){
    //     device_dis_array[blockIdx.y*blockDim.x+threadIdx.x]=sum;
    //     // __threadfence();
    // }
    // if (blockIdx.y*blockDim.x+threadIdx.x==line_idx){
    //     device_dis_array[blockIdx.x]=sum;
    //     // __threadfence();
    // }
    // __threadfence();
    // atomicAdd(device_count_array+(blockIdx.y*blockDim.x+threadIdx.x),1);
    // __syncthreads();
    // atomicAdd(device_count,1);
    // if (i==Pos && j==Pos)  printf("dis[%ld,%ld]=%f\n",i,j,sum);
    // if (sum<6000.0) sum=0.0;
    return;
}

float l2(const float* data, int data_i, int data_j, int data_dim){
    float sum=0.0;
    for (int dim=0;dim<data_dim;++dim){
        float rest=data[data_i*data_dim+dim]-data[data_j*data_dim+dim];
        sum+=rest*rest;
    }
    // printf("dis[%d,%d]=%f\n",data_i,data_j,sum);
    return sum;
}

float ip(const float* data, int data_i, int data_j, int data_dim){
    float sum=0.0;
    for (int dim=0;dim<data_dim;++dim){
        sum+=data[data_i*data_dim+dim]*data[data_j*data_dim+dim];
    }
    // printf("dis[%d,%d]=%f\n",data_i,data_j,sum);
    return sum;
}

void call_cuda(const float* data,int data_num,int data_dim,bool test=0){
    // config grid & block
    dim3 grid_dim(Grid_dim_x,Grid_dim_y,1);
    dim3 block_dim(Block_dim_x,1,1);
    printf("grid setting: (%d,%d,%d).\n",grid_dim.x,grid_dim.y,grid_dim.z);
    printf("block setting: (%d,%d,%d).\n",block_dim.x,block_dim.y,block_dim.z);
    printf("data copying to device.\n");

    // unified memory allocated
    float *data_d, *result_d;
    int * count_d;
    cudaMallocManaged((void **) &result_d, data_num*data_num*sizeof(int));
    unsigned long long int data_size=data_num*data_dim*sizeof(float);
    cudaMallocManaged((void **) &data_d, data_size);
    cudaMallocManaged((void **) &count_d, data_num*sizeof(int));
    memset(count_d,0,data_num*sizeof(int));
    memset(result_d,0,data_num*data_num*sizeof(float));
    printf("data copied to device.\n");

    memcpy(data_d,data,data_size);
    cudaMemPrefetchAsync(data_d,data_size,0,NULL);
    cudaMemPrefetchAsync(result_d,data_num*data_num*sizeof(float),0,NULL);
    printf("array ready.\n");


    L2<<<grid_dim,block_dim,256*sizeof(float)>>>(data_d,result_d,count_d,data_num,data_dim);
    cudaDeviceSynchronize();
    printf("inner product done.\n");
    
    // int *result_h=new int[data_num*Len];
    // memcpy(result_h,result_d,data_num*Len*sizeof(int));
    printf("data back to host.\n");

    if (test) {
        unsigned long long int sum=0;
        for (int i=0;i<data_num;++i){
            for (int j=0;j<data_num;++j){
                if (fabs(l2(data,i,j,data_dim)-result_d[i*data_num+j])<0.01){
                    sum++;
                }
            }
        }
        printf("sum of hit: %lld\n",sum);
    }
    


    cudaFree(data_d);
    cudaFree(result_d);
    cudaFree(count_d);
    return;
}

void call_cuda_128_128_8(const float* data,int data_num,int data_dim,bool test=0){
    // config grid & block
    dim3 grid_dim(data_num/8,data_num/8,data_dim/8);
    dim3 block_dim(8,8,8);
    printf("grid setting: (%d,%d,%d).\n",grid_dim.x,grid_dim.y,grid_dim.z);
    printf("block setting: (%d,%d,%d).\n",block_dim.x,block_dim.y,block_dim.z);
    printf("data copying to device.\n");

    // unified memory allocated
    float *data_d, *result_d;
    cudaMallocManaged((void **) &result_d, data_num*data_num*sizeof(int));
    unsigned long long int data_size=data_num*data_dim*sizeof(float);
    cudaMallocManaged((void **) &data_d, data_size);
    printf("data copied to device.\n");

    memcpy(data_d,data,data_size);
    cudaMemPrefetchAsync(data_d,data_size,0,NULL);
    cudaMemPrefetchAsync(result_d,data_num*data_num*sizeof(float),0,NULL);
    printf("array ready.\n");


    L2_128_128_8<<<grid_dim,block_dim>>>(data_d,result_d,data_num,data_dim);
    cudaDeviceSynchronize();
    printf("inner product done.\n");
    
    // int *result_h=new int[data_num*Len];
    // memcpy(result_h,result_d,data_num*Len*sizeof(int));
    printf("data back to host.\n");

    if (test) {
        unsigned long long int sum=0;
        for (int i=0;i<data_num;++i){
            for (int j=0;j<data_num;++j){
                if (fabs(l2(data,i,j,data_dim)-result_d[i*data_num+j])<0.01){
                    sum++;
                }
            }
        }
        printf("sum of hit: %lld\n",sum);
    }
    


    cudaFree(data_d);
    cudaFree(result_d);
    return;
}

void call_cuda_reduce(const float* data,int data_num,int data_dim,bool test=0){
    // config grid & block
    dim3 grid_dim(data_num,data_num,1);
    dim3 block_dim(256,1,1);
    printf("grid setting: (%d,%d,%d).\n",grid_dim.x,grid_dim.y,grid_dim.z);
    printf("block setting: (%d,%d,%d).\n",block_dim.x,block_dim.y,block_dim.z);
    printf("data copying to device.\n");

    // unified memory allocated
    float *data_d, *result_d;
    int * count_d;
    cudaMallocManaged((void **) &result_d, data_num*data_num*sizeof(int));
    unsigned long long int data_size=data_num*data_dim*sizeof(float);
    cudaMallocManaged((void **) &data_d, data_size);
    cudaMallocManaged((void **) &count_d, data_num*sizeof(int));
    memset(count_d,0,data_num*sizeof(int));
    memset(result_d,0,data_num*data_num*sizeof(float));
    printf("data copied to device.\n");

    memcpy(data_d,data,data_size);
    cudaMemPrefetchAsync(data_d,data_size,0,NULL);
    cudaMemPrefetchAsync(result_d,data_num*data_num*sizeof(float),0,NULL);
    printf("array ready.\n");


    L2Reduce<<<grid_dim,block_dim,data_dim*sizeof(float)>>>(data_d,result_d,count_d,data_num,data_dim);
    cudaDeviceSynchronize();
    printf("inner product done.\n");
    
    // int *result_h=new int[data_num*Len];
    // memcpy(result_h,result_d,data_num*Len*sizeof(int));
    printf("data back to host.\n");

    if (test) {
        unsigned long long int sum=0;
        for (int i=0;i<data_num;++i){
            for (int j=0;j<data_num;++j){
                if (fabs(l2(data,i,j,data_dim)-result_d[i*data_num+j])<0.01){
                    sum++;
                }
            }
        }
        printf("sum of hit: %lld\n",sum);
    }

    cudaFree(data_d);
    cudaFree(result_d);
    cudaFree(count_d);
    return;
}

void call_cuda_opt(const float* data,int data_num,int data_dim,bool test=0){
    // config grid & block
    dim3 grid_dim(data_num,data_num,1);
    dim3 block_dim(256,1,1);
    printf("grid setting: (%d,%d,%d).\n",grid_dim.x,grid_dim.y,grid_dim.z);
    printf("block setting: (%d,%d,%d).\n",block_dim.x,block_dim.y,block_dim.z);
    printf("data copying to device.\n");

    // unified memory allocated
    float *data_d, *result_d;
    int * count_d;
    cudaMallocManaged((void **) &result_d, data_num*data_num*sizeof(int));
    unsigned long long int data_size=data_num*data_dim*sizeof(float);
    cudaMallocManaged((void **) &data_d, data_size);
    cudaMallocManaged((void **) &count_d, data_num*sizeof(int));
    memset(count_d,0,data_num*sizeof(int));
    memset(result_d,0,data_num*data_num*sizeof(float));
    printf("data copied to device.\n");

    memcpy(data_d,data,data_size);
    cudaMemPrefetchAsync(data_d,data_size,0,NULL);
    cudaMemPrefetchAsync(result_d,data_num*data_num*sizeof(float),0,NULL);
    printf("array ready.\n");


    L2Opt<<<grid_dim,block_dim,data_dim*sizeof(float)>>>(data_d,result_d,count_d,data_num,data_dim);
    cudaDeviceSynchronize();
    printf("inner product done.\n");
    
    // int *result_h=new int[data_num*Len];
    // memcpy(result_h,result_d,data_num*Len*sizeof(int));
    printf("data back to host.\n");

    if (test) {
        unsigned long long int sum=0;
        for (int i=0;i<data_num;++i){
            for (int j=0;j<data_num;++j){
                if (fabs(l2(data,i,j,data_dim)-result_d[i*data_num+j])<0.01){
                    sum++;
                }
            }
        }
        printf("sum of hit: %lld\n",sum);
    }

    cudaFree(data_d);
    cudaFree(result_d);
    cudaFree(count_d);
    return;
}

void call_cuda_all(const float* data,int data_num,int data_dim,bool test=0){
    dim3 grid_dim(Grid_dim_x,Grid_dim_y,1);
    dim3 block_dim(Block_dim_x,1,1);

    // unified memory allocated
    printf("-------- data being allocated --------\n");
    unsigned long long int data_size=data_num*data_dim*sizeof(float);
    float *data_d, *result_d;
    int * count_d;
    cudaMallocManaged((void **) &data_d, data_size);
    cudaMallocManaged((void **) &result_d, data_num*data_num*sizeof(int));
    cudaMallocManaged((void **) &count_d, data_num*sizeof(int));
    memset(count_d,0,data_num*sizeof(int));
    memset(result_d,0,data_num*data_num*sizeof(float));
    memcpy(data_d,data,data_size);
    printf("-------- data allocated --------\n");
    printf("-------- data being prefetched --------\n");
    cudaMemPrefetchAsync(data_d,data_size,0,NULL);
    cudaMemPrefetchAsync(result_d,data_num*data_num*sizeof(float),0,NULL);
    printf("-------- data prefetched --------\n");
    

    // BF L2
    printf("-------- BF L2 start --------\n");
    grid_dim=dim3(Grid_dim_x,Grid_dim_y,1);
    block_dim=dim3(Block_dim_x,1,1);
    printf("grid setting: (%d,%d,%d).\n",grid_dim.x,grid_dim.y,grid_dim.z);
    printf("block setting: (%d,%d,%d).\n",block_dim.x,block_dim.y,block_dim.z);
    L2<<<grid_dim,block_dim,256*sizeof(float)>>>(data_d,result_d,count_d,data_num,data_dim);
    cudaDeviceSynchronize();
    printf("-------- BF L2 done --------\n");

    // simple reduce
    printf("-------- Reduce L2 start --------\n");
    grid_dim=dim3(data_num,data_num,1);
    block_dim=dim3(256,1,1);
    printf("grid setting: (%d,%d,%d).\n",grid_dim.x,grid_dim.y,grid_dim.z);
    printf("block setting: (%d,%d,%d).\n",block_dim.x,block_dim.y,block_dim.z);
    L2<<<grid_dim,block_dim,256*sizeof(float)>>>(data_d,result_d,count_d,data_num,data_dim);
    cudaDeviceSynchronize();
    printf("-------- Reduce L2 done --------\n");

    // optimized reduce
    printf("-------- OPT L2 start --------\n");
    grid_dim=dim3(data_num/BLOCK_BATCH,data_num/BLOCK_BATCH,1);
    block_dim=dim3(256,1,1);
    printf("grid setting: (%d,%d,%d).\n",grid_dim.x,grid_dim.y,grid_dim.z);
    printf("block setting: (%d,%d,%d).\n",block_dim.x,block_dim.y,block_dim.z);
    L2<<<grid_dim,block_dim,256*sizeof(float)>>>(data_d,result_d,count_d,data_num,data_dim);
    cudaDeviceSynchronize();
    printf("-------- OPT L2 done --------\n");
    
    // int *result_h=new int[data_num*Len];
    // memcpy(result_h,result_d,data_num*Len*sizeof(int));
    printf("data back to host.\n");

    // test if needed
    if (test) {
        unsigned long long int sum=0;
        for (int i=0;i<data_num;++i){
            for (int j=0;j<data_num;++j){
                if (fabs(l2(data,i,j,data_dim)-result_d[i*data_num+j])<0.01){
                    sum++;
                }
            }
        }
        printf("sum of hit: %lld\n",sum);
    }

    cudaFree(data_d);
    cudaFree(result_d);
    cudaFree(count_d);
    return;
}


void call_cublas(const float* data,int data_num,int data_dim,int flag=0){
    // unified memory allocated
    float *data_d, *result_d, *module_d;
    unsigned long long int data_size=data_num*data_dim*sizeof(float);
    cudaMallocManaged((void **) &data_d, data_size);
    cudaMallocManaged((void **) &result_d, data_num*data_num*sizeof(float));
    cudaMallocManaged((void **) &module_d, data_num*sizeof(float));
    memset(result_d,0,data_num*data_num*sizeof(float));
    memcpy(data_d,data,data_size);
    // cudaMemPrefetchAsync(data_d,data_size,0,NULL);
    cudaMemPrefetchAsync(result_d,data_num*data_num*sizeof(float),0,NULL);
    // cudaMemPrefetchAsync(module_d,data_num*sizeof(float),0,NULL);
    printf("array ready.\n");

    dim3 grid_dim(data_num/512,1,1);
    dim3 block_dim(512,1,1);
    SetModuleVectorL<<<grid_dim,block_dim>>>(data_d,module_d,data_num,data_dim);
    SetModuleVectorR<<<grid_dim,block_dim>>>(data_d,module_d,data_num,data_dim);
    SetModule<<<grid_dim,block_dim>>>(module_d,module_d,result_d,data_num,data_num,data_dim);
    cudaDeviceSynchronize();

    // cublas init
    cublasHandle_t handle;
    cublasCreate(&handle);

    float a=-2.f;
    float b=1.f;
    //C=a*opt(A)*opt(B)+b*C
    cublasSgemm(handle,
            CUBLAS_OP_T,CUBLAS_OP_N,
            data_num,data_num,data_dim,//
            &a,
            data_d,data_dim,//A
            data_d,data_dim,//B
            &b,
            NULL,data_num);
    cudaDeviceSynchronize();
    printf("cublas done.\n");


    grid_dim=dim3(data_dim/512,1,1);
    block_dim=dim3(512,1,1);
    CondTake<<<grid_dim,block_dim>>>(result_d,data_num,data_num);
    cudaDeviceSynchronize();

    if (flag==1) {
        unsigned long long int sum=0;
        for (int i=0;i<data_num;++i){
            for (int j=0;j<data_num;++j){
                float ref=l2(data,i,j,data_dim);
                float cal=result_d[i*data_num+j];
                if (fabs(ref-cal)<1){
                    sum++;
                }
                else{
                    printf("ref:%f | cal:%f\n",ref,cal);
                }
            }
        }
        printf("sum of hit: %lld\n",sum);
    }

    if (flag==2) {
        #pragma omp parallel for
        for (int i=0;i<data_num;++i){
            for (int j=0;j<data_num;++j){
                result_d[i*data_num+j]+=data[i]+data[j];
            }
        }
    }

    cudaFree(data_d);
    cudaFree(result_d);

    cublasDestroy(handle);
    return;
}

void line_test(const float* data,int data_num,int data_dim, int line_idx){
    dim3 grid_dim(Grid_dim_x,Grid_dim_y,1);
    dim3 block_dim(Block_dim_x,1,1);
    printf("grid setting: (%d,%d,%d).\n",grid_dim.x,grid_dim.y,grid_dim.z);
    printf("block setting: (%d,%d,%d).\n",block_dim.x,block_dim.y,block_dim.z);

    float * device_data=NULL;
    unsigned long long int data_size=data_num*data_dim*sizeof(float);
    cudaMalloc((void **) &device_data, data_size);
    cudaMemcpy(device_data,data,data_size,cudaMemcpyHostToDevice);
    printf("data copied to device.\n");

    float * device_dis_array=NULL;
    float * dis_array=new float[data_num];
    float * cpu_dis_array=new float[data_num];
    unsigned long long int array_size=data_num*sizeof(float);
    cudaMalloc((void **) &device_dis_array,array_size);
    printf("array ready.\n");

    unsigned long long int* device_count=NULL;
    unsigned long long int* count=new unsigned long long(0);
    cudaMalloc((void **) &device_count, sizeof(unsigned long long int));
    printf("count ready.\n");

    IP_line_test<<<grid_dim,block_dim>>>(device_data,data_num,data_dim,device_count,device_dis_array,line_idx);
    printf("gpu runnig.\n");
    for (int i=0;i<data_num;++i){
        cpu_dis_array[i]=l2(data,line_idx,i,data_dim);
    }
    // wait gpu
    cudaDeviceSynchronize();
    printf("gpu done.\n");
    printf("mul then add done.\n");

    cudaMemcpy(count,device_count,sizeof(unsigned long long int),cudaMemcpyDeviceToHost);
    cudaMemcpy(dis_array,device_dis_array,array_size,cudaMemcpyDeviceToHost);

    int diff_count=0;
    for (int i=0;i<data_num;++i){
        if (fabs(cpu_dis_array[i]-dis_array[i])>0.01){
            diff_count++;
            printf("cpu:%f | gpu:%f\n",cpu_dis_array[i],dis_array[i]);
        }
    }
    printf("diff count:%d\n",diff_count);

    printf("count: %lld\n=%lldM\n",*count,*count/MB);
    delete [] cpu_dis_array;
    delete [] dis_array;
    delete count;
    cudaFree(device_count);
    cudaFree(device_data);
    cudaFree(device_dis_array);
    return;
}


int main(){

    clock_t start_t,end_t;
    // 随机生成输入向量列表

    float * data=data_controller.get_feature();
    start_t=clock();
    Gemm(data,Num,Dim,0);
    ModuleTake();
    end_t=clock();
    double total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("cpu-gpu-cpu time: %lf s\n",total_t);
    CorrectnessTest();
    // delete [] data;

    return 0;
}