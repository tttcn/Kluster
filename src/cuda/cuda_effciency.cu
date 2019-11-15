#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include "data_controller/data_controller.h"

typedef char int8;

// const unsigned long long int GB=1024*1024*1024;
const unsigned long long int MB=1024*1024;
// const float TH = 100.0;
const int Num=MB/16;
const int Dim=512;
const int Grid_dim_x=Num;
const int Block_dim_x=Dim/2;
const int Grid_dim_y=Num/Block_dim_x;
// const unsigned long long int Pos=(Num-8)*Dim;

// to be deleted
void load_data_float32(float* data,int data_num,int data_dim){
    FILE *fd=fopen("/mnt/data/tao/tr5kw/tr05/boysenberry.feature.data","rb");
    fread(data,sizeof(float),data_num*data_dim,fd);
    unsigned long long int data_size=data_num*data_dim*sizeof(float);
    printf("data loaded: %lldMB\n",data_size/MB);
    fclose(fd);
    return;
}


__global__
void IP_float32(const float* device_data, int data_num, int data_dim,unsigned long long int* device_count,volatile int *device_count_array){
    unsigned long long int i=blockIdx.x*data_dim;
    unsigned long long int j=(blockIdx.y*blockDim.x+threadIdx.x)*data_dim;
    float sum=0.0;
    for (int dim=0;dim<data_dim;++dim){
        sum+=(device_data[i+dim]*device_data[j+dim]);
    }
    // if (sum<TH) printf("bingo!");
}

__global__
void L2_float32(const float* device_data, int data_num, int data_dim,unsigned long long int* device_count,volatile int *device_count_array){
    unsigned long long int i=blockIdx.x*data_dim;
    unsigned long long int j=(blockIdx.y*blockDim.x+threadIdx.x)*data_dim;
    float sum=0.0;
    for (int dim=0;dim<data_dim;++dim){
        float rest=(device_data[i+dim]-device_data[j+dim]);
        sum+=rest*rest;
    }
    
    // device_count_array[threadIdx.x]=blockIdx.y;
    // __syncthreads();
    // __threadfence();
    // atomicAdd(device_count_array+(blockIdx.x),1);
    
    // atomicAdd(device_count_array+(blockIdx.y*blockDim.x+threadIdx.x),1);
    
    // atomicAdd(device_count,1);
    // if (i==Pos && j==Pos)  printf("dis[%ld,%ld]=%f\n",i,j,sum);
    atomicAdd(device_count,1);
    return;
}

__global__
void IP_line_test_float32(const float* device_data, int data_num, int data_dim,unsigned long long int* device_count,float *device_dis_array,int line_idx){
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

void call_cuda_float32(const float* data,int data_num,int data_dim){
    dim3 grid_dim(Grid_dim_x,Grid_dim_y,1);
    dim3 block_dim(Block_dim_x,1,1);
    printf("grid setting: (%d,%d,%d).\n",grid_dim.x,grid_dim.y,grid_dim.z);
    printf("block setting: (%d,%d,%d).\n",block_dim.x,block_dim.y,block_dim.z);
    printf("data copying to device.\n");

    float * device_data=NULL;
    unsigned long long int data_size=data_num*data_dim*sizeof(float);
    cudaMalloc((void **) &device_data, data_size);
    cudaMemcpy(device_data,data,data_size,cudaMemcpyHostToDevice);
    printf("data copied to device.\n");

    int * device_count_array=NULL;
    int * count_array=new int[data_num];
    unsigned long long int array_size=data_num*sizeof(int);
    memset(count_array,0,array_size);
    cudaMalloc((void **) &device_count_array,array_size);
    cudaMemset(device_count_array,0,array_size);
    // cudaMemcpy(device_count_array,count_array,array_size,cudaMemcpyHostToDevice);
    printf("array ready.\n");

    unsigned long long int* device_count=NULL;
    unsigned long long int* count=new unsigned long long int(0);
    cudaMalloc((void **) &device_count, sizeof(unsigned long long int));


    IP_float32<<<grid_dim,block_dim>>>(device_data,data_num,data_dim,device_count,device_count_array);
    cudaDeviceSynchronize();
    printf("inner product done.\n");
    cudaMemcpy(count,device_count,sizeof(unsigned long long int),cudaMemcpyDeviceToHost);
    cudaMemcpy(count_array,device_count_array,array_size,cudaMemcpyDeviceToHost);

    unsigned long long int sum_array=0;
    for (int i=0;i<data_num;++i){
        sum_array+=count_array[i];
    }
    printf("sum of array: %lld\n",sum_array);

    printf("count: %lld\n=%lldM\n",*count,*count/MB);
    delete [] count_array;
    delete count;
    cudaFree(device_count);
    cudaFree(device_data);
    cudaFree(device_count_array);
    return;
}


float l2_float32(const float* data, int data_i, int data_j, int data_dim){
    float sum=0.0;
    for (int dim=0;dim<data_dim;++dim){
        float rest=data[data_i*data_dim+dim]-data[data_j*data_dim+dim];
        sum+=rest*rest;
    }
    // printf("dis[%d,%d]=%f\n",data_i,data_j,sum);
    return sum;
}

void line_test_float32(const float* data,int data_num,int data_dim, int line_idx){
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

    IP_line_test_float32<<<grid_dim,block_dim>>>(device_data,data_num,data_dim,device_count,device_dis_array,line_idx);
    printf("gpu runnig.\n");
    for (int i=0;i<data_num;++i){
        cpu_dis_array[i]=l2_float32(data,line_idx,i,data_dim);
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
    // clock_t start_t,end_t;
    DataController data_controller;
    data_controller.load_feature("/mnt/data/tao/tr5kw/tr05/boysenberry.feature.data",Dim,Num);
    
    float * data_float32=data_controller.get_feature();
    // l2_float32(data_float32,Pos,Pos,Dim);
    call_cuda_float32(data_float32,Num,Dim);
    // line_test_float32(data_float32,Num,Dim,1231);
    
    delete [] data_float32;

    // int8 * data_int8=new int8[Num*Dim];W
    // load_data_int8(data_int8,Num,Dim);
    // call_cuda_int8(data_int8,Num,Dim);
    // delete [] data_int8;
    return 0;
}