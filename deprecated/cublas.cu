/*
*   使用cuda的官方库所实现的距离计算方法，
*   cuBLAS实现的gemm效率极高，经过了汇编指令优化，几乎可以完全发挥出gpu的全部算力，唯一可能不够好的地方可能是没有做strassen算法优化。
*   --------------------------------------------------------------------------------------------------------------- 
*   编译说明：
*   使用cuBLAS，需要在nvcc编译时加上"-lcublas"选项
*   使用cuSparse，需要在nvcc编译时加上"-lsparse"选项
*   使用OpenMP，需要在gcc编译时加上"-fopenmp"；或者使用nvcc编译，使用"-Xcompiler -fopenmp"将"-fopenmp"参数直接传递给gcc。
*/
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>

// const unsigned long long int GB=1024*1024*1024;
const unsigned long long int MB=1024*1024;
// const float TH = 100.0;
const int Num=1024*1024;
const int Dim=512;
const int Batch=1024*32;
const int Grid_dim_x=Num;
const int Block_dim_x=Dim/2;
const int Grid_dim_y=Num/Block_dim_x;
const int Batch_TH=Batch*Batch;

struct coo{
    int row_id;
    int col_id;
    float distance;
    float for_align;
};

// 复杂度O(nd)，1M数据400ms左右，只要执行一次，优化意义不大
__global__
void SetModuleArray(const float* data_d, float *module_d, int data_num, int data_dim){
    int tid=(blockIdx.x*blockDim.x+threadIdx.x);
    float tmodule=0.f;
    for (int i=0;i<data_dim;++i){
        float pos=data_d[tid*data_dim+i];
        tmodule+=pos*pos;
    }
    module_d[tid]=tmodule;
    return;
}

// 复杂度O(n^2)，1M数据10s左右，耗时比较稳定，与batch size没有太大关系，优化意义较大。
__global__
void SetModule(const float* lm, const float *rm, float *result_d, int row, int col){
    int cid=(blockIdx.x*blockDim.x+threadIdx.x);
    #pragma unroll 
    for (int i=0;i<row;++i){
        result_d[i*col+cid]=lm[i]+rm[cid];
    }
    return;
}

__global__
void SetModuleSingle(const float* lm, const float *rm, float *result_d, int row, int col){
    int rid=blockIdx.x;
    int cid=(blockIdx.y*blockDim.x+threadIdx.x);
    result_d[rid*col+cid]=lm[rid]+rm[cid];
    return;
}

// 复杂度O(n^2)，1M数据5s左右，与batch size有关，更大的batch size时间更短，应该是掩盖了访存延迟，优化意义较大。
__global__
void CondTake(float *result_d, coo* output_d,unsigned int *condtake_num_d,int row, int col){
    int cid=(blockIdx.x*blockDim.x+threadIdx.x);
    for (int i=0;i<row;++i){
        if (i==cid) continue;
        float dis=result_d[i*col+cid];
        if (dis<4000){
            int output_id= atomicAdd(condtake_num_d,1);
            output_d[output_id].row_id=i;
            output_d[output_id].col_id=cid;
            output_d[output_id].distance=dis;
            // result_d[i*col+cid]=0;
        }
    }
    return;
}

__global__
void CondTakeSingle(float *result_d, coo* output_d,unsigned int *condtake_num_d,int row, int col){
    int rid=blockIdx.x;
    int cid=(blockIdx.y*blockDim.x+threadIdx.x);
    float dis=result_d[rid*col+cid];
    if (dis<4000){
        int output_id= atomicAdd(condtake_num_d,1);
        output_d[output_id].row_id=rid;
        output_d[output_id].col_id=cid;
        output_d[output_id].distance=dis;
    }
    return;
}

float l2(const float* lhs_d,const float *rhs_d, int data_i, int data_j, int data_dim){
    float sum=0.0;
    for (int dim=0;dim<data_dim;++dim){
        float rest=lhs_d[data_i*data_dim+dim]-rhs_d[data_j*data_dim+dim];
        sum+=rest*rest;
    }
    // printf("dis[%d,%d]=%f\n",data_i,data_j,sum);
    return sum;
}

float ip(const float* lhs_d,const float *rhs_d, int data_i, int data_j, int data_dim){
    float sum=0.0;
    for (int dim=0;dim<data_dim;++dim){
        sum+=lhs_d[data_i*data_dim+dim]*rhs_d[data_j*data_dim+dim];
    }
    // printf("dis[%d,%d]=%f\n",data_i,data_j,sum);
    return sum;
}

void condtake(float *result_d, int row, int col){
    int condtake_num=0;
    for (int i=0;i<row;++i){
        for(int j=0;j<col;++j){
            if (result_d[i*col+j]<4000){
                condtake_num++;
            }
        }
    }
    printf("cpu condtake:%d\n",condtake_num);
    return;
}

void set_module(const float* lm, const float *rm, float *result_d, int row, int col){
    #pragma omp parallel for
    for (int i=0;i<row;++i){
        for (int j=0;j<col;++j){
            result_d[i*col+j]=lm[i]+rm[j];
        }
    }
    return;
}

void set_module_array(const float *data_d,float *module_d,int data_num,int data_dim){
    dim3 grid_dim(data_num/512,1,1);
    dim3 block_dim(512,1,1);
    SetModuleArray<<<grid_dim,block_dim>>>(data_d,module_d,data_num,data_dim);
    cudaDeviceSynchronize();
}

void do_batch(const float *data_d, const float* module_d, int lid, int rid,
    float *result_d, int batch_num, int data_dim, int flag=0){
    const float *lhs_d=data_d+lid*batch_num*data_dim;
    const float *rhs_d=data_d+rid*batch_num*data_dim;
    const float *lmod_d=module_d+lid*batch_num;
    const float *rmod_d=module_d+rid*batch_num;

    // setmodule(lmod_d,rmod_d,result_d,batch_num,batch_num);
    dim3 grid_dim(batch_num,batch_num/512,1);
    dim3 block_dim(512,1,1);
    // SetModule<<<grid_dim,block_dim>>>(lmod_d,rmod_d,result_d,batch_num,batch_num);
    SetModuleSingle<<<grid_dim,block_dim>>>(lmod_d,rmod_d,result_d,batch_num,batch_num);
    cudaDeviceSynchronize();

    // cublas init
    cublasHandle_t handle;
    cublasCreate(&handle);

    float a=-2.f;
    float b=1.f;
    //C=a*opt(A)*opt(B)+b*C
    // result = -2*trans(rhs)*lhs + mod_mat
    cublasSgemm(handle,
            CUBLAS_OP_T,CUBLAS_OP_N,
            batch_num,batch_num,data_dim,//
            &a,
            rhs_d,data_dim,//A
            lhs_d,data_dim,//B
            &b,
            result_d,batch_num);
    cudaDeviceSynchronize();
    

    if (flag==1) {
        printf("-------------- test inner product -----------\n");
        unsigned long long int hit=0;
        unsigned long long int not_hit=0;
        for (int i=0;i<batch_num;++i){
            for (int j=0;j<batch_num;++j){
                float ref=l2(lhs_d,rhs_d,i,j,data_dim);
                float cal=result_d[i*batch_num+j];
                if (fabs(ref-cal)<0.1){
                    hit++;
                }
                else{
                    // printf("ref:%f | cal:%f\n",ref,cal);
                    // printf("i=%d,j=%d\n",lid,rid);
                    not_hit++;
                }
            }
        }
        printf("sum of hit: %lld\nnot hit:%lld\n",hit,not_hit);
    }
    
    if (flag==2) {
        printf("-------------- test L2 distance -----------\n");
        unsigned long long int hit=0;
        unsigned long long int not_hit=0;
        for (int i=0;i<batch_num;++i){
            for (int j=0;j<batch_num;++j){
                float ref=l2(lhs_d,rhs_d,i,j,data_dim);
                float cal=result_d[i*batch_num+j];
                if (fabs(ref-cal)<0.1){
                    hit++;
                }
                else{
                    // printf("ref:%f | cal:%f\n",ref,cal);
                    // printf("i=%d,j=%d\n",lid,rid);
                    not_hit++;
                }
            }
        }
        printf("sum of hit: %lld\nnot hit:%lld\n",hit,not_hit);
    }

    cublasDestroy(handle);
    return;
}

unsigned long long int cond_take(float *result_d, coo* output_d, unsigned int * condtake_num_d,
    int batch_num,int flag=0){
    dim3 grid_dim(batch_num,batch_num/512,1);
    dim3 block_dim(512,1,1);
    *condtake_num_d=0;
    // printf("batch %d,%d: condtake init:%d.\n",lid,rid,*condtake_num_d);
    CondTakeSingle<<<grid_dim,block_dim>>>(result_d,output_d,condtake_num_d,batch_num,batch_num);
    cudaDeviceSynchronize();

    if (flag==1){
        printf("------- test condtake ---------\n");
        printf("gpu condtake: %d\n",*condtake_num_d);
        condtake(result_d,batch_num,batch_num);
    }
    return *condtake_num_d;
    
}

void call_cublas(const float* data,int data_num,int data_dim,int batch_num,int device=0,int flag=0){
    cudaSetDevice(device);

    // seg_num代表了需要分段计算的段数，每一个seg之间都要依次计算距离。
    int seg_num=data_num/batch_num;
    printf("batch len is %d\n",batch_num);

    // 预先分配空间
    float *data_d, *result_d, *module_d;
    coo * output_d;
    unsigned int * condtake_num_d;
    unsigned long long int data_size=data_num*data_dim*sizeof(float);
    cudaMallocManaged((void **) &data_d, data_size);
    memcpy(data_d,data,data_size);
    // cudaMemPrefetchAsync(data_d,data_size,0,NULL);

    cudaMallocManaged((void **) &module_d, data_num*sizeof(float));
    // cudaMemPrefetchAsync(module_d,data_num*sizeof(float),0,NULL);

    cudaMallocManaged((void **) &result_d, batch_num*batch_num*sizeof(float));    
    // cudaMemPrefetchAsync(result_d,batch_num*batch_num*sizeof(float),0,NULL);

    cudaMallocManaged((void **) &output_d, batch_num*batch_num*sizeof(coo));    
    // cudaMemPrefetchAsync(output_d,batch_num*batch_num*sizeof(coo),0,NULL);

    
    cudaMallocManaged((void **) &condtake_num_d, sizeof(int)); 
    // cudaMemPrefetchAsync(condtake_num_d,sizeof(int),0,NULL);
    
    printf("pre-allocation done.\n");

    // 先计算module array
    set_module_array(data_d,module_d,data_num,data_dim);
    

    // 循环计算各个seg
    unsigned long long int total_condtake_num;
    for (int i=0;i<seg_num;++i){
        for (int j=i;j<seg_num;++j){
            // if (i<j) break;
            do_batch(data_d,module_d,i,j,result_d,batch_num,data_dim);
            total_condtake_num+=cond_take(result_d,output_d,condtake_num_d,batch_num);
        }
    }
    printf("condtake num:%u\n",*condtake_num_d);
    printf("total condtake num:%lld\n",total_condtake_num);

    // 回收空间
    cudaFree(data_d);
    cudaFree(result_d);
    cudaFree(module_d);
    cudaFree(output_d);
    return;
}



int main(){
    clock_t start_t,end_t;
    DataController data_controller;
    data_controller.load_feature("/mnt/data/tao/tr5kw/tr05/boysenberry.feature.data",Dim,Num);
    
    float * data=data_controller.get_feature();
    start_t=clock();
    call_cublas(data,Num,Dim,Batch);
    end_t=clock();
    double total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("cpu-gpu-cpu time: %lf s\n",total_t);

    return 0;
}