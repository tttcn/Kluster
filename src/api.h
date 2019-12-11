/*
*   整合的算法过程
*   DistanceLinker生成距离阈值
*/

#ifndef KLUSTER_API_H_
#define KLUSTER_API_H_

#include <memory>
#include <unistd.h>
#include "src/cuda/moduletake.cuh"
#include "src/cuda/gemm.cuh"
#include "src/config.h"

void DistanceLinker(const void *node_data,
                    const void *edge_data,
                    int data_num,
                    int data_dim,
                    float threshold);
                    
void Knn(const void *node_data,
         const void *edge_data,
         float threshold, int neighest_k);

#endif //KLUSTER_API_H_