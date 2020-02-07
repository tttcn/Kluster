/*
*   整合的算法过程
*   DistanceLinker生成基于距离阈值的边表
*/

#ifndef KLUSTER_API_H_
#define KLUSTER_API_H_

#include "src/config.h"

int DistanceLinker(const void *node_data,
                    void *edge_data,
                    size_t data_num,
                    size_t data_dim,
                    size_t edge_limit,
                    float threshold,
                    int batch_len);

void Knn(const void *node_data,
         const void *edge_data,
         float threshold, int neighest_k);

#endif //KLUSTER_API_H_