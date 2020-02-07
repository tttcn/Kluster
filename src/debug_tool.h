#ifndef KLUSTER_DEBUG_TOOL_H_
#define KLUSTER_DEBUG_TOOL_H_

#include <math.h>
#include <stdio.h>

#include "config.h"

namespace Kluster {
#ifdef DEBUG_
#define DEBUG(...) printf(__VA_ARGS__)
#else
#define DEBUG(format, ...)
#endif

float l2_float32(const float *data, int data_i, int data_j, int data_dim);
float dot_float32(const float *data, int data_i, int data_j, int data_dim);
int gemmtest(const float *data, float *result, int batch_len, int x, int y);
int taketest(const float *data, float *result, Coo *output, int batch_len,
             int edge_num, int lid, int rid);

// float l2_float32(const float *data, int data_i, int data_j, int data_dim) {
//   float sum = 0.0;
//   for (int dim = 0; dim < data_dim; ++dim) {
//     float rest = data[data_i * data_dim + dim] - data[data_j * data_dim + dim];
//     sum += rest * rest;
//   }
//   // printf("dis[%d,%d]=%f\n",data_i,data_j,sum);
//   return sum;
// }

// float dot_float32(const float *data, int data_i, int data_j, int data_dim) {
//   float sum = 0.0;
//   for (int dim = 0; dim < data_dim; ++dim) {
//     sum += data[data_i * data_dim + dim] * data[data_j * data_dim + dim];
//   }
//   // printf("dis[%d,%d]=%f\n",data_i,data_j,sum);
//   return sum;
// }

// int gemmtest(const float *data, float *result, int batch_len, int x, int y) {
//   int count = 0;
//   for (int i = 0; i < batch_len; ++i) {
//     for (int j = 0; j < batch_len; ++j) {
//       float diff =
//           fabs(result[i * batch_len + j] -
//                dot_float32(data, x * batch_len + i, y * batch_len + j, 20));
//       if (diff > 0.01) {
//         printf("error:[%d,%d]%f-%f\n", i, j, result[i * batch_len + j],
//                dot_float32(data, x + i, y + j, 20));
//         ++count;
//       }
//     }
//   }
//   return count;
// }

// int taketest(const float *data, float *result, Coo *output, int batch_len,
//              int edge_num, int lid, int rid) {
//   int count = 0;
//   for (int edge_id = 0; edge_id < edge_num; ++edge_id) {
//     int base_id = output[edge_id].base_id + lid * batch_len;
//     int query_id = output[edge_id].query_id + rid * batch_len;
//     float distance = output[edge_id].distance;
//     float diff = fabs(distance - l2_float32(data, base_id, query_id, 20));
//     if (diff > 0.01) {
//       printf("error:[%d,%d]%f-%f\n", base_id, query_id, distance,
//              l2_float32(data, base_id, query_id, 20));
//       ++count;
//     }
//   }
//   return count;
// }
}

#endif // KLUSTER_DEBUG_TOOL_H_