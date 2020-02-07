
#include "api.h"

#include <cmath>
#include <cstring>

#include "src/config.h"
#include "src/cuda/gemm.h"
#include "src/cuda/moduletake.h"
#include "src/debug_tool.h"
#include "src/device.h"

namespace Kluster {
int DistanceLinker(const void *node_data, void *edge_data, size_t node_num,
                   size_t node_dim, size_t edge_limit, float threshold,
                   int batch_len) {

  DEBUG("data num is %lu\n", node_num);
  DEBUG("data dim is %lu\n", node_dim);
  DEBUG("threshold is %f\n", threshold);

  // 检查device状况
  Device device(CUDA, 0);
  Device device1(CUDA, 1);
  DEBUG("device checked\n");

  // 预先分配空间
  size_t node_size = node_num * node_dim * sizeof(float);
  DEBUG("data size is %lu\n", node_size);

  // _m means managed
  Matrix<Float32> node_m(node_num, node_dim, CUDA);
  memcpy(node_m.Get(), node_data, node_size);
  Matrix<Coo> edge_m(edge_limit, 1, CUDA);
  Matrix<Float32> module_m(node_num, 1, CUDA);
  Matrix<Float32> result_m(batch_len, batch_len, CUDA);
  Matrix<Coo> output_m(batch_len, batch_len, CUDA);
  Matrix<Uint32> take_num_m(1, 1, CUDA);

  DEBUG("managed vram allocated.\n");

  // memcpy(node_m, node_data,10);
  DEBUG("data moved.\n");
  // 分块计算
  // 先计算module array
  SetModule(node_m, module_m);
  // module_m = SetModule(node_m.Slice());
  DEBUG("module set.\n");

  // seg_num代表了需要分段计算的段数，每一个seg之间都要依次计算距离。
  int seg_num = node_num / batch_len;

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

  for (int lid = 0; cal_flag == 0 && lid < seg_num; ++lid) {
    for (int rid = lid; cal_flag == 0 && rid < seg_num; ++rid) {
      if (rid % 2 == 0) {
        device.Set();
      } else {
        device1.Set();
      }
      auto lhs_m = node_m.Slice(lid, lid + batch_len, 0, node_dim);
      auto rhs_m = node_m.Slice(rid, rid + batch_len, 0, node_dim);
      auto module_base_m = module_m.Slice(lid, lid + batch_len, 0, 1);
      auto module_query_m = module_m.Slice(rid, rid + batch_len, 0, 1);

      Gemm(lhs_m, rhs_m, result_m);
      // #ifdef DEBUG_
      //       gemmtest(node_m, result_m, batch_len, lid, rid);
      // #endif
      Uint32 take_num = ModuleTake(result_m, module_base_m, module_query_m,
                                   output_m, threshold);
      // #ifdef DEBUG_
      //       taketest(node_m, result_m, output_m, batch_len, *take_num_m, lid,
      //       rid);
      // #endif
      // 写回结果

      // edge_m.Append(output_m); id的偏移需要修正
      for (Uint32 edge_id = 0; cal_flag == 0 && edge_id < take_num; ++edge_id) {
        int base_id = output_m[edge_id].base_id + lid * batch_len;
        int query_id = output_m[edge_id].query_id + rid * batch_len;
        float distance = output_m[edge_id].distance;
        if (base_id < query_id) {
          edge_m[total_take_num].base_id = base_id;
          edge_m[total_take_num].query_id = query_id;
          edge_m[total_take_num].distance = distance;
          ++total_take_num;
          if (total_take_num == MAX_EDGE_NUM || total_take_num == edge_limit) {
            total_take_num = -1;
            cal_flag = 1;
          }
          // #ifdef DEBUG_
          //           if (distance < PRECISION_LIMIT) {
          //             float cpu_distance =
          //                 l2_float32(node_m, base_id, query_id, node_dim);
          //             float diff = cpu_distance - distance;
          //             if (diff > PRECISION_LIMIT * 0.01) {
          //               printf("gpu:%f |cpu:%f\n", distance, cpu_distance);
          //             }
          //           }
          // #endif
        }
      }
    }
  }
  DEBUG("total take num (edge num) is %d\n", total_take_num);

  return total_take_num;
}

void Knn(const void *node_data, const void *edge_data, float threshold,
         int neighest_k) {
  return;
}
}
