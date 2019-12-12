/* 
*   算子的集中声明。只需引用本文件即可。
*   
*/


void Gemm(const void *data_h, const int data_type,
          int data_num, int data_dim, int batch_num,
          int device_type = 0);

// 这个算子列出来，但是不暴露在对外API中，因为必须要？？？？
void ModuleTake(const float *product_h, const float *base_module_h, const float *query_module_h,
                int data_num, int data_dim, float threshold);