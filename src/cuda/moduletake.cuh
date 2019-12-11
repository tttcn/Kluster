#include <cuda_runtime.h>
#include "src/config.h"

void SetModule(const float *data_d,
    float *module_d,
    int data_num,
    int data_dim);
void ModuleTake(const float *product_d,
    const float *module_base_d,
    const float *module_query_d,
    coo *output_d,
    int base_len,
    int query_len,
    float threshold);