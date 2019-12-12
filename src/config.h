#ifndef KLUSTER_CONFIG_H_
#define KLUSTER_CONFIG_H_

struct coo
{
    int base_id;
    int query_id;
    float distance;
    float for_align;
};

// const unsigned long long int GB=1024*1024*1024;
const unsigned long long int MB = 1024 * 1024;
// const float TH = 100.0;
const int Num = 1024 * 1024;
const int Dim = 512;
const int Batch = 1024 * 32;
const int Grid_dim_x = Num;
const int Block_dim_x = Dim / 2;
const int Grid_dim_y = Num / Block_dim_x;
const int Batch_TH = Batch * Batch;

#endif // KLUSTER_CONFIG_H_