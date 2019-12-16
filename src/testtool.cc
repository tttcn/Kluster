#include "testtool.h"

float l2_float32(const float *data, int data_i, int data_j, int data_dim)
{
    float sum = 0.0;
    for (int dim = 0; dim < data_dim; ++dim)
    {
        float rest = data[data_i * data_dim + dim] - data[data_j * data_dim + dim];
        sum += rest * rest;
    }
    // printf("dis[%d,%d]=%f\n",data_i,data_j,sum);
    return sum;
}
float dot_float32(const float *data, int data_i, int data_j, int data_dim)
{
    float sum = 0.0;
    for (int dim = 0; dim < data_dim; ++dim)
    {
        sum += data[data_i * data_dim + dim] * data[data_j * data_dim + dim];
    }
    // printf("dis[%d,%d]=%f\n",data_i,data_j,sum);
    return sum;
}

void gemmtest(const float *data, float *result, int batch_len, int x, int y)
{
    for (int i = 0; i < batch_len; ++i)
    {
        for (int j = 0; j < batch_len; ++j)
        {
            float diff = fabs(result[i * batch_len + j] - dot_float32(data, x * batch_len + i, y * batch_len + j, 20));
            if (diff > 0.01)
            {
                printf("error:[%d,%d]%f-%f\n", i, j, result[i * batch_len + j], dot_float32(data, x + i, y + j, 20));
                return;
            }
        }
    }
    return;
}