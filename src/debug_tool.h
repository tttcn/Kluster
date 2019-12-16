#ifndef DEBUG_TOOL_H_
#define DEBUG_TOOL_H_

#include <stdio.h>
#include <math.h>

#include "config.h"

#ifdef DEBUG_
#define DEBUG(...) printf(__VA_ARGS__)
#else
#define DEBUG(format, ...)
#endif

float l2_float32(const float *data, int data_i, int data_j, int data_dim);
float dot_float32(const float *data, int data_i, int data_j, int data_dim);
int gemmtest(const float *data, float *result, int batch_len, int x, int y);
int taketest(const float *data, float *result, coo *output, int batch_len, int edge_num, int lid, int rid);

#endif // DEBUG_TOOL_H_