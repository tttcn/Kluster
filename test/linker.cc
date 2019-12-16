/*
*   用于处理csv文件
*   数据格式：idx,v1,...,v20
*/

#include "src/api.cuh"
#include "fstream"
#include "iostream"
#include <string>
#include <omp.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        printf("usage: program node_file_name edge_file_name data_num data_dim threshold\n");
    }

    char *node_file_name = argv[1];
    char *edge_file_name = argv[2];
    char *data_num_string = argv[3];
    char *data_dim_string = argv[4];
    char *threshold_string = argv[5];
    int data_num = atoi(data_num_string);
    int data_dim = atoi(data_dim_string);
    float threshold = atof(threshold_string);

    int edge_num = MAX_EDGE_NUM;
    coo *edge_data = new coo[edge_num];

    // printf("%d\n%d\n%f\n", data_num, data_dim, threshold);

    // 打开并读取文件
    int fd_node = open(node_file_name, O_RDONLY);
    unsigned int node_file_size = data_num * data_dim * sizeof(float);
    float *node_data = (float *)mmap(NULL, node_file_size, PROT_READ, MAP_SHARED, fd_node, 0); // mmap映射数据至内存中

    // // 读取测试
    // for (int line_id = 0; line_id < 10; ++line_id)
    // {
    //     for (int element_id = 0; element_id < data_dim; ++element_id)
    //     {
    //         printf("%f,", node_data[line_id * data_dim + element_id]);
    //     }
    //     printf("\n");
    // }
    // printf("ready to run\n");

    int batch_len = 16384;
    edge_num = DistanceLinker(node_data, edge_data, data_num, data_dim, threshold, batch_len);
    printf("linked\n");

    // 写回结果
    if (edge_num == -1) // 阈值太高导致取得边数超过了容纳上限
    {
        printf("edge num overflow, lower your threshold!\n");
    }
    else
    {
        FILE *edge_file = fopen(edge_file_name, "w");
        fprintf(edge_file, "node1_position,node2_position,distance\n");
        for (int edge_id = 0; edge_id < edge_num; ++edge_id)
        {
            fprintf(edge_file, "%d,%d,%f\n", edge_data[edge_id].base_id, edge_data[edge_id].query_id, edge_data[edge_id].distance);
        }
        fclose(edge_file);
        // int fd_edge = open(edge_file_name, O_RDWR|O_CREAT);
        // close(fd_edge);
    }

    delete[] edge_data;

    munmap(node_data, node_file_size); // 解除映射关系
    close(fd_node);

    return 0;
}