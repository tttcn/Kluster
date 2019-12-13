/*
*   用于处理csv文件
*/

#include "src/api.h"
#include "fstream"
#include "iostream"
#include <string>


int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("usage: program node_file_name edge_file_name threshold\n");
    }

    char *node_file_name=argv[1];
    char *edge_file_name=argv[2];
    char *threshold_string=argv[3];
    float * node_data;
    coo * edge_data;
    int data_num=1;
    int data_dim=1;
    float threshold;

    printf("%s\n%s\n%s\n",node_file_name,edge_file_name,threshold_string);
    
    // 打开并读取文件
    // open(node_file_name);
    // mmap(); // mmap映射数据至内存中
    // close();

    
    // int batch_len = 8192;
    // DistanceLinker(node_data, edge_data, data_num, data_dim, threshold, batch_len);

    // // 写回结果
    // open(edge_file_name);
    // write();
    // close();
    return 0;
}