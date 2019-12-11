/*
*   用于处理csv文件
*/

#include "src/api.h"
#include "fstream"
#include "iostream"

int main(int argv, char **argc[])
{
    if (argv!=4){
        printf("usage: program node_file_name edge_file_name threshold\n");
    }

    

    char *node_file_name;
    char *edge_file_name;
    // 打开并读取文件
    open(node_file_name);
    mmap(); // mmap映射数据至内存中
    close();

    DataLinker(node_data,edge_data,threshold);

    // 写回结果
    open(edge_file_name);
    write();
    close();
}