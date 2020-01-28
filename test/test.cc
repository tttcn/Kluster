/*
    测试包括：
    随机产生数据填充至Matrix结构中
    Matrix各项算子测试
    单机多卡测试
    集群测试 -- 计划中。
*/

#include "src/compute_graph.h"
#include "src/operator.h"
#include "src/matrix.h"
#include "src/device.h"

int main(){
    DEBUG("compute graph test")
    ComputeGraph compute_graph();
    compute_graph.start_point();
    compute_graph.Execute();
    compute_graph.output().show();
    return 0;
}