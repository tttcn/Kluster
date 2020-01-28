#ifndef COMPUTE_GRAPH_H_
#define COMPUTE_GRAPH_H_

/*
    Operator类负责算子的执行过程，接收n个Matrix，输出m个Matrix。
    算子的执行涉及到GPU，则会在GPU上申请空间，并负责算子过程的调度。
    应该定义为虚函数。
*/

class ComputeNode{
private:
    input_1;
    input_2;
    output_1;
    Operator operator_;
public:
    ComputeNode();
    ~ComputeNode();
    Exec();
}

class ComputeGraph{
private:
    ComputeNode top_node_;
public:
    ComputeGraph();
    ~ComputeGraph();
    Exec();
}



#endif  // COMPUTE_GRAPH_H_