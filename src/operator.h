#ifndef OPERATOR_H_
#define OPERATOR_H_

/*
    Operator类负责算子的执行过程，接收n个Matrix，输出m个Matrix。
    算子的执行涉及到GPU，则会在GPU上申请空间，并负责算子过程的调度。
    应该定义为虚函数。
*/

class Operator{
private:
    batch_length_;
    queue<Device> device_queue_;
public:
    Operator();
    ~Operator();
    Exec();
    Batch();
}

#endif  // OPERATOR_H_