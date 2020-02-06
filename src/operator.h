#ifndef KLUSTER_OPERATOR_H_
#define KLUSTER_OPERATOR_H_

/*
    Operator类负责算子的执行过程，接收n个Matrix，输出m个Matrix。
    算子的执行涉及到GPU，则会在GPU上申请空间，并负责算子过程的调度。
    类中函数应该定义为虚函数。
    目前暂不考虑用Operator类，因为构建函数时开销很大，不如计算图形式或者直接函数调用。
*/

class Operator {

public:
  virtual Operator();
  virtual ~Operator();
  checkParamters();

  Exec();
  Batch();

private:
  batch_length_;
}

#endif // KLUSTER_OPERATOR_H_