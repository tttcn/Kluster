#include "operator.h"

Operator::Operator(){
    for (iter in device_list){
        this->device_quque_.push();
    }
    // set batch
}

Operator::~Operator(){
    // 释放device并更改其状态
    for (iter in device_list){
        this->device_quque_.push();
    }
    // free device memory？
}

Operator::Exec(){
    // 调用device来执行
    // 按照batch来执行？
    for (iter in device_list){
        this->device_quque_.push();
    }
    // 释放device的缓存？
}