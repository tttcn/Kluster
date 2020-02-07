/*
    device是设备的抽象
    device的显存是硬资源，不能被复用，而计算能力或许可以被复用？
*/

#ifndef KLUSTER_DEVICE_H_
#define KLUSTER_DEVICE_H_

#include "cuda_wrapper.h"

class Device{
    DeviceType  device_type_;
    DeviceId device_id_;
    // bool   is_occupied_;
public:
    Device(DeviceType device_type, DeviceId device_id);
    ~Device();
    // void Check();
    // void Get();
    void Set();
    void Exec();
    void * Malloc(size_t buffer_size);
    void Free(void * pointer);
};

#endif  // KLUSTER_DEVICE_H_