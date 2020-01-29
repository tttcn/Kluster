/*
    device是设备的抽象
    device的显存是硬资源，不能被复用，而计算能力或许可以被复用？
*/

#ifndef KLUSTER_DEVICE_H_
#define KLUSTER_DEVICE_H_

#include "cuda_wrapper.h"

class Device{
    DeviceType  device_type_;
    DeviceIndex device_id_;
    bool   is_occupied_;
    // buffer_list_;
public:
    Device(DeviceType device_type,DeviceIndex device_index);
    ~Device();
    // void Check();
    // void Get();
    void Set();
    void Exec();
    void Malloc(void * pointer_managed);
    void Free(void * pointer_managed);
}

#endif  // KLUSTER_DEVICE_H_