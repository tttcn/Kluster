/*
    device是设备的抽象
    device的显存是硬资源，不能被复用，而计算能力或许可以被复用？
*/

#ifndef DEVICE_H_
#define DEVICE_H_

#include "cuda_wrapper.h"

class Device{
    DeviceType  device_type_;
    DeviceIndex device_id_;
    bool   is_occupied_;
    // buffer_list_;
public:
    Device(DeviceType device_type,DeviceIndex device_index);
    ~Device();
    Check();
    Get();
    Free();
    void * Malloc();
    Set();
}

#endif  // DEVICE_H_