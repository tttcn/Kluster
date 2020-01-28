/*
    device是设备的抽象
    device的显存是硬资源，不能被复用，而计算能力或许可以被复用？
*/

#ifndef DEVICE_H_
#define DEVICE_H_

class Device{
    DeviceType device_type;
    DeviceId;
    bool is_occupied_;
public:
    Device();
    ~Device();
    Check();
    Get();
    Free();
}

#endif  // DEVICE_H_