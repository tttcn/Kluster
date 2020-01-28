#include "device.h"

void * Device::Malloc(void * ptr_m,size_t buffer_size){
    this->Set();
    if (device_type_ == CUDA){
        CudaMallocManaged((void **)&ptr_m, buffer_size);
    }

    return;
}

Device::Device(DeviceType device_type, DeviceIndex device_index){
    // device check

    device_type_ = device_type;
    device_index_ = device_index;
}


Device::~Device(int id){
    // 析构时释放buffer池
}

void Device::Set(){
    if (device_type_ == CUDA){
        CudaSetDevice(device_id_);
    }
}
