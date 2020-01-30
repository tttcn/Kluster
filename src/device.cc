#include "device.h"

Device::Device(DeviceType device_type, DeviceIndex device_index){
    // device check
    device_type_ = device_type;
    device_index_ = device_index;
}


Device::~Device(){
    // 析构时释放buffer池
}

void Device::Set(){
    if (device_type_ == CUDA){
        CudaSetDevice(device_index_);
    }
    return;
}

// 没用，exec会先执行括号内的函数
void Device::Exec(){
    this->Set();
    return;
}

void * Device::Malloc(size_t buffer_size){
    void * pointer=NULL;
    if (device_type_ == CUDA){
        CudaMallocManaged(&pointer, buffer_size);
        DEBUG("pointer address: %p\n",pointer);
    }
    return pointer;
}

void Device::Free(void * pointer){
    if (device_type_ == CUDA){
        CudaFree(pointer);
    }
    return;
}




