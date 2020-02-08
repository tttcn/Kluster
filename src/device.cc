#include "device.h"
#include "debug_tool.h"

namespace Kluster {

Device::Device(DeviceType device_type, DeviceId device_id) {
  // device check
  device_type_ = device_type;
  device_id_ = device_id;
}

Device::~Device() {
  // 析构时释放buffer池
}

void Device::Set() {
  if (device_type_ == CUDA) {
    CudaSetDevice(device_id_);
  }
  return;
}

// 没用，exec会先执行括号内的函数
void Device::Exec() {
  this->Set();
  return;
}

void Device::Cache(size_t cache_size){
  // DeviceCache();
  // cache_list_.pushback(Malloc(cache_size));
}

void *Device::Malloc(size_t buffer_size) {
  void *pointer = nullptr;
  if (device_type_ == CUDA) {
    CudaMallocManaged(&pointer, buffer_size);
    DEBUG("pointer address: %p\n", pointer);
  }
  return pointer;
}

void Device::Free(void *pointer) {
  if (device_type_ == CUDA) {
    CudaFree(pointer);
  }
  return;
}
}
