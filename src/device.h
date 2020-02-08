/*
    device是设备的抽象
    device的显存是硬资源，不能被复用，而计算能力或许可以被复用？
*/

#ifndef KLUSTER_DEVICE_H_
#define KLUSTER_DEVICE_H_

#include "cuda_wrapper.h"
#include "debug_tool.h"

namespace Kluster {
class Device {
  DeviceType device_type_;
  DeviceId device_id_;
  // bool   is_occupied_;
public:
  Device(DeviceType device_type, DeviceId device_id);
  ~Device();
  // void Check();
  // void Get();
  void Set();
  void Exec();
  void Cache(size_t cache_size);
  void *Malloc(size_t buffer_size);
  void Free(void *pointer);
};

// Device::Device(DeviceType device_type, DeviceId device_id) {
//   // device check
//   device_type_ = device_type;
//   device_id_ = device_id;
// }

// Device::~Device() {
//   // 析构时释放buffer池
// }

// void Device::Set() {
//   if (device_type_ == CUDA) {
//     CudaSetDevice(device_id_);
//   }
//   return;
// }

// // 没用，exec会先执行括号内的函数
// void Device::Exec() {
//   this->Set();
//   return;
// }

// void *Device::Malloc(size_t buffer_size) {
//   void *pointer = nullptr;
//   if (device_type_ == CUDA) {
//     CudaMallocManaged(&pointer, buffer_size);
//     DEBUG("pointer address: %p\n", pointer);
//   }
//   return pointer;
// }

// void Device::Free(void *pointer) {
//   if (device_type_ == CUDA) {
//     CudaFree(pointer);
//   }
//   return;
// }
}

#endif // KLUSTER_DEVICE_H_