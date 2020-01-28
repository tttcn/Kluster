#include "src/device.h"
#include "src/config.h"

int main(){
    DEBUG("test start\n");
    device_list
    device = Device(CUDA,0);
    device.Malloc();
    memcpy();
    ModuleTake(Gemm(device));
    device.Batch()
    return 0;
}