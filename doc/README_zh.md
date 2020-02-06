# Kluster
一个和参数K有关的cluster底层算法库。编译为动态库后引入头文件即可使用。

## 编译
在Kluster目录下执行一系列make指令即可
- `make Kluster` 将会在Kluster/bin文件夹中生成libKluster.so文件
- `make debug` 将会在Kluster/bin文件夹中生成libKluster.so文件，附加了一些debug信息，只计算3x3=9个batch之间的运算，有cpu和gpu运算的检测

## 依赖
算法依赖cuda库，目前包括
- libcudart.so: CUDA runtime lib
- libcublas.so: cuBLAS lib，英伟达官方基于CUDA的BLAS库

## 特性
### 编程者须知
- 数据：如何将数据转化为数据管理类Matrix。
- 操作：各算子的作用，以及如何使用算子来完成任务。
### 隐藏的实现
- 显存和内存的调度
- 设备信息
- 多设备的扩展
- [ ] 可能存在的批处理？
### 如何开发新的算子
cuda核，封装Matrix解析
### 显存内存隐藏调度
cuda的内存调度是隐藏的，相比于GPU全局内存的稀少，内存或许是更多的，在Matrix初始化时保存的是CPU中的内存地址，
只有在调用slice方法的时候才会转到申请UM，是否进行
### 统一内存寻址 Unified Memory
所有的CUDA设备内存管理都使用了统一内存寻址方式管理，支持按需数据传输、多设备透明以及超量内存订阅功能。
> 详情可参考：[统一内存参考资料](http://lianguanglei.com/2019/06/11/CUDA%E7%BB%9F%E4%B8%80%E5%86%85%E5%AD%98UVA/)

## 代码
文件可以分为几个层次：
- Kluster：顶层文件夹，包含总体的Makefile
    - bin：生成的库文件和二进制程序都在这里
    - data：存放数据
    - test：测试有关的源文件和Makefile
    - src：包含有cuda文件夹以及一些C/C++文件，有一些封装了cuda算子的算法，一些配置文件以及Makefile
        - cuda：cuda算子的实现 


## 已知问题
- cuda浮点计算存在误差，不过一般可以接受，在靠近0附近比较敏感，甚至可能出现差平方小于零的情况。
