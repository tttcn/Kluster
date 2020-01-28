# Kluster
一个和参数K有关的cluster底层算法库。
> 目前正在重构代码

## 用法
在Kluster目录下执行一系列make指令即可
- `make Kluster` 将会在Kluster/bin文件夹中生成libKluster.so文件
- `make debug` 将会在Kluster/bin文件夹中生成libKluster.so文件，附加了一些debug信息，只计算3x3=9个batch之间的运算，有cpu和gpu运算的检测

## 依赖
算法依赖cuda库，目前包括
- libcudart.so: CUDA runtime lib
- libcublas.so: cuBLAS lib，英伟达官方基于CUDA的BLAS库

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
