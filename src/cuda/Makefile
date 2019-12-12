CC = g++
NVCC = nvcc
CXXFLAGS = -fopenmp -O3 -Wextra -std=c++11 -fPIC
CUDAFLAGS = -std=c++11 -c -Xcompiler -fPIC
LDFLAGS = -lpthread -lcudart -lcublas
LIBDIRS = -L/usr/local/cuda-10.1/lib64
INCDIRS = -I/usr/local/cuda-10.1/include,../..
BINDIR = ../../bin

all: moduletake.o gemm.o

gemm.o: gemm.cu
	$(NVCC) $(CUDAFLAGS) gemm.cu 

moduletake.o: moduletake.cu
	$(NVCC) $(CUDAFLAGS) moduletake.cu $(INCDIRS)
	