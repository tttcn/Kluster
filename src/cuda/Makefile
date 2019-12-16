CC = g++
NVCC = nvcc
CXXFLAGS = -fopenmp -O3 -Wextra -std=c++11 -fPIC
CUDAFLAGS = -std=c++11 -c -Xcompiler -fPIC
LDFLAGS = -lcuda -lcudart -lcublas -lcurand
LIBDIRS = -L/usr/local/cuda/lib64
INCDIRS = -I/usr/local/cuda/include -I../..
BINDIR = ../../bin

all: moduletake.o gemm.o

gemm.o: gemm.cu
	$(NVCC) $(CUDAFLAGS) gemm.cu 

moduletake.o: moduletake.cu
	$(NVCC) $(CUDAFLAGS) $(INCDIRS) moduletake.cu 
	