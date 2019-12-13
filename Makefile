CC = g++
NVCC = nvcc
CXXFLAGS = -fopenmp -O3 -Wextra -std=c++11
CUDAFLAGS = -std=c++11 -c -arch=sm_60
LDFLAGS = -lpthread -lcudart -lcublas
LIBDIRS = -L/usr/local/cuda-10.1/lib64 -L./bin
INCDIRS = -I/usr/local/cuda-10.1/include -I./src -I./test
BINDIR = ./bin

all:
	make linker

linker: Kluster
	cd test; make linker

Kluster:
	cd src; make libKluster.so

cuda:
	cd src/cuda; make all

clean:
	rm -f $(BINDIR)/*
	find . -name "*.o" | xargs rm -f
