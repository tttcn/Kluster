CC = g++
NVCC = nvcc
CXXFLAGS = -fopenmp -O3 -Wextra -std=c++11
CUDAFLAGS = -std=c++11 -c 
LDFLAGS = -lpthread -lcudart -lcublas -lKluster
LIBDIRS = -L/usr/local/cuda-10.1/lib64 -L../bin
INCDIRS = -I/usr/local/cuda-10.1/include -I..
BINDIR = ../bin

linker.o: linker.cc
	$(CC) $(CXXFLAGS) $(INCDIRS) linker.cc 

linker: linker.cc
	$(CC) linker.cc $(LIBDIRS) $(INCDIRS) $(LDFLAGS) $(CXXFLAGS) -o $(BINDIR)/$@