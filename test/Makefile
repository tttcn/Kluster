CC = g++
NVCC = nvcc
CXXFLAGS = -fopenmp -O3 -Wextra -std=c++11
CUDAFLAGS = -std=c++11 -c 
LDFLAGS = -lpthread -lcudart -lcublas
LIBDIRS = -L/usr/local/cuda-10.1/lib64
INCDIRS = -I/usr/local/cuda-10.1/include,../src
BINDIR = ../bin

linker: $(BINDIR)/Kluster.so linker.cc
	$(CC) linker.cc $(BINDIR)/Kluster.so $(LIBDIRS) $(INCDIRS) $(LIBS) $(CXXFLAGS) -o $(BINDIR)/$@