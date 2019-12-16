CC = g++
NVCC = nvcc
CXXFLAGS = -fopenmp -O3 -std=c++11 -fPIC -Wextra
CUDAFLAGS = -std=c++11 -c 
LDFLAGS = -lKluster
LIBDIRS = -L/usr/local/cuda/lib64 -L../bin
INCDIRS = -I/usr/local/cuda/include -I..
BINDIR = ../bin

linker.o: linker.cc
	$(CC) $(CXXFLAGS) $(INCDIRS) linker.cc -o linker.o

linker: linker.cc
	$(CC) $(CXXFLAGS) $(INCDIRS) linker.cc -o $(BINDIR)/$@ $(LIBDIRS) $(LDFLAGS)