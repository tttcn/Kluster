CC = g++
NVCC = nvcc
CXXFLAGS = -fopenmp -O3 -Wextra -std=c++11 -fPIC
CUDAFLAGS = -std=c++11 -c -Xcompiler -fPIC
LDFLAGS = -lcuda -lcudart -lcublas -lcurand
LIBDIRS = -L/usr/local/cuda/lib64
INCDIRS = -I/usr/local/cuda/include -I..
BINDIR = ../bin

libKluster.so: api.o
	$(CC) $(CXXFLAGS) $(INCDIRS)  api.o cuda/*.o  $(LIBDIRS) $(LDFLAGS) -shared -o $(BINDIR)/$@

libKluster.a:
	$(NVCC) $(CUDAFLAGS) api.o cuda/*.o -o $@

api.o: api.cu api.cuh
	cd cuda; make all
	$(NVCC) $(CUDAFLAGS) $(INCDIRS) api.cu