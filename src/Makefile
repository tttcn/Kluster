CC = g++
NVCC = nvcc
CXXFLAGS = -fopenmp -O3 -Wextra -std=c++11 -fPIC
CUDAFLAGS = -std=c++11 -c -Xcompiler -fPIC
LDFLAGS = -lpthread -lcudart -lcublas
LIBDIRS = -L/usr/local/cuda-10.1/lib64
INCDIRS = -I/usr/local/cuda-10.1/include,..
BINDIR = ../bin

Kluster.so: api.o
	cd cuda; make all
	$(CC) $(CXXFLAGS) api.o cuda/*.o -shared -o $@
	mv $@ $(BINDIR)/$@

Kluster.a:
	$(NVCC) $(CUDAFLAGS) api.o cuda/*.o -o $@

api.o: api.cc
	$(NVCC) $(CUDAFLAGS) api.cc $(INCDIRS)