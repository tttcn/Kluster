CXX = g++
NVCC = nvcc
CXXFLAGS = -fopenmp -O3 -Wextra -std=c++11 -fPIC
CUDAFLAGS = -std=c++11 -c -Xcompiler -fPIC
LDFLAGS = -lcuda -lcudart -lcublas -lcurand
LIBDIRS = -L/usr/local/cuda/lib64
INCDIRS = -I/usr/local/cuda/include -I..
BINDIR = ../bin


libKluster.so: debug_tool.o cuda_wrapper.o device.o api.o
	$(CXX) $(CXXFLAGS) $(INCDIRS)  *.o cuda/*.o  $(LIBDIRS) $(LDFLAGS) -shared -o $(BINDIR)/$@

debug: api.h api.cc
	cd cuda; make all
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c -o api.o api.cc -D DEBUG_ -g

cuda_wrapper.o: cuda_wrapper.h cuda_wrapper.cu
	$(NVCC) $(CUDAFLAGS) $(INCDIRS) cuda_wrapper.cu

device.o: device.h device.cc
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c -o $@ device.cc

api.o: api.h api.cc
	cd cuda; make all
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c -o $@ api.cc