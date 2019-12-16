CC = g++
NVCC = nvcc
CXXFLAGS = -fopenmp -O3 -Wextra -std=c++11
CUDAFLAGS = -std=c++11 -c
LDFLAGS = -lcuda -lcudart -lcublas -lcurand
LIBDIRS = -L/usr/local/cuda/lib64 -L./bin
INCDIRS = -I/usr/local/cuda/include -I./src -I./test
BINDIR = ./bin


.PHONY: test clean prof data check

all:
	make linker
	make test

check:
	cd data; python check.py

test:
	cd bin; ./linker ../data/eco_nodes ../data/eco_edges.csv 65536 20 1.0

prof:
	cd bin; nvprof ./linker ../data/eco_nodes ../data/eco_edges.csv 822802 20 1000

linker: Kluster
	cd test; make linker

dlinker: debug
	cd test; make linker

debug:
	cd src; make debug; make libKluster.so

Kluster:
	cd src; make libKluster.so

cuda:
	cd src/cuda; make all

clean:
	rm -f $(BINDIR)/*
	find . -name "*.o" | xargs rm -f

basic:
	cd test;nvcc test.cu -o ../bin/basic; nvprof ../bin/basic

data:
	cd test; python csv2bin.py