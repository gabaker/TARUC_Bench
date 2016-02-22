CC=nvcc
BENCH_FILES=CUDA_Bandwidth.cu
BENCH_EXE=bench
CFLAGS=-lpthread -lhwloc -Xcompiler -fopenmp -lnuma  -std=c++11
IFLAGS=-I/usr/local/cuda-7.5/samples/common/inc

all: bench

topo.o: topology.cpp
	$(CC) $(IFLAGS) $(CFLAGS) -c topology.cpp -o topo.o

params.o: params.cpp
	$(CC) $(IFLAGS) $(CFLAGS) -c params.cpp -o params.o

bench: $(BENCH_FILES) topo.o params.o
	$(CC) -D USING_CPP $(IFLAGS) $(CFLAGS) params.o topo.o $(BENCH_FILES) -o $(BENCH_EXE)

nocpp:  $(BENCH_FILES) topo.o params.o
	$(CC) $(IFLAGS) $(CFLAGS) params.o topo.o $(BENCH_FILES) -o $(BENCH_EXE)

clean:
	rm  *.o $(BENCH_EXE) $(NUMA_EXE)
