CC=nvcc
BENCH_FILES=benchmark.cu
BENCH_EXE=run
CFLAGS=-lpthread -lhwloc -Xcompiler -fopenmp -lnuma  -std=c++11
IFLAGS=-I/usr/local/cuda-7.5/samples/common/inc
OFILES=topology.o parameters.o

all: bench

topology.o: topology.cu
	$(CC) $(CFLAGS) -c topology.cu -o topology.o

parameters.o: parameters.cpp
	$(CC) $(CFLAGS) -c parameters.cpp -o parameters.o

bench: $(BENCH_FILES) $(OFILES)
	$(CC) -D USING_CPP $(CFLAGS) $(OFILES) $(BENCH_FILES) -o $(BENCH_EXE)

nocpp:  $(BENCH_FILES) $(OFILES)
	$(CC) $(CFLAGS) $(OFILES) $(BENCH_FILES) -o $(BENCH_EXE)

clean:
	rm  *.o $(BENCH_EXE) $(NUMA_EXE)
