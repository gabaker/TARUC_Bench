CC=nvcc
BENCH_FILES=CUDA_Bandwidth.cu
NUMA_FILES=NUMA_Test.cu
BENCH_EXE=bench
NUMA_EXE=numa
CFLAGS=-lpthread -Xcompiler -fopenmp -lnuma  -std=c++11
DFLAGS=-D USING_CPP
IFLAGS=-I/usr/local/cuda-7.5/samples/common/inc

all: bench

bench: $(BENCH_FILES) $(NUMA_FILES)
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(BENCH_FILES) -o $(BENCH_EXE)
	$(CC) $(IFLAGS) $(CFLAGS) $(NUMA_FILES) -o $(NUMA_EXE)

numa:
	$(CC) $(IFLAGS) $(CFLAGS) $(NUMA_FILES) -o $(NUMA_EXE)

nocpp:
	$(CC) $(IFLAGS) $(CFLAGS) $(BENCH_FILES) -o $(BENCH_EXE)

cpp:
	$(CC) $(IFLAGS) $(DFLAGS) $(CFLAGS) $(BENCH_FILES) -o $(BENCH_EXE)

clean:
	rm $(BENCH_EXE) $(NUMA_EXE)
