CC=nvcc
BENCH_FILES=benchmark.cu
BENCH_EXE=run
O_FILES=topology.o parameters.o
LDFLAGS=-lpthread -lhwloc -Xcompiler -fopenmp -lnuma -std=c++11 -L/usr/src/gdk/nvml/lib/ -lnvidia-ml

all: bench

topology.o: topology.cu
	$(CC) $(LDFLAGS) -c topology.cu -o topology.o

parameters.o: parameters.cpp
	$(CC) $(LDFLAGS) -c parameters.cpp -o parameters.o

bench: $(BENCH_FILES) $(O_FILES)
	$(CC) -D USING_CPP $(LDFLAGS) $(O_FILES) $(BENCH_FILES) -o $(BENCH_EXE)

nocpp:  $(BENCH_FILES) $(O_FILES)
	$(CC) $(LDFLAGS) $(O_FILES) $(BENCH_FILES) -o $(BENCH_EXE)

debug:
	$(CC) -g -G -D USING_CPP $(LDFLAGS) topology.cu parameters.cpp $(BENCH_FILES) -o $(BENCH_EXE)

debug_nocpp:
	$(CC) -g -G $(LDFLAGS) topology.cu parameters.cpp $(BENCH_FILES) -o $(BENCH_EXE)

clean:
	rm  *.o $(BENCH_EXE) $(NUMA_EXE)
