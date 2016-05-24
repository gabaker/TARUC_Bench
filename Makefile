SHELL = /bin/sh

CC=nvcc
SOURCES=topology.cu parameters.cpp
BENCH_EXE=run
O_FILES=topology.o parameters.o
BENCH_FILES=benchmark.cu benchmark.h
FLAGS= -std=c++11 -O3 -lhwloc -Xcompiler -fopenmp 
#-L/usr/src/gdk/nvml/lib/ -lnvidia-ml -D_MWAITXINTRIN_H_INCLUDED
#-lpthread 
#-lnuma

default: taruc

nocpp: taruc_nocpp

topology.o: topology.cu topology.h
	$(CC) $(FLAGS) -c topology.cu -o topology.o

parameters.o: parameters.cpp parameters.h
	$(CC) $(FLAGS) -c parameters.cpp -o parameters.o

timer.o: timer.cu timer.h
	$(CC) -D USING_CPP $(FLAGS) -c timer.cu -o timer.o

timer_nocpp.o: timer.cu timer.h
	$(CC) $(FLAGS) -c timer.cu -o timer_nocpp.o

benchmark.o: $(BENCH_FILES) $(O_FILES) timer.o
	$(CC) -D USING_CPP $(FLAGS) -c benchmark.cu -o benchmark.o

benchmark_nocpp.o: $(BENCH_FILES) $(O_FILES) timer_nocpp.o
	$(CC) $(FLAGS) -c benchmark.cu -o benchmark_nocpp.o

taruc: $(O_FILES) benchmark.o
	$(CC) -D USING_CPP $(FLAGS) $(O_FILES) timer.o benchmark.o -o $(BENCH_EXE)

taruc_nocpp: $(O_FILES) benchmark_nocpp.o
	$(CC) $(FLAGS) $(O_FILES) timer_nocpp.o benchmark_nocpp.o -o $(BENCH_EXE)_nocpp

.PHONY: clean

clean:
	rm -f *.o $(BENCH_EXE) $(BENCH_EXE)_nocpp


