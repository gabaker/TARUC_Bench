CC = nvcc
CFLAGS = -O3 -Xcompiler -fopenmp -std=c++11 -lhwloc

default: stream

topology.o: topology.cu
	$(CC) $(CFLAGS) topology.cu -c topology.o
	
stream: stream.cpp topology.o
	$(CC) $(CFLAGS) topology.o stream.cpp -o run

shared: stream.cpp topology.o
	$(CC) -D SHARED_BLOCK $(CFLAGS) topology.o stream.cpp -o run

clean:
	rm -f *.o run

