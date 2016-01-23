CC=nvcc
FILES=CUDA_Bandwidth
EXE=run
LDFLAGS=-Wall -std=c++0x
NONE=-lhwloc

all:
	$(CC) $(LDFLAGS) $(FILES).cu -o $(EXE)

clean:
	rm $(EXE) *.o
