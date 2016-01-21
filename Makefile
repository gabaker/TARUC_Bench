CC=nvcc
FILES=CUDA_Bandwidth
EXE=run
LDFLAGS= 
NONE=-lhwloc

all:
	$(CC) $(LDFLAGS) $(FILES).cu -o $(EXE)

clean:
	rm $(EXE) *.o
