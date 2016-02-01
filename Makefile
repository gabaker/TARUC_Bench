CC=nvcc
FILES=CUDA_Bandwidth
EXE=run
LDFLAGS=-std=c++11
#-Xcompiler=-std=c++0x
NONE=-lhwloc
DFLAGS=-D USING_CPP
all:
	$(CC) $(LDFLAGS) $(FILES).cu -o $(EXE)

cpp:
	$(CC) $(DFLAGS) $(LDFLAGS) $(FILES).cu -o $(EXE)

clean:
	rm $(EXE) 
