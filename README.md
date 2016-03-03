CUDA PCIe Bandwidth, Memory and Multi-GPU Benchmarks that accounts for multi socket systems with many variable model GPUs

####Author: Gavin Baker
####Contact:
            gabaker@calpoly.edu
            gavin.m.baker@gmail.com

##Acknowledgements: 

####Craig Ulmer (Sandia)
####Jerry Friesen (Sandia)
####Chris Lupo (Cal Poly)

##Explaination of Goals:

1. Goal 1
2. Goal 2
3. Goal 3

##Package Requirements:

- C++11
- CUDA toolkit
- HWLOC
- numa.h
- python2.7+
- openmp (omp.h)
- Linux/Unix system (untested on OSX or Windows)

##Features:
- CUDA support for multi-gpu systems
- Support for diverse topology recognition using hwloc, numa.h and cpuset.h 
- matplot lib plotting of complete benchmark runs using provided parameter file

##Non-Features:
- Distributed system awareness
- Network communicaton


##Run Instructions:

./scripts/run_benchmark.sh bench_params.in

##Explaination of Directory Structure and Project Files:

- scripts/
- results/
- misc/
- sample_images/

- benchmark.cu
- benchmark.h
- helper_cuda.h
- helper_string.h
- parameters.cpp
- parameters.h
- parameters.in
- topology.cu
- topology.h

