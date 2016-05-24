# Topology Aware Resource Usability and Contention (TARUC) Benchmark

####Author: Gavin Baker
####Contact:
            gavin.m.baker@gmail.com
            gabaker@calpoly.edu

##Benchmark Overview:

CUDA PCIe Bandwidth, Memory and Multi-GPU Benchmarks that accounts for multi socket systems with many variable model GPUs


##Package Requirements:

- C++11
- CUDA 7.5 (untested on earlier)
- NVML
- HWLOC
- python2.7
- OpenMP (omp.h)
- Unix operating system

##Features:
- CUDA support for multi-gpu systems
- Support for diverse topology recognition using hwloc, numa.h and cpuset.h 
- matplot lib plotting of complete benchmark runs using provided parameter file

##Non-Features:
- Distributed system awareness
- Network communicaton

##Run Instructions:

Run w/ Default parameters:

      TARUC_Bench/scripts/run_benchmark.sh parameters.in

Run w/ provided parameter file (parameters.in)

      TARUC_Bench/scripts/run_benchmark.sh parameters.in

##Explaination of Directory Structure and Project Files:

Folders:

- scripts/
- results/
- misc/
- sample_images/

Files:

- benchmark.cu
- benchmark.h
- helper_cuda.h
- helper_string.h
- parameters.cpp
- parameters.h
- parameters.in
- topology.cu
- topology.h
- timer.cu
- timer.h
- nvml.h
- Makefile
