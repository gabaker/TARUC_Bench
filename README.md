# Topology Aware Resource Usability and Contention (TARUC) Benchmark

####Author: Gavin Baker
####Contact:
            gavin.m.baker@gmail.com
            gabaker@calpoly.edu

##Benchmark Overview:

CUDA PCIe Bandwidth, Memory and Multi-GPU Benchmarks that accounts for multi socket systems with many variable model GPUs

###Memory Management Overhead

###Memory Transfer Bandwidth

####Host-Host Transfer Bandwidth (single-threaded)

####Host-Device Transfer Bandwidth (single-threaded)

####Device-Device Transfer Bandwidth (single-threaded)

###Resource Contention

####Local Host Memory Bandwidth (multi-threaded)

####Inter-Socket (QPI) Host Memory Bandwidth (multi-threaded)

####Single Device PCIe Memory Bandwidth (multi-threaded)

####GPU Pair Memory Bandwidth (multi-threaded)

####Single Host Multi-Device Memory Bandwidth (multi-threaded)

###NUMA Latency

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

##Parameter Explaination

The following is a sample parameter file demonstrating correct format. 

   ------- Benchmark Parameters --------

   Run Tag = anatta

   ------------- All Tests -------------

   Use All GPUs = true

   Test All Mem Types = true

   Run Burst Tests = false

   Run Range Tests = true

   Run Sustained Tests = true

   Run Socket Effect Tests = true

   # Repeated Steps = 20

   # Steps Per Magnitude = 10

   Burst Block Size (bytes) = 64000000

   ---------- Memory Overhead ----------

   Run Memory Overhead Test = false

   Range Min = 100

   Range Max = 2500000000

   --------- Host-Host Bandwidth -------

   Run Host-Host Tests = false

   Vary Access Patterns = true

   Range Min = 100

   Range Max = 2500000000

   ------- Host-Device Bandwidth -------

   Run Host-Device Tests = false

   Vary Access Patterns = true

   Range Min = 100

   Range Max = 2500000000

   ----------- P2P Bandwidth -----------

   Run P2P Tests = false

   Range Min = 100

   Range Max = 1500000000

   -------- Resource Congestion --------

   Run Congestion Test = true

   # Repeated Operations = 40

   Local Host Block Size =  100000000

   QPI Host Block Size =    100000000

   Host-Device Block Size = 100000000

   -------------------------------------


##Directory Structure and Project Files:

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
