//cuda headers and helper functions
#include<cuda_runtime.h>
#include<cuda.h>
#include "helper_cuda.h"
#include "nvml.h"

// C/C++ standard includes
#include<memory>
#include<iostream>
#include<stdio.h>
#include<string>
#include<vector>
#include<time.h>
#include<string>
#include<fstream>
#include<iostream>
#include<ios>
#include<vector>
#include<unistd.h>
#include<sys/time.h>
#include<sstream>
#include<iomanip>

// Newer c++ timing lib 
// Does not compile with older C++ compiler versions (i.e. RHEL 6 standard g++ version)
#ifdef USING_CPP
#include<chrono>
#endif

// OpenMP threading includes
#include<omp.h>

// NUMA Locality includes
#include<hwloc.h>
#include<numa.h>
#include<sched.h>

// Time conversions for varying timing functions that are compatible 
// with cuda, c++11 or older C++ versions
#define MILLI_TO_MICRO (1.0 / 1000.0)
#define MICRO_TO_MILLI (1000.0)
#define NANO_TO_MILLI (1.0 / 1000000.0)
#define NANO_TO_MICRO (1.0 / 1000.0)
#define NUM_PATTERNS 4
#define MAX_PATTERN_SIZE 10000000
#define BURST_BLOCK_SIZE pow(2, 24)
#define HOST_MEM_TYPES 2


// Memory access patterns to test help adjust small PCI transfer latency for possible Host 
// caching effects with repeated transfers of memory blocks
typedef enum {
   REPEATED,
   RANDOM,
   PERIODIC,
   LINEAR_INC,
   LINEAR_DEC,
   BURST
} MEM_PATTERN;

// Memory allocation types to be used in benchmark suppost functions to adjust behavior of 
// test run depending on benchmark parameters and cases being studied
typedef enum {
   DEVICE_MALLOC,             // Device memory allocation, single memory block
   HOST_MALLOC,               // Host pageable memory allocation, single memory block
   HOST_PINNED_MALLOC,        // Host pinned memory allocation, single memory block
   DEVICE_FREE,               // Device memory deallocation, single memory block
   HOST_FREE,                 // Host pageable memory deallocation, single memory block
   HOST_PINNED_FREE,          // Host pinned memory deallocation, single memory block

   HOST_HOST_COPY,            // Host-To-Host Copy, pageable memory
   HOST_PINNED_HOST_COPY,     // Host-To-Host Copy, src pinned, dest pageable
   HOST_HOST_PINNED_COPY,     // Host-To-Host Copy, dest pinned, src pageable
   HOST_HOST_COPY_PINNED,     // Host-To-Host Copy, both src/dest pinned memory

   HOST_DEVICE_COPY,          // Host-To-Device copy, pageable host memory
   HOST_PINNED_DEVICE_COPY,   // Host-To-Device copy, pinned host memory
   DEVICE_HOST_COPY,          // Device-To-Host copy, pageable host memory
   DEVICE_HOST_PINNED_COPY,   // Device-To-Host copy, pinned host memory
   DEVICE_DEVICE_COPY,        // Device-To-Device copy, no peer support
   PEER_COPY_NO_UVA,          // Peer-to-Peer device copy, no uva support
   COPY_UVA                   // General UVA copy, CUDA runtime copy based on pointer addressing

} MEM_OP;

// Header for BenchParams class, user to read in and print out benchmark parameters
#ifndef PARAM_CLASS_INC
#include "parameters.h"
#define PARAM_CLASS_INC
#endif

// Header for system topology detection; abstraction for numa.h and hwloc libraries
#ifndef TOPOLOGY_CLASS_INC
#include "topology.h"
#define TOPOLOGY_CLASS_INC
#endif

