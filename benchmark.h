//cuda headers and helper functions
#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"
#include "nvml.h"

// C/C++ standard includes
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <ios>
#include <vector>
#include <unistd.h>
#include <sstream>
#include <iomanip>
#include <algorithm>

// OpenMP threading includes
#include <omp.h>

// NUMA Locality includes
#include <sched.h>
#include <hwloc.h>

// Time conversions for varying timing functions that are compatible 
// with cuda, c++11 or older C++ versions
#define MILLI_TO_MICRO ((double) 1.0 / (double) 1e3f)
#define MICRO_TO_MILLI ((double) 1e3f)
#define NANO_TO_MILLI ((double) 1.0 / (double) 1e6f)
#define NANO_TO_MICRO ((double) 1.0 / (double) 1e3f)
#define NUM_PATTERNS 3
#define HOST_MEM_TYPES 3
#define BYTES_TO_MEGA ((double) pow(2.0, 20))

// Memory access patterns to test help adjust small PCI transfer latency for possible Host 
// caching effects with repeated transfers of memory blocks
typedef enum {
   REPEATED,
   LINEAR_INC,
   LINEAR_DEC,
} MEM_PATTERN;

typedef enum {
   HD,
   HH,
   P2P
} BW_RANGED_TYPE;

typedef enum {
   PAGE,
   PINNED,
   WRITE_COMBINED,
   MANAGED,
   MAPPED,
   DEVICE
} MEM_TYPE;

// Memory allocation types to be used in benchmark suppost functions to adjust behavior of 
// test run depending on benchmark parameters and cases being studied
typedef enum {
   HOST_MALLOC,               // Host pageable memory allocation, single memory block
   HOST_PINNED_MALLOC,        // Host pinned memory allocation, single memory block
   HOST_COMBINED_MALLOC,
   MANAGED_MALLOC,
   MAPPED_MALLOC,
   DEVICE_MALLOC,             // Device memory allocation, single memory block

   HOST_FREE,                 // Host pageable memory deallocation, single memory block
   HOST_PINNED_FREE,          // Host pinned memory deallocation, single memory block
   HOST_COMBINED_FREE,
   MANAGED_FREE,
   MAPPED_FREE,
   DEVICE_FREE,               // Device memory deallocation, single memory block

   HOST_HOST_COPY,            // Host-To-Host Copy, pageable memory
   DEVICE_HOST_COPY,          // Device-To-Host copy, pageable host memory
   HOST_DEVICE_COPY,          // Host-To-Device copy, pageable host memory
   
   HOST_PINNED_HOST_COPY,     // Host-To-Host Copy, src pinned, dest pageable
   HOST_HOST_PINNED_COPY,     // Host-To-Host Copy, dest pinned, src pageable
   HOST_HOST_COPY_PINNED,     // Host-To-Host Copy, both src/dest pinned memory
   HOST_PINNED_DEVICE_COPY,   // Host-To-Device copy, pinned host memory
   DEVICE_HOST_PINNED_COPY,   // Device-To-Host copy, pinned host memory

   HOST_COMBINED_HOST_COPY,
   HOST_HOST_COMBINED_COPY,
   HOST_HOST_COPY_COMBINED,
   HOST_COMBINED_DEVICE_COPY,
   DEVICE_HOST_COMBINED_COPY,

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

