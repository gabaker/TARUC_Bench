// CUDA headers and helper functions
#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

// C/C++ standard includes
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <algorithm>
//#include <mwaitxintrin.h>

// OpenMP threading includes
#include <omp.h>

// NUMA Locality includes
#include <sched.h>
#include <hwloc.h>

#define HOST_MEM_TYPES 2

// Memory block size conversions
#define BYTES_TO_MEGA ((double) pow(2.0, 20))

// Memory access patterns to test help adjust small PCI transfer latency for possible Host 
// caching effects with repeated transfers of memory blocks
typedef enum {
   REPEATED,      // Same initial address for each transfer
   LINEAR_INC,    // Linearly increasing address (skip a block)
   LINEAR_DEC,    // Linearly decreasing address (skip a block)
} MEM_PATTERN;

typedef enum {
   HD,            // Host-Device Transfer
   HH,            // Host-Host Transfer
   P2P            // Device-Device Transfer
} BW_RANGED_TYPE;

// Memory types for host and device memory
typedef enum {
   PAGE,                // Pageable host memory
   PINNED,              // Pinned (non-pageable) host memory
   WRITE_COMBINED,      // Write combined host memory
   MANAGED,             // CUDA managed memory block
   MAPPED,              // CUDA host-device mapped memory block
   DEVICE               // CUDA device memory
} MEM_TYPE;

// Memory types to be used in micro-benchmark support functions to adjust behavior of 
// test run depending on benchmark parameters and cases being studied
typedef enum {
   HOST_MALLOC,               // Host pageable memory allocation, single memory block
   HOST_PINNED_MALLOC,        // Host pinned memory allocation, single memory block
   HOST_COMBINED_MALLOC,      // Host write-combined memory allocation, single memory block
   MANAGED_MALLOC,            // CUDA managed memory allocation, single memory block
   MAPPED_MALLOC,             // CUDA mapped memory allocation, single memory block
   DEVICE_MALLOC,             // Device memory allocation, single memory block

   HOST_FREE,                 // Host pageable memory deallocation, single memory block
   HOST_PINNED_FREE,          // Host pinned memory deallocation, single memory block
   HOST_COMBINED_FREE,        // Host write-combined memory deallocation, single memory block
   MANAGED_FREE,              // CUDA managed memory deallocation, single memory block
   MAPPED_FREE,               // CUDA mapped memory deallocation, single memory block
   DEVICE_FREE,               // Device memory deallocation, single memory block

   HOST_HOST_COPY,            // Host-To-Host Copy, pageable memory
   DEVICE_HOST_COPY,          // Device-To-Host copy, pageable host memory
   HOST_DEVICE_COPY,          // Host-To-Device copy, pageable host memory
   
   HOST_PINNED_HOST_COPY,     // Host-To-Host Copy, src pinned, dest pageable
   HOST_HOST_PINNED_COPY,     // Host-To-Host Copy, dest pinned, src pageable
   HOST_HOST_COPY_PINNED,     // Host-To-Host Copy, both src/dest pinned memory
   HOST_PINNED_DEVICE_COPY,   // Host-To-Device copy, pinned host memory
   DEVICE_HOST_PINNED_COPY,   // Device-To-Host copy, pinned host memory

   HOST_COMBINED_HOST_COPY,   // Host-To-Host Copy, src write-combined, dest pageable
   HOST_HOST_COMBINED_COPY,   // Host-To-Host Copy, src pageable, dest write-combined
   HOST_HOST_COPY_COMBINED,   // Host-To-Host Copy, both write-combined host memories
   HOST_COMBINED_DEVICE_COPY, // Host-To-Device Copy, write-combined host memory
   DEVICE_HOST_COMBINED_COPY, // Device-To-Host Copy, write-combined host memory

   DEVICE_DEVICE_COPY,        // Device-To-Device copy, no peer support
   PEER_COPY_NO_UVA,          // Peer-to-Peer device copy, no uva support
   COPY_UVA                   // General UVA copy, CUDA runtime copy based on pointer addressing

} MEM_OP;

