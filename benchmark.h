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

//newer c++ timing lib
#ifdef USING_CPP
#include<chrono>
#endif

// OpenMP threading includes
#include<omp.h>

// NUMA Locality includes
#include<hwloc.h>
#include<numa.h>
#include<sched.h>

#define MILLI_TO_MICRO (1.0 / 1000.0)
#define MICRO_TO_MILLI (1000.0)
#define NANO_TO_MILLI (1.0 / 1000000.0)
#define NANO_TO_MICRO (1.0 / 1000.0)

#define MAX_PATTERN_SIZE 10000000

typedef enum {
   REPEATED,
   RANDOM,
   PERIODIC,
   LINEAR_INC,
   LINEAR_DEC,
   BURST
} MEM_PATTERN;

typedef enum {
   DEVICE_MALLOC,
   HOST_MALLOC,
   HOST_PINNED_MALLOC,
   DEVICE_FREE,
   HOST_FREE,
   HOST_PINNED_FREE,
   HOST_DEVICE_COPY,
   HOST_DEVICE_COPY_PINNED,
   DEVICE_HOST_COPY,
   DEVICE_HOST_COPY_PINNED,
   DEVICE_DEVICE_COPY,
   HOST_HOST_COPY,
   HOST_HOST_COPY_PINNED,
   PEER_COPY_NO_UVA,
   COPY_UVA
} MEM_OP;

#ifndef PARAM_CLASS_INC
#include "parameters.h"
#define PARAM_CLASS_INC
#endif

#ifndef TOPOLOGY_CLASS_INC
#include "topology.h"
#define TOPOLOGY_CLASS_INC
#endif

