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

typedef struct TestParams {

   std::string resultsFile;
   std::string inputFile;
   bool useDefaultParams;

   bool printDevProps;
   std::string devPropFile;

   std::string topoFile;
   bool runTopoAware;

   int nDevices;

   // Overhead memory test for allocation and deallocation of Host and Device memory
   bool runMemoryOverheadTest;
   bool runAllDevices;
   long rangeMemOverhead[3]; //min, max and step size (in bytes)
 
   // Device-Peer PCIe Baseline bandwidth test
   bool runHostDeviceBandwidthTest;
   bool varyBlockSizeHD;
   bool usePinnedHD;
   bool runBurstHD;
   bool runSustainedHD;
   long rangeHostDeviceBW[3]; //min, max and step size (in bytes)

   // Peer-to-peer device memory transfer bandwidth
   bool runP2PBandwidthTest;
   bool varyBlockSizeP2P;
   bool runBurstP2P;
   bool runSustainedP2P;
   long rangeDeviceP2P[3]; //min, max and step size (in bytes)

   // PCIe Congestion tests
   bool runPCIeCongestionTest;

   // CUDA kernel task scalability and load balancing
   bool runTaskScalabilityTest;

} TestParams;

typedef struct PU {

   int puID;
   int coreID;
   int cpuID;
   int numCoreSiblings;
   int numCPUSiblings;
   std::vector<int> coreSiblings;
   std::vector<int> cpuSiblings;

} PUInfo;

typedef struct CPU {

   int physical_ID;
   int numCores;
   int numPUs; 
   std::vector<PU> PUs;

} SocketInfo;

typedef struct SystemInfo {

   int numCPUs;
   std::vector<CPU> sockets;
   int numPUs;

} System;

typedef enum {
   DEVICE_MALLOC,
   HOST_MALLOC,
   HOST_PINNED_MALLOC,
   DEVICE_FREE,
   HOST_FREE,
   HOST_PINNED_FREE
} MEM_OP;

