#ifndef TOPOLOGY_CLASS_INC
#define TOPOLOGY_CLASS_INC

//cuda headers and helper functions
#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"
//#include "nvml.h"

// C/C++ standard includes
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <math.h>

// OpenMP threading includes
#include <omp.h>

// NUMA Locality includes
#include <sched.h>
#include <hwloc.h>
#include <hwloc/helper.h>
class SystemTopo
{
   public:

      //Topology pinning - numa and cpuset
      void PinNode(int NodeIdx);
      void PinSocket(int SocketIdx);
      void PinCore(int CoreIdx);
      void PinPU(int PUIdx);
      void PinPUBySocket(int SocketIdx, int PUIdx);
      void PinCoreBySocket(int SocketIdx, int CoreIdx);

      // Memory Allocation Functions 
      void * AllocMemByCore(int CoreIdx, long long NumBytes);
      void * AllocMemByNode(int NodeIdx, long long NumBytes);
      void * AllocMemBySocket(int SocketIdx, long long NumBytes);
      void * AllocPinMemByNode(int NodeIdx, long long NumBytes);
      void * AllocWCMemByNode(int NodeIdx, long long NumBytes);
      void * AllocManagedMemByNode(int NodeIdx, int DevIdx, long long NumBytes);
      void * AllocMappedMemByNode(int NodeIdx, int DevIdx, long long NumBytes);
      void * AllocDeviceMem(int DevIdx, long long NumBytes);

      // Memory Deallocation Functions
      void FreeHostMem(void *Addr, long long NumBytes);     
      void FreePinMem(void *Addr, long long NumBytes);
      void FreeWCMem(void *Addr);
      void FreeMappedMem(void *Addr);
      void FreeManagedMem(void *Addr);
      void FreeDeviceMem(void *Addr, int DevIdx);

      // Other Memory Utility Functions 
      void SetHostMem(void *Addr, int Value, long long NumBytes);
      void SetDeviceMem(void *Addr, int Value, long long NumBytes, int DevIdx);
      void PinHostMemory(void *Addr, long long NumBytes);

      // Device UVA and P2P Functions
      bool DeviceUVA(int DevIdx);  
      bool DeviceGroupUVA(int DevA, int DevB); 
      bool DeviceGroupCanP2P(int DevA, int DevB);
      void DeviceGroupSetP2P(int DevA, int DevB, bool Status); 

      // Device Utility Functions
      void SetActiveDevice(int DevIdx);
      int NumPeerGroups();      
      std::vector<std::vector<int> > GetPeerGroups();
      std::string GetDeviceName(int DevIdx);

      // System Topology Info and Utility
      int NumNodes();
      int NumSockets();
      int NumCores();
      int NumPUs();  
      int NumCoresPerSocket();
      int NumPUsPerCore();
      int NumPUsPerSocket();
      int NumGPUs();
      void GetTopology(hwloc_topology_t &Copy);
      void PrintTopology(std::ofstream &OutFile);
      void PrintDeviceProps(std::string FileName);

      SystemTopo();
      ~SystemTopo();

   private: 
      // HWLOC topology object
      hwloc_topology_t Topology;

      //Device Info
      cudaDeviceProp *DevProps;
      std::vector<std::vector<int> > PeerGroups;
      int PeerGroupCount;

      // Depths of obj types
      int TopoDepth;
      int NodeDepth;
      int SocketDepth;
      int CoreDepth;
      int PUDepth;

      // Numbers by type and hierarchy level
      int NodesInSystem;
      int SocketsInSystem;
      int PUsInSystem;
      int CoresInSystem;
      int CoresPerSocket;
      int PUsPerCore;
      int GPUs;

      // Structure types
      bool HyperThreaded;
      bool SymmetricTopo;

      void GetAllDeviceProps(); 
      void InitTopology();
      void ParseTopology();
      void FreeTopology();
};
#endif

