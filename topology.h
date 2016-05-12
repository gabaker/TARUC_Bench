
//benchmark header
#ifndef BENCH_HEADER_INC
#define BENCH_HEADER_INC
#include "benchmark.h"
#endif

#ifndef TOPOLOGY_CLASS_INC
#define TOPOLOGY_CLASS_INC
class SystemTopo
{
   public:

      //Topology pinning - numa and cpuset
      void PinNumaNode(int nodeIdx);
      void PinSocket(int socketIdx);
      void PinCore(int coreIdx);
      void PinPU(int puIdx);
      void PinPUBySocket(int socketIdx, int puIdx);
      void PinCoreBySocket(int socketIdx, int coreIdx);

      // Memory Allocation Functions 
      void * AllocMemByCore(int coreIdx, long long numBytes);
      void * AllocMemByNode(int nodeIdx, long long numBytes);
      void * AllocMemBySocket(int socketIdx, long long numBytes);
      void * AllocPinMemByNode(int nodeIdx, long long numBytes);
      void * AllocWCMemByNode(int nodeIdx, long long numBytes);
      void * AllocManagedMemByNode(int nodeIdx, int devIdx, long long numBytes);
      void * AllocMappedMemByNode(int nodeIdx, int devIdx, long long numBytes);
      void * AllocDeviceMem(int devIdx, long long numBytes);
      
      // Memory Deallocation Functions
      void FreeHostMem(void *addr, long long numBytes);     
      void FreePinMem(void *addr, long long numBytes);
      void FreeWCMem(void *addr);
      void FreeMappedMem(void *addr);
      void FreeManagedMem(void *addr);
      void FreeDeviceMem(void *addr, int deviceIdx);

      // Other Memory Utility Functions 
      void SetHostMem(void *addr, int value, long long numBytes);
      void SetDeviceMem(void *addr, int value, long long numBytes, int deviceIdx);
      void PinHostMemory(void *addr, long long numBytes);

      // Device UVA and P2P Functions
      bool DeviceUVA(int deviceIdx);  
      bool DeviceGroupUVA(int deviceA, int deviceB); 
      bool DeviceGroupCanP2P(int deviceA, int deviceB);
      void DeviceGroupSetP2P(int deviceA, int deviceB, bool status); 

      // Device Utility Functions
      void SetActiveDevice(int devIdx);
      void ResetDevices(); 
      int NumPeerGroups();      
      std::vector<std::vector<int> > GetPeerGroups();
      std::string GetDeviceName(int devIdx);

      // System Topology Info and Utility
      int NumNodes();
      int NumSockets();
      int NumCores();
      int NumPUs();  
      int NumCoresPerSocket();
      int NumPUsPerCore();
      int NumPUsPerSocket();
      int NumGPUs();
      void GetTopology(hwloc_topology_t &dupTopology);
      
      void PrintTopology(std::ofstream &OutFile);
      void PrintDeviceProps(BenchParams &params);

      SystemTopo();
      ~SystemTopo();

   private: 
      // HWLOC topology object
      hwloc_topology_t topology;

      //Device Info
      cudaDeviceProp *devProps;
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
      int NumDevices;

      // Structure types
      bool HyperThreaded;
      bool SymmetricTopo;

      void GetAllDeviceProps(); 
      void InitTopology();
      void ParseTopology();
      void FreeTopology();
};
#endif

