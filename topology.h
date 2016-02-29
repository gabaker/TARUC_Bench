
//benchmark header
#ifndef BENCH_HEADER_INC
#define BENCH_HEADER_INC
#include "benchmark.h"
#endif

#ifndef TOPOLOGY_CLASS_INC
#define TOPOLOGY_CLASS_INC
class SystemTopo
{

   //std::vector<std::vector<std::vector<hwloc_obj_t> > > SysObjs;

   public:
      //useful values extrapolated from hwloc topology

      void GetTopology(hwloc_topology_t &dupTopology);
      void PrintTopology(std::ofstream &OutFile);

      //Topology pinning - numa and cpuset
      void PinNumaNode(int nodeIdx);
      void PinSocket(int socketIdx);

      //void PinCoreBySocket(int coreIdx);
      //void PinPUBySocket(int socketIdx, int puIdx);
      //void PinPURange(int FirstPU, int LastPU);
      //void PinNumaNodeByPU(int puIdx); 
      
      //return class local variables
      int NumNodes();
      int NumSockets();
      int NumCores();
      int NumPUs();  
      int NumCoresPerSocket();
      int NumPUsPerCore();

      SystemTopo();
      ~SystemTopo();

   private: 
      // HWLOC topology object
      hwloc_topology_t topology;

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
      int NumGPUs;

      // Structure types
      bool HyperThreaded;
      bool SymmetricTopo;

      void InitTopology();
      void ParseTopology();
      void FreeTopology();
};
#endif

