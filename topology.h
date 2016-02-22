
//benchmark header
#ifndef BENCH_HEADER_INC
#define BENCH_HEADER_INC
#include "bench.h"
#endif


#ifndef TOPOLOGY_CLASS_INC
#define TOPOLOGY_CLASS_INC
class SystemTopo
{

   //useful values extrapolated from hwloc topology
   int NumNodesInSystem;
   int NumSocketsInSystem;
   int NumPUsInSystem;
   int NumCoresInSystem;

   int NumCoresPerSocket;
   int NumPUsPerCore;

   bool HyperThreaded;
   bool SymmetricTopo;

   //std::vector<std::vector<std::vector<hwloc_obj_t> > > SysObjs;

   public:

      void GetTopology(hwloc_topology_t &dupTopology);
      void PrintTopology();
      void PinPURange(int FirstPU, int LastPU);
      void PinPUBySocket();
      void PinNumaNodeBySocket();
      void PinNumaNodeByPU(); 
      void GetNumSockets();
      void GetNumPUs();  
      int TotalNumCores();
      int TotalNumPUs();
      int TotalNumSockets();
      SystemTopo();
      ~SystemTopo();

   private: 
      //values needed for hwloc functionality
      hwloc_topology_t topology;
      int TopoDepth;
      void InitTopology();
      void ParseTopology();
      void FreeTopology();
};
#endif

