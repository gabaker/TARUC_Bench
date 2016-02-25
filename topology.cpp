
//#ifndef TOPOLOGY_CLASS_INC
//#define TOPOLOGY_CLASS_INC
#include "topology.h"
//#endif

void SystemTopo::PinNumaNode(int nodeIdx) {
   hwloc_obj_t node = hwloc_get_obj_by_depth(topology, NodeDepth, nodeIdx); 

   hwloc_nodeset_t nodeSet = node->nodeset;

  /* char *str;
   hwloc_bitmap_asprintf(&str, nodeSet);

   std::cout << str << std::endl;

   free(str);*/
   //std::cout << hwloc_get_api_version() << std::endl;
   hwloc_set_membind_nodeset(topology, nodeSet, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_NOCPUBIND | HWLOC_MEMBIND_THREAD);

}


void SystemTopo::PinSocket(int socketIdx) {
   hwloc_obj_t socket = hwloc_get_obj_by_depth(topology, SocketDepth, socketIdx);
   
   hwloc_set_cpubind(topology, socket->cpuset, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_NOMEMBIND);

}

/*void SystemTopo::PinCoreBySocket(int coreIdx) {


}

void SystemTopo::PinPUBySocket(int socketIdx, int puIdx) {

}*/

void SystemTopo::PrintTopology() {

   int s_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_SOCKET);
   int c_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
   int p_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);

   hwloc_obj_t s_obj;// = hwloc_get_obj_by_depth(topology, s_depth, 0);
   hwloc_obj_t c_obj;// = hwloc_get_obj_by_depth(topology, c_depth, 0);
   hwloc_obj_t p_obj;// = hwloc_get_obj_by_depth(topology, p_depth, 0);

   char *str;
   std::cout << "\n------------------------ System Topology ------------------------" << std::endl;
   std::cout << "\tSockets:\t\t" << SocketsInSystem << std::endl; 
   std::cout << "\tTotal Cores\t\t" << CoresInSystem << std::endl;
   std::cout << "\tTotal PUs:\t\t" << PUsInSystem << std::endl;
   std::cout << "\tCores Per Socket:\t" << CoresPerSocket << std::endl;
   std::cout << "\tPUs Per Core:\t\t" << PUsPerCore << std::endl;
   std::cout << "\tHyperthreaded:\t\t" << std::boolalpha << HyperThreaded << std::noboolalpha << std::endl;
   std::cout << "\tSymmetric Topo:\t\t" << std::boolalpha << SymmetricTopo << std::noboolalpha << std::endl;
   std::cout << "\n------------------------- Topology Tree -------------------------" << std::endl;
   
   int m_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_MACHINE); 
   hwloc_obj_t m_obj = hwloc_get_obj_by_depth(topology, m_depth, 0);

   hwloc_bitmap_asprintf(&str, m_obj->cpuset);
   std::cout << "Machine: " << "P#" << m_obj->os_index << " CPUSET=" << str << std::endl; 
   free(str); 
   
   for (int sNum = 0; sNum < SocketsInSystem; sNum++) {
      s_obj = hwloc_get_obj_by_depth(topology, s_depth, sNum);
      
      hwloc_bitmap_asprintf(&str, s_obj->cpuset);
      std::cout << "\tSocket: " << s_obj->os_index << " " << str << std::endl;
      free(str);
     
      for (int cNum = 0; cNum < SocketsInSystem * CoresPerSocket; cNum++) {    
         c_obj = hwloc_get_obj_by_depth(topology, c_depth, cNum);
         
         if (hwloc_obj_is_in_subtree(topology, c_obj, s_obj)) {  
            hwloc_bitmap_asprintf(&str, c_obj->cpuset);
            std::cout << "\t\tCore:" << " L#" << c_obj->logical_index << " P#" << c_obj->os_index << " CPUSET=" << str << std::endl;
            free(str);
      
            for (int pNum = 0; pNum < SocketsInSystem * PUsPerCore * CoresPerSocket; pNum++) {
               p_obj = hwloc_get_obj_by_depth(topology, p_depth, pNum);

               if (hwloc_obj_is_in_subtree(topology, p_obj, c_obj)) {  
                  hwloc_bitmap_asprintf(&str, p_obj->cpuset); 
                  std::cout << "\t\t\tPU:" << " L#" << p_obj->logical_index << " P#" << p_obj->os_index << " CPUSET=" << str << std::endl;
                  free(str);
               }
            }
         }
      }
      //for (int dNum = 0; dNum <  
      //   hwloc_bitmap_asprintf(&str, p_obj->cpuset); 
      //   std::cout << "\t\t\tPU:" << " L#" << p_obj->logical_index << " P#" << p_obj->os_index << " CPUSET=" << str << std::endl;
      //}
   }
   //std::cout << hwloc_get_nobj_by
   //std::cout << "-----------------------------------------------------------------" << std::endl;
}

void SystemTopo::ParseTopology() {

   TopoDepth = hwloc_topology_get_depth(topology);
   NodeDepth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);   
   SocketDepth = hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE);
   CoreDepth = hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
   PUDepth = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);
   
   NodesInSystem = hwloc_get_nbobjs_by_depth(topology, NodeDepth);
   SocketsInSystem = hwloc_get_nbobjs_by_depth(topology, SocketDepth);
   CoresInSystem = hwloc_get_nbobjs_by_depth(topology, CoreDepth); 
   PUsInSystem = hwloc_get_nbobjs_by_depth(topology, PUDepth);

   CoresPerSocket = CoresInSystem / SocketsInSystem;
   PUsPerCore = PUsInSystem / CoresInSystem;
   
   HyperThreaded = (PUsPerCore != 1) ? true : false;
   SymmetricTopo = (hwloc_get_root_obj(topology)->symmetric_subtree != 0) ? true : false;
   
}

//SystemTopo Constructor, initializes hwloc topology
SystemTopo::SystemTopo() {
   InitTopology();
   ParseTopology();

}

//SystemTopo destructor, free hwloc topology
SystemTopo::~SystemTopo() {
   FreeTopology();
}

//return a copy of the system topology
void SystemTopo::GetTopology(hwloc_topology_t &dupTopology) {
   hwloc_topology_dup(&dupTopology, topology);
}

void SystemTopo::InitTopology() {
   hwloc_topology_init(&topology);
   hwloc_topology_load(topology);
   //hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_IO_DEVICES | HWLOC_TOPOLOGY_FLAG_IO_BRIDGES);

}

void SystemTopo::FreeTopology() { 
   hwloc_topology_destroy(topology);
}

int SystemTopo::NumNodes() {
   return NodesInSystem;
}

int SystemTopo::NumSockets() {
   return SocketsInSystem;
}

int SystemTopo::NumCores() {
   return CoresInSystem;
}
 
int SystemTopo::NumPUs() {
   return PUsInSystem;
} 

int SystemTopo::NumCoresPerSocket() {
   return CoresPerSocket;
} 

int SystemTopo::NumPUsPerCore() {
   return PUsPerCore;
}

