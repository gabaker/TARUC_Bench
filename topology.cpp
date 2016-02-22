
//#ifndef TOPOLOGY_CLASS_INC
//#define TOPOLOGY_CLASS_INC
#include "topology.h"
//#endif

void SystemTopo::PrintTopology() {

   int s_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_SOCKET);
   int c_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
   int p_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);

   hwloc_obj_t s_obj;// = hwloc_get_obj_by_depth(topology, s_depth, 0);
   hwloc_obj_t c_obj;// = hwloc_get_obj_by_depth(topology, c_depth, 0);
   hwloc_obj_t p_obj;// = hwloc_get_obj_by_depth(topology, p_depth, 0);

   char *str;
   std::cout << "\n------------------------ System Topology ------------------------" << std::endl;
   std::cout << "\tSockets:\t\t" << NumSocketsInSystem << std::endl; 
   std::cout << "\tTotal Cores\t\t" << NumCoresInSystem << std::endl;
   std::cout << "\tTotal PUs:\t\t" << NumPUsInSystem << std::endl;
   std::cout << "\tCores Per Socket:\t" << NumCoresPerSocket << std::endl;
   std::cout << "\tPUs Per Core:\t\t" << NumPUsPerCore << std::endl;
   std::cout << "\tHyperthreaded:\t\t" << std::boolalpha << HyperThreaded << std::noboolalpha << std::endl;
   std::cout << "\tSymmetric Topo:\t\t" << std::boolalpha << SymmetricTopo << std::noboolalpha << std::endl;
   std::cout << "\n------------------------- Topology Tree -------------------------" << std::endl;
   
   int m_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_MACHINE); 
   hwloc_obj_t m_obj = hwloc_get_obj_by_depth(topology, m_depth, 0);

   hwloc_bitmap_asprintf(&str, m_obj->cpuset);
   std::cout << "Machine: " << "P#" << m_obj->os_index << " CPUSET=" << str << std::endl; 
   free(str); 
   
   for (int sNum = 0; sNum < NumSocketsInSystem; sNum++) {
      s_obj = hwloc_get_obj_by_depth(topology, s_depth, sNum);
      
      hwloc_bitmap_asprintf(&str, s_obj->cpuset);
      std::cout << "\tSocket: " << s_obj->os_index << " " << str << std::endl;
      free(str);
     
      for (int cNum = 0; cNum < NumSocketsInSystem * NumCoresPerSocket; cNum++) {    
         c_obj = hwloc_get_obj_by_depth(topology, c_depth, cNum);
         
         if (hwloc_obj_is_in_subtree(topology, c_obj, s_obj)) {  
            hwloc_bitmap_asprintf(&str, c_obj->cpuset);
            std::cout << "\t\tCore:" << " L#" << c_obj->logical_index << " P#" << c_obj->os_index << " CPUSET=" << str << std::endl;
            free(str);
      
            for (int pNum = 0; pNum < NumSocketsInSystem * NumPUsPerCore * NumCoresPerSocket; pNum++) {
               p_obj = hwloc_get_obj_by_depth(topology, p_depth, pNum);

               if (hwloc_obj_is_in_subtree(topology, p_obj, c_obj)) {  
                  hwloc_bitmap_asprintf(&str, p_obj->cpuset); 
                  std::cout << "\t\t\tPU:" << " L#" << p_obj->logical_index << " P#" << p_obj->os_index << " CPUSET=" << str << std::endl;
                  free(str);
               }
            }
         }
      }
   }
   //std::cout << "-----------------------------------------------------------------" << std::endl;
}

void SystemTopo::ParseTopology() {

   TopoDepth = hwloc_topology_get_depth(topology);
   
   NumNodesInSystem = hwloc_get_nbobjs_by_depth(topology, hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE));
   NumSocketsInSystem = hwloc_get_nbobjs_by_depth(topology, hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE));
   NumCoresInSystem = hwloc_get_nbobjs_by_depth(topology, hwloc_get_type_depth(topology, HWLOC_OBJ_CORE));
   NumPUsInSystem = hwloc_get_nbobjs_by_depth(topology, hwloc_get_type_depth(topology, HWLOC_OBJ_PU));

   NumCoresPerSocket = NumCoresInSystem / NumSocketsInSystem;
   NumPUsPerCore = NumPUsInSystem / NumCoresInSystem;
   
   HyperThreaded = (NumPUsPerCore != 1) ? true : false;
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
}

void SystemTopo::FreeTopology() { 
   hwloc_topology_destroy(topology);
}

