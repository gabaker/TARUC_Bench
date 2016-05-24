#include <cuda.h>
#include <cuda_runtime.h>
#include <numa.h> //all numa calls

#include <unistd.h> //sysconf, usleep
#include <sched.h> //sched_getcpu, sched_setaffinity (process)

#include <hwloc.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <limits>
#include <errno.h>
#include <stdio.h>

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
   std::cout << "-----------------------------------------------------------------" << std::endl;
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

int main (int argc, char *argv[]) {

   //hwloc_topology_t topology;

   //std::cout << "Depth: " << topodepth << std::endl;


   SystemTopo SysInfo;

   SysInfo.PrintTopology(); 

/*   int socket_d = ;
   int pu_d = ;
   int core_d = ;
   int numa_d = ;
   int 

   int num_machines = hwloc_get_nobj_by_depth(topology, );
   hwloc_obj_t machine = hwloc_get_obj_by_depth(topology, );
*/   

   //long  numCores = sysconf( _SC_NPROCESSORS_CONF );

   /*if (numa_available() < 0) {
      std::cout << "This system does not support the NUMA API...exiting" << std::endl;
   }*/


//   cpu_set_t CPUs;
//   CPU_ZERO(&CPUs);
//   sched_getaffinity(0, sizeof(CPUs), &CPUs);

//   int currCore = sched_getcpu(); 

/*   int count = 0;
   for (int i = 0; i < CPU_SETSIZE; ++i) {
      if (CPU_ISSET(i, &CPUs))
         count++;
   }*/


/*   currCore = sched_getcpu(); 
   std::cout << "Main Program thread: " << currCore << std::endl;

   std::cout << "numa_num_task_nodes: " << numa_num_task_nodes() << std::endl;
   std::cout << "numa_max_node:" << numa_max_node() << std::endl;
   std::cout << "numa_max_possible_node:" << numa_max_possible_node() << std::endl;
   std::cout << "numa_num_possible_nodes:" << numa_num_possible_nodes() << std::endl;
   
   std::cout << "numa_num_configured_cpus: " << numa_num_configured_cpus() << std::endl;
   std::cout << "numa_num_configured_nodes: " << numa_num_configured_nodes() << std::endl;
   std::cout << "numa_get_mems_allowed: " << numa_get_mems_allowed() << std::endl;
   std::cout << "numa_num_task_cpus: " << numa_num_task_cpus() << std::endl;
   std::cout << "numa_preferred: " << numa_preferred() << std::endl;
   
   std::cout << "numa_distance 0 => 1: " << numa_distance(0, 1) << std::endl;

   int OmpMaxThreads = 16;//omp_get_max_threads();
   omp_set_num_threads(OmpMaxThreads);
   std::cout << "Max Omp Threads: " << OmpMaxThreads << std::endl;

   for (int i = 0; i < numa_num_configured_cpus(); ++i) {
      std::cout << "Physical core : " << i << " NUMA node: " << numa_node_of_cpu(i) << std::endl;
   }

   count = 0;
   while(1) {

      //numa_run_on_node(count % 2);
      std::cout << "RUN ON NUMA NODE: " << (count % 2) << std::endl;
      #pragma omp parallel
      {

         cpu_set_t cpuset;
         CPU_ZERO(&cpuset);

         int thread_ID = omp_get_thread_num();
         int node = (count % 2) == 0 ? 0 : 1;
         
         if (thread_ID < 8) {
            CPU_SET(thread_ID + node * 8, &cpuset);
         } else {
            CPU_SET(thread_ID + node * 8 + 8, &cpuset);
         }
         pthread_t thread = pthread_self();
         int s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
         
         if (s != 0)
            std::cout << "Problem in setaffinity function: ThreadID: " << omp_get_thread_num() << "Core:" << sched_getcpu() << std::endl;

         long long loop = 0;
         while(loop < 1000000000)
            loop++;

         #pragma omp critical
         {
            bitmask *numa_mask = numa_allocate_cpumask();
            numa_node_to_cpus(0, numa_mask);
            std::cout << "Numa CPU mask " << numa_all_nodes_ptr->maskp << std::endl;

            std::cout << "CPU_MEM_PREF/Sched_Core/OMP_Thread_Count/OMP_Thread_ID: ";
            std::cout << numa_preferred() << " " << sched_getcpu() << " " << omp_get_num_threads() << " " << omp_get_thread_num();
            std::cout << std::endl;
         }
      }
   }*/
   return 0;
}

