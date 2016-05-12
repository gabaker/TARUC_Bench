#include "topology.h"

// --------------------------- Memory Allocation Functions -----------------------------


void * SystemTopo::AllocMemByCore(int coreIdx, long long numBytes) {
   hwloc_obj_t core = hwloc_get_obj_by_depth(topology, CoreDepth, coreIdx);

   return hwloc_alloc_membind_policy_nodeset(topology, numBytes, core->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_NOCPUBIND | HWLOC_MEMBIND_STRICT);
}

void * SystemTopo::AllocMemByNode(int nodeIdx, long long numBytes) {
   hwloc_obj_t node = hwloc_get_obj_by_depth(topology, NodeDepth, nodeIdx);

   return hwloc_alloc_membind_policy_nodeset(topology, numBytes, node->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_NOCPUBIND | HWLOC_MEMBIND_STRICT);
}

void * SystemTopo::AllocMemBySocket(int socketIdx, long long numBytes) {
   hwloc_obj_t socket = hwloc_get_obj_by_depth(topology, SocketDepth, socketIdx);

   return hwloc_alloc_membind_policy_nodeset(topology, numBytes, socket->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_NOCPUBIND | HWLOC_MEMBIND_STRICT);
}

void * SystemTopo::AllocPinMemByNode(int nodeIdx, long long numBytes) {
    hwloc_obj_t node = hwloc_get_obj_by_depth(topology, NodeDepth, nodeIdx);

   void *addr = hwloc_alloc_membind_policy_nodeset(topology, numBytes, node->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_NOCPUBIND | HWLOC_MEMBIND_STRICT);

   // Pin memory after allocation; hwloc has no interface for 
   // allocating pinned memory directly
   checkCudaErrors(cudaHostRegister(addr, numBytes, cudaHostRegisterPortable));

   return addr;
}

void * SystemTopo::AllocWCMemByNode(int nodeIdx, long long numBytes) {
   // Get the initial topology binding to make prevent changes in 
   // binding from user perspective; must bind allocator since 
   // write-combined memory cannot be allocated directly from hwloc
   hwloc_nodeset_t initNodeSet = hwloc_bitmap_alloc();
   hwloc_membind_policy_t policy;
   hwloc_get_membind_nodeset(topology, initNodeSet, &policy, HWLOC_MEMBIND_THREAD);

   // Bind to node requested in nodeIdx
   hwloc_obj_t node = hwloc_get_obj_by_depth(topology, NodeDepth, nodeIdx);
   hwloc_nodeset_t nodeSet = node->nodeset;
   hwloc_set_membind_nodeset(topology, nodeSet, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_NOCPUBIND | HWLOC_MEMBIND_THREAD | HWLOC_MEMBIND_STRICT);
   
   // CUDA call to alloc portable pinned write combined memory
   void *blkAddr;
   checkCudaErrors(cudaHostAlloc(&blkAddr, numBytes, cudaHostAllocWriteCombined | cudaHostAllocPortable)); 

   // Reset memory binding to initial state
   hwloc_set_membind_nodeset(topology, initNodeSet, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_NOCPUBIND | HWLOC_MEMBIND_THREAD | HWLOC_MEMBIND_STRICT);

   return blkAddr;
}

void * SystemTopo::AllocManagedMemByNode(int nodeIdx, int devIdx, long long numBytes) {
   // Get the initial topology binding to make prevent changes in 
   // binding from user perspective; must bind allocator since 
   // write-combined memory cannot be allocated directly from hwloc
   hwloc_nodeset_t initNodeSet = hwloc_bitmap_alloc();
   hwloc_membind_policy_t policy;
   hwloc_get_membind_nodeset(topology, initNodeSet, &policy, HWLOC_MEMBIND_THREAD);

   // Bind to node requested in nodeIdx
   hwloc_obj_t node = hwloc_get_obj_by_depth(topology, NodeDepth, nodeIdx);
   hwloc_nodeset_t nodeSet = node->nodeset;
   hwloc_set_membind_nodeset(topology, nodeSet, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_NOCPUBIND | HWLOC_MEMBIND_THREAD | HWLOC_MEMBIND_STRICT);
   
   // CUDA call to alloc portable pinned write combined memory
   // Again save initial device state to prevent unintended behavior to caller
   void *blkAddr;
   int initDevIdx;
   checkCudaErrors(cudaGetDevice(&initDevIdx));
   checkCudaErrors(cudaSetDevice(devIdx));
   // Check CUDA flags on given device
   // cudaGetDeviceFlags();
   // cudaSetDeviceFlags();
   // This flag is implicit on init, so not nessesary except for more complicated situations (TODO)
   checkCudaErrors(cudaMallocManaged(&blkAddr, numBytes, cudaMemAttachGlobal));
   checkCudaErrors(cudaSetDevice(initDevIdx));

   // Reset memory binding to initial state
   hwloc_set_membind_nodeset(topology, initNodeSet, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_NOCPUBIND | HWLOC_MEMBIND_THREAD | HWLOC_MEMBIND_STRICT);

   return blkAddr;  

}

void * SystemTopo::AllocMappedMemByNode(int nodeIdx, int devIdx, long long numBytes) {
   // Get the initial topology binding to make prevent changes in 
   // binding from user perspective; must bind allocator since 
   // write-combined memory cannot be allocated directly from hwloc
   hwloc_nodeset_t initNodeSet = hwloc_bitmap_alloc();
   hwloc_membind_policy_t policy;
   hwloc_get_membind_nodeset(topology, initNodeSet, &policy, HWLOC_MEMBIND_THREAD);

   // Bind to node requested in nodeIdx
   hwloc_obj_t node = hwloc_get_obj_by_depth(topology, NodeDepth, nodeIdx);
   hwloc_nodeset_t nodeSet = node->nodeset;
   hwloc_set_membind_nodeset(topology, nodeSet, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_NOCPUBIND | HWLOC_MEMBIND_THREAD | HWLOC_MEMBIND_STRICT);
   
   // CUDA call to alloc portable pinned write combined memory
   // Again save initial device state to prevent unintended behavior to caller
   void *blkAddr;
   int initDevIdx;
   checkCudaErrors(cudaGetDevice(&initDevIdx));
   checkCudaErrors(cudaSetDevice(devIdx));
   checkCudaErrors(cudaHostAlloc(&blkAddr, numBytes, cudaHostAllocPortable | cudaHostAllocMapped));
   checkCudaErrors(cudaSetDevice(initDevIdx));

   // Reset memory binding to initial state
   hwloc_set_membind_nodeset(topology, initNodeSet, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_NOCPUBIND | HWLOC_MEMBIND_THREAD | HWLOC_MEMBIND_STRICT);

   return blkAddr;  
}

void * SystemTopo::AllocDeviceMem(int devIdx, long long numBytes) {
   int currDevice = 0;
   void * addr;

   checkCudaErrors(cudaGetDevice(&currDevice));

   checkCudaErrors(cudaSetDevice(devIdx));
   checkCudaErrors(cudaMalloc(&addr, numBytes));

   checkCudaErrors(cudaSetDevice(currDevice));

   return addr;
}

// ----------------------- Memory Deallocation Functions --------------------------------

void SystemTopo::FreeHostMem(void *addr, long long numBytes) {
   hwloc_free(topology, addr, numBytes);
}  

void SystemTopo::FreeWCMem(void *addr) {
   checkCudaErrors(cudaFreeHost(addr));
}
 
void SystemTopo::FreePinMem(void *addr, long long numBytes) {
   checkCudaErrors(cudaHostUnregister((void*) addr));
   hwloc_free(topology, addr, numBytes);
}

void SystemTopo::FreeMappedMem(void *addr) {
   checkCudaErrors(cudaFreeHost(addr));
}

void SystemTopo::FreeManagedMem(void *addr) {
   checkCudaErrors(cudaFree(addr));
}

void SystemTopo::FreeDeviceMem(void *addr, int deviceIdx) {
    int currDevice = 0;

   checkCudaErrors(cudaGetDevice(&currDevice));

   checkCudaErrors(cudaSetDevice(deviceIdx));
   checkCudaErrors(cudaFree(addr));

   checkCudaErrors(cudaSetDevice(currDevice));
}

// -------------------------- Other Memory Utility Functions ---------------------------

void SystemTopo::PinHostMemory(void *addr, long long numBytes) {
   checkCudaErrors(cudaHostRegister(addr, numBytes, cudaHostRegisterPortable));
}

void SystemTopo::SetDeviceMem(void *addr, int value, long long numBytes, int deviceIdx) {
    int currDevice = 0;

   checkCudaErrors(cudaGetDevice(&currDevice));

   checkCudaErrors(cudaSetDevice(deviceIdx));
   checkCudaErrors(cudaMemset(addr, value, numBytes));

   checkCudaErrors(cudaSetDevice(currDevice));
}

void SystemTopo::SetHostMem(void *addr, int value, long long numBytes) {
   memset(addr, value, numBytes);
}

// ------------------- UVA and P2P Utility Functions ---------------------------------

bool SystemTopo::DeviceUVA(int deviceIdx) {
   return (devProps[deviceIdx].unifiedAddressing ? true : false);
}
  
bool SystemTopo::DeviceGroupUVA(int deviceA, int deviceB) {
   bool aUVA = (devProps[deviceA].unifiedAddressing ? true : false);
   bool bUVA = (devProps[deviceB].unifiedAddressing ? true : false);

   return aUVA && bUVA;
}

bool SystemTopo::DeviceGroupCanP2P(int deviceA, int deviceB) {
   int aCanUVA = 0, bCanUVA = 0;
   
   checkCudaErrors(cudaDeviceCanAccessPeer(&aCanUVA, deviceA, deviceB));
   checkCudaErrors(cudaDeviceCanAccessPeer(&bCanUVA, deviceB, deviceA));
   return ((aCanUVA && bCanUVA) ? true : false);
}

void SystemTopo::DeviceGroupSetP2P(int deviceA, int deviceB, bool status) {
   int currDevice = 0;

   checkCudaErrors(cudaGetDevice(&currDevice));

   if (status) {
         checkCudaErrors(cudaSetDevice(deviceB));
         checkCudaErrors(cudaDeviceEnablePeerAccess(deviceA, 0));
         checkCudaErrors(cudaSetDevice(deviceA));
         checkCudaErrors(cudaDeviceEnablePeerAccess(deviceB, 0));
   } else {
         checkCudaErrors(cudaSetDevice(deviceB));
         checkCudaErrors(cudaDeviceDisablePeerAccess(deviceA));
         checkCudaErrors(cudaSetDevice(deviceA));
         checkCudaErrors(cudaDeviceDisablePeerAccess(deviceB));
   }

   checkCudaErrors(cudaSetDevice(currDevice));
}

// ---------------------------- NUMA and CPUSET Pinning Functions ----------------------

void SystemTopo::PinNumaNode(int nodeIdx) {
   hwloc_obj_t node = hwloc_get_obj_by_depth(topology, NodeDepth, nodeIdx); 

   hwloc_nodeset_t nodeSet = node->nodeset;

   hwloc_set_membind_nodeset(topology, nodeSet, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_NOCPUBIND | HWLOC_MEMBIND_THREAD | HWLOC_MEMBIND_STRICT);

}

void SystemTopo::PinSocket(int socketIdx) {
   hwloc_obj_t socket = hwloc_get_obj_by_depth(topology, SocketDepth, socketIdx);
   
   hwloc_set_cpubind(topology, socket->cpuset, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_NOMEMBIND | HWLOC_CPUBIND_STRICT);
}

void SystemTopo::PinCore(int coreIdx) {
   hwloc_obj_t core = hwloc_get_obj_by_depth(topology, CoreDepth, coreIdx);

   hwloc_set_cpubind(topology, core->cpuset, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_NOMEMBIND | HWLOC_CPUBIND_STRICT);
}

void SystemTopo::PinPU(int puIdx) {
   hwloc_obj_t pu = hwloc_get_obj_by_depth(topology, PUDepth, puIdx);

   hwloc_set_cpubind(topology, pu->cpuset, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_NOMEMBIND | HWLOC_CPUBIND_STRICT);
}

void SystemTopo::PinPUBySocket(int socketIdx, int puIdx) {
   hwloc_obj_t socket = hwloc_get_obj_by_depth(topology, SocketDepth, socketIdx);
   hwloc_obj_t pu = hwloc_get_obj_inside_cpuset_by_depth(topology, socket->cpuset, CoreDepth, puIdx);
   
   hwloc_set_cpubind(topology, pu->cpuset, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_NOMEMBIND | HWLOC_CPUBIND_STRICT);

}

void SystemTopo::PinCoreBySocket(int socketIdx, int coreIdx) {
   hwloc_obj_t socket = hwloc_get_obj_by_depth(topology, SocketDepth, socketIdx);
   hwloc_obj_t core = hwloc_get_obj_inside_cpuset_by_depth(topology, socket->cpuset, CoreDepth, coreIdx);
   
   hwloc_set_cpubind(topology, core->cpuset, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_NOMEMBIND | HWLOC_CPUBIND_STRICT);
}


// --------------------------- Device Utility Functions -------------------------------

void SystemTopo::SetActiveDevice(int devIdx) {
   checkCudaErrors(cudaSetDevice(devIdx));
}
 
std::string SystemTopo::GetDeviceName(int devIdx) {
   return std::string(devProps[devIdx].name);
}

std::vector<std::vector<int> > SystemTopo::GetPeerGroups() {
   return PeerGroups;
}
// ------------------------------- System Topology Info ----------------------------

int SystemTopo::NumPeerGroups() {
   return PeerGroupCount;
}     
 
void SystemTopo::PrintTopology(std::ofstream &OutFile) {
   int s_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_SOCKET);
   int c_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
   int p_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);

   hwloc_obj_t s_obj;// = hwloc_get_obj_by_depth(topology, s_depth, 0);
   hwloc_obj_t c_obj;// = hwloc_get_obj_by_depth(topology, c_depth, 0);
   hwloc_obj_t p_obj;// = hwloc_get_obj_by_depth(topology, p_depth, 0);

   char *str;

   std::stringstream outTopoStr;

   outTopoStr << "-----------------------------------------------------------------" << std::endl;
   outTopoStr << "------------------------ System Topology ------------------------" << std::endl;
   outTopoStr << "-----------------------------------------------------------------" << std::endl;
   outTopoStr << "\tSockets:\t\t" << SocketsInSystem << std::endl; 
   outTopoStr << "\tNUMA Nodes:\t\t" << NodesInSystem << std::endl; 
   outTopoStr << "\tTotal Cores\t\t" << CoresInSystem << std::endl;
   outTopoStr << "\tTotal PUs:\t\t" << PUsInSystem << std::endl;
   outTopoStr << "\tCores Per Socket:\t" << CoresPerSocket << std::endl;
   outTopoStr << "\tPUs Per Core:\t\t" << PUsPerCore << std::endl;
   outTopoStr << "\tHyperthreaded:\t\t" << std::boolalpha << HyperThreaded << std::noboolalpha << std::endl;
   outTopoStr << "\tSymmetric Topology:\t" << std::boolalpha << SymmetricTopo << std::noboolalpha << std::endl;
   outTopoStr << "\tGPU Accelerators:\t" << NumDevices << std::endl;
   outTopoStr << "-----------------------------------------------------------------" << std::endl;
   outTopoStr << "------------------------- Topology Tree -------------------------" << std::endl;
   outTopoStr << "-----------------------------------------------------------------" << std::endl;
   
   int m_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_MACHINE); 
   hwloc_obj_t m_obj = hwloc_get_obj_by_depth(topology, m_depth, 0);

   hwloc_bitmap_asprintf(&str, m_obj->cpuset);
   outTopoStr << "Machine: " << "P#" << m_obj->os_index << " CPUSET=" << str << std::endl; 
   free(str); 
   
   for (int sNum = 0; sNum < SocketsInSystem; sNum++) {
      s_obj = hwloc_get_obj_by_depth(topology, s_depth, sNum);
      
      hwloc_bitmap_asprintf(&str, s_obj->cpuset);
      outTopoStr << "\tSocket: " << s_obj->os_index << " " << str << std::endl;
      free(str);
     
      for (int cNum = 0; cNum < SocketsInSystem * CoresPerSocket; cNum++) {    
         c_obj = hwloc_get_obj_by_depth(topology, c_depth, cNum);
         
         if (hwloc_obj_is_in_subtree(topology, c_obj, s_obj)) {  
            hwloc_bitmap_asprintf(&str, c_obj->cpuset);
            outTopoStr << "\t\tCore:" << " L#" << c_obj->logical_index << " P#" << c_obj->os_index << " CPUSET=" << str << std::endl;
            free(str);
      
            for (int pNum = 0; pNum < SocketsInSystem * PUsPerCore * CoresPerSocket; pNum++) {
               p_obj = hwloc_get_obj_by_depth(topology, p_depth, pNum);

               if (hwloc_obj_is_in_subtree(topology, p_obj, c_obj)) {  
                  hwloc_bitmap_asprintf(&str, p_obj->cpuset); 
                  outTopoStr << "\t\t\tPU:" << " L#" << p_obj->logical_index << " P#" << p_obj->os_index << " CPUSET=" << str << std::endl;
                  free(str);
               }
            }
         }
      }
   }


   std::cout << outTopoStr.str();
   if (OutFile.is_open()) {
      OutFile << outTopoStr.str();
   }
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

   checkCudaErrors(cudaGetDeviceCount(&NumDevices));
   GetAllDeviceProps();

   PeerGroupCount = 0;
   std::vector<bool> inGroup(NumDevices, false);

   if (NumDevices > 0) {
      for (int i = 0; i < NumDevices; i++) {
         if (inGroup[i])
            continue;
         
         std::vector<int> group;
         group.push_back(i);
         inGroup[i] = true;

         for (int j = i + 1; j < NumDevices; j++) {
            if (DeviceGroupCanP2P(i,j)) {
               group.push_back(j);
               inGroup[j] = true;
            }
         }
         PeerGroups.push_back(group);
      } 
   }
   PeerGroupCount = PeerGroups.size();
}

// Prints the device properties out to file based named depending on the 
void SystemTopo::PrintDeviceProps(BenchParams &params) {
   std::string devFileName = "./results/" + params.devPropFile;
   std::ofstream devicePropsFile(devFileName.c_str());
   std::stringstream devicePropsSS;

   devicePropsSS << "\n-----------------------------------------------------------------" << std::endl;
   devicePropsSS << "------------------------ Device Properties ----------------------" << std::endl;
   devicePropsSS << "-----------------------------------------------------------------" << std::endl;

   cudaDeviceProp *props = devProps;

   int driverVersion = 0, runtimeVersion = 0;
   for (int i = 0; i < NumDevices; i++) {
      checkCudaErrors(cudaSetDevice(i));
      checkCudaErrors(cudaDriverGetVersion(&driverVersion));
      checkCudaErrors(cudaDriverGetVersion(&runtimeVersion));

      devicePropsSS << "Device " << i << ":\t\t\t\t" <<props[i].name << std::endl;
      devicePropsSS << "CUDA Capability:\t\t\t" << props[i].major << "." << props[i].minor << std::endl;
      devicePropsSS << "Driver Version / Runtime Version:\t" << (driverVersion / 1000) << "." << ((float) (driverVersion % 100) / 10) << " / " << (runtimeVersion / 1000) << "." << ((float) (runtimeVersion % 100) / 10) << std::endl;
      devicePropsSS << "PCI Domain/Bus/Device ID:\t\t" <<   props[i].pciDomainID << ":" <<  props[i].pciBusID << ":" <<  props[i].pciDeviceID << std::endl; 
      devicePropsSS << "Device Clock:\t\t\t\t" << ((float) props[i].clockRate * 1e-3f) << " (MHz)" << std::endl; 
      devicePropsSS << "Memory Clock:\t\t\t\t" << ((float) props[i].memoryClockRate * 1e-3f) << " (MHz)" << std::endl; 
      devicePropsSS << "Global Memory Bus Width:\t\t" << props[i].memoryBusWidth << " (Bits)" << std::endl; 
      devicePropsSS << "Theoretical Memory BW:\t\t\t" << (((float) props[i].memoryClockRate * props[i].memoryBusWidth * 2) / 8.0 / pow(2,20)) << " (GB/s)" << std::endl;
      devicePropsSS << "Global Memory Size:\t\t\t" << (props[i].totalGlobalMem / pow(2.0,20.0)) << " (MB)" << std::endl;
      devicePropsSS << "Shared Memory Per Block:\t\t" << props[i].sharedMemPerBlock << " (Bytes)" << std::endl;
      devicePropsSS << "Shared Memory Per Multiprocessor:\t" << props[i].sharedMemPerMultiprocessor << " (Bytes)" << std::endl;
      devicePropsSS << "Total Constant Memory:\t\t\t" << props[i].totalConstMem << " (Bytes)" << std::endl;
      devicePropsSS << "L2 Cache Size:\t\t\t\t" << ((float) props[i].l2CacheSize / pow(2.0, 10.0)) << " (KB)" << std::endl;
      devicePropsSS << std::boolalpha;
      devicePropsSS << "UVA Support:\t\t\t\t" << (props[i].unifiedAddressing ? true : false) << std::endl;
      devicePropsSS << "Managed Memory Support:\t\t\t" << (props[i].managedMemory ? true : false) << std::endl;
      devicePropsSS << "Mapped Memory Support:\t\t\t" << (props[i].canMapHostMemory ? true : false) << std::endl;
      devicePropsSS << "Global L1 Cache Support:\t\t" << (props[i].globalL1CacheSupported ? true : false) << std::endl;
      devicePropsSS << "Local L1 Cache Support:\t\t\t" << (props[i].localL1CacheSupported ? true : false) << std::endl;
      devicePropsSS << "ECC Enables:\t\t\t\t" << (props[i].ECCEnabled ? true : false) << std::endl;
      devicePropsSS << "Multi-GPU Board:\t\t\t" << (props[i].isMultiGpuBoard ? true : false) << std::endl;
      devicePropsSS << "Multi-GPU Board Group ID:\t\t" << props[i].multiGpuBoardGroupID << std::endl;
      devicePropsSS << "Comm/Exec Overlap Support:\t\t" << (props[i].asyncEngineCount ? true : false) << std::endl;
      devicePropsSS << "Async Engine Count:\t\t\t" << props[i].asyncEngineCount << std::endl;
      devicePropsSS << "Compute Mode:\t\t\t\t" << props[i].computeMode << std::endl;
      devicePropsSS << "Integrated Device:\t\t\t" << (props[i].integrated ? true : false) << std::endl;
      devicePropsSS << std::noboolalpha;
      devicePropsSS << "-----------------------------------------------------------------" << std::endl;
   }

   if (params.printDevProps) 
      std::cout << devicePropsSS.str(); 
   else
      std::cout << "\nSee " << params.devPropFile << " for information about your device's properties." << std::endl; 
   devicePropsFile << devicePropsSS.str();
   
   devicePropsFile.close();
}

//SystemTopo Constructor, initializes hwloc topology
SystemTopo::SystemTopo() {
   InitTopology();
   ParseTopology();
}

//SystemTopo destructor, free hwloc topology
SystemTopo::~SystemTopo() {
   hwloc_topology_destroy(topology);
   free(devProps);
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

int SystemTopo::NumPUsPerSocket() {
   return PUsPerCore * CoresPerSocket;
}

int SystemTopo::NumGPUs() {
   return NumDevices;
}

// function for cleaning up device state including profile data
// to be used before and after any test in benchmark suite.
void SystemTopo::ResetDevices() {
   for (int devNum = 0; devNum < NumDevices; ++devNum) {
      checkCudaErrors(cudaSetDevice(devNum));
      checkCudaErrors(cudaDeviceReset());
   }
}

// Creates an array of cudaDeviceProp structs with populated data
// located in a pre-allocated section of memory
void SystemTopo::GetAllDeviceProps() {
   devProps = (cudaDeviceProp *) calloc (sizeof(cudaDeviceProp), NumDevices);
   
   for (int i = 0; i < NumDevices; ++i) {
      checkCudaErrors(cudaGetDeviceProperties(&devProps[i], i));
   }
}

