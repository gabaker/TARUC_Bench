
#include "topology.h"

// --------------------------- Memory Allocation Functions -----------------------------

// Allocate pageable memory by core
void * SystemTopo::AllocMemByCore(int CoreIdx, long long NumBytes) {

   // Get requested core object from hwloc topology tree
   hwloc_obj_t core = hwloc_get_obj_by_depth(Topology, CoreDepth, CoreIdx);

   // Allocate the memory on specified core without changing the CPU thread pinning policy
   void *MemBlock = hwloc_alloc_membind_policy(Topology, NumBytes, core->nodeset, HWLOC_MEMBIND_BIND, 
																											 HWLOC_MEMBIND_BYNODESET |
                                                                                  HWLOC_MEMBIND_NOCPUBIND | 
                                                                                  HWLOC_MEMBIND_THREAD | 
                                                                                  HWLOC_MEMBIND_STRICT);

   // Check for errors thrown during allocation by hwloc
   // errors returned along with NULL
   if (MemBlock == NULL)
      perror("Error during AllocMemByCore()");

   return MemBlock;
}

// Allocate pageable memory by NUMA node
void * SystemTopo::AllocMemByNode(int NodeIdx, long long NumBytes) {

   // Get node object from hwloc topology tree
   hwloc_obj_t node = hwloc_get_obj_by_depth(Topology, NodeDepth, NodeIdx);

   void *MemBlock = hwloc_alloc_membind_policy(Topology, NumBytes, node->nodeset, HWLOC_MEMBIND_BIND, 
                                                                                  HWLOC_MEMBIND_BYNODESET |
                                                                                  HWLOC_MEMBIND_NOCPUBIND |
                                                                                  HWLOC_MEMBIND_THREAD | 
                                                                                  HWLOC_MEMBIND_STRICT);
   // Check for errors thrown during allocation by hwloc
   // errors returned along with NULL
   if (MemBlock == NULL)
      perror("Error during AllocMemByNode()");

   return MemBlock;
}

// Allocate pageable host memory block by socket
void * SystemTopo::AllocMemBySocket(int SocketIdx, long long NumBytes) {
   
   // Get CPU socket (vs NUMA node) object from hwloc topology tree
   hwloc_obj_t socket = hwloc_get_obj_by_depth(Topology, SocketDepth, SocketIdx);

   void *MemBlock = hwloc_alloc_membind_policy(Topology, NumBytes, socket->nodeset, HWLOC_MEMBIND_BIND, 
																												HWLOC_MEMBIND_BYNODESET |
                                                                                    HWLOC_MEMBIND_NOCPUBIND | 
                                                                                    HWLOC_MEMBIND_THREAD | 
                                                                                    HWLOC_MEMBIND_STRICT);
   // Check for errors thrown during allocation by hwloc
   // errors returned along with NULL
   if (MemBlock == NULL)
      perror("Error during AllocMemBySocket()");

   return MemBlock;
}

// Allocate pinned host memory block by NUMA node
void * SystemTopo::AllocPinMemByNode(int NodeIdx, long long NumBytes) {

   // Get NUMA node object from hwloc topology tree
   hwloc_obj_t node = hwloc_get_obj_by_depth(Topology, NodeDepth, NodeIdx);

   void *MemBlock = hwloc_alloc_membind_policy(Topology, NumBytes, node->nodeset, HWLOC_MEMBIND_BIND, 
                                                                                  HWLOC_MEMBIND_BYNODESET |
                                                                                  HWLOC_MEMBIND_NOCPUBIND |
                                                                                  HWLOC_MEMBIND_THREAD | 
                                                                                  HWLOC_MEMBIND_STRICT);
   // Pin memory after allocation; hwloc has no interface for 
   // allocating pinned memory directly
   if (MemBlock == NULL)
      perror("Error during AllocPinMemByNode()");

   // Register memory as pinned with CUDA runtime (alternative to mlock that is cuda aware)
   checkCudaErrors(cudaHostRegister(MemBlock, NumBytes, cudaHostRegisterPortable));
   
   return MemBlock;
}

// Allocate write combined memory block by NUMA node index
void * SystemTopo::AllocWCMemByNode(int NodeIdx, long long NumBytes) {

   // Get the initial topology binding to make prevent changes in 
   // binding from user perspective; must bind allocator since 
   // write-combined memory cannot be allocated directly from hwloc
   hwloc_nodeset_t initNodeSet = hwloc_bitmap_alloc();
   hwloc_membind_policy_t policy;
   
   int error = hwloc_get_membind(Topology, initNodeSet, &policy, HWLOC_MEMBIND_BYNODESET |
																									HWLOC_MEMBIND_THREAD); 
   if (error < 0)
      perror("Error during AllocWCMemByNode() trying to get existing membind policy from hwloc_get_membind()");

   // Bind to node requested in NodeIdx
   hwloc_obj_t node = hwloc_get_obj_by_depth(Topology, NodeDepth, NodeIdx);
   hwloc_nodeset_t nodeSet = node->nodeset;
   
	error = hwloc_set_membind(Topology, nodeSet, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET | 
																										HWLOC_MEMBIND_NOCPUBIND |
																										HWLOC_CPUBIND_STRICT |
																										HWLOC_MEMBIND_THREAD);
   if (error < 0)
      perror("Error during AllocWCMemByNode() trying to set new membind policy with hwloc_set_membind()");
   
   // CUDA call to allocate portable pinned write combined memory
   void *MemBlock;
   checkCudaErrors(cudaHostAlloc(&MemBlock, NumBytes, cudaHostAllocWriteCombined | cudaHostAllocPortable)); 

   // Reset memory binding to initial state
   error = hwloc_set_membind(Topology, initNodeSet, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET | 
																											 HWLOC_MEMBIND_NOCPUBIND |
																											 HWLOC_CPUBIND_STRICT |
																											 HWLOC_MEMBIND_THREAD);
    if (error < 0)
      perror("Error during AllocWCMemByNode() trying to reset membind policy to previous type with hwloc_set_membind()");

   return MemBlock;
}

// Allocate CUDA managed memory block
// This function is experimental
void * SystemTopo::AllocManagedMemByNode(int NodeIdx, int DevIdx, long long NumBytes) {
   // Get the initial topology binding to make prevent changes in 
   // binding from user perspective; must bind allocator since 
   // write-combined memory cannot be allocated directly from hwloc
   hwloc_nodeset_t initNodeSet = hwloc_bitmap_alloc();
   hwloc_membind_policy_t policy;
   int error = hwloc_get_membind(Topology, initNodeSet, &policy, HWLOC_MEMBIND_BYNODESET |
																									HWLOC_MEMBIND_THREAD);    
	if (error < 0)
      perror("Error during AllocManagedMemByNode() trying to get existing membind policy from hwloc_get_membind()");

   // Bind to node requested in NodeIdx
   hwloc_obj_t node = hwloc_get_obj_by_depth(Topology, NodeDepth, NodeIdx);
   hwloc_nodeset_t nodeSet = node->nodeset;
   
	error = hwloc_set_membind(Topology, nodeSet, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET | 
																										HWLOC_MEMBIND_NOCPUBIND |
																										HWLOC_CPUBIND_STRICT |
																										HWLOC_MEMBIND_THREAD);
    if (error < 0)
      perror("Error during AllocManagedMemByNode() trying to set new membind policy with hwloc_set_membind()");

  
   // CUDA call to alloc portable pinned write combined memory
   // Again save initial device state to prevent unintended behavior to caller
   void *MemBlock;
   int initDevIdx;
   checkCudaErrors(cudaGetDevice(&initDevIdx));
   checkCudaErrors(cudaSetDevice(DevIdx));
   checkCudaErrors(cudaMallocManaged(&MemBlock, NumBytes, cudaMemAttachGlobal));
   checkCudaErrors(cudaSetDevice(initDevIdx));

   // Reset memory binding to initial state
   
	error = hwloc_set_membind(Topology, initNodeSet, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET | 
																											 HWLOC_MEMBIND_NOCPUBIND |
																											 HWLOC_CPUBIND_STRICT |
																											 HWLOC_MEMBIND_THREAD);
   if (error < 0)
      perror("Error during AllocManagedMemByNode() trying to reset initial membind policy with hwloc_set_membind()");

   return MemBlock;  

}

// Allocate CUDA device mapped memory 
// This function is experimental; likely errors will occur for off socked host/device memory mappings
void * SystemTopo::AllocMappedMemByNode(int NodeIdx, int DevIdx, long long NumBytes) {
   // Get the initial topology binding to make prevent changes in 
   // binding from user perspective; must bind allocator since 
   // write-combined memory cannot be allocated directly from hwloc
   hwloc_nodeset_t initNodeSet = hwloc_bitmap_alloc();
   hwloc_membind_policy_t policy;
   int error = hwloc_get_membind(Topology, initNodeSet, &policy, HWLOC_MEMBIND_BYNODESET |
																									HWLOC_MEMBIND_THREAD); 
   if (error < 0)
      perror("Error during AllocManagedMemByNode() trying to get existing membind policy from hwloc_get_membind()");

   // Bind to node requested in NodeIdx
   hwloc_obj_t node = hwloc_get_obj_by_depth(Topology, NodeDepth, NodeIdx);
   hwloc_nodeset_t nodeSet = node->nodeset;
   
	error = hwloc_set_membind(Topology, nodeSet, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET | 
																										HWLOC_MEMBIND_NOCPUBIND |
																										HWLOC_CPUBIND_STRICT |
																										HWLOC_MEMBIND_THREAD);
   if (error < 0)
      perror("Error during AllocManagedMemByNode() trying to set new membind policy with hwloc_set_membind()");
  
   // CUDA call to alloc portable pinned write combined memory
   // Again save initial device state to prevent unintended behavior to caller
   void *blkAddr;
   int initDevIdx;
   checkCudaErrors(cudaGetDevice(&initDevIdx));
   checkCudaErrors(cudaSetDevice(DevIdx));
   checkCudaErrors(cudaHostAlloc(&blkAddr, NumBytes, cudaHostAllocPortable | cudaHostAllocMapped));
   checkCudaErrors(cudaSetDevice(initDevIdx));

   // Reset memory binding to initial state
   
	error = hwloc_set_membind(Topology, initNodeSet, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET | 
																											 HWLOC_MEMBIND_NOCPUBIND |
																											 HWLOC_CPUBIND_STRICT |
																											 HWLOC_MEMBIND_THREAD);
   if (error < 0)
      perror("Error during AllocMappedMemByNode() trying to reset initial membind policy from hwloc_set_membind()");

   return blkAddr;  
}

// Allocate device memory block on specified device
// Save current device state (current set device)
void * SystemTopo::AllocDeviceMem(int DevIdx, long long NumBytes) {
   int currDevice = 0;
   void * Addr;
   
   // Get initial active device for later
   checkCudaErrors(cudaGetDevice(&currDevice));

   // Set current device to device for allocation
   checkCudaErrors(cudaSetDevice(DevIdx));
   checkCudaErrors(cudaMalloc(&Addr, NumBytes));

   // Return active device to initial device index
   checkCudaErrors(cudaSetDevice(currDevice));

   return Addr;
}


// ----------------------- Memory Deallocation Functions --------------------------------

// Free memory allocated with HWLOC calls
void SystemTopo::FreeHostMem(void *Addr, long long NumBytes) {
   hwloc_free(Topology, Addr, NumBytes);
}  

// Free host write-combined memory allocated with CUDA runtime
void SystemTopo::FreeWCMem(void *Addr) {
   checkCudaErrors(cudaFreeHost(Addr));
}

// Free host pinned memory
// Unregister memory with CUDA runtime and then free with HWLOC 
void SystemTopo::FreePinMem(void *Addr, long long NumBytes) {
   checkCudaErrors(cudaHostUnregister((void*) Addr));
   hwloc_free(Topology, Addr, NumBytes);
}

// Free CUDA device/host mapped memory block
void SystemTopo::FreeMappedMem(void *Addr) {
   checkCudaErrors(cudaFreeHost(Addr));
}

// Free CUDA device/host managed memory block
void SystemTopo::FreeManagedMem(void *Addr) {
   checkCudaErrors(cudaFree(Addr));
}

// Free device memory; save current device state
void SystemTopo::FreeDeviceMem(void *Addr, int DevIdx) {
    int currDevice = 0;

   checkCudaErrors(cudaGetDevice(&currDevice));

   checkCudaErrors(cudaSetDevice(DevIdx));
   checkCudaErrors(cudaFree(Addr));

   checkCudaErrors(cudaSetDevice(currDevice));
}


// -------------------------- Other Memory Utility Functions ---------------------------

// Mark a block of pageable host memory to be pinned and portable within the CUDA runtime
void SystemTopo::PinHostMemory(void *Addr, long long NumBytes) {
   checkCudaErrors(cudaHostRegister(Addr, NumBytes, cudaHostRegisterPortable));
}

// Set block of device memory located on provided device index
void SystemTopo::SetDeviceMem(void *Addr, int value, long long NumBytes, int DevIdx) {
    int currDevice = 0;

   checkCudaErrors(cudaGetDevice(&currDevice));

   checkCudaErrors(cudaSetDevice(DevIdx));
   checkCudaErrors(cudaMemset(Addr, value, NumBytes));

   checkCudaErrors(cudaSetDevice(currDevice));
}

// Set block of pageable host memory 
void SystemTopo::SetHostMem(void *Addr, int value, long long NumBytes) {
   memset(Addr, value, NumBytes);
}


// ------------------- UVA and P2P Utility Functions ---------------------------------

// Returns boolean indicating if a specific device can use inified virtual addressing
// Only useful for older GPU devices
bool SystemTopo::DeviceUVA(int DevIdx) {
   return (DevProps[DevIdx].unifiedAddressing ? true : false);
}

// Return whether UVA is supported within two objects
// Only useful for older GPU devices  
bool SystemTopo::DeviceGroupUVA(int DevA, int DevB) {
   bool aUVA = (DevProps[DevA].unifiedAddressing ? true : false);
   bool bUVA = (DevProps[DevB].unifiedAddressing ? true : false);

   return aUVA && bUVA;
}

// Return boolean indicating if two device indexs have peer to peer support
bool SystemTopo::DeviceGroupCanP2P(int DevA, int DevB) {
   int aCanP2P = 0, bCanP2P = 0;
   
   checkCudaErrors(cudaDeviceCanAccessPeer(&aCanP2P, DevA, DevB));
   checkCudaErrors(cudaDeviceCanAccessPeer(&bCanP2P, DevB, DevA));

   return ((aCanP2P && bCanP2P) ? true : false);
}

// Set the peer status of two GPU devices based on provided index and boolean balue
void SystemTopo::DeviceGroupSetP2P(int DevA, int DevB, bool Status) {
   int currDevice = 0;

   checkCudaErrors(cudaGetDevice(&currDevice));

   if (Status) {
         checkCudaErrors(cudaSetDevice(DevB));
         checkCudaErrors(cudaDeviceEnablePeerAccess(DevA, 0));
         checkCudaErrors(cudaSetDevice(DevA));
         checkCudaErrors(cudaDeviceEnablePeerAccess(DevB, 0));
   } else {
         checkCudaErrors(cudaSetDevice(DevB));
         checkCudaErrors(cudaDeviceDisablePeerAccess(DevA));
         checkCudaErrors(cudaSetDevice(DevA));
         checkCudaErrors(cudaDeviceDisablePeerAccess(DevB));
   }

   checkCudaErrors(cudaSetDevice(currDevice));
}


// ---------------------------- NUMA and CPUSET Pinning Functions ----------------------

void SystemTopo::PinNode(int NodeIdx) {
   hwloc_obj_t node = hwloc_get_obj_by_depth(Topology, NodeDepth, NodeIdx); 

   hwloc_nodeset_t nodeSet = node->nodeset;
   int error = hwloc_set_membind(Topology, nodeSet, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET | 
																							   HWLOC_MEMBIND_NOCPUBIND |
																								HWLOC_CPUBIND_STRICT |
																								HWLOC_MEMBIND_THREAD);
   if (error < 0)
      perror("Error in PinNode() while setting binding policy in hwloc_set_membind()");
}

// Pin current thread to given CPU socket 
void SystemTopo::PinSocket(int SocketIdx) {
   hwloc_obj_t socket = hwloc_get_obj_by_depth(Topology, SocketDepth, SocketIdx);
   
   int error = hwloc_set_cpubind(Topology, socket->cpuset, HWLOC_CPUBIND_THREAD | 
                                                           HWLOC_CPUBIND_NOMEMBIND | 
                                                           HWLOC_CPUBIND_STRICT);
   if (error < 0)
      perror("Error in PinSocket() while setting binding policy in hwloc_set_cpubind()");
}

// Pin current thread to given processing core index at system level
void SystemTopo::PinCore(int CoreIdx) {
   hwloc_obj_t core = hwloc_get_obj_by_depth(Topology, CoreDepth, CoreIdx);

   int error = hwloc_set_cpubind(Topology, core->cpuset, HWLOC_CPUBIND_THREAD | 
                                                         HWLOC_CPUBIND_NOMEMBIND | 
                                                         HWLOC_CPUBIND_STRICT);
   if (error < 0)
      perror("Error in PinCore() while setting binding policy in hwloc_set_cpubind()");
}

// Pin current thread to given processing unit index at the system level
void SystemTopo::PinPU(int PUIdx) {
   hwloc_obj_t pu = hwloc_get_obj_by_depth(Topology, PUDepth, PUIdx);

   int error = hwloc_set_cpubind(Topology, pu->cpuset, HWLOC_CPUBIND_THREAD | 
                                                       HWLOC_CPUBIND_NOMEMBIND | 
                                                       HWLOC_CPUBIND_STRICT);
   if (error < 0)
      perror("Error in PinPU() while setting binding policy in hwloc_set_cpubind()");
}

// Pin current thread to given processing unit (PU) by specified socket
void SystemTopo::PinPUBySocket(int SocketIdx, int PUIdx) {
   hwloc_obj_t socket = hwloc_get_obj_by_depth(Topology, SocketDepth, SocketIdx);
   hwloc_obj_t pu = hwloc_get_obj_inside_cpuset_by_depth(Topology, socket->cpuset, PUDepth, PUIdx);
   
   int error = hwloc_set_cpubind(Topology, pu->cpuset, HWLOC_CPUBIND_THREAD | 
                                                       HWLOC_CPUBIND_NOMEMBIND | 
                                                       HWLOC_CPUBIND_STRICT);
   if (error < 0)
      perror("Error in PinPUBySocket() while setting binding policy in hwloc_set_cpubind()");
}

// Pin current thread to given core with a specific socket
void SystemTopo::PinCoreBySocket(int SocketIdx, int CoreIdx) {
   hwloc_obj_t socket = hwloc_get_obj_by_depth(Topology, SocketDepth, SocketIdx);
   hwloc_obj_t core = hwloc_get_obj_inside_cpuset_by_depth(Topology, socket->cpuset, CoreDepth, CoreIdx);
   
   int error = hwloc_set_cpubind(Topology, core->cpuset, HWLOC_CPUBIND_THREAD | 
                                                         HWLOC_CPUBIND_NOMEMBIND | 
                                                         HWLOC_CPUBIND_STRICT);
   if (error < 0)
      perror("Error in PinCoreBySocket() while setting binding policy in hwloc_set_cpubind()");
}


// --------------------------- Device Utility Functions -------------------------------

// Set active GPU device within the CUDA runtime
void SystemTopo::SetActiveDevice(int DevIdx) {
   checkCudaErrors(cudaSetDevice(DevIdx));
}

// Return string of device name for device at given index 
std::string SystemTopo::GetDeviceName(int DevIdx) {
   return std::string(DevProps[DevIdx].name);
}

// Return the number of Peer groups detected during topology initialization
std::vector<std::vector<int> > SystemTopo::GetPeerGroups() {
   return PeerGroups;
}

// ------------------------------- System Topology Info ----------------------------     

// Print basic topology information and tree structure; I/O and GPUs note included 
void SystemTopo::PrintTopology(std::ofstream &OutFile) {
   int s_depth = hwloc_get_type_depth(Topology, HWLOC_OBJ_SOCKET);
   int c_depth = hwloc_get_type_depth(Topology, HWLOC_OBJ_CORE);
   int p_depth = hwloc_get_type_depth(Topology, HWLOC_OBJ_PU);

   hwloc_obj_t s_obj;
   hwloc_obj_t c_obj;
   hwloc_obj_t p_obj;

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
   outTopoStr << "\tGPU Accelerators:\t" << GPUs << std::endl;
   outTopoStr << "-----------------------------------------------------------------" << std::endl;
   outTopoStr << "------------------------- Topology Tree -------------------------" << std::endl;
   outTopoStr << "-----------------------------------------------------------------" << std::endl;
   
   int m_depth = hwloc_get_type_depth(Topology, HWLOC_OBJ_MACHINE); 
   hwloc_obj_t m_obj = hwloc_get_obj_by_depth(Topology, m_depth, 0);

   hwloc_bitmap_asprintf(&str, m_obj->cpuset);
   outTopoStr << "Machine: " << "P#" << m_obj->os_index << " CPUSET=" << str << std::endl; 
   free(str); 
   
   for (int sNum = 0; sNum < SocketsInSystem; sNum++) {
      s_obj = hwloc_get_obj_by_depth(Topology, s_depth, sNum);
      
      hwloc_bitmap_asprintf(&str, s_obj->cpuset);
      outTopoStr << "\tSocket: " << s_obj->os_index << " " << str << std::endl;
      free(str);
     
      for (int cNum = 0; cNum < SocketsInSystem * CoresPerSocket; cNum++) {    
         c_obj = hwloc_get_obj_by_depth(Topology, c_depth, cNum);
         
         if (hwloc_obj_is_in_subtree(Topology, c_obj, s_obj)) {  
            hwloc_bitmap_asprintf(&str, c_obj->cpuset);
            outTopoStr << "\t\tCore:" << " L#" << c_obj->logical_index << " P#" << c_obj->os_index << " CPUSET=" << str << std::endl;
            free(str);
      
            for (int pNum = 0; pNum < SocketsInSystem * PUsPerCore * CoresPerSocket; pNum++) {
               p_obj = hwloc_get_obj_by_depth(Topology, p_depth, pNum);

               if (hwloc_obj_is_in_subtree(Topology, p_obj, c_obj)) {  
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

// Basic topology parser; object levels and counts
// Aquire basic GPU device info from CUDA runtime
void SystemTopo::ParseTopology() {

   TopoDepth = hwloc_topology_get_depth(Topology);
   NodeDepth = hwloc_get_type_depth(Topology, HWLOC_OBJ_NUMANODE);   
   SocketDepth = hwloc_get_type_depth(Topology, HWLOC_OBJ_PACKAGE);
   PUDepth = hwloc_get_type_depth(Topology, HWLOC_OBJ_PU);
   CoreDepth = hwloc_get_type_depth(Topology, HWLOC_OBJ_CORE);
   NodesInSystem = hwloc_get_nbobjs_by_type(Topology, HWLOC_OBJ_NUMANODE);
   SocketsInSystem = hwloc_get_nbobjs_by_type(Topology, HWLOC_OBJ_PACKAGE);
   CoresInSystem = hwloc_get_nbobjs_by_type(Topology, HWLOC_OBJ_CORE);
   PUsInSystem = hwloc_get_nbobjs_by_type(Topology, HWLOC_OBJ_PU); 
/*
	for (int i = 0; i < hwloc_topology_get_depth(Topology); ++i)
		std::cout << "Depth " << i << ": " << hwloc_get_depth_type(Topology, i) << " x" << hwloc_get_nbobjs_by_depth(Topology, i) << std::endl;

	std::cout << "System: " << HWLOC_OBJ_SYSTEM << std::endl;
	std::cout << "Machine: " << HWLOC_OBJ_MACHINE << std::endl;
	std::cout << "PACKAGE: " << HWLOC_OBJ_NUMANODE << std::endl;
	std::cout << "NUMA Node: " << HWLOC_OBJ_PACKAGE << std::endl;
	std::cout << "CORE: " << HWLOC_OBJ_CORE << std::endl;
	std::cout << "PU: " << HWLOC_OBJ_PU << std::endl;
	std::cout << "L1 Cache: " << HWLOC_OBJ_L1CACHE << std::endl;
	std::cout << "L2 Cache: " << HWLOC_OBJ_L2CACHE << std::endl;
	std::cout << "L3 Cache: " << HWLOC_OBJ_L3CACHE << std::endl;
	std::cout << "L4 Cache: " << HWLOC_OBJ_L4CACHE << std::endl;
	std::cout << "L5 Cache: " << HWLOC_OBJ_L5CACHE << std::endl;


	
	std::cout << "Topology Depth: " << TopoDepth << std::endl;
	std::cout << "Node Depth: " << NodeDepth << std::endl;
	std::cout << "Socket Depth: " << SocketDepth << std::endl;
	std::cout << "Core Depth: " << CoreDepth << std::endl;
	std::cout << "PU Depth: " << PUDepth << std::endl;
	std::cout << "Nodes: " << NodesInSystem << std::endl;
	std::cout << "Sockets: " << SocketsInSystem << std::endl;
	std::cout << "Cores: " << CoresInSystem << std::endl;
	std::cout << "PUs: " << PUsInSystem << std::endl;
	*/

	CoresPerSocket = CoresInSystem / SocketsInSystem;
   PUsPerCore = PUsInSystem / CoresInSystem;
   
   HyperThreaded = (PUsPerCore != 1) ? true : false;
   SymmetricTopo = (hwloc_get_root_obj(Topology)->symmetric_subtree != 0) ? true : false;

   checkCudaErrors(cudaGetDeviceCount(&GPUs));
   GetAllDeviceProps();

   PeerGroupCount = 0;
   std::vector<bool> inGroup(GPUs, false);
   
	if (GPUs > 0) {
      for (int i = 0; i < GPUs; i++) {
         if (inGroup[i])
            continue;
         
         std::vector<int> group;
         group.push_back(i);
         inGroup[i] = true;

         for (int j = i + 1; j < GPUs; j++) {
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
void SystemTopo::PrintDeviceProps(std::string FileName) {
   std::string devFileName = "./results/" + FileName;
   std::ofstream devicePropsFile(devFileName.c_str());
   std::stringstream devicePropsSS;

   devicePropsSS << "\n-----------------------------------------------------------------" << std::endl;
   devicePropsSS << "------------------------ Device Properties ----------------------" << std::endl;
   devicePropsSS << "-----------------------------------------------------------------" << std::endl;

   cudaDeviceProp *props = DevProps;

   int driverVersion = 0, runtimeVersion = 0;
   for (int i = 0; i < GPUs; i++) {
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

   std::cout << devicePropsSS.str(); 
   devicePropsFile << devicePropsSS.str();
   
   devicePropsFile.close();
}

// Return the number of peer groups
int SystemTopo::NumPeerGroups() {
   return PeerGroupCount;
}

// Return a copy of the system topology
void SystemTopo::GetTopology(hwloc_topology_t &copy) {
   hwloc_topology_dup(&copy, Topology);
}

// Initialize and load HWLOC topology tree
void SystemTopo::InitTopology() {
   int error = hwloc_topology_init(&Topology);

   if (error < 0)
      perror("Failed to Initialize Topology");

   error = hwloc_topology_load(Topology);

   if (error < 0)
      perror("Failed to Load Topology");

   //hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_IO_DEVICES | HWLOC_TOPOLOGY_FLAG_IO_BRIDGES);
}

// Return the number of detected NUMA nodes
int SystemTopo::NumNodes() {
   return NodesInSystem;
}

// Returns the number of CPU sockets 
// (vs NUMA nodes which may be different depending on bios/system)
int SystemTopo::NumSockets() {
   return SocketsInSystem;
}

// Returns the total number of processing cores within a system
// For multi-socket systems this is sum of all cores on all sockets 
int SystemTopo::NumCores() {
   return CoresInSystem;
}

// Returns the total number of processing units (PUs) in a system
// For multi-socket systems this is sum of all PUs on all sockets 
int SystemTopo::NumPUs() {
   return PUsInSystem;
} 

// Returns number of processing cores per socket
// will return 2 for hyperthreaded systems by default
int SystemTopo::NumCoresPerSocket() {
   return CoresPerSocket;
} 

// Returns number of processing units (PUs) per processing core
int SystemTopo::NumPUsPerCore() {
   return PUsPerCore;
}

// Calculates and returns number of processing units (PUs) per CPU socket
int SystemTopo::NumPUsPerSocket() {
   return PUsPerCore * CoresPerSocket;
}

// Returns number of detected GPUs during initializatin of class by CUDA runtime
int SystemTopo::NumGPUs() {
   return GPUs;
}

// Creates an array of cudaDeviceProp structs with populated data
// located in a pre-allocated section of memory
void SystemTopo::GetAllDeviceProps() {
   DevProps = (cudaDeviceProp *) calloc (sizeof(cudaDeviceProp), GPUs);
   
   for (int i = 0; i < GPUs; ++i)
      checkCudaErrors(cudaGetDeviceProperties(&DevProps[i], i));
}

//------------------------ Object Constructor/Destructors ------------------------

//SystemTopo Constructor, initializes hwloc topology and parses tree for basic information
SystemTopo::SystemTopo() {
   InitTopology();
   ParseTopology();
}

// SystemTopo destructor, free hwloc topology object as well as device props array
SystemTopo::~SystemTopo() {
   hwloc_topology_destroy(Topology);
   free(DevProps);
}

