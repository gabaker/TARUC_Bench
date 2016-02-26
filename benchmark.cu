
// Benchmark includes and defines
#ifndef BENCH_HEADER_INC
#define BENCH_HEADER_INC
#include "bench.h"
#endif

// BenchParams class definition
#ifndef PARAM_CLASS_INC
#include "params.h"
#define PARAM_CLASS_INC
#endif

// SystemTopo class definition
#ifndef TOPOLOGY_CLASS_INC
#include "topology.h"
#define TOPOLOGY_CLASS_INC
#endif

//Benchmark Tests
void RunBandwidthTestSuite(BenchParams &params, SystemTopo &topo);
void TestMemoryOverhead(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo);
void TestHostDeviceBandwidth(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo);
void TestP2PDeviceBandwidth(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo);
void TestPCIeCongestion(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo);
void TestTaskScalability(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo);

//Device Properties
void GetAllDeviceProps(cudaDeviceProp *props, int dCount);
void PrintDeviceProps(cudaDeviceProp *props, BenchParams &params);
void ResetDevices(int numToReset);

//Results output
void PrintResults(std::ofstream &outFile, std::vector<long> &steps, std::vector<std::vector<float> > &results, BenchParams &params);

int main (int argc, char **argv) {
   BenchParams params;  
   SystemTopo topo;
   
   std::cout << "Starting Multi-GPU Performance Test Suite...\n" << std::endl; 
   
   // Determine the number of recognized CUDA enabled devices
   cudaGetDeviceCount(&(params.nDevices));

   if (params.nDevices <= 0) {
      std::cout << "No devices found...aborting benchmarks." << std::endl;
      exit(-1);
   }

   // Setup benchmark parameters
   if (argc == 1) { //No input file, use default parameters
   
      params.SetDefault();
   
   } else if (argc == 2) { //Parse input file
      
      params.ParseParamFile(std::string(argv[1]));
       
   } else { //Unknown input parameter list, abort test
      
      std::cout << "Aborting test: Incorrect number of input parameters" << std::endl;
      exit(-1);
   
   }

   std::string topoFileName ="./results/topology.out";
   std::ofstream topoFile(topoFileName.c_str());
   if (params.runTopoAware)
      topo.PrintTopology(topoFile);

   params.PrintParams();
   RunBandwidthTestSuite(params, topo);

   return 0;
}

void RunBandwidthTestSuite(BenchParams &params, SystemTopo &topo) {
   cudaDeviceProp *props = (cudaDeviceProp *) calloc (sizeof(cudaDeviceProp), params.nDevices);

   // Aquire device properties for each CUDA enabled GPU
   GetAllDeviceProps(props, params.nDevices);

   if (params.runMemoryOverheadTest != false ) {
      
      TestMemoryOverhead(props, params, topo);
   
   }

   if (params.runHostDeviceBandwidthTest != false) {

      TestHostDeviceBandwidth(props, params, topo);

   }

   if (params.runP2PBandwidthTest != false) {  
      
      TestP2PDeviceBandwidth(props, params, topo);
   
   }

   if (params.runPCIeCongestionTest != false) {

      TestPCIeCongestion(props, params, topo);

   }

   if (params.runTaskScalabilityTest != false) { 

      TestTaskScalability(props, params, topo);

   }

   // Output device properties for each CUDA enabled GPU
   if (params.printDevProps != false) {
      PrintDeviceProps(props, params);
   }

   std::cout << "\nBenchmarks complete!\n" << std::endl;

   free(props);
}

float TimedMemOp(void **MemBlk, long NumBytes, MEM_OP TimedOp) {
   #ifdef USING_CPP
   std::chrono::high_resolution_clock::time_point start_c, stop_c;
   auto total_c = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_c - start_c);
   #else
   struct timeval stop_t, start_t, total_t;
   #endif
   
   cudaEvent_t start_e, stop_e; 
   checkCudaErrors(cudaEventCreate(&start_e));
   checkCudaErrors(cudaEventCreate(&stop_e));
   float OpTime = 0;
  
   switch (TimedOp) {
      case HOST_MALLOC:
         #ifdef USING_CPP
         start_c = std::chrono::high_resolution_clock::now();
         *MemBlk = malloc(NumBytes);
         stop_c = std::chrono::high_resolution_clock::now();
         total_c = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_c - start_c);      
         OpTime = (float) total_c.count() * NANO_TO_MILLI;
         #else
         gettimeofday(&start_t, NULL);
         *MemBlk = malloc(NumBytes); 
         gettimeofday(&stop_t, NULL);
         timersub(&stop_t, &start_t, &total_t);
         OpTime = (float) total_t.tv_usec * MICRO_TO_MILLI;
         #endif
         break;
      case HOST_PINNED_MALLOC:
         cudaEventRecord(start_e, 0);      
         cudaMallocHost(MemBlk, NumBytes);
         cudaEventRecord(stop_e, 0);
         cudaEventSynchronize(stop_e);
         cudaEventElapsedTime(&OpTime, start_e, stop_e);
         break;
      case DEVICE_MALLOC:
         checkCudaErrors(cudaEventRecord(start_e, 0));
         checkCudaErrors(cudaMalloc(MemBlk, NumBytes));
         checkCudaErrors(cudaEventRecord(stop_e, 0));
         checkCudaErrors(cudaEventSynchronize(stop_e));
         checkCudaErrors(cudaEventElapsedTime(&OpTime, start_e, stop_e)); 
         break;
      case HOST_FREE:
         #ifdef USING_CPP
         start_c = std::chrono::high_resolution_clock::now();
         free(*MemBlk);
         stop_c = std::chrono::high_resolution_clock::now(); 
         total_c = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_c - start_c);
         OpTime = (float) total_c.count() * NANO_TO_MILLI;
         #else
         gettimeofday(&start_t, NULL);
         free(*MemBlk); 
         gettimeofday(&stop_t, NULL); 
         timersub(&stop_t, &start_t, &total_t); 
         OpTime = (float) total_t.tv_usec * MICRO_TO_MILLI;
         #endif
         break;
      case HOST_PINNED_FREE:
         cudaEventRecord(start_e, 0);
         cudaFreeHost(*MemBlk);
         cudaEventRecord(stop_e, 0);
         cudaEventSynchronize(stop_e);
         cudaEventElapsedTime(&OpTime, start_e, stop_e);
         break;
      case DEVICE_FREE:
         checkCudaErrors(cudaEventRecord(start_e, 0));
         checkCudaErrors(cudaFree(*MemBlk)); 
         checkCudaErrors(cudaEventRecord(stop_e, 0));
         checkCudaErrors(cudaEventSynchronize(stop_e));   
         checkCudaErrors(cudaEventElapsedTime(&OpTime, start_e, stop_e));  
         break;
      default:
         std::cout << "Error: unrecognized timed memory operation type" << std::endl; 
         break;
   }
   cudaEventDestroy(start_e);
   cudaEventDestroy(stop_e);

   return OpTime;
}

//TODO: add numa awareness and pin threads to different sockets for runs
void TestMemoryOverhead(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo) {
      char *deviceMem = NULL;
      char *hostMem = NULL;
      char *hostPinnedMem = NULL;
      int nDevices = params.nDevices;

      std::vector<long> blockSteps;
      std::vector<std::vector<float> > overheadData;

      // Only run overhead device cases on a single device
      // default to device 0
      if (!params.runAllDevices)
         nDevices = 1;
      
      // Memory overhead test will run for each device utilizing the cudaMalloc and cudaFree functions
      // on the first iteration of the look, assuming there is atleast one device, the host will run the 
      // pinned and un-pinned memory tests
      for (int numaIdx = 0; numaIdx < topo.NumNodes(); numaIdx++) { 
         topo.PinNumaNode(numaIdx);
         
         for (int socketIdx = 0; socketIdx < topo.NumSockets(); socketIdx++) {
            topo.PinSocket(socketIdx);
         
            for (int currDev = 0; currDev < nDevices; currDev++) {
               checkCudaErrors(cudaSetDevice(currDev));
        
               std::vector<float> chunkData;
               long stepNum = 0;
               long stepSize = params.rangeMemOverhead[0];// / params.rangeMemOverhead[2];
               //stepSize = (stepSize) ? stepSize : params.rangeMemOverhead[0]; 
               for ( long chunkSize = params.rangeMemOverhead[0]; 
                     chunkSize <= params.rangeMemOverhead[1]; 
                     chunkSize += stepSize) { 

                  if (currDev == 0) {
                     blockSteps.push_back(chunkSize); 
                     //CASE 1: Host Pinned Memory Overhead
                     chunkData.push_back(TimedMemOp((void **) &hostPinnedMem, chunkSize, HOST_PINNED_MALLOC));
                     chunkData.push_back(TimedMemOp((void **) &hostPinnedMem, 0, HOST_PINNED_FREE)); 
                     //CASE 2: Host UnPinned Memory Overhead
                     chunkData.push_back(TimedMemOp((void **) &hostMem, 0, HOST_FREE));
                     chunkData.push_back(TimedMemOp((void **) &hostMem, chunkSize, HOST_MALLOC));
                  }
                  // CASE 3: Allocation of device memory  
                  chunkData.push_back(TimedMemOp((void **) &deviceMem, chunkSize, DEVICE_MALLOC));
                  // CASE 4: DeAllocation of device memory 
                  chunkData.push_back(TimedMemOp((void **) &deviceMem, 0, DEVICE_FREE));
                 
                  //Add device/host run data to correct location of data vector
                  if (currDev == 0 && numaIdx == 0 && socketIdx == 0) {
                     overheadData.push_back(chunkData); 
                  } else {
                     for (int i = 0; i < chunkData.size(); i++) {
                     overheadData[stepNum].push_back(chunkData[i]);
                     //overheadData[stepNum].push_back(chunkData[1]);
                     }
                  }
                  chunkData.clear(); //clear chunkData for next mem step 

                  //Move to next stepSize after every numSteps as set by the param file
                  long stride = (params.rangeMemOverhead[2] - 1) ? (params.rangeMemOverhead[2] - 1) : 1;
                  if (stepNum && (stepNum % stride) == 0) {
                     stepSize *= 5;
                  }
                  stepNum++; 
               }
            }
         }
      }
      std::string dataFileName = "./results/" + params.resultsFile + "_overhead.csv";
      std::ofstream overheadResultsFile(dataFileName.c_str());
      PrintResults(overheadResultsFile, blockSteps, overheadData, params);
}

void TestHostDeviceBandwidth(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo) {
   std::cout << "Running host-device bandwidth test" << std::endl;
}

void TestP2PDeviceBandwidth(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo){
   std::cout << "Running P2P device bandwidth test" << std::endl;
}

void TestPCIeCongestion(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo) {
   std::cout << "Running PCIe congestion test" << std::endl;
}

void TestTaskScalability(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo) {
   std::cout << "Running task scalability test" << std::endl;
}

// Prints the device properties out to file based named depending on the 
void PrintDeviceProps(cudaDeviceProp *props, BenchParams &params) {
   std::cout << "\nSee " << params.devPropFile << " for information about your device's properties." << std::endl; 
   std::string devFileName = "./results/" + params.devPropFile;
   std::ofstream deviceProps(devFileName.c_str());

   deviceProps << "-------- Device Properties --------" << std::endl;

   for (int i = 0; i < params.nDevices; i++) {
      deviceProps << props[i].name << std::endl;
      deviceProps << "PCI Bus/Device/Domain ID: " <<   props[i].pciBusID << ":" <<  props[i].pciDeviceID << ":" <<  props[i].pciDomainID << std::endl; 
/*
      deviceProps << 
      deviceProps << 
      deviceProps << 
      deviceProps << 
      deviceProps << 
      deviceProps << 
      deviceProps << 
      deviceProps << 
      deviceProps << 
      deviceProps << 
      deviceProps << 
      deviceProps << 
      deviceProps << 
      deviceProps << 
      deviceProps << 
      deviceProps << 
      deviceProps << 
      deviceProps << 
      deviceProps << 
*/
   }
   deviceProps << "-----------------------------------" << std::endl;

   deviceProps.close();
}

void PrintResults(std::ofstream &outFile, std::vector<long> &steps, std::vector<std::vector<float> > &results, BenchParams &params) {
   
   if (!outFile.is_open()) {
      std::cout << "Failed to open file to print results" << std::endl;
      return;
   }
   std::vector<std::vector<float> >::iterator iter_o;
   std::vector<float>::iterator iter_i;
   std::vector<long>::iterator iter_l = steps.begin();
   std::cout << results[0].size() << std::endl;
   
   for (iter_o = results.begin(); iter_o != results.end(); ++iter_o) {
      outFile << std::fixed << *iter_l++ << ",";
      for (iter_i = (*iter_o).begin(); iter_i != (*iter_o).end(); ++iter_i) {
         outFile << std::fixed << *iter_i;
         if (iter_i + 1 != (*iter_o).end())
            outFile << ",";
      }
      outFile << std::endl;
   }
}

// Creates an array of cudaDeviceProp structs with populated data
// located in a pre-allocated section of memory
void GetAllDeviceProps(cudaDeviceProp *props, int dCount) {
   for (int i = 0; i < dCount; ++i) {
      cudaGetDeviceProperties(&props[i], i);
   }
}

// function for cleaning up device state including profile data
// to be used before and after any test in benchmark suite.
void ResetDevices(int numToReset) {
   for (int devNum = 0; devNum < numToReset; ++devNum) {
      cudaSetDevice(devNum);
      cudaDeviceReset();
   }
}

