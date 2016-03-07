
// Benchmark includes and defines
#ifndef BENCH_HEADER_INC
#define BENCH_HEADER_INC
#include "benchmark.h"
#endif

// BenchParams class definition
#ifndef PARAM_CLASS_INC
#include "parameters.h"
#define PARAM_CLASS_INC
#endif

// SystemTopo class definition
#ifndef TOPOLOGY_CLASS_INC
#include "topology.h"
#define TOPOLOGY_CLASS_INC
#endif

// Benchmark Tests
void RunTestSuite(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo);
void TestMemoryOverhead(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo);
void TestHostDeviceBandwidth(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo);
void TestP2PDeviceBandwidth(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo);
void TestPCIeCongestion(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo);
void TestTaskScalability(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo);

// Test Subfunctions
void MemCopyRun(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData, MEM_OP copyType, MEM_PATTERN patternType, int destIdx, int srcIdx); 
//void MemCopyRun(BenchParams &params, char *destPtr, char *srcPtr, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData, MEM_OP copyType, MEM_PATTERN patternType, int destIdx = 0, int srcIdx = 0);
float TimedMemOp(void **MemBlk, long long NumBytes, MEM_OP TimedOp); 
float TimedMemCopyStep(char * destPtr, char *srcPtr, long stepSize, long long blockSize, int numCopiesPerStep, MEM_OP copyType, MEM_PATTERN patternType, int destIdx = 0, int srcIdx = 0);
void MemCopyOp(char * destPtr, char *srcPtr, long stepSize, long long blockSize, int numCopiesPerStep, MEM_OP copyType, int destIdx = 0, int srcIdx = 0);

// Support functions
void AllocateMemBlock(SystemTopo &topo, void **destPtr, void **srcPtr, long long numBytes, MEM_OP copyType, int destIdx = 0, int srcIdx = 0);
void FreeMemBlock(SystemTopo &topo, void* destPtr, void *srcPtr, long long numBytes, MEM_OP copyType, int destIdx = 0, int srcIdx = 0);
int CalcRunSteps(std::vector<long long> &blockSteps, long long startStep, long long stopStep, long long numSteps); 

// Device Properties
void GetAllDeviceProps(cudaDeviceProp *props, int dCount);
void PrintDeviceProps(cudaDeviceProp *props, BenchParams &params);
void ResetDevices(int numToReset);

// Results output
void PrintResults(std::ofstream &outFile, std::vector<long long> &steps, std::vector<std::vector<float> > &results, BenchParams &params);

/* Benchmark main()
 * 
 * 
 */
int main (int argc, char **argv) {
   BenchParams benchParams;  
   SystemTopo sysTopo;
   
   std::cout << "\nStarting Multi-GPU Performance Test Suite...\n" << std::endl; 
   
   // Determine the number of recognized CUDA enabled devices
   checkCudaErrors(cudaGetDeviceCount(&(benchParams.nDevices)));

   // Exit if system contains no devices
   if (benchParams.nDevices <= 0) {
      std::cout << "No devices found...aborting benchmarks." << std::endl;
      exit(-1);
   }

   // Setup benchmark parameters
   if (argc == 1) { 
      // No input file, use default parameters
      benchParams.SetDefault();
   
   } else if (argc == 2) {       
      // Parse input file and set parameter class local variables
      benchParams.ParseParamFile(std::string(argv[1]));

   } else {
      // Unknown input parameter list, abort test
      std::cout << "Aborting test: Incorrect number of input parameters" << std::endl;
      exit(-1);
   }

   // Print HWLOC topology info
   // Class constructor parses system topology from device files (linux)
   std::string topoFileName ="./results/topology.out";
   std::ofstream topoFile(topoFileName.c_str());
   sysTopo.PrintTopology(topoFile);

   cudaDeviceProp *devProps = (cudaDeviceProp *) calloc (sizeof(cudaDeviceProp), benchParams.nDevices);

   // Aquire device properties for each CUDA enabled GPU
   GetAllDeviceProps(devProps, benchParams.nDevices);

   // Output device properties for each CUDA enabled GPU to file
   PrintDeviceProps(devProps, benchParams);

   // Print device parameters for user/script parsing
   benchParams.PrintParams();

   // Run the benchmark per parameters defines in params
   RunTestSuite(devProps, benchParams, sysTopo);

   free(devProps);
   
   return 0;
}

void RunTestSuite(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo) {

   if (params.runMemoryOverheadTest) {
      
      TestMemoryOverhead(props, params, topo);
   
   }

   if (params.runHDBandwidthTest) {

      TestHostDeviceBandwidth(props, params, topo);

   }

   if (params.runP2PBandwidthTest) {  
      
      TestP2PDeviceBandwidth(props, params, topo);
   
   }

   if (params.runPCIeCongestionTest) {

      TestPCIeCongestion(props, params, topo);

   }

   if (params.runTaskScalabilityTest) { 

      TestTaskScalability(props, params, topo);

   }

   std::cout << "\nBenchmarks complete!\n" << std::endl;

}

void TestMemoryOverhead(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo) {
   char *deviceMem = NULL;
   char *hostMem = NULL;
   char *hostPinnedMem = NULL;
   int nDevices = params.nDevices;
   long long chunkSize = 0;
   int testNum = 0;

   std::cout << "\nRunning Memory Overhead Test...\n" << std::endl;

   // Only run overhead device cases on a single device
   // default to device 0
   if (!params.runAllDevices)
      nDevices = 1;

   std::vector<long long> blockSteps;
   CalcRunSteps(blockSteps, params.rangeMemOverhead[0], params.rangeMemOverhead[1], params.rangeMemOverhead[2]);  
   std::vector<std::vector<float> > overheadData;
   overheadData.resize(blockSteps.size());
   
   // Memory overhead test will run for each device utilizing the cudaMalloc and cudaFree functions
   // on the first iteration of the look, assuming there is atleast one device, the host will run the 
   // pinned and un-pinned memory tests
   for (int numaIdx = 0; numaIdx < topo.NumNodes(); numaIdx++) { 
      topo.PinNumaNode(numaIdx);
      
      for (int socketIdx = 0; socketIdx < topo.NumSockets(); socketIdx++) {
         topo.PinSocket(socketIdx);
         
            std::cout << "Test " << testNum++ << " Host Alloc/Free, Pinned/Pageable\t" << "|numa node: " << numaIdx << " CPU " << socketIdx << "|" << std::endl;            
         // Host based management for CASE 1 & 2
         for (long stepIdx = 0; stepIdx < blockSteps.size(); stepIdx++) {
            chunkSize = blockSteps[stepIdx];
            float pinAllocTime = 0, pinFreeTime = 0, hostAllocTime = 0, hostFreeTime = 0;
            
            // repeat same block run and average times
            for (int reIdx = 0; reIdx < params.numStepRepeatsOH; reIdx++) {
               if (params.usePinnedMem) {
                  //CASE 1: Host Pinned Memory Overhead
                  pinAllocTime += TimedMemOp((void **) &hostPinnedMem, chunkSize, HOST_PINNED_MALLOC);
                  pinFreeTime += TimedMemOp((void **) &hostPinnedMem, 0, HOST_PINNED_FREE); 
               }
               //CASE 2: Host UnPinned Memory Overhead
               hostAllocTime += TimedMemOp((void **) &hostMem, 0, HOST_FREE);
               hostFreeTime += TimedMemOp((void **) &hostMem, chunkSize, HOST_MALLOC);
            }

            overheadData[stepIdx].push_back(pinAllocTime / (float) params.numStepRepeatsOH);
            overheadData[stepIdx].push_back(pinFreeTime / (float) params.numStepRepeatsOH);
            overheadData[stepIdx].push_back(hostAllocTime / (float) params.numStepRepeatsOH);
            overheadData[stepIdx].push_back(hostFreeTime / (float) params.numStepRepeatsOH);
         }
         
         // Device based memory management for CASE 3 & 4
         for (int currDev = 0; currDev < nDevices; currDev++) {
            checkCudaErrors(cudaSetDevice(currDev)); 
            std::cout << "Test " << testNum++ << " Device Alloc/Free \t\t" << "|numa node: " << numaIdx << " CPU " << socketIdx << " device:" << currDev << "|" << std::endl;            
            
            for (long stepIdx = 0; stepIdx < blockSteps.size(); stepIdx++) {
               chunkSize = blockSteps[stepIdx];
               float devAllocTime = 0, devFreeTime = 0;

               // repeat same block run and average times
               for (int reIdx = 0; reIdx < params.numStepRepeatsOH; reIdx++) {
                  // CASE 3: Allocation of device memory  
                  devAllocTime += TimedMemOp((void **) &deviceMem, chunkSize, DEVICE_MALLOC);
                  // CASE 4: DeAllocation of device memory 
                  devFreeTime += TimedMemOp((void **) &deviceMem, 0, DEVICE_FREE);
               }

               overheadData[stepIdx].push_back(devAllocTime / (float) params.numStepRepeatsOH);
               overheadData[stepIdx].push_back(devFreeTime / (float) params.numStepRepeatsOH);
            }
         }
      }
   }
   std::string dataFileName = "./results/" + params.resultsFile + "_overhead.csv";
   std::ofstream overheadResultsFile(dataFileName.c_str());
   PrintResults(overheadResultsFile, blockSteps, overheadData, params);

   std::cout << "\nMemory Overhead Test Complete!" << std::endl;
   
}

void TestHostDeviceBandwidth(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo) {
   std::cout << "\nRunning Host-Device and Device-Host Bandwidth Test...\n" << std::endl;

   params.numCopiesPerStepHD = 20;
   
   if (params.runSustainedHD == false) {
      params.numCopiesPerStepHD = 1;
   }

   std::vector<std::vector<float> > bandwidthData;
   std::vector<long long> blockSteps;
   CalcRunSteps(blockSteps, params.rangeHostDeviceBW[0], params.rangeHostDeviceBW[1], params.rangeHostDeviceBW[2]); 
   bandwidthData.resize(blockSteps.size());
   int testNum = 0;
   for (int socketIdx = 0; socketIdx < 1/*topo.NumSockets()*/; socketIdx++) {
      //topo.PinSocket(socketIdx);
 
      for (int numaSrc = 0; numaSrc < topo.NumNodes(); numaSrc++) { 
         //topo.PinNumaNode(numaSrc);

         //Host To Host Memory Transfers
         for (int numaDest = 0; numaDest < topo.NumNodes(); numaDest++) { 
            // HtoH Ranged Transfer - Pageable Memory
            std::cout << "Test " << testNum++ << " HtoH, Pageable Memory, Repeated Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Numa:" << numaDest << "|" << std::endl;
            MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_HOST_COPY, REPEATED, numaDest, numaSrc); 
            /*std::cout << "Test " << testNum++ << " HtoH, Pageable Memory, Random\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Numa:" << numaDest << "|" << std::endl;
            MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_HOST_COPY, RANDOM, numaDest, numaSrc); 
            std::cout << "Test " << testNum++ << " HtoH, Pageable Memory, Linear Inc Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Numa:" << numaDest << "|" << std::endl;
            MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_HOST_COPY, LINEAR_INC, numaDest, numaSrc); 
            std::cout << "Test " << testNum++ << " HtoH, Pageable Memory, Linear Dec Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Numa:" << numaDest << "|" << std::endl;
            MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_HOST_COPY, LINEAR_DEC, numaDest, numaSrc); 
*/
            //HtoH Ranged Transfer - Pinned Memory
            std::cout << "Test " << testNum++ << " HtoH, Pinned Memory, Repeated Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Numa:" << numaDest << "|" << std::endl;
            MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_HOST_COPY_PINNED, REPEATED, numaDest, numaSrc); 
  /*          std::cout << "Test " << testNum++ << " HtoH, Pinned Memory, Random Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Numa:" << numaDest << "|" << std::endl;
            //MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_HOST_COPY_PINNED, RANDOM, numaDest, numaSrc); 
            std::cout << "Test " << testNum++ << " HtoH, Pinned Memory, Linear Inc Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Numa:" << numaDest << "|" << std::endl;
            //MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_HOST_COPY_PINNED, LINEAR_INC, numaDest, numaSrc); 
            std::cout << "Test " << testNum++ << " HtoH, Pinned Memory, Linear Dec Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Numa:" << numaDest << "|" << std::endl;
            //MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_HOST_COPY_PINNED, LINEAR_DEC, numaDest, numaSrc); 
*/
         }

         //Host-Device PCIe Memory Transfers
         for (int currDev = 0; currDev < params.nDevices; currDev++) {
            //checkCudaErrors(cudaSetDevice(currDev));

            // HtoD Ranged Transfer - Pageable Memory
            std::cout << "Test " << testNum++ << " HtoD, Pageable Memory, Repeated Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Dev:" << currDev << "|" << std::endl;
            MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_DEVICE_COPY, REPEATED, currDev, numaSrc); 
          /*  std::cout << "Test " << testNum++ << " HtoD, Pageable Memory, Random Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Dev:" << currDev << "|" << std::endl;
            //MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_DEVICE_COPY, RANDOM, currDev, numaSrc); 
            std::cout << "Test " << testNum++ << " HtoD, Pageable Memory, Linear Inc Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Dev:" << currDev << "|" << std::endl;
            //MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_DEVICE_COPY, LINEAR_INC, currDev, numaSrc); 
            std::cout << "Test " << testNum++ << " HtoD, Pageable Memory, Linear Dec Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Dev:" << currDev << "|" << std::endl;
            //MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_DEVICE_COPY, LINEAR_DEC, currDev, numaSrc); 
*/
            // DtoH Ranged Transfer - Pageable Memory
            std::cout << "Test " << testNum++ << " DtoH, Pageable Memory, Repeated Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Dev:" << currDev << "|" << std::endl;
            MemCopyRun(params, topo, blockSteps, bandwidthData, DEVICE_HOST_COPY, REPEATED, numaSrc, currDev); 
  /*          std::cout << "Test " << testNum++ << " DtoH, Pageable Memory, Random Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Dev:" << currDev << "|" << std::endl;
            //MemCopyRun(params, topo, blockSteps, bandwidthData, DEVICE_HOST_COPY, RANDOM, numaSrc, currDev); 
            std::cout << "Test " << testNum++ << " DtoH, Pageable Memory, Linear Inc Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Dev:" << currDev << "|" << std::endl;
            //MemCopyRun(params, topo, blockSteps, bandwidthData, DEVICE_HOST_COPY, LINEAR_INC, numaSrc, currDev); 
            std::cout << "Test " << testNum++ << " DtoH, Pageable Memory, Linear Dec Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Dev:" << currDev << "|" << std::endl;
            //MemCopyRun(params, topo, blockSteps, bandwidthData, DEVICE_HOST_COPY, LINEAR_DEC, numaSrc, currDev); 
*/
            // HtoD Ranged Transfer - Pinned Memory
            std::cout << "Test " << testNum++ << " HtoD, Pinned Memory, Repeated Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Dev:" << currDev << "|" << std::endl;
            MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_DEVICE_COPY_PINNED, REPEATED, currDev, numaSrc); 
  /*          std::cout << "Test " << testNum++ << " HtoD, Pinned Memory, Random Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Dev:" << currDev << "|" << std::endl;
            //MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_DEVICE_COPY_PINNED, RANDOM, currDev, numaSrc); 
            std::cout << "Test " << testNum++ << " HtoD, Pinned Memory, Linear Inc Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Dev:" << currDev << "|" << std::endl;
            //MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_DEVICE_COPY_PINNED, LINEAR_INC, currDev, numaSrc); 
            std::cout << "Test " << testNum++ << " HtoD, Pinned Memory, Linear Dec Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Dev:" << currDev << "|" << std::endl;
            //MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_DEVICE_COPY_PINNED, LINEAR_DEC, currDev, numaSrc); 
*/
            // DtoH Ranged Transfer - Pinned Memory
            std::cout << "Test " << testNum++ << " DtoH, Pinned Memory, Repeated Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Dev:" << currDev << "|" << std::endl;
            MemCopyRun(params, topo, blockSteps, bandwidthData, DEVICE_HOST_COPY_PINNED, REPEATED, numaSrc, currDev); 
  /*          std::cout << "Test " << testNum++ << " DtoH, Pinned Memory, Random Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Dev:" << currDev << "|" << std::endl;
            //MemCopyRun(params, topo, blockSteps, bandwidthData, DEVICE_HOST_COPY_PINNED, RANDOM, numaSrc, currDev); 
            std::cout << "Test " << testNum++ << " DtoH, Pinned Memory, Linear Inc\t\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Dev:" << currDev << "|" << std::endl;
            //MemCopyRun(params, topo, blockSteps, bandwidthData, DEVICE_HOST_COPY_PINNED, LINEAR_INC, numaSrc, currDev); 
            std::cout << "Test " << testNum++ << " DtoH, Pinned Memory, Linear Dec Addr\t|CPU:" << socketIdx << " Numa Src:" << numaSrc << " Dest Dev:" << currDev << "|" << std::endl;
            //MemCopyRun(params, topo, blockSteps, bandwidthData, DEVICE_HOST_COPY_PINNED, LINEAR_DEC, numaSrc, currDev); 
    */     }
      }
   }

   std::string dataFileName = "./results/" + params.resultsFile + "_bandwidth.csv";
   std::ofstream bandwidthResultsFile(dataFileName.c_str());
   PrintResults(bandwidthResultsFile, blockSteps, bandwidthData, params);

   std::cout << "\nHost-Device and Host-Host Bandwidth Test complete!" << std::endl;

}

void TestP2PDeviceBandwidth(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo){
   std::cout << "Running P2P Device Bandwidth Test..." << std::endl;

   //Device to Device transfers
   /*for (int srcDev = 0; currDev < params.nDevices; currDev++) {
      checkCudaErrors(cudaSetDevice(currDev));
      for (int destDev = 0; currDev < nDevices; currDev++) {
         checkCudaErrors(cudaSetDevice(currDev));  
      
         //must support p2p to allow direct transfer
         if (srcDev != destDev) {
            
         }
      }
   } */     

   std::cout << " P2P Device Bandwidth Test Complete!" << std::endl;
}

void TestPCIeCongestion(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo) {
   std::cout << "Running PCIe congestion test" << std::endl;
   return;
}

void TestTaskScalability(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo) {
   std::cout << "Running task scalability test" << std::endl;
   return;
}

void MemCopyRun(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData, MEM_OP copyType, MEM_PATTERN patternType, int destIdx, int srcIdx) {
   char *destPtr, *srcPtr; 
   long totalSteps = blockSteps.size();
   
   std::vector<float> timedRun(totalSteps, 0.0);
   long long blockSize = blockSteps[totalSteps - 1 ];

   AllocateMemBlock(topo, (void **) &destPtr, (void **) &srcPtr, blockSize, copyType, destIdx, srcIdx);

   for (long stepNum = 0; stepNum < totalSteps; ++stepNum) { 

      bandwidthData[stepNum].push_back(TimedMemCopyStep((char *) destPtr, (char *) srcPtr, blockSteps[stepNum], blockSize, params.numCopiesPerStepHD, copyType, patternType, destIdx, srcIdx));

   }
   
   FreeMemBlock(topo, (void *) destPtr, (void *) srcPtr, blockSize, copyType, destIdx, srcIdx);
}

float TimedMemCopyStep(char * destPtr, char *srcPtr, long stepSize, long long blockSize, int numCopiesPerStep, MEM_OP copyType, MEM_PATTERN patternType, int destIdx, int srcIdx) {
   long long offset = 0;
   float totalTime = 0; 

   #ifdef USING_CPP
   std::chrono::high_resolution_clock::time_point start_c, stop_c;
   auto total_c = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_c - start_c);
   #else
   struct timeval stop_t, start_t, total_t;
   #endif
   
   cudaEvent_t start_e, stop_e; 
   checkCudaErrors(cudaEventCreate(&start_e));
   checkCudaErrors(cudaEventCreate(&stop_e)); 

   if (HOST_HOST_COPY) {
      #ifdef USING_CPP
      start_c = std::chrono::high_resolution_clock::now();
      #else
      gettimeofday(&start_t, NULL);
      #endif
   } else{
      checkCudaErrors(cudaEventRecord(start_e, 0));
   }

   for (int copyIdx = 0; copyIdx < numCopiesPerStep; copyIdx++) {

      MemCopyOp(destPtr + offset, srcPtr + offset, stepSize, blockSize, numCopiesPerStep, copyType, destIdx, srcIdx); 

      //TODO: add options to change 
      if (numCopiesPerStep > 1 && MAX_PATTERN_SIZE) {
         switch (patternType) {
       
           case REPEATED:
               offset = 0;
               break;
            case RANDOM:
               break;
            case PERIODIC:
               break;
            case LINEAR_INC:
               break;
            case LINEAR_DEC:
               break;
            default: //BURST
               break;
         }
      }
   }

   if (HOST_HOST_COPY) {
      #ifdef USING_CPP
      stop_c = std::chrono::high_resolution_clock::now(); 
      total_c = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_c - start_c);
      totalTime = (float) total_c.count() * NANO_TO_MILLI; 
      #else
      gettimeofday(&stop_t, NULL); 
      timersub(&stop_t, &start_t, &total_t); 
      totalTime = (float) total_t.tv_usec * MICRO_TO_MILLI;
      #endif
   } else{
      checkCudaErrors(cudaEventRecord(stop_e, 0));
      checkCudaErrors(cudaEventSynchronize(stop_e));   
      checkCudaErrors(cudaEventElapsedTime(&totalTime, start_e, stop_e));  
   }

   return totalTime;
}

void MemCopyOp(char * destPtr, char *srcPtr, long stepSize, long long blockSize, int numCopiesPerStep, MEM_OP copyType, int destIdx, int srcIdx) {
   switch (copyType) {
      case HOST_HOST_COPY:  
         memcpy((void *) (destPtr), (void *) (srcPtr), stepSize);
         break;
      case HOST_HOST_COPY_PINNED:  
         checkCudaErrors(cudaMemcpyAsync((void *)(destPtr), (void *) (srcPtr), stepSize, cudaMemcpyHostToHost, 0));
         break;
      case DEVICE_HOST_COPY:
         checkCudaErrors(cudaMemcpy((void *) (destPtr), (void *) (srcPtr), stepSize, cudaMemcpyDeviceToHost));
         break;
      case DEVICE_HOST_COPY_PINNED:
         checkCudaErrors(cudaMemcpyAsync((void *) (destPtr), (void *) (srcPtr), stepSize, cudaMemcpyDeviceToHost, 0));
         break;
      case HOST_DEVICE_COPY:
         checkCudaErrors(cudaMemcpy((void *) (destPtr), (void *) (srcPtr), stepSize, cudaMemcpyHostToDevice));
         break;
      case HOST_DEVICE_COPY_PINNED:
         checkCudaErrors(cudaMemcpyAsync((void *) (destPtr), (void *) (srcPtr), stepSize, cudaMemcpyHostToDevice, 0));
         break;
      case PEER_COPY_NO_UVA:
         checkCudaErrors(cudaMemcpyPeerAsync((void *) (destPtr), destIdx, (void *) (srcPtr), srcIdx, 0));
         break;
      case DEVICE_DEVICE_COPY:
         checkCudaErrors(cudaMemcpyAsync((void *) (destPtr), (void *) (srcPtr), stepSize, cudaMemcpyDeviceToDevice));
         break;
      case COPY_UVA:
         checkCudaErrors(cudaMemcpyAsync((void *) (destPtr), (void *) (srcPtr), stepSize, cudaMemcpyDefault, 0));
         break;
      default:
         std::cout << "Error: unrecognized timed memory copy operation type" << std::endl; 
         break;
   }
}

void FreeMemBlock(SystemTopo &topo, void* destPtr, void *srcPtr, long long numBytes, MEM_OP copyType, int destIdx, int srcIdx) {
   switch (copyType) {
      case HOST_HOST_COPY: 
         topo.FreeMem((void *) destPtr, numBytes);
         topo.FreeMem((void *) srcPtr, numBytes);
         break;
      case HOST_HOST_COPY_PINNED:  
         checkCudaErrors(cudaHostUnregister((void*) srcPtr));
         topo.FreeMem((void *) srcPtr, numBytes);
         checkCudaErrors(cudaHostUnregister((void*) destPtr));
         topo.FreeMem((void *) destPtr, numBytes);
         break;
      case DEVICE_HOST_COPY:
         checkCudaErrors(cudaSetDevice(srcIdx));
         checkCudaErrors(cudaFree((void *) srcPtr));
         topo.FreeMem((void *) destPtr, numBytes);
         break;
      case DEVICE_HOST_COPY_PINNED:
         checkCudaErrors(cudaSetDevice(srcIdx));
         checkCudaErrors(cudaFree((void *) srcPtr));
         checkCudaErrors(cudaHostUnregister((void*) destPtr));
         topo.FreeMem((void *) destPtr, numBytes);
         break;
      case HOST_DEVICE_COPY:
         topo.FreeMem((void *) srcPtr, numBytes);
         checkCudaErrors(cudaSetDevice(destIdx));
         checkCudaErrors(cudaFree((void *) destPtr));
         break;
      case HOST_DEVICE_COPY_PINNED:
         checkCudaErrors(cudaHostUnregister((void *) srcPtr));
         topo.FreeMem((void *) srcPtr, numBytes);
         checkCudaErrors(cudaSetDevice(destIdx));
         checkCudaErrors(cudaFree((void *) destPtr));
         break;
      case PEER_COPY_NO_UVA: 
         checkCudaErrors(cudaSetDevice(srcIdx));
         checkCudaErrors(cudaFree((void *) srcPtr));
         checkCudaErrors(cudaSetDevice(destIdx));
         checkCudaErrors(cudaFree((void *) destPtr));
         break;
      case DEVICE_DEVICE_COPY:
         checkCudaErrors(cudaSetDevice(srcIdx));
         checkCudaErrors(cudaFree((void *) srcPtr));
         checkCudaErrors(cudaSetDevice(destIdx));
         checkCudaErrors(cudaFree((void *) destPtr));
         break;
      case COPY_UVA:
         checkCudaErrors(cudaSetDevice(srcIdx));
         checkCudaErrors(cudaFree((void *) srcPtr));
         checkCudaErrors(cudaSetDevice(destIdx));
         checkCudaErrors(cudaFree((void *) destPtr));
         break;
      default:
         std::cout << "Error: unrecognized memory copy operation type for deallocation" << std::endl; 
         break;
   }
}

void AllocateMemBlock(SystemTopo &topo, void **destPtr, void **srcPtr,long  long numBytes, MEM_OP copyType, int destIdx, int srcIdx) {
   switch (copyType) {

      case HOST_HOST_COPY: 
         *destPtr = topo.AllocMemByNode(destIdx, numBytes);
         *srcPtr =topo.AllocMemByNode(srcIdx, numBytes);
         break;
      case HOST_HOST_COPY_PINNED:  
         *srcPtr =topo.AllocMemByNode(srcIdx, numBytes);
         checkCudaErrors(cudaHostRegister(*srcPtr, numBytes, cudaHostRegisterPortable));
         *destPtr = topo.AllocMemByNode(destIdx, numBytes);
         checkCudaErrors(cudaHostRegister(*destPtr, numBytes, cudaHostRegisterPortable));
         break;
      case DEVICE_HOST_COPY:
         checkCudaErrors(cudaSetDevice(srcIdx));
         checkCudaErrors(cudaMalloc(srcPtr, numBytes));
         *destPtr = topo.AllocMemByNode(destIdx, numBytes);
         break;
      case DEVICE_HOST_COPY_PINNED:
         checkCudaErrors(cudaSetDevice(srcIdx));
         checkCudaErrors(cudaMalloc(srcPtr, numBytes));
         *destPtr = topo.AllocMemByNode(destIdx, numBytes);
         checkCudaErrors(cudaHostRegister(*destPtr, numBytes, cudaHostRegisterPortable));
         break;
      case HOST_DEVICE_COPY:
         *srcPtr = topo.AllocMemByNode(srcIdx, numBytes);
         checkCudaErrors(cudaSetDevice(destIdx));
         checkCudaErrors(cudaMalloc(destPtr, numBytes));
         break;
      case HOST_DEVICE_COPY_PINNED:
         *srcPtr = topo.AllocMemByNode(srcIdx, numBytes);
         checkCudaErrors(cudaHostRegister(*srcPtr, numBytes, cudaHostRegisterPortable));
         checkCudaErrors(cudaSetDevice(destIdx));
         checkCudaErrors(cudaMalloc(destPtr, numBytes));
         break;
      case PEER_COPY_NO_UVA: 
         checkCudaErrors(cudaSetDevice(srcIdx));
         checkCudaErrors(cudaMalloc(srcPtr, numBytes));
         checkCudaErrors(cudaSetDevice(destIdx));
         checkCudaErrors(cudaMalloc(destPtr, numBytes));
         break;
      case DEVICE_DEVICE_COPY:
         checkCudaErrors(cudaSetDevice(srcIdx));
         checkCudaErrors(cudaMalloc(srcPtr, numBytes));
         checkCudaErrors(cudaSetDevice(destIdx));
         checkCudaErrors(cudaMalloc(destPtr, numBytes));
         break;
      case COPY_UVA:
         checkCudaErrors(cudaSetDevice(srcIdx));
         checkCudaErrors(cudaMalloc(srcPtr, numBytes));
         checkCudaErrors(cudaSetDevice(destIdx));
         checkCudaErrors(cudaMalloc(destPtr, numBytes));
         break;
      default:
         std::cout << "Error: unrecognized memory copy operation type for allocation" << std::endl; 
         break;
   }
}

int CalcRunSteps(std::vector<long long> &blockSteps, long long startStep, long long stopStep, long long numSteps) {
   int magStart = max((int)log10(startStep), 1);
   int magStop = log10(stopStep);

   long long start = pow(10, magStart);
   double stepSize = 10 * start / numSteps;
   int extra = (stopStep - pow(10, magStop)) / pow(10, magStop) * numSteps;
   long long stop = pow(10, magStop - 1) * (10 + extra); 
   int rangeSkip = numSteps / start;
   int totalSteps = (magStop - magStart) * (numSteps - rangeSkip) + extra + 1;  
   double step = start;

   for (long stepNum = 0; stepNum < totalSteps; ++stepNum) { 
      blockSteps.push_back(step);
      
      if ((stepNum) && (stepNum) % (numSteps - rangeSkip) == 0 && (stepSize * numSteps * 10) <= stop) {
         stepSize *= 10.0;
      } 
      
      step += stepSize; 
   }

   return totalSteps;
}

float TimedMemOp(void **MemBlk, long long NumBytes, MEM_OP TimedOp) {
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
         checkCudaErrors(cudaEventRecord(start_e, 0));      
         checkCudaErrors(cudaMallocHost(MemBlk, NumBytes));
         checkCudaErrors(cudaEventRecord(stop_e, 0));
         checkCudaErrors(cudaEventSynchronize(stop_e));
         checkCudaErrors(cudaEventElapsedTime(&OpTime, start_e, stop_e));
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
         checkCudaErrors(cudaEventRecord(start_e, 0));
         checkCudaErrors(cudaFreeHost(*MemBlk));
         checkCudaErrors(cudaEventRecord(stop_e, 0));
         checkCudaErrors(cudaEventSynchronize(stop_e));
         checkCudaErrors(cudaEventElapsedTime(&OpTime, start_e, stop_e));
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
   checkCudaErrors(cudaEventDestroy(start_e));
   checkCudaErrors(cudaEventDestroy(stop_e));

   return OpTime;
}

// Prints the device properties out to file based named depending on the 
void PrintDeviceProps(cudaDeviceProp *props, BenchParams &params) {
   std::string devFileName = "./results/" + params.devPropFile;
   std::ofstream devicePropsFile(devFileName.c_str());
   std::stringstream devicePropsSS;

   devicePropsSS << "\n-----------------------------------------------------------------" << std::endl;
   devicePropsSS << "------------------------ Device Properties ----------------------" << std::endl;
   devicePropsSS << "-----------------------------------------------------------------" << std::endl;

   int driverVersion = 0, runtimeVersion;
   for (int i = 0; i < params.nDevices; i++) {
      cudaSetDevice(i);
      cudaDriverGetVersion(&driverVersion);
      cudaDriverGetVersion(&runtimeVersion);

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

   std::cout << nvmlInit() << std::endl;

   unsigned int count = 0;
   nvmlDevice_t deviceArray;// = (nvmlDevice_t *) malloc(sizeof(nvmlDevice_t) * 4);
   //nvmlSystemGetTopologyGpuSet(1, &count, deviceArray);
   std::cout << "cpuset dev count " << count << std::endl;
   nvmlUnitGetCount(&count);
   std::cout << "unit count " << count << std::endl;
   nvmlDeviceGetCount(&count);
   std::cout << "device count " << count << std::endl;
   std::cout << nvmlShutdown() << std::endl;
   
   nvmlDeviceGetHandleByIndex(1, &deviceArray);
   unsigned long cpuSet[2];
   cpuSet[0] = 0;
   cpuSet[1] = 0;


   nvmlDeviceGetCpuAffinity(deviceArray, 2, cpuSet);
   std::cout << "CPUSET: " << cpuSet[0] << " " << cpuSet[0] << std::endl;
   if (params.printDevProps) 
      std::cout << devicePropsSS.str(); 
   else
      std::cout << "\nSee " << params.devPropFile << " for information about your device's properties." << std::endl; 
   devicePropsFile << devicePropsSS.str();
   
   devicePropsFile.close();
}

void PrintResults(std::ofstream &outFile, std::vector<long long> &steps, std::vector<std::vector<float> > &results, BenchParams &params) {
   
   if (!outFile.is_open()) {
      std::cout << "Failed to open file to print results" << std::endl;
      return;
   }
   std::vector<std::vector<float> >::iterator iter_o;
   std::vector<float>::iterator iter_i;
   std::vector<long long>::iterator iter_l = steps.begin();
   //std::cout << results[0].size() << std::endl;
   
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
      checkCudaErrors(cudaGetDeviceProperties(&props[i], i));
   }
}

// function for cleaning up device state including profile data
// to be used before and after any test in benchmark suite.
void ResetDevices(int numToReset) {
   for (int devNum = 0; devNum < numToReset; ++devNum) {
      checkCudaErrors(cudaSetDevice(devNum));
      checkCudaErrors(cudaDeviceReset());
   }
}

