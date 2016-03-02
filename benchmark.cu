
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
void RunTestSuite(BenchParams &params, SystemTopo &topo);
void TestMemoryOverhead(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo);
void TestHostDeviceBandwidth(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo);
void TestP2PDeviceBandwidth(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo);
void TestPCIeCongestion(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo);
void TestTaskScalability(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo);

// Test Subfunctions

void MemCopyRun(BenchParams &params, char *destPtr, char *srcPtr, std::vector<long> &blockSteps, std::vector<std::vector<float> > &bandwidthData, MEM_OP copyType, MEM_PATTERN patternType, int destIdx = 0, int srcIdx = 0);
float TimedMemOp(void **MemBlk, long NumBytes, MEM_OP TimedOp); 
float TimedMemCopyStep(char * destPtr, char *srcPtr, long stepSize, long blockSize, int numCopiesPerStep, MEM_OP copyType, MEM_PATTERN patternType, int destIdx = 0, int srcIdx = 0);
void MemCopyOp(char * destPtr, char *srcPtr, long stepSize, long blockSize, int numCopiesPerStep, MEM_OP copyType, int destIdx = 0, int srcIdx = 0);

// Support functions
void AllocateMemBlock(SystemTopo &topo, void **destPtr, void **srcPtr, long numBytes, MEM_OP copyType, int destIdx = 0, int srcIdx = 0);
void FreeMemBlock(SystemTopo &topo, void* destPtr, void *srcPtr, long numBytes, MEM_OP copyType, int destIdx = 0, int srcIdx = 0);
int CalcRunSteps(std::vector<long> &blockSteps, long startStep, long stopStep, long numSteps); 

// Device Properties
void GetAllDeviceProps(cudaDeviceProp *props, int dCount);
void PrintDeviceProps(cudaDeviceProp *props, BenchParams &params);
void ResetDevices(int numToReset);

// Results output
void PrintResults(std::ofstream &outFile, std::vector<long> &steps, std::vector<std::vector<float> > &results, BenchParams &params);

/* Benchmark main()
 * 
 * 
 */

int main (int argc, char **argv) {
   BenchParams params;  
   SystemTopo topo;
   
   std::cout << "Starting Multi-GPU Performance Test Suite...\n" << std::endl; 
   
   // Determine the number of recognized CUDA enabled devices
   checkCudaErrors(cudaGetDeviceCount(&(params.nDevices)));

   // Exit if system contains no devices
   if (params.nDevices <= 0) {
      std::cout << "No devices found...aborting benchmarks." << std::endl;
      exit(-1);
   }

   // Setup benchmark parameters
   if (argc == 1) { 
      // No input file, use default parameters
      params.SetDefault();
   
   } else if (argc == 2) {       
      // Parse input file and set parameter class local variables
      params.ParseParamFile(std::string(argv[1]));

   } else {
      // Unknown input parameter list, abort test
      std::cout << "Aborting test: Incorrect number of input parameters" << std::endl;
      exit(-1);
   }

   // If running topology aware print HWLOC topology info 
   if (params.runTopoAware) {
      std::string topoFileName ="./results/topology.out";
      std::ofstream topoFile(topoFileName.c_str());
      topo.PrintTopology(topoFile);
   }

   // Print device parameters for user/script parsing
   params.PrintParams();

   // Run the benchmark per parameters defines in params
   RunTestSuite(params, topo);

   return 0;
}

void RunTestSuite(BenchParams &params, SystemTopo &topo) {
   cudaDeviceProp *props = (cudaDeviceProp *) calloc (sizeof(cudaDeviceProp), params.nDevices);

   // Aquire device properties for each CUDA enabled GPU
   GetAllDeviceProps(props, params.nDevices);

   if (params.runMemoryOverheadTest) {
      
      TestMemoryOverhead(props, params, topo);
   
   }

   if (params.runHostDeviceBandwidthTest) {

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

   // Output device properties for each CUDA enabled GPU
   if (params.printDevProps) {
      PrintDeviceProps(props, params);
   }

   std::cout << "\nBenchmarks complete!\n" << std::endl;

   free(props);

   return;
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

   return;
}

void TestHostDeviceBandwidth(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo) {
   std::cout << "Running host-device bandwidth test" << std::endl;

   params.numCopiesPerStepHD = 1;
   
   if (params.runSustainedHD == false) {
      params.numCopiesPerStepHD = 1;
   }

   void *destPtr, *srcPtr;
   std::vector<std::vector<float> > bandwidthData;
   std::vector<long> blockSteps;
   CalcRunSteps(blockSteps, params.rangeHostDeviceBW[0], params.rangeHostDeviceBW[1], params.rangeHostDeviceBW[2]); 
   bandwidthData.resize(blockSteps.size());

   AllocateMemBlock(topo, &destPtr, &srcPtr, blockSteps[blockSteps.size()-1], HOST_HOST_COPY);
   MemCopyRun(params, (char *) destPtr, (char *) srcPtr, blockSteps, bandwidthData, HOST_HOST_COPY, REPEATED); 
   FreeMemBlock(topo, destPtr, srcPtr, blockSteps[blockSteps.size()-1], HOST_HOST_COPY);

   for (int socketIdx = 0; socketIdx < topo.NumSockets(); socketIdx++) {
      topo.PinSocket(socketIdx);
 
      for (int numaSrc = 0; numaSrc < topo.NumNodes(); numaSrc++) { 
         topo.PinNumaNode(numaSrc);

         //Host to host memory transfers
         for (int numaDest = 0; numaDest < topo.NumNodes(); numaDest++) { 


         }

         //Host/Device bandwidth PCIetransfers
         for (int currDev = 0; currDev < params.nDevices; currDev++) {
            checkCudaErrors(cudaSetDevice(currDev));


         }
      }
   }

   std::string dataFileName = "./results/" + params.resultsFile + "_bandwidth.csv";
   std::ofstream bandwidthResultsFile(dataFileName.c_str());
   PrintResults(bandwidthResultsFile, blockSteps, bandwidthData, params);

   return;
}

void TestP2PDeviceBandwidth(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo){
   std::cout << "Running P2P device bandwidth test" << std::endl;

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
   return;
}

void TestPCIeCongestion(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo) {
   std::cout << "Running PCIe congestion test" << std::endl;
   return;
}

void TestTaskScalability(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo) {
   std::cout << "Running task scalability test" << std::endl;
   return;
}

void MemCopyRun(BenchParams &params, char *destPtr, char *srcPtr, std::vector<long> &blockSteps, std::vector<std::vector<float> > &bandwidthData, MEM_OP copyType, MEM_PATTERN patternType, int destIdx, int srcIdx) {
   long totalSteps = blockSteps.size();
   
   std::vector<float> timedRun(totalSteps, 0.0);
   long blockSize = blockSteps[totalSteps - 1 ];
   
   for (long stepNum = 0; stepNum < totalSteps; ++stepNum) { 

      bandwidthData[stepNum].push_back(TimedMemCopyStep((char *) destPtr, (char *) srcPtr, blockSteps[stepNum], blockSize, params.numCopiesPerStepHD, copyType, patternType, destIdx, srcIdx));

   }
}

float TimedMemCopyStep(char * destPtr, char *srcPtr, long stepSize, long blockSize, int numCopiesPerStep, MEM_OP copyType, MEM_PATTERN patternType, int destIdx, int srcIdx) {
   long offset = 0;
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

void MemCopyOp(char * destPtr, char *srcPtr, long stepSize, long blockSize, int numCopiesPerStep, MEM_OP copyType, int destIdx, int srcIdx) {
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

void FreeMemBlock(SystemTopo &topo, void* destPtr, void *srcPtr, long numBytes, MEM_OP copyType, int destIdx, int srcIdx) {
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

void AllocateMemBlock(SystemTopo &topo, void **destPtr, void **srcPtr, long numBytes, MEM_OP copyType, int destIdx, int srcIdx) {
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

int CalcRunSteps(std::vector<long> &blockSteps, long startStep, long stopStep, long numSteps) {
   int magStart = max((int)log10(startStep), 1);
   int magStop = log10(stopStep);

   long start = pow(10, magStart);
   double stepSize = 10 * start / numSteps;
   int extra = (stopStep - pow(10, magStop)) / pow(10, magStop) * numSteps;
   long stop = pow(10, magStop - 1) * (10 + extra); 
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
   std::cout << "\nSee " << params.devPropFile << " for information about your device's properties." << std::endl; 
   std::string devFileName = "./results/" + params.devPropFile;
   std::ofstream deviceProps(devFileName.c_str());

   deviceProps << "-------- Device Properties --------" << std::endl;

   for (int i = 0; i < params.nDevices; i++) {
      deviceProps << props[i].name << std::endl;
      deviceProps << "CUDA Capability: " << props[i].major << "." << props[i].minor << std::endl;
      deviceProps << "PCI Bus/Device/Domain ID: " <<   props[i].pciBusID << ":" <<  props[i].pciDeviceID << ":" <<  props[i].pciDomainID << std::endl; 
      deviceProps << "Clock: " << props[i].clockRate << std::endl; 
      deviceProps << "Memory Clock: " << props[i].memoryClockRate << std::endl; 
      deviceProps << "Memory Bus Width: " << props[i].memoryBusWidth << std::endl; 
      deviceProps << "Theoretical BW: " << props[i].clockRate << std::endl;
      deviceProps << "Global Mem: " << props[i].totalGlobalMem << std::endl;

 
/*        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

        SPRINTF(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
        printf("%s", msg);

        printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
               deviceProp.multiProcessorCount,
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
        printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000
        // This is supported in CUDA 5.0 (runtime API device properties)
        printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n",   deviceProp.memoryBusWidth);

        if (deviceProp.l2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
        }
#else
        // This only available in CUDA 4.0-4.2 (but these were only exposed in the CUDA Driver API)
        int memoryClock;
        getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
        int memBusWidth;
        getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
        printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
        int L2CacheSize;
        getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

        if (L2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
        }
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

