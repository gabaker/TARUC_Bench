
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



float TimedMemCopy(void * destPtr, void *srcPtr, long numBytes, int numCopysPerStep, MEM_OP copyType) {

   switch (copyType) {

      case HOST_HOST_COPY:  
         break;
      case DEVICE_HOST_COPY:
         break;
      case HOST_DEVICE_COPY:
         break;
      case DEVICE_DEVICE_COPY:
         break;
      default:
         std::cout << "Error: unrecognized timed memory copy operation type" << std::endl; 
         break;
   }
   return 0;
}

void MemCopyRun(BenchParams &params, int numCopiesPerStep, MEM_OP copyType) {
   //Calculate the steps for the run;
   //useful for knowing the number of total steps for various randomization methods

   long startStep = params.rangeHostDeviceBW[0];
   long stopStep = params.rangeHostDeviceBW[1];
   long numSteps = params.rangeHostDeviceBW[2];

   int magStart = max((int)log10(startStep), 1);
   int magStop = log10(stopStep);

   int start = max((int)pow(10, magStart), 10);
   double stepSize = 10 * start / numSteps;
   int extra = (stopStep - pow(10, magStop)) / pow(10, magStop - 1);// (pow(10, magStop - 2) * stepSize);
   int stop = pow(10, magStop - 1) * (10 + extra); 
   int totalSteps = (magStop - magStart) * (numSteps) + extra * numSteps / 10; 
   
   double step = start;

   std::cout << extra << std::endl;
   std::cout << totalSteps << std::endl;
   std::cout << start << std::endl;
   std::cout << stop << std::endl;
   for (long stepNum = 1; stepNum <= totalSteps; ++stepNum) {     
      std::cout << stepNum << " " <<  (long) step << " "<< stepSize << std::endl;
      
      if (stepNum && (stepNum % (numSteps - 1) == 0) && (stop >= stepSize * 10 * numSteps + step)) {
         stepSize *= 10.0;
         std::cout << stepSize << std::endl;
      }
 
      step += stepSize;
   }
}

void TestHostDeviceBandwidth(cudaDeviceProp *props, BenchParams &params, SystemTopo &topo) {
   std::cout << "Running host-device bandwidth test" << std::endl;
   int numCopiesPerStep = 20;

   if (params.runSustainedHD == false) {
      numCopiesPerStep = 1;
   }

   //for (int socketIdx = 0; socketIdx < topo.NumSockets(); socketIdx++) {
      //topo.PinSocket(socketIdx);
 
      /*for (int numaSrc = 0; numaSrc < topo.NumNodes(); numaSrc++) { 
         topo.PinNumaNode(numaSrc);


         //Host to host memory transfers
         for (int numaDest = 0; numaDest < topo.NumNodes(); numaDest++) { 
            topo.PinNumaNode(numaDest);

            MemCopyRun(params, numCopiesPerStep, HOST_HOST_COPY); 
         }

         //Host/Device bandwidth PCIetransfers
         for (int currDev = 0; currDev < nDevices; currDev++) {
            checkCudaErrors(cudaSetDevice(currDev));
         
            MemCopyRun(params, numCopiesPerStep, DEVICE_HOST_COPY); 
            MemCopyRun(params, numCopiesPerStep, HOST_DEVICE_COPY); 
         }
      }*/
   //}

   MemCopyRun(params, numCopiesPerStep, DEVICE_HOST_COPY); 
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
      deviceProps << "CUDA Capability: " << props[i].major << "." << props[i].minor << std::endl;
      deviceProps << "PCI Bus/Device/Domain ID: " <<   props[i].pciBusID << ":" <<  props[i].pciDeviceID << ":" <<  props[i].pciDomainID << std::endl; 
      deviceProps << "Clock: " << props[i].clockRate << std::endl; 
      deviceProps << "Memory Clock: " << props[i].memoryClockRate << std::endl; 
      deviceProps << "Memory Bus Width: " << props[i].memoryBusWidth << std::endl; 
      deviceProps << "Theoretical BW: " << props[i].clockRate << std::endl;
      deviceProps << "Global Mem: " << props[i].totalGlobalMem << std::endl;

 
/*        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

        char msg[256];
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

#endif

        printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
               deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
               deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
               deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
        printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
               deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);*/
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

