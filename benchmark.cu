
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
void TestMemoryOverhead(BenchParams &params, SystemTopo &topo);
void TestBandwidth(BenchParams &params, SystemTopo &topo);
void TestP2PDeviceBandwidth(BenchParams &params, SystemTopo &topo);
void TestPCIeCongestion(BenchParams &params, SystemTopo &topo);
void TestTaskScalability(BenchParams &params, SystemTopo &topo);

// Test Subfunctions
void MemCopyRun(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData, MEM_OP copyType, MEM_PATTERN patternType, int destIdx, int srcIdx); 
float TimedMemOp(void **MemBlk, long long NumBytes, MEM_OP TimedOp); 
float TimedMemCopyStep(char * destPtr, char *srcPtr, long stepSize, long long blockSize, int numCopiesPerStep, MEM_OP copyType, MEM_PATTERN patternType, int destIdx = 0, int srcIdx = 0);
void MemCopyOp(char * destPtr, char *srcPtr, long stepSize, long long blockSize, int numCopiesPerStep, MEM_OP copyType, int destIdx = 0, int srcIdx = 0);
//void TestHDBandwidth(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData, int socketIdx, int destIdx, int srcIdx, bool HostTest, int &testNum); 
void TestRangeBandwidth(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData, bool testSockets, int &testNum); 

// Support functions
void AllocateMemBlock(SystemTopo &topo, void **destPtr, void **srcPtr, long long numBytes, MEM_OP copyType, int destIdx = 0, int srcIdx = 0);
void FreeMemBlock(SystemTopo &topo, void* destPtr, void *srcPtr, long long numBytes, MEM_OP copyType, int destIdx = 0, int srcIdx = 0);
int CalcRunSteps(std::vector<long long> &blockSteps, long long startStep, long long stopStep, long long numSteps);
void SetMemBlockTransfer(SystemTopo &topo, void *destPtr, void *srcPtr, long long numBytes, MEM_OP copyType, int destIdx, int srcIdx, long long value); 

// Device Utility Functions
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

   // Output system topology to file and screen
   sysTopo.PrintTopology(topoFile);

   // Output device properties for each CUDA enabled GPU to file
   sysTopo.PrintDeviceProps(benchParams);

   // Print actual benchmark parameters for user/script parsing
   benchParams.PrintParams();

   // Run the benchmark per parameters defines in params
   RunTestSuite(benchParams, sysTopo);
  
   return 0;
}

void RunTestSuite(BenchParams &params, SystemTopo &topo) {

   if (params.runMemoryOverheadTest) {
      
      TestMemoryOverhead(params, topo);
   
   }

   if (params.runHDBandwidthTest) {

      TestBandwidth(params, topo);

   }

   if (params.runP2PBandwidthTest) {  
      
      TestP2PDeviceBandwidth(params, topo);
   
   }

   if (params.runPCIeCongestionTest) {

      TestPCIeCongestion(params, topo);

   }

   if (params.runTaskScalabilityTest) { 

      TestTaskScalability(params, topo);

   }

   std::cout << "\nBenchmarks complete!\n" << std::endl;

}

void TestMemoryOverhead(BenchParams &params, SystemTopo &topo) {
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
         
            std::cout << "Test " << testNum++ << " Host Alloc/Free, Pinned/Pageable\t" << "NUMA node: " << numaIdx << " CPU " << socketIdx << std::endl;            
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
            std::cout << "Test " << testNum++ << " Device Alloc/Free \t\t" << "NUMA node: " << numaIdx << " CPU " << socketIdx << " Dev:" << currDev << std::endl;            
            
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

float BurstMemCopy(SystemTopo &topo, long long blockSize, MEM_OP copyType, int destIdx, int srcIdx, int numSteps) {
   float elapsedTime = 0;
   char *destPtr, *srcPtr;

   AllocateMemBlock(topo, (void **) &destPtr, (void **) &srcPtr, blockSize, copyType, destIdx, srcIdx);
   SetMemBlockTransfer(topo, (void *) destPtr, (void *) srcPtr, blockSize, copyType, destIdx, srcIdx, -1); 

//float TimedMemCopyStep(char * destPtr, char *srcPtr, long stepSize, long long blockSize, int numCopiesPerStep, MEM_OP copyType, MEM_PATTERN patternType, int destIdx, int srcIdx) {
   elapsedTime = TimedMemCopyStep((char *) destPtr, (char *) srcPtr, blockSize, blockSize, numSteps, copyType, REPEATED, destIdx, srcIdx);

   FreeMemBlock(topo, (void *) destPtr, (void *) srcPtr, blockSize, copyType, destIdx, srcIdx);

   return elapsedTime;
}

void TestBurstBandwidth(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData, bool testSockets, int &testNum) { 
   int numSockets = 1;
   long long blockSize = pow(2, 26); //set test size for 16 MB
   int numPatterns = 1;

   int stride = 2;
   int matrixWidth = 2 * topo.NumNodes();
   int matrixHeight = numPatterns * matrixWidth;
   burstData.resize(matrixHeight);
   for (int idx = 0; idx < matrixHeight; ++idx) {
      burstData[idx].resize(matrixWidth);
   }
   
   double convConst =(double) blockSize * 1e3f / (double) pow(2.0, 30.0); //(double) blockSize * (double) params.numCopiesPerStepHD * 1000 / (double) pow(2.0, 30.0);
   if (testSockets)
      numSockets = topo.NumSockets();
   
   for (int socketIdx = 0; socketIdx < numSockets; socketIdx++) {
      topo.PinSocket(socketIdx);
 
      for (int srcIdx = 0; srcIdx < topo.NumNodes(); srcIdx++) { 

         //Host-To-Host Memory Transfers
         for (int destIdx = 0; destIdx < topo.NumNodes(); destIdx++) { 
            // HtoH Ranged Transfer - Pageable Memory
            burstData[srcIdx * stride][destIdx * stride] = convConst / BurstMemCopy(topo, blockSize, HOST_HOST_COPY, destIdx, srcIdx, params.numCopiesPerStepHD); 

            // HtoH Ranged Transfer - Pinned Memory Src Host
            burstData[srcIdx * stride + 1][destIdx * stride] = convConst / BurstMemCopy(topo, blockSize, HOST_PINNED_HOST_COPY, destIdx, srcIdx, params.numCopiesPerStepHD); 
            
            // HtoH Ranged Transfer - Pinned Memory Dest Host
            burstData[srcIdx * stride][destIdx * stride + 1] = convConst / BurstMemCopy(topo, blockSize, HOST_HOST_PINNED_COPY, destIdx, srcIdx, params.numCopiesPerStepHD); 

            // HtoH Ranged Transfer - Pinned Memory Both Hosts
            burstData[srcIdx * stride + 1][destIdx * stride + 1] = convConst / BurstMemCopy(topo, blockSize, HOST_HOST_COPY_PINNED, destIdx, srcIdx, params.numCopiesPerStepHD);        
         }

         //Host-Device Memory Transfers
         for (int destIdx = 0; destIdx < params.nDevices; destIdx++) {
            // HtoD Ranged Transfer - Pageable Memory
            std::cout << convConst / BurstMemCopy(topo, blockSize, HOST_DEVICE_COPY, destIdx, srcIdx, params.numCopiesPerStepHD) << ",";        
            
            // DtoH Ranged Transfer - Pageable Memory
            std::cout << convConst / BurstMemCopy(topo, blockSize, DEVICE_HOST_COPY, srcIdx, destIdx, params.numCopiesPerStepHD) << ",";        
            
            // HtoD Ranged Transfer - Pinned Memory
            std::cout << convConst / BurstMemCopy(topo, blockSize, HOST_PINNED_DEVICE_COPY, destIdx, srcIdx, params.numCopiesPerStepHD) << ",";

            // DtoH Ranged Transfer - Pinned Memory
            std::cout << convConst / BurstMemCopy(topo, blockSize, DEVICE_HOST_PINNED_COPY, srcIdx, destIdx, params.numCopiesPerStepHD) << ","; 
         }
         std::cout << std::endl;
      }
   }
}

void PrintBurstMatrix(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData, bool testSockets) {
   int numSockets = 1;
   long long blockSize = pow(2,24); //set test size for 16 MB
   int numPatterns = 1;

   //int stride = 2;
   int matrixWidth = 2 * numSockets * topo.NumNodes();
   int matrixHeight = numPatterns * matrixWidth;
  std::cout << params.numCopiesPerStepHD << " " << 1e3 << std::endl;; 
   if (testSockets)
      numSockets = topo.NumSockets();
   std::cout << "Host-Host Multi-Numa Unidirectional Memory Transfers:" << std::endl;
   std::cout << "Transfer Block Size: " << blockSize << " (Bytes)"<< std::endl;
   std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;
   std::cout << "|\t\t|---------------|-------------------------- Destination ------------------------|" << std::endl;
   std::cout << "|   Transfer \t|---------------|---------------------------------------------------------------|" << std::endl;
   std::cout << "|   Point\t| NUMA \t\t|";
   for (int i = 0; i < topo.NumNodes(); i++)
      std::cout << "\t\t" << i << "\t\t|";
   std::cout << "" << std::endl;
   
   std::cout << "|\t\t| Node \t\t|---------------------------------------------------------------|" << std::endl;
   std::cout << "|\t\t| #     Mem Type";
   for (int i = 0; i < matrixWidth; i++){
      if (i % 2)
         std::cout << "|    Pinned\t";
      else
         std::cout << "|    Pageable\t";
   }
   std::cout << "|"<< std::endl;
   std::cout << "|---------------|-------|-------|---------------------------------------------------------------|" << std::endl;

   for (int i = 0; i < matrixHeight; ++i) {
      std::cout << "|\t\t|\t|";//<< std::endl;
      for (int j = 0; j < matrixWidth + 1; ++j)
         std::cout << "\t|\t";
      std::cout << std::endl; 

      std::cout << "|\t\t| " << i / (matrixHeight / topo.NumNodes()) <<  "\t|";
      if (i % 2)
         std::cout << " Pin\t|    ";
      else
         std::cout << " Page\t|    ";
 
      for (int j = 0; j < matrixWidth; ++j) {
         std::cout << burstData[i][j] << "\t|    ";
      }
          
      std::cout << "\n|\t\t|\t|";
      for (int j = 0; j < matrixWidth + 1; ++j)
         std::cout << "\t|\t";
      std::cout << std::endl;
      
      if (i + 1 < matrixHeight && (i + 1 != ((float) matrixHeight / 2.0)))
         std::cout << "|\t\t|-------|-----------------------------------------------------------------------|" << std::endl;
      else if (i + 1 < matrixHeight)
         std::cout << "|    Source     |-------|-----------------------------------------------------------------------|" << std::endl;
   }
   std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;

   std::cout << "\nHost-To-Device and Device-To-Host Unidirectional Memory Transfers" << std::endl;
   std::cout << "Transfer Block Size: " << blockSize << std::endl;

}

//void TestHDBandwidth(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData, int socketIdx, int srcIdx, int destIdx, bool HostTest, int &testNum) {
void TestBandwidth(BenchParams &params, SystemTopo &topo) {
   std::cout << "\nRunning Host-Device and Device-Host Bandwidth Test...\n" << std::endl;

   std::vector<std::vector<float> > rangeData;
   std::vector<std::vector<float> > burstData;
   std::vector<long long> blockSteps;
   int testNum = 0;

   bool testSockets = false;  

   if (params.runSustainedHD == false)
      params.numCopiesPerStepHD = 1;

   if (params.runBurstHD) {
      std::cout << "Running Burst Bandwidth test...\n" << std::endl;
      TestBurstBandwidth(params, topo, burstData, testSockets, testNum); 
      
      PrintBurstMatrix(params, topo, burstData, testSockets);
      std::cout << "\nFinished Burst Bandwidth test!" << std::endl;
   }

   if (params.runRangeTestHD) {
      std::cout << "\nRunning Ranged Bandwidth test...\n" << std::endl;

      CalcRunSteps(blockSteps, params.rangeHostDeviceBW[0], params.rangeHostDeviceBW[1], params.rangeHostDeviceBW[2]); 
      rangeData.resize(blockSteps.size());
      TestRangeBandwidth(params, topo, blockSteps, rangeData, testSockets, testNum);

      // Output average transfer time (ms) over entire repeated transfer set and block size
      /*for (int blkIdx = 0; blkIdx < blockSteps.size(); ++blkIdx) {
         for (int runIdx = 0; runIdx < rangeData[blkIdx].size(); ++runIdx) {
            rangeData[blkIdx][runIdx] /= (float) params.numCopiesPerStepHD;
         }
      }*/

      // tt == Transfer Time
      std::string dataFileName = "./results/" + params.resultsFile + "_ranged_tt.csv";
      std::ofstream ttResultsFile(dataFileName.c_str());
      PrintResults(ttResultsFile, blockSteps, rangeData, params);

      // Output throughput (GB/S) and block size
      for (int blkIdx = 0; blkIdx < blockSteps.size(); ++blkIdx) {
         for (int runIdx = 0; runIdx < rangeData[blkIdx].size(); ++runIdx) {
            rangeData[blkIdx][runIdx] = ((double) blockSteps[blkIdx]) / rangeData[blkIdx][runIdx];
            rangeData[blkIdx][runIdx] /= pow(2.0, 30.0);
            rangeData[blkIdx][runIdx] *= 10e3f;
         }
      }

      dataFileName = "./results/" + params.resultsFile + "_ranged_bw.csv";
      std::ofstream bwResultsFile(dataFileName.c_str());
      PrintResults(bwResultsFile, blockSteps, rangeData, params);
      
      std::cout << "\nRanged Bandwidth Test Complete!" << std::endl;
   }

   std::cout << "\nHost-Device and Host-Host Bandwidth Test complete!" << std::endl;
}

void TestP2PDeviceBandwidth(BenchParams &params, SystemTopo &topo){
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

void TestPCIeCongestion(BenchParams &params, SystemTopo &topo) {
   std::cout << "Running PCIe congestion test" << std::endl;
   return;
}

void TestTaskScalability(BenchParams &params, SystemTopo &topo) {
   std::cout << "Running task scalability test" << std::endl;
   return;
}

void TestRangeBandwidth(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData, bool testSockets, int &testNum) {
   int numSockets = 1;
   
   if (testSockets)
      numSockets = topo.NumSockets();
   
   for (int socketIdx = 0; socketIdx < numSockets; socketIdx++) {
      topo.PinSocket(socketIdx);
 
      for (int srcIdx = 0; srcIdx < topo.NumNodes(); srcIdx++) { 
         //topo.PinNumaNode(numaSrc);

         //Host To Host Memory Transfers
         for (int destIdx = 0; destIdx < topo.NumNodes(); destIdx++) { 
            // HtoH Ranged Transfer - Pageable Memory
            std::cout << "Test " << testNum++ << " HtoH, Pageable Memory, Repeated Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
            MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_HOST_COPY, REPEATED, destIdx, srcIdx); 
            if (params.runAllPatternsHD) {
               std::cout << "Test " << testNum++ << " HtoH, Pageable Memory, Random\t\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_HOST_COPY, RANDOM, destIdx, srcIdx); 
               std::cout << "Test " << testNum++ << " HtoH, Pageable Memory, Linear Inc Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_HOST_COPY, LINEAR_INC, destIdx, srcIdx); 
               std::cout << "Test " << testNum++ << " HtoH, Pageable Memory, Linear Dec Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_HOST_COPY, LINEAR_DEC, destIdx, srcIdx);
            }

            //HtoH Ranged Transfer - Pinned Memory Src Host
            std::cout << "Test " << testNum++ << " HtoH, Pinned Memory Src, Repeated Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
            MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_PINNED_HOST_COPY, REPEATED, destIdx, srcIdx);
            if (params.runAllPatternsHD){ 
               std::cout << "Test " << testNum++ << " HtoH, Pinned Memory Src, Random Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_PINNED_HOST_COPY, RANDOM, destIdx, srcIdx); 
               std::cout << "Test " << testNum++ << " HtoH, Pinned Memory Src, Linear Inc Addr \tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_PINNED_HOST_COPY, LINEAR_INC, destIdx, srcIdx); 
               std::cout << "Test " << testNum++ << " HtoH, Pinned Memory Src, Linear Dec Addr \tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_PINNED_HOST_COPY, LINEAR_DEC, destIdx, srcIdx); 
            }

            //HtoH Ranged Transfer - Pinned Memory Dest Host
            std::cout << "Test " << testNum++ << " HtoH, Pinned Memory Dest, Repeated Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
            MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_HOST_PINNED_COPY, REPEATED, destIdx, srcIdx); 
            if (params.runAllPatternsHD) {
               std::cout << "Test " << testNum++ << " HtoH, Pinned Memory Dest, Random Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_HOST_PINNED_COPY, RANDOM, destIdx, srcIdx); 
               std::cout << "Test " << testNum++ << " HtoH, Pinned Memory Dest, Linear Inc Addr\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_HOST_PINNED_COPY, LINEAR_INC, destIdx, srcIdx); 
               std::cout << "Test " << testNum++ << " HtoH, Pinned Memory Dest, Linear Dec Addr\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_HOST_PINNED_COPY, LINEAR_DEC, destIdx, srcIdx); 
            }

           //HtoH Ranged Transfer - Pinned Memory Both Hosts
            std::cout << "Test " << testNum++ << " HtoH, Both Pinned Memory, Repeated Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
            MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_HOST_COPY_PINNED, REPEATED, destIdx, srcIdx); 
            if (params.runAllPatternsHD) {
               std::cout << "Test " << testNum++ << " HtoH, Both Pinned Memory, Random Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx  << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_HOST_COPY_PINNED, RANDOM, destIdx, srcIdx); 
               std::cout << "Test " << testNum++ << " HtoH, Both Pinned Memory, Linear Inc Addr\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_HOST_COPY_PINNED, LINEAR_INC, destIdx, srcIdx); 
               std::cout << "Test " << testNum++ << " HtoH, Both Pinned Memory, Linear Dec Addr\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_HOST_COPY_PINNED, LINEAR_DEC, destIdx, srcIdx); 
            }
         }

         //Host-Device PCIe Memory Transfers
         for (int destIdx = 0; destIdx < params.nDevices; destIdx++) {
             // HtoD Ranged Transfer - Pageable Memory
            std::cout << "Test " << testNum++ << " HtoD, Pageable Memory, Repeated Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest Dev: " << destIdx << std::endl;
            MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_DEVICE_COPY, REPEATED, destIdx, srcIdx); 
            if (params.runAllPatternsHD) {
               std::cout << "Test " << testNum++ << " HtoD, Pageable Memory, Random Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest Dev: " << destIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_DEVICE_COPY, RANDOM, destIdx, srcIdx); 
               std::cout << "Test " << testNum++ << " HtoD, Pageable Memory, Linear Inc Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest Dev: " << destIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_DEVICE_COPY, LINEAR_INC, destIdx, srcIdx); 
               std::cout << "Test " << testNum++ << " HtoD, Pageable Memory, Linear Dec Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest Dev: " << destIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_DEVICE_COPY, LINEAR_DEC, destIdx, srcIdx); 
            }

            // DtoH Ranged Transfer - Pageable Memory
            std::cout << "Test " << testNum++ << " DtoH, Pageable Memory, Repeated Addr\t\tCPU: " << socketIdx << "\t\tDev Src: " << srcIdx << "\tNUMA dest: " << srcIdx << std::endl;
            MemCopyRun(params, topo, blockSteps, bandwidthData, DEVICE_HOST_COPY, REPEATED, srcIdx, destIdx); 
            if (params.runAllPatternsHD) {
               std::cout << "Test " << testNum++ << " DtoH, Pageable Memory, Random Addr\t\tCPU: " << socketIdx << "\t\tDev Src: " << destIdx << "\tNUMA dest: " << srcIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, DEVICE_HOST_COPY, RANDOM, srcIdx, destIdx); 
               std::cout << "Test " << testNum++ << " DtoH, Pageable Memory, Linear Inc Addr\t\tCPU: " << socketIdx << "\t\tDev Src: " << destIdx << "\tNUMA dest: " << srcIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, DEVICE_HOST_COPY, LINEAR_INC, srcIdx, destIdx); 
               std::cout << "Test " << testNum++ << " DtoH, Pageable Memory, Linear Dec Addr\t\tCPU: " << socketIdx << "\t\tDev Src: " << destIdx << "\tNUMA dest: " << srcIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, DEVICE_HOST_COPY, LINEAR_DEC, srcIdx, destIdx); 
            }
            
            // HtoD Ranged Transfer - Pinned Memory
            std::cout << "Test " << testNum++ << " HtoD, Pinned Memory, Repeated Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest Dev: " << destIdx << std::endl;
            MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_PINNED_DEVICE_COPY, REPEATED, destIdx, srcIdx); 
            if (params.runAllPatternsHD) {
               std::cout << "Test " << testNum++ << " HtoD, Pinned Memory, Random Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest Dev: " << destIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_PINNED_DEVICE_COPY, RANDOM, destIdx, srcIdx); 
               std::cout << "Test " << testNum++ << " HtoD, Pinned Memory, Linear Inc Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest Dev: " << destIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_PINNED_DEVICE_COPY, LINEAR_INC, destIdx, srcIdx); 
               std::cout << "Test " << testNum++ << " HtoD, Pinned Memory, Linear Dec Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest Dev: " << destIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, HOST_PINNED_DEVICE_COPY, LINEAR_DEC, destIdx, srcIdx); 
            } 

            // DtoH Ranged Transfer - Pinned Memory
            std::cout << "Test " << testNum++ << " DtoH, Pinned Memory, Repeated Addr\t\tCPU: " << socketIdx << "\t\tSrc Dev: " << srcIdx << "\tNUMA Dest: " << srcIdx << std::endl;
            MemCopyRun(params, topo, blockSteps, bandwidthData, DEVICE_HOST_PINNED_COPY, REPEATED, srcIdx, destIdx); 
            if (params.runAllPatternsHD) {
               std::cout << "Test " << testNum++ << " DtoH, Pinned Memory, Random Addr\t\tCPU: " << socketIdx << "\t\tDev Src: " << destIdx << "\tNUMA dest: " << srcIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, DEVICE_HOST_PINNED_COPY, RANDOM, srcIdx, destIdx); 
               std::cout << "Test " << testNum++ << " DtoH, Pinned Memory, Linear Inc Addr\t\tCPU: " << socketIdx << "\t\tDev Src: " << destIdx << "\tNUMA dest: " << srcIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, DEVICE_HOST_PINNED_COPY, LINEAR_INC, srcIdx, destIdx); 
               std::cout << "Test " << testNum++ << " DtoH, Pinned Memory, Linear Dec Addr\t\tCPU: " << socketIdx << "\t\tDev Src: " << destIdx << "\tNUMA dest: " << srcIdx << std::endl;
               MemCopyRun(params, topo, blockSteps, bandwidthData, DEVICE_HOST_PINNED_COPY, LINEAR_DEC, srcIdx, destIdx); 
            }               
         }
      }
   }
}

void MemCopyRun(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData, MEM_OP copyType, MEM_PATTERN patternType, int destIdx, int srcIdx) {
   char *destPtr, *srcPtr; 
   long totalSteps = blockSteps.size();
   
   std::vector<float> timedRun(totalSteps, 0.0);
   long long blockSize = blockSteps[totalSteps - 1 ];

   AllocateMemBlock(topo, (void **) &destPtr, (void **) &srcPtr, blockSize, copyType, destIdx, srcIdx);
   SetMemBlockTransfer(topo, (void *) destPtr, (void *) srcPtr, blockSize, copyType, destIdx, srcIdx, -1);
   
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

   if (copyType == HOST_HOST_COPY) {
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
            default:
               std::cout << "Error: unrecognized memory access pattern during copy operation" << std::endl; 
               break;
         }
      }
   }

   if (copyType == HOST_HOST_COPY) {
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

   return totalTime / (float) numCopiesPerStep;
}

void MemCopyOp(char * destPtr, char *srcPtr, long stepSize, long long blockSize, int numCopiesPerStep, MEM_OP copyType, int destIdx, int srcIdx) {
   switch (copyType) {
      case HOST_HOST_COPY: 
         memcpy((void *) (destPtr), (void *) (srcPtr), stepSize);
         break;
      case HOST_PINNED_HOST_COPY: 
      case HOST_HOST_PINNED_COPY: 
      case HOST_HOST_COPY_PINNED: 
         checkCudaErrors(cudaMemcpyAsync((void *)(destPtr), (void *) (srcPtr), stepSize, cudaMemcpyHostToHost, 0));
         break;
      case DEVICE_HOST_COPY:
         checkCudaErrors(cudaMemcpy((void *) (destPtr), (void *) (srcPtr), stepSize, cudaMemcpyDeviceToHost));
         break;
      case DEVICE_HOST_PINNED_COPY:
         checkCudaErrors(cudaMemcpyAsync((void *) (destPtr), (void *) (srcPtr), stepSize, cudaMemcpyDeviceToHost, 0));
         break;
      case HOST_DEVICE_COPY:
         checkCudaErrors(cudaMemcpy((void *) (destPtr), (void *) (srcPtr), stepSize, cudaMemcpyHostToDevice));
         break;
      case HOST_PINNED_DEVICE_COPY:
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
      case HOST_PINNED_HOST_COPY:  
         checkCudaErrors(cudaHostUnregister((void*) srcPtr));
         topo.FreeMem((void *) srcPtr, numBytes);
         topo.FreeMem((void *) destPtr, numBytes);
         break;
     case HOST_HOST_PINNED_COPY:  
         topo.FreeMem((void *) srcPtr, numBytes);
         checkCudaErrors(cudaHostUnregister((void*) destPtr));
         topo.FreeMem((void *) destPtr, numBytes);
         break;
     case HOST_HOST_COPY_PINNED:  
         checkCudaErrors(cudaHostUnregister((void*) srcPtr));
         topo.FreeMem((void *) srcPtr, numBytes);
         checkCudaErrors(cudaHostUnregister((void*) destPtr));
         topo.FreeMem((void *) destPtr, numBytes);
         break;
      case DEVICE_HOST_COPY:
         topo.FreeDeviceMem(srcPtr, srcIdx);
         topo.FreeMem((void *) destPtr, numBytes);
         break;
      case DEVICE_HOST_PINNED_COPY:
         topo.FreeDeviceMem(srcPtr, srcIdx);
         checkCudaErrors(cudaHostUnregister((void*) destPtr));
         topo.FreeMem((void *) destPtr, numBytes);
         break;
      case HOST_DEVICE_COPY:
         topo.FreeMem((void *) srcPtr, numBytes);
         topo.FreeDeviceMem(destPtr, destIdx);
         break;
      case HOST_PINNED_DEVICE_COPY:
         checkCudaErrors(cudaHostUnregister((void *) srcPtr));
         topo.FreeMem((void *) srcPtr, numBytes);
         topo.FreeDeviceMem(destPtr, destIdx);
         break;
      case PEER_COPY_NO_UVA: 
      case DEVICE_DEVICE_COPY:
      case COPY_UVA:
         topo.FreeDeviceMem(srcPtr, srcIdx);
         topo.FreeDeviceMem(destPtr, destIdx);
         break;
      default:
         std::cout << "Error: unrecognized memory copy operation type for deallocation" << std::endl; 
         break;
   }
}

void SetMemBlockTransfer(SystemTopo &topo, void *destPtr, void *srcPtr, long long numBytes, MEM_OP copyType, int destIdx, int srcIdx, long long value) {
   switch (copyType) {
      case HOST_HOST_COPY: 
      case HOST_PINNED_HOST_COPY: 
      case HOST_HOST_PINNED_COPY: 
      case HOST_HOST_COPY_PINNED: 
         topo.SetHostMem(srcPtr, value, numBytes);
         topo.SetHostMem(destPtr, value, numBytes);
         break;
      case DEVICE_HOST_COPY:
      case DEVICE_HOST_PINNED_COPY:
         topo.SetDeviceMem(srcPtr, value, numBytes, srcIdx);
         topo.SetHostMem(destPtr, value, numBytes);
         break;
      case HOST_DEVICE_COPY:
      case HOST_PINNED_DEVICE_COPY:
         topo.SetHostMem(srcPtr, value, numBytes);
         topo.SetDeviceMem(destPtr, value, numBytes, destIdx);
         break;
      case PEER_COPY_NO_UVA: 
      case DEVICE_DEVICE_COPY:
      case COPY_UVA:
         topo.SetDeviceMem(srcPtr, value, numBytes, srcIdx);
         topo.SetDeviceMem(destPtr, value, numBytes, destIdx);
         break;
      default:
         std::cout << "Error: unrecognized memory copy operation type for mem set" << std::endl; 
         break;
   }
}

void AllocateMemBlock(SystemTopo &topo, void **destPtr, void **srcPtr,long  long numBytes, MEM_OP copyType, int destIdx, int srcIdx) {
   switch (copyType) {
      case HOST_HOST_COPY: 
         *destPtr = topo.AllocMemByNode(destIdx, numBytes);
         *srcPtr =topo.AllocMemByNode(srcIdx, numBytes);
         break;
      case HOST_PINNED_HOST_COPY: 
         *srcPtr =topo.AllocMemByNode(srcIdx, numBytes);
         checkCudaErrors(cudaHostRegister(*srcPtr, numBytes, cudaHostRegisterPortable));
         *destPtr = topo.AllocMemByNode(destIdx, numBytes);
         break;
      case HOST_HOST_PINNED_COPY: 
         *srcPtr =topo.AllocMemByNode(srcIdx, numBytes);
         *destPtr = topo.AllocMemByNode(destIdx, numBytes);
         checkCudaErrors(cudaHostRegister(*destPtr, numBytes, cudaHostRegisterPortable));
         break;
      case HOST_HOST_COPY_PINNED: 
         *srcPtr =topo.AllocMemByNode(srcIdx, numBytes);
         checkCudaErrors(cudaHostRegister(*srcPtr, numBytes, cudaHostRegisterPortable));
         *destPtr = topo.AllocMemByNode(destIdx, numBytes);
         checkCudaErrors(cudaHostRegister(*destPtr, numBytes, cudaHostRegisterPortable));
         break;
      case DEVICE_HOST_COPY:
         topo.AllocDeviceMem(srcPtr, numBytes, srcIdx);
         *destPtr = topo.AllocMemByNode(destIdx, numBytes);
         break;
      case DEVICE_HOST_PINNED_COPY:
         topo.AllocDeviceMem(srcPtr, numBytes, srcIdx);
         *destPtr = topo.AllocMemByNode(destIdx, numBytes);
         checkCudaErrors(cudaHostRegister(*destPtr, numBytes, cudaHostRegisterPortable));
         break;
      case HOST_DEVICE_COPY:
         *srcPtr = topo.AllocMemByNode(srcIdx, numBytes);
         topo.AllocDeviceMem(destPtr, numBytes, destIdx);
         break;
      case HOST_PINNED_DEVICE_COPY:
         *srcPtr = topo.AllocMemByNode(srcIdx, numBytes);
         checkCudaErrors(cudaHostRegister(*srcPtr, numBytes, cudaHostRegisterPortable));
         topo.AllocDeviceMem(destPtr, numBytes, destIdx);
         break;
      case PEER_COPY_NO_UVA: 
      case DEVICE_DEVICE_COPY:
      case COPY_UVA:
         topo.AllocDeviceMem(srcPtr, numBytes, srcIdx);
         topo.AllocDeviceMem(destPtr, numBytes, destIdx);
         break;
      default:
         std::cout << "Error: unrecognized memory copy operation type for allocation" << std::endl; 
         break;
   }
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

void PrintResults(std::ofstream &outFile, std::vector<long long> &steps, std::vector<std::vector<float> > &results, BenchParams &params) {
   
   if (!outFile.is_open()) {
      std::cout << "Failed to open file to print results" << std::endl;
      return;
   }
   std::vector<std::vector<float> >::iterator iter_o;
   std::vector<float>::iterator iter_i;
   std::vector<long long>::iterator iter_l = steps.begin();
   
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

