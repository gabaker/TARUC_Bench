
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
void RunBenchmarkSuite(BenchParams &params, SystemTopo &topo);
void TestMemoryOverhead(BenchParams &params, SystemTopo &topo);
void HHBurstTransferTest(BenchParams &params, SystemTopo &topo);
void HDBurstTransferTest(BenchParams &params, SystemTopo &topo);
void P2PBurstTransferTest(BenchParams &params, SystemTopo &topo);
void HHRangeTransferTest(BenchParams &params, SystemTopo &topo);
void HDRangeTransferTest(BenchParams &params, SystemTopo &topo);
void P2PRangeTransferTest(BenchParams &params, SystemTopo &topo);
void TestPCIeCongestion(BenchParams &params, SystemTopo &topo);
void TestTaskScalability(BenchParams &params, SystemTopo &topo);

// Test Subfunctions
void MemCopyRun(SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData, MEM_OP copyType, MEM_PATTERN patternType, int destIdx, int srcIdx, int numCopiesPerStep); 
float TimedMemOp(void **MemBlk, long long NumBytes, MEM_OP TimedOp); 
float TimedMemCopyStep(char * destPtr, char *srcPtr, long stepSize, long long blockSize, int numCopiesPerStep, MEM_OP copyType, MEM_PATTERN patternType, int destIdx = 0, int srcIdx = 0);
float BurstMemCopy(SystemTopo &topo, long long blockSize, MEM_OP copyType, int destIdx, int srcIdx, int numSteps, MEM_PATTERN pattern = REPEATED); 
void MemCopyOp(char * destPtr, char *srcPtr, long stepSize, long long blockSize, int numCopiesPerStep, MEM_OP copyType, int destIdx = 0, int srcIdx = 0);

void RangeHDBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData); 
void RangeHHBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData); 
void RangeP2PBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData);

void BurstHDBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData); 
void BurstHHBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData); 
void BurstP2PBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData);  

// Support functions
void AllocateMemBlock(SystemTopo &topo, void **destPtr, void **srcPtr, long long numBytes, MEM_OP copyType, int destIdx = 0, int srcIdx = 0);
void FreeMemBlock(SystemTopo &topo, void* destPtr, void *srcPtr, long long numBytes, MEM_OP copyType, int destIdx = 0, int srcIdx = 0);
void SetMemBlockTransfer(SystemTopo &topo, void *destPtr, void *srcPtr, long long numBytes, MEM_OP copyType, int destIdx, int srcIdx, long long value); 
int CalcRunSteps(std::vector<long long> &blockSteps, long long startStep, long long stopStep, long long numSteps);

// Results output
void PrintRangedHeader(BenchParams &params, SystemTopo &topo, std::ofstream &fileStream, BW_RANGED_TYPE testType); 
void PrintResults(std::ofstream &outFile, std::vector<long long> &steps, std::vector<std::vector<float> > &results, BenchParams &params);
void PrintHHBurstMatrix(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData);
void PrintHDBurstMatrix(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData);
void PrintP2PBurstMatrix(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData);

std::vector<std::string> PatternNames{"Repeated","Random", "Linear Increasing","Linear Decreasing"};
 
/* Benchmark main()
 * 
 * 
 */
int main (int argc, char **argv) {
   BenchParams benchParams;  
   SystemTopo sysTopo;
   
   std::cout << "\nStarting Multi-GPU, Multi-NUMA Performance Test Suite...\n" << std::endl; 
   
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
   std::string topoFileName ="./results/" + benchParams.topoFile;
   std::ofstream topoFile(topoFileName.c_str());

   // Output system topology to file and screen
   sysTopo.PrintTopology(topoFile);

   // Output device properties for each CUDA enabled GPU to file
   sysTopo.PrintDeviceProps(benchParams);

   // Check parameters and fix parameters associated with boolean flags
   if (benchParams.runSustainedTests == false)
      benchParams.numStepRepeats = 1;

   if (benchParams.runSocketTests)
      benchParams.nSockets = sysTopo.NumSockets();

   if (!benchParams.runAllDevices)
      benchParams.nDevices = 1;

   // Print actual benchmark parameters for user/script parsing
   benchParams.PrintParams();

   // Run the benchmark per parameters defined in params
   RunBenchmarkSuite(benchParams, sysTopo);

   std::cout << "\nBenchmarks complete!\n" << std::endl;
  
   return 0;
}

void RunBenchmarkSuite(BenchParams &params, SystemTopo &topo) {

   // Memory overhead tests (allocation and deallocations)
   if (params.runMemoryOverheadTest)
      TestMemoryOverhead(params, topo);

   //Burst bandwidth tests (matricies printed)
   if (params.runHHBandwidthTest && params.runBurstTests)
      HHBurstTransferTest(params, topo);
   
   if (params.runHDBandwidthTest && params.runBurstTests)
      HDBurstTransferTest(params, topo);

   if (params.runP2PBandwidthTest && params.runBurstTests && params.runAllDevices)
      P2PBurstTransferTest(params, topo);

   //Ranged bandwidth tests (cvs files printed)
   if (params.runHHBandwidthTest && params.runRangeTests)
      HHRangeTransferTest(params, topo);
   
   if (params.runHDBandwidthTest && params.runRangeTests)
      HDRangeTransferTest(params, topo);

   if (params.runP2PBandwidthTest && params.runRangeTests && params.runAllDevices)
      P2PRangeTransferTest(params, topo);

   // Congestion benchmark tests
   if (params.runPCIeCongestionTest)
      TestPCIeCongestion(params, topo);

   if (params.runTaskScalabilityTest) 
      TestTaskScalability(params, topo);

}

void TestMemoryOverhead(BenchParams &params, SystemTopo &topo) {
   std::cout << "\nRunning Ranged Memory Overhead Test...\n" << std::endl;
   
   char *deviceMem = NULL, *hostMem = NULL, *hostPinnedMem = NULL;
   std::vector<long long> steps;
   std::vector<std::vector<float> > overheadData;
   int testNum = 0;
  
   CalcRunSteps(steps, params.rangeMemOverhead[0], params.rangeMemOverhead[1], params.rangeMemOverhead[2]);  
   overheadData.resize(steps.size());
   
   // Memory overhead test will run for each device utilizing the cudaMalloc and cudaFree functions
   // on the first iteration of the look, assuming there is atleast one device, the host will run the 
   // pinned and un-pinned memory tests
   for (int socketIdx = 0; socketIdx < params.nSockets; socketIdx++) {
      topo.PinSocket(socketIdx);
 
      for (int numaIdx = 0; numaIdx < topo.NumNodes(); numaIdx++) { 
         topo.PinNumaNode(numaIdx);
        
         std::cout << "Test " << testNum++ << " Host Alloc/Free, Pinned/Pageable\t" << "NUMA node: " << numaIdx << " CPU " << socketIdx << std::endl;            
         // Host based management for CASE 1 & 2
         for (long stepIdx = 0; stepIdx < steps.size(); stepIdx++) {
            long long chunkSize = steps[stepIdx];
            float pinAllocTime = 0, pinFreeTime = 0, hostAllocTime = 0, hostFreeTime = 0;
            
            // repeat same block run and average times
            for (int reIdx = 0; reIdx < params.numStepRepeats; reIdx++) {
               //CASE 2: Host UnPinned Memory Overhead
               hostFreeTime += TimedMemOp((void **) &hostMem, chunkSize, HOST_MALLOC);
               hostAllocTime += TimedMemOp((void **) &hostMem, 0, HOST_FREE);

               if (params.usePinnedMem) {
                  //CASE 1: Host Pinned Memory Overhead
                  pinAllocTime += TimedMemOp((void **) &hostPinnedMem, chunkSize, HOST_PINNED_MALLOC);
                  pinFreeTime += TimedMemOp((void **) &hostPinnedMem, 0, HOST_PINNED_FREE); 
               }
            }
            overheadData[stepIdx].push_back(hostAllocTime / (float) params.numStepRepeats);
            overheadData[stepIdx].push_back(hostFreeTime / (float) params.numStepRepeats);

            overheadData[stepIdx].push_back(pinAllocTime / (float) params.numStepRepeats);
            overheadData[stepIdx].push_back(pinFreeTime / (float) params.numStepRepeats);
         }   
      }

      topo.PinNumaNode(0);
      // Device based memory management for CASE 3 & 4
      for (int currDev = 0; currDev < params.nDevices; currDev++) {
         checkCudaErrors(cudaSetDevice(currDev)); 
         std::cout << "Test " << testNum++ << " Device Alloc/Free \t\t" << " CPU " << socketIdx << " Dev:" << currDev << std::endl;            
         
         for (long stepIdx = 0; stepIdx < steps.size(); stepIdx++) {
            long long chunkSize = steps[stepIdx];
            float devAllocTime = 0, devFreeTime = 0;

            // repeat same block run and average times
            for (int reIdx = 0; reIdx < params.numStepRepeats; reIdx++) {
               // CASE 3: Allocation of device memory  
               devAllocTime += TimedMemOp((void **) &deviceMem, chunkSize, DEVICE_MALLOC);
               // CASE 4: DeAllocation of device memory 
               devFreeTime += TimedMemOp((void **) &deviceMem, 0, DEVICE_FREE);
            }

            overheadData[stepIdx].push_back(devAllocTime / (float) params.numStepRepeats);
            overheadData[stepIdx].push_back(devFreeTime / (float) params.numStepRepeats);
         }
      }
   }
   
   std::string dataFileName = "./results/" + params.resultsFile + "_overhead.csv";
   std::ofstream overheadResultsFile(dataFileName.c_str());
   overheadResultsFile << params.nSockets << ",";
   overheadResultsFile << topo.NumNodes() << ",";
   overheadResultsFile << params.nDevices;
   if (params.usePinnedMem)
      overheadResultsFile << ",t";
   else 
      overheadResultsFile << ",f";

   for (int i = 0; i < params.nDevices; i++)
      overheadResultsFile << "," << topo.GetDeviceName(i);
   overheadResultsFile << std::endl;
   PrintResults(overheadResultsFile, steps, overheadData, params);

   std::cout << "\nMemory Overhead Test Complete!" << std::endl;
   
}

void HHBurstTransferTest(BenchParams &params, SystemTopo &topo) {
   std::cout << "\nRunning Host-Host Burst Bandwidth Tests...\n" << std::endl;

   std::vector<std::vector<float> > burstData;

   BurstHHBandwidthRun(params, topo, burstData); 
   PrintHHBurstMatrix(params, topo, burstData);
}

void HDBurstTransferTest(BenchParams &params, SystemTopo &topo) {
   std::cout << "\nRunning Host-Device Burst Bandwidth Tests...\n" << std::endl;

   std::vector<std::vector<float> > burstData;

   BurstHDBandwidthRun(params, topo, burstData);  
   
   PrintHDBurstMatrix(params, topo, burstData);
   
}

void P2PBurstTransferTest(BenchParams &params, SystemTopo &topo) {
   std::cout << "\nRunning Device-Device Burst Bandwidth Tests...\n" << std::endl;
   
   std::vector<std::vector<float> > burstData;

   BurstP2PBandwidthRun(params, topo, burstData);
 
   PrintP2PBurstMatrix(params, topo, burstData);
}

void HHRangeTransferTest(BenchParams &params, SystemTopo &topo) {
   std::cout << "\nRunning Ranged Host-Host Bandwidth Tests...\n" << std::endl;
   
   std::vector<std::vector<float> > rangeData;
   std::vector<long long> steps;
  
   CalcRunSteps(steps, params.rangeHostHostBW[0], params.rangeHostHostBW[1], params.rangeHostHostBW[2]); 
   rangeData.resize(steps.size());
   
   RangeHHBandwidthRun(params, topo, steps, rangeData);

   // tt == Transfer Time
   std::string dataFileName = "./results/" + params.resultsFile + "_ranged_hh_tt.csv";
   std::ofstream ttResultsFileHH(dataFileName.c_str());
   PrintRangedHeader(params, topo, ttResultsFileHH, HH); 
   PrintResults(ttResultsFileHH, steps, rangeData, params);

   // Output throughput (GB/S) and block size
   for (int blkIdx = 0; blkIdx < steps.size(); ++blkIdx) {
      for (int runIdx = 0; runIdx < rangeData[blkIdx].size(); ++runIdx) {
         rangeData[blkIdx][runIdx] = ((double) steps[blkIdx]) / rangeData[blkIdx][runIdx];
         rangeData[blkIdx][runIdx] /= pow(2.0, 30.0);
      }
   }

   dataFileName = "./results/" + params.resultsFile + "_ranged_hh_bw.csv";
   std::ofstream bwResultsFileHH(dataFileName.c_str());
   PrintRangedHeader(params, topo, bwResultsFileHH, HH); 
   PrintResults(bwResultsFileHH, steps, rangeData, params);

   std::cout << "\nRanged Host-Host Bandwidth Tests complete!" << std::endl;
}

void HDRangeTransferTest(BenchParams &params, SystemTopo &topo) {
   std::cout << "\nRunning Ranged Host-Device Bandwidth Tests...\n" << std::endl;
   
   std::vector<std::vector<float> > rangeData;
   std::vector<long long> steps;

   CalcRunSteps(steps, params.rangeHostDeviceBW[0], params.rangeHostDeviceBW[1], params.rangeHostDeviceBW[2]); 
   rangeData.resize(steps.size());
   
   RangeHDBandwidthRun(params, topo, steps, rangeData);
   
   // tt == Transfer Time
   std::string dataFileName = "./results/" + params.resultsFile + "_ranged_hd_tt.csv";
   std::ofstream ttResultsFileHD(dataFileName.c_str());
   PrintRangedHeader(params, topo, ttResultsFileHD, HD); 
   PrintResults(ttResultsFileHD, steps, rangeData, params);

   // Output throughput (GB/S) and block size
   for (int blkIdx = 0; blkIdx < steps.size(); ++blkIdx) {
      for (int runIdx = 0; runIdx < rangeData[blkIdx].size(); ++runIdx) {
         rangeData[blkIdx][runIdx] = ((double) steps[blkIdx]) / rangeData[blkIdx][runIdx];
         rangeData[blkIdx][runIdx] /= pow(2.0, 30.0);
      }
   }

   dataFileName = "./results/" + params.resultsFile + "_ranged_hd_bw.csv";
   std::ofstream bwResultsFileHD(dataFileName.c_str());
   PrintRangedHeader(params, topo, bwResultsFileHD, HD); 
   PrintResults(bwResultsFileHD, steps, rangeData, params);

   std::cout << "\nHost-Device Ranged Bandwidth Tests complete!" << std::endl;
}

void P2PRangeTransferTest(BenchParams &params, SystemTopo &topo){
   std::cout << "\nRunning P2P Device Ranged Bandwidth test..." << std::endl;

   std::vector<std::vector<float> > rangeData;
   std::vector<long long> steps;

   CalcRunSteps(steps, params.rangeDeviceBW[0], params.rangeDeviceBW[1], params.rangeDeviceBW[2]); 
   rangeData.resize(steps.size());
   
   RangeP2PBandwidthRun(params, topo, steps, rangeData);

   // tt == Transfer Time
   std::string dataFileName = "./results/" + params.resultsFile + "_ranged_p2p_tt.csv";
   std::ofstream ttResultsFileP2P(dataFileName.c_str());
   PrintRangedHeader(params, topo, ttResultsFileP2P, P2P); 
   PrintResults(ttResultsFileP2P, steps, rangeData, params);

   // Output throughput (GB/S) and block size
   for (int blkIdx = 0; blkIdx < steps.size(); ++blkIdx) {
      for (int runIdx = 0; runIdx < rangeData[blkIdx].size(); ++runIdx) {
         rangeData[blkIdx][runIdx] = ((double) steps[blkIdx]) / rangeData[blkIdx][runIdx];
         rangeData[blkIdx][runIdx] /= pow(2.0, 30.0);
      }
   }

   dataFileName = "./results/" + params.resultsFile + "_ranged_p2p_bw.csv";
   std::ofstream bwResultsFileP2P(dataFileName.c_str());
   PrintRangedHeader(params, topo, bwResultsFileP2P, P2P); 
   PrintResults(bwResultsFileP2P, steps, rangeData, params);

   std::cout << "\nP2P Device Ranged Bandwidth Test Complete!" << std::endl;
}

void TestPCIeCongestion(BenchParams &params, SystemTopo &topo) {
   std::cout << "Running PCIe congestion tests..." << std::endl;


   std::cout << "PCIe congestion tests complete!" << std::endl;
}

void TestTaskScalability(BenchParams &params, SystemTopo &topo) {
   std::cout << "\nRunning task scalability tests..." << std::endl;


   std::cout << "\nTask scalability tests complete!" << std::endl;
}

void BurstHHBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData) { 
   long long blockSize = params.burstBlockSize;
   int numNodes = topo.NumNodes();
   int numSockets = params.nSockets;
   int numPatterns = 1;

   if (params.runPatternsHD)
      numPatterns = NUM_PATTERNS;

   burstData.resize(numPatterns * numSockets);
   double convConst =(double) blockSize / (double) pow(2.0, 30.0); 

   for (int socketIdx = 0; socketIdx < numSockets; socketIdx++) {
      topo.PinSocket(socketIdx);
      
      for (int patternNum = 0; patternNum < numPatterns; patternNum ++) {
   
         MEM_PATTERN pattern = REPEATED;
         if (patternNum == 1)
            pattern = LINEAR_INC;
         if (patternNum == 2)
            pattern = LINEAR_DEC;
      
         for (int srcIdx = 0; srcIdx < numNodes; srcIdx++) { 

            for (int destIdx = 0; destIdx < numNodes; destIdx++) { 
               // HtoH Ranged Transfer - Pageable Memory
               int rowIdx = socketIdx * numPatterns + patternNum;
               burstData[rowIdx].push_back(convConst / BurstMemCopy(topo, blockSize, HOST_HOST_COPY, destIdx, srcIdx, params.numStepRepeats, pattern));        

               // HtoH Ranged Transfer - Pinned Memory Src Host
               burstData[rowIdx].push_back(convConst / BurstMemCopy(topo, blockSize, HOST_PINNED_HOST_COPY, destIdx, srcIdx, params.numStepRepeats, pattern));        
               
               if (params.usePinnedMem) {
                  // HtoH Ranged Transfer - Pinned Memory Dest
                  burstData[rowIdx].push_back(convConst / BurstMemCopy(topo, blockSize, HOST_HOST_PINNED_COPY, destIdx, srcIdx, params.numStepRepeats, pattern));        

                  // HtoH Ranged Transfer - Pinned Memory Both Hosts
                  burstData[rowIdx].push_back(convConst / BurstMemCopy(topo, blockSize, HOST_HOST_COPY_PINNED, destIdx, srcIdx, params.numStepRepeats, pattern));
               }       
            }
         }
      }
   }
}

void BurstHDBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData) { 
   long long blockSize = params.burstBlockSize;
   double convConst = (double) blockSize / (double) pow(2.0, 30.0); 

   int numSockets = params.nSockets;
   int numPatterns = 1;
   if (params.runPatternsHD)
      numPatterns = NUM_PATTERNS;
   
   burstData.resize(numPatterns * numSockets);
   for (int socketIdx = 0; socketIdx < numSockets; socketIdx++) {
      topo.PinSocket(socketIdx);
      
      for (int patternNum = 0; patternNum < numPatterns; patternNum++) {
      
         MEM_PATTERN pattern = REPEATED;
         if (patternNum == 1)
            pattern = LINEAR_INC;
         if (patternNum == 2)
            pattern = LINEAR_DEC;
    
         for (int srcIdx = 0; srcIdx < topo.NumNodes(); srcIdx++) { 

            //Host-Device Memory Transfers
            for (int destIdx = 0; destIdx < params.nDevices; destIdx++) {
               topo.SetActiveDevice(destIdx); 
               int rowIdx = socketIdx * numPatterns + patternNum; 

               // HtoD Ranged Transfer - Pageable Memory
               burstData[rowIdx].push_back( convConst / BurstMemCopy(topo, blockSize, HOST_DEVICE_COPY, destIdx, srcIdx, params.numStepRepeats, pattern));        
               
               // DtoH Ranged Transfer - Pageable Memory
               burstData[rowIdx].push_back( convConst / BurstMemCopy(topo, blockSize, DEVICE_HOST_COPY, srcIdx, destIdx, params.numStepRepeats, pattern));        
               
               if ( params.usePinnedMem) {      
                  // HtoD Ranged Transfer - Pinned Memory
                  burstData[rowIdx].push_back( convConst / BurstMemCopy(topo, blockSize, HOST_PINNED_DEVICE_COPY, destIdx, srcIdx, params.numStepRepeats, pattern));

                  // DtoH Ranged Transfer - Pinned Memory
                  burstData[rowIdx].push_back( convConst / BurstMemCopy(topo, blockSize, DEVICE_HOST_PINNED_COPY, srcIdx, destIdx, params.numStepRepeats, pattern)); 
               }
            }
         }
      }
   }
}

void BurstP2PBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData) { 
   long long blockSize = params.burstBlockSize;
   double convConst = (double) blockSize / (double) pow(2.0, 30.0); 
   
   burstData.resize(topo.NumGPUs() * params.nSockets);
   for (int socketIdx = 0; socketIdx < params.nSockets; socketIdx++) {
      topo.PinSocket(socketIdx);
 
      for (int srcIdx = 0; srcIdx < topo.NumGPUs(); srcIdx++) { 
         //topo.SetActiveDevice(srcIdx); 
         for (int destIdx = 0; destIdx < topo.NumGPUs(); destIdx++) { 
            // DtoD Burst Transfer - No Peer, No UVA
            burstData[socketIdx * topo.NumGPUs() + srcIdx].push_back(convConst / BurstMemCopy(topo, blockSize, DEVICE_DEVICE_COPY, destIdx, srcIdx, params.numStepRepeats)); 
            // DtoD Burst Transfer - Peer, No UVA
            if (topo.DeviceGroupCanP2P(srcIdx, destIdx)) {
               topo.DeviceGroupSetP2P(srcIdx, destIdx, true);
               burstData[socketIdx * topo.NumGPUs() + srcIdx].push_back(convConst / BurstMemCopy(topo, blockSize, PEER_COPY_NO_UVA, destIdx, srcIdx, params.numStepRepeats)); 
               topo.DeviceGroupSetP2P(srcIdx, destIdx, false);
            }

            if (topo.DeviceGroupUVA(srcIdx, destIdx)) {  
               // DtoD Burst Transfer - No Peer, UVA
               burstData[socketIdx * topo.NumGPUs() + srcIdx].push_back(convConst / BurstMemCopy(topo, blockSize, COPY_UVA, destIdx, srcIdx, params.numStepRepeats)); 
               
               // DtoD Burst Transfer - Peer, UVA
               if (topo.DeviceGroupCanP2P(srcIdx, destIdx)) {
                  topo.DeviceGroupSetP2P(srcIdx, destIdx, true);
                  burstData[socketIdx * topo.NumGPUs() + srcIdx].push_back( convConst / BurstMemCopy(topo, blockSize, COPY_UVA, destIdx, srcIdx, params.numStepRepeats));        
                  topo.DeviceGroupSetP2P(srcIdx, destIdx, false);
               }
            }
         }
      }
   }
}

void RangeHHBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData) {
   int testNum = 0;
   long numRepeats = params.numStepRepeats;
   
   for (int socketIdx = 0; socketIdx < params.nSockets; socketIdx++) {
      topo.PinSocket(socketIdx);
 
      for (int srcIdx = 0; srcIdx < topo.NumNodes(); srcIdx++) { 

         //Host To Host Memory Transfers
         for (int destIdx = 0; destIdx < topo.NumNodes(); destIdx++) { 
            // HtoH Ranged Transfer - Pageable Memory
            std::cout << "Test " << testNum++ << " HtoH, Pageable Memory, Repeated Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
            MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COPY, REPEATED, destIdx, srcIdx, numRepeats); 
            if (params.runPatternsHD) {
               std::cout << "Test " << testNum++ << " HtoH, Pageable Memory, Linear Inc Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COPY, LINEAR_INC, destIdx, srcIdx, numRepeats); 
               std::cout << "Test " << testNum++ << " HtoH, Pageable Memory, Linear Dec Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COPY, LINEAR_DEC, destIdx, srcIdx, numRepeats);
            }

            //HtoH Ranged Transfer - Pinned Memory Src Host
            std::cout << "Test " << testNum++ << " HtoH, Pinned Memory Src, Repeated Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
            MemCopyRun(topo, blockSteps, bandwidthData, HOST_PINNED_HOST_COPY, REPEATED, destIdx, srcIdx, numRepeats);
            if (params.runPatternsHD){ 
               std::cout << "Test " << testNum++ << " HtoH, Pinned Memory Src, Linear Inc Addr \tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_PINNED_HOST_COPY, LINEAR_INC, destIdx, srcIdx, numRepeats); 
               std::cout << "Test " << testNum++ << " HtoH, Pinned Memory Src, Linear Dec Addr \tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_PINNED_HOST_COPY, LINEAR_DEC, destIdx, srcIdx, numRepeats); 
            }

            if (params.usePinnedMem) {
               //HtoH Ranged Transfer - Pinned Memory Dest Host
               std::cout << "Test " << testNum++ << " HtoH, Pinned Memory Dest, Repeated Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_PINNED_COPY, REPEATED, destIdx, srcIdx, numRepeats); 
               if (params.runPatternsHD) {
                  std::cout << "Test " << testNum++ << " HtoH, Pinned Memory Dest, Linear Inc Addr\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_PINNED_COPY, LINEAR_INC, destIdx, srcIdx, numRepeats); 
                  std::cout << "Test " << testNum++ << " HtoH, Pinned Memory Dest, Linear Dec Addr\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_PINNED_COPY, LINEAR_DEC, destIdx, srcIdx, numRepeats); 
               }

              //HtoH Ranged Transfer - Pinned Memory Both Hosts
               std::cout << "Test " << testNum++ << " HtoH, Both Pinned Memory, Repeated Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COPY_PINNED, REPEATED, destIdx, srcIdx, numRepeats); 
               if (params.runPatternsHD) {
                  std::cout << "Test " << testNum++ << " HtoH, Both Pinned Memory, Linear Inc Addr\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COPY_PINNED, LINEAR_INC, destIdx, srcIdx, numRepeats); 
                  std::cout << "Test " << testNum++ << " HtoH, Both Pinned Memory, Linear Dec Addr\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COPY_PINNED, LINEAR_DEC, destIdx, srcIdx, numRepeats);
               } 
            }
         }
      }
   }
}

void RangeHDBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData) {
   int testNum = 0;
   long numRepeats = params.numStepRepeats;  
 
   for (int socketIdx = 0; socketIdx < params.nSockets; socketIdx++) {
      topo.PinSocket(socketIdx);
 
      for (int srcIdx = 0; srcIdx < topo.NumNodes(); srcIdx++) { 

         //Host-Device PCIe Memory Transfers
         for (int destIdx = 0; destIdx < params.nDevices; destIdx++) {
             // HtoD Ranged Transfer - Pageable Memory
            std::cout << "Test " << testNum++ << " HtoD, Pageable Memory, Repeated Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest Dev: " << destIdx << std::endl;
            MemCopyRun(topo, blockSteps, bandwidthData, HOST_DEVICE_COPY, REPEATED, destIdx, srcIdx, numRepeats); 
            if (params.runPatternsHD) {
               std::cout << "Test " << testNum++ << " HtoD, Pageable Memory, Linear Inc Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest Dev: " << destIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_DEVICE_COPY, LINEAR_INC, destIdx, srcIdx, numRepeats); 
               std::cout << "Test " << testNum++ << " HtoD, Pageable Memory, Linear Dec Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest Dev: " << destIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_DEVICE_COPY, LINEAR_DEC, destIdx, srcIdx, numRepeats); 
            }

            // DtoH Ranged Transfer - Pageable Memory
            std::cout << "Test " << testNum++ << " DtoH, Pageable Memory, Repeated Addr\t\tCPU: " << socketIdx << "\t\tDev Src: " << srcIdx << "\tNUMA dest: " << srcIdx << std::endl;
            MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_COPY, REPEATED, srcIdx, destIdx, numRepeats); 
            if (params.runPatternsHD) {
               std::cout << "Test " << testNum++ << " DtoH, Pageable Memory, Linear Inc Addr\t\tCPU: " << socketIdx << "\t\tDev Src: " << destIdx << "\tNUMA dest: " << srcIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_COPY, LINEAR_INC, srcIdx, destIdx, numRepeats); 
               std::cout << "Test " << testNum++ << " DtoH, Pageable Memory, Linear Dec Addr\t\tCPU: " << socketIdx << "\t\tDev Src: " << destIdx << "\tNUMA dest: " << srcIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_COPY, LINEAR_DEC, srcIdx, destIdx, numRepeats); 
            }
            
            if (params.usePinnedMem) {
               // HtoD Ranged Transfer - Pinned Memory
               std::cout << "Test " << testNum++ << " HtoD, Pinned Memory, Repeated Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest Dev: " << destIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_PINNED_DEVICE_COPY, REPEATED, destIdx, srcIdx, numRepeats); 
               if (params.runPatternsHD) {
                  std::cout << "Test " << testNum++ << " HtoD, Pinned Memory, Linear Inc Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest Dev: " << destIdx << std::endl;
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_PINNED_DEVICE_COPY, LINEAR_INC, destIdx, srcIdx, numRepeats); 
                  std::cout << "Test " << testNum++ << " HtoD, Pinned Memory, Linear Dec Addr\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest Dev: " << destIdx << std::endl;
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_PINNED_DEVICE_COPY, LINEAR_DEC, destIdx, srcIdx, numRepeats); 
               } 

               // DtoH Ranged Transfer - Pinned Memory
               std::cout << "Test " << testNum++ << " DtoH, Pinned Memory, Repeated Addr\t\tCPU: " << socketIdx << "\t\tSrc Dev: " << srcIdx << "\tNUMA Dest: " << srcIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_PINNED_COPY, REPEATED, srcIdx, destIdx, numRepeats); 
               if (params.runPatternsHD) {
                  std::cout << "Test " << testNum++ << " DtoH, Pinned Memory, Linear Inc Addr\t\tCPU: " << socketIdx << "\t\tDev Src: " << destIdx << "\tNUMA dest: " << srcIdx << std::endl;
                  MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_PINNED_COPY, LINEAR_INC, srcIdx, destIdx, numRepeats); 
                  std::cout << "Test " << testNum++ << " DtoH, Pinned Memory, Linear Dec Addr\t\tCPU: " << socketIdx << "\t\tDev Src: " << destIdx << "\tNUMA dest: " << srcIdx << std::endl;
                  MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_PINNED_COPY, LINEAR_DEC, srcIdx, destIdx, numRepeats);
               } 
            }               
         }
      }
   }
}

void RangeP2PBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData) {
   int testNum = 0;
   long numRepeats = params.numStepRepeats;  
   
   for (int socketIdx = 0; socketIdx < params.nSockets; socketIdx++) {
      topo.PinSocket(socketIdx);
 
      for (int srcIdx = 0; srcIdx < topo.NumGPUs(); srcIdx++) { 

         for (int destIdx = 0; destIdx < topo.NumGPUs(); destIdx++) { 
            // DtoD Ranged Transfer - No Peer, No UVA
            std::cout << "Test " << testNum++ << " Device-To-Device, No Peer, No UVA\tCPU: " << socketIdx << "\tSrc Device: " << srcIdx << "\tDest Device: " << destIdx << std::endl;
            MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_DEVICE_COPY, REPEATED, destIdx, srcIdx, numRepeats); 

            // DtoD Ranged Transfer - Peer, No UVA
            if (topo.DeviceGroupCanP2P(srcIdx, destIdx)) {
            std::cout << "Test " << testNum++ << " Device-To-Device, Peer Enabled, No UVA\tCPU: " << socketIdx << "\tSrc Device: " << srcIdx << "\tDest Device: " << destIdx << std::endl;
               topo.DeviceGroupSetP2P(srcIdx, destIdx, true);
               MemCopyRun(topo, blockSteps, bandwidthData, PEER_COPY_NO_UVA, REPEATED, destIdx, srcIdx, numRepeats);  
               topo.DeviceGroupSetP2P(srcIdx, destIdx, false);
            }
            
            if (topo.DeviceGroupUVA(srcIdx, destIdx)) {  
               // DtoD Ranged Transfer - No Peer, UVA
               std::cout << "Test " << testNum++ << " Device-To-Device, No Peer, UVA\t\tCPU: " << socketIdx << "\tSrc Device: " << srcIdx << "\tDest Device: " << destIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, COPY_UVA, REPEATED, destIdx, srcIdx, numRepeats); 
 
               // DtoD Ranged Transfer - Peer, UVA
               if (topo.DeviceGroupCanP2P(srcIdx, destIdx)) {
                  std::cout << "Test " << testNum++ << " Device-To-Device, Peer Enabled, No UVA\tCPU: " << socketIdx << "\tSrc Device: " << srcIdx << "\tDest Device: " << destIdx << std::endl;
                  topo.DeviceGroupSetP2P(srcIdx, destIdx, true);
                  MemCopyRun(topo, blockSteps, bandwidthData, COPY_UVA, REPEATED, destIdx, srcIdx, numRepeats); 
                  topo.DeviceGroupSetP2P(srcIdx, destIdx, false);
               }
            }
         }
      }
   }
}

void MemCopyRun(SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData, MEM_OP copyType, MEM_PATTERN patternType, int destIdx, int srcIdx, int numCopiesPerStep) {
   char *destPtr, *srcPtr; 
   long totalSteps = blockSteps.size();
   
   std::vector<float> timedRun(totalSteps, 0.0);
   long long blockSize = blockSteps[totalSteps - 1 ];

   AllocateMemBlock(topo, (void **) &destPtr, (void **) &srcPtr, blockSize, copyType, destIdx, srcIdx);
   SetMemBlockTransfer(topo, (void *) destPtr, (void *) srcPtr, blockSize, copyType, destIdx, srcIdx, -1);
   
   for (long stepNum = 0; stepNum < totalSteps; ++stepNum) { 

      bandwidthData[stepNum].push_back(TimedMemCopyStep((char *) destPtr, (char *) srcPtr, blockSteps[stepNum], blockSize, numCopiesPerStep, copyType, patternType, destIdx, srcIdx));

   }
   
   FreeMemBlock(topo, (void *) destPtr, (void *) srcPtr, blockSize, copyType, destIdx, srcIdx);
}

float BurstMemCopy(SystemTopo &topo, long long blockSize, MEM_OP copyType, int destIdx, int srcIdx, int numSteps, MEM_PATTERN pattern) {  
   float elapsedTime = 0;
   char *destPtr, *srcPtr;

   AllocateMemBlock(topo, (void **) &destPtr, (void **) &srcPtr, blockSize, copyType, destIdx, srcIdx);
   SetMemBlockTransfer(topo, (void *) destPtr, (void *) srcPtr, blockSize, copyType, destIdx, srcIdx, -1); 

   elapsedTime = TimedMemCopyStep((char *) destPtr, (char *) srcPtr, blockSize, blockSize, numSteps, copyType, pattern, destIdx, srcIdx);

   FreeMemBlock(topo, (void *) destPtr, (void *) srcPtr, blockSize, copyType, destIdx, srcIdx);

   return elapsedTime;
}

float TimedMemCopyStep(char * destPtr, char *srcPtr, long stepSize, long long blockSize, int numCopiesPerStep, MEM_OP copyType, MEM_PATTERN patternType, int destIdx, int srcIdx) {
   long long offset = 0;
   float totalTime = 0; 
   long long maxFrameSize = pow(2, 27);
   long long gap = maxFrameSize - stepSize;

   bool usingPattern = false;
   if (blockSize < maxFrameSize) {
      numCopiesPerStep *= 5;
      switch (patternType) {
         case LINEAR_INC:
            usingPattern = true;
            offset = 0;
            break;
         case LINEAR_DEC:
            usingPattern = true;
            offset = blockSize - stepSize;
            break;
         default:
            usingPattern = false;
            break;
      }
   }

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

      if (usingPattern) {
         switch (patternType) {
       
           case REPEATED:
               offset = 0;
               break;
            case LINEAR_INC:
               offset += gap;
               if (offset > blockSize)
                  offset = 0;
               break;
            case LINEAR_DEC:
               offset -= gap;
               if (offset < 0)
                  offset = blockSize - stepSize;
               break;
            default:
               offset = 0;
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

   return totalTime / 1000.0 / (double) numCopiesPerStep;
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
         std::cout << "Error: unrecognized memory copy operation type for deallocation!" << std::endl; 
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
         *srcPtr = topo.AllocMemByNode(srcIdx, numBytes);
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
         std::cout << "Error: unrecognized memory copy operation type for allocation!" << std::endl; 
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
   
   float OpTime = 0;
   
   #ifdef USING_CPP
   start_c = std::chrono::high_resolution_clock::now();
   #else
   gettimeofday(&start_t, NULL);
   #endif

   switch (TimedOp) {
      case HOST_MALLOC:
         *MemBlk = malloc(NumBytes); 
         break;
      case HOST_PINNED_MALLOC:
         checkCudaErrors(cudaMallocHost(MemBlk, NumBytes));
         break;
      case DEVICE_MALLOC:
         checkCudaErrors(cudaMalloc(MemBlk, NumBytes));
         break;
      case HOST_FREE:
         free(*MemBlk);
         break;
      case HOST_PINNED_FREE:
         checkCudaErrors(cudaFreeHost(*MemBlk));
         break;
      case DEVICE_FREE:
         checkCudaErrors(cudaFree(*MemBlk)); 
         break;
      default:
         std::cout << "Error: unrecognized timed memory operation type!" << std::endl; 
         break;
   }

   #ifdef USING_CPP
   stop_c = std::chrono::high_resolution_clock::now();
   total_c = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_c - start_c);      
   OpTime = (float) total_c.count() * NANO_TO_MILLI;
   #else
   gettimeofday(&stop_t, NULL);
   timersub(&stop_t, &start_t, &total_t);
   OpTime = (float) total_t.tv_usec * MICRO_TO_MILLI;
   #endif

   return OpTime;
}
/*float TimedMemOp(void **MemBlk, long long NumBytes, MEM_OP TimedOp) {
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
         std::cout << "Error: unrecognized timed memory operation type!" << std::endl; 
         break;
   }
   checkCudaErrors(cudaEventDestroy(start_e));
   checkCudaErrors(cudaEventDestroy(stop_e));

   return OpTime;
}*/

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

void PrintP2PBurstMatrix(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData) {
   long long blockSize = params.burstBlockSize;
   int numSockets = params.nSockets;
   std::vector<int> deviceIdxs;
   deviceIdxs.resize(params.nDevices, 0);
   int dataIdx = 0;
   
   int matrixWidth = params.nDevices;
   int matrixHeight = params.nDevices * 4;
   std::cout << "\nDevice-To-Device Unidirectional Memory Transfers:" << std::endl;
   std::cout << "Transfer Block Size: " << blockSize / BYTES_TO_MEGA << " (MB)"<< std::endl;
  
   for (int socketIdx = 0; socketIdx < numSockets; socketIdx++) {
      std::cout << "\nInitiating Socket: " << socketIdx << std::endl;
      
      std::cout << "-----------------------------------------"; 
      for (int i = 0; i < matrixWidth; i++)
         std::cout << "----------------";
      std::cout << std::endl;

      std::cout << "|\t\t|-----------------------|"; 
      for (int i = 0; i < matrixWidth * 8 - 7; i++)
         std::cout << "-";

      std::cout << " Destination ";
      for (int i = 0; i < matrixWidth * 8 - 7; i++)
         std::cout << "-";
      std::cout << "|" << std::endl;
      
      std::cout << "|\t\t| GPU   | Transfer\t";
      for (int i = 0; i < matrixWidth; i++)
         std::cout << "|---------------";
      std::cout << "|" << std::endl;

      std::cout << "|\t\t|   #   | Type\t\t|";
      for (int i = 0; i < matrixWidth; i++)
         std::cout << "\t" << i << "\t|";
      std::cout << std::endl;

      std::cout << "|---------------|-----------------------"; 
      for (int i = 0; i < matrixWidth; i++)
         std::cout << "|---------------";
      std::cout << "|" << std::endl;


      std::cout << std::setprecision(2) << std::fixed;          
      
      std::fill(deviceIdxs.begin(), deviceIdxs.end(), 0);
      for (int i = 0; i < matrixHeight; ++i) {

         std::cout << "|\t\t|  " << i  / 4 <<  "\t|";
         if (i % 4 == 0) {
            std::cout << " Standard D2D\t|";
         } else if (i % 4 == 1) {
            std::cout << " Peer, No UVA\t|";
         } else if (i % 4 == 2) {
            std::cout << " No Peer, UVA\t|";
         } else { 
            std::cout << " Peer, UVA\t|";
         }
         
         if (i % 4 == 0) {
            //deviceIdxs.resize(matrixWidth, 0);
            //deviceIdxs.assign(deviceIdxs.begin(), deviceIdxs.end(), 0);
            std::fill(deviceIdxs.begin(), deviceIdxs.end(), 0);
         }
         dataIdx = 0;
         for (int j = 0; j < matrixWidth; ++j) {
            if (i % 4 == 0) {
               std::cout << "      " << burstData[socketIdx * matrixWidth + i / 4][dataIdx + deviceIdxs[j]] << "\t|";
               deviceIdxs[j]++;
            } else if ((i % 4 == 1) && topo.DeviceGroupCanP2P(i / 4, j)) {
               std::cout << "      " << burstData[socketIdx * matrixWidth + i / 4][dataIdx + deviceIdxs[j]] << "\t|";
               deviceIdxs[j]++;
            } else if ((i % 4 == 2) && topo.DeviceGroupUVA(i / 4, j)) {
               std::cout << "      " << burstData[socketIdx * matrixWidth + i / 4][dataIdx + deviceIdxs[j]] << "\t|";
               deviceIdxs[j]++;
            } else if ((i % 4 == 3) && topo.DeviceGroupUVA(i / 4, j) && topo.DeviceGroupCanP2P(i / 4, j)) { 
               std::cout << "      " << burstData[socketIdx * matrixWidth + i / 4][dataIdx + deviceIdxs[j]] << "\t|";
               deviceIdxs[j]++;
            } else { 
               std::cout << "\t-\t|";
            }

            dataIdx++;
            if (topo.DeviceGroupCanP2P(i / 4, j))
               dataIdx++;
            if (topo.DeviceGroupUVA(i / 4, j)) {
               dataIdx++;
               if (topo.DeviceGroupCanP2P(i / 4, j)) 
                  dataIdx++;
            }
         }
         
         std::cout << std::endl;
         
         if (i + 1 < matrixHeight && (i + 1 == ((float) matrixHeight / 2.0))) {
            std::cout << "|   Source\t|-----------------------";
            for (int i = 0; i < matrixWidth; i++)
               std::cout << "|---------------";
            std::cout << "|" << std::endl;
         } else if (i + 1 < matrixHeight && (i + 1) % 4  ==  0) {
            std::cout << "|\t\t|-----------------------";
            for (int i = 0; i < matrixWidth; i++)
               std::cout << "|---------------";
            std::cout << "|" << std::endl;
         }
      }
      std::cout << std::setprecision(4) << std::fixed;          
      
      std::cout << "-----------------------------------------"; 
      for (int i = 0; i < matrixWidth; i++)
         std::cout << "----------------";
      std::cout << std::endl;
   }
}

void PrintHDBurstMatrix(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData) {
   long long blockSize = params.burstBlockSize;
   int numSockets = params.nSockets;
   
   int numPatterns = 1;
   if (params.runPatternsHD)
      numPatterns = NUM_PATTERNS;
   
   int matrixWidth = topo.NumNodes();
   int matrixHeight = params.nDevices;
   
   std::cout << "\nHost/Device Unidirectional Memory Transfers:" << std::endl;
   std::cout << "Transfer Block Size: " << blockSize / BYTES_TO_MEGA << " (MB)"<< std::endl;
   std::cout << "Num Patterns: " << numPatterns << std::endl;

   std::cout << std::setprecision(2) << std::fixed;          
   for (int socketIdx = 0; socketIdx < numSockets; socketIdx++) {
      std::cout << "\nInitiating Socket: " << socketIdx << std::endl;
      
      for (int patternNum = 0; patternNum < numPatterns; patternNum++) {
         std::cout << "Memory Access Pattern: " <<  PatternNames[patternNum] << std::endl;   
   
         std::cout << "-------------------------"; 
         for (int i = 0; i < matrixWidth * 2; i++)
            std::cout << "----------------";
         std::cout << std::endl;

         std::cout << "|\t\t\t|"; 
         for (int i = 0; i < matrixWidth * 16 - 6; i++)
            std::cout << "-";

         std::cout << " Host CPU ";
         for (int i = 0; i < matrixWidth * 16 - 5; i++)
            std::cout << "-";
         std::cout << "|" << std::endl;

         std::cout << "|\t\t\t|";
         for (int i = 0; i < matrixWidth; i++)
            std::cout << "\t\t" << i << "\t\t|";
         std::cout << std::endl;

         std::cout << "|\t\t\t|"; 
         for (int i = 0; i < matrixWidth * 2; i++){
            if (i + 1 < matrixWidth * 2)
               std::cout << "----------------";
            else 
               std::cout << "---------------";
         }
         std::cout << "|" << std::endl;
       
         std::cout << "|\t\t\t";
         for (int i = 0; i < matrixWidth; i++)
            std::cout << "| Host-2-Device | Device-2-Host ";
         std::cout << "|" << std::endl;
           
         std::cout << "|\t       Transfer\t|";
         for (int i = 0; i < matrixWidth * 2; i++){
            if (i + 1 < matrixWidth * 2)
               std::cout << "----------------";
            else 
               std::cout << "---------------";
         }
         std::cout << "|" << std::endl;

         std::cout << "|\t\t  Type\t";
         for (int i = 0; i < matrixWidth * 2; i++) {
            std::cout << "| Page\t";
            std::cout << "|  Pin\t";
         }
         std::cout << "|" << std::endl;

         std::cout << "|-----------------------"; 
         for (int i = 0; i < matrixWidth * 2; i++)
            std::cout << "----------------";
         std::cout << "|" << std::endl;
         
         std::cout << std::setprecision(2) << std::fixed;          
         for (int i = 0; i < matrixHeight; ++i) {

            std::cout << "|\t\t|  " << i <<  "\t|";
            int rowIdx = socketIdx * numPatterns + patternNum;
            for (int j = 0; j < matrixWidth; ++j) {
                  int colIdx = j * topo.NumGPUs() * 4 + i * 4;
                  std::cout << " " << burstData[rowIdx][colIdx + 0] << "\t|";
                  std::cout << " " << burstData[rowIdx][colIdx + 2] << "\t|";
                  std::cout << " " << burstData[rowIdx][colIdx + 1] << "\t|";
                  std::cout << " " << burstData[rowIdx][colIdx + 3] << "\t|";
            }
            std::cout << std::endl;
            
            if (i + 1 < matrixHeight && (i + 1 == ((float) matrixHeight / 2.0))) {
               std::cout << "|     Device\t|-------";
               for (int i = 0; i < matrixWidth * 2; i++)
                  std::cout << "----------------";
               std::cout << "|" << std::endl;
            } else if (i + 1 < matrixHeight) {
               std::cout << "|\t\t|-------";
               for (int i = 0; i < matrixWidth * 2; i++)
                  std::cout << "----------------";
               std::cout << "|" << std::endl;
            }
         }
         std::cout << std::setprecision(4) << std::fixed;          

         std::cout << "-------------------------"; 
         for (int i = 0; i < matrixWidth * 2; i++)
            std::cout << "----------------";
         std::cout << std::endl;
      }
   }
}

void PrintHHBurstMatrix(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData) {
   long long blockSize = params.burstBlockSize;
   int numSockets = params.nSockets;

   int numPatterns = 1;
   if (params.runPatternsHH)
      numPatterns = NUM_PATTERNS;
   int nodeWidth = pow(HOST_MEM_TYPES * topo.NumNodes(), 2) / topo.NumNodes();

   int matrixWidth = HOST_MEM_TYPES * topo.NumNodes();
   int matrixHeight = HOST_MEM_TYPES * topo.NumNodes();
   
   std::cout << "\nHost-Host Multi-NUMA Unidirectional Memory Transfers:" << std::endl;
   std::cout << "Transfer Block Size: " << blockSize / BYTES_TO_MEGA << " (MB)"<< std::endl;
   std::cout << "Num Patterns: " << numPatterns << std::endl;

   std::cout << std::setprecision(2) << std::fixed;          
   for (int socketIdx = 0; socketIdx < numSockets; socketIdx++) {
      std::cout << "\nInitiating Socket: " << socketIdx << std::endl;
      
      for (int patternNum = 0; patternNum < numPatterns; patternNum++) {
         std::cout << "Memory Access Pattern: " <<  PatternNames[patternNum] << std::endl;   
         
         std::cout << "---------------------------------"; 
         for (int i = 0; i < matrixWidth; i++)
            std::cout << "----------------";
         std::cout << std::endl;

         std::cout << "|\t\t|----------------"; 
         for (int i = 0; i < matrixWidth * 8 - 7; i++)
            std::cout << "-";

         std::cout << " Destination ";
         for (int i = 0; i < matrixWidth * 8 - 7; i++)
            std::cout << "-";
         std::cout << "|" << std::endl;

         std::cout << "|   Transfer \t|---------------";// << std::endl;
         for (int i = 0; i < matrixWidth; i++)
            std::cout << "----------------";
         std::cout << "|" << std::endl;

         std::cout << "|   Point\t| NUMA \t\t|";
         for (int i = 0; i < topo.NumNodes(); i++)
            std::cout << "\t\t" << i << "\t\t|";
         std::cout << "" << std::endl;
         
         std::cout << "|\t\t| Node \t\t|";
         for (int i = 0; i < matrixWidth; i++) {
            if (i + 1 < matrixWidth)
               std::cout << "----------------";
            else 
               std::cout << "---------------";
         }
         std::cout << "|" << std::endl;
    
         std::cout << "|\t\t| #     Mem Type";
         for (int i = 0; i < matrixWidth; i++){
            if (i % 2)
               std::cout << "|    Pinned\t";
            else
               std::cout << "|    Pageable\t";
         }
         std::cout << "|"<< std::endl;
    
         std::cout << "|-------------------------------"; 
         for (int i = 0; i < matrixWidth; i++)
            std::cout << "----------------";
         std::cout << "|" << std::endl;
        
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
       
            int rowIdx = socketIdx * numPatterns + patternNum;
            for (int j = 0; j < matrixWidth; ++j) {
               int colIdx = (i / HOST_MEM_TYPES * nodeWidth) + j * HOST_MEM_TYPES + i % HOST_MEM_TYPES;
               std::cout << burstData[rowIdx][colIdx] << "\t|    ";
            }
                
            std::cout << "\n|\t\t|\t|";
            for (int j = 0; j < matrixWidth + 1; ++j)
               std::cout << "\t|\t";
            std::cout << std::endl;
            
            if (i + 1 < matrixHeight && (i + 1 != ((float) matrixHeight / 2.0))) {
               std::cout << "|\t\t|-------|-------|";
               for (int i = 0; i < matrixWidth; i++) {
                  if (i + 1 < matrixWidth)
                     std::cout << "----------------";
                  else 
                     std::cout << "---------------";
               }
               std::cout << "|" << std::endl; 
            } else if (i + 1 < matrixHeight) {
               std::cout << "|    Source     |-------|-------|";
               for (int i = 0; i < matrixWidth; i++) {
                  if (i + 1 < matrixWidth)
                     std::cout << "----------------";
                  else 
                     std::cout << "---------------";
               }
               std::cout << "|" << std::endl; 
            }
         }

         std::cout << "---------------------------------"; 
         for (int i = 0; i < matrixWidth; i++)
            std::cout << "----------------";
         std::cout << std::endl;
      }
      std::cout << std::setprecision(2) << std::fixed;          
   }
}

void PrintRangedHeader(BenchParams &params, SystemTopo &topo, std::ofstream &fileStream, BW_RANGED_TYPE testType) {

   std::vector<std::vector<int> > peerGroups;// = topo.GetPeerGroups();
   switch (testType) {
      case HH: 
         if (!params.runSocketTests) 
            fileStream << "0,";
         else
            fileStream << topo.NumSockets() << ",";

         fileStream << topo.NumNodes();
         if (params.usePinnedMem)
            fileStream << ",t";
         else 
           fileStream  << ",f";

         if (params.runSocketTests) 
            fileStream << ",t";
         else
            fileStream << ",f";

         fileStream << ",Repeated";
         if (params.runPatternsHD) {
            fileStream << ",Linear Inc";
            fileStream << ",Linear Dec";
         }
         fileStream << std::endl;
         break;
      case HD:
         if (!params.runSocketTests) 
            fileStream << "0,";
         else
            fileStream << topo.NumSockets() << ",";

         fileStream << topo.NumNodes() << ",";
         fileStream << params.nDevices;
         if (params.usePinnedMem)
            fileStream << ",t";
         else 
           fileStream  << ",f";

         if (params.runSocketTests) 
            fileStream << ",t";
         else
            fileStream << ",f";

         for (int i = 0; i < params.nDevices; i++) {
            fileStream << "," << topo.GetDeviceName(i);
         }

         fileStream << ",Repeated";
         if (params.runPatternsHD) {
            fileStream << ",Linear Inc";
            fileStream << ",Linear Dec";
         }

         fileStream << std::endl;
         break;
      case P2P:
         if (!params.runSocketTests) 
            fileStream << "0,";
         else
            fileStream << topo.NumSockets() << ",";

         fileStream << params.nDevices;
         fileStream << "," << topo.NumPeerGroups();
         
         if (params.runSocketTests) 
            fileStream << ",t";
         else
            fileStream << ",f";

         for (int i = 0; i < params.nDevices; i++) {
            fileStream << "," << topo.GetDeviceName(i);
         }
 
         for (int i = 0; i < params.nDevices; i++) {
            fileStream << "," << std::boolalpha << topo.DeviceUVA(i) << std::noboolalpha;
         }
         
         fileStream << std::endl;
         peerGroups = topo.GetPeerGroups();
         for (int i = 0; i < peerGroups.size(); i++) {
            for (int j = 0; j < peerGroups[i].size(); j++) {
               fileStream << peerGroups[i][j];
               if (j + 1 < peerGroups[i].size()) {
                  fileStream << ",";
               }
            }
            fileStream << std::endl;
         }
         break;
      default:
         std::cout << "Error: unrecognized ranged transfer test type!" << std::endl; 
         break;

   }
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

