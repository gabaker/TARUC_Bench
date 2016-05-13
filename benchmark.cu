
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

// SystemTopo class definition
#ifndef TIMER_CLASS_INC
#include "timer.h"
#define TIMER_CLASS_INC
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
void TestContention(BenchParams &params, SystemTopo &topo);

// Test Subfunctions
void RangeHDBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData); 
void RangeHHBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData); 
void RangeP2PBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData);
void BurstHDBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData); 
void BurstHHBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData); 
void BurstP2PBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData);  
void ContentionSubTestMemAccess(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps);
void ContentionSubTestQPI(BenchParams &params, SystemTopo &topo);
void ContentionSubTestPCIe(BenchParams &params, SystemTopo &topo);
void ContentionSubTestP2P(BenchParams &params, SystemTopo &topo);

// Support functions
void MemCopyRun(SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData, MEM_OP copyType, MEM_PATTERN patternType, int destIdx, int srcIdx, int numCopies); 
float TimedMemManageOp(void **MemBlk, long long NumBytes, MEM_OP TimedOp); 
float TimedMemCopyStep(void * destPtr, void *srcPtr, long long stepSize, long long blockSize, int numCopiesPerStep, MEM_OP copyType, MEM_PATTERN patternType, int destIdx = 0, int srcIdx = 0);
float BurstMemCopy(SystemTopo &topo, long long blockSize, MEM_OP copyType, int destIdx, int srcIdx, int numSteps, MEM_PATTERN pattern = REPEATED); 
void MemCopyOp(void * destPtr, void *srcPtr, long long stepSize, MEM_OP copyType, int destIdx = 0, int srcIdx = 0, cudaStream_t stream = 0);
void AllocMemBlocks(SystemTopo &topo, void **destPtr, void **srcPtr, long long numBytes, MEM_OP copyType, int destIdx = 0, int srcIdx = 0);
void AllocMemBlock(SystemTopo &topo, void **blkPtr, long long numBytes, MEM_TYPE blockType, int srcIdx, int extIdx = 0);
void FreeMemBlocks(SystemTopo &topo, void* destPtr, void *srcPtr, long long numBytes, MEM_OP copyType, int destIdx = 0, int srcIdx = 0);
void SetMemBlocks(SystemTopo &topo, void *destPtr, void *srcPtr, long long numBytes, MEM_OP copyType, int destIdx, int srcIdx, int value); 
void SetMemBlock(SystemTopo &topo, void *blkPtr, long long numBytes, long long value, MEM_TYPE memType, int devIdx = 0);
int CalcRunSteps(std::vector<long long> &blockSteps, long long startStep, long long stopStep, long long numSteps);

// Results output
void PrintRangedHeader(BenchParams &params, SystemTopo &topo, std::ofstream &fileStream, BW_RANGED_TYPE testType); 
void PrintResults(std::ofstream &outFile, std::vector<long long> &steps, std::vector<std::vector<float> > &results);
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

   benchParams.nSockets = 1;
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
   if (params.runBandwidthTestHH && params.runBurstTests)
      HHBurstTransferTest(params, topo);
   
   if (params.runBandwidthTestHD && params.runBurstTests)
      HDBurstTransferTest(params, topo);

   if (params.runBandwidthTestP2P && params.runBurstTests && params.runAllDevices)
      P2PBurstTransferTest(params, topo);

   //Ranged bandwidth tests (cvs files printed)
   if (params.runBandwidthTestHH && params.runRangeTests)
      HHRangeTransferTest(params, topo);
   
   if (params.runBandwidthTestHD && params.runRangeTests)
      HDRangeTransferTest(params, topo);

   if (params.runBandwidthTestP2P && params.runRangeTests && params.runAllDevices)
      P2PRangeTransferTest(params, topo);

   // Contention benchmark tests
   if (params.runContentionTest)
      TestContention(params, topo);
}

void TestMemoryOverhead(BenchParams &params, SystemTopo &topo) {
   std::cout << "\nRunning Ranged Memory Overhead Test...\n" << std::endl;
   
   char *deviceMem = NULL;//, * managedMem = NULL, * mappedMem = NULL; 
   char *hostPageMem = NULL, *hostPinnedMem = NULL, * hostCombinedMem = NULL;
   std::vector<long long> steps;
   std::vector<std::vector<float> > overheadData;
   int testNum = 0;
  
   CalcRunSteps(steps, params.rangeMemOverhead[0], params.rangeMemOverhead[1], params.numRangeSteps);  
   overheadData.resize(steps.size());
   
   // Memory overhead test will run for each device utilizing the cudaMalloc and cudaFree functions
   // on the first iteration of the look, assuming there is atleast one device, the host will run the 
   // pinned and un-pinned memory tests
   for (int socketIdx = 0; socketIdx < params.nSockets; socketIdx++) {
      topo.PinSocket(socketIdx);
 
      for (int numaIdx = 0; numaIdx < topo.NumNodes(); numaIdx++) { 
         topo.PinNumaNode(numaIdx);
        
         std::cout << "Test " << testNum++ << " Host Alloc/Free, Pinned/Pageable/Write-Combined\t" << "NUMA node: " << numaIdx << " CPU " << socketIdx << std::endl;            
         // Host based management for CASE 1 & 2
         for (long stepIdx = 0; stepIdx < steps.size(); stepIdx++) {
            long long chunkSize = steps[stepIdx];
            
            float hostAllocTime = 0, pinAllocTime = 0, combAllocTime = 0;//, managedAllocTime = 0, mappedAllocTime = 0;
            float hostFreeTime = 0, pinFreeTime = 0, combFreeTime = 0;//, managedFreeTime = 0, mappedFreeTime = 0; 
            // repeat same block run and average times
            for (int reIdx = 0; reIdx < params.numStepRepeats; reIdx++) {
               hostFreeTime += TimedMemManageOp((void **) &hostPageMem, chunkSize, HOST_MALLOC);
               SetMemBlock(topo, hostPageMem, chunkSize, 0, PAGE);
               hostAllocTime += TimedMemManageOp((void **) &hostPageMem, chunkSize, HOST_FREE);

               if (params.testAllMemTypes) {
                  pinAllocTime += TimedMemManageOp((void **) &hostPinnedMem, chunkSize, HOST_PINNED_MALLOC);
                  SetMemBlock(topo, hostPinnedMem, chunkSize, 0, PINNED);
                  pinFreeTime += TimedMemManageOp((void **) &hostPinnedMem, chunkSize, HOST_PINNED_FREE); 
               
                  combAllocTime += TimedMemManageOp((void **) &hostCombinedMem, chunkSize, HOST_COMBINED_MALLOC);
                  SetMemBlock(topo, hostCombinedMem, chunkSize, 0, WRITE_COMBINED);
                  combFreeTime += TimedMemManageOp((void **) &hostCombinedMem, chunkSize, HOST_COMBINED_FREE);

                  /*managedAllocTime += TimedMemManageOp((void **) &managedMem, chunkSize, MANAGED_MALLOC);
                  SetMemBlock(topo, managedMem, chunkSize, 0, MANAGED);
                  managedFreeTime += TimedMemManageOp((void **) &managedMem, chunkSize, MANAGED_FREE);

                  mappedAllocTime += TimedMemManageOp((void **) &mappedMem, chunkSize, MAPPED_MALLOC);
                  SetMemBlock(topo, mappedMem, chunkSize, 0, MAPPED);
                  mappedFreeTime += TimedMemManageOp((void **) &mappedMem, chunkSize, MAPPED_FREE); */
               }
            }
            hostAllocTime /= (float) params.numStepRepeats;
            hostFreeTime /= (float) params.numStepRepeats;
            if (hostAllocTime < 0.2)
               hostAllocTime = 0.2;
            if (hostFreeTime < 0.2)            
               hostFreeTime = 0.2;

            overheadData[stepIdx].push_back(hostAllocTime);
            overheadData[stepIdx].push_back(hostFreeTime);

            overheadData[stepIdx].push_back(pinAllocTime / (float) params.numStepRepeats);
            overheadData[stepIdx].push_back(pinFreeTime / (float) params.numStepRepeats);

            overheadData[stepIdx].push_back(combAllocTime / (float) params.numStepRepeats);
            overheadData[stepIdx].push_back(combFreeTime / (float) params.numStepRepeats);

            //overheadData[stepIdx].push_back(managedAllocTime / (float) params.numStepRepeats);
            //overheadData[stepIdx].push_back(managedFreeTime / (float) params.numStepRepeats);
            
            //overheadData[stepIdx].push_back(mappedFreeTime / (float) params.numStepRepeats);
            //overheadData[stepIdx].push_back(mappedFreeTime / (float) params.numStepRepeats);

         }   
      }

      topo.PinSocket(socketIdx);
      topo.PinNumaNode(socketIdx);
      // Device based memory management for CASE 3 & 4
      for (int currDev = 0; currDev < params.nDevices; currDev++) {
         topo.SetActiveDevice(currDev);
         std::cout << "Test " << testNum++ << " Device Alloc/Free \t\t\t\t" << "CPU " << socketIdx << " Dev:" << currDev << std::endl;            
         
         for (long stepIdx = 0; stepIdx < steps.size(); stepIdx++) {
            long long chunkSize = steps[stepIdx];
            float devAllocTime = 0, devFreeTime = 0;

            // repeat same block run and average times
            for (int reIdx = 0; reIdx < params.numStepRepeats; reIdx++) {
               // CASE 3: Allocation of device memory  
               devAllocTime += TimedMemManageOp((void **) &deviceMem, chunkSize, DEVICE_MALLOC);
               SetMemBlock(topo, deviceMem, chunkSize, 0, DEVICE, currDev);
               // CASE 4: DeAllocation of device memory 
               devFreeTime += TimedMemManageOp((void **) &deviceMem, chunkSize, DEVICE_FREE);
            }

            overheadData[stepIdx].push_back(devAllocTime / (float) params.numStepRepeats);
            overheadData[stepIdx].push_back(devFreeTime / (float) params.numStepRepeats);
         }
      }
   }
   
   std::string dataFileName = "./results/overhead/" + params.runTag + "_overhead.csv";
   std::ofstream overheadResultsFile(dataFileName.c_str());
   overheadResultsFile << params.nSockets << ",";
   overheadResultsFile << topo.NumNodes() << ",";
   overheadResultsFile << params.nDevices;
   if (params.testAllMemTypes)
      overheadResultsFile << ",t";
   else 
      overheadResultsFile << ",f";

   for (int i = 0; i < params.nDevices; i++)
      overheadResultsFile << "," << topo.GetDeviceName(i);
   overheadResultsFile << std::endl;
   PrintResults(overheadResultsFile, steps, overheadData);

   std::cout << "\nMemory Overhead Test Complete!" << std::endl;
}

void HHRangeTransferTest(BenchParams &params, SystemTopo &topo) {
   std::cout << "\nRunning Ranged Host-Host Bandwidth Tests...\n" << std::endl;
   
   std::vector<std::vector<float> > rangeData;
   std::vector<long long> steps;
    
   CalcRunSteps(steps, params.rangeHostHostBW[0], params.rangeHostHostBW[1], params.numRangeSteps); 
   rangeData.resize(steps.size());
   
   RangeHHBandwidthRun(params, topo, steps, rangeData);

   // tt == Transfer Time
   std::string dataFileName = "./results/bandwidth/" + params.runTag + "_ranged_hh_tt.csv";
   std::ofstream ttResultsFileHH(dataFileName.c_str());
   PrintRangedHeader(params, topo, ttResultsFileHH, HH); 
   PrintResults(ttResultsFileHH, steps, rangeData);

   // Output throughput (GB/S) and block size
   for (int blkIdx = 0; blkIdx < steps.size(); ++blkIdx) {
      for (int runIdx = 0; runIdx < rangeData[blkIdx].size(); ++runIdx) {
         rangeData[blkIdx][runIdx] = ((double) steps[blkIdx]) / rangeData[blkIdx][runIdx] * 1.0e6;
         rangeData[blkIdx][runIdx] /= pow(2.0, 30.0);
      }
   }

   dataFileName = "./results/bandwidth/" + params.runTag + "_ranged_hh_bw.csv";
   std::ofstream bwResultsFileHH(dataFileName.c_str());
   PrintRangedHeader(params, topo, bwResultsFileHH, HH); 
   PrintResults(bwResultsFileHH, steps, rangeData);

   std::cout << "\nRanged Host-Host Bandwidth Tests complete!" << std::endl;
}

void HDRangeTransferTest(BenchParams &params, SystemTopo &topo) {
   std::cout << "\nRunning Ranged Host-Device Bandwidth Tests...\n" << std::endl;
   
   std::vector<std::vector<float> > rangeData;
   std::vector<long long> steps;

   CalcRunSteps(steps, params.rangeHostDeviceBW[0], params.rangeHostDeviceBW[1], params.numRangeSteps); 
   rangeData.resize(steps.size());
   
   RangeHDBandwidthRun(params, topo, steps, rangeData);
   
   // tt == Transfer Time
   std::string dataFileName = "./results/bandwidth/" + params.runTag + "_ranged_hd_tt.csv";
   std::ofstream ttResultsFileHD(dataFileName.c_str());
   PrintRangedHeader(params, topo, ttResultsFileHD, HD); 
   PrintResults(ttResultsFileHD, steps, rangeData);

   // Output throughput (GB/S) and block size
   for (int blkIdx = 0; blkIdx < steps.size(); ++blkIdx) {
      for (int runIdx = 0; runIdx < rangeData[blkIdx].size(); ++runIdx) {
         rangeData[blkIdx][runIdx] = ((double) steps[blkIdx]) / rangeData[blkIdx][runIdx] * 1.0e6;
         rangeData[blkIdx][runIdx] /= pow(2.0, 30.0);
      }
   }

   dataFileName = "./results/bandwidth/" + params.runTag + "_ranged_hd_bw.csv";
   std::ofstream bwResultsFileHD(dataFileName.c_str());
   PrintRangedHeader(params, topo, bwResultsFileHD, HD); 
   PrintResults(bwResultsFileHD, steps, rangeData);

   std::cout << "\nHost-Device Ranged Bandwidth Tests complete!" << std::endl;
}

void P2PRangeTransferTest(BenchParams &params, SystemTopo &topo){
   std::cout << "\nRunning P2P Device Ranged Bandwidth test..." << std::endl;

   std::vector<std::vector<float> > rangeData;
   std::vector<long long> steps;

   CalcRunSteps(steps, params.rangeDeviceBW[0], params.rangeDeviceBW[1], params.numRangeSteps); 
   rangeData.resize(steps.size());
   
   RangeP2PBandwidthRun(params, topo, steps, rangeData);

   // tt == Transfer Time
   std::string dataFileName = "./results/bandwidth/" + params.runTag + "_ranged_p2p_tt.csv";
   std::ofstream ttResultsFileP2P(dataFileName.c_str());
   PrintRangedHeader(params, topo, ttResultsFileP2P, P2P); 
   PrintResults(ttResultsFileP2P, steps, rangeData);

   // Output throughput (GB/S) and block size
   for (int blkIdx = 0; blkIdx < steps.size(); ++blkIdx) {
      for (int runIdx = 0; runIdx < rangeData[blkIdx].size(); ++runIdx) {
         rangeData[blkIdx][runIdx] = ((double) steps[blkIdx]) / rangeData[blkIdx][runIdx] * 1.0e6;
         rangeData[blkIdx][runIdx] /= pow(2.0, 30.0);
      }
   }

   dataFileName = "./results/bandwidth/" + params.runTag + "_ranged_p2p_bw.csv";
   std::ofstream bwResultsFileP2P(dataFileName.c_str());
   PrintRangedHeader(params, topo, bwResultsFileP2P, P2P); 
   PrintResults(bwResultsFileP2P, steps, rangeData);

   std::cout << "\nP2P Device Ranged Bandwidth Test Complete!" << std::endl;
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
            std::cout << "Test " << testNum++ << " HtoH, Pageable Memory\t\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
            MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COPY, REPEATED, destIdx, srcIdx, numRepeats); 
            if (params.runPatternsHD) {
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COPY, LINEAR_INC, destIdx, srcIdx, numRepeats); 
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COPY, LINEAR_DEC, destIdx, srcIdx, numRepeats);
            }

            if (params.testAllMemTypes) {
               // HtoH Ranged Transfer - Pinned Memory Src Host
               std::cout << "Test " << testNum++ << " HtoH, Pinned Memory Src  \t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_PINNED_HOST_COPY, REPEATED, destIdx, srcIdx, numRepeats);
               if (params.runPatternsHD){ 
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_PINNED_HOST_COPY, LINEAR_INC, destIdx, srcIdx, numRepeats); 
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_PINNED_HOST_COPY, LINEAR_DEC, destIdx, srcIdx, numRepeats); 
               }

               // HtoH Ranged Transfer - Pinned Memory Dest Host
               std::cout << "Test " << testNum++ << " HtoH, Pinned Memory Dest \t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_PINNED_COPY, REPEATED, destIdx, srcIdx, numRepeats); 
               if (params.runPatternsHD) {
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_PINNED_COPY, LINEAR_INC, destIdx, srcIdx, numRepeats); 
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_PINNED_COPY, LINEAR_DEC, destIdx, srcIdx, numRepeats); 
               }

              // HtoH Ranged Transfer - Pinned Memory Both Hosts
               std::cout << "Test " << testNum++ << " HtoH, Both Pinned Memory \t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COPY_PINNED, REPEATED, destIdx, srcIdx, numRepeats); 
               if (params.runPatternsHD) {
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COPY_PINNED, LINEAR_INC, destIdx, srcIdx, numRepeats); 
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COPY_PINNED, LINEAR_DEC, destIdx, srcIdx, numRepeats);
               } 

               // HtoH Ranged Transfer - Write-Combined Memory Src Host
               /*std::cout << "Test " << testNum++ << " HtoH, Write-Combined Memory Src    \tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_COMBINED_HOST_COPY, REPEATED, destIdx, srcIdx, numRepeats);
               if (params.runPatternsHD){ 
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_COMBINED_HOST_COPY, LINEAR_INC, destIdx, srcIdx, numRepeats); 
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_COMBINED_HOST_COPY, LINEAR_DEC, destIdx, srcIdx, numRepeats); 
               }*/

               // HtoH Ranged Transfer - Write-Combined Memory Dest Host
               std::cout << "Test " << testNum++ << " HtoH, Write-Combined Memory Dest  \tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COMBINED_COPY, REPEATED, destIdx, srcIdx, numRepeats); 
               if (params.runPatternsHD) {
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COMBINED_COPY, LINEAR_INC, destIdx, srcIdx, numRepeats); 
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COMBINED_COPY, LINEAR_DEC, destIdx, srcIdx, numRepeats); 
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
 
      for (int hostIdx = 0; hostIdx < topo.NumNodes(); hostIdx++) { 
         topo.PinNumaNode(hostIdx);
         //Host-Device PCIe Memory Transfers
         for (int devIdx = 0; devIdx < params.nDevices; devIdx++) {
            topo.SetActiveDevice(devIdx);
            // HtoD Ranged Transfer - Pageable Memory
            std::cout << "Test " << testNum++ << " HtoD, Pageable Memory\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << hostIdx << "\tDev Dev: " << devIdx << std::endl;
            MemCopyRun(topo, blockSteps, bandwidthData, HOST_DEVICE_COPY, REPEATED, devIdx, hostIdx, numRepeats); 
            if (params.runPatternsHD) {
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_DEVICE_COPY, LINEAR_INC, devIdx, hostIdx, numRepeats); 
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_DEVICE_COPY, LINEAR_DEC, devIdx, hostIdx, numRepeats); 
            }

            //blockSteps, bandwidthData, copyType, patternType, int devIdx, int srcIdx, int numCopies); 
            // DtoH Ranged Transfer - Pageable Memory
            std::cout << "Test " << testNum++ << " DtoH, Pageable Memory\t\tCPU: " << socketIdx << "\t\tDev Src: " << devIdx << "\tNUMA Dest: " << hostIdx << std::endl;
            MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_COPY, REPEATED, hostIdx, devIdx, numRepeats); 
            if (params.runPatternsHD) {
               MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_COPY, LINEAR_INC, hostIdx, devIdx, numRepeats); 
               MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_COPY, LINEAR_DEC, hostIdx, devIdx, numRepeats); 
            }
            
            if (params.testAllMemTypes) {
               // HtoD Ranged Transfer - Pinned Memory
               std::cout << "Test " << testNum++ << " HtoD, Pinned Memory\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << hostIdx << "\tDest Dev: " << devIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_PINNED_DEVICE_COPY, REPEATED, devIdx, hostIdx, numRepeats); 
               if (params.runPatternsHD) {
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_PINNED_DEVICE_COPY, LINEAR_INC, devIdx, hostIdx, numRepeats); 
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_PINNED_DEVICE_COPY, LINEAR_DEC, devIdx, hostIdx, numRepeats); 
               } 

               // DtoH Ranged Transfer - Pinned Memory
               std::cout << "Test " << testNum++ << " DtoH, Pinned Memory\t\tCPU: " << socketIdx << "\t\tSrc Dev: " << hostIdx << "\tNUMA Dest: " << hostIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_PINNED_COPY, REPEATED, hostIdx, devIdx, numRepeats); 
               if (params.runPatternsHD) {
                  MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_PINNED_COPY, LINEAR_INC, hostIdx, devIdx, numRepeats); 
                  MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_PINNED_COPY, LINEAR_DEC, hostIdx, devIdx, numRepeats);
               } 

               // HtoD Ranged Transfer - Write-Combined Memory
               std::cout << "Test " << testNum++ << " HtoD, Write-Combined Memory \tCPU: " << socketIdx << "\t\tNUMA Src: " << hostIdx << "\tDest Dev: " << devIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_COMBINED_DEVICE_COPY, REPEATED, devIdx, hostIdx, numRepeats); 
               if (params.runPatternsHD) {
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_COMBINED_DEVICE_COPY, LINEAR_INC, devIdx, hostIdx, numRepeats); 
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_COMBINED_DEVICE_COPY, LINEAR_DEC, devIdx, hostIdx, numRepeats); 
               } 

               // DtoH Ranged Transfer - Write-Combined Memory
               std::cout << "Test " << testNum++ << " DtoH, Write-Combined Memory \tCPU: " << socketIdx << "\t\tSrc Dev: " << hostIdx << "\tNUMA Dest: " << hostIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_COMBINED_COPY, REPEATED, hostIdx, devIdx, numRepeats); 
               if (params.runPatternsHD) {
                  MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_COMBINED_COPY, LINEAR_INC, hostIdx, devIdx, numRepeats); 
                  MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_COMBINED_COPY, LINEAR_DEC, hostIdx, devIdx, numRepeats);
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
            
            /*if (topo.DeviceGroupUVA(srcIdx, destIdx)) {  
               // DtoD Ranged Transfer - No Peer, UVA
               std::cout << "Test " << testNum++ << " Device-To-Device, No Peer, UVA\t\tCPU: " << socketIdx << "\tSrc Device: " << srcIdx << "\tDest Device: " << destIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, COPY_UVA, REPEATED, destIdx, srcIdx, numRepeats); 
 
               // DtoD Ranged Transfer - Peer, UVA
               if (topo.DeviceGroupCanP2P(srcIdx, destIdx)) {
                  std::cout << "Test " << testNum++ << " Device-To-Device, Peer Enabled, UVA\tCPU: " << socketIdx << "\tSrc Device: " << srcIdx << "\tDest Device: " << destIdx << std::endl;
                  topo.DeviceGroupSetP2P(srcIdx, destIdx, true);
                  MemCopyRun(topo, blockSteps, bandwidthData, COPY_UVA, REPEATED, destIdx, srcIdx, numRepeats); 
                  topo.DeviceGroupSetP2P(srcIdx, destIdx, false);
               }
            }*/
         }
      }
   }
}

void TestContention(BenchParams &params, SystemTopo &topo) {
   std::cout << "Running Contention tests..." << std::endl;

   // No parameters for this test, set default here
   // TODO: migrate relevent parameters to param file input

   if (params.testAllMemTypes) 
      params.numContMemTypes = 2;
   else
      params.numContMemTypes = 1;
  
   std::vector<long long> blockSteps;
   if (params.testContRange)
      CalcRunSteps(blockSteps, params.rangeCont[0], params.rangeCont[1], params.numContRepeats);
   else
      CalcRunSteps(blockSteps, params.rangeCont[1], params.rangeCont[1], params.numContRepeats);
 
   /* Memory Access: Single Socket, Single Node
    *
    * Host-Host single node memory access
    * Inherently bidirectional transfer (actual bandwidth is double)
    * since destination is same as source.
    */

   ContentionSubTestMemAccess(params, topo, blockSteps);

   /* QPI Bus Test (Multiple Sockets)
    *
    * Host-to-Host: bidirectional and unidirectional
    * Pin multiple cores on a single 
    */
   if (topo.NumSockets() >= 2)
      ContentionSubTestQPI(params, topo);
   else
      std::cout << "One Socket Detected: No inter-CPU communication bus to test!" << std::endl;

   /* PCIe (Single and Multiple Sockets)
    * 
    * Host-to-Device & Device-to-Host: bidirectional and unidirectional
    * Single socket (avoid QPI effects) to each combination of GPUs
    */
   //ContentionSubTestPCIe(params, topo);

   /* P2P
    * 
    * Host-Host Transfers: bidirectional and unidirectionsal 
    * Every combination up to one per transfer
    * Multiple from one to all devices (if more than one)
    */
   //ContentionSubTestP2P(params, topo);

   std::cout << "Contention tests complete!" << std::endl;
}

void TestMemoryUsage(BenchParams &params, SystemTopo &topo) {
   std::cout << "\nRunning memory usage pattern tests..." << std::endl;

   std::cout << "Test not yet implemented!" << std::endl;

   std::cout << "\nMemory usage patterns tests complete!" << std::endl;
}

void ContentionSubTestMemAccess(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps) {
   //int numThreads = 1;
   //int maxThreads = topo.NumPUsPerCore() * topo.NumCoresPerSocket();//topo.NumCores();//topo.NumCoresPerSocket();
/*   int PUsPerSocket = topo.NumCoresPerSocket() * topo.NumPUsPerCore();
   //float threadBW[topo.NumPUs()];
   float aggBW = 0;
   MEM_TYPE memType;
   //MEM_OP copyType = HOST_PINNED_HOST_COPY;
   float conv = 1.0e-6; 
   int NumOps = 2; 
   std::vector<std::vector<float> > data;
   data.resize(blockSteps.size());
   std::cout << PUsPerSocket << std::endl;
   std::cout << "Main Memory Contention Test" << std::endl;

   long long blockSize = blockSteps[blockSteps.size() - 1] / sizeof(double);
   //static double srcBlk[blockSize];
   //static double destBlk[blockSize];

   for (int socketCount = 0; socketCount < topo.NumSockets(); socketCount++) {
      for (int memIdx = 0; memIdx < params.numContMemTypes; memIdx++) {
         for (int opIdx = 0; opIdx < NumOps; opIdx++) {
            if (memIdx == 0)
               memType = PAGE;
            else
               memType = PINNED;

            std::cout << "- " << socketCount << " - " << memIdx << " - " << opIdx << std::endl;
            int maxThreads = (socketCount + 1) * topo.NumCoresPerSocket() * topo.NumPUsPerCore();
            int numThreads = 1;
            do {
               omp_set_num_threads(numThreads);

               #pragma omp parallel
               {
                  // Get local thread ID
                  int threadIdx = omp_get_thread_num();
                  double * srcBlk, * destBlk;
                  // Set the memory type and timers as indicated by memIdx

                  // pin threads to execution space (socket)
                  // TODO: Check to see pinning per core works then change this
                  //topo.PinPUBySocket(threadIdx / PUsPerSocket, threadIdx % PUsPerSocket);
                  topo.PinSocket(threadIdx / PUsPerSocket);
                  //topo.PinPU(threadIdx * (socketIdx + 1));
                  //todo.PinCoreBySocket();
    
                  // allocate src and dest blocks to NUMA nodes
                  AllocMemBlock(topo, (void **) &srcBlk, blockSteps[blockSteps.size() - 1], memType, threadIdx / PUsPerSocket);
                  AllocMemBlock(topo, (void **) &destBlk, blockSteps[blockSteps.size() - 1], memType, threadIdx / PUsPerSocket);
                  SetMemBlock(topo, srcBlk, blockSize, 1, memType);
                  SetMemBlock(topo, destBlk, blockSize, 0, memType);
                  

                  // Run ranged test for each thread, sync between steps
                  for (int stepIdx = 0; stepIdx < blockSteps.size(); stepIdx++) {
                     double totalTime = 0;
                     #pragma omp barrier

                     #pragma omp master
                     {
                     static Timer threadTimer(true);//useHostTimer);
                     threadTimer.StartTimer();
                     }
                     #pragma omp for
                     for (int repCount = 0; repCount < params.numContRepeats; repCount++) {
                        
                        #pragma omp for 
                        for (register long long i = 0; i < blockSize; ++i)
                           destBlk[i] = srcBlk[i];   
                        
                        //threadTimer.StopTimer();                     
                        //totalTime += (double) threadTimer.ElapsedTime();
                     }
                     
                     #pragma omp barrier*/
                     // initiate transfers on each thread simultaneously 
       
                     /*for (int repCount = 0; repCount < params.numContRepeats; repCount++) {
                        if (opIdx == 0)
                           MemCopyOp(destBlk, srcBlk, blockSteps[stepIdx], copyType);
                        else
                           SetMemBlock(topo, srcBlk, blockSteps[blockSteps.size() - 1], repCount % 2, memType);
                           //SetMemBlock(topo, destBlk, blockSteps[blockSteps.size() - 1], repCount % 2, memType);
                     }*/
                     /*#pragma omp master
                     {
                     totalTime = totalTime / (double) params.numContRepeats;
                     long long totalBytes = blockSteps[blockSteps.size() - 1];
                     double bandwidth = ((double) totalBytes / (double) pow(2.0, 30.0)) / (totalTime * conv);
                     //threadBW[threadIdx] = bandwidth; 

                     // sum aggragite bandwidths
                     //#pragma omp atomic

                     //#pragma omp barrier

                     //#pragma omp single
                     //{
                        data[0].push_back(aggBW);
                        aggBW = bandwidth;
                        //for (int i = 0; i < omp_get_num_threads(); ++i) {
                        //   aggBW += threadBW[i];
                        //   data[stepIdx].push_back(threadBW[i]);
                        //}
                        std::cout << numThreads << "|" << blockSize << ": " << aggBW << std::endl;
                        aggBW = 0;
                     }
                  }
                  if (memType == PAGE) {
                     topo.FreeHostMem(srcBlk, params.rangeCont[1]);
                     topo.FreeHostMem(destBlk, params.rangeCont[1]);
                  } else {
                     topo.FreePinMem(srcBlk, params.rangeCont[1]);
                     topo.FreePinMem(destBlk, params.rangeCont[1]);
                  }
               }

               if (numThreads == 1)
                  numThreads++;
               else 
                  numThreads *= 2;

            } while (numThreads <= maxThreads);
         }
      }
   }

   // Output results
   // Header: sockets, memtypes, max thread count, test range
   std::string dataFileName = "./results/contention/" + params.runTag + "_Contention_host_mem_.csv";
   std::ofstream resultsFile(dataFileName.c_str());
   resultsFile << topo.NumSockets() << ","; 
   resultsFile << params.numContMemTypes << ","; 
   resultsFile << topo.NumCoresPerSocket() * topo.NumPUsPerCore() << ","; 
   if (params.testContRange) 
      resultsFile << "t" << std::endl;
   else  
      resultsFile << "f" << std::endl;
   PrintResults(resultsFile, blockSteps, data);
   */
}

void ContentionSubTestQPI(BenchParams &params, SystemTopo &topo) {
   int NumDirs = 2; // Copy Directions: 0->1 unidirectional, bidirectional
   MEM_TYPE memType;
   MEM_OP copyType = HOST_PINNED_HOST_COPY;

   float aggBW = 0;
   float conv = 1.0e-6; 
   long long blockSize = params.rangeCont[1];
   float threadBW[topo.NumPUs()];
   std::vector<long long> blockSteps;
   blockSteps.push_back(blockSize); 
   std::vector<std::vector<float> > data;
   data.resize(1);

   std::cout << "Socket-Socket Communication Contention" << std::endl;
   for (int copyDir = 0; copyDir < NumDirs; copyDir++) {
      for (int memIdx = 0; memIdx < params.numContMemTypes; memIdx++) {
         std::cout << "- " << copyDir << " - " << memIdx << " - " << std::endl;

         if (memIdx == 0)
            memType = PAGE;
         else
            memType = PINNED;
          
         int numThreads = 1;
         int MaxThreads = topo.NumPUs();
         if (copyDir == 0)
            MaxThreads /= topo.NumSockets();
         
         do {
            omp_set_num_threads(numThreads);
            #pragma omp parallel
            {
               // Get local thread ID
               int threadIdx = omp_get_thread_num();
               void * srcBlk, * destBlk;
               int srcNode = 0, destNode = 0, coreIdx = 0;
               Timer threadTimer(true);

               // Set the memory type and timers as indicated by memIdx
               coreIdx = threadIdx % topo.NumCoresPerSocket();
               if (copyDir == 0) { // unidirectional, only testing one direction; should be equivalent 
                  srcNode = 0;
                  destNode = 1;
               } else if (copyDir == 1) { // bidirectional
                  srcNode = (threadIdx / topo.NumPUsPerSocket()) % 2;
                  destNode = (threadIdx / topo.NumPUsPerSocket() + 1) % 2;
               } 
               
               // allocate memory and pin threads to execution space
               // TODO: Check to see pinning per core works then change this
               //topo.PinCore(coreIdx); //
               topo.PinCoreBySocket(srcNode, coreIdx);
               //AllocMemBlock(topo, &srcBlk, blockSize, memType, srcNode);
               //AllocMemBlock(topo, &destBlk, blockSize, memType, destNode);
               //SetMemBlock(topo, srcBlk, blockSize, 0x0, memType);
               //SetMemBlock(topo, destBlk, blockSize, 0x0, memType);
  
               // initiate transfers on each thread simultaneously 
               #pragma omp barrier
               threadTimer.StartTimer();
               
               for (register int repCount = 0; repCount < params.numContRepeats; repCount++) 
                  MemCopyOp(destBlk, srcBlk, blockSize, copyType);
               
               //#pragma omp barrier
               threadTimer.StopTimer();     
                  
               // calculate thread local bandwidth
               double time = (double) threadTimer.ElapsedTime() / (double) params.numContRepeats;
               double bandwidth = ((double) blockSize / (double) pow(2.0, 30.0)) / (time * conv);
              
               threadBW[threadIdx] = bandwidth; 

               // sum aggragite bandwidths
               #pragma omp atomic
               aggBW += bandwidth;

               #pragma omp barrier

               #pragma omp single
               {
                  for (int i = 0; i < omp_get_num_threads(); ++i)
                     data[0].push_back(threadBW[i]);
                  data[0].push_back(aggBW);
                  
                  std::cout << numThreads << ": " << aggBW << std::endl;
                  aggBW = 0;
               }
            
               if (memType == PAGE) {
                  topo.FreeHostMem(srcBlk, blockSize);
                  topo.FreeHostMem(destBlk, blockSize);
               } else {
                  topo.FreePinMem(srcBlk, blockSize);
                  topo.FreePinMem(destBlk, blockSize);
               }

            }
            if (numThreads == 1)
               numThreads++;
            else 
               //numThreads*=2;
               numThreads+=2;
         } while (numThreads <= MaxThreads);
      }
   }

   // Output results
   std::string dataFileName = "./results/contention/" + params.runTag + "_contention_inter_socket_.csv";
   std::ofstream resultsFile(dataFileName.c_str());

   //TODO Confirm these are correct header values
   resultsFile << topo.NumSockets() << ","; 
   resultsFile << params.numContMemTypes << ","; 
   resultsFile << topo.NumPUsPerCore() * topo.NumCoresPerSocket() << ","; 
   if (params.testContRange) 
      resultsFile << "t" << std::endl;
   else  
      resultsFile << "f" << std::endl;
   PrintResults(resultsFile, blockSteps, data);

}

void ContentionSubTestPCIe(BenchParams &params, SystemTopo &topo) {
   int maxThreads = 1;   

   // Output results
   std::string dataFileName = "./results/contention/" + params.runTag + "_contention_pcie_.csv";
   std::ofstream resultsFile(dataFileName.c_str());

   //TODO Confirm these are correct header values
   resultsFile << topo.NumSockets() << ","; 
   resultsFile << params.numContMemTypes << ","; 
   resultsFile << maxThreads << ","; 
   if (params.testContRange) 
      resultsFile << "t" << std::endl;
   else  
      resultsFile << "f" << std::endl;
   //PrintResults(resultsFile, blockSteps, data);
}

void ContentionSubTestP2P(BenchParams &params, SystemTopo &topo) {
   int maxThreads = 1;

   std::string dataFileName = "./results/contention/" + params.runTag + "_contention_p2p_.csv";
   std::ofstream resultsFile(dataFileName.c_str());

   //TODO Confirm these are correct header values
   resultsFile << topo.NumSockets() << ","; 
   resultsFile << params.numContMemTypes << ","; 
   resultsFile << maxThreads << ","; 
   if (params.testContRange) 
      resultsFile << "t" << std::endl;
   else  
      resultsFile << "f" << std::endl;
   //PrintResults(resultsFile, blockSteps, data);
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

void BurstHHBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData) { 
   long long blockSize = params.burstBlockSize;
   int numNodes = topo.NumNodes();
   int numSockets = params.nSockets;
   int numPatterns = 1;

   if (params.runPatternsHD)
      numPatterns = NUM_PATTERNS;

   burstData.resize(numPatterns * numSockets);
   double convConst = (double) blockSize / (double) pow(2.0, 30.0) * (double) 1.0e6; 

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
              
               if (params.testAllMemTypes) {
                  // HtoH Ranged Transfer - Pinned Memory Src
                  burstData[rowIdx].push_back(convConst / BurstMemCopy(topo, blockSize, HOST_PINNED_HOST_COPY, destIdx, srcIdx, params.numStepRepeats, pattern)); 
                  // HtoH Ranged Transfer - Pinned Memory Dest
                  burstData[rowIdx].push_back(convConst / BurstMemCopy(topo, blockSize, HOST_HOST_PINNED_COPY, destIdx, srcIdx, params.numStepRepeats, pattern));        
                  // HtoH Ranged Transfer - Pinned Memory Both
                  burstData[rowIdx].push_back(convConst / BurstMemCopy(topo, blockSize, HOST_HOST_COPY_PINNED, destIdx, srcIdx, params.numStepRepeats, pattern));
                  // HtoH Ranged Transfer - WC Memory Src
                  //burstData[rowIdx].push_back(convConst / BurstMemCopy(topo, blockSize, HOST_COMBINED_HOST_COPY, destIdx, srcIdx, params.numStepRepeats, pattern));        
                  // HtoH Ranged Transfer - WC Memory Dest
                  //burstData[rowIdx].push_back(convConst / BurstMemCopy(topo, blockSize, HOST_HOST_COMBINED_COPY, destIdx, srcIdx, params.numStepRepeats, pattern));
               }       
            }
         }
      }
   }
}

void BurstHDBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData) { 
   long long blockSize = params.burstBlockSize;
   double convConst = (double) blockSize / (double) pow(2.0, 30.0) * (double) 1.0e6; 

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
               
               if ( params.testAllMemTypes) {      
                  // HtoD Ranged Transfer - Pinned Memory
                  burstData[rowIdx].push_back( convConst / BurstMemCopy(topo, blockSize, HOST_PINNED_DEVICE_COPY, destIdx, srcIdx, params.numStepRepeats, pattern));
                  // DtoH Ranged Transfer - Pinned Memory
                  burstData[rowIdx].push_back( convConst / BurstMemCopy(topo, blockSize, DEVICE_HOST_PINNED_COPY, srcIdx, destIdx, params.numStepRepeats, pattern)); 
                  // HtoD Ranged Transfer - WC Memory
                  //burstData[rowIdx].push_back( convConst / BurstMemCopy(topo, blockSize, HOST_COMBINED_DEVICE_COPY, destIdx, srcIdx, params.numStepRepeats, pattern));
                  // DtoH Ranged Transfer - WC Memory
                  //burstData[rowIdx].push_back( convConst / BurstMemCopy(topo, blockSize, DEVICE_HOST_COMBINED_COPY, srcIdx, destIdx, params.numStepRepeats, pattern)); 
               }
            }
         }
      }
   }
}

void BurstP2PBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData) { 
   long long blockSize = params.burstBlockSize;
   double convConst = (double) blockSize / (double) pow(2.0, 30.0) * (double) 1.0e-6; 
   
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

void MemCopyRun(SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &data, MEM_OP copyType, MEM_PATTERN pattern, int destIdx, int srcIdx, int numCopies) {
   void *destPtr, *srcPtr; 
   long long totalSteps = blockSteps.size();
   long long blockSize = blockSteps[totalSteps - 1];

   AllocMemBlocks(topo, &destPtr, &srcPtr, blockSize, copyType, destIdx, srcIdx);
   SetMemBlocks(topo, destPtr, srcPtr, blockSize, copyType, destIdx, srcIdx, 0);
   for (long long stepNum = 0; stepNum < totalSteps; stepNum++) {
      data[stepNum].push_back(TimedMemCopyStep(destPtr, srcPtr, blockSteps[stepNum], blockSize, numCopies, copyType, pattern, destIdx, srcIdx));
   }
   FreeMemBlocks(topo, destPtr, srcPtr, blockSize, copyType, destIdx, srcIdx);
}

float BurstMemCopy(SystemTopo &topo, long long blockSize, MEM_OP copyType, int destIdx, int srcIdx, int numSteps, MEM_PATTERN pattern) {  
   float elapsedTime = 0;
   void *destPtr, *srcPtr;

   AllocMemBlocks(topo, &destPtr, &srcPtr, blockSize, copyType, destIdx, srcIdx);
   SetMemBlocks(topo, destPtr, srcPtr, blockSize, copyType, destIdx, srcIdx, 0); 

   elapsedTime = TimedMemCopyStep(destPtr, srcPtr, blockSize, blockSize, numSteps, copyType, pattern, destIdx, srcIdx);

   FreeMemBlocks(topo, destPtr, srcPtr, blockSize, copyType, destIdx, srcIdx);

   return elapsedTime;
}

float TimedMemCopyStep(void * destPtr, void *srcPtr, long long stepSize, long long blockSize, int numCopiesPerStep, MEM_OP copyType, MEM_PATTERN patternType, int destIdx, int srcIdx) {
   long long offset = 0;
   long long maxFrameSize = 100000000;
   long long gap = 100000000;
   float time = 0.0;
   bool usingPattern = false;
   
   if (stepSize < maxFrameSize) {
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

   Timer CopyTimer(true); 

   if (  copyType == HOST_PINNED_DEVICE_COPY || copyType == DEVICE_HOST_PINNED_COPY ||
         copyType == HOST_COMBINED_DEVICE_COPY || copyType == DEVICE_HOST_COMBINED_COPY ||
         copyType == DEVICE_DEVICE_COPY || copyType == PEER_COPY_NO_UVA || copyType == COPY_UVA) {
      CopyTimer.SetHostTiming(false);
   }
   
   CopyTimer.StartTimer();
   for (int copyIdx = 0; copyIdx < numCopiesPerStep; copyIdx++) {
      char * dest = ((char *) destPtr) + offset;
      char * src = ((char *) srcPtr) + offset;
      
      if (CopyTimer.UseHostTimer) 
         MemCopyOp((void *) dest, (void *) src, stepSize, copyType, destIdx, srcIdx);//, CopyTimer.stream); 
      else 
         MemCopyOp((void *) dest, (void *) src, stepSize, copyType, destIdx, srcIdx, CopyTimer.stream); 

      if (usingPattern) {
         switch (patternType) {
            case LINEAR_INC:
               offset += gap;
               if ((offset + stepSize) >= blockSize)
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
   CopyTimer.StopTimer();
   time += CopyTimer.ElapsedTime();

   return time / (float) numCopiesPerStep;
}

void SetMemBlock(SystemTopo &topo, void *blkPtr, long long numBytes, long long value, MEM_TYPE memType, int devIdx) {
   switch (memType) {
      case PAGE:
      case PINNED:
      case WRITE_COMBINED:
      case MANAGED:
      case MAPPED:
         topo.SetHostMem(blkPtr, value, numBytes);
         break;
      case DEVICE:
         topo.SetDeviceMem(blkPtr, value, numBytes, devIdx);
         break;
      default:
         std::cout << "Error: unrecognized memory set operation type for block set!" << std::endl; 
         break;
   }
}

void SetMemBlocks(SystemTopo &topo, void *destPtr, void *srcPtr, long long numBytes, MEM_OP copyType, int destIdx, int srcIdx, int value) {
   switch (copyType) {
      case HOST_HOST_COPY: 
      case HOST_PINNED_HOST_COPY: 
      case HOST_HOST_PINNED_COPY: 
      case HOST_HOST_COPY_PINNED: 
      case HOST_COMBINED_HOST_COPY:
      case HOST_HOST_COMBINED_COPY:
      case HOST_HOST_COPY_COMBINED:
         topo.SetHostMem(srcPtr, value, numBytes);
         topo.SetHostMem(destPtr, value, numBytes);
         break;
      case DEVICE_HOST_COPY:
      case DEVICE_HOST_PINNED_COPY:
      case DEVICE_HOST_COMBINED_COPY:
         topo.SetDeviceMem(srcPtr, value, numBytes, srcIdx);
         topo.SetHostMem(destPtr, value, numBytes);
         break;
      case HOST_DEVICE_COPY:
      case HOST_PINNED_DEVICE_COPY:
      case HOST_COMBINED_DEVICE_COPY:
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
         std::cout << "Error: unrecognized memory set operation type for setting blocks!" << std::endl; 
         break;
   }
}

void AllocMemBlock(SystemTopo &topo, void **blkPtr, long long numBytes, MEM_TYPE blockType, int srcIdx, int extIdx) {
   switch (blockType) {
      case PAGE:
         *blkPtr = topo.AllocMemByNode(srcIdx, numBytes);
         break;
      case PINNED:
         *blkPtr = topo.AllocPinMemByNode(srcIdx, numBytes);
         break;
      case WRITE_COMBINED:
         *blkPtr = topo.AllocWCMemByNode(srcIdx, numBytes);
         break;
      case MANAGED:
         *blkPtr = topo.AllocManagedMemByNode(srcIdx, extIdx, numBytes);
         break;
      case MAPPED:
         *blkPtr = topo.AllocMappedMemByNode(srcIdx, extIdx, numBytes);
         break;
      case DEVICE:
         *blkPtr = topo.AllocDeviceMem(srcIdx, numBytes);
         break;
      default:
         std::cout << "Error: unrecognized memory type for allocation!" << std::endl; 
         break;
   }
}

void AllocMemBlocks(SystemTopo &topo, void **destPtr, void **srcPtr, long  long numBytes, MEM_OP copyType, int destIdx, int srcIdx) {
   switch (copyType) {
      case HOST_HOST_COPY: 
         *srcPtr = topo.AllocMemByNode(srcIdx, numBytes);
         *destPtr = topo.AllocMemByNode(destIdx, numBytes);
         break;
      case HOST_PINNED_HOST_COPY: 
         *srcPtr = topo.AllocPinMemByNode(srcIdx, numBytes);
         *destPtr = topo.AllocMemByNode(destIdx, numBytes);
         break;
      case HOST_HOST_PINNED_COPY: 
         *srcPtr = topo.AllocMemByNode(srcIdx, numBytes);
         *destPtr = topo.AllocPinMemByNode(destIdx, numBytes);
         break;
      case HOST_HOST_COPY_PINNED: 
         *srcPtr = topo.AllocPinMemByNode(srcIdx, numBytes);
         *destPtr = topo.AllocPinMemByNode(destIdx, numBytes);
         break;
      case HOST_COMBINED_HOST_COPY:
         *srcPtr = topo.AllocWCMemByNode(srcIdx, numBytes);
         *destPtr = topo.AllocMemByNode(destIdx, numBytes);
         break;
      case HOST_HOST_COMBINED_COPY:
         *srcPtr =topo.AllocMemByNode(srcIdx, numBytes);
         *destPtr = topo.AllocWCMemByNode(destIdx, numBytes);
          break;
      case HOST_HOST_COPY_COMBINED:
         *srcPtr =topo.AllocWCMemByNode(srcIdx, numBytes);
         *destPtr = topo.AllocWCMemByNode(destIdx, numBytes);
         break;
      case DEVICE_HOST_COPY:
         *srcPtr = topo.AllocDeviceMem(srcIdx, numBytes);
         *destPtr = topo.AllocMemByNode(destIdx, numBytes);
         break;
      case DEVICE_HOST_PINNED_COPY:
         *srcPtr = topo.AllocDeviceMem(srcIdx, numBytes);
         *destPtr = topo.AllocPinMemByNode(destIdx, numBytes);
         break;
      case DEVICE_HOST_COMBINED_COPY:
         *srcPtr = topo.AllocDeviceMem(srcIdx, numBytes);
         *destPtr = topo.AllocWCMemByNode(destIdx, numBytes);
         break;
      case HOST_DEVICE_COPY:
         *srcPtr = topo.AllocMemByNode(srcIdx, numBytes);
         *destPtr = topo.AllocDeviceMem(destIdx, numBytes);
         break;
      case HOST_PINNED_DEVICE_COPY:
         *srcPtr = topo.AllocPinMemByNode(srcIdx, numBytes);
         *destPtr = topo.AllocDeviceMem(destIdx, numBytes);
         break;
      case HOST_COMBINED_DEVICE_COPY:
         *srcPtr = topo.AllocWCMemByNode(srcIdx, numBytes);
         *destPtr = topo.AllocDeviceMem(destIdx, numBytes);
         break;
      case PEER_COPY_NO_UVA: 
      case DEVICE_DEVICE_COPY:
      case COPY_UVA:
         *srcPtr = topo.AllocDeviceMem(srcIdx, numBytes);
         *destPtr = topo.AllocDeviceMem(destIdx, numBytes);
         break;
      default:
         std::cout << "Error: unrecognized memory copy operation type for allocation!" << std::endl;
         break;
   }
}

void MemCopyOp(void * destPtr, void *srcPtr, long long stepSize, MEM_OP copyType, int destIdx, int srcIdx, cudaStream_t stream) {
   switch (copyType) {
      case HOST_HOST_COPY: 
         memcpy(destPtr, srcPtr, stepSize);
         break;
      case HOST_PINNED_HOST_COPY: 
      case HOST_HOST_PINNED_COPY:
      case HOST_COMBINED_HOST_COPY:
      case HOST_HOST_COMBINED_COPY: 
      case HOST_HOST_COPY_PINNED: 
      case HOST_HOST_COPY_COMBINED:
         checkCudaErrors(cudaMemcpy(destPtr, srcPtr, stepSize, cudaMemcpyHostToHost));
         break;
      case DEVICE_HOST_COPY:
         checkCudaErrors(cudaMemcpy(destPtr, srcPtr, stepSize, cudaMemcpyDeviceToHost));
         break;
      case DEVICE_HOST_PINNED_COPY:
      case DEVICE_HOST_COMBINED_COPY:
         checkCudaErrors(cudaMemcpyAsync(destPtr, srcPtr, stepSize, cudaMemcpyDeviceToHost, stream));
         break;
      case HOST_DEVICE_COPY:
         checkCudaErrors(cudaMemcpy(destPtr, srcPtr, stepSize, cudaMemcpyHostToDevice));
         break;
      case HOST_PINNED_DEVICE_COPY:
      case HOST_COMBINED_DEVICE_COPY:
         checkCudaErrors(cudaMemcpyAsync(destPtr, srcPtr, stepSize, cudaMemcpyHostToDevice, stream));
         break;
      case PEER_COPY_NO_UVA:
         checkCudaErrors(cudaMemcpyPeerAsync(destPtr, destIdx, srcPtr, srcIdx, stepSize, stream));
         break;
      case DEVICE_DEVICE_COPY:
         checkCudaErrors(cudaMemcpyAsync(destPtr, srcPtr, stepSize, cudaMemcpyDeviceToDevice, stream));
         break;
      case COPY_UVA:
         checkCudaErrors(cudaMemcpyAsync(destPtr, srcPtr, stepSize, cudaMemcpyDefault, stream));
         break;
      default:
         std::cout << "Error: unrecognized timed memory copy operation type" << std::endl; 
         break;
   }
}

void FreeMemBlocks(SystemTopo &topo, void* destPtr, void *srcPtr, long long numBytes, MEM_OP copyType, int destIdx, int srcIdx) {
   switch (copyType) {
      case HOST_HOST_COPY: 
         topo.FreeHostMem(destPtr, numBytes);
         topo.FreeHostMem(srcPtr, numBytes);
         break;
      case HOST_PINNED_HOST_COPY:  
         topo.FreePinMem(srcPtr, numBytes);
         topo.FreeHostMem(destPtr, numBytes);
         break;
      case HOST_HOST_PINNED_COPY:  
         topo.FreeHostMem(srcPtr, numBytes);
         topo.FreePinMem(destPtr, numBytes);
         break;
      case HOST_HOST_COPY_PINNED:  
         topo.FreePinMem(srcPtr, numBytes);
         topo.FreePinMem(destPtr, numBytes);
         break;
      case HOST_COMBINED_HOST_COPY:
         topo.FreeWCMem(srcPtr);
         topo.FreeHostMem(destPtr, numBytes);
         break;
      case HOST_HOST_COMBINED_COPY:
         topo.FreeHostMem(srcPtr, numBytes);
         topo.FreeWCMem(destPtr);
         break;
      case HOST_HOST_COPY_COMBINED:
         topo.FreeWCMem(srcPtr);
         topo.FreeWCMem(destPtr);
         break;
      case DEVICE_HOST_COPY:
         topo.FreeDeviceMem(srcPtr, srcIdx);
         topo.FreeHostMem(destPtr, numBytes);
         break;
      case DEVICE_HOST_PINNED_COPY:
         topo.FreeDeviceMem(srcPtr, srcIdx);
         topo.FreePinMem(destPtr, numBytes);
         break;
      case DEVICE_HOST_COMBINED_COPY:
         topo.FreeDeviceMem(srcPtr, srcIdx);
         topo.FreeWCMem(destPtr);
         break;
      case HOST_DEVICE_COPY:
         topo.FreeHostMem(srcPtr, numBytes);
         topo.FreeDeviceMem(destPtr, destIdx);
         break;
      case HOST_PINNED_DEVICE_COPY:
         topo.FreePinMem(srcPtr, numBytes);
         topo.FreeDeviceMem(destPtr, destIdx);
         break;
      case HOST_COMBINED_DEVICE_COPY:
         topo.FreeWCMem(srcPtr);
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

float TimedMemManageOp(void **MemBlk, long long NumBytes, MEM_OP TimedOp) {
   Timer OpTimer(true); 
   OpTimer.StartTimer();
 
   switch (TimedOp) {
      case HOST_MALLOC:
         *MemBlk = malloc(NumBytes); 
         break;
      case HOST_PINNED_MALLOC:
         checkCudaErrors(cudaHostAlloc(MemBlk, NumBytes, cudaHostAllocPortable));
         break;
      case HOST_COMBINED_MALLOC:
         checkCudaErrors(cudaHostAlloc(MemBlk, NumBytes, cudaHostAllocPortable | cudaHostAllocWriteCombined));
         break;
      case MANAGED_MALLOC:
         checkCudaErrors(cudaMallocManaged(MemBlk, NumBytes));
         break;
      case MAPPED_MALLOC: 
         checkCudaErrors(cudaHostAlloc(MemBlk, NumBytes, cudaHostAllocPortable | cudaHostAllocMapped));
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
      case HOST_COMBINED_FREE:
         checkCudaErrors(cudaFreeHost(*MemBlk));
         break;
      case MANAGED_FREE:
         checkCudaErrors(cudaFree(*MemBlk));
         break;
      case MAPPED_FREE:
         checkCudaErrors(cudaFreeHost(*MemBlk));
         break;
      case DEVICE_FREE:
         checkCudaErrors(cudaFree(*MemBlk)); 
         break;
      default:
         std::cout << "Error: unrecognized timed memory operation type!" << std::endl; 
         break;
   }

   OpTimer.StopTimer();

   return OpTimer.ElapsedTime();
}

int CalcRunSteps(std::vector<long long> &blockSteps, long long startStep, long long stopStep, long long numSteps) {
   int magStart = max((int) log10(startStep), 1);
   int magStop = log10(stopStep);
   long long totalSteps = (magStop - magStart) * numSteps;
   long long start = pow(10, magStart);
   long long stop = pow(10, magStop); 
   long long step = start;

   double expStep = ((double) (magStop  - magStart)) / (double) totalSteps;
   double exp = 1.0;

   if (stop == step) {
      blockSteps.push_back(start);      
      totalSteps = 1;
   }

   while (step < stop) {
      step = pow(10, exp);
      blockSteps.push_back(step); 
      exp += expStep;
   }

   //for (int i = 0; i < blockSteps.size(); ++i) 
   //   std::cout << blockSteps[i] << std::endl;

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
         fileStream << topo.NumSockets() << ",";
         fileStream << topo.NumNodes() << ",";

         if (params.runPatternsHD)
            fileStream << NUM_PATTERNS;
         else
            fileStream << 1;

         if (params.testAllMemTypes)
            fileStream << ",t";
         else 
            fileStream << ",f";

         if (params.runSocketTests) 
            fileStream << ",t";
         else
            fileStream << ",f";

         fileStream << std::endl;
         break;
      case HD:
         fileStream << topo.NumSockets() << ",";
         fileStream << topo.NumNodes() << ",";
         fileStream << params.nDevices << ",";

         if (params.runPatternsHD)
            fileStream << NUM_PATTERNS;
         else
            fileStream << 1;

         if (params.testAllMemTypes)
            fileStream << ",t";
         else 
            fileStream << ",f";

         if (params.runSocketTests) 
            fileStream << ",t";
         else
            fileStream << ",f";

         for (int i = 0; i < params.nDevices; i++) {
            fileStream << "," << topo.GetDeviceName(i);
         }
         
         fileStream << std::endl;
         break;
      case P2P:
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
 
         /*for (int i = 0; i < params.nDevices; i++) {
            fileStream << "," << std::boolalpha << topo.DeviceUVA(i) << std::noboolalpha;
         }*/
         
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

void PrintResults(std::ofstream &outFile, std::vector<long long> &steps, std::vector<std::vector<float> > &results) {
   
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

