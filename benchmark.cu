
// Benchmark includes and defines
#ifndef BENCH_HEADER_INC
#include "benchmark.h"
#define BENCH_HEADER_INC
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
void NURMATest(BenchParams &params, SystemTopo &topo);

// Test Subfunctions
void RangeHDBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData); 
void RangeHHBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData); 
void RangeP2PBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData);
void BurstHDBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData); 
void BurstHHBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData); 
void BurstP2PBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData);  
void ContentionSubTestMem(BenchParams &params, SystemTopo &topo);
void ContentionSubTestQPI(BenchParams &params, SystemTopo &topo);
void ContentionSubTestPCIe(BenchParams &params, SystemTopo &topo);
void ContentionSubTestP2P(BenchParams &params, SystemTopo &topo);

// Support functions
void MemCopyRun(SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData, MEM_OP copyType, MEM_PATTERN patternType, int destIdx, int srcIdx, int numCopies); 
float TimedMemManageOp(void **MemBlk, long long NumBytes, MEM_OP TimedOp); 
float TimedMemCopyStep(void * destPtr, void *srcPtr, long long stepSize, long long blockSize, int numCopiesPerStep, MEM_OP copyType, MEM_PATTERN patternType, int destIdx = 0, int srcIdx = 0);
float BurstMemCopy(SystemTopo &topo, long long blockSize, MEM_OP copyType, int destIdx, int srcIdx, int numSteps, MEM_PATTERN pattern = REPEATED); 
void MemCopyOp(void * destPtr, void *srcPtr, long long stepSize, MEM_OP copyType, int destIdx = 0, int srcIdx = 0, cudaStream_t stream = 0);
int CalcRunSteps(std::vector<long long> &blockSteps, long long startStep, long long stopStep, long long numSteps);

// Memory Operation Functions Based on enum Types (see benchmark.h)
void AllocMemBlocks(SystemTopo &topo, void **destPtr, void **srcPtr, long long numBytes, MEM_OP copyType, int destIdx = 0, int srcIdx = 0);
void AllocMemBlock(SystemTopo &topo, void **blkPtr, long long numBytes, MEM_TYPE blockType, int srcIdx, int extIdx = 0);
void FreeMemBlocks(SystemTopo &topo, void* destPtr, void *srcPtr, long long numBytes, MEM_OP copyType, int destIdx = 0, int srcIdx = 0);
void SetMemBlocks(SystemTopo &topo, void *destPtr, void *srcPtr, long long numBytes, MEM_OP copyType, int destIdx, int srcIdx, int value); 
void SetMemBlock(SystemTopo &topo, void *blkPtr, long long numBytes, long long value, MEM_TYPE memType, int devIdx = 0);

// Results Output (CSV or stdio)
void PrintRangedHeader(BenchParams &params, SystemTopo &topo, std::ofstream &fileStream, BW_RANGED_TYPE testType); 
void PrintResults(std::ofstream &outFile, std::vector<std::vector<float> > &results);
void PrintResults(std::ofstream &outFile, std::vector<long long> &steps, std::vector<std::vector<float> > &results);
void PrintHHBurstMatrix(BenchParams &params, std::vector<std::vector<float> > &burstData);
void PrintHDBurstMatrix(BenchParams &params, std::vector<std::vector<float> > &burstData);
void PrintP2PBurstMatrix(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData);

std::vector<std::string> PatternNames{"Repeated","Random", "Linear Increasing","Linear Decreasing"};
 
// TARUC Benchmark main()
int main (int argc, char **argv) {
   BenchParams benchParams;  
   SystemTopo sysTopo;
   
   std::cout << "\nStarting Topology Aware Resource Usability and Contention (TARUC) Benchmark...\n" << std::endl; 
   
   // Determine the number of recognized CUDA enabled devices
   benchParams.nDevices = sysTopo.NumGPUs();

   // Exit if system contains no devices
   if (benchParams.nDevices <= 0) {
      std::cout << "No devices found...aborting benchmarks." << std::endl;
      exit(-1);
   }

   // Setup benchmark parameters from file (non-default) if parameter file provided
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
   sysTopo.PrintDeviceProps(benchParams.devPropFile);

   // Check parameters and fix parameters associated with boolean flags
   if (benchParams.runSustainedTests == false)
      benchParams.numStepRepeats = 1;

   // Check if socket tests are enabled
   benchParams.nSockets = 1;
   if (benchParams.runSocketTests)
      benchParams.nSockets = sysTopo.NumSockets();

   // Check if multiple devices are to be used for bandwidth and overhead tests
   if (!benchParams.runAllDevices)
      benchParams.nDevices = 1;
   
   // Set test parameter nNodes = # NUMA nodes
   benchParams.nNodes = sysTopo.NumNodes();

   // Print actual benchmark parameters for user/script parsing
   benchParams.PrintParams();

   // Run the benchmark per parameters defined in params
   RunBenchmarkSuite(benchParams, sysTopo);

   std::cout << "\nTARUC Micro-Benchmarks complete!\n" << std::endl;
   
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

   // Non-Uniform Random Memory Access (NURMA) Test 
   if (params.runNURMATest)
      NURMATest(params, topo);

   // Contention benchmark tests
   if (params.runContentionTest)
      TestContention(params, topo);

}

void TestMemoryOverhead(BenchParams &params, SystemTopo &topo) {
   void *deviceMem = NULL, *hostPageMem = NULL, *hostPinnedMem = NULL, * hostCombinedMem = NULL;
   std::vector<long long> steps;
   std::vector<std::vector<float> > overheadData;
   int testNum = 0;
   
   std::cout << "\nRunning Ranged Memory Overhead Test...\n" << std::endl;
 
   // Calculate memory block size range form input parameters
   CalcRunSteps(steps, params.rangeMemOH[0], params.rangeMemOH[1], params.numRangeSteps);  
   overheadData.resize(steps.size());
   
   // Iterate through combinations of CPU sockets and NUMA Nodes
   for (int socketIdx = 0; socketIdx < params.nSockets; socketIdx++) {
      topo.PinSocket(socketIdx);
 
      for (int nodeIdx = 0; nodeIdx < params.nNodes; nodeIdx++) { 
         topo.PinNode(nodeIdx);
        
         std::cout << "Test " << testNum++ << " Host Alloc/Free, Pinned/Pageable/Write-Combined\t";
         std::cout << "NUMA Node: " << nodeIdx << " CPU: " << socketIdx << std::endl;            
         
         // Host based management memory types
         for (long stepIdx = 0; stepIdx < steps.size(); stepIdx++) {
            
            long long chunkSize = steps[stepIdx];
            float hostAllocTime = 0, pinAllocTime = 0, combAllocTime = 0;
            float hostFreeTime = 0, pinFreeTime = 0, combFreeTime = 0; 
            
            // Repeat each memory block size allocation/deallocation reIdx number of times
            for (int reIdx = 0; reIdx < params.numStepRepeats; reIdx++) {
               hostFreeTime += TimedMemManageOp(&hostPageMem, chunkSize, HOST_MALLOC);
               SetMemBlock(topo, hostPageMem, chunkSize, 0, PAGE);
               hostAllocTime += TimedMemManageOp(&hostPageMem, chunkSize, HOST_FREE);

               if (params.testAllMemTypes) {
                  pinAllocTime += TimedMemManageOp(&hostPinnedMem, chunkSize, HOST_PINNED_MALLOC);
                  SetMemBlock(topo, hostPinnedMem, chunkSize, 0, PINNED);
                  pinFreeTime += TimedMemManageOp(&hostPinnedMem, chunkSize, HOST_PINNED_FREE); 
               
                  combAllocTime += TimedMemManageOp(&hostCombinedMem, chunkSize, HOST_COMBINED_MALLOC);
                  SetMemBlock(topo, hostCombinedMem, chunkSize, 0, WRITE_COMBINED);
                  combFreeTime += TimedMemManageOp(&hostCombinedMem, chunkSize, HOST_COMBINED_FREE);
               }
            }

            // Average host timings for all repeated steps
            hostAllocTime /= (float) params.numStepRepeats;
            hostFreeTime /= (float) params.numStepRepeats;

            // Set min value for 0 returns; this is essential maximum accuracy
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
         }   
      }

      topo.PinSocket(socketIdx);

      // Device based memory management overhead; iterate through all devices
      for (int currDev = 0; currDev < params.nDevices; currDev++) {
         topo.SetActiveDevice(currDev);

         std::cout << "Test " << testNum++ << " Device Alloc/Free \t\t\t\t" << "CPU: " << socketIdx << " Dev: " << currDev << std::endl;            
        
         // Test each block size in order 
         for (long stepIdx = 0; stepIdx < steps.size(); stepIdx++) {

            long long chunkSize = steps[stepIdx];
            float devAllocTime = 0, devFreeTime = 0;

            // Repeat device allocations/deallocations
            for (int reIdx = 0; reIdx < params.numStepRepeats; reIdx++) {

               // Allocation of device memory  
               devAllocTime += TimedMemManageOp(&deviceMem, chunkSize, DEVICE_MALLOC);
               SetMemBlock(topo, deviceMem, chunkSize, 0, DEVICE, currDev);

               // DeAllocation of device memory 
               devFreeTime += TimedMemManageOp(&deviceMem, chunkSize, DEVICE_FREE);

            }

            // Average all runs of each management operation
            overheadData[stepIdx].push_back(devAllocTime / (float) params.numStepRepeats);
            overheadData[stepIdx].push_back(devFreeTime / (float) params.numStepRepeats);
         }
      }
   }

   // Open Overhead results file   
   std::string dataFileName = "./results/overhead/" + params.runTag + "_overhead.csv";
   std::ofstream overheadResultsFile(dataFileName.c_str());

   // Output Header to results file
   overheadResultsFile << params.nSockets << ",";
   overheadResultsFile << params.nNodes << ",";
   overheadResultsFile << params.nDevices;
   
   if (params.testAllMemTypes)
      overheadResultsFile << ",t";
   else 
      overheadResultsFile << ",f";

   for (int i = 0; i < params.nDevices; i++)
      overheadResultsFile << "," << topo.GetDeviceName(i);
   overheadResultsFile << std::endl;

   // Print results file to .csv file
   PrintResults(overheadResultsFile, steps, overheadData);

   std::cout << "\nMemory Overhead Test Complete!" << std::endl;
}

void HHRangeTransferTest(BenchParams &params, SystemTopo &topo) {
   std::vector<std::vector<float> > rangeData;
   std::vector<long long> steps;
   
   std::cout << "\nRunning Ranged Host-Host Bandwidth Tests...\n" << std::endl;
    
   // Calculate memory block size range form input parameters
   CalcRunSteps(steps, params.rangeHHBW[0], params.rangeHHBW[1], params.numRangeSteps); 
   rangeData.resize(steps.size());
  
   // Run ranged block size host-host memory transfer test 
   RangeHHBandwidthRun(params, topo, steps, rangeData);

   // Output results as transfer time (aka function call time)
   // tt == Transfer Time
   std::string dataFileName = "./results/bandwidth/" + params.runTag + "_ranged_hh_tt.csv";
   std::ofstream ttResultsFileHH(dataFileName.c_str());
   PrintRangedHeader(params, topo, ttResultsFileHH, HH); 
   PrintResults(ttResultsFileHH, steps, rangeData);

   // Output results as bandwidth/throughput (GB/S)
   for (int blkIdx = 0; blkIdx < steps.size(); ++blkIdx) {
      for (int runIdx = 0; runIdx < rangeData[blkIdx].size(); ++runIdx) {
         rangeData[blkIdx][runIdx] = ((double) steps[blkIdx]) / rangeData[blkIdx][runIdx] * 1.0E6;
         rangeData[blkIdx][runIdx] /= pow(2.0, 30.0);
      }
   }

   // Output header and results to CSV file
   dataFileName = "./results/bandwidth/" + params.runTag + "_ranged_hh_bw.csv";
   std::ofstream bwResultsFileHH(dataFileName.c_str());
   PrintRangedHeader(params, topo, bwResultsFileHH, HH); 
   PrintResults(bwResultsFileHH, steps, rangeData);

   std::cout << "\nRanged Host-Host Bandwidth Tests Complete!" << std::endl;
}

void HDRangeTransferTest(BenchParams &params, SystemTopo &topo) {
   std::vector<std::vector<float> > rangeData;
   std::vector<long long> steps;
   
   std::cout << "\nRunning Ranged Host-Device Bandwidth Tests...\n" << std::endl;

   // Calculate memory block size range form input parameters
   CalcRunSteps(steps, params.rangeHDBW[0], params.rangeHDBW[1], params.numRangeSteps); 
   rangeData.resize(steps.size());
  
   // Run ranged block size host-device memory transfer test 
   RangeHDBandwidthRun(params, topo, steps, rangeData);
   
   // Output results as transfer time (aka function call time)
   // tt == Transfer Time
   std::string dataFileName = "./results/bandwidth/" + params.runTag + "_ranged_hd_tt.csv";
   std::ofstream ttResultsFileHD(dataFileName.c_str());
   PrintRangedHeader(params, topo, ttResultsFileHD, HD); 
   PrintResults(ttResultsFileHD, steps, rangeData);

   // Output results as bandwidth/throughput (GB/S)
   for (int blkIdx = 0; blkIdx < steps.size(); ++blkIdx) {
      for (int runIdx = 0; runIdx < rangeData[blkIdx].size(); ++runIdx) {
         rangeData[blkIdx][runIdx] = ((double) steps[blkIdx]) / rangeData[blkIdx][runIdx] * 1.0E6;
         rangeData[blkIdx][runIdx] /= pow(2.0, 30.0);
      }
   }

   // Output header and results to CSV file
   dataFileName = "./results/bandwidth/" + params.runTag + "_ranged_hd_bw.csv";
   std::ofstream bwResultsFileHD(dataFileName.c_str());
   PrintRangedHeader(params, topo, bwResultsFileHD, HD); 
   PrintResults(bwResultsFileHD, steps, rangeData);

   std::cout << "\nHost-Device Ranged Bandwidth Tests Complete!" << std::endl;
}

void P2PRangeTransferTest(BenchParams &params, SystemTopo &topo){
   std::vector<std::vector<float> > rangeData;
   std::vector<long long> steps;
   
   std::cout << "\nRunning P2P Device Ranged Bandwidth Test..." << std::endl;

   // Calculate memory block size range form input parameters
   CalcRunSteps(steps, params.rangeP2PBW[0], params.rangeP2PBW[1], params.numRangeSteps); 
   rangeData.resize(steps.size());
   
   RangeP2PBandwidthRun(params, topo, steps, rangeData);

   // Output results as transfer time (aka function call time)
   // tt == Transfer Time
   std::string dataFileName = "./results/bandwidth/" + params.runTag + "_ranged_p2p_tt.csv";
   std::ofstream ttResultsFileP2P(dataFileName.c_str());
   PrintRangedHeader(params, topo, ttResultsFileP2P, P2P); 
   PrintResults(ttResultsFileP2P, steps, rangeData);

   // Output results as bandwidth/throughput (GB/S)
   for (int blkIdx = 0; blkIdx < steps.size(); ++blkIdx) {
      for (int runIdx = 0; runIdx < rangeData[blkIdx].size(); ++runIdx) {
         rangeData[blkIdx][runIdx] = ((double) steps[blkIdx]) / rangeData[blkIdx][runIdx] * 1.0E6;
         rangeData[blkIdx][runIdx] /= pow(2.0, 30.0);
      }
   }

   // Output header and results to CSV file
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
 
      for (int srcIdx = 0; srcIdx < params.nNodes; srcIdx++) { 
         //topo.PinNode(srcIdx);

         //Host To Host Memory Transfers
         for (int destIdx = 0; destIdx < params.nNodes; destIdx++) { 
            
            // HtoH Ranged Transfer - Pageable Memory
            std::cout << "Test " << testNum++ << " HtoH, Pageable Memory\t\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
            MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COPY, REPEATED, destIdx, srcIdx, numRepeats); 
            if (params.runPatternTests) {
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COPY, LINEAR_INC, destIdx, srcIdx, numRepeats); 
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COPY, LINEAR_DEC, destIdx, srcIdx, numRepeats);
            }

            if (params.testAllMemTypes) {
               
               // HtoH Ranged Transfer - Pinned Memory Src Host
               std::cout << "Test " << testNum++ << " HtoH, Pinned Memory Src  \t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_PINNED_HOST_COPY, REPEATED, destIdx, srcIdx, numRepeats);
               if (params.runPatternTests){ 
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_PINNED_HOST_COPY, LINEAR_INC, destIdx, srcIdx, numRepeats); 
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_PINNED_HOST_COPY, LINEAR_DEC, destIdx, srcIdx, numRepeats); 
               }

               // HtoH Ranged Transfer - Pinned Memory Dest Host
               std::cout << "Test " << testNum++ << " HtoH, Pinned Memory Dest \t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_PINNED_COPY, REPEATED, destIdx, srcIdx, numRepeats); 
               if (params.runPatternTests) {
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_PINNED_COPY, LINEAR_INC, destIdx, srcIdx, numRepeats); 
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_PINNED_COPY, LINEAR_DEC, destIdx, srcIdx, numRepeats); 
               }

               // HtoH Ranged Transfer - Pinned Memory Both Hosts
               std::cout << "Test " << testNum++ << " HtoH, Both Pinned Memory \t\tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COPY_PINNED, REPEATED, destIdx, srcIdx, numRepeats); 
               if (params.runPatternTests) {
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COPY_PINNED, LINEAR_INC, destIdx, srcIdx, numRepeats); 
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COPY_PINNED, LINEAR_DEC, destIdx, srcIdx, numRepeats);
               } 

               // HtoH Ranged Transfer - Write-Combined Memory Dest Host
               std::cout << "Test " << testNum++ << " HtoH, Write-Combined Memory Dest  \tCPU: " << socketIdx << "\t\tNUMA Src: " << srcIdx << "\tDest NUMA: " << destIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_HOST_COMBINED_COPY, REPEATED, destIdx, srcIdx, numRepeats); 
               if (params.runPatternTests) {
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
 
      for (int hostIdx = 0; hostIdx < params.nNodes; hostIdx++) { 
         topo.PinNode(hostIdx);

         //Host-Device PCIe Memory Transfers
         for (int devIdx = 0; devIdx < params.nDevices; devIdx++) {
            topo.SetActiveDevice(devIdx);
            
            // HtoD Ranged Transfer - Pageable Memory
            std::cout << "Test " << testNum++ << " HtoD, Pageable Memory\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << hostIdx << "\tDev Dev: " << devIdx << std::endl;
            MemCopyRun(topo, blockSteps, bandwidthData, HOST_DEVICE_COPY, REPEATED, devIdx, hostIdx, numRepeats); 
            if (params.runPatternTests) {
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_DEVICE_COPY, LINEAR_INC, devIdx, hostIdx, numRepeats); 
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_DEVICE_COPY, LINEAR_DEC, devIdx, hostIdx, numRepeats); 
            }

            // DtoH Ranged Transfer - Pageable Memory
            std::cout << "Test " << testNum++ << " DtoH, Pageable Memory\t\tCPU: " << socketIdx << "\t\tDev Src: " << devIdx << "\tNUMA Dest: " << hostIdx << std::endl;
            MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_COPY, REPEATED, hostIdx, devIdx, numRepeats); 
            if (params.runPatternTests) {
               MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_COPY, LINEAR_INC, hostIdx, devIdx, numRepeats); 
               MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_COPY, LINEAR_DEC, hostIdx, devIdx, numRepeats); 
            }
            
            if (params.testAllMemTypes) {
               
               // HtoD Ranged Transfer - Pinned Memory
               std::cout << "Test " << testNum++ << " HtoD, Pinned Memory\t\tCPU: " << socketIdx << "\t\tNUMA Src: " << hostIdx << "\tDest Dev: " << devIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_PINNED_DEVICE_COPY, REPEATED, devIdx, hostIdx, numRepeats); 
               if (params.runPatternTests) {
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_PINNED_DEVICE_COPY, LINEAR_INC, devIdx, hostIdx, numRepeats); 
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_PINNED_DEVICE_COPY, LINEAR_DEC, devIdx, hostIdx, numRepeats); 
               } 

               // DtoH Ranged Transfer - Pinned Memory
               std::cout << "Test " << testNum++ << " DtoH, Pinned Memory\t\tCPU: " << socketIdx << "\t\tSrc Dev: " << hostIdx << "\tNUMA Dest: " << hostIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_PINNED_COPY, REPEATED, hostIdx, devIdx, numRepeats); 
               if (params.runPatternTests) {
                  MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_PINNED_COPY, LINEAR_INC, hostIdx, devIdx, numRepeats); 
                  MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_PINNED_COPY, LINEAR_DEC, hostIdx, devIdx, numRepeats);
               } 

               // HtoD Ranged Transfer - Write-Combined Memory
               std::cout << "Test " << testNum++ << " HtoD, Write-Combined Memory \tCPU: " << socketIdx << "\t\tNUMA Src: " << hostIdx << "\tDest Dev: " << devIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, HOST_COMBINED_DEVICE_COPY, REPEATED, devIdx, hostIdx, numRepeats); 
               if (params.runPatternTests) {
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_COMBINED_DEVICE_COPY, LINEAR_INC, devIdx, hostIdx, numRepeats); 
                  MemCopyRun(topo, blockSteps, bandwidthData, HOST_COMBINED_DEVICE_COPY, LINEAR_DEC, devIdx, hostIdx, numRepeats); 
               } 

               // DtoH Ranged Transfer - Write-Combined Memory
               std::cout << "Test " << testNum++ << " DtoH, Write-Combined Memory \tCPU: " << socketIdx << "\t\tSrc Dev: " << hostIdx << "\tNUMA Dest: " << hostIdx << std::endl;
               MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_COMBINED_COPY, REPEATED, hostIdx, devIdx, numRepeats); 
               if (params.runPatternTests) {
                  MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_COMBINED_COPY, LINEAR_INC, hostIdx, devIdx, numRepeats); 
                  MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_HOST_COMBINED_COPY, LINEAR_DEC, hostIdx, devIdx, numRepeats);
               }
            }               
         }
      }
   }
}

void RangeP2PBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<long long> &blockSteps, std::vector<std::vector<float> > &bandwidthData) {

   long numRepeats = params.numStepRepeats;  
   int testNum = 0;
  
   // Iterate through each socker and GPU device pair 
   for (int socketIdx = 0; socketIdx < params.nSockets; socketIdx++) {
      topo.PinSocket(socketIdx);
      //topo.PinNode(socketIdx);
 
      for (int srcIdx = 0; srcIdx < params.nDevices; srcIdx++) { 

         for (int destIdx = srcIdx; destIdx < params.nDevices; destIdx++) {

            // DtoD Ranged Transfer - No Peer, No UVA
            std::cout << "Test " << testNum++ << " Device-To-Device, No Peer\tCPU: " << socketIdx << "\tSrc Device: " << srcIdx << "\tDest Device: " << destIdx << std::endl;
            topo.SetActiveDevice(srcIdx);
            MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_DEVICE_COPY, REPEATED, destIdx, srcIdx, numRepeats); 

            // DtoD Ranged Transfer - No Peer, No UVA
            if (destIdx != srcIdx) {

               std::cout << "Test " << testNum++ << " Device-To-Device, No Peer\tCPU: " << socketIdx << "\tDest Device: " << srcIdx << "\tSrc Device: " << destIdx << std::endl;
               topo.SetActiveDevice(destIdx);
               MemCopyRun(topo, blockSteps, bandwidthData, DEVICE_DEVICE_COPY, REPEATED, srcIdx, destIdx, numRepeats); 

            }

            // DtoD Ranged Transfer - Peer, No UVA
            if (topo.DeviceGroupCanP2P(srcIdx, destIdx)) {
               // For device pairs with peer access, enable P2P flags
               topo.DeviceGroupSetP2P(srcIdx, destIdx, true);
 
               std::cout << "Test " << testNum++ << " Device-To-Device, Peer Enabled\tCPU: " << socketIdx << "\tSrc Device: " << srcIdx << "\tDest Device: " << destIdx << std::endl;
               topo.SetActiveDevice(srcIdx);
               MemCopyRun(topo, blockSteps, bandwidthData, PEER_COPY_NO_UVA, REPEATED, destIdx, srcIdx, numRepeats);  

               if (destIdx != srcIdx) {

                  std::cout << "Test " << testNum++ << " Device-To-Device, Peer Enabled\tCPU: " << socketIdx << "\tDest Device: " << srcIdx << "\tSrc Device: " << destIdx << std::endl;
                  topo.SetActiveDevice(destIdx);
                  MemCopyRun(topo, blockSteps, bandwidthData, PEER_COPY_NO_UVA, REPEATED, srcIdx, destIdx, numRepeats);  

               }

               // Return P2P flags back to false for GPU pair 
               topo.DeviceGroupSetP2P(srcIdx, destIdx, false);
            }
         }
      }
   }
}

void NURMATest(BenchParams &params, SystemTopo &topo) {

   std::vector<std::vector<float> > data;
   std::vector<long long> steps;
   long long NumSteps = CalcRunSteps(steps, params.rangeNURMA[0] * sizeof(double), 
                                            params.rangeNURMA[1] * sizeof(double), 
                                            params.numRangeSteps);
   data.resize(steps.size());
	
   int testNum = 0; 
	long long blockSize = params.blockSizeNURMA * sizeof(double);
	long long numDoubles = params.blockSizeNURMA;

   int NumMemTypes = (params.testAllMemTypes ? 2 : 1);

   std::cout << "\nRunning Non-Uniform Random Memory Access (NURMA) Test..." << std::endl;

	for (int memType = 0; memType < NumMemTypes; ++memType) {
		
		for (int socket = 0; socket < topo.NumSockets(); ++socket) {

			for (int srcNode = 0; srcNode < topo.NumNodes(); ++srcNode) {

				for (int destNode = 0; destNode < topo.NumNodes(); ++destNode) {
				
               // Print Benchmark Test Status Info	
					std::cout << "Test: " << testNum++;
					if (memType == 0)
						std::cout << " Pageable Memory"; 
					else 
						std::cout << " Pinned Memory  "; 
					std::cout << "   \tCPU: " << socket << " Src Node: " << srcNode << \
                           " Dest Node: " << destNode << std::endl;

               // Restrict is important for ensuring compiler knows there is no pointer aliasing 
               // (for dynamic memory vs static stack memory where compiler check before runtime)
					double * __restrict__ srcBlk = NULL;
					double * __restrict__ destBlk = NULL;

					// Pin, allocate and set source mem block to srcNode 
					//topo.PinNode(srcNode);
					if (memType == 0)
						srcBlk = (double *) topo.AllocMemByNode(srcNode, blockSize);
					else 
						srcBlk = (double *) topo.AllocPinMemByNode(srcNode, blockSize); 
					topo.SetHostMem(srcBlk, 10, blockSize);
					
					// Pin, allocate and set dest node mem block to destNode 
					//topo.PinNode(destNode);
					if (memType == 0) 
						destBlk = (double *) topo.AllocMemByNode(destNode, blockSize);
					else 
						destBlk = (double *) topo.AllocPinMemByNode(destNode, blockSize);
					topo.SetHostMem(destBlk, 0, blockSize);
				    
					//topo.PinNode(destNode);
					topo.PinSocket(socket);
               
               // Spin to ensure thread has switched context
               for (volatile long long spin = 0; spin < 100; ++spin) {}

					for (long long stepIdx = 0; stepIdx < steps.size(); ++stepIdx) {
	
						long long rangeWidth = steps[stepIdx];
                  long long startIdx = 0;
                  long long gap = params.gapNURMA; // access skip width (in doubles)
					
                  // Start timer for step test	
						Timer timer(true);
						timer.StartTimer();
						
						for (int repIdx = 0; repIdx < rangeWidth; ++repIdx) {
							
							destBlk[startIdx] = srcBlk[startIdx];
							
							startIdx = (startIdx + gap) % numDoubles;
						}
						
						timer.StopTimer();
                  double time = timer.ElapsedTime();

                  // Push elapsed time for step test onto data vector at block size index
                  data[stepIdx].push_back(time);
                  
                  //std::cout << "Num accesses: "<< rangeWidth << " Time: " << time << std::endl;	
					}	

               // Free Memory block after all steps complete for given test			
					if (memType == 0) {
						topo.FreeHostMem(srcBlk, blockSize);
						topo.FreeHostMem(destBlk, blockSize);
					} else {
						topo.FreePinMem(srcBlk, blockSize);
						topo.FreePinMem(destBlk, blockSize);
					}                  
            }
         }
      }
   }

   // Output results
   std::string dataFileName = "./results/random_access/" + params.runTag + "_random_access.csv";
   std::ofstream resultsFile(dataFileName.c_str());

   // Print results header values
   resultsFile << topo.NumSockets() << ","; 
   resultsFile << topo.NumNodes() << ",";
   resultsFile << NumMemTypes << ","; 
   resultsFile << std::endl;

   // Print benchmark results to .csv file
   PrintResults(resultsFile, steps, data);

   std::cout << "\nNURMA Test Complete!" << std::endl;
}

void TestContention(BenchParams &params, SystemTopo &topo) {
   
   std::cout << "\nRunning Contention tests..." << std::endl;

   /* Memory Access: Single Socket, Single Node
    *
    * Host-Host single node memory access
    * Inherently bidirectional transfer (actual bandwidth is double)
    * since destination is same as source.
    */

   ContentionSubTestMem(params, topo);

   /* QPI Bus Test (Multiple Sockets)
    *
    * Host-to-Host: bidirectional and unidirectional
    * Pin multiple cores on a single 
    */

   if (params.nSockets >= 2)
      ContentionSubTestQPI(params, topo);
   else
      std::cout << "One Socket Detected: No inter-CPU communication bus to test!" << std::endl;
  
   /* PCIe (Single and Multiple Sockets)
    * 
    * Host-to-Device & Device-to-Host: bidirectional and unidirectional
    * Single socket (avoid QPI effects) to each combination of GPUs
    */

   ContentionSubTestPCIe(params, topo);

   std::cout << "\nContention tests complete!" << std::endl;
}

void ContentionSubTestMem(BenchParams &params, SystemTopo &topo) {

   float threadBW[topo.NumPUs()];
   std::vector<std::vector<float> > data;
   long long blockSize = params.contBlockSize[0] / sizeof(double);
   
   int testNum = 0;
   int MaxThreads = topo.NumCores();
   int NumOps = 3;
   float conv = 1.0E-6; 

   std::cout << "\nLocal Memory Contention Micro-Benchmarks:" << std::endl;
   std::cout << "Test Options: \n\tHost Count: \n\t\t1...N (N = # NUMA Nodes)" << std::endl;
   std::cout << "\tOperations:\n\t\t0 = memcopy()\n\t\t1 = manual copy\n\t\t2 = triad (manual copy/scale/add)\n" << std::endl;

   for (int hostCount = 1; hostCount <= topo.NumSockets(); ++hostCount) {
      
      for (int opIdx = 0; opIdx < NumOps; ++opIdx) {

         std::cout << "Test " << testNum++ << "   \tNum Host Nodes: " << hostCount;
         std::cout << " Operation: " << opIdx << " Max Threads: " << MaxThreads << std::endl; 

         for (int numThreads = 1; numThreads <= MaxThreads; ++numThreads) {
            
            omp_set_num_threads(numThreads);
            #pragma omp parallel
            {
               // Get local thread ID
               int threadIdx = omp_get_thread_num();
               double* __restrict__ srcBlk;
               double* __restrict__ destBlk;
               double* __restrict__ addBlk;

               double scale = 12;
               // Place transfer onto each socket; alternating sockets for each transfer
               int socket = threadIdx % hostCount;
               int core = (threadIdx / hostCount) % topo.NumCoresPerSocket();
               
               // pin threads to execution space
               topo.PinCoreBySocket(socket, core);               
               topo.PinNode(socket);               

               // Allocate thread local memory to correct NUMA node
               AllocMemBlock(topo, (void **) &srcBlk, blockSize * sizeof(double), PAGE, socket);
               AllocMemBlock(topo, (void **) &destBlk, blockSize * sizeof(double), PAGE, socket);
               if (opIdx != 0)
                  AllocMemBlock(topo, (void **) &addBlk, blockSize * sizeof(double), PAGE, socket);
              
               // Set memory to initial values 
               SetMemBlock(topo, (void *) srcBlk, blockSize * sizeof(double), 8, PAGE);
               SetMemBlock(topo, (void *) destBlk, blockSize * sizeof(double), 0, PAGE);
               if (opIdx != 0)
                  SetMemBlock(topo, (void *) addBlk, blockSize * sizeof(double), 3, PAGE);
               
               // Initialize timer with host timer 
               Timer threadTimer(true);
               
               // start timer and initiate memory operations on each thread simultaneously
               // wait for all threads to be fully initialized (allocation, timer, memset) 
               #pragma omp barrier
               threadTimer.StartTimer();
               
               for (int repCount = 0; repCount < params.numContRepeats; repCount++) {
                  
                  if (opIdx == 0) {          // Memcopy
                  
                     MemCopyOp(destBlk, srcBlk, blockSize * sizeof(double), HOST_HOST_COPY);
                  
                  } else if (opIdx == 1) {   // Manual Copy (assignment operator)
                  
                     for (long long i = 0; i < blockSize; ++i)
                        destBlk[i] = srcBlk[i];
                  
                  } else {                   // Manual Triad (copy, scale, add)
                  
                     #pragma omp simd
                     for (long long i = 0; i < blockSize; ++i)
                        destBlk[i] = srcBlk[i] + scale * addBlk[i];
                  }
               }      
         
               threadTimer.StopTimer();     
                  
               // calculate thread local bandwidth
               double time = (double) threadTimer.ElapsedTime() / (double) params.numContRepeats;
               double bandwidth = ((double) blockSize / (double) pow(2.0, 30.0)) / (time * conv) * (double) sizeof(double);
              
               if (opIdx == 2)
                  bandwidth *= 3;
               else //opIdx ==  1 or 0
                  bandwidth *= 2;
 
               // place thread local bandwidth into bandwidth array 
               threadBW[threadIdx] = bandwidth; 

               // Wait for all threads to complete operations and calculate bandwidth
               #pragma omp barrier
               
               // Calculate aggregate bandwidth and push thread bandwidths to data vector
               #pragma omp single
               {
                  double aggBW = 0;
                  for (int i = 0; i < numThreads; ++i)
                     aggBW += threadBW[i];
                  
                  std::vector<float> testData;
                  testData.push_back(aggBW);
                  for (int i = 0; i < omp_get_num_threads(); ++i)
                     testData.push_back(threadBW[i]);
                  data.push_back(testData);
               }
               
               // Free thread local memory blocks 
               topo.FreeHostMem(srcBlk, blockSize * sizeof(double));
               topo.FreeHostMem(destBlk, blockSize * sizeof(double));
               if (opIdx != 0)
                  topo.FreeHostMem(addBlk, blockSize * sizeof(double));
            }
         } 
      }
   }

   // Open results file for output
   std::string dataFileName = "./results/contention/mem/" + params.runTag + "_mem_contention.csv";
   std::ofstream resultsFile(dataFileName.c_str());

   // Print header to output file
   resultsFile << topo.NumSockets() << ","; 
   resultsFile << NumOps << ","; 
   resultsFile << MaxThreads << std::endl;

   // Print results to output file
   PrintResults(resultsFile, data);
}

void ContentionSubTestQPI(BenchParams &params, SystemTopo &topo) {
   
   float threadBW[topo.NumPUs()];
   std::vector<std::vector<float> > data;
   long long blockSize = params.contBlockSize[1] / sizeof(double);

   int MaxThreads = topo.NumCores();
   int testNum = 0;
   int NumDirs = 2; // Copy Directions: 0->1 unidirectional, bidirectional
   int NumOps = 3;
   float conv = 1.0E-6; 

   std::cout << "\nSocket-Socket Communication (QPI) Contention Micro-Benchmarks" << std::endl;
   std::cout << "Test Options: \n\tDirections:\n\t\t0 = unidirectional\n\t\t1 = bidirectional" << std::endl;
   std::cout << "\tOperations:\n\t\t0 = memcopy()\n\t\t1 = manual copy\n\t\t2 = triad (manual copy/scale/add)\n" << std::endl;

   for (int copyDir = 0; copyDir < NumDirs; copyDir++) {
      
      for (int opIdx = 0; opIdx < NumOps; ++opIdx) {

         std::cout << "Test " << testNum++ << "   \tDirection: " << copyDir;
         std::cout << " Operation: " << opIdx << " Max Threads: " << MaxThreads << std::endl; 

         for (int numThreads = 1; numThreads <= MaxThreads; ++numThreads) {
            omp_set_num_threads(numThreads);
            
            #pragma omp parallel
            {
               // Get local thread ID
               int threadIdx = omp_get_thread_num();
               double* __restrict__ srcBlk;
               double* __restrict__ destBlk;
               double* __restrict__ addBlk;

               double scale = 12;
               int srcNode = 0, destNode = 0;
               int core = 0;

               // Calculate correct NUMA nodes and execution core for each copy direction
               if (copyDir == 0) { // unidirectional, only testing one direction; should be equivalent 
                  srcNode = 0;
                  destNode = 1;
                  core = threadIdx % topo.NumCoresPerSocket();
               
               } else { // bidirectional; alternate src and dest nodes for each thread
                  srcNode = threadIdx % 2; 
                  destNode = (threadIdx + 1) % 2; 
                  core = (threadIdx / 2) % topo.NumCoresPerSocket();
               } 
               
               // pin threads to execution space
               topo.PinCoreBySocket(srcNode, core);
              
               // Allocate thread local memory to correct NUMA node
               AllocMemBlock(topo, (void **) &srcBlk, blockSize * sizeof(double), PAGE, destNode);
               AllocMemBlock(topo, (void **) &destBlk, blockSize * sizeof(double), PAGE, destNode);
               if (opIdx != 0)
                  AllocMemBlock(topo, (void **) &addBlk, blockSize * sizeof(double), PAGE, destNode);
              
               // Set thread memory to initial values 
               SetMemBlock(topo, (void *) srcBlk, blockSize * sizeof(double), 8, PAGE);
               SetMemBlock(topo, (void *) destBlk, blockSize * sizeof(double), 0, PAGE);
               if (opIdx != 0)
                  SetMemBlock(topo, (void *) addBlk, blockSize * sizeof(double), 3, PAGE);
              
               topo.PinNode(destNode);
 
               // Initialize timer to use host based timing (vs CUDA events)
               Timer threadTimer(true);
 
               // start timer and initiate memory operations on each thread simultaneously
               // wait for all threads to be fully initialized (allocation, timer, memset) 
               #pragma omp barrier
               threadTimer.StartTimer();
               
               for (int repCount = 0; repCount < params.numContRepeats; repCount++) {
                  
                  if (opIdx == 0) {          
                     
                     // Memcopy
                     MemCopyOp(destBlk, srcBlk, blockSize * sizeof(double), HOST_HOST_COPY);
                  
                  } else if (opIdx == 1) {   
                     
                     // Manual Copy (assignment operator)
                     for (long long i = 0; i < blockSize; ++i)
                        destBlk[i] = srcBlk[i];
                  
                  } else {                   

                     // Manual Triad (copy, scale, add)
                     #pragma omp simd
                     for (long long i = 0; i < blockSize; ++i)
                        destBlk[i] = srcBlk[i] + scale * addBlk[i];
                  }
               }      
         
               threadTimer.StopTimer();     
            
               // calculate thread local bandwidth
               double time = (double) threadTimer.ElapsedTime() / (double) params.numContRepeats;
               double bandwidth = ((double) blockSize / (double) pow(2.0, 30.0)) / (time * conv) * (double) sizeof(double);

               if (opIdx == 2)
                  bandwidth *= 3;
               else //opIdx ==  1 or 0
                  bandwidth *= 2;
               
               // place thread local bandwidth into bandwidth array 
               threadBW[threadIdx] = bandwidth; 

               // Wait for all threads to complete transfers and add thread bandwidths to array
               #pragma omp barrier

               // Calculate aggregate bandwidth and push thread bandwidths to data vector
               #pragma omp single
               {  
                  float aggBW = 0;
                  for (int i = 0; i < numThreads; ++i)
                     aggBW += threadBW[i];
                  
                  std::vector<float> testData;
                  testData.push_back(aggBW);
                     
                  for (int i = 0; i < numThreads; ++i)
                     testData.push_back(threadBW[i]);
                  data.push_back(testData);
               }
           
               // Free thread local memory  
               topo.FreeHostMem(srcBlk, blockSize * sizeof(double));
               topo.FreeHostMem(destBlk, blockSize * sizeof(double));
               if (opIdx != 0)
                  topo.FreeHostMem(addBlk, blockSize * sizeof(double));
            }
         } 
      }
   }

   // Open results file for output
   std::string dataFileName = "./results/contention/qpi/" + params.runTag + "_qpi_contention.csv";
   std::ofstream resultsFile(dataFileName.c_str());

   // Print header to output file
   //resultsFile << topo.NumSockets() << ","; 
   resultsFile << MaxThreads << ",";
   resultsFile << NumDirs << ",";
   resultsFile << NumOps << std::endl;

   // Print results to output file
   PrintResults(resultsFile, data);
}

void ContentionSubTestPCIe(BenchParams &params, SystemTopo &topo) {

   std::vector<std::vector<float> > data;
   long long blockSize = params.contBlockSize[2];
   float threadBW[topo.NumCores()];
   
   // Set Local Test Parameters  
   int testNum = 0; 
   int numDirs = 3; // to, from, both
   int maxThreads = topo.NumCores();
   int numSocketTests = topo.NumSockets() + 1; // one per socket + all sockets
   if (topo.NumSockets() == 1)
      numSocketTests = 1;

   std::cout << "\nPCIe Contention Micro-Benchmark" << std::endl;
   std::cout << "Test Options: \n\tDirections:\n\t\t0,1 = unidirectional\n\t\t2 = bidirectional" << std::endl;
   std::cout << "\tSocket Num: \n\t0...n (n = all sockets)\n\tDevice: \n\t\t0, N-1 (N = # GPUs)" << std::endl;

   std::cout << "\nSingle Device Contention Tests: " << std::endl;

   // Single Device Contention w/ Multiple Host Threads
   for (int devIdx = 0; devIdx < topo.NumGPUs(); ++devIdx) {
      
      for (int dirIdx = 0; dirIdx < numDirs; ++dirIdx) {
         
         for (int socketIdx = 0; socketIdx < numSocketTests; ++socketIdx) {
            
            std::cout << "Test " << testNum++ << "   \tDevice: " << devIdx << " Socket: " << socketIdx;
            std::cout << " Direction: " << dirIdx << " Max Threads: " << maxThreads << std::endl; 

            for (int numThreads = 1; numThreads <= maxThreads; ++numThreads) {
               
               omp_set_num_threads(numThreads);
               #pragma omp parallel
               {
                  // Get local thread ID
                  int threadIdx = omp_get_thread_num();
                  void * __restrict__ hostBlkA, * __restrict__ devBlkA;
                  void * __restrict__ hostBlkB, * __restrict__ devBlkB;
                  int node, core;

                  node = socketIdx;
                  core = threadIdx % topo.NumCoresPerSocket();
                  if (socketIdx == 2) {
                     node = threadIdx % topo.NumSockets();
                     core = threadIdx / topo.NumSockets();
                  }                  

                  // Pin Cores for execution and NUMA memory regions; set GPU device
                  topo.PinCoreBySocket(node, core); 
                  topo.SetActiveDevice(devIdx);
                  topo.PinNode(node);
                  
                  // Allocate Host/Device memory and set initial values
                  hostBlkA = topo.AllocPinMemByNode(node, blockSize);
                  topo.SetHostMem(hostBlkA, 2, blockSize);
                  devBlkA = topo.AllocDeviceMem(devIdx, blockSize);
                  topo.SetDeviceMem(devBlkA, 0, blockSize, devIdx);

                  hostBlkB = topo.AllocPinMemByNode(node, blockSize);
                  topo.SetHostMem(hostBlkB, 1, blockSize);
                  devBlkB = topo.AllocDeviceMem(devIdx, blockSize);
                  topo.SetDeviceMem(devBlkB, 0, blockSize, devIdx);
            
                  // Initialize local thread timer with CUDA event timing and wait for all threads to finish allocation steps
                  Timer threadTimer(false); 
                  #pragma omp barrier
            
                  threadTimer.StartTimer();
                   
                  for (int repCount = 0; repCount < params.numContRepeats; ++repCount) {
                     if (dirIdx == 0) {
                        MemCopyOp(devBlkA, hostBlkA, blockSize, HOST_PINNED_DEVICE_COPY, 0, 0, threadTimer.stream);
                     } else if (dirIdx == 1) { 
                        MemCopyOp(hostBlkA, devBlkA, blockSize, DEVICE_HOST_PINNED_COPY, 0, 0, threadTimer.stream);
                     } else {
                        if (repCount % 2)
                           MemCopyOp(devBlkA, hostBlkA, blockSize, HOST_PINNED_DEVICE_COPY, 0, 0, threadTimer.stream);
                        else
                           MemCopyOp(hostBlkB, devBlkB, blockSize, DEVICE_HOST_PINNED_COPY, 0, 0, threadTimer.stream);
                     }
                  }
                  
                  threadTimer.StopTimer();     
                                   
                  // calculate thread local bandwidth
                  double time = (double) threadTimer.ElapsedTime() / (double) params.numContRepeats;
                  double bandwidth = ((double) blockSize / (double) pow(2.0, 30.0)) / (time * 1.0E-6);
                  
                  // place thread local bandwidth into bandwidth array 
                  threadBW[threadIdx] = bandwidth; 

                  // Wait for all threads to complete transfers and add thread bandwidths to array
                  #pragma omp barrier

                  // Calculate aggregate bandwidth and push thread bandwidths to data vector
                  #pragma omp single
                  {  
                     double aggBW = 0;
                     for (int i = 0; i < numThreads; ++i)
                        aggBW += threadBW[i];

                     std::vector<float> testData;
                     testData.push_back(aggBW);
                        
                     for (int i = 0; i < numThreads; ++i)
                        testData.push_back(threadBW[i]);
                     data.push_back(testData);
                 
                  }
                  // Free thread local memory  
                  topo.FreePinMem(hostBlkA, blockSize);
                  topo.FreeDeviceMem(devBlkA, devIdx);
                  topo.FreePinMem(hostBlkB, blockSize);
                  topo.FreeDeviceMem(devBlkB, devIdx);


               }
            }
         }      
      }
   }

   std::cout << "\nGPU Pair Contenion Tests: " << std::endl;
   
   // GPU Pair PCIe Contention w/ Multiple Host Threads/Sockets
   if (topo.NumGPUs() >= 2) {
      
      for (int socketIdx = 0; socketIdx < numSocketTests; ++socketIdx) {
         
         for (int devIdx1 = 0; devIdx1 < topo.NumGPUs(); ++devIdx1) {
            
            for (int devIdx2 = devIdx1 + 1; devIdx2 < topo.NumGPUs(); ++devIdx2) {
               
               for (int dirIdx = 0; dirIdx < numDirs; ++dirIdx) {
                 
                  std::cout << "Test " << testNum++ << "   \tDevice 1: " << devIdx1 << " Device 2: " << devIdx2;
                  std::cout << " Socket: " << socketIdx << " Direction: " << dirIdx << " Max Threads: " << maxThreads << std::endl; 

                  for (int numThreads = 1; numThreads <= maxThreads; ++numThreads) {
                     
                     omp_set_num_threads(numThreads);
                     #pragma omp parallel
                     {
                        // Get local thread ID
                        int threadIdx = omp_get_thread_num();
                        void * __restrict__ hostBlkA, * __restrict__ devBlkA;
                        void * __restrict__ hostBlkB, * __restrict__ devBlkB;
                        int socket = socketIdx;
                        int core = threadIdx % topo.NumCoresPerSocket();
                        
                        if (socketIdx == topo.NumSockets()) {
                           socket = (threadIdx / 2) % topo.NumSockets(); 
                           core =  2 * (threadIdx / (2 * topo.NumSockets())) + threadIdx % 2;
                        }

                        int device = devIdx2;
                        if (threadIdx % 2 ==  0)
                           device = devIdx1;
                        
                        topo.SetActiveDevice(device);
                        topo.PinCoreBySocket(socket, core);
                        topo.PinNode(socket);
                        
                        // Allocate Host/Device Memory
                        hostBlkA = topo.AllocPinMemByNode(socket, blockSize);
                        topo.SetHostMem(hostBlkA, 2, blockSize);
                        devBlkA = topo.AllocDeviceMem(device, blockSize);
                        topo.SetDeviceMem(devBlkA, 0, blockSize, device);

                        hostBlkB = topo.AllocPinMemByNode(socket, blockSize);
                        topo.SetHostMem(hostBlkB, 1, blockSize);
                        devBlkB = topo.AllocDeviceMem(device, blockSize);
                        topo.SetDeviceMem(devBlkB, 0, blockSize, device);
                  
                        // Initialize local thread timer with CUDA event timing and wait for all threads to finish allocation steps
                        Timer threadTimer(false); 
                        #pragma omp barrier
                  
                        threadTimer.StartTimer();
          
                        for (int repCount = 0; repCount < params.numContRepeats; repCount++) {
                           if (dirIdx == 0) {
                              MemCopyOp(devBlkA, hostBlkA, blockSize, HOST_PINNED_DEVICE_COPY, 0, 0, threadTimer.stream);
                           } else if (dirIdx == 1) { 
                              MemCopyOp(hostBlkA, devBlkA, blockSize, DEVICE_HOST_PINNED_COPY, 0, 0, threadTimer.stream);
                           } else {
                              if (repCount % 2)
                                 MemCopyOp(devBlkA, hostBlkA, blockSize, HOST_PINNED_DEVICE_COPY, 0, 0, threadTimer.stream);
                              else
                                 MemCopyOp(hostBlkB, devBlkB, blockSize, DEVICE_HOST_PINNED_COPY, 0, 0, threadTimer.stream);
                           }
                        }
                        
                        threadTimer.StopTimer();     
                        
                        topo.FreePinMem(hostBlkA, blockSize);
                        topo.FreeDeviceMem(devBlkA, device);
                        topo.FreePinMem(hostBlkB, blockSize);
                        topo.FreeDeviceMem(devBlkB, device);

                              
                        // calculate thread local bandwidth
                        double time = (double) threadTimer.ElapsedTime() / (double) params.numContRepeats;
                        double bandwidth = ((double) blockSize / (double) pow(2.0, 30.0)) / (time * 1.0E-6);
                        
                        threadBW[threadIdx] = bandwidth; 

                        // Wait for all threads to complete transfers and add thread bandwidths to array
                        #pragma omp barrier
 
                        // Calculate aggregate bandwidth and push thread bandwidths to data vector
                        #pragma omp single
                        {  
                           double aggBW = 0;
                           for (int i = 0; i < numThreads; ++i)
                              aggBW += threadBW[i];
                           
                           std::vector<float> testData;
                           testData.push_back(aggBW);
                              
                           for (int i = 0; i < numThreads; ++i)
                              testData.push_back(threadBW[i]);
                           data.push_back(testData);
                       
                        }        
                     }
                  }
               }
            }
         }
      }   
  
      std::cout << "\nSingle Host Multiple Device PCIe Contention Tests: " << std::endl;
    
      // Single Host PCIe/Mem Contention
      for (int socketIdx = 0; socketIdx < topo.NumSockets(); ++socketIdx) {
         
         for (int dirIdx = 0; dirIdx < numDirs; ++dirIdx) {
            
            std::cout << "Test " << testNum++ << "   \tSocket: " << socketIdx;
            std::cout << " Direction: " << dirIdx << " Max Threads: " << maxThreads << std::endl; 

            for (int numThreads = 1; numThreads <= maxThreads; ++numThreads) {
       
               omp_set_num_threads(numThreads);
               #pragma omp parallel
               {
                  // Get local thread ID
                  int threadIdx = omp_get_thread_num();
                  void * __restrict__ hostBlkA, * __restrict__ devBlkA;
                  void * __restrict__ hostBlkB, * __restrict__ devBlkB;
                       
                  int socket = socketIdx;
                  int core = threadIdx % topo.NumCoresPerSocket();
                  int devIdx = threadIdx % topo.NumGPUs();
                  
                  topo.SetActiveDevice(devIdx);
                  topo.PinCoreBySocket(socket, core);
                  topo.PinNode(socket);
                  
                  // Allocate Device Memory
                  hostBlkA = topo.AllocPinMemByNode(socket, blockSize);
                  topo.SetHostMem(hostBlkA, 2, blockSize);
                  devBlkA = topo.AllocDeviceMem(devIdx, blockSize);
                  topo.SetDeviceMem(devBlkA, 0, blockSize, devIdx);

                  hostBlkB = topo.AllocPinMemByNode(socket, blockSize);
                  topo.SetHostMem(hostBlkB, 1, blockSize);
                  devBlkB = topo.AllocDeviceMem(devIdx, blockSize);
                  topo.SetDeviceMem(devBlkB, 0, blockSize, devIdx);
                        
                  // Initialize timer and wait for all threads to finish allocation steps 
                  Timer threadTimer(false); 
                  #pragma omp barrier
            
                  threadTimer.StartTimer();
    
                  for (int repCount = 0; repCount < params.numContRepeats; repCount++) {
                     if (dirIdx == 0) {
                        MemCopyOp(devBlkA, hostBlkA, blockSize, HOST_PINNED_DEVICE_COPY, 0, 0, threadTimer.stream);
                     } else if (dirIdx == 1) { 
                        MemCopyOp(hostBlkA, devBlkA, blockSize, DEVICE_HOST_PINNED_COPY, 0, 0, threadTimer.stream);
                     } else {
                        if (repCount % 2)
                           MemCopyOp(devBlkA, hostBlkA, blockSize, HOST_PINNED_DEVICE_COPY, 0, 0, threadTimer.stream);
                        else
                           MemCopyOp(hostBlkB, devBlkB, blockSize, DEVICE_HOST_PINNED_COPY, 0, 0, threadTimer.stream);
                     }
                  }
                  
                  threadTimer.StopTimer();     
                 
                  // calculate thread local bandwidth
                  double time = (double) threadTimer.ElapsedTime() / (double) params.numContRepeats;
                  double bandwidth = ((double) blockSize / (double) pow(2.0, 30.0)) / (time * 1.0E-6);
                  
                  threadBW[threadIdx] = bandwidth; 

                  // Wait for all threads to complete transfers and add thread bandwidths to array
                  #pragma omp barrier

                  // Calculate aggregate bandwidth and push thread bandwidths to data vector
                  #pragma omp single
                  {  
                     double aggBW = 0;
                     for (int i = 0; i < numThreads; ++i)
                        aggBW += threadBW[i];

                     std::vector<float> testData;
                     testData.push_back(aggBW);
                        
                     for (int i = 0; i < numThreads; ++i)
                        testData.push_back(threadBW[i]);
                     data.push_back(testData);
                  } 

                  topo.FreePinMem(hostBlkA, blockSize);
                  topo.FreeDeviceMem(devBlkA, devIdx);
                  topo.FreePinMem(hostBlkB, blockSize);
                  topo.FreeDeviceMem(devBlkB, devIdx);

               }
            }
         }      
      }
   }

   std::cout << "Testing Aggregate GPU-Device Bandwidth" << std::endl;
   for (int mapIdx = 0; mapIdx < 2; ++mapIdx){
      for (int numThreads = 1; numThreads <= 4 * topo.NumGPUs(); ++numThreads) {
         
         omp_set_num_threads(numThreads);
         #pragma omp parallel
         {
            // Get local thread ID
            int threadIdx = omp_get_thread_num();
            void * __restrict__ hostBlkA, * __restrict__ devBlkA;
            void * __restrict__ hostBlkB, * __restrict__ devBlkB;
            int node, devIdx;

            devIdx = threadIdx % topo.NumGPUs();
            if (mapIdx == 0) {
               node = threadIdx % topo.NumNodes();
            } else {
               node = (threadIdx / topo.NumNodes()) % topo.NumSockets();
            }           
 
            // Pin Cores for execution and NUMA memory regions; set GPU device
            topo.SetActiveDevice(devIdx);
            topo.PinSocket(node); 
            topo.PinNode(node);
            
            // Allocate Host/Device memory and set initial values
            hostBlkA = topo.AllocPinMemByNode(node, blockSize);
            topo.SetHostMem(hostBlkA, 2, blockSize);
            devBlkA = topo.AllocDeviceMem(devIdx, blockSize);
            topo.SetDeviceMem(devBlkA, 0, blockSize, devIdx);

            hostBlkB = topo.AllocPinMemByNode(node, blockSize);
            topo.SetHostMem(hostBlkB, 1, blockSize);
            devBlkB = topo.AllocDeviceMem(devIdx, blockSize);
            topo.SetDeviceMem(devBlkB, 0, blockSize, devIdx);
 
            
            // Initialize local thread timer with CUDA event timing and wait for all threads to finish allocation steps
            Timer threadTimer(false); 
            #pragma omp barrier

            threadTimer.StartTimer();
             
            for (int repCount = 0; repCount < params.numContRepeats; ++repCount) {
               if (repCount % 2 == 0)
                  MemCopyOp(devBlkA, hostBlkA, blockSize, HOST_PINNED_DEVICE_COPY, 0, 0, threadTimer.stream);
               else
                  MemCopyOp(hostBlkB, devBlkB, blockSize, DEVICE_HOST_PINNED_COPY, 0, 0, threadTimer.stream);
            }
            
            threadTimer.StopTimer();     
                             
            // calculate thread local bandwidth
            double time = (double) threadTimer.ElapsedTime() / (double) params.numContRepeats;
            double bandwidth = ((double) blockSize / (double) pow(2.0, 30.0)) / (time * 1.0E-6);
            
            // place thread local bandwidth into bandwidth array 
            threadBW[threadIdx] = bandwidth; 

            // Wait for all threads to complete transfers and add thread bandwidths to array
            #pragma omp barrier

            // Calculate aggregate bandwidth and push thread bandwidths to data vector
            #pragma omp single
            {  
               double aggBW = 0;
               for (int i = 0; i < numThreads; ++i)
                  aggBW += threadBW[i];

               std::vector<float> testData;
               testData.push_back(aggBW);
                  
               for (int i = 0; i < numThreads; ++i)
                  testData.push_back(threadBW[i]);
               data.push_back(testData);
           
            }
            // Free thread local memory  
            topo.FreePinMem(hostBlkA, blockSize);
            topo.FreeDeviceMem(devBlkA, devIdx);
            topo.FreePinMem(hostBlkB, blockSize);
            topo.FreeDeviceMem(devBlkB, devIdx);



         }
      }
   }
      
   // Output results
   std::string dataFileName = "./results/contention/pcie/" + params.runTag + "_pcie_contention.csv";
   std::ofstream resultsFile(dataFileName.c_str());

   // Print results header values
   resultsFile << topo.NumSockets() << ","; 
   resultsFile << maxThreads << ","; 
   resultsFile << topo.NumGPUs() << ","; 
   resultsFile << numDirs << ","; 
   resultsFile << numSocketTests; 
   
   for (int i = 0; i < topo.NumGPUs(); i++)
      resultsFile << "," << topo.GetDeviceName(i);
   
   resultsFile << std::endl;

   // Print benchmark results to .csv file
   PrintResults(resultsFile, data);
}

void HHBurstTransferTest(BenchParams &params, SystemTopo &topo) {
   std::cout << "\nRunning Host-Host Burst Bandwidth Tests...\n" << std::endl;

   std::vector<std::vector<float> > burstData;

   BurstHHBandwidthRun(params, topo, burstData);
 
   PrintHHBurstMatrix(params, burstData);
}

void HDBurstTransferTest(BenchParams &params, SystemTopo &topo) {
   std::cout << "\nRunning Host-Device Burst Bandwidth Tests...\n" << std::endl;

   std::vector<std::vector<float> > burstData;

   BurstHDBandwidthRun(params, topo, burstData);  
   
   PrintHDBurstMatrix(params, burstData);
   
}

void P2PBurstTransferTest(BenchParams &params, SystemTopo &topo) {
   std::cout << "\nRunning Device-Device Burst Bandwidth Tests...\n" << std::endl;
   
   std::vector<std::vector<float> > burstData;

   BurstP2PBandwidthRun(params, topo, burstData);
 
   PrintP2PBurstMatrix(params, topo, burstData);
}

void BurstHHBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData) { 

   long long blockSize = params.burstBlockSize;
   burstData.resize(params.numPatterns * params.nSockets);
   double convConst = (double) blockSize / (double) pow(2.0, 30.0) * (double) 1.0E6; 

   for (int socketIdx = 0; socketIdx < params.nSockets; socketIdx++) {

      topo.PinSocket(socketIdx);
      
      for (int patternNum = 0; patternNum < params.numPatterns; patternNum ++) {
   
         MEM_PATTERN pattern = REPEATED;
         if (patternNum == 1)
            pattern = LINEAR_INC;
         if (patternNum == 2)
            pattern = LINEAR_DEC;
      
         for (int srcIdx = 0; srcIdx < params.nNodes; srcIdx++) { 

            for (int destIdx = 0; destIdx < params.nNodes; destIdx++) { 
               int rowIdx = socketIdx * params.numPatterns + patternNum;

               // HtoH Ranged Transfer - Pageable Memory
               burstData[rowIdx].push_back(convConst / BurstMemCopy(topo, blockSize, HOST_HOST_COPY, destIdx, srcIdx, params.numStepRepeats, pattern));        
              
               if (params.testAllMemTypes) {

                  // HtoH Ranged Transfer - Pinned Memory Src
                  burstData[rowIdx].push_back(convConst / BurstMemCopy(topo, blockSize, HOST_PINNED_HOST_COPY, destIdx, srcIdx, params.numStepRepeats, pattern)); 

                  // HtoH Ranged Transfer - Pinned Memory Dest
                  burstData[rowIdx].push_back(convConst / BurstMemCopy(topo, blockSize, HOST_HOST_PINNED_COPY, destIdx, srcIdx, params.numStepRepeats, pattern));        

                  // HtoH Ranged Transfer - Pinned Memory Both
                  burstData[rowIdx].push_back(convConst / BurstMemCopy(topo, blockSize, HOST_HOST_COPY_PINNED, destIdx, srcIdx, params.numStepRepeats, pattern));
               }       
            }
         }
      }
   }
}

void BurstHDBandwidthRun(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData) { 
   long long blockSize = params.burstBlockSize;
   double convConst = (double) blockSize / (double) pow(2.0, 30.0) * (double) 1.0E6; 
   burstData.resize(params.numPatterns * params.nSockets);

   for (int socketIdx = 0; socketIdx < params.nSockets; socketIdx++) {
      topo.PinSocket(socketIdx);
      
      for (int patternNum = 0; patternNum < params.numPatterns; patternNum++) {
      
         MEM_PATTERN pattern = REPEATED;
         if (patternNum == 1)
            pattern = LINEAR_INC;
         if (patternNum == 2)
            pattern = LINEAR_DEC;
    
         for (int srcIdx = 0; srcIdx < params.nNodes; srcIdx++) { 

            //Host-Device Memory Transfers
            for (int destIdx = 0; destIdx < params.nDevices; destIdx++) {
               int rowIdx = socketIdx * params.numPatterns + patternNum; 
               
               topo.SetActiveDevice(destIdx); 
               topo.PinNode(srcIdx); 

               // HtoD Ranged Transfer - Pageable Memory
               burstData[rowIdx].push_back( convConst / BurstMemCopy(topo, blockSize, HOST_DEVICE_COPY, destIdx, srcIdx, params.numStepRepeats, pattern));        
               
               // DtoH Ranged Transfer - Pageable Memory
               burstData[rowIdx].push_back( convConst / BurstMemCopy(topo, blockSize, DEVICE_HOST_COPY, srcIdx, destIdx, params.numStepRepeats, pattern));        
               
               if ( params.testAllMemTypes) {      

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
   double convConst = (double) blockSize / (double) pow(2.0, 30.0) * (double) 1.0E-6; 
   int nGPUs = params.nDevices; 
   burstData.resize(nGPUs * params.nSockets);
   
   for (int socketIdx = 0; socketIdx < params.nSockets; socketIdx++) {

      topo.PinSocket(socketIdx);
      topo.PinNode(socketIdx);
 
      for (int srcIdx = 0; srcIdx < nGPUs; srcIdx++) { 
         
         for (int destIdx = 0; destIdx < nGPUs; destIdx++) { 
            
            // DtoD Burst Transfer - No Peer, No UVA
            burstData[socketIdx * nGPUs + srcIdx].push_back(convConst / BurstMemCopy(topo, blockSize, DEVICE_DEVICE_COPY, destIdx, srcIdx, params.numStepRepeats)); 
            
            // DtoD Burst Transfer - Peer, No UVA
            if (topo.DeviceGroupCanP2P(srcIdx, destIdx)) {
               topo.DeviceGroupSetP2P(srcIdx, destIdx, true);
               burstData[socketIdx * nGPUs + srcIdx].push_back(convConst / BurstMemCopy(topo, blockSize, PEER_COPY_NO_UVA, destIdx, srcIdx, params.numStepRepeats)); 
               topo.DeviceGroupSetP2P(srcIdx, destIdx, false);
            }

            if (topo.DeviceGroupUVA(srcIdx, destIdx)) {  
            
               // DtoD Burst Transfer - No Peer, UVA
               burstData[socketIdx * nGPUs + srcIdx].push_back(convConst / BurstMemCopy(topo, blockSize, COPY_UVA, destIdx, srcIdx, params.numStepRepeats)); 
               
               // DtoD Burst Transfer - Peer, UVA
               if (topo.DeviceGroupCanP2P(srcIdx, destIdx)) {
                  topo.DeviceGroupSetP2P(srcIdx, destIdx, true);
                  burstData[socketIdx * nGPUs + srcIdx].push_back( convConst / BurstMemCopy(topo, blockSize, COPY_UVA, destIdx, srcIdx, params.numStepRepeats));        
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

   for (long long stepNum = 0; stepNum < totalSteps; stepNum++)
      data[stepNum].push_back(TimedMemCopyStep(destPtr, srcPtr, blockSteps[stepNum], blockSize, numCopies, copyType, pattern, destIdx, srcIdx));

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

   int magStart = max((int) log10(startStep), 0);
   int magStop = log10(stopStep);
   long long totalSteps = (magStop - magStart) * numSteps;
   long long start = pow(10, magStart);
   long long stop = pow(10, magStop); 
   long long step = start;
   double expStep = ((double) (magStop  - magStart)) / (double) totalSteps;
   double exp = magStart;
   
   if (stop == step) {
      blockSteps.push_back(start);      
      totalSteps = 1;
   }

   while (pow(10, exp) < stopStep) {
      step = pow(10, exp);
      blockSteps.push_back(step); 
      exp += expStep;
   }

   return totalSteps;
}

void PrintP2PBurstMatrix(BenchParams &params, SystemTopo &topo, std::vector<std::vector<float> > &burstData) {

   long long blockSize = params.burstBlockSize;
   std::vector<int> deviceIdxs;
   deviceIdxs.resize(params.nDevices, 0);
   int dataIdx = 0;
   
   int matrixWidth = params.nDevices;
   int matrixHeight = params.nDevices * 4;
   std::cout << "\nDevice-To-Device Unidirectional Memory Transfers:" << std::endl;
   std::cout << "Transfer Block Size: " << blockSize / BYTES_TO_MEGA << " (MB)"<< std::endl;
  
   for (int socketIdx = 0; socketIdx < params.nSockets; socketIdx++) {
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

void PrintHDBurstMatrix(BenchParams &params, std::vector<std::vector<float> > &burstData) {

   long long blockSize = params.burstBlockSize;
   int matrixWidth = params.nNodes;
   int matrixHeight = params.nDevices;
   
   std::cout << "\nHost/Device Unidirectional Memory Transfers:" << std::endl;
   std::cout << "Transfer Block Size: " << blockSize / BYTES_TO_MEGA << " (MB)"<< std::endl;
   std::cout << "Num Patterns: " << params.numPatterns << std::endl;

   std::cout << std::setprecision(2) << std::fixed;          

   for (int socketIdx = 0; socketIdx < params.nSockets; socketIdx++) {
      std::cout << "\nInitiating Socket: " << socketIdx << std::endl;
      
      for (int patternNum = 0; patternNum < params.numPatterns; patternNum++) {
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
            int rowIdx = socketIdx * params.numPatterns + patternNum;

            for (int j = 0; j < matrixWidth; ++j) {
                  int colIdx = j * params.nDevices * 4 + i * 4;
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

void PrintHHBurstMatrix(BenchParams &params, std::vector<std::vector<float> > &burstData) {

   long long blockSize = params.burstBlockSize;
   int nodeWidth = pow(HOST_MEM_TYPES * params.nNodes, 2) / params.nNodes;
   int matrixWidth = HOST_MEM_TYPES * params.nNodes;
   int matrixHeight = HOST_MEM_TYPES * params.nNodes;
   
   std::cout << "\nHost-Host Multi-NUMA Unidirectional Memory Transfers:" << std::endl;
   std::cout << "Transfer Block Size: " << blockSize / BYTES_TO_MEGA << " (MB)"<< std::endl;
   std::cout << "Num Patterns: " << params.numPatterns << std::endl;

   std::cout << std::setprecision(2) << std::fixed;          

   for (int socketIdx = 0; socketIdx < params.nSockets; socketIdx++) {
      std::cout << "\nInitiating Socket: " << socketIdx << std::endl;
      
      for (int patternNum = 0; patternNum < params.numPatterns; patternNum++) {
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
         for (int i = 0; i < params.nNodes; i++)
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

            std::cout << "|\t\t| " << i / (matrixHeight / params.nNodes) <<  "\t|";
            if (i % 2)
               std::cout << " Pin\t|    ";
            else
               std::cout << " Page\t|    ";
       
            int rowIdx = socketIdx * params.numPatterns + patternNum;
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
         fileStream << params.nSockets << ",";
         fileStream << params.nNodes << ",";
         fileStream << params.numPatterns;

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
         fileStream << params.nSockets << ",";
         fileStream << params.nNodes << ",";
         fileStream << params.nDevices << ",";
         fileStream << params.numPatterns;

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
         fileStream << params.nSockets << ",";
         fileStream << params.nDevices;
         fileStream << "," << topo.NumPeerGroups();
         
         if (params.runSocketTests) 
            fileStream << ",t";
         else
            fileStream << ",f";

         for (int i = 0; i < params.nDevices; i++) {
            fileStream << "," << topo.GetDeviceName(i);
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

void PrintResults(std::ofstream &outFile, std::vector<std::vector<float> > &results) {
   
   if (!outFile.is_open()) {
      std::cout << "Failed to open file to print results" << std::endl;
      return;
   }

   std::vector<std::vector<float> >::iterator iter_o;
   std::vector<float>::iterator iter_i;
   
   for (iter_o = results.begin(); iter_o != results.end(); ++iter_o) {

      for (iter_i = (*iter_o).begin(); iter_i != (*iter_o).end(); ++iter_i) {
         outFile << std::fixed << *iter_i;

         if (iter_i + 1 != (*iter_o).end())
            outFile << ",";

      }

      outFile << std::endl;

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

