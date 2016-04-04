
#ifndef BENCH_HEADER_INC
#define BENCH_HEADER_INC
#include "benchmark.h"
#endif

#ifndef PARAM_CLASS_INC
#define PARAM_CLASS_INC

#include<math.h>
class BenchParams {

   public:
      std::string inputFile;
      bool useDefaultParams;
      
      std::string resultsFile;
      bool printDevProps;
      std::string devPropFile;
      std::string topoFile;

      // All Tests (parameter file)
      bool runAllDevices;
      bool usePinnedMem;
      bool runBurstTests;
      bool runRangeTests; 
      bool runSustainedTests;
      bool runSocketTests;
      long numStepRepeats;
      long long burstBlockSize;
 
      int nDevices;
      int nSockets;

      // Overhead memory test for allocation and deallocation of Host and Device memory
      bool runMemoryOverheadTest;
      long long rangeMemOverhead[3]; //min, max and step size (in bytes)

      // Host-Host memory bandwidth test
      bool runHHBandwidthTest;
      bool runPatternsHH;
      long long rangeHostHostBW[3]; //min, max and step size (in bytes)
      
      // Host-Device PCIe Baseline bandwidth test
      bool runHDBandwidthTest;
      bool runPatternsHD;
      long long rangeHostDeviceBW[3]; //min, max and step size (in bytes)

      // Peer-to-peer device memory transfer bandwidth
      bool runP2PBandwidthTest; 
      long long rangeDeviceBW[3]; //min, max and step size (in bytes)

      // PCIe Congestion tests
      bool runPCIeCongestionTest;

      // CUDA kernel task scalability and load balancing
      bool runTaskScalabilityTest;

      void ParseParamFile(std::string fileStr);
      void SetDefault();
      void PrintParams();      
      BenchParams() {};
      ~BenchParams() {};
      
   private:
      void GetNextLine(std::ifstream &inFile, std::string &lineStr);
      bool GetNextLineBool(std::ifstream &inFile, std::string &lineStr);

} ;

#endif

