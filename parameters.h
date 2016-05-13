
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
      
      std::string runTag;
      bool printDevProps;
      std::string devPropFile;
      std::string topoFile;

      // All Tests (parameter file)
      bool runAllDevices;
      bool testAllMemTypes;
      bool runBurstTests;
      bool runRangeTests; 
      bool runSustainedTests;
      bool runSocketTests;
      long numStepRepeats;
      long numRangeSteps;
      long long burstBlockSize;
 
      int nDevices;
      int nSockets;

      // Overhead memory test for allocation and deallocation of Host and Device memory
      bool runMemoryOverheadTest;
      long long rangeMemOverhead[2]; //min, max range block size (in bytes)

      // Host-Host memory bandwidth test
      bool runBandwidthTestHH;
      bool runPatternsHH;
      long long rangeHostHostBW[2]; //min, max range block size (in bytes)
      
      // Host-Device PCIe Baseline bandwidth test
      bool runBandwidthTestHD;
      bool runPatternsHD;
      long long rangeHostDeviceBW[2]; //min, max range block size(in bytes)

      // Peer-to-peer device memory transfer bandwidth
      bool runBandwidthTestP2P; 
      long long rangeDeviceBW[2]; //min, max range block size(in bytes)

      // Resource Congestion tests
      bool runContentionTest;
      long long rangeCont[2];
      long long numContRepeats;
      int numContMemTypes;
      bool testContRange;

      void ParseParamFile(std::string fileStr);
      void SetDefault();
      void PrintParams();      
      BenchParams() {};
      ~BenchParams() {};
      
   private:
      bool GetNextBool(std::ifstream &inFile);
      long long GetNextInteger(std::ifstream &inFile);
      std::string GetNextString(std::ifstream &inFile);
      std::string GetNextLine(std::ifstream &inFile);

} ;

#endif

