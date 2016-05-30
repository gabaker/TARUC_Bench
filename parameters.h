#ifndef PARAM_CLASS_INC
#define PARAM_CLASS_INC

#define NUM_ACCESS_PATTERNS 3

// C/C++ standard includes
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
//#include<math.h>

class BenchParams {

   public:
      // Output file names 
      std::string runTag;
      std::string devPropFile;
      std::string topoFile;

      // All Tests (parameter file)
      bool runAllDevices;
      bool testAllMemTypes;
      bool runBurstTests;
      bool runRangeTests; 
      bool runSustainedTests;
      bool runSocketTests;
      bool runPatternTests;
      long numStepRepeats;
      long numRangeSteps;
      long long burstBlockSize;

      // Parameters determined by input values
      int numPatterns;

      // Benchmark hardware information (combination of parameters and system topology info) 
      int nDevices;
      int nNodes;
      int nSockets;

      // Overhead memory test for allocation and deallocation of Host and Device memory
      bool runMemoryOverheadTest;
      long long rangeMemOH[2]; //min, max range block size (in bytes)

      // Host-Host memory bandwidth test
      bool runBandwidthTestHH;
      long long rangeHHBW[2]; //min, max range block size (in bytes)

      // Host-Device PCIe baseline bandwidth test
      bool runBandwidthTestHD;
      long long rangeHDBW[2]; //min, max range block size(in bytes)

      // Peer-to-peer device memory transfer bandwidth
      bool runBandwidthTestP2P; 
      long long rangeP2PBW[2]; //min, max range block size(in bytes)

      // Non-Uniform Random Memory Access (NURMA) micro-benchmark
      bool runNURMATest;
      long long gapNURMA;
      long long blockSizeNURMA;
      long long rangeNURMA[2];

      // Resource congestion tests
      bool runContentionTest;
      long long contBlockSize[3];
      long long numContRepeats;

      // Class initialization and setup
      void ParseParamFile(std::string FileStr);
      void SetDefault();
      void PrintParams();      
      BenchParams() {};
      ~BenchParams() {};

   private:
      // Input information
      std::string inputFile;
      bool useDefaultParams;

      // File parsing utility functions for value scanning
      bool GetNextBool(std::ifstream &InFile);
      long long GetNextInteger(std::ifstream &InFile);
      std::string GetNextString(std::ifstream &InFile);
      std::string GetNextLine(std::ifstream &InFile);

} ;

#endif

